//! Immutable package compilation before any main-world atlas publication.

use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
};

use super::*;
use crate::stream::{
    hierarchy::CompiledManifestLodHierarchyCompiler,
    persistent_cache::{PersistentCachePageIdentities, PersistentCachePageIdentitiesBuilder},
    preparation::PreparationBudget,
    runtime::PreparedLodRuntime,
    transport::{ManifestPageLocations, ManifestPageLocationsBuilder},
};

pub(super) struct PreparedPackage {
    pub plan: GaussianLodPackageAtlasPlan,
    pub runtime: PreparedLodRuntime,
    pub locations: ManifestPageLocations,
    pub identities: Option<PersistentCachePageIdentities>,
    pub effective: GaussianLodSettings,
    pub structural: PackageStructuralSettings,
    pub runtime_streaming: GaussianStreamingSettings,
    pub debug_index: Option<Arc<LodDebugManifestIndex>>,
    pub memory_leases: Vec<LodMemoryLease>,
}

impl PreparedPackage {
    fn reserve_storage(&mut self, ledger: &LodMemoryLedger) -> Result<(), LodMemoryAdmissionError> {
        if self
            .memory_leases
            .iter()
            .any(|lease| lease.category() == LodMemoryCategory::AtlasGpu)
        {
            return Ok(());
        }
        let encoded = self.runtime_streaming.effective_max_encoded_page_bytes();
        let transport_bytes = encoded
            .checked_mul(u64::from(self.runtime_streaming.max_concurrent_requests))
            .ok_or(LodMemoryAdmissionError::ByteOverflow)?;
        let preprocess_bytes = encoded
            .checked_add(self.effective.budgets.max_upload_bytes_per_frame)
            .ok_or(LodMemoryAdmissionError::ByteOverflow)?;
        let atlas_bytes = self
            .plan
            .physical_bytes
            .checked_add(16)
            .ok_or(LodMemoryAdmissionError::ByteOverflow)?;
        let leases = ledger.try_reserve_many(&[
            (
                LodMemoryCategory::MetadataCpu,
                crate::stream::cache::LodPageCache::eviction_index_bytes(
                    self.effective.budgets.max_resident_pages,
                ),
            ),
            (LodMemoryCategory::AtlasGpu, atlas_bytes),
            (
                LodMemoryCategory::DecodedPagesCpu,
                self.effective.budgets.max_resident_bytes,
            ),
            (
                LodMemoryCategory::RecoveryStagingCpu,
                self.plan.cpu_recovery_capacity_bytes(),
            ),
            (LodMemoryCategory::TransportCpu, transport_bytes),
            (LodMemoryCategory::PreprocessCpu, preprocess_bytes),
        ])?;
        self.memory_leases.extend(leases);
        Ok(())
    }
}

pub(super) async fn prepare_package(
    asset: GaussianLodAsset,
    source: GaussianLodPackageSource,
    settings: GaussianLodSettings,
    config: GaussianLodPackageConfig,
    streaming: GaussianStreamingSettings,
    debug_metadata: bool,
    work: PreparationBudget,
) -> Result<PreparedPackage, GaussianLodPackageError> {
    settings
        .validate()
        .map_err(|error| GaussianLodPackageError::InvalidLodSettings(error.to_string()))?;
    config.validate_limits()?;
    let manifest = asset.shared_manifest();
    let mut stride = 0;
    for page in &manifest.pages {
        work.record().await;
        stride = stride.max(page.gaussian_count);
    }
    if stride == 0 {
        return Err(GaussianLodPackageError::ManifestHasNoPages);
    }
    let plan = GaussianLodPackageAtlasPlan::from_limits(
        manifest.header.source_gaussian_count,
        stride,
        &settings,
        &config,
    )?;
    let bytes_per_slot = u64::from(plan.gaussians_per_slot)
        .checked_mul(gaussian_3d_gpu_bytes_per_record())
        .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
    let staging_step_bytes = package_gpu_staging_step_byte_limit(&settings);
    if bytes_per_slot > staging_step_bytes {
        return Err(GaussianLodPackageError::GpuUploadCommitTooLarge {
            dirty_slots: 1,
            bytes: bytes_per_slot,
            limit: staging_step_bytes,
        });
    }
    let mut compiler = CompiledManifestLodHierarchyCompiler::new(Arc::clone(&manifest));
    let hierarchy = loop {
        work.record().await;
        if let Some(hierarchy) = compiler.advance(1) {
            break hierarchy;
        }
    };
    let mut root_pages = BTreeSet::new();
    for &root in &manifest.roots {
        work.record().await;
        let node = hierarchy
            .node(root)
            .ok_or_else(|| GaussianLodPackageError::InvalidManifest("missing root".to_owned()))?;
        root_pages.insert(node.representation.page);
    }
    if root_pages.len() > plan.slot_count as usize {
        return Err(GaussianLodPackageError::RootFallbackExceedsAtlas {
            root_pages: root_pages.len() as u64,
            slots: plan.slot_count,
        });
    }
    let page_index = hierarchy.compiled_page_index();
    let mut locations = ManifestPageLocationsBuilder::new(manifest.pages.len());
    let mut identities = streaming
        .persistent_cache
        .then(|| PersistentCachePageIdentitiesBuilder::new(&manifest));
    for page in &manifest.pages {
        work.record().await;
        locations.push(page).map_err(|error| match source {
            GaussianLodPackageSource::NativeDirectory { .. } => {
                GaussianLodPackageError::NativeTransport(error.to_string())
            }
            GaussianLodPackageSource::Url { .. } => {
                GaussianLodPackageError::HttpTransport(error.to_string())
            }
        })?;
        if let Some(identities) = &mut identities {
            identities
                .push(page)
                .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string()))?;
        }
    }
    let locations = locations.finish(page_index.clone());
    validate_package_locations(&source, &streaming, &locations, &work).await?;
    let identities = identities.map(|identities| identities.finish(page_index));
    let runtime_streaming = package_runtime_streaming_settings(&source, &streaming);
    let mut effective = settings.clone();
    effective.budgets.max_resident_pages = plan.slot_count;
    effective.budgets.max_resident_gaussians = u64::from(plan.physical_gaussians);
    effective.budgets.max_resident_bytes = u64::from(plan.physical_gaussians)
        .checked_mul(size_of::<Gaussian3d>() as u64)
        .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
    effective.budgets.max_active_gaussians = effective
        .budgets
        .max_active_gaussians
        .min(u64::from(plan.physical_gaussians));
    let structural = PackageStructuralSettings {
        max_resident_gaussians: effective.budgets.max_resident_gaussians,
        max_resident_bytes: effective.budgets.max_resident_bytes,
        max_resident_pages: effective.budgets.max_resident_pages,
        max_pending_requests: effective.budgets.max_pending_requests,
    };
    let bootstrap = LodPackageBootstrapBudget {
        max_pages: plan.slot_count.min(PACKAGE_BOOTSTRAP_MAX_PAGES),
        max_active_gaussians: effective
            .budgets
            .max_active_gaussians
            .min(PACKAGE_BOOTSTRAP_MAX_ACTIVE_GAUSSIANS),
        max_encoded_bytes: PACKAGE_BOOTSTRAP_MAX_ENCODED_BYTES,
        max_decoded_bytes: PACKAGE_BOOTSTRAP_MAX_DECODED_BYTES,
        max_gpu_bytes: package_gpu_staging_step_byte_limit(&effective)
            .min(PACKAGE_BOOTSTRAP_MAX_GPU_BYTES),
        gpu_bytes_per_slot: bytes_per_slot,
    };
    let runtime = PreparedLodRuntime::new(
        hierarchy,
        &effective,
        &runtime_streaming,
        Some(bootstrap),
        &work,
    )
    .await
    .map_err(GaussianLodPackageError::Runtime)?;
    let debug_index = if debug_metadata {
        Some(Arc::new(
            LodDebugManifestIndex::prepare_from_validated_manifest(&manifest, &work)
                .await
                .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?,
        ))
    } else {
        None
    };
    Ok(PreparedPackage {
        plan,
        runtime,
        locations,
        identities,
        effective,
        structural,
        runtime_streaming,
        debug_index,
        memory_leases: Vec::new(),
    })
}

pub(super) async fn validate_package_locations(
    source: &GaussianLodPackageSource,
    streaming: &GaussianStreamingSettings,
    locations: &ManifestPageLocations,
    work: &PreparationBudget,
) -> Result<(), GaussianLodPackageError> {
    match source {
        GaussianLodPackageSource::NativeDirectory { root } => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                platform::validate_native_root(root)?;
                for page in locations.page_ids() {
                    work.record().await;
                    crate::stream::transport::validate_native_page_location(
                        page,
                        locations.get(page).expect("compiled page location"),
                        streaming.effective_max_encoded_page_bytes(),
                    )
                    .map_err(|error| GaussianLodPackageError::NativeTransport(error.to_string()))?;
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = root;
                return Err(GaussianLodPackageError::NativeSourceUnsupportedInBrowser);
            }
        }
        GaussianLodPackageSource::Url { base_url } => {
            let config = package_http_config(base_url, streaming)?;
            for page in locations.page_ids() {
                work.record().await;
                let location = locations.get(page).expect("compiled page location");
                crate::stream::http::validate_http_location(
                    page,
                    location,
                    config.max_encoded_page_bytes,
                )
                .and_then(|()| {
                    crate::stream::http::resolve_page_url(base_url, &location.uri).map(|_| ())
                })
                .map_err(|error| GaussianLodPackageError::HttpTransport(error.to_string()))?;
            }
        }
    }
    Ok(())
}

type PreparationFuture =
    Pin<Box<dyn Future<Output = Result<PreparedPackage, GaussianLodPackageError>> + Send>>;

enum PreparationExecution {
    Cooperative {
        future: Mutex<PreparationFuture>,
        work: PreparationBudget,
    },
    #[cfg(not(target_arch = "wasm32"))]
    Native(bevy::tasks::Task<Option<Result<PreparedPackage, GaussianLodPackageError>>>),
    // Move only the fixed result header out of the enum. Payload allocations
    // and their leases remain owned by the same PreparedPackage until taken.
    Ready(Box<Mutex<Option<Result<PreparedPackage, GaussianLodPackageError>>>>),
}

pub(super) struct PackagePreparationJob {
    pub manifest: AssetId<GaussianLodAsset>,
    pub source: GaussianLodPackageSource,
    pub config: GaussianLodPackageConfig,
    pub streaming: GaussianStreamingSettings,
    pub structural: PackageStructuralSignature,
    settings: GaussianLodSettings,
    pub charged_bytes: u64,
    pub failure: Option<GaussianLodPackageError>,
    metadata_lease: Option<LodMemoryLease>,
    cancelled: Arc<AtomicBool>,
    execution: PreparationExecution,
}

impl PackagePreparationJob {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        manifest: AssetId<GaussianLodAsset>,
        asset: &GaussianLodAsset,
        source: &GaussianLodPackageSource,
        settings: &GaussianLodSettings,
        config: &GaussianLodPackageConfig,
        streaming: &GaussianStreamingSettings,
        debug_metadata: bool,
    ) -> Self {
        Self::new_reserved(
            manifest,
            asset,
            source,
            settings,
            config,
            streaming,
            debug_metadata,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_reserved(
        manifest: AssetId<GaussianLodAsset>,
        asset: &GaussianLodAsset,
        source: &GaussianLodPackageSource,
        settings: &GaussianLodSettings,
        config: &GaussianLodPackageConfig,
        streaming: &GaussianStreamingSettings,
        debug_metadata: bool,
        metadata_lease: Option<LodMemoryLease>,
    ) -> Self {
        let work = PreparationBudget::new(0);
        let compiler = prepare_package(
            asset.clone(),
            source.clone(),
            settings.clone(),
            config.clone(),
            streaming.clone(),
            debug_metadata,
            work.clone(),
        );
        let future_lease = metadata_lease.clone();
        let future: PreparationFuture = Box::pin(async move {
            let mut prepared = compiler.await?;
            if let Some(lease) = future_lease {
                prepared.memory_leases.push(lease);
            }
            Ok(prepared)
        });
        let cancelled = Arc::new(AtomicBool::new(false));
        #[cfg(not(target_arch = "wasm32"))]
        let execution = if !cfg!(test) {
            let cancellation = Arc::clone(&cancelled);
            let pool = bevy::tasks::AsyncComputeTaskPool::get_or_init(|| {
                bevy::tasks::TaskPoolBuilder::new()
                    .num_threads(2)
                    .thread_name("lod-package-preparation".to_owned())
                    .build()
            });
            PreparationExecution::Native(pool.spawn(async move {
                let mut future = future;
                loop {
                    if cancellation.load(Ordering::Acquire) {
                        return None;
                    }
                    work.reset(4096);
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        future
                            .as_mut()
                            .poll(&mut Context::from_waker(Waker::noop()))
                    }));
                    match result {
                        Ok(Poll::Ready(result)) => {
                            return (!cancellation.load(Ordering::Acquire)).then_some(result);
                        }
                        Err(_) => return Some(Err(GaussianLodPackageError::PreparationPanicked)),
                        Ok(Poll::Pending) => bevy::tasks::futures_lite::future::yield_now().await,
                    }
                }
            }))
        } else {
            PreparationExecution::Cooperative {
                future: Mutex::new(future),
                work,
            }
        };
        #[cfg(target_arch = "wasm32")]
        let execution = PreparationExecution::Cooperative {
            future: Mutex::new(future),
            work,
        };
        Self {
            manifest,
            source: source.clone(),
            config: config.clone(),
            streaming: streaming.clone(),
            structural: PackageStructuralSignature::new(settings),
            settings: settings.clone(),
            charged_bytes: asset.preparation_bytes(),
            failure: None,
            metadata_lease,
            cancelled,
            execution,
        }
    }

    pub(super) fn matches(
        &self,
        signature: PackageBuildSignature<'_>,
        settings: &GaussianLodSettings,
    ) -> bool {
        (self.failure.is_none() || self.settings == *settings)
            && signature
                == PackageBuildSignature {
                    manifest: self.manifest,
                    source: &self.source,
                    config: &self.config,
                    streaming: &self.streaming,
                    structural: self.structural,
                }
    }

    pub(super) fn cancel(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns consumed cooperative record allowance. Native work consumes no
    /// application-thread allowance; cancelled work remains charged until ack.
    pub(super) fn advance(&mut self, records: usize) -> usize {
        match &mut self.execution {
            PreparationExecution::Cooperative { future, work } => {
                if self.cancelled.load(Ordering::Acquire) {
                    self.execution = PreparationExecution::Ready(Box::default());
                    return 0;
                }
                work.reset(records);
                let future = future
                    .get_mut()
                    .expect("preparation future has one mutable owner");
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    future
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                }));
                let consumed = records.saturating_sub(work.remaining());
                match result {
                    Ok(Poll::Ready(result)) => {
                        self.execution =
                            PreparationExecution::Ready(Box::new(Mutex::new(Some(result))))
                    }
                    Err(_) => {
                        self.execution = PreparationExecution::Ready(Box::new(Mutex::new(Some(
                            Err(GaussianLodPackageError::PreparationPanicked),
                        ))))
                    }
                    Ok(Poll::Pending) => {}
                }
                consumed
            }
            #[cfg(not(target_arch = "wasm32"))]
            PreparationExecution::Native(task) => {
                if let Poll::Ready(result) =
                    Pin::new(task).poll(&mut Context::from_waker(Waker::noop()))
                {
                    self.execution = PreparationExecution::Ready(Box::new(Mutex::new(result)));
                }
                0
            }
            PreparationExecution::Ready(_) => 0,
        }
    }

    pub(super) fn reserve_ready_storage(
        &mut self,
        ledger: &LodMemoryLedger,
    ) -> Result<(), LodMemoryAdmissionError> {
        if let PreparationExecution::Ready(result) = &mut self.execution
            && let Some(Ok(prepared)) = result
                .get_mut()
                .expect("prepared result has one owner")
                .as_mut()
        {
            prepared.reserve_storage(ledger)?;
        }
        Ok(())
    }

    pub(super) fn release_failed_reservation(&mut self) {
        self.charged_bytes = 0;
        self.metadata_lease = None;
    }

    pub(super) fn is_ready(&self) -> bool {
        matches!(self.execution, PreparationExecution::Ready(_))
    }
    pub(super) fn take_result(
        &mut self,
    ) -> Option<Result<PreparedPackage, GaussianLodPackageError>> {
        let result = match &mut self.execution {
            PreparationExecution::Ready(result) => result
                .get_mut()
                .expect("prepared result has one mutable owner")
                .take(),
            _ => None,
        };
        if let Some(Err(error)) = &result {
            self.failure = Some(error.clone());
            self.release_failed_reservation();
        }
        result
    }
}

impl Drop for PackagePreparationJob {
    fn drop(&mut self) {
        self.cancel();
    }
}
