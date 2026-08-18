//! Automatic Bevy wiring for prebuilt `.gsplatlod` packages.
//!
//! Package manifests remain Bevy assets while page payloads are fetched by the
//! bounded streaming runtime. Native files and immutable HTTP(S) objects share
//! the same two-phase atlas/render commit path; optionally, encoded pages pass
//! through a content-addressed persistent cache before decoding.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
    mem::size_of,
    sync::{Mutex, atomic::Ordering},
    time::Duration,
};

use bevy::prelude::*;
use bevy::{
    camera::{CameraUpdateSystems, Projection, primitives::Aabb, visibility::VisibilitySystems},
    transform::TransformSystems,
};
use bevy_interleave::prelude::{Planar, PlanarHandle};

#[cfg_attr(target_arch = "wasm32", path = "package/browser.rs")]
#[cfg_attr(not(target_arch = "wasm32"), path = "package/native.rs")]
mod platform;

use crate::{
    CloudSettings,
    gaussian::{
        formats::{
            planar_3d::{Gaussian3d, gaussian_3d_gpu_bytes_per_record},
            planar_3d_chunked::{LodNodeId, LodPageId},
        },
        lod_settings::{GaussianLodSettings, GaussianStreamingSettings},
    },
    io::lod::GaussianLodHandle,
};
use platform::{
    PackageCacheRegistry, PackageManagerParam, PackagePageTransport, init_package_manager,
    package_page_transport, validate_cache_config,
};

use crate::{
    gaussian::{
        formats::planar_3d::{PlanarGaussian3d, PlanarGaussian3dHandle},
        lod_debug::{
            LodDebugAnnotationAtlas, LodDebugManifestIndex, LodDebugMetadata, LodDebugResidency,
        },
    },
    io::lod::GaussianLodAsset,
    stream::{
        LodRenderPathSupportError,
        atlas_upload::{GaussianLodAtlasUploadPlugin, LodAtlasUploadQueue},
        cache::AtlasSlot,
        hierarchy::LodView,
        http::HttpRangeTransportConfig,
        persistent_cache::PersistentCachePackageIdentity,
        render_commit::{
            GaussianLodRenderCommitPlugin, LOD_RENDER_ACTIVE, LodOrchestrationFailure,
            LodOrchestrationFailureCode, LodOrchestrationSource, LodOrchestrationTransition,
            LodOrchestrationTransitionKind, LodPageAtlasMirror, LodRenderCandidate,
            LodRenderCandidates, LodRenderCommitError,
        },
        require_lod_render_path,
        runtime::{
            LodCandidateFrontier, LodPhysicalRange, LodRuntimeError, LodRuntimeViewId,
            LodStreamFrame, LodStreamingRuntime,
        },
        transport::{
            LodPageTransport, LodPageTransportFailure, LodPageTransportFailureKind, PagePoll,
        },
    },
};

const PACKAGE_ROOT_FALLBACK_VIEW: LodRuntimeViewId = LodRuntimeViewId(u64::MAX);

/// Page-byte source paired with a loaded [`GaussianLodHandle`].
#[derive(Component, Clone, Debug, PartialEq, Eq, Reflect)]
#[reflect(Component)]
pub enum GaussianLodPackageSource {
    /// Manifest page URIs are resolved below this explicit directory.
    NativeDirectory { root: String },
    /// Manifest page URIs are resolved below this absolute HTTP(S) base URL.
    Url { base_url: String },
}

impl GaussianLodPackageSource {
    pub fn native_directory(root: impl Into<String>) -> Self {
        Self::NativeDirectory { root: root.into() }
    }

    pub fn url(base_url: impl Into<String>) -> Self {
        Self::Url {
            base_url: base_url.into(),
        }
    }
}

/// Global hard bounds for package-owned physical atlases.
#[derive(Resource, Clone, Debug, PartialEq, Reflect)]
#[reflect(Resource)]
pub struct GaussianLodPackageConfig {
    pub max_atlas_gaussians: u32,
    /// Hard canonical-plus-derived GPU storage bound for the physical atlas.
    pub max_atlas_bytes: u64,
    pub max_views_per_cloud: u32,
    /// Explicit parent directory for native persistent caches. Enabling cache
    /// persistence without setting this field fails closed on native targets.
    pub persistent_cache_root: Option<String>,
    /// Explicit safe namespace used as a native subdirectory and browser Cache
    /// Storage name. It is required whenever persistence is enabled.
    pub persistent_cache_namespace: Option<String>,
    /// Hard persistent metadata/file count bound independent of the byte bound.
    pub persistent_cache_max_entries: u32,
    pub streaming: GaussianStreamingSettings,
}

impl Default for GaussianLodPackageConfig {
    fn default() -> Self {
        let streaming = GaussianStreamingSettings {
            persistent_cache: false,
            ..default()
        };
        Self {
            max_atlas_gaussians: 524_288,
            max_atlas_bytes: 512 * 1024 * 1024,
            max_views_per_cloud: 16,
            persistent_cache_root: None,
            persistent_cache_namespace: None,
            persistent_cache_max_entries: 16_384,
            streaming,
        }
    }
}

impl GaussianLodPackageConfig {
    pub fn validate(&self) -> Result<(), GaussianLodPackageError> {
        self.validate_limits()?;
        let streaming = package_streaming_settings(&self.streaming)?;
        if streaming.persistent_cache {
            validate_cache_config(self)?;
        }
        Ok(())
    }

    fn validate_limits(&self) -> Result<(), GaussianLodPackageError> {
        if self.max_atlas_gaussians == 0 {
            return Err(GaussianLodPackageError::ZeroLimit("max_atlas_gaussians"));
        }
        if self.max_atlas_bytes == 0 {
            return Err(GaussianLodPackageError::ZeroLimit("max_atlas_bytes"));
        }
        if self.max_views_per_cloud == 0 {
            return Err(GaussianLodPackageError::ZeroLimit("max_views_per_cloud"));
        }
        if self.persistent_cache_max_entries == 0 {
            return Err(GaussianLodPackageError::ZeroLimit(
                "persistent_cache_max_entries",
            ));
        }
        Ok(())
    }
}

fn package_streaming_settings(
    settings: &GaussianStreamingSettings,
) -> Result<GaussianStreamingSettings, GaussianLodPackageError> {
    settings
        .validate()
        .map_err(|error| GaussianLodPackageError::InvalidStreaming(error.to_string()))?;
    Ok(settings.clone())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Reflect)]
pub enum GaussianLodPackagePhase {
    Loading,
    Active,
    Degraded,
    Failed,
}

/// Observable package loading and streaming state.
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component)]
pub struct GaussianLodPackageStatus {
    pub phase: GaussianLodPackagePhase,
    pub resident_pages: u32,
    pub active_gaussians: u64,
    pub terminal_failures: u32,
    pub failure: Option<LodOrchestrationFailure>,
}

impl GaussianLodPackageStatus {
    fn loading() -> Self {
        Self {
            phase: GaussianLodPackagePhase::Loading,
            resident_pages: 0,
            active_gaussians: 0,
            terminal_failures: 0,
            failure: None,
        }
    }

    fn failed(error: GaussianLodPackageError) -> Self {
        Self {
            phase: GaussianLodPackagePhase::Failed,
            failure: Some(LodOrchestrationFailure::from(&error)),
            ..Self::loading()
        }
    }

    /// Human-readable context retained for compatibility with logging and UI
    /// code that previously consumed an untyped error string.
    pub fn error_detail(&self) -> Option<&str> {
        self.failure.as_ref().and_then(|failure| failure.detail())
    }
}

/// Bounded allocation derived only from page stride and explicit budgets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GaussianLodPackageAtlasPlan {
    pub virtual_source_gaussians: u64,
    pub gaussians_per_slot: u32,
    pub slot_count: u32,
    pub physical_gaussians: u32,
    /// Total canonical plus feature-enabled derived GPU storage bytes.
    pub physical_bytes: u64,
}

impl GaussianLodPackageAtlasPlan {
    pub fn from_manifest(
        manifest: &crate::GaussianLodManifest,
        settings: &GaussianLodSettings,
        config: &GaussianLodPackageConfig,
    ) -> Result<Self, GaussianLodPackageError> {
        manifest
            .validate()
            .map_err(|error| GaussianLodPackageError::InvalidManifest(error.to_string()))?;
        settings
            .validate()
            .map_err(|error| GaussianLodPackageError::InvalidLodSettings(error.to_string()))?;
        config.validate_limits()?;
        let stride = manifest
            .pages
            .iter()
            .map(|page| page.gaussian_count)
            .max()
            .ok_or(GaussianLodPackageError::ManifestHasNoPages)?;
        let plan = Self::from_limits(
            manifest.header.source_gaussian_count,
            stride,
            settings,
            config,
        )?;
        let nodes = manifest
            .nodes
            .iter()
            .map(|node| (node.id, node))
            .collect::<HashMap<_, _>>();
        let root_pages = manifest
            .roots
            .iter()
            .filter_map(|root| nodes.get(root).map(|node| node.representation.page))
            .collect::<BTreeSet<_>>();
        let bytes_per_slot = u64::from(plan.gaussians_per_slot)
            .checked_mul(gaussian_3d_gpu_bytes_per_record())
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        let root_gpu_bytes = (root_pages.len() as u64)
            .checked_mul(bytes_per_slot)
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        if root_gpu_bytes > settings.budgets.max_gpu_upload_bytes_per_commit {
            return Err(
                GaussianLodPackageError::RootFallbackExceedsGpuUploadBudget {
                    root_pages: root_pages.len() as u64,
                    required: root_gpu_bytes,
                    limit: settings.budgets.max_gpu_upload_bytes_per_commit,
                },
            );
        }
        if root_pages.len() > plan.slot_count as usize {
            return Err(GaussianLodPackageError::RootFallbackExceedsAtlas {
                root_pages: root_pages.len() as u64,
                slots: plan.slot_count,
            });
        }
        Ok(plan)
    }

    fn from_limits(
        virtual_source_gaussians: u64,
        gaussians_per_slot: u32,
        settings: &GaussianLodSettings,
        config: &GaussianLodPackageConfig,
    ) -> Result<Self, GaussianLodPackageError> {
        if gaussians_per_slot == 0 {
            return Err(GaussianLodPackageError::ManifestHasNoPages);
        }
        let canonical_bytes_per_slot = u64::from(gaussians_per_slot)
            .checked_mul(size_of::<Gaussian3d>() as u64)
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        let gpu_bytes_per_slot = u64::from(gaussians_per_slot)
            .checked_mul(gaussian_3d_gpu_bytes_per_record())
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        let slots = settings
            .budgets
            .max_resident_pages
            .min(config.max_atlas_gaussians / gaussians_per_slot)
            .min((config.max_atlas_bytes / gpu_bytes_per_slot).min(u64::from(u32::MAX)) as u32)
            .min(
                (settings.budgets.max_resident_gaussians / u64::from(gaussians_per_slot))
                    .min(u64::from(u32::MAX)) as u32,
            )
            .min(
                (settings.budgets.max_resident_bytes / canonical_bytes_per_slot)
                    .min(u64::from(u32::MAX)) as u32,
            )
            .min(
                (settings.budgets.max_gpu_upload_bytes_per_commit / gpu_bytes_per_slot)
                    .min(u64::from(u32::MAX)) as u32,
            );
        if slots == 0 {
            return Err(GaussianLodPackageError::AtlasCannotFitPage {
                gaussians_per_slot,
                bytes_per_slot: gpu_bytes_per_slot,
            });
        }
        let physical_gaussians = slots
            .checked_mul(gaussians_per_slot)
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        let physical_bytes = u64::from(physical_gaussians)
            .checked_mul(gaussian_3d_gpu_bytes_per_record())
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        Ok(Self {
            virtual_source_gaussians,
            gaussians_per_slot,
            slot_count: slots,
            physical_gaussians,
            physical_bytes,
        })
    }
}

struct PackageInstantiation {
    manifest: AssetId<GaussianLodAsset>,
    source: GaussianLodPackageSource,
    config: GaussianLodPackageConfig,
    atlas: Handle<PlanarGaussian3d>,
    plan: GaussianLodPackageAtlasPlan,
    runtime: Mutex<LodStreamingRuntime<PackagePageTransport>>,
    mirror: LodPageAtlasMirror,
    /// Allocated only while cloud debug settings actually require metadata.
    debug: Option<PackageDebugAnnotations>,
    current: Option<LodRenderCandidates>,
    pending: Option<LodRenderCandidates>,
    current_fallback_nodes: BTreeSet<LodNodeId>,
    pending_fallback_nodes: BTreeSet<LodNodeId>,
    views: BTreeSet<Entity>,
    /// Slots that currently contain the complete fallback cut.
    /// A replacement cut only dirties the union of this set and its new set.
    visible_slots: BTreeMap<u32, AtlasSlot>,
    /// Exact normalized retained range/provenance state already materialized.
    visible_ranges: Vec<LodPhysicalRange>,
    visible_fallback_nodes: BTreeSet<LodNodeId>,
    requested_structural: PackageStructuralSignature,
    structural: PackageStructuralSettings,
    /// Retry budget seen by the generic runtime. HTTP owns the configured
    /// retry/backoff budget internally, so its outer runtime budget is zero.
    runtime_streaming: GaussianStreamingSettings,
    streaming: GaussianStreamingSettings,
    /// Pages that have already spent their one cache-repair allowance since
    /// their last successful decode. This keeps cache repair available even
    /// when the ordinary outer retry budget is zero, without creating an
    /// unbounded preprocess/invalidate/retry loop.
    preprocess_cache_repairs: BTreeSet<LodPageId>,
    resident_pages: u32,
    active_gaussians: u64,
    terminal_failures: u32,
    root_fallback: bool,
    last_failure: Option<LodOrchestrationFailure>,
}

struct PackageDebugAnnotations {
    atlas: LodDebugAnnotationAtlas,
    /// Owned, validated once per package generation and reused per page.
    index: LodDebugManifestIndex,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PackageStructuralSettings {
    max_resident_gaussians: u64,
    max_resident_bytes: u64,
    max_resident_pages: u32,
    max_pending_requests: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PackageStructuralSignature {
    max_resident_gaussians: u64,
    max_resident_bytes: u64,
    max_resident_pages: u32,
    max_pending_requests: u32,
    max_gpu_upload_bytes_per_commit: u64,
}

impl PackageStructuralSignature {
    fn new(settings: &GaussianLodSettings) -> Self {
        Self {
            max_resident_gaussians: settings.budgets.max_resident_gaussians,
            max_resident_bytes: settings.budgets.max_resident_bytes,
            max_resident_pages: settings.budgets.max_resident_pages,
            max_pending_requests: settings.budgets.max_pending_requests,
            max_gpu_upload_bytes_per_commit: settings.budgets.max_gpu_upload_bytes_per_commit,
        }
    }
}

impl PackageStructuralSettings {
    fn apply(self, settings: &GaussianLodSettings) -> GaussianLodSettings {
        let mut effective = settings.clone();
        effective.budgets.max_resident_gaussians = self.max_resident_gaussians;
        effective.budgets.max_resident_bytes = self.max_resident_bytes;
        effective.budgets.max_resident_pages = self.max_resident_pages;
        effective.budgets.max_pending_requests = self.max_pending_requests;
        effective.budgets.max_active_gaussians = effective
            .budgets
            .max_active_gaussians
            .min(self.max_resident_gaussians);
        effective
    }
}

#[cfg_attr(not(target_arch = "wasm32"), derive(Resource))]
#[derive(Default)]
struct GaussianLodPackageManager {
    clouds: HashMap<Entity, PackageInstantiation>,
    caches: PackageCacheRegistry,
}

impl GaussianLodPackageManager {
    fn prune_unused_caches(&mut self) {
        self.caches.prune_unused();
    }
}

#[derive(PartialEq)]
struct PackageBuildSignature<'a> {
    manifest: AssetId<GaussianLodAsset>,
    source: &'a GaussianLodPackageSource,
    config: &'a GaussianLodPackageConfig,
    streaming: &'a GaussianStreamingSettings,
    structural: PackageStructuralSignature,
    debug_metadata: bool,
}

/// Stable backend category retained by the package runtime instead of
/// immediately erasing transport failures into strings.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GaussianLodPackageTransportError {
    NativeFile(String),
    Http(String),
    PersistentCache(String),
}

impl fmt::Display for GaussianLodPackageTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for GaussianLodPackageTransportError {}

impl GaussianLodPackageTransportError {
    fn runtime_failure(&self) -> LodPageTransportFailure {
        match self {
            Self::NativeFile(_) | Self::Http(_) => {
                LodPageTransportFailure::transport(self.to_string())
            }
            Self::PersistentCache(_) => LodPageTransportFailure::cache(self.to_string()),
        }
    }
}

impl From<&GaussianLodPackageTransportError> for LodOrchestrationFailure {
    fn from(error: &GaussianLodPackageTransportError) -> Self {
        let code = match error {
            GaussianLodPackageTransportError::NativeFile(_)
            | GaussianLodPackageTransportError::Http(_) => {
                LodOrchestrationFailureCode::TransportRequestFailed
            }
            GaussianLodPackageTransportError::PersistentCache(_) => {
                LodOrchestrationFailureCode::CacheFailed
            }
        };
        Self::with_detail(code, error.to_string())
    }
}

fn map_package_poll<Error>(
    poll: PagePoll<Error>,
    map_error: impl FnOnce(Error) -> GaussianLodPackageTransportError,
) -> PagePoll<GaussianLodPackageTransportError> {
    match poll {
        PagePoll::Pending => PagePoll::Pending,
        PagePoll::Ready(payload) => PagePoll::Ready(payload),
        PagePoll::Failed(error) => PagePoll::Failed(map_error(error)),
    }
}

#[derive(Default)]
pub struct GaussianLodPackagePlugin;

impl Plugin for GaussianLodPackagePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GaussianLodPackageConfig>()
            .register_type::<GaussianLodPackageSource>()
            .register_required_components::<GaussianLodHandle, GaussianLodSettings>()
            .register_required_components::<GaussianLodHandle, CloudSettings>()
            .register_required_components::<GaussianLodHandle, Transform>()
            .register_required_components::<GaussianLodHandle, Visibility>();
        if !app.is_plugin_added::<GaussianLodAtlasUploadPlugin>() {
            app.add_plugins(GaussianLodAtlasUploadPlugin);
        }
        if !app.is_plugin_added::<GaussianLodRenderCommitPlugin>() {
            app.add_plugins(GaussianLodRenderCommitPlugin);
        }
        init_package_manager(app);
        app.add_systems(
            PostUpdate,
            update_lod_packages
                .after(CameraUpdateSystems)
                .after(VisibilitySystems::CheckVisibility)
                .after(TransformSystems::Propagate),
        )
        .add_systems(
            PostUpdate,
            publish_package_status_transitions.after(update_lod_packages),
        );
    }
}

fn publish_package_status_transitions(
    statuses: Query<(Entity, &GaussianLodPackageStatus), Changed<GaussianLodPackageStatus>>,
    mut removed: RemovedComponents<GaussianLodPackageStatus>,
    mut previous: Local<
        HashMap<Entity, (GaussianLodPackagePhase, Option<LodOrchestrationFailureCode>)>,
    >,
    mut transitions: MessageWriter<LodOrchestrationTransition>,
) {
    for entity in removed.read() {
        previous.remove(&entity);
    }
    for (entity, status) in &statuses {
        let next = (
            status.phase,
            status.failure.as_ref().map(LodOrchestrationFailure::code),
        );
        let old = previous.insert(entity, next);
        if old == Some(next) {
            continue;
        }
        let kind = match status.phase {
            GaussianLodPackagePhase::Failed => Some(LodOrchestrationTransitionKind::Failed),
            GaussianLodPackagePhase::Degraded => Some(LodOrchestrationTransitionKind::Degraded),
            GaussianLodPackagePhase::Active
                if old.is_some_and(|(phase, _)| {
                    matches!(
                        phase,
                        GaussianLodPackagePhase::Degraded | GaussianLodPackagePhase::Failed
                    )
                }) =>
            {
                Some(LodOrchestrationTransitionKind::Recovered)
            }
            GaussianLodPackagePhase::Loading | GaussianLodPackagePhase::Active => None,
        };
        if let Some(kind) = kind {
            transitions.write(LodOrchestrationTransition {
                entity,
                source: LodOrchestrationSource::Package,
                kind,
                failure: status.failure.clone(),
            });
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_lod_packages(
    mut commands: Commands,
    config: Res<GaussianLodPackageConfig>,
    mut manager: PackageManagerParam<'_>,
    manifests: Res<Assets<GaussianLodAsset>>,
    mut manifest_events: MessageReader<AssetEvent<GaussianLodAsset>>,
    mut clouds: ResMut<Assets<PlanarGaussian3d>>,
    mut atlas_uploads: ResMut<LodAtlasUploadQueue>,
    cameras: Query<(Entity, &Camera, &Projection, &GlobalTransform), With<crate::GaussianCamera>>,
    cloud_handles: Query<&PlanarGaussian3dHandle>,
    packages: Query<(
        Entity,
        &GaussianLodHandle,
        &GaussianLodPackageSource,
        &GaussianLodSettings,
        &CloudSettings,
        Option<&GaussianStreamingSettings>,
        Option<&ViewVisibility>,
        &GlobalTransform,
    )>,
) {
    let camera_views = collect_package_camera_views(&cameras, config.max_views_per_cloud);
    let changed_manifests = manifest_events
        .read()
        .filter_map(|event| match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::Removed { id }
            | AssetEvent::LoadedWithDependencies { id } => Some(*id),
            AssetEvent::Unused { .. } => None,
        })
        .collect::<HashSet<_>>();
    let mut seen = BTreeSet::new();
    for (
        entity,
        handle,
        source,
        settings,
        cloud_settings,
        per_cloud_streaming,
        visibility,
        transform,
    ) in &packages
    {
        seen.insert(entity);
        let effective_streaming =
            match package_streaming_settings(per_cloud_streaming.unwrap_or(&config.streaming)) {
                Ok(streaming) => streaming,
                Err(error) => {
                    if let Some(previous) = manager.clouds.remove(&entity) {
                        release_package_instantiation(
                            entity,
                            previous,
                            &mut clouds,
                            &mut commands,
                            &cloud_handles,
                            false,
                        );
                    }
                    commands
                        .entity(entity)
                        .remove::<LodRenderCandidates>()
                        .remove::<LodDebugMetadata>()
                        .insert(GaussianLodPackageStatus::failed(error));
                    continue;
                }
            };
        if let Err(error) = validate_package_render_path(&cloud_settings.sort_mode) {
            if let Some(previous) = manager.clouds.remove(&entity) {
                release_package_instantiation(
                    entity,
                    previous,
                    &mut clouds,
                    &mut commands,
                    &cloud_handles,
                    false,
                );
            }
            commands
                .entity(entity)
                .remove::<LodRenderCandidates>()
                .remove::<LodDebugMetadata>()
                .insert(GaussianLodPackageStatus::failed(error));
            continue;
        }
        let structural = PackageStructuralSignature::new(settings);
        let debug_metadata = cloud_settings.lod_debug.requires_metadata();
        let unchanged = !changed_manifests.contains(&handle.0.id())
            && manager.clouds.get(&entity).is_some_and(|state| {
                PackageBuildSignature {
                    manifest: state.manifest,
                    source: &state.source,
                    config: &state.config,
                    streaming: &state.streaming,
                    structural: state.requested_structural,
                    debug_metadata: state.debug.is_some(),
                } == PackageBuildSignature {
                    manifest: handle.0.id(),
                    source,
                    config: &config,
                    streaming: &effective_streaming,
                    structural,
                    debug_metadata,
                }
            });
        if !unchanged {
            if let Some(previous) = manager.clouds.remove(&entity) {
                release_package_instantiation(
                    entity,
                    previous,
                    &mut clouds,
                    &mut commands,
                    &cloud_handles,
                    false,
                );
            }
            commands
                .entity(entity)
                .remove::<LodRenderCandidates>()
                .insert(GaussianLodPackageStatus::loading());
            let Some(asset) = manifests.get(&handle.0) else {
                continue;
            };
            let result = instantiate_package(
                asset,
                source,
                settings,
                &config,
                &effective_streaming,
                debug_metadata,
                &mut manager,
                &mut clouds,
            );
            match result {
                Ok(state) => {
                    let bounds = asset.manifest.scene_bounds.map(|bounds| {
                        Aabb::from_min_max(Vec3::from(bounds.min), Vec3::from(bounds.max))
                    });
                    let atlas = state.atlas.clone();
                    manager.clouds.insert(
                        entity,
                        PackageInstantiation {
                            manifest: handle.0.id(),
                            source: source.clone(),
                            config: config.clone(),
                            ..state
                        },
                    );
                    let mut entity_commands = commands.entity(entity);
                    entity_commands.insert(PlanarGaussian3dHandle(atlas));
                    if let Some(bounds) = bounds {
                        entity_commands.insert(bounds);
                    }
                }
                Err(error) => {
                    commands
                        .entity(entity)
                        .insert(GaussianLodPackageStatus::failed(error));
                    continue;
                }
            }
        }

        let Some(state) = manager.clouds.get_mut(&entity) else {
            continue;
        };
        if visibility.is_some_and(|visibility| !visibility.get()) {
            match suspend_package_state(state) {
                Ok(()) => publish_package_state(entity, state, &mut commands),
                Err(error) => publish_package_failure(entity, state, error, &mut commands),
            }
            continue;
        }
        let views = match &camera_views {
            Ok(views) => views,
            Err(error) => {
                publish_package_failure(entity, state, error.clone(), &mut commands);
                continue;
            }
        };
        match drive_package_state(
            state,
            settings,
            transform,
            views,
            &mut clouds,
            &mut atlas_uploads,
        ) {
            Ok(()) => publish_package_state(entity, state, &mut commands),
            Err(error) => publish_package_failure(entity, state, error, &mut commands),
        }
    }

    let stale = manager
        .clouds
        .keys()
        .filter(|entity| !seen.contains(entity))
        .copied()
        .collect::<Vec<_>>();
    for entity in stale {
        let Some(state) = manager.clouds.remove(&entity) else {
            continue;
        };
        release_package_instantiation(
            entity,
            state,
            &mut clouds,
            &mut commands,
            &cloud_handles,
            true,
        );
    }
    manager.prune_unused_caches();
}

/// Releases camera/root fallback pins and cancels queued, in-flight, and
/// preprocessing work while an entity is not visible. The materialized atlas
/// and last complete render candidate remain available for a cheap resume, but
/// hidden clouds no longer consume asynchronous pipeline capacity forever.
fn suspend_package_state(state: &mut PackageInstantiation) -> Result<(), GaussianLodPackageError> {
    let camera_views = std::mem::take(&mut state.views);
    let runtime = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    let _ = runtime.transport_mut().maintain_cache()?;
    for view in camera_views {
        runtime
            .remove_view(LodRuntimeViewId(view.to_bits()))
            .map_err(GaussianLodPackageError::Runtime)?;
    }
    runtime
        .remove_view(PACKAGE_ROOT_FALLBACK_VIEW)
        .map_err(GaussianLodPackageError::Runtime)?;
    let frame = runtime.begin_frame();
    runtime
        .finish_frame(frame)
        .map_err(GaussianLodPackageError::Runtime)?;
    state.resident_pages = runtime.cache().stats().resident_pages;
    Ok(())
}

fn release_package_instantiation(
    entity: Entity,
    state: PackageInstantiation,
    clouds: &mut Assets<PlanarGaussian3d>,
    commands: &mut Commands,
    cloud_handles: &Query<&PlanarGaussian3dHandle>,
    remove_status: bool,
) {
    clouds.remove(state.atlas.id());
    let Ok(mut entity_commands) = commands.get_entity(entity) else {
        return;
    };
    entity_commands
        .remove::<LodRenderCandidates>()
        .remove::<LodDebugMetadata>();
    if remove_status {
        entity_commands.remove::<GaussianLodPackageStatus>();
    }
    if cloud_handles
        .get(entity)
        .is_ok_and(|handle| handle.handle().id() == state.atlas.id())
    {
        entity_commands
            .remove::<PlanarGaussian3dHandle>()
            .remove::<Aabb>();
    }
}

fn package_http_config(
    base_url: &str,
    streaming: &GaussianStreamingSettings,
) -> Result<HttpRangeTransportConfig, GaussianLodPackageError> {
    streaming
        .validate()
        .map_err(|error| GaussianLodPackageError::InvalidStreaming(error.to_string()))?;
    let timeout = Duration::try_from_secs_f32(streaming.request_timeout_seconds)
        .map_err(|error| GaussianLodPackageError::InvalidStreaming(error.to_string()))?;
    let retry_base = Duration::try_from_secs_f32(streaming.retry_base_delay_seconds)
        .map_err(|error| GaussianLodPackageError::InvalidStreaming(error.to_string()))?;
    let retry_multiplier = 1_u32
        .checked_shl(streaming.retry_limit.min(31))
        .unwrap_or(u32::MAX);
    let retry_max = retry_base.saturating_mul(retry_multiplier).max(retry_base);
    let http = HttpRangeTransportConfig {
        base_url: base_url.to_owned(),
        request_timeout: timeout,
        retry_limit: streaming.retry_limit,
        retry_base_delay: retry_base,
        retry_max_delay: retry_max,
        max_encoded_page_bytes: streaming.effective_max_encoded_page_bytes(),
        max_concurrent_requests: streaming.max_concurrent_requests,
        require_content_length: true,
        require_object_validator: true,
        object_version_header: None,
    };
    http.validate()
        .map_err(|error| GaussianLodPackageError::HttpTransport(error.to_string()))?;
    Ok(http)
}

fn validated_cache_namespace(
    config: &GaussianLodPackageConfig,
) -> Result<&str, GaussianLodPackageError> {
    let namespace = config
        .persistent_cache_namespace
        .as_deref()
        .ok_or(GaussianLodPackageError::MissingPersistentCacheNamespace)?;
    if namespace.is_empty()
        || namespace.len() > 96
        || !namespace
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(GaussianLodPackageError::InvalidPersistentCacheNamespace(
            namespace.to_owned(),
        ));
    }
    Ok(namespace)
}

fn package_cache_name(
    manifest: &crate::GaussianLodManifest,
    config: &GaussianLodPackageConfig,
) -> Result<String, GaussianLodPackageError> {
    let namespace = validated_cache_namespace(config)?;
    let identity = PersistentCachePackageIdentity::from_manifest(manifest)
        .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string()))?;
    Ok(format!("{namespace}-{:016x}", identity.stable_hash()))
}

#[allow(clippy::too_many_arguments)]
fn instantiate_package(
    asset: &GaussianLodAsset,
    source: &GaussianLodPackageSource,
    settings: &GaussianLodSettings,
    config: &GaussianLodPackageConfig,
    streaming: &GaussianStreamingSettings,
    debug_metadata: bool,
    manager: &mut GaussianLodPackageManager,
    clouds: &mut Assets<PlanarGaussian3d>,
) -> Result<PackageInstantiation, GaussianLodPackageError> {
    let plan = GaussianLodPackageAtlasPlan::from_manifest(&asset.manifest, settings, config)?;
    let transport = package_page_transport(
        &asset.manifest,
        source,
        config,
        streaming,
        &mut manager.caches,
    )?;
    let runtime_streaming = package_runtime_streaming_settings(source, streaming);
    let physical_len = plan.physical_gaussians as usize;
    let mut physical = Vec::new();
    physical.try_reserve_exact(physical_len).map_err(|_| {
        GaussianLodPackageError::AtlasAllocationFailed {
            gaussian_count: plan.physical_gaussians,
            bytes: plan.physical_bytes,
        }
    })?;
    physical.resize(physical_len, Gaussian3d::default());

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
    let runtime = LodStreamingRuntime::new(
        asset.manifest.clone(),
        transport,
        &effective,
        &runtime_streaming,
    )
    .map_err(GaussianLodPackageError::Runtime)?;
    let mirror = LodPageAtlasMirror::new(runtime.atlas_layout(), plan.slot_count)
        .map_err(GaussianLodPackageError::RenderCommit)?;
    let debug = if debug_metadata {
        let index = LodDebugManifestIndex::new(&asset.manifest)
            .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?;
        let atlas = LodDebugAnnotationAtlas::new(plan.slot_count, plan.gaussians_per_slot)
            .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?;
        Some(PackageDebugAnnotations { atlas, index })
    } else {
        None
    };
    let atlas = clouds.add(PlanarGaussian3d::from(physical));
    Ok(PackageInstantiation {
        manifest: AssetId::default(),
        source: source.clone(),
        config: config.clone(),
        atlas,
        plan,
        runtime: Mutex::new(runtime),
        mirror,
        debug,
        current: None,
        pending: None,
        current_fallback_nodes: BTreeSet::new(),
        pending_fallback_nodes: BTreeSet::new(),
        views: BTreeSet::new(),
        visible_slots: BTreeMap::new(),
        visible_ranges: Vec::new(),
        visible_fallback_nodes: BTreeSet::new(),
        requested_structural: PackageStructuralSignature::new(settings),
        structural,
        runtime_streaming,
        streaming: streaming.clone(),
        preprocess_cache_repairs: BTreeSet::new(),
        resident_pages: 0,
        active_gaussians: 0,
        terminal_failures: 0,
        root_fallback: false,
        last_failure: None,
    })
}

fn package_runtime_streaming_settings(
    source: &GaussianLodPackageSource,
    streaming: &GaussianStreamingSettings,
) -> GaussianStreamingSettings {
    let mut runtime = streaming.clone();
    if matches!(source, GaussianLodPackageSource::Url { .. }) {
        // HttpRangePageTransport owns the single configured retry/backoff
        // budget. Disable generic outer retries to avoid (R + 1)^2 requests.
        runtime.retry_limit = 0;
    }
    runtime
}

fn package_sort_is_supported(sort_mode: &crate::sort::SortMode) -> bool {
    #[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
    {
        matches!(sort_mode, crate::sort::SortMode::Radix)
    }
    #[cfg(not(all(feature = "sort_radix", not(feature = "buffer_texture"))))]
    {
        let _ = sort_mode;
        false
    }
}

fn validate_package_render_path(
    sort_mode: &crate::sort::SortMode,
) -> Result<(), GaussianLodPackageError> {
    require_lod_render_path().map_err(GaussianLodPackageError::UnsupportedRenderPath)?;
    if package_sort_is_supported(sort_mode) {
        Ok(())
    } else {
        Err(GaussianLodPackageError::UnsupportedSortMode(format!(
            "{sort_mode:?}"
        )))
    }
}

#[derive(Clone, Copy)]
struct PackageCameraView {
    entity: Entity,
    view: LodView,
}

fn collect_package_camera_views(
    cameras: &Query<(Entity, &Camera, &Projection, &GlobalTransform), With<crate::GaussianCamera>>,
    max_views: u32,
) -> Result<Vec<PackageCameraView>, GaussianLodPackageError> {
    let active = cameras
        .iter()
        .filter(|(_, camera, _, _)| camera.is_active)
        .collect::<Vec<_>>();
    if active.len() > max_views as usize {
        return Err(GaussianLodPackageError::ViewLimitExceeded {
            actual: active.len() as u64,
            limit: max_views,
        });
    }
    let mut views = Vec::with_capacity(active.len());
    for (entity, camera, projection, transform) in active {
        let viewport_height = camera
            .physical_viewport_size()
            .map(|size| size.y as f32)
            .filter(|height| *height > 0.0)
            .ok_or(GaussianLodPackageError::UnsupportedCamera(entity))?;
        let clip_from_world = projection.get_clip_from_view() * transform.to_matrix().inverse();
        let view = match projection {
            Projection::Perspective(perspective) => LodView::perspective(
                transform.translation(),
                viewport_height,
                perspective.fov,
                perspective.near.max(f32::EPSILON),
            ),
            Projection::Orthographic(orthographic) => LodView::orthographic(
                transform.translation(),
                viewport_height,
                (orthographic.area.max.y - orthographic.area.min.y)
                    .abs()
                    .max(f32::EPSILON),
                orthographic.near.abs().max(f32::EPSILON),
            ),
            Projection::Custom(_) => {
                return Err(GaussianLodPackageError::UnsupportedCamera(entity));
            }
        };
        views.push(PackageCameraView {
            entity,
            view: view.with_clip_from_world(clip_from_world),
        });
    }
    views.sort_by_key(|view| view.entity);
    Ok(views)
}

fn drive_package_state(
    state: &mut PackageInstantiation,
    settings: &GaussianLodSettings,
    transform: &GlobalTransform,
    camera_views: &[PackageCameraView],
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), GaussianLodPackageError> {
    if let Some(pending) = state.pending.as_ref() {
        if pending
            .by_camera
            .values()
            .any(|candidate| candidate.failed())
        {
            state.pending = None;
            state.pending_fallback_nodes.clear();
            // The render world has overwritten/rejected the previous GPU
            // candidate state. Keep the complete root-only fallback atlas and
            // require a fresh handshake instead of relabeling stale state.
            state.current = None;
            state.current_fallback_nodes.clear();
            state.last_failure = Some(LodOrchestrationFailure::with_detail(
                LodOrchestrationFailureCode::RenderCommitFailed,
                "render candidate commit failed; retained complete root fallback",
            ));
            return Ok(());
        }
        if pending
            .by_camera
            .values()
            .all(|candidate| candidate.render_is_prepared())
        {
            let pending = state.pending.take().expect("checked pending cut");
            let fallback_nodes = std::mem::take(&mut state.pending_fallback_nodes);
            materialize_package_cut(state, &pending, &fallback_nodes, assets, atlas_uploads)?;
            for candidate in pending.by_camera.values() {
                if !candidate.activate() {
                    return Err(GaussianLodPackageError::RenderCommitFailed);
                }
            }
            state.active_gaussians = pending
                .by_camera
                .values()
                .map(|candidate| u64::from(candidate.rendered_candidate_count()))
                .max()
                .unwrap_or(0);
            state.current = Some(pending);
            state.current_fallback_nodes = fallback_nodes;
            state.root_fallback = false;
            state.last_failure = None;
        }
        // Do not poll or evict while render-world staging is in progress. This
        // keeps the last complete cut's physical slots valid until activation.
        return Ok(());
    }

    if state
        .current
        .as_ref()
        .is_some_and(|current| !package_candidate_set_is_active(current))
    {
        state.current = None;
        state.current_fallback_nodes.clear();
        state.last_failure = Some(LodOrchestrationFailure::with_detail(
            LodOrchestrationFailureCode::RenderCommitFailed,
            "render state was revoked; staging a fresh cut from root fallback",
        ));
    }

    let next_views = camera_views
        .iter()
        .map(|view| view.entity)
        .collect::<BTreeSet<_>>();
    let effective = state.structural.apply(settings);
    // A package bridge never presents a partial forest as a complete scene.
    let runtime = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    let _ = runtime.transport_mut().maintain_cache()?;
    for removed in state
        .views
        .difference(&next_views)
        .copied()
        .collect::<Vec<_>>()
    {
        runtime
            .remove_view(LodRuntimeViewId(removed.to_bits()))
            .map_err(GaussianLodPackageError::Runtime)?;
    }
    state.views = next_views;

    let frame = runtime.begin_frame();
    let world_from_local = transform.to_matrix();
    let mut root_settings = effective.clone();
    root_settings.quality = 0.0;
    root_settings.frustum_culling = false;
    let root_frame = runtime
        .update_view_in_frame(
            frame,
            PACKAGE_ROOT_FALLBACK_VIEW,
            LodView::perspective(Vec3::ZERO, 1.0, 1.0, 0.1).with_world_from_local(world_from_local),
            &root_settings,
            &state.runtime_streaming,
        )
        .map_err(GaussianLodPackageError::Runtime)?;
    let mut stream_frames = Vec::with_capacity(camera_views.len());
    for camera in camera_views {
        let stream_frame = runtime
            .update_view_in_frame(
                frame,
                LodRuntimeViewId(camera.entity.to_bits()),
                camera.view.with_world_from_local(world_from_local),
                &effective,
                &state.runtime_streaming,
            )
            .map_err(GaussianLodPackageError::Runtime)?;
        state.resident_pages = stream_frame.cache_stats().resident_pages;
        stream_frames.push((camera.entity, stream_frame));
    }
    runtime
        .finish_frame(frame)
        .map_err(GaussianLodPackageError::Runtime)?;
    let rejected_cached_pages = std::iter::once(&root_frame)
        .chain(stream_frames.iter().map(|(_, stream_frame)| stream_frame))
        .flat_map(|stream_frame| stream_frame.preprocess_failed_pages().iter().copied())
        .collect::<BTreeSet<_>>();
    for page in rejected_cached_pages {
        // Encoded cache records are integrity-checked without duplicating the
        // full codec decode. A typed downstream preprocessing rejection is the
        // authoritative signal to evict the record before the next retry.
        runtime.transport_mut().invalidate_cached_page(page)?;
        let first_cache_repair =
            state.streaming.persistent_cache && state.preprocess_cache_repairs.insert(page);
        if first_cache_repair && runtime.is_terminal_failure(page) {
            // A bad cache hit can exhaust a zero outer retry budget without ever
            // reaching the canonical transport. Grant exactly one explicit
            // repair after invalidation; a second preprocessing rejection stays
            // terminal because the page remains in this bounded set.
            let retried = runtime
                .retry_terminal_failure(page)
                .map_err(GaussianLodPackageError::Runtime)?;
            debug_assert!(retried, "the page was terminal immediately above");
        }
    }
    let _ = runtime.transport_mut().maintain_cache()?;
    for stream_frame in std::iter::once(&root_frame)
        .chain(stream_frames.iter().map(|(_, stream_frame)| stream_frame))
    {
        for &page in stream_frame.completed_pages() {
            state.preprocess_cache_repairs.remove(&page);
            let slot = runtime
                .cache()
                .get(page)
                .map(|resident| resident.slot)
                .ok_or(GaussianLodPackageError::CompletedPageNotResident(page))?;
            state
                .mirror
                .stage_page(page, slot)
                .map_err(GaussianLodPackageError::RenderCommit)?;
        }
    }
    let root_frontier =
        match root_frame.candidate_frontier(root_settings.max_active_gaussians_u32()) {
            Ok(frontier) => Some(frontier),
            Err(LodRuntimeError::NoResidentFrontier) => None,
            Err(error) => return Err(GaussianLodPackageError::Runtime(error)),
        };
    if camera_views.is_empty() {
        state.resident_pages = root_frame.cache_stats().resident_pages;
    }
    let fallback_nodes = selected_package_ancestor_fallback_nodes(runtime, &stream_frames);
    let mut candidates = LodRenderCandidates::default();
    let mut complete = true;
    for (camera, stream_frame) in &stream_frames {
        match stream_frame.candidate_frontier(effective.max_active_gaussians_u32()) {
            Ok(frontier) => {
                let mut candidate = LodRenderCandidate::new(frontier);
                if let Some(previous) = state
                    .current
                    .as_ref()
                    .and_then(|current| current.get(*camera))
                    .filter(|previous| {
                        previous.phase.load(Ordering::Acquire) == LOD_RENDER_ACTIVE
                            && previous.same_payload(&candidate)
                    })
                {
                    candidate.phase = previous.phase.clone();
                }
                candidates.by_camera.insert(*camera, candidate);
            }
            Err(LodRuntimeError::NoResidentFrontier) => complete = false,
            Err(error) => return Err(GaussianLodPackageError::Runtime(error)),
        }
    }
    state.terminal_failures = runtime
        .terminal_failures()
        .len()
        .try_into()
        .unwrap_or(u32::MAX);
    if state.terminal_failures > 0 {
        state.last_failure = Some(terminal_runtime_failure(runtime, state.terminal_failures));
    } else {
        state.last_failure = None;
    }
    if camera_views.is_empty() {
        if let Some(root_frontier) = root_frontier
            && !state.root_fallback
        {
            materialize_package_frontiers(
                state,
                std::slice::from_ref(&root_frontier),
                &BTreeSet::new(),
                assets,
                atlas_uploads,
            )?;
            state.root_fallback = true;
            state.current = None;
            state.current_fallback_nodes.clear();
            state.active_gaussians = u64::from(root_frontier.candidate_count());
        }
        return Ok(());
    }
    if !complete || candidates.len() != camera_views.len() {
        return Ok(());
    }
    let debug_fallback_nodes = fallback_nodes.clone();
    if let Some(current) = state.current.as_ref()
        && package_candidate_sets_equal(current, &candidates)
    {
        if state.current_fallback_nodes != fallback_nodes
            || state.visible_fallback_nodes != debug_fallback_nodes
        {
            let current = current.clone();
            materialize_package_cut(
                state,
                &current,
                &debug_fallback_nodes,
                assets,
                atlas_uploads,
            )?;
            state.current_fallback_nodes = fallback_nodes;
        }
        state.active_gaussians = candidates
            .by_camera
            .values()
            .map(|candidate| u64::from(candidate.rendered_candidate_count()))
            .max()
            .unwrap_or(0);
        state.current = Some(candidates);
        return Ok(());
    }
    let Some(root_frontier) = root_frontier else {
        return Ok(());
    };
    materialize_package_frontiers(
        state,
        std::slice::from_ref(&root_frontier),
        &BTreeSet::new(),
        assets,
        atlas_uploads,
    )?;
    state.root_fallback = true;
    state.current = None;
    state.current_fallback_nodes.clear();
    state.active_gaussians = u64::from(root_frontier.candidate_count());
    state.pending = Some(candidates);
    state.pending_fallback_nodes = fallback_nodes;
    Ok(())
}

fn terminal_requests_exhausted_failure(count: u32) -> LodOrchestrationFailure {
    LodOrchestrationFailure::with_detail(
        LodOrchestrationFailureCode::TransportRequestsExhausted,
        format!("{count} page request(s) exhausted retry budget; retained complete ancestor cut"),
    )
}

fn terminal_runtime_failure<T: LodPageTransport>(
    runtime: &LodStreamingRuntime<T>,
    count: u32,
) -> LodOrchestrationFailure {
    if let Some((page, error)) = runtime.terminal_failures().iter().find_map(|page| {
        runtime
            .page_preprocess_error(*page)
            .map(|error| (*page, error))
    }) {
        return LodOrchestrationFailure::with_detail(
            LodOrchestrationFailureCode::DecodeValidationFailed,
            format!(
                "page {} failed decode/validation after bounded retries: {error}; {count} terminal page(s), retained complete ancestor cut",
                page.0
            ),
        );
    }
    if let Some((page, error)) = runtime.terminal_failures().iter().find_map(|page| {
        runtime
            .page_transport_failure(*page)
            .map(|error| (*page, error))
    }) {
        let code = match error.kind() {
            LodPageTransportFailureKind::Transport => {
                LodOrchestrationFailureCode::TransportRequestFailed
            }
            LodPageTransportFailureKind::Cache => LodOrchestrationFailureCode::CacheFailed,
        };
        return LodOrchestrationFailure::with_detail(
            code,
            format!(
                "page {} failed after bounded retries: {}; {count} terminal page(s), retained complete ancestor cut",
                page.0,
                error.detail()
            ),
        );
    }
    terminal_requests_exhausted_failure(count)
}

fn package_candidate_sets_equal(left: &LodRenderCandidates, right: &LodRenderCandidates) -> bool {
    left.by_camera.len() == right.by_camera.len()
        && left.by_camera.iter().all(|(camera, candidate)| {
            right
                .by_camera
                .get(camera)
                .is_some_and(|other| candidate.same_payload(other))
        })
}

fn selected_package_ancestor_fallback_nodes<T: LodPageTransport>(
    runtime: &LodStreamingRuntime<T>,
    frames: &[(Entity, LodStreamFrame)],
) -> BTreeSet<LodNodeId> {
    let mut nodes = BTreeSet::new();
    for (_, frame) in frames {
        let selected = frame
            .frontier()
            .nodes
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        for &requested in &frame.frontier().requested_nodes {
            let mut cursor = runtime
                .hierarchy()
                .node(requested)
                .and_then(|node| node.parent);
            while let Some(ancestor) = cursor {
                if selected.contains(&ancestor) {
                    nodes.insert(ancestor);
                    break;
                }
                cursor = runtime
                    .hierarchy()
                    .node(ancestor)
                    .and_then(|node| node.parent);
            }
        }
    }
    nodes
}

fn package_candidate_set_is_active(candidates: &LodRenderCandidates) -> bool {
    !candidates.is_empty()
        && candidates
            .by_camera
            .values()
            .all(|candidate| candidate.phase.load(Ordering::Acquire) == LOD_RENDER_ACTIVE)
}

fn materialize_package_cut(
    state: &mut PackageInstantiation,
    cut: &LodRenderCandidates,
    fallback_nodes: &BTreeSet<LodNodeId>,
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), GaussianLodPackageError> {
    let ranges = cut
        .by_camera
        .values()
        .flat_map(LodRenderCandidate::render_ranges)
        .copied()
        .collect::<Vec<_>>();
    materialize_package_ranges(state, &ranges, fallback_nodes, assets, atlas_uploads)
}

fn materialize_package_frontiers(
    state: &mut PackageInstantiation,
    frontiers: &[LodCandidateFrontier],
    fallback_nodes: &BTreeSet<LodNodeId>,
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), GaussianLodPackageError> {
    let ranges = frontiers
        .iter()
        .flat_map(|frontier| frontier.physical_ranges())
        .copied()
        .collect::<Vec<_>>();
    materialize_package_ranges(state, &ranges, fallback_nodes, assets, atlas_uploads)
}

fn materialize_package_ranges(
    state: &mut PackageInstantiation,
    ranges: &[LodPhysicalRange],
    fallback_nodes: &BTreeSet<LodNodeId>,
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), GaussianLodPackageError> {
    let normalized_ranges = normalize_package_ranges(ranges);
    if state.visible_ranges == normalized_ranges && state.visible_fallback_nodes == *fallback_nodes
    {
        return Ok(());
    }
    validate_atomic_package_upload_commit_ranges(
        state.plan,
        &normalized_ranges,
        &state.visible_slots,
        state.requested_structural.max_gpu_upload_bytes_per_commit,
    )?;
    let runtime = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    let atlas = assets
        .get_mut_untracked(&state.atlas)
        .ok_or(GaussianLodPackageError::MissingAtlasAsset)?;
    let rewrite = rewrite_atlas_to_ranges(
        runtime,
        &mut state.mirror,
        state.debug.as_mut(),
        state.plan,
        &normalized_ranges,
        fallback_nodes,
        &state.visible_slots,
        atlas,
    )?;
    enqueue_package_atlas_uploads(atlas_uploads, state.atlas.id(), state.plan, &rewrite)?;
    state.visible_slots = rewrite.selected_slots;
    state.visible_ranges = normalized_ranges;
    state.visible_fallback_nodes.clone_from(fallback_nodes);
    Ok(())
}

fn normalize_package_ranges(ranges: &[LodPhysicalRange]) -> Vec<LodPhysicalRange> {
    let mut normalized = ranges.to_vec();
    normalized.sort_by_key(|range| {
        (
            range.node.0,
            range.page.0,
            range.slot.index,
            range.slot.generation,
            range.physical_start,
            range.count,
        )
    });
    normalized.dedup();
    normalized
}

fn validate_atomic_package_upload_commit_ranges(
    plan: GaussianLodPackageAtlasPlan,
    ranges: &[LodPhysicalRange],
    previous_slots: &BTreeMap<u32, AtlasSlot>,
    limit: u64,
) -> Result<(), GaussianLodPackageError> {
    let mut dirty_slots = previous_slots.keys().copied().collect::<BTreeSet<_>>();
    for range in ranges {
        if range.slot.index >= plan.slot_count {
            return Err(GaussianLodPackageError::AtlasSizeOverflow);
        }
        dirty_slots.insert(range.slot.index);
    }
    let bytes_per_slot = u64::from(plan.gaussians_per_slot)
        .checked_mul(gaussian_3d_gpu_bytes_per_record())
        .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
    let dirty_slot_count =
        u64::try_from(dirty_slots.len()).map_err(|_| GaussianLodPackageError::AtlasSizeOverflow)?;
    let bytes = dirty_slot_count
        .checked_mul(bytes_per_slot)
        .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
    if bytes > limit {
        return Err(GaussianLodPackageError::GpuUploadCommitTooLarge {
            dirty_slots: dirty_slot_count,
            bytes,
            limit,
        });
    }
    Ok(())
}

#[derive(Debug)]
struct PackageAtlasRewrite {
    dirty_slots: BTreeSet<u32>,
    selected_slots: BTreeMap<u32, AtlasSlot>,
    #[cfg(all(test, not(target_arch = "wasm32")))]
    selection_scratch: PackageAtlasSelectionScratch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PackageAtlasSelectedInterval {
    start: u32,
    end: u32,
}

#[derive(Debug)]
struct PackageAtlasSparseSelection {
    intervals_by_slot: BTreeMap<u32, Vec<PackageAtlasSelectedInterval>>,
    selected_slots: BTreeMap<u32, AtlasSlot>,
    materializations: BTreeSet<(LodPageId, AtlasSlot)>,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PackageAtlasSelectionScratch {
    slots: usize,
    intervals: usize,
    materializations: usize,
}

impl PackageAtlasSparseSelection {
    #[cfg(all(test, not(target_arch = "wasm32")))]
    fn scratch(&self) -> PackageAtlasSelectionScratch {
        PackageAtlasSelectionScratch {
            slots: self.intervals_by_slot.len(),
            intervals: self.intervals_by_slot.values().map(Vec::len).sum(),
            materializations: self.materializations.len(),
        }
    }
}

fn package_atlas_slot_bounds(
    plan: GaussianLodPackageAtlasPlan,
    slot_index: u32,
) -> Result<(u32, u32), GaussianLodPackageError> {
    if slot_index >= plan.slot_count {
        return Err(GaussianLodPackageError::AtlasSizeOverflow);
    }
    let start = slot_index
        .checked_mul(plan.gaussians_per_slot)
        .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
    let end = start
        .checked_add(plan.gaussians_per_slot)
        .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
    if end > plan.physical_gaussians {
        return Err(GaussianLodPackageError::AtlasSizeOverflow);
    }
    Ok((start, end))
}

fn plan_package_atlas_selection(
    plan: GaussianLodPackageAtlasPlan,
    ranges: &[LodPhysicalRange],
) -> Result<PackageAtlasSparseSelection, GaussianLodPackageError> {
    let mut intervals_by_slot = BTreeMap::<u32, Vec<PackageAtlasSelectedInterval>>::new();
    let mut selected_slots = BTreeMap::new();
    let mut selected_pages = BTreeMap::new();
    let mut materializations = BTreeSet::new();

    for &range in ranges {
        let (slot_start, slot_end) = package_atlas_slot_bounds(plan, range.slot.index)?;
        let range_end = range
            .end()
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        if range.physical_start < slot_start || range_end > slot_end {
            return Err(GaussianLodPackageError::RenderCommit(
                LodRenderCommitError::FrontierReferencesUnsynchronizedPage {
                    page: range.page,
                    slot: range.slot,
                },
            ));
        }
        if let Some(previous) = selected_slots.insert(range.slot.index, range.slot)
            && previous != range.slot
        {
            return Err(GaussianLodPackageError::ConflictingAtlasSlot {
                index: range.slot.index,
                first: previous,
                second: range.slot,
            });
        }
        if let Some(previous) = selected_pages.insert(range.slot.index, range.page)
            && previous != range.page
        {
            return Err(GaussianLodPackageError::ConflictingAtlasPage {
                index: range.slot.index,
                first: previous,
                second: range.page,
            });
        }
        materializations.insert((range.page, range.slot));
        if range.count != 0 {
            intervals_by_slot.entry(range.slot.index).or_default().push(
                PackageAtlasSelectedInterval {
                    start: range.physical_start,
                    end: range_end,
                },
            );
        }
    }

    for intervals in intervals_by_slot.values_mut() {
        intervals.sort_unstable_by_key(|interval| (interval.start, interval.end));
        for pair in intervals.windows(2) {
            if pair[1].start < pair[0].end {
                return Err(GaussianLodPackageError::Runtime(
                    LodRuntimeError::OverlappingPhysicalRanges {
                        previous_end: pair[0].end,
                        next_start: pair[1].start,
                    },
                ));
            }
        }
    }

    Ok(PackageAtlasSparseSelection {
        intervals_by_slot,
        selected_slots,
        materializations,
    })
}

fn enqueue_package_atlas_uploads(
    atlas_uploads: &mut LodAtlasUploadQueue,
    atlas: AssetId<PlanarGaussian3d>,
    plan: GaussianLodPackageAtlasPlan,
    rewrite: &PackageAtlasRewrite,
) -> Result<(), GaussianLodPackageError> {
    for &index in &rewrite.dirty_slots {
        if let Some(&slot) = rewrite.selected_slots.get(&index) {
            atlas_uploads
                .enqueue_slot(atlas, slot, plan.gaussians_per_slot)
                .map_err(|error| GaussianLodPackageError::AtlasUpload(error.to_string()))?;
        } else {
            atlas_uploads
                .enqueue_cleared_slot(atlas, index, plan.gaussians_per_slot)
                .map_err(|error| GaussianLodPackageError::AtlasUpload(error.to_string()))?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(test, not(target_arch = "wasm32")))]
fn rewrite_atlas_to_frontiers<T: LodPageTransport>(
    runtime: &LodStreamingRuntime<T>,
    mirror: &mut LodPageAtlasMirror,
    debug: Option<&mut PackageDebugAnnotations>,
    plan: GaussianLodPackageAtlasPlan,
    frontiers: &[LodCandidateFrontier],
    fallback_nodes: &BTreeSet<LodNodeId>,
    previous_slots: &BTreeMap<u32, AtlasSlot>,
    atlas: &mut PlanarGaussian3d,
) -> Result<PackageAtlasRewrite, GaussianLodPackageError> {
    let ranges = frontiers
        .iter()
        .flat_map(|frontier| frontier.physical_ranges())
        .copied()
        .collect::<Vec<_>>();
    let ranges = normalize_package_ranges(&ranges);
    rewrite_atlas_to_ranges(
        runtime,
        mirror,
        debug,
        plan,
        &ranges,
        fallback_nodes,
        previous_slots,
        atlas,
    )
}

#[allow(clippy::too_many_arguments)]
fn rewrite_atlas_to_ranges<T: LodPageTransport>(
    runtime: &LodStreamingRuntime<T>,
    mirror: &mut LodPageAtlasMirror,
    mut debug: Option<&mut PackageDebugAnnotations>,
    plan: GaussianLodPackageAtlasPlan,
    ranges: &[LodPhysicalRange],
    fallback_nodes: &BTreeSet<LodNodeId>,
    previous_slots: &BTreeMap<u32, AtlasSlot>,
    atlas: &mut PlanarGaussian3d,
) -> Result<PackageAtlasRewrite, GaussianLodPackageError> {
    if atlas.len() != plan.physical_gaussians as usize {
        return Err(GaussianLodPackageError::RenderCommit(
            LodRenderCommitError::AtlasLengthMismatch {
                expected: plan.physical_gaussians,
                actual: atlas.len(),
            },
        ));
    }
    let selection = plan_package_atlas_selection(plan, ranges)?;
    #[cfg(all(test, not(target_arch = "wasm32")))]
    let selection_scratch = selection.scratch();
    let PackageAtlasSparseSelection {
        intervals_by_slot,
        selected_slots,
        materializations,
    } = selection;
    for (page_id, slot) in materializations {
        let page = runtime
            .decoded_page(page_id)
            .ok_or(GaussianLodPackageError::ResidentPageNotDecoded(page_id))?;
        mirror
            .materialize_page(atlas, page, slot)
            .map_err(GaussianLodPackageError::RenderCommit)?;
        if let Some(debug) = debug.as_deref_mut() {
            debug
                .atlas
                .write_page_indexed_with_node_residency(&debug.index, page, slot, |node| {
                    if fallback_nodes.contains(&node) {
                        LodDebugResidency::AncestorFallback
                    } else {
                        LodDebugResidency::Resident
                    }
                })
                .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?;
        }
    }
    mirror
        .validate_ranges(ranges)
        .map_err(GaussianLodPackageError::RenderCommit)?;

    let dirty_slots = previous_slots
        .keys()
        .chain(selected_slots.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    for &slot_index in &dirty_slots {
        let (slot_start, slot_end) = package_atlas_slot_bounds(plan, slot_index)?;
        if !selected_slots.contains_key(&slot_index) {
            let previous = previous_slots
                .get(&slot_index)
                .copied()
                .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
            if let Some(debug) = debug.as_deref_mut() {
                debug.atlas.clear_slot(previous).map_err(|error| {
                    GaussianLodPackageError::DebugAnnotations(error.to_string())
                })?;
            }
        }
        let mut clear_start = slot_start;
        if let Some(intervals) = intervals_by_slot.get(&slot_index) {
            for interval in intervals {
                for index in clear_start..interval.start {
                    Planar::set(&mut *atlas, index as usize, Gaussian3d::default());
                }
                clear_start = interval.end;
            }
        }
        for index in clear_start..slot_end {
            Planar::set(&mut *atlas, index as usize, Gaussian3d::default());
        }
    }
    Ok(PackageAtlasRewrite {
        dirty_slots,
        selected_slots,
        #[cfg(all(test, not(target_arch = "wasm32")))]
        selection_scratch,
    })
}

fn publish_package_state(
    entity: Entity,
    state: &mut PackageInstantiation,
    commands: &mut Commands,
) {
    let candidates = state.pending.as_ref().or(state.current.as_ref());
    let mut entity_commands = commands.entity(entity);
    if let Some(candidates) = candidates {
        entity_commands.insert(candidates.clone());
    } else {
        entity_commands.remove::<LodRenderCandidates>();
    }
    if let Some(debug) = &state.debug {
        entity_commands.insert(debug.atlas.metadata());
    } else {
        entity_commands.remove::<LodDebugMetadata>();
    }
    let phase = if state.current.is_some() {
        if state.terminal_failures > 0 || state.last_failure.is_some() {
            GaussianLodPackagePhase::Degraded
        } else {
            GaussianLodPackagePhase::Active
        }
    } else if state.root_fallback {
        GaussianLodPackagePhase::Degraded
    } else if state.terminal_failures > 0 || state.last_failure.is_some() {
        GaussianLodPackagePhase::Failed
    } else {
        GaussianLodPackagePhase::Loading
    };
    entity_commands.insert(GaussianLodPackageStatus {
        phase,
        resident_pages: state.resident_pages,
        active_gaussians: state.active_gaussians,
        terminal_failures: state.terminal_failures,
        failure: state.last_failure.clone(),
    });
}

fn publish_package_failure(
    entity: Entity,
    state: &mut PackageInstantiation,
    error: GaussianLodPackageError,
    commands: &mut Commands,
) {
    state.pending = None;
    state.last_failure = Some(LodOrchestrationFailure::from(&error));
    publish_package_state(entity, state, commands);
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum GaussianLodPackageError {
    ZeroLimit(&'static str),
    InvalidManifest(String),
    InvalidLodSettings(String),
    InvalidStreaming(String),
    UnsupportedSortMode(String),
    UnsupportedRenderPath(LodRenderPathSupportError),
    NativeSourceUnsupportedInBrowser,
    EmptyNativeRoot,
    UnsupportedUrlScheme(String),
    MissingPersistentCacheRoot,
    InvalidPersistentCacheRoot(String),
    MissingPersistentCacheNamespace,
    InvalidPersistentCacheNamespace(String),
    ManifestHasNoPages,
    AtlasCannotFitPage {
        gaussians_per_slot: u32,
        bytes_per_slot: u64,
    },
    RootFallbackExceedsAtlas {
        root_pages: u64,
        slots: u32,
    },
    RootFallbackExceedsGpuUploadBudget {
        root_pages: u64,
        required: u64,
        limit: u64,
    },
    AtlasSizeOverflow,
    AtlasAllocationFailed {
        gaussian_count: u32,
        bytes: u64,
    },
    ConflictingAtlasSlot {
        index: u32,
        first: AtlasSlot,
        second: AtlasSlot,
    },
    ConflictingAtlasPage {
        index: u32,
        first: LodPageId,
        second: LodPageId,
    },
    NativeTransport(String),
    HttpTransport(String),
    PersistentCache(String),
    PersistentCacheConfigConflict {
        key: String,
    },
    Runtime(LodRuntimeError),
    RenderCommit(LodRenderCommitError),
    DebugAnnotations(String),
    AtlasUpload(String),
    GpuUploadCommitTooLarge {
        dirty_slots: u64,
        bytes: u64,
        limit: u64,
    },
    RuntimePoisoned,
    MissingAtlasAsset,
    CompletedPageNotResident(LodPageId),
    ResidentPageNotDecoded(LodPageId),
    RenderCommitFailed,
    ViewLimitExceeded {
        actual: u64,
        limit: u32,
    },
    UnsupportedCamera(Entity),
}

impl fmt::Display for GaussianLodPackageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for GaussianLodPackageError {}

impl From<&GaussianLodPackageError> for LodOrchestrationFailure {
    fn from(error: &GaussianLodPackageError) -> Self {
        let code = match error {
            GaussianLodPackageError::ZeroLimit(_)
            | GaussianLodPackageError::InvalidManifest(_)
            | GaussianLodPackageError::InvalidLodSettings(_)
            | GaussianLodPackageError::InvalidStreaming(_)
            | GaussianLodPackageError::EmptyNativeRoot
            | GaussianLodPackageError::UnsupportedUrlScheme(_)
            | GaussianLodPackageError::MissingPersistentCacheRoot
            | GaussianLodPackageError::InvalidPersistentCacheRoot(_)
            | GaussianLodPackageError::MissingPersistentCacheNamespace
            | GaussianLodPackageError::InvalidPersistentCacheNamespace(_) => {
                LodOrchestrationFailureCode::InvalidConfiguration
            }
            GaussianLodPackageError::UnsupportedSortMode(_)
            | GaussianLodPackageError::UnsupportedRenderPath(_)
            | GaussianLodPackageError::NativeSourceUnsupportedInBrowser
            | GaussianLodPackageError::UnsupportedCamera(_) => {
                LodOrchestrationFailureCode::UnsupportedConfiguration
            }
            GaussianLodPackageError::ManifestHasNoPages => {
                LodOrchestrationFailureCode::SourceUnavailable
            }
            GaussianLodPackageError::NativeTransport(_)
            | GaussianLodPackageError::HttpTransport(_) => {
                LodOrchestrationFailureCode::TransportRequestFailed
            }
            GaussianLodPackageError::PersistentCache(_)
            | GaussianLodPackageError::PersistentCacheConfigConflict { .. } => {
                LodOrchestrationFailureCode::CacheFailed
            }
            GaussianLodPackageError::Runtime(_) => LodOrchestrationFailureCode::RuntimeFailed,
            GaussianLodPackageError::RenderCommit(_)
            | GaussianLodPackageError::AtlasUpload(_)
            | GaussianLodPackageError::MissingAtlasAsset => {
                LodOrchestrationFailureCode::AtlasCommitFailed
            }
            GaussianLodPackageError::RenderCommitFailed => {
                LodOrchestrationFailureCode::RenderCommitFailed
            }
            GaussianLodPackageError::AtlasCannotFitPage { .. }
            | GaussianLodPackageError::RootFallbackExceedsAtlas { .. }
            | GaussianLodPackageError::RootFallbackExceedsGpuUploadBudget { .. }
            | GaussianLodPackageError::AtlasSizeOverflow
            | GaussianLodPackageError::AtlasAllocationFailed { .. }
            | GaussianLodPackageError::GpuUploadCommitTooLarge { .. }
            | GaussianLodPackageError::ViewLimitExceeded { .. } => {
                LodOrchestrationFailureCode::CapacityExceeded
            }
            GaussianLodPackageError::ConflictingAtlasSlot { .. }
            | GaussianLodPackageError::ConflictingAtlasPage { .. }
            | GaussianLodPackageError::DebugAnnotations(_)
            | GaussianLodPackageError::RuntimePoisoned
            | GaussianLodPackageError::CompletedPageNotResident(_)
            | GaussianLodPackageError::ResidentPageNotDecoded(_) => {
                LodOrchestrationFailureCode::InternalInvariant
            }
        };
        Self::with_detail(code, error.to_string())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
