//! Automatic Bevy wiring for prebuilt `.gsplatlod` packages.
//!
//! Package manifests remain Bevy assets while page payloads are fetched by the
//! bounded streaming runtime. Native files and immutable HTTP(S) objects share
//! the same two-phase atlas/render commit path; optionally, encoded pages pass
//! through a content-addressed persistent cache before decoding.

use std::{
    any::TypeId,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    fmt,
    mem::size_of,
    sync::{Arc, Mutex, atomic::Ordering},
    time::Duration,
};

use bevy::prelude::*;
use bevy::{
    camera::{
        CameraUpdateSystems, Projection,
        primitives::Aabb,
        visibility::{VisibilitySystems, VisibleEntities},
    },
    transform::TransformSystems,
};
#[cfg(all(test, not(target_arch = "wasm32")))]
use bevy_interleave::prelude::Planar;
use bevy_interleave::prelude::PlanarHandle;

#[cfg_attr(target_arch = "wasm32", path = "package/browser.rs")]
#[cfg_attr(not(target_arch = "wasm32"), path = "package/native.rs")]
mod platform;

use crate::{
    CloudSettings, GaussianMode,
    gaussian::{
        cloud::CloudVisibilityClass,
        formats::{
            planar_3d::{Gaussian3d, gaussian_3d_gpu_bytes_per_record},
            planar_3d_chunked::{LodNodeId, LodPageId},
        },
        lod_settings::{
            GaussianLodSettings, GaussianStreamingSettings, LodQualityTarget, LodSelectionMode,
        },
    },
    io::{lod::GaussianLodHandle, lodge::GaussianLodgeHandle},
};
use platform::{
    PackageCacheRegistry, PackageManagerParam, PackagePageTransport, init_package_manager,
    package_page_transport, validate_cache_config,
};

use crate::{
    gaussian::{
        formats::planar_3d::{PlanarGaussian3d, PlanarGaussian3dHandle},
        lod_debug::{
            LodDebugAnnotationAtlas, LodDebugManifestIndex, LodDebugMetadata, LodDebugRecord,
            LodDebugResidency,
        },
    },
    io::lod::GaussianLodAsset,
    stream::{
        LodRenderPathSupportError,
        atlas_upload::{
            GaussianLodAtlasUploadPlugin, LodAtlasUploadBudget, LodAtlasUploadBudgetError,
            LodAtlasUploadQueue, LodTransientAtlas, LodTransientAtlasRegistry,
        },
        cache::AtlasSlot,
        hierarchy::{LodHierarchy, LodView},
        http::HttpRangeTransportConfig,
        persistent_cache::PersistentCachePackageIdentity,
        render_commit::{
            GaussianLodRenderCommitPlugin, LOD_RENDER_ACTIVE, LOD_RENDER_WAITING,
            LodOrchestrationFailure, LodOrchestrationFailureCode, LodOrchestrationSource,
            LodOrchestrationTransition, LodOrchestrationTransitionKind, LodPageAtlasMirror,
            LodRenderActivePresentation, LodRenderCandidate, LodRenderCandidates,
            LodRenderCommitError, LodViewBlendEndpoint, LodViewBlendPredecessorAttestation,
            LodViewBlendRetirementRequirement, LodViewBlendStatusSnapshot,
        },
        require_lod_render_path,
        runtime::{
            LodCandidateFrontier, LodPackageBootstrapBudget, LodPackageTargetPlan,
            LodPhysicalRange, LodRuntimeError, LodRuntimeViewId, LodSplitCohortCapacityStall,
            LodStreamFrame, LodStreamingRuntime, LodTemporalTransitionMode, LodViewBlendEdge,
            LodViewBlendMetric,
        },
        transport::{
            LodPageTransport, LodPageTransportFailure, LodPageTransportFailureKind, PagePoll,
        },
    },
};

const PACKAGE_ROOT_FALLBACK_VIEW: LodRuntimeViewId = LodRuntimeViewId(u64::MAX);
const PACKAGE_BOOTSTRAP_MAX_PAGES: u32 = 8;
const PACKAGE_BOOTSTRAP_MAX_ACTIVE_GAUSSIANS: u64 = 8_192;
const PACKAGE_BOOTSTRAP_MAX_ENCODED_BYTES: u64 = 2 * 1024 * 1024;
const PACKAGE_BOOTSTRAP_MAX_DECODED_BYTES: u64 = 2 * 1024 * 1024;
const PACKAGE_BOOTSTRAP_MAX_GPU_BYTES: u64 = 2 * 1024 * 1024;

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

/// Testing-only bounded-work snapshot for package liveness qualification.
#[cfg(feature = "testing")]
#[derive(Component, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GaussianLodPackageTestingSnapshot {
    pub current_cut_request_identity_present: bool,
    pub current_cut_matches_live_request: bool,
    pub pending_present: bool,
    pub pending_all_prepared_or_later: bool,
    pub pending_all_render_active: bool,
    pub pending_any_render_transitioning: bool,
    pub pending_any_view_blend_replan_requested: bool,
    /// False means the package runtime mutex was poisoned; the zero-valued
    /// gauges below must not then be interpreted as an idle runtime.
    pub runtime_work_available: bool,
    pub runtime_request_queue_len: u32,
    pub runtime_transport_in_flight_requests: u32,
    pub preprocess_waiting_jobs: u32,
    pub preprocess_backend_tracked_jobs: u32,
    pub preprocess_ready_pages: u32,
    pub runtime_capacity_blocked_requests: u32,
    pub runtime_max_last_observed_view_requested_pages: u32,
    pub runtime_split_cohort_admitted: bool,
    pub view_blend_retirement_attestation_retry_count: u64,
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
        Self::from_validated_manifest(manifest, settings, config)
    }

    fn from_validated_manifest(
        manifest: &crate::GaussianLodManifest,
        settings: &GaussianLodSettings,
        config: &GaussianLodPackageConfig,
    ) -> Result<Self, GaussianLodPackageError> {
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
        let root_ids = manifest.roots.iter().copied().collect::<BTreeSet<_>>();
        let root_pages = manifest
            .nodes
            .iter()
            .filter(|node| root_ids.contains(&node.id))
            .map(|node| node.representation.page)
            .collect::<BTreeSet<_>>();
        let bytes_per_slot = u64::from(plan.gaussians_per_slot)
            .checked_mul(gaussian_3d_gpu_bytes_per_record())
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        let staging_step_bytes = package_gpu_staging_step_byte_limit(settings);
        if bytes_per_slot > staging_step_bytes {
            return Err(GaussianLodPackageError::GpuUploadCommitTooLarge {
                dirty_slots: 1,
                bytes: bytes_per_slot,
                limit: staging_step_bytes,
            });
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
    /// Fixed-size GPU atlas with sparse, per-slot CPU staging payloads. Package
    /// startup never allocates or zeros `plan.physical_gaussians` on the CPU.
    transient_atlas: LodTransientAtlas,
    /// Last transient allocation generation whose materialized slots were
    /// queued. A recreated GPU atlas is empty and must replay those slots.
    transient_atlas_generation: u64,
    plan: GaussianLodPackageAtlasPlan,
    runtime: Mutex<LodStreamingRuntime<PackagePageTransport>>,
    mirror: LodPageAtlasMirror,
    /// Immutable annotation lookup compiled once at package instantiation.
    /// Debug Off drops only live sidecar payloads, never this manifest index.
    debug_index: Arc<LodDebugManifestIndex>,
    /// Immutable all-Resident page bases retained while debug presentation is
    /// Off. The live sidecar takes ownership while enabled; moving the bounded
    /// slot cache back here avoids regenerating support/color fields on a later
    /// Off-to-On toggle without retaining cut-dependent Residency state.
    retained_debug_page_bases: HashMap<u32, PackageDebugPageBasis>,
    /// Allocated only while cloud debug settings actually require metadata.
    debug: Option<PackageDebugAnnotations>,
    current: Option<LodRenderCandidates>,
    pending: Option<LodRenderCandidates>,
    /// Additively materialized replacement awaiting generation-current GPU
    /// pages and one complete compaction/radix publication.
    staged: Option<PackageStagedCut>,
    /// Exact selection request that produced `pending`. Candidate payloads do
    /// not retain the camera snapshot or every selector budget, so the
    /// handshake must carry that identity separately.
    pending_request: Option<PackageCutRequestSignature>,
    /// True only when the stream frames which produced `pending` were already
    /// stable and applied no bounded temporal substitution. A transition
    /// endpoint is drawable work toward the request, not ownership of its
    /// stationary fixed point.
    pending_request_fixed_point: bool,
    /// Whether this transaction used the ABI16 progressive fixed-point
    /// exception. A later render capability veto may never inherit that
    /// admission as a categorical cut.
    pending_progressive_view_blend: bool,
    /// Effective presentation mode finalized before the retirement and
    /// progressive-admission gates. ACTIVE publication must attest the same
    /// mode; RenderWorld cannot silently convert an admitted blend into hard
    /// output.
    pending_presentation_modes: BTreeMap<Entity, Option<LodTemporalTransitionMode>>,
    /// Device capability is stable for this package allocation. Once render
    /// rejects an authored view-blend table, subsequent candidates are planned
    /// categorically in the main world before ordinary publication gates.
    render_view_blend_unsupported: bool,
    /// A stale request which has already published morph entries must reach
    /// its exact ACTIVE endpoint before the package may release the staged
    /// parent/child union. The next live request supersedes it immediately
    /// after this one infallible logical commit.
    pending_transition_must_commit: bool,
    /// Exact request satisfied by the retained current cut. A direct package
    /// target deliberately bypasses transient navigation rungs; retaining its
    /// request identity prevents an unchanged stationary view from being
    /// overwritten by a residency-dependent ancestor wave on the next frame.
    /// Bootstrap fallbacks never populate this field because they are only a
    /// presentation-safe intermediate, not the requested target.
    current_request: Option<PackageCutRequestSignature>,
    /// Updated from the complete live request at the start of every visible
    /// orchestration turn, then corrected after same-turn activation. Render
    /// status uses this to distinguish a drawable retained cut from a cut that
    /// actually satisfies the current camera as well as its quality target.
    current_request_matches_live: bool,
    current_fallback_nodes: BTreeSet<LodNodeId>,
    pending_fallback_nodes: BTreeSet<LodNodeId>,
    /// Extra package-owned cache leases for every page referenced by `current`.
    /// Runtime view pins may move to a pending cut without making the published
    /// render output's physical slots reusable.
    current_page_leases: BTreeSet<LodPageId>,
    /// Independent leases for a complete replacement while its deterministic
    /// slot prefix is materialized across bounded frames.
    pending_page_leases: BTreeSet<LodPageId>,
    /// True after the retained current slots have been re-enqueued for one GPU
    /// generation-loss recovery. Cleared only after that same complete cut is
    /// ACTIVE again, so the bounded uploader can drain without the main world
    /// repeatedly invalidating and restarting its queued generations.
    current_recovery_queued: bool,
    /// The bounded bootstrap is eligible only before this package generation
    /// has published any per-camera cut. Render recovery retains/rebuilds the
    /// latest complete cut instead of replaying the cold-start sequence.
    has_published_cut: bool,
    /// Capacity admission for an atomic bootstrap-to-target handoff, keyed to
    /// the exact selection request. A rejected request remains quiescent until
    /// camera or policy identity changes.
    bootstrap_handoff: Option<PackageBootstrapHandoff>,
    /// Direct exact-leaf demand for legacy packages that have no admissible
    /// presentation bootstrap. The all-resident target is planned once per
    /// selection request and its completed pages are retained until the first
    /// exact transaction takes ownership.
    cold_direct_target: Option<PackageColdDirectTarget>,
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
    #[cfg(feature = "testing")]
    view_blend_retirement_attestation_retry_count: u64,
}

#[derive(Debug)]
struct PackageStagedCut {
    ranges: Vec<LodPhysicalRange>,
    slots: BTreeMap<u32, AtlasSlot>,
    materializations: Vec<(LodPageId, AtlasSlot)>,
    next_materialization: usize,
    complete: bool,
    fallback_nodes: BTreeSet<LodNodeId>,
    debug: PackageStagedDebugPreparation,
}

#[derive(Clone, Debug, Default)]
struct PackageStagedDebugPreparation {
    sidecar_identity: Option<u64>,
    /// All target page/generation pairs already represented by `pending` or
    /// `prepared`. This makes materialization admission logarithmic instead of
    /// repeatedly scanning a growing pair of queues.
    targets: BTreeSet<(LodPageId, AtlasSlot)>,
    /// Union of every slot referenced by the retained logical cut. These
    /// records stay byte-for-byte untouched until publication, regardless of
    /// camera frustum visibility.
    retained_current_targets: BTreeSet<(LodPageId, AtlasSlot)>,
    pending: VecDeque<(LodPageId, AtlasSlot)>,
    prepared: Vec<(LodPageId, AtlasSlot, Arc<[LodDebugRecord]>)>,
    /// Prepared targets installed early because the retained cut cannot read
    /// their physical slots. Candidate activation waits for their invariant
    /// revisions to reach the GPU.
    prepublished: BTreeSet<(LodPageId, AtlasSlot)>,
    complete: bool,
}

#[derive(Clone, Debug)]
enum PackageBootstrapHandoff {
    Admitted(PackageCutRequestSignature),
    CapacityExceeded {
        request: PackageCutRequestSignature,
        error: GaussianLodPackageError,
    },
}

impl PackageBootstrapHandoff {
    fn request(&self) -> &PackageCutRequestSignature {
        match self {
            Self::Admitted(request) | Self::CapacityExceeded { request, .. } => request,
        }
    }
}

#[derive(Clone, Debug)]
struct PackageColdDirectTarget {
    request: PackageCutRequestSignature,
    plan: LodPackageTargetPlan,
}

struct PackageDebugAnnotations {
    atlas: LodDebugAnnotationAtlas,
    /// Owned, validated once per package generation and reused per page.
    index: Arc<LodDebugManifestIndex>,
    /// Existing visible pages captured when annotations are enabled after the
    /// package is already active. They are populated in bounded waves without
    /// replacing the package runtime, atlas, or retained render cut.
    initialization: VecDeque<(LodPageId, AtlasSlot)>,
    /// Immutable all-Resident page-local records. Sparse atlas slots share
    /// these Arcs directly; cut-dependent fallback variants only copy and patch
    /// the low Residency bits.
    page_bases: HashMap<u32, PackageDebugPageBasis>,
}

struct PackageDebugPageBasis {
    page: LodPageId,
    records: Arc<[LodDebugRecord]>,
}

impl PackageDebugAnnotations {
    fn page_basis_is_current(&self, page: LodPageId, slot: AtlasSlot) -> bool {
        self.page_bases
            .get(&slot.index)
            .is_some_and(|basis| basis.page == page)
    }

    fn prepared_page_record_work(
        &self,
        page: LodPageId,
        slot: AtlasSlot,
        records_per_slot: u32,
        fallback_nodes: &BTreeSet<LodNodeId>,
    ) -> usize {
        let basis_is_current = self.page_basis_is_current(page, slot);
        let needs_residency_patch = !fallback_nodes.is_empty()
            && self
                .index
                .node_ids(page)
                .is_some_and(|mut nodes| nodes.any(|node| fallback_nodes.contains(&node)));
        if basis_is_current && !needs_residency_patch {
            0
        } else {
            records_per_slot as usize
        }
    }

    fn prepared_page_records(
        &mut self,
        page: &crate::PlanarGaussian3dPage,
        slot: AtlasSlot,
        records_per_slot: u32,
        fallback_nodes: &BTreeSet<LodNodeId>,
    ) -> Result<Arc<[LodDebugRecord]>, GaussianLodPackageError> {
        let basis_is_current = self
            .page_bases
            .get(&slot.index)
            .is_some_and(|basis| basis.page == page.id);
        if !basis_is_current {
            let mut records = self
                .index
                .records_for_page_with_node_residency_trusted_decoded(page, |_| {
                    LodDebugResidency::Resident
                })
                .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?;
            records.resize(records_per_slot as usize, LodDebugRecord::default());
            self.page_bases.insert(
                slot.index,
                PackageDebugPageBasis {
                    page: page.id,
                    records: records.into(),
                },
            );
        }
        let basis = Arc::clone(&self.page_bases[&slot.index].records);
        let has_fallback = !fallback_nodes.is_empty()
            && self
                .index
                .node_ids(page.id)
                .is_some_and(|mut nodes| nodes.any(|node| fallback_nodes.contains(&node)));
        if !has_fallback {
            return Ok(basis);
        }
        let mut records = basis.as_ref().to_vec();
        self.index
            .patch_page_record_residency(page.id, &mut records, |node| {
                if fallback_nodes.contains(&node) {
                    LodDebugResidency::AncestorFallback
                } else {
                    LodDebugResidency::Resident
                }
            })
            .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?;
        Ok(records.into())
    }
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

#[derive(Clone, Debug, PartialEq)]
struct PackageCutRequestSignature {
    target: LodQualityTarget,
    selection_mode: LodSelectionMode,
    hysteresis: f32,
    frustum_culling: bool,
    frustum_margin: f32,
    max_active_gaussians: u64,
    max_traversal_nodes_per_view: u32,
    structural: PackageStructuralSignature,
    cameras: Vec<PackageCutCameraSignature>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PackageCutCameraSignature {
    entity: Entity,
    /// Dynamic selection follows the live view exactly. Frozen selection owns
    /// its snapshot inside the runtime, so later live-camera motion is not a
    /// new selection request until the mode returns to Dynamic.
    view: Option<LodView>,
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

impl PackageCutRequestSignature {
    fn new(
        settings: &GaussianLodSettings,
        transform: &GlobalTransform,
        camera_views: &[PackageCameraView],
    ) -> Self {
        let dynamic = settings.selection_mode == LodSelectionMode::Dynamic;
        let world_from_local = transform.to_matrix();
        Self {
            target: settings.quality_target(),
            selection_mode: settings.selection_mode,
            hysteresis: settings.hysteresis,
            frustum_culling: settings.frustum_culling,
            frustum_margin: settings.frustum_margin,
            max_active_gaussians: settings.budgets.max_active_gaussians,
            max_traversal_nodes_per_view: settings.budgets.max_traversal_nodes_per_view,
            structural: PackageStructuralSignature::new(settings),
            cameras: camera_views
                .iter()
                .map(|camera| PackageCutCameraSignature {
                    entity: camera.entity,
                    view: dynamic.then_some(camera.view.with_world_from_local(world_from_local)),
                })
                .collect(),
        }
    }

    /// Identity that may never change beneath an extracted candidate. Dynamic
    /// camera samples are deliberately excluded: one render-claimed cut may
    /// activate under motion even before its private GPU state reaches
    /// PREPARED, so continuous camera updates cannot starve the handshake.
    fn same_critical_request(&self, other: &Self) -> bool {
        self.target == other.target
            && self.selection_mode == other.selection_mode
            && self.hysteresis == other.hysteresis
            && self.frustum_culling == other.frustum_culling
            && self.frustum_margin == other.frustum_margin
            && self.max_active_gaussians == other.max_active_gaussians
            && self.max_traversal_nodes_per_view == other.max_traversal_nodes_per_view
            && self.structural == other.structural
            && self.cameras.len() == other.cameras.len()
            && self
                .cameras
                .iter()
                .zip(&other.cameras)
                .all(|(left, right)| left.entity == right.entity)
    }
}

#[cfg_attr(not(target_arch = "wasm32"), derive(Resource))]
#[derive(Default)]
struct GaussianLodPackageManager {
    clouds: HashMap<Entity, PackageInstantiation>,
    caches: PackageCacheRegistry,
}

/// Persistent cursor for deterministic package-cloud staging admission.
///
/// The render-world uploader has its own fair atlas scheduler because it also
/// owns retries. This cursor bounds the earlier, main-world materialization
/// step so many package entities cannot create `N * per-cloud-limit` sparse
/// CPU payloads before extraction applies the global upload budget.
#[derive(Resource, Default)]
struct GaussianLodPackageStagingScheduler {
    next_owner: Option<u64>,
}

impl GaussianLodPackageStagingScheduler {
    fn rotate_to_next_owner<Package>(
        &mut self,
        packages: &mut [Package],
        entity: impl Fn(&Package) -> Entity,
    ) {
        if packages.is_empty() {
            return;
        }
        packages.sort_unstable_by_key(|package| entity(package).to_bits());
        let start = self.next_owner.map_or(0, |next| {
            let offset = packages.partition_point(|package| entity(package).to_bits() < next);
            if offset == packages.len() { 0 } else { offset }
        });
        packages.rotate_left(start);
        self.next_owner = Some(entity(&packages[1 % packages.len()]).to_bits());
    }
}

/// One package update's globally bounded staging tokens.
struct PackageStagingFrame {
    budget: LodAtlasUploadBudget,
    remaining_canonical_bytes: u64,
    remaining_slots: u32,
}

impl PackageStagingFrame {
    fn new(budget: LodAtlasUploadBudget) -> Self {
        Self {
            budget,
            remaining_canonical_bytes: budget.max_canonical_bytes_per_frame(),
            remaining_slots: budget.max_slots_per_frame(),
        }
    }

    fn begin_owner(&mut self) -> PackageStagingPermit<'_> {
        PackageStagingPermit {
            frame: self,
            used_canonical_bytes: 0,
            used_gpu_bytes: 0,
            used_slots: 0,
        }
    }
}

/// Work-conserving share for one package cloud in the scheduler's rotated
/// order. The owner may consume the live aggregate remainder; persistent
/// round-robin rotation changes priority next frame so this remains starvation
/// free without reserving tokens for hidden, stable, or invalid tail owners.
struct PackageStagingPermit<'a> {
    frame: &'a mut PackageStagingFrame,
    used_canonical_bytes: u64,
    used_gpu_bytes: u64,
    used_slots: u32,
}

impl PackageStagingPermit<'_> {
    fn try_consume_slot(
        &mut self,
        atlas: AssetId<PlanarGaussian3d>,
        slot_index: u32,
        gaussians_per_slot: u32,
        max_gpu_bytes: u64,
    ) -> Result<bool, GaussianLodPackageError> {
        let canonical_bytes = u64::from(gaussians_per_slot)
            .checked_mul(size_of::<Gaussian3d>() as u64)
            .ok_or(GaussianLodPackageError::MainWorldStagingBudget(
                LodAtlasUploadBudgetError::SlotCanonicalByteOverflow { atlas, slot_index },
            ))?;
        if canonical_bytes > self.frame.budget.max_canonical_bytes_per_frame() {
            return Err(GaussianLodPackageError::MainWorldStagingBudget(
                LodAtlasUploadBudgetError::SlotExceedsCanonicalByteLimit {
                    atlas,
                    slot_index,
                    required: canonical_bytes,
                    limit: self.frame.budget.max_canonical_bytes_per_frame(),
                },
            ));
        }
        let gpu_bytes = u64::from(gaussians_per_slot)
            .checked_mul(gaussian_3d_gpu_bytes_per_record())
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        if gpu_bytes > max_gpu_bytes {
            return Err(GaussianLodPackageError::GpuUploadCommitTooLarge {
                dirty_slots: 1,
                bytes: gpu_bytes,
                limit: max_gpu_bytes,
            });
        }

        let next_canonical_bytes = self
            .used_canonical_bytes
            .checked_add(canonical_bytes)
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        let next_gpu_bytes = self
            .used_gpu_bytes
            .checked_add(gpu_bytes)
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        let next_slots = self
            .used_slots
            .checked_add(1)
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        let fits_global_remainder = next_canonical_bytes <= self.frame.remaining_canonical_bytes
            && next_slots <= self.frame.remaining_slots;
        // A physical slot is atomic: proportional division must never reject a
        // whole slot that fits the live aggregate budget. The current owner may
        // continue into otherwise stranded idle-tail capacity; rotated priority
        // gives every queried owner the same opportunity on later frames.
        if !fits_global_remainder || next_gpu_bytes > max_gpu_bytes {
            return Ok(false);
        }
        self.used_canonical_bytes = next_canonical_bytes;
        self.used_gpu_bytes = next_gpu_bytes;
        self.used_slots = next_slots;
        Ok(true)
    }
}

impl Drop for PackageStagingPermit<'_> {
    fn drop(&mut self) {
        debug_assert!(self.used_canonical_bytes <= self.frame.remaining_canonical_bytes);
        debug_assert!(self.used_slots <= self.frame.remaining_slots);
        self.frame.remaining_canonical_bytes -= self.used_canonical_bytes;
        self.frame.remaining_slots -= self.used_slots;
    }
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

/// Main-world point after which a package's reserved render handle is stable
/// for extraction. Sort storage performs its final insertion/sizing pass after
/// this set because packages may create or replace sparse atlases in
/// `PostUpdate`.
#[derive(SystemSet, Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GaussianLodPackageUpdate;

impl Plugin for GaussianLodPackagePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GaussianLodPackageConfig>()
            .init_resource::<GaussianLodPackageStagingScheduler>()
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
                .in_set(GaussianLodPackageUpdate)
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
    upload_budget: Res<LodAtlasUploadBudget>,
    mut staging_scheduler: ResMut<GaussianLodPackageStagingScheduler>,
    mut manager: PackageManagerParam<'_>,
    manifests: Res<Assets<GaussianLodAsset>>,
    mut manifest_events: MessageReader<AssetEvent<GaussianLodAsset>>,
    mut clouds: ResMut<Assets<PlanarGaussian3d>>,
    mut transient_atlases: ResMut<LodTransientAtlasRegistry>,
    mut atlas_uploads: ResMut<LodAtlasUploadQueue>,
    cameras: Query<PackageCameraQueryItem, With<crate::GaussianCamera>>,
    cloud_handles: Query<&PlanarGaussian3dHandle>,
    packages: Query<
        (
            Entity,
            &GaussianLodHandle,
            &GaussianLodPackageSource,
            &GaussianLodSettings,
            &CloudSettings,
            Option<&GaussianStreamingSettings>,
            Option<&ViewVisibility>,
            &GlobalTransform,
        ),
        Without<GaussianLodgeHandle>,
    >,
) {
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
    let mut packages = packages.iter().collect::<Vec<_>>();
    staging_scheduler.rotate_to_next_owner(&mut packages, |package| package.0);
    let mut staging_frame = PackageStagingFrame::new(*upload_budget);
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
    ) in packages
    {
        let mut staging = staging_frame.begin_owner();
        seen.insert(entity);
        let effective_streaming =
            match package_streaming_settings(per_cloud_streaming.unwrap_or(&config.streaming)) {
                Ok(streaming) => streaming,
                Err(error) => {
                    if let Some(previous) = manager.clouds.remove(&entity) {
                        release_package_instantiation(
                            PackageReleaseTarget::replacement(entity),
                            previous,
                            &mut clouds,
                            &mut transient_atlases,
                            &mut atlas_uploads,
                            &mut commands,
                            &cloud_handles,
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
                    PackageReleaseTarget::replacement(entity),
                    previous,
                    &mut clouds,
                    &mut transient_atlases,
                    &mut atlas_uploads,
                    &mut commands,
                    &cloud_handles,
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
                } == PackageBuildSignature {
                    manifest: handle.0.id(),
                    source,
                    config: &config,
                    streaming: &effective_streaming,
                    structural,
                }
            });
        if !unchanged {
            if let Some(previous) = manager.clouds.remove(&entity) {
                release_package_instantiation(
                    PackageReleaseTarget::replacement(entity),
                    previous,
                    &mut clouds,
                    &mut transient_atlases,
                    &mut atlas_uploads,
                    &mut commands,
                    &cloud_handles,
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
                &mut transient_atlases,
                &mut atlas_uploads,
            );
            match result {
                Ok(state) => {
                    let bounds = asset.manifest().scene_bounds.map(|bounds| {
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
                    entity_commands.insert((
                        PlanarGaussian3dHandle(atlas),
                        LodRenderCandidates::package_required(),
                    ));
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
        let mut debug_work = PackageDebugPreparationWork::default();
        if let Err(error) = sync_package_debug_annotations(state, debug_metadata)
            .and_then(|()| advance_package_debug_initialization(state, &mut debug_work))
        {
            publish_package_failure(entity, state, error, &mut commands);
            continue;
        }
        if visibility.is_some_and(|visibility| !visibility.get()) {
            match suspend_package_state(state) {
                Ok(()) => publish_package_state(entity, state, &mut commands),
                Err(error) => publish_package_failure(entity, state, error, &mut commands),
            }
            continue;
        }
        let views =
            match package_camera_views_for_cloud(&cameras, entity, config.max_views_per_cloud) {
                Ok(views) => views,
                Err(error) => {
                    publish_package_failure(entity, state, error, &mut commands);
                    continue;
                }
            };
        match drive_package_state(
            state,
            settings,
            cloud_settings.gaussian_mode,
            transform,
            &views,
            &mut atlas_uploads,
            &mut staging,
            &mut debug_work,
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
            PackageReleaseTarget::removed(entity),
            state,
            &mut clouds,
            &mut transient_atlases,
            &mut atlas_uploads,
            &mut commands,
            &cloud_handles,
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
    let leased_pages = std::mem::take(&mut state.current_page_leases);
    let pending_pages = std::mem::take(&mut state.pending_page_leases);
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
    for page in leased_pages {
        runtime
            .release_resident_page(page)
            .map_err(GaussianLodPackageError::Runtime)?;
    }
    for page in pending_pages {
        runtime
            .release_resident_page(page)
            .map_err(GaussianLodPackageError::Runtime)?;
    }
    if let Some(pending) = state.pending.take() {
        for candidate in pending.by_camera.values() {
            candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
        }
    }
    state.pending_request = None;
    state.pending_request_fixed_point = false;
    state.pending_progressive_view_blend = false;
    state.pending_presentation_modes.clear();
    state.pending_transition_must_commit = false;
    state.pending_fallback_nodes.clear();
    let frame = runtime.begin_frame();
    runtime
        .finish_frame(frame)
        .map_err(GaussianLodPackageError::Runtime)?;
    state.resident_pages = runtime.cache().stats().resident_pages;
    discard_package_unpublished_staged_cut(state);
    Ok(())
}

#[derive(Clone, Copy)]
struct PackageReleaseTarget {
    entity: Entity,
    remove_status: bool,
}

impl PackageReleaseTarget {
    fn replacement(entity: Entity) -> Self {
        Self {
            entity,
            remove_status: false,
        }
    }

    fn removed(entity: Entity) -> Self {
        Self {
            entity,
            remove_status: true,
        }
    }
}

fn release_package_instantiation(
    target: PackageReleaseTarget,
    state: PackageInstantiation,
    clouds: &mut Assets<PlanarGaussian3d>,
    transient_atlases: &mut LodTransientAtlasRegistry,
    atlas_uploads: &mut LodAtlasUploadQueue,
    commands: &mut Commands,
    cloud_handles: &Query<&PlanarGaussian3dHandle>,
) {
    let atlas = state.atlas.id();
    atlas_uploads.remove_atlas(atlas);
    transient_atlases.unregister(atlas);
    // Dropping the sole strong transient owner cancels any already-extracted
    // generation before the reserved handle can be reused.
    drop(state);
    clouds.remove(atlas);
    let Ok(mut entity_commands) = commands.get_entity(target.entity) else {
        return;
    };
    entity_commands
        .remove::<LodRenderCandidates>()
        .remove::<LodDebugMetadata>();
    #[cfg(feature = "testing")]
    entity_commands.remove::<GaussianLodPackageTestingSnapshot>();
    if target.remove_status {
        entity_commands.remove::<GaussianLodPackageStatus>();
    }
    if cloud_handles
        .get(target.entity)
        .is_ok_and(|handle| handle.handle().id() == atlas)
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
    let identity = PersistentCachePackageIdentity::from_validated_manifest(manifest);
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
    transient_atlases: &mut LodTransientAtlasRegistry,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<PackageInstantiation, GaussianLodPackageError> {
    let manifest = asset.manifest();
    let plan = GaussianLodPackageAtlasPlan::from_validated_manifest(manifest, settings, config)?;
    let transport =
        package_page_transport(manifest, source, config, streaming, &mut manager.caches)?;
    let runtime_streaming = package_runtime_streaming_settings(source, streaming);
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
    let gpu_bytes_per_slot = u64::from(plan.gaussians_per_slot)
        .checked_mul(gaussian_3d_gpu_bytes_per_record())
        .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
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
        gpu_bytes_per_slot,
    };
    let runtime = LodStreamingRuntime::from_validated_shared_manifest_with_package_bootstrap(
        asset.shared_manifest(),
        transport,
        &effective,
        &runtime_streaming,
        bootstrap,
    )
    .map_err(GaussianLodPackageError::Runtime)?;
    let mirror = LodPageAtlasMirror::new(runtime.atlas_layout(), plan.slot_count)
        .map_err(GaussianLodPackageError::RenderCommit)?;
    let debug_index = Arc::new(
        LodDebugManifestIndex::from_validated_manifest(manifest)
            .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?,
    );
    let debug = if debug_metadata {
        let atlas =
            LodDebugAnnotationAtlas::new_sparse(plan.slot_count, plan.gaussians_per_slot)
                .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?;
        Some(PackageDebugAnnotations {
            atlas,
            index: Arc::clone(&debug_index),
            initialization: VecDeque::new(),
            page_bases: HashMap::new(),
        })
    } else {
        None
    };
    let atlas = clouds.reserve_handle();
    let transient_atlas = LodTransientAtlas::new_empty(plan.physical_gaussians)
        .map_err(|error| GaussianLodPackageError::AtlasUpload(error.to_string()))?;
    transient_atlases
        .register(
            atlas.id(),
            atlas.id(),
            plan.physical_gaussians,
            plan.gaussians_per_slot,
            &transient_atlas,
        )
        .and_then(|()| transient_atlases.queue_pending_initialization(atlas_uploads))
        .map_err(|error| GaussianLodPackageError::AtlasUpload(error.to_string()))?;
    let transient_atlas_generation = transient_atlas.ticket().generation();
    Ok(PackageInstantiation {
        manifest: AssetId::default(),
        source: source.clone(),
        config: config.clone(),
        atlas,
        transient_atlas,
        transient_atlas_generation,
        plan,
        runtime: Mutex::new(runtime),
        mirror,
        debug_index,
        retained_debug_page_bases: HashMap::new(),
        debug,
        current: None,
        pending: None,
        staged: None,
        pending_request: None,
        pending_request_fixed_point: false,
        pending_progressive_view_blend: false,
        pending_presentation_modes: BTreeMap::new(),
        render_view_blend_unsupported: false,
        pending_transition_must_commit: false,
        current_request: None,
        current_request_matches_live: false,
        current_fallback_nodes: BTreeSet::new(),
        pending_fallback_nodes: BTreeSet::new(),
        current_page_leases: BTreeSet::new(),
        pending_page_leases: BTreeSet::new(),
        current_recovery_queued: false,
        has_published_cut: false,
        bootstrap_handoff: None,
        cold_direct_target: None,
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
        #[cfg(feature = "testing")]
        view_blend_retirement_attestation_retry_count: 0,
    })
}

const PACKAGE_DEBUG_MAX_RECORDS_PER_FRAME: usize = 32 * 1024;
const PACKAGE_DEBUG_MAX_SLOTS_PER_FRAME: usize = 256;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PackageDebugPreparationWork {
    slots: usize,
    records: usize,
    /// Record fields regenerated from decoded page data. Cut-dependent
    /// Residency-only copies remain part of `records` but not this counter.
    regenerated_records: usize,
}

impl PackageDebugPreparationWork {
    fn can_consume(self, records: usize) -> bool {
        self.slots < PACKAGE_DEBUG_MAX_SLOTS_PER_FRAME
            && (self.slots == 0
                || self.records.saturating_add(records) <= PACKAGE_DEBUG_MAX_RECORDS_PER_FRAME)
    }

    fn consume(&mut self, records: usize) {
        self.slots += 1;
        self.records = self.records.saturating_add(records);
    }

    fn consume_prepared_page(&mut self, records: usize, regenerated_records: usize) {
        self.consume(records);
        self.regenerated_records = self.regenerated_records.saturating_add(regenerated_records);
    }
}

fn sync_package_debug_annotations(
    state: &mut PackageInstantiation,
    required: bool,
) -> Result<(), GaussianLodPackageError> {
    if !required {
        if let Some(debug) = state.debug.take() {
            state.retained_debug_page_bases = debug.page_bases;
        }
        if let Some(staged) = state.staged.as_mut() {
            staged.debug = PackageStagedDebugPreparation {
                complete: true,
                ..default()
            };
        }
        return Ok(());
    }
    if state.debug.is_some() {
        return Ok(());
    }
    let mut atlas =
        LodDebugAnnotationAtlas::new_sparse(state.plan.slot_count, state.plan.gaussians_per_slot)
            .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?;
    let initialization = state
        .visible_ranges
        .iter()
        .map(|range| (range.page, range.slot))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<VecDeque<_>>();
    atlas.set_complete(initialization.is_empty());
    state.debug = Some(PackageDebugAnnotations {
        atlas,
        index: Arc::clone(&state.debug_index),
        initialization,
        page_bases: std::mem::take(&mut state.retained_debug_page_bases),
    });
    Ok(())
}

fn advance_package_debug_initialization(
    state: &mut PackageInstantiation,
    work: &mut PackageDebugPreparationWork,
) -> Result<(), GaussianLodPackageError> {
    let PackageInstantiation {
        debug: Some(debug),
        runtime,
        mirror,
        visible_fallback_nodes,
        ..
    } = state
    else {
        return Ok(());
    };
    if debug.initialization.is_empty() {
        debug.atlas.set_complete(true);
        return Ok(());
    }

    debug.atlas.set_complete(false);
    let runtime = runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    while work.slots < PACKAGE_DEBUG_MAX_SLOTS_PER_FRAME {
        let Some(&(page_id, slot)) = debug.initialization.front() else {
            break;
        };
        if !mirror.is_page_current(page_id, slot) {
            debug.initialization.pop_front();
            continue;
        }
        let Some(page) = runtime.decoded_page(page_id) else {
            // Retained package leases normally keep this payload decoded. If a
            // recovery boundary temporarily removes it, preserve the bounded
            // initialization cursor and retry after residency is restored.
            break;
        };
        let record_work = debug.prepared_page_record_work(
            page_id,
            slot,
            state.plan.gaussians_per_slot,
            visible_fallback_nodes,
        );
        let regenerated_records = if !debug.page_basis_is_current(page_id, slot) {
            state.plan.gaussians_per_slot as usize
        } else {
            0
        };
        if !work.can_consume(record_work) {
            break;
        }
        debug.initialization.pop_front();
        if debug
            .atlas
            .page_matches_indexed_node_residency(&debug.index, page_id, slot, |node| {
                if visible_fallback_nodes.contains(&node) {
                    LodDebugResidency::AncestorFallback
                } else {
                    LodDebugResidency::Resident
                }
            })
        {
            continue;
        }
        let records = debug.prepared_page_records(
            page,
            slot,
            state.plan.gaussians_per_slot,
            visible_fallback_nodes,
        )?;
        debug
            .atlas
            .write_prepared_sparse_page(page_id, slot, records)
            .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?;
        work.consume_prepared_page(record_work, regenerated_records);
    }
    debug.atlas.set_complete(debug.initialization.is_empty());
    Ok(())
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

type PackageCameraQueryItem = (
    Entity,
    &'static Camera,
    &'static Projection,
    &'static GlobalTransform,
    Option<&'static VisibleEntities>,
);

fn package_camera_views_for_cloud(
    cameras: &Query<PackageCameraQueryItem, With<crate::GaussianCamera>>,
    cloud: Entity,
    max_views: u32,
) -> Result<Vec<PackageCameraView>, GaussianLodPackageError> {
    // Cameras without `VisibleEntities` intentionally observe every package.
    // Otherwise both conversion and limits are scoped to the cloud's explicit
    // visibility membership so unrelated views cannot poison this package.
    // Store at most the configured per-cloud bound. In particular, do not
    // clone every camera's potentially large visibility set before filtering.
    let mut applicable = Vec::new();
    for observation @ (_, camera, _, _, visible_entities) in cameras.iter() {
        if !camera.is_active
            || visible_entities.is_some_and(|visible| {
                !visible
                    .iter(TypeId::of::<CloudVisibilityClass>())
                    .any(|visible_cloud| *visible_cloud == cloud)
            })
        {
            continue;
        }
        if applicable.len() == max_views as usize {
            return Err(GaussianLodPackageError::ViewLimitExceeded {
                actual: u64::from(max_views) + 1,
                limit: max_views,
            });
        }
        applicable.push(observation);
    }
    applicable.sort_by_key(|(entity, _, _, _, _)| *entity);
    let mut views = Vec::with_capacity(applicable.len());
    for (entity, camera, projection, transform, _) in applicable {
        let viewport_height = camera
            .physical_viewport_size()
            .map(|size| size.y as f32)
            .filter(|height| *height > 0.0)
            .ok_or(GaussianLodPackageError::UnsupportedCamera(entity))?;
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
        let clip_from_world = projection.get_clip_from_view() * transform.to_matrix().inverse();
        views.push(PackageCameraView {
            entity,
            view: view.with_clip_from_world(clip_from_world),
        });
    }
    Ok(views)
}

#[allow(clippy::too_many_arguments)]
fn drive_package_state(
    state: &mut PackageInstantiation,
    settings: &GaussianLodSettings,
    gaussian_mode: GaussianMode,
    transform: &GlobalTransform,
    camera_views: &[PackageCameraView],
    atlas_uploads: &mut LodAtlasUploadQueue,
    staging: &mut PackageStagingPermit<'_>,
    debug_work: &mut PackageDebugPreparationWork,
) -> Result<(), GaussianLodPackageError> {
    if state.transient_atlas.ticket().is_failed() {
        return Err(GaussianLodPackageError::AtlasUpload(
            "transient package atlas GPU initialization or slot snapshot failed".to_owned(),
        ));
    }
    let transient_generation = state.transient_atlas.ticket().generation();
    if transient_generation != state.transient_atlas_generation {
        // A recreated GPU allocation is empty. Replay every sparse CPU payload
        // already materialized for the current/root/pending transaction before
        // any candidate can regain ACTIVE for the new generation.
        enqueue_package_materialized_slots(state, atlas_uploads)?;
        for candidates in state.current.iter().chain(state.pending.iter()) {
            for candidate in candidates.by_camera.values() {
                if package_candidate_requires_atlas(candidate) {
                    candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
                } else if !candidate.failed() {
                    candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
                }
            }
        }
        state.transient_atlas_generation = transient_generation;
        state.current_recovery_queued = state.current.as_ref().is_some_and(|current| {
            current
                .by_camera
                .values()
                .any(package_candidate_requires_atlas)
        });
        if state.current_recovery_queued {
            state.last_failure = Some(LodOrchestrationFailure::with_detail(
                LodOrchestrationFailureCode::AtlasCommitFailed,
                "GPU atlas generation changed; replaying the retained complete package cut",
            ));
        }
        return Ok(());
    }
    // Current-cut leases transition atomically when a cut publishes. The only
    // ordinary path that empties them without retiring `current` is hidden-
    // cloud suspension, so resume reacquires once instead of rebuilding a
    // BTreeSet on every stable frame.
    if state.current_page_leases.is_empty() {
        let current_pages = state
            .current
            .as_ref()
            .map(package_candidate_pages)
            .unwrap_or_default();
        if !current_pages.is_empty() {
            replace_package_current_page_leases(state, &current_pages)?;
        }
    }
    let effective = state.structural.apply(settings);
    let request = PackageCutRequestSignature::new(&effective, transform, camera_views);
    state.current_request_matches_live =
        state.current.is_some() && state.current_request.as_ref() == Some(&request);
    let pending_requested_view_blend_replan = state.pending.as_ref().is_some_and(|pending| {
        pending
            .by_camera
            .values()
            .any(LodRenderCandidate::view_blend_replan_requested)
    });
    if pending_requested_view_blend_replan {
        // Render observed a newer predecessor endpoint/pressure state than the
        // proof used to author this pending replacement. No replacement bytes
        // were synchronized. Restore selector topology to the retained cut,
        // cancel the token, and reselect from a fresh camera snapshot without
        // treating an ordinary pipelined race as a hard capability failure.
        state.pending_transition_must_commit = false;
        clear_package_pending_transaction(state)?;
        state.current_request = None;
        state.current_request_matches_live = false;
        state.last_failure = None;
        return Ok(());
    }
    let pending_invalid_view_blend_pressure = state
        .pending
        .as_ref()
        .is_some_and(package_candidate_set_has_invalid_view_blend_pressure);
    if pending_invalid_view_blend_pressure {
        // Render preflight found a non-finite or threshold-contradictory
        // pressure pair before this token became drawable. Keep the retained
        // current output, cancel the unrendered candidate, and give the next
        // main-world turn a fresh camera snapshot for a categorical/recovered
        // plan.
        state.pending_transition_must_commit = false;
        clear_package_pending_transaction(state)?;
        state.current_request = None;
        state.current_request_matches_live = false;
        state.last_failure = Some(invalid_view_blend_pressure_failure());
        return Ok(());
    }
    let render_requested_hard_fallback = state.pending.as_ref().is_some_and(|pending| {
        pending
            .by_camera
            .values()
            .any(LodRenderCandidate::render_hard_fallback_requested)
    });
    let static_mode_requires_hard_replan = gaussian_mode != GaussianMode::Gaussian3d
        && state.pending.as_ref().is_some_and(|pending| {
            pending.by_camera.values().any(|candidate| {
                candidate.view_blend_mode() == Some(LodTemporalTransitionMode::Morphing)
            })
        });
    if render_requested_hard_fallback || static_mode_requires_hard_replan {
        // A capability downgrade is a request for a new package-authored hard
        // transaction, never permission for RenderWorld to expose this token's
        // target. Cancel while the retained current output/table is untouched,
        // then let the ordinary fixed-point/capacity gates judge the replan.
        state.render_view_blend_unsupported |= render_requested_hard_fallback;
        state.pending_transition_must_commit = false;
        clear_package_pending_transaction(state)?;
    }
    let next_views = camera_views
        .iter()
        .map(|view| view.entity)
        .collect::<BTreeSet<_>>();
    {
        let runtime = state
            .runtime
            .get_mut()
            .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
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
    }
    state.views = next_views;
    let pending_request_is_stale = state.pending.as_ref().is_some_and(|pending| {
        let prepared = pending
            .by_camera
            .values()
            .all(LodRenderCandidate::render_is_prepared);
        let render_claimed = pending
            .by_camera
            .values()
            .all(LodRenderCandidate::render_is_claimed);
        let render_owned_motion_commit =
            (prepared || render_claimed) && !package_candidate_set_is_complete_empty(pending);
        state.pending_request.as_ref().is_none_or(|staged| {
            !staged.same_critical_request(&request)
                || !package_candidate_set_matches_request(pending, &request)
                || (staged != &request && !render_owned_motion_commit)
        })
    });
    if pending_request_is_stale || state.pending_transition_must_commit {
        let disposition = state.pending.as_mut().map(|pending| {
            reconcile_stale_package_pending_transition(
                pending,
                &state.views,
                state.pending_transition_must_commit,
            )
        });
        match disposition {
            Some(PackagePendingStaleDisposition::FinishAndCommit) => {
                // Once a transition has published destination-cardinality
                // entries, its parent/child union is GPU ownership, not
                // speculative demand. Finish and logically commit that exact
                // endpoint before selecting the superseding request.
                state.pending_transition_must_commit = true;
            }
            Some(PackagePendingStaleDisposition::CancelSafe) => {
                state.pending_transition_must_commit = false;
                clear_package_pending_transaction(state)?;
            }
            None => {}
        }
    }
    if state
        .pending
        .as_ref()
        .is_some_and(package_candidate_set_is_active)
        && !package_pending_active_presentation_is_safe(state)?
    {
        let invalid_pressure = state
            .pending
            .as_ref()
            .is_some_and(package_candidate_set_has_invalid_view_blend_pressure);
        let surprise_hard_fallback = state.pending.as_ref().is_some_and(|pending| {
            pending.by_camera.iter().any(|(camera, candidate)| {
                state.pending_presentation_modes.get(camera)
                    == Some(&Some(LodTemporalTransitionMode::Morphing))
                    && candidate.active_presentation()
                        == Some(LodRenderActivePresentation::HardTarget)
            })
        });
        state.render_view_blend_unsupported |= surprise_hard_fallback;
        state.pending_transition_must_commit = false;
        clear_package_pending_transaction(state)?;
        if invalid_pressure {
            state.last_failure = Some(invalid_view_blend_pressure_failure());
        }
        return Ok(());
    }
    if state
        .pending
        .as_ref()
        .is_some_and(package_candidate_set_is_active)
    {
        let mut staged =
            state
                .staged
                .take()
                .ok_or_else(|| GaussianLodPackageError::RenderCommitFailed {
                    detail: "render activated a pending cut without staged transaction ownership"
                        .to_owned(),
                })?;
        let preparation = advance_package_staged_debug_preparation(state, &mut staged, debug_work);
        let debug_complete = staged.debug.complete;
        state.staged = Some(staged);
        preparation?;
        if !debug_complete {
            // The GPU may already have a drawable pending compaction output,
            // but debug Residency variants are still being built under the CPU
            // record budget. Explicitly gate the live sidecar so the activated
            // replacement can only use authored colors until the pointer-only
            // annotation commit is ready. Pending-only preparation before
            // activation still leaves the retained current epoch ready.
            if let Some(debug) = state.debug.as_mut() {
                debug.atlas.set_complete(false);
            }
            return Ok(());
        }
    }

    if let Some(pending) = state.pending.as_ref() {
        if pending
            .by_camera
            .values()
            .any(|candidate| candidate.failed())
        {
            if state.pending_transition_must_commit {
                // A sibling view may still have drawable morph entries. Keep
                // every union lease pinned instead of turning a render failure
                // into stale physical reads. Recovery can replace this cloud
                // only after the render capability itself is retired.
                state.last_failure = Some(LodOrchestrationFailure::with_detail(
                    LodOrchestrationFailureCode::RenderCommitFailed,
                    "an in-flight multi-view transition failed; retaining its atlas union until render retirement",
                ));
                return Ok(());
            }
            clear_package_pending_transaction(state)?;
            if state.current.is_some() {
                state.last_failure = Some(LodOrchestrationFailure::with_detail(
                    LodOrchestrationFailureCode::RenderCommitFailed,
                    "replacement render commit failed; retained previous complete cut",
                ));
            } else {
                state.last_failure = Some(LodOrchestrationFailure::with_detail(
                    LodOrchestrationFailureCode::RenderCommitFailed,
                    "render candidate commit failed; retained root pages for a fresh complete candidate",
                ));
            }
            return Ok(());
        }
        if package_candidate_set_is_active(pending) {
            let (staged_required_ranges, staged_fallback_nodes, staged_debug, target_ranges) = {
                let staged = state.staged.as_ref().ok_or_else(|| {
                    GaussianLodPackageError::RenderCommitFailed {
                        detail:
                            "render activated a pending cut without staged transaction ownership"
                                .to_owned(),
                    }
                })?;
                validate_package_staged_cut(state, staged)?;
                let target_ranges = package_candidate_target_ranges(pending);
                state
                    .mirror
                    .validate_ranges(&target_ranges)
                    .map_err(GaussianLodPackageError::RenderCommit)?;
                let target_selection = plan_package_atlas_selection(state.plan, &target_ranges)?;
                if target_selection
                    .selected_slots
                    .iter()
                    .any(|(index, slot)| staged.slots.get(index) != Some(slot))
                {
                    return Err(GaussianLodPackageError::RenderCommitFailed {
                        detail: "active morph target is not a subset of its staged atlas union"
                            .to_owned(),
                    });
                }
                (
                    staged.ranges.clone(),
                    staged.fallback_nodes.clone(),
                    staged.debug.clone(),
                    target_ranges,
                )
            };
            let staged_required_pages = staged_required_ranges
                .iter()
                .map(|range| range.page)
                .collect::<BTreeSet<_>>();
            validate_package_pending_page_leases(state, &staged_required_pages)?;
            let target_pages = target_ranges
                .iter()
                .map(|range| range.page)
                .collect::<BTreeSet<_>>();
            if !target_pages.is_subset(&staged_required_pages) {
                return Err(GaussianLodPackageError::RenderCommitFailed {
                    detail: "active morph target pages are not a subset of staged union leases"
                        .to_owned(),
                });
            }
            let first_published_cut = !state.has_published_cut;
            let committed_bootstrap = pending
                .by_camera
                .values()
                .all(|candidate| candidate.frontier().is_coverage_guard());
            let published_bootstrap = first_published_cut && committed_bootstrap;
            if first_published_cut {
                // The pending transaction already leases every page needed by
                // the first visible cut. Transfer protection away from the
                // runtime's cold-start reserve before the infallible logical
                // commit so bounded out-of-core refinement can evict bootstrap
                // pages as later complete cuts replace them.
                state
                    .runtime
                    .get_mut()
                    .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?
                    .release_package_bootstrap_reserve()
                    .map_err(GaussianLodPackageError::Runtime)?;
            }

            // Snapshot only the immutable acknowledgement payload before any
            // mutable package commit work. Candidate state remains owned by
            // `state.pending` until the infallible take below.
            let rendered_frontiers = pending
                .by_camera
                .values()
                .map(|candidate| {
                    (
                        candidate.frontier().view(),
                        candidate
                            .target_render_ranges()
                            .iter()
                            .map(|range| range.node)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();

            // Shared physical pages remain drawable by `current` until this
            // exact transaction publishes. Commit their cut-dependent debug
            // provenance only after every other fallible validation step, so
            // the retained cut can never be paired with pending residency
            // labels. Render extraction gates the newly revised sparse slots
            // behind authored colors until every GPU subrange write drains.
            // ACTIVE means a complete drawable presentation and radix output
            // exist for the current per-edge weights. It does not mean a
            // camera-conditioned edge is categorical: stationary fractional
            // edges may remain ACTIVE indefinitely. Publish debug provenance
            // and ownership for the complete staged parent/children union.
            commit_package_staged_debug_annotations(
                state,
                &staged_required_ranges,
                &staged_fallback_nodes,
                &staged_debug,
            )?;
            let runtime = state
                .runtime
                .get_mut()
                .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
            for (view, nodes) in rendered_frontiers {
                runtime.acknowledge_rendered_frontier(view, &nodes);
            }

            // All fallible validation precedes transaction ownership changes.
            // From this point the candidate/current/lease/visible swap is one
            // infallible logical commit; obsolete cut-lease release follows.
            let pending = state.pending.take().expect("validated active pending cut");
            state.pending_transition_must_commit = false;
            state.pending_progressive_view_blend = false;
            state.pending_presentation_modes.clear();
            let staged = state.staged.take().expect("validated staged cut");
            let previous_page_leases = commit_package_current_page_leases(state);
            publish_package_staged_cut(state, staged);
            let committed_request = state.pending_request.take();
            let committed_request_fixed_point =
                std::mem::take(&mut state.pending_request_fixed_point)
                    && !pending
                        .by_camera
                        .values()
                        .any(LodRenderCandidate::view_blend_is_lagging);
            let fallback_nodes = std::mem::take(&mut state.pending_fallback_nodes);
            state.active_gaussians = pending
                .by_camera
                .values()
                .map(|candidate| u64::from(candidate.rendered_candidate_count()))
                .max()
                .unwrap_or(0);
            state.current = Some(pending);
            state.current_request = package_request_ownership_after_commit(
                committed_request,
                committed_bootstrap,
                committed_request_fixed_point,
            );
            state.current_request_matches_live = state.current_request.as_ref() == Some(&request);
            state.has_published_cut = true;
            state.cold_direct_target = None;
            if !published_bootstrap {
                state.bootstrap_handoff = None;
            }
            state.current_fallback_nodes = fallback_nodes;
            state.current_recovery_queued = false;
            state.root_fallback = false;
            state.last_failure = None;
            state.last_failure =
                retire_package_page_leases_best_effort(state, previous_page_leases);
            if published_bootstrap {
                // Capacity admission uses the now-ACTIVE bootstrap's exact
                // package leases on the next application frame. Do not start a
                // speculative target request between publication and that
                // all-resident footprint check.
                return Ok(());
            }
        } else if pending
            .by_camera
            .values()
            .all(|candidate| candidate.render_is_prepared())
        {
            let pending = pending.clone();
            let fallback_nodes = state.pending_fallback_nodes.clone();
            stage_package_cut_bounded(
                state,
                &pending,
                &fallback_nodes,
                atlas_uploads,
                staging,
                package_gpu_staging_step_byte_limit(&effective),
                debug_work,
            )?;
        }
        // Do not poll or evict while render-world staging is in progress. This
        // keeps the last complete cut's physical slots valid until activation.
        return Ok(());
    }

    if let Some(current) = state.current.as_ref() {
        for candidate in current.by_camera.values() {
            if !candidate.failed() && !package_candidate_requires_atlas(candidate) {
                candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
            }
        }
        if current.by_camera.values().any(LodRenderCandidate::failed) {
            state.current = None;
            state.current_request = None;
            state.current_request_matches_live = false;
            state.bootstrap_handoff = None;
            state.cold_direct_target = None;
            state.current_fallback_nodes.clear();
            replace_package_current_page_leases(state, &BTreeSet::new())?;
            state.current_recovery_queued = false;
            state.root_fallback = false;
            state.last_failure = Some(LodOrchestrationFailure::with_detail(
                LodOrchestrationFailureCode::RenderCommitFailed,
                "retained render state failed; staging a fresh cut from root fallback",
            ));
        } else if package_candidate_set_is_active(current) {
            state.current_recovery_queued = false;
            if package_candidate_set_has_invalid_view_blend_pressure(current) {
                // The render world keeps the last drawable suffix/table
                // bit-exact. This is not request ownership or successful
                // convergence. Hold this exact ACTIVE transaction rather than
                // selecting a categorical replacement: a fractional edge has
                // no endpoint evidence with which such a replacement could be
                // retired safely. A later valid render evaluation republishes
                // this same candidate, after which ordinary orchestration may
                // resume and reclaim request ownership.
                state.current_request = None;
                state.current_request_matches_live = false;
                state.last_failure = Some(invalid_view_blend_pressure_failure());
                return Ok(());
            } else if package_candidate_set_has_missing_view_blend_consumers(current) {
                // ACTIVE is not fixed-point ownership until every expected
                // private view has published one coherent radix-proven
                // snapshot. Retain the unanimous Fractional hold and allow
                // Render Cleanup to complete the same token.
                state.current_request = None;
                state.current_request_matches_live = false;
                state.last_failure = None;
                return Ok(());
            } else {
                state.last_failure = None;
            }
        } else {
            // GPU recovery reuses the same staged transaction: keep every
            // current page leased, re-enqueue its immutable CPU slots exactly
            // once, and do not select another cut until the bounded uploader
            // restores every generation and all per-view outputs republish.
            if state.last_failure.is_none() {
                state.last_failure = Some(LodOrchestrationFailure::with_detail(
                    LodOrchestrationFailureCode::AtlasCommitFailed,
                    "retained package cut is waiting for GPU atlas and compaction recovery",
                ));
            }
            if !state.current_recovery_queued {
                enqueue_package_current_recovery(state, atlas_uploads)?;
                state.current_recovery_queued = true;
            }
            return Ok(());
        }
    }

    let predictive_maintenance_required = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?
        .has_predictive_view_blend_work();
    if package_current_request_can_short_circuit(
        state.current.is_some(),
        state.current_request.as_ref() == Some(&request),
        predictive_maintenance_required,
    ) {
        // The retained cut already satisfies this exact camera/policy request.
        // In particular, do not restart hierarchy navigation after a direct
        // target publication: its final pages are resident, but intermediate
        // ancestor pages intentionally were not loaded and must not generate a
        // second, regressive presentation wave for a stationary view.
        // Optional predictive cohorts are the sole maintenance exception: they
        // remain outside request ownership/degradation, yet their bounded I/O
        // must keep advancing at a stationary pose until ready or isolated as
        // speculative terminal work.
        return Ok(());
    }

    if rebind_retained_package_bootstrap(state, &request, camera_views, &effective, transform)? {
        return Ok(());
    }
    if preflight_package_bootstrap_handoff(
        state,
        &request,
        camera_views,
        &effective,
        transform.to_matrix(),
    )? {
        return Ok(());
    }
    prepare_package_cold_direct_target(
        state,
        &request,
        camera_views,
        &effective,
        transform.to_matrix(),
    )?;
    let force_hard_view_blend =
        gaussian_mode != GaussianMode::Gaussian3d || state.render_view_blend_unsupported;

    // A package bridge never presents a partial forest as a complete scene.
    let runtime = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    let _ = runtime.transport_mut().maintain_cache()?;
    let frame = runtime.begin_frame();
    let world_from_local = transform.to_matrix();
    let direct_target_streaming = state.cold_direct_target.is_some() && !camera_views.is_empty();
    if let (Some(target), Some(owner)) = (&state.cold_direct_target, camera_views.first()) {
        // Mark the exact target demand before the one runtime update which
        // polls, publishes, and starts bounded page work. Legacy direct mode
        // deliberately omits ordinary camera navigation demand: the immutable
        // plan is the selection result and transient ancestors must not compete
        // with it for request or cache capacity.
        runtime
            .prime_package_pages_in_frame(
                frame,
                LodRuntimeViewId(owner.entity.to_bits()),
                target.plan.pages(),
            )
            .map_err(GaussianLodPackageError::Runtime)?;
    }
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
    state.resident_pages = root_frame.cache_stats().resident_pages;
    let mut stream_frames = Vec::with_capacity(camera_views.len());
    if !direct_target_streaming {
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
    }
    if let Some(target) = &state.cold_direct_target {
        let newly_resident = target
            .plan
            .pages()
            .difference(&state.pending_page_leases)
            .filter(|page| {
                runtime.cache().contains(**page) && runtime.decoded_page(**page).is_some()
            })
            .copied()
            .collect::<Vec<_>>();
        for page in newly_resident {
            runtime
                .retain_resident_page(page)
                .map_err(GaussianLodPackageError::Runtime)?;
            state.pending_page_leases.insert(page);
        }
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
    let mut fallback_nodes = stream_frames
        .iter()
        .flat_map(|(_, frame)| frame.ancestor_fallback_nodes())
        .collect::<BTreeSet<_>>();
    let mut candidates = LodRenderCandidates::default();
    let mut complete = true;
    for (camera, stream_frame) in &stream_frames {
        match stream_frame.candidate_frontier(effective.max_active_gaussians_u32()) {
            Ok(frontier) => {
                let mut candidate = LodRenderCandidate::new(frontier);
                if force_hard_view_blend {
                    package_author_hard_candidate_mode(&candidate);
                }
                if let Some(previous) = state
                    .current
                    .as_ref()
                    .and_then(|current| current.get(*camera))
                    .filter(|previous| {
                        previous.render_is_active()
                            && (!force_hard_view_blend
                                || previous.view_blend_mode()
                                    != Some(LodTemporalTransitionMode::Morphing))
                            && previous.same_payload(&candidate)
                    })
                {
                    candidate.inherit_active_payload_state(previous);
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
        if state.pending.is_none() && state.cold_direct_target.take().is_some() {
            for page in std::mem::take(&mut state.pending_page_leases) {
                runtime
                    .release_resident_page(page)
                    .map_err(GaussianLodPackageError::Runtime)?;
            }
            let mut view_ids = Vec::with_capacity(camera_views.len() + 1);
            view_ids.push(PACKAGE_ROOT_FALLBACK_VIEW);
            view_ids.extend(
                camera_views
                    .iter()
                    .map(|camera| LodRuntimeViewId(camera.entity.to_bits())),
            );
            runtime
                .cancel_package_view_work(&view_ids)
                .map_err(GaussianLodPackageError::Runtime)?;
        }
    } else if let Some(stall) = runtime.split_cohort_capacity_stall() {
        let error = split_cohort_capacity_error(stall);
        state.last_failure = Some(LodOrchestrationFailure::from(&error));
    } else if state
        .current
        .as_ref()
        .is_some_and(package_candidate_set_has_invalid_view_blend_pressure)
    {
        state.current_request = None;
        state.current_request_matches_live = false;
        state.last_failure = Some(invalid_view_blend_pressure_failure());
    } else {
        state.last_failure = None;
    }
    if camera_views.is_empty() {
        if let Some(root_frontier) = root_frontier
            && !state.root_fallback
        {
            let complete = materialize_package_frontiers_bounded(
                state,
                std::slice::from_ref(&root_frontier),
                &BTreeSet::new(),
                atlas_uploads,
                staging,
                package_gpu_staging_step_byte_limit(&effective),
                debug_work,
            )?;
            if !complete {
                return Ok(());
            }
            let (staged_ranges, staged_fallback_nodes, staged_debug) = {
                let staged = state
                    .staged
                    .as_ref()
                    .expect("complete root fallback retains its staged cut");
                validate_package_staged_cut(state, staged)?;
                (
                    staged.ranges.clone(),
                    staged.fallback_nodes.clone(),
                    staged.debug.clone(),
                )
            };
            replace_package_pending_page_leases(state, &BTreeSet::new())?;
            replace_package_current_page_leases(state, &BTreeSet::new())?;
            commit_package_staged_debug_annotations(
                state,
                &staged_ranges,
                &staged_fallback_nodes,
                &staged_debug,
            )?;
            let staged = state
                .staged
                .take()
                .expect("validated root fallback staged cut");
            publish_package_staged_cut(state, staged);
            state.root_fallback = true;
            state.current = None;
            state.current_request = None;
            state.current_request_matches_live = false;
            state.current_fallback_nodes.clear();
            state.current_recovery_queued = false;
            state.active_gaussians = u64::from(root_frontier.candidate_count());
        }
        return Ok(());
    }
    let mut direct_target_selected = false;
    if let Some(target) = state.cold_direct_target.as_ref()
        && let Some(frontiers) = runtime
            .package_target_candidates(&target.plan, effective.max_active_gaussians_u32())
            .map_err(GaussianLodPackageError::Runtime)?
    {
        let mut direct = LodRenderCandidates::default();
        for (view, frontier) in frontiers {
            let camera = camera_views
                .iter()
                .find(|camera| camera.entity.to_bits() == view.0)
                .ok_or(GaussianLodPackageError::RenderCommitFailed {
                    detail: "direct target plan references a stale package camera".to_owned(),
                })?;
            let mut candidate = LodRenderCandidate::new(frontier);
            if force_hard_view_blend {
                package_author_hard_candidate_mode(&candidate);
            }
            if let Some(previous) = state
                .current
                .as_ref()
                .and_then(|current| current.get(camera.entity))
                .filter(|previous| {
                    previous.render_is_active()
                        && (!force_hard_view_blend
                            || previous.view_blend_mode()
                                != Some(LodTemporalTransitionMode::Morphing))
                        && previous.same_payload(&candidate)
                })
            {
                candidate.inherit_active_payload_state(previous);
            }
            direct.by_camera.insert(camera.entity, candidate);
        }
        candidates = direct;
        fallback_nodes.clear();
        complete = true;
        direct_target_selected = true;
    }
    let publish_fixed_point =
        package_stream_frames_reached_publish_fixed_point(runtime, &stream_frames);
    let request_fixed_point = direct_target_selected
        || (publish_fixed_point
            && stream_frames.iter().all(|(_, frame)| {
                frame.selection_stable() && !frame.temporal_transition_applied()
            }));
    let mut package_bootstrap_selected = false;
    if !publish_fixed_point && state.current.is_none() && !state.has_published_cut {
        let mut bootstrap = LodRenderCandidates::default();
        for camera in camera_views {
            let Some(frontier) = runtime
                .package_bootstrap_candidate(
                    LodRuntimeViewId(camera.entity.to_bits()),
                    camera.view.with_world_from_local(world_from_local),
                    &effective,
                )
                .map_err(GaussianLodPackageError::Runtime)?
            else {
                bootstrap.by_camera.clear();
                break;
            };
            bootstrap
                .by_camera
                .insert(camera.entity, LodRenderCandidate::new(frontier));
        }
        if bootstrap.len() == camera_views.len() {
            fallback_nodes = package_candidate_fallback_nodes(&bootstrap);
            candidates = bootstrap;
            complete = true;
            package_bootstrap_selected = true;
        }
    }
    if !complete || candidates.len() != camera_views.len() {
        return Ok(());
    }
    if force_hard_view_blend {
        package_author_hard_candidate_modes(&candidates);
    }
    // Capacity is part of the package-authored presentation decision. Resolve
    // it before endpoint retirement or the progressive fixed-point exception,
    // so those gates reason about the exact mode RenderWorld is allowed to
    // activate rather than an optimistic authored morph.
    let pending_ranges = package_candidate_staging_ranges(state.plan, &candidates)?;
    let retirement_attestations = if let Some(current) = state.current.as_ref() {
        let Some(attestations) = package_candidate_set_view_blend_retirement_attestations(
            runtime.hierarchy(),
            current,
            &candidates,
        ) else {
            // The selector may optimistically reach the next adjacent cut
            // before the previously published edge is drawable at the matching
            // endpoint. Rebase only topology history to the retained target
            // cut; residency and disjoint demand remain live for the next
            // attempt.
            for candidate in current.by_camera.values() {
                let nodes = candidate
                    .target_render_ranges()
                    .iter()
                    .map(|range| range.node)
                    .collect::<Vec<_>>();
                runtime
                    .retry_from_rendered_frontier(candidate.frontier().view(), &nodes)
                    .map_err(GaussianLodPackageError::Runtime)?;
            }
            #[cfg(feature = "testing")]
            {
                state.view_blend_retirement_attestation_retry_count = state
                    .view_blend_retirement_attestation_retry_count
                    .saturating_add(1);
            }
            return Ok(());
        };
        attestations
    } else {
        BTreeMap::new()
    };
    for (camera, attestation) in retirement_attestations {
        let Some(candidate) = candidates.by_camera.get_mut(&camera) else {
            return Err(GaussianLodPackageError::RenderCommitFailed {
                detail: "retirement attestation references a missing replacement camera".to_owned(),
            });
        };
        candidate.set_predecessor_view_blend_attestation(attestation);
    }
    let capacity_relief = !publish_fixed_point
        && state.current.as_ref().is_some_and(|_| {
            package_candidate_releases_retained_capacity(
                runtime,
                &stream_frames,
                &state.current_page_leases,
                &candidates,
            )
        });
    let applied_view_blend_frame_count = stream_frames
        .iter()
        .filter(|(_, frame)| frame.temporal_transition_applied())
        .count();
    let applied_progressive_view_blend = state.current.is_some()
        && applied_view_blend_frame_count != 0
        && stream_frames
            .iter()
            .filter(|(_, frame)| frame.temporal_transition_applied())
            .all(|(camera, _)| {
                candidates.get(*camera).is_some_and(|candidate| {
                    candidate.view_blend_mode() == Some(LodTemporalTransitionMode::Morphing)
                        && candidate
                            .temporal_transition()
                            .and_then(|transition| transition.morph())
                            .is_some()
                })
            });
    let presentation_only_progressive_view_blend = applied_view_blend_frame_count == 0
        && state.current.as_ref().is_some_and(|current| {
            let identity = |candidate: &LodRenderCandidate| {
                (candidate.view_blend_mode() == Some(LodTemporalTransitionMode::Morphing))
                    .then(|| {
                        candidate
                            .temporal_transition()
                            .and_then(|transition| transition.morph())
                            .map(|morph| morph.identity())
                    })
                    .flatten()
            };
            let mut changed = false;
            let safe = candidates.by_camera.iter().all(|(camera, next)| {
                let Some(previous) = current.get(*camera) else {
                    return false;
                };
                let previous_identity = identity(previous);
                let next_identity = identity(next);
                if previous_identity == next_identity {
                    return true;
                }
                changed = true;
                // With no exact-cut substitution this is only a persistent
                // boundary-table addition/removal. Endpoint-safe removals were
                // already proved above; at least one side must remain authored
                // view-blend topology so categorical legacy churn cannot use
                // this global fixed-point exception.
                previous_identity.is_some() || next_identity.is_some()
            });
            safe && changed
        });
    let effective_view_blend_downgraded = candidates
        .by_camera
        .values()
        .any(package_candidate_has_downgraded_view_blend);
    let progressive_view_blend = package_progressive_view_blend_is_allowed(
        effective_view_blend_downgraded,
        applied_progressive_view_blend,
        presentation_only_progressive_view_blend,
    );
    if !publish_fixed_point
        && !capacity_relief
        && !progressive_view_blend
        && !package_bootstrap_selected
        && !direct_target_selected
    {
        // A complete resident ancestor frontier is a spatial safety proof, not
        // a presentation signal. Native and HTTP transports can complete many
        // page waves for one stationary view; publishing every wave repeatedly
        // swaps parent representatives for their children and visibly pops even
        // though each individual cut is atomic. Keep the exact retained cut
        // byte-for-byte while demand converges. A cold package deliberately
        // remains Loading instead of exposing a visually useless root wave.
        // Progressive v3 packages have one separate exception: the runtime may
        // admit a deterministic, payload-capped global bootstrap antichain.
        // That transaction is complete before extraction and remains unchanged
        // until this fixed-point gate admits the final target.
        //
        // ABI16 view-blend tables are a separate progressive exception: every
        // ready independent edge joins one persistent ACTIVE table while common
        // fractional edges inherit state. This removes the old global
        // fixed-point burst cadence without exposing categorical legacy waves.
        //
        // Explicit retained-cut capacity pressure is the other narrow exception: a
        // complete replacement which releases a page held only by the current
        // package lease must commit so blocked detail work can make progress.
        return Ok(());
    }
    let debug_fallback_nodes = fallback_nodes.clone();
    if let Some(current) = state.current.as_ref()
        && package_candidate_sets_equal(current, &candidates)
    {
        let selection_view_frozen_changed = current.by_camera.iter().any(|(camera, previous)| {
            candidates.get(*camera).is_none_or(|next| {
                previous.frontier().selection_view_frozen()
                    != next.frontier().selection_view_frozen()
            })
        });
        if state.current_fallback_nodes != fallback_nodes
            || state.visible_fallback_nodes != debug_fallback_nodes
        {
            let current = current.clone();
            let complete = materialize_package_cut(
                state,
                &current,
                &debug_fallback_nodes,
                atlas_uploads,
                staging,
                package_gpu_staging_step_byte_limit(&effective),
                debug_work,
            )?;
            if !complete {
                return Ok(());
            }
            state.current_fallback_nodes = fallback_nodes;
        }
        state.active_gaussians = candidates
            .by_camera
            .values()
            .map(|candidate| u64::from(candidate.rendered_candidate_count()))
            .max()
            .unwrap_or(0);
        let request_fixed_point = package_same_payload_request_fixed_point(
            request_fixed_point,
            selection_view_frozen_changed,
            candidates
                .by_camera
                .values()
                .any(LodRenderCandidate::view_blend_is_lagging),
        );
        state.current = Some(candidates);
        state.current_request = package_request_ownership_after_commit(
            Some(request.clone()),
            false,
            request_fixed_point,
        );
        state.current_request_matches_live = request_fixed_point;
        if direct_target_selected {
            state.cold_direct_target = None;
            state.bootstrap_handoff = None;
        }
        return Ok(());
    }
    // Every existing view keeps its own current candidate output while this
    // replacement stages. New views have no prior output and remain in the
    // package loading skip until their first complete candidate is published.
    let pending_pages = pending_ranges
        .iter()
        .map(|range| range.page)
        .collect::<BTreeSet<_>>();
    replace_package_pending_page_leases(state, &pending_pages)?;
    discard_package_unpublished_staged_cut(state);
    // Establish transaction ownership before extraction. A replacement whose
    // entire frontier is already current can pass PREPARED and reach ACTIVE in
    // one render frame, so the main world may never observe a staging turn.
    // Current slots are finalized below without an atlas write; dirty slots
    // remain in `materializations` until render has validated the candidate.
    let staged = prepare_package_staged_cut(state, &pending_ranges, &fallback_nodes)?;
    state.staged = Some(staged);
    state.pending = Some(candidates);
    state.pending_request = Some(request);
    state.pending_request_fixed_point = request_fixed_point;
    state.pending_progressive_view_blend = progressive_view_blend && !publish_fixed_point;
    state.pending_presentation_modes = state
        .pending
        .as_ref()
        .map(package_candidate_presentation_modes)
        .unwrap_or_default();
    state.pending_transition_must_commit = false;
    state.pending_fallback_nodes = fallback_nodes;
    Ok(())
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
fn drive_package_state_for_test(
    state: &mut PackageInstantiation,
    settings: &GaussianLodSettings,
    transform: &GlobalTransform,
    camera_views: &[PackageCameraView],
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), GaussianLodPackageError> {
    let budget = LodAtlasUploadBudget::try_new(u64::MAX, u32::MAX)
        .expect("unbounded test staging budget is non-zero");
    let mut frame = PackageStagingFrame::new(budget);
    let mut staging = frame.begin_owner();
    let mut debug_work = PackageDebugPreparationWork::default();
    drive_package_state(
        state,
        settings,
        GaussianMode::Gaussian3d,
        transform,
        camera_views,
        atlas_uploads,
        &mut staging,
        &mut debug_work,
    )
}

fn package_candidate_pages(candidates: &LodRenderCandidates) -> BTreeSet<LodPageId> {
    candidates
        .by_camera
        .values()
        .flat_map(LodRenderCandidate::required_atlas_ranges)
        .map(|range| range.page)
        .collect()
}

fn package_candidate_target_ranges(candidates: &LodRenderCandidates) -> Vec<LodPhysicalRange> {
    normalize_package_ranges(
        &candidates
            .by_camera
            .values()
            .flat_map(LodRenderCandidate::target_render_ranges)
            .copied()
            .collect::<Vec<_>>(),
    )
}

fn package_candidate_has_downgraded_view_blend(candidate: &LodRenderCandidate) -> bool {
    candidate
        .temporal_transition()
        .and_then(|transition| transition.morph())
        .is_some()
        && candidate.view_blend_mode() != Some(LodTemporalTransitionMode::Morphing)
}

const fn package_progressive_view_blend_is_allowed(
    effective_view_blend_downgraded: bool,
    applied: bool,
    presentation_only: bool,
) -> bool {
    !effective_view_blend_downgraded && (applied || presentation_only)
}

fn package_author_hard_candidate_modes(candidates: &LodRenderCandidates) {
    for candidate in candidates.by_camera.values() {
        package_author_hard_candidate_mode(candidate);
    }
}

fn package_author_hard_candidate_mode(candidate: &LodRenderCandidate) {
    if candidate.view_blend_mode() == Some(LodTemporalTransitionMode::Morphing) {
        candidate.publish_temporal_transition_mode(LodTemporalTransitionMode::BoundedHardCohort);
    }
}

fn package_candidate_presentation_modes(
    candidates: &LodRenderCandidates,
) -> BTreeMap<Entity, Option<LodTemporalTransitionMode>> {
    candidates
        .by_camera
        .iter()
        .map(|(camera, candidate)| (*camera, candidate.view_blend_mode()))
        .collect()
}

/// Resolves the bounded atlas payload before a candidate is extracted. A
/// morph union is optional presentation data; if that union cannot be planned
/// while the exact target remains valid, atomically downgrade the candidate to
/// its complete categorical endpoint instead of failing an otherwise valid
/// quality request with CapacityExceeded.
fn package_candidate_staging_ranges(
    plan: GaussianLodPackageAtlasPlan,
    candidates: &LodRenderCandidates,
) -> Result<Vec<LodPhysicalRange>, GaussianLodPackageError> {
    let required = normalize_package_ranges(
        &candidates
            .by_camera
            .values()
            .flat_map(LodRenderCandidate::required_atlas_ranges)
            .copied()
            .collect::<Vec<_>>(),
    );
    let target = package_candidate_target_ranges(candidates);
    let morphing = candidates.by_camera.values().any(|candidate| {
        candidate.temporal_transition_mode()
            == Some(crate::stream::runtime::LodTemporalTransitionMode::Morphing)
    });
    let (ranges, downgraded) = resolve_package_staging_ranges(plan, required, target, morphing)?;
    if downgraded {
        for candidate in candidates.by_camera.values() {
            if candidate.temporal_transition_mode()
                == Some(crate::stream::runtime::LodTemporalTransitionMode::Morphing)
            {
                candidate.publish_temporal_transition_mode(
                    crate::stream::runtime::LodTemporalTransitionMode::BoundedHardCohort,
                );
            }
        }
        debug_assert_eq!(
            ranges,
            normalize_package_ranges(
                &candidates
                    .by_camera
                    .values()
                    .flat_map(LodRenderCandidate::required_atlas_ranges)
                    .copied()
                    .collect::<Vec<_>>()
            )
        );
    }
    Ok(ranges)
}

fn resolve_package_staging_ranges(
    plan: GaussianLodPackageAtlasPlan,
    required: Vec<LodPhysicalRange>,
    target: Vec<LodPhysicalRange>,
    morphing: bool,
) -> Result<(Vec<LodPhysicalRange>, bool), GaussianLodPackageError> {
    match plan_package_atlas_selection(plan, &required) {
        Ok(_) => Ok((required, false)),
        Err(_) if morphing => {
            plan_package_atlas_selection(plan, &target)?;
            Ok((target, true))
        }
        Err(error) => Err(error),
    }
}

/// CPU debug metadata is a cloud-wide union, but each node enters it only when
/// at least one camera candidate classifies that exact rendered range as an
/// ancestor fallback. Merely belonging to a bootstrap cut is not sufficient:
/// an all-resident selector can retain that node because of active/traversal
/// budgets or frustum policy.
fn package_candidate_fallback_nodes(candidates: &LodRenderCandidates) -> BTreeSet<LodNodeId> {
    candidates
        .by_camera
        .values()
        .flat_map(|candidate| {
            candidate.render_ranges().iter().filter_map(|range| {
                candidate
                    .frontier()
                    .is_ancestor_fallback(range.node)
                    .then_some(range.node)
            })
        })
        .collect()
}

/// Replaces an already-published package bootstrap with the same resident
/// payload bound to the complete live camera set. This runs before capacity
/// memoization so a newly added camera never inherits a stall whose retained
/// candidate map cannot render it.
fn rebind_retained_package_bootstrap(
    state: &mut PackageInstantiation,
    request: &PackageCutRequestSignature,
    camera_views: &[PackageCameraView],
    settings: &GaussianLodSettings,
    transform: &GlobalTransform,
) -> Result<bool, GaussianLodPackageError> {
    let Some(current) = state.current.as_ref() else {
        return Ok(false);
    };
    if current.is_empty()
        || !current
            .by_camera
            .values()
            .all(|candidate| candidate.frontier().is_coverage_guard())
        || current.by_camera.keys().copied().collect::<BTreeSet<_>>() == state.views
    {
        return Ok(false);
    }
    if camera_views.is_empty() {
        return Ok(false);
    }

    let bootstrap_pages = package_candidate_pages(current);
    if !bootstrap_pages.is_subset(&state.current_page_leases) {
        return Err(GaussianLodPackageError::RenderCommitFailed {
            detail: "retained package bootstrap lost its current page leases".to_owned(),
        });
    }
    let current = current.clone();
    let world_from_local = transform.to_matrix();
    let mut candidates = LodRenderCandidates::default();
    {
        let runtime = state
            .runtime
            .get_mut()
            .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
        for camera in camera_views {
            let Some(frontier) = runtime
                .retained_package_bootstrap_candidate(
                    LodRuntimeViewId(camera.entity.to_bits()),
                    camera.view.with_world_from_local(world_from_local),
                    settings,
                )
                .map_err(GaussianLodPackageError::Runtime)?
            else {
                return Err(GaussianLodPackageError::RenderCommitFailed {
                    detail: "retained package bootstrap pages are no longer resident and decoded"
                        .to_owned(),
                });
            };
            let mut candidate = LodRenderCandidate::new(frontier);
            if let Some(previous) = current
                .get(camera.entity)
                .filter(|previous| previous.render_is_active() && previous.same_payload(&candidate))
            {
                candidate.inherit_active_payload_state(previous);
            }
            candidates.by_camera.insert(camera.entity, candidate);
        }
    }
    if package_candidate_pages(&candidates) != bootstrap_pages {
        return Err(GaussianLodPackageError::RenderCommitFailed {
            detail: "rebound package bootstrap changed its retained page payload".to_owned(),
        });
    }

    let fallback_nodes = package_candidate_fallback_nodes(&candidates);
    replace_package_pending_page_leases(state, &bootstrap_pages)?;
    discard_package_unpublished_staged_cut(state);
    let ranges = candidates
        .by_camera
        .values()
        .flat_map(LodRenderCandidate::required_atlas_ranges)
        .copied()
        .collect::<Vec<_>>();
    state.staged = Some(prepare_package_staged_cut(state, &ranges, &fallback_nodes)?);
    state.pending = Some(candidates);
    state.pending_request = Some(request.clone());
    state.pending_request_fixed_point = false;
    state.pending_progressive_view_blend = false;
    state.pending_presentation_modes = state
        .pending
        .as_ref()
        .map(package_candidate_presentation_modes)
        .unwrap_or_default();
    state.pending_transition_must_commit = false;
    state.pending_fallback_nodes = fallback_nodes;
    state.bootstrap_handoff = None;
    state.last_failure = None;
    Ok(true)
}

fn preflight_package_bootstrap_handoff(
    state: &mut PackageInstantiation,
    request: &PackageCutRequestSignature,
    camera_views: &[PackageCameraView],
    settings: &GaussianLodSettings,
    world_from_local: Mat4,
) -> Result<bool, GaussianLodPackageError> {
    let current_is_bootstrap = state.current.as_ref().is_some_and(|current| {
        !current.is_empty()
            && current
                .by_camera
                .values()
                .all(|candidate| candidate.frontier().is_coverage_guard())
    });
    if !current_is_bootstrap {
        state.bootstrap_handoff = None;
        return Ok(false);
    }

    // A reachable terminal page must be observed by the ordinary runtime
    // fixed-point path before capacity memoization can suppress updates. Its
    // complete resident ancestor is an honest degraded result, whereas an
    // all-resident footprint deliberately assumes every target page exists.
    if !state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?
        .terminal_failures()
        .is_empty()
    {
        state.bootstrap_handoff = None;
        return Ok(false);
    }

    if let Some(handoff) = &state.bootstrap_handoff
        && handoff.request() == request
    {
        return match handoff {
            PackageBootstrapHandoff::Admitted(_) => Ok(false),
            PackageBootstrapHandoff::CapacityExceeded { error, .. } => {
                state.last_failure = Some(LodOrchestrationFailure::from(error));
                Ok(true)
            }
        };
    }
    if state
        .cold_direct_target
        .as_ref()
        .is_some_and(|target| target.request != *request)
    {
        clear_package_direct_target(state)?;
    }
    state.bootstrap_handoff = None;

    let runtime = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    let views = camera_views
        .iter()
        .map(|camera| {
            (
                LodRuntimeViewId(camera.entity.to_bits()),
                camera.view.with_world_from_local(world_from_local),
            )
        })
        .collect::<Vec<_>>();
    let plan = runtime
        .package_all_resident_target_plan(&views, settings)
        .map_err(GaussianLodPackageError::Runtime)?;
    let mut pages = plan.pages().clone();
    pages.extend(state.current_page_leases.iter().copied());
    pages.extend(
        runtime
            .hierarchy()
            .roots()
            .iter()
            .filter_map(|root| runtime.hierarchy().page(*root)),
    );

    let mut required_decoded_bytes = 0_u64;
    let mut required_gaussians = 0_u64;
    for &page in &pages {
        let descriptor =
            runtime
                .hierarchy()
                .page_descriptor(page)
                .ok_or(GaussianLodPackageError::Runtime(
                    LodRuntimeError::MissingPageDescriptor(page),
                ))?;
        required_decoded_bytes = required_decoded_bytes
            .checked_add(descriptor.decoded_len)
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        required_gaussians = required_gaussians
            .checked_add(u64::from(descriptor.gaussian_count))
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
    }
    let required_pages = u64::try_from(pages.len()).unwrap_or(u64::MAX);
    let limits = runtime.cache().limits();
    let fits = required_pages <= u64::from(limits.max_pages)
        && required_decoded_bytes <= limits.max_bytes
        && required_gaussians <= limits.max_gaussians;
    if fits {
        state.cold_direct_target = Some(PackageColdDirectTarget {
            request: request.clone(),
            plan,
        });
        state.bootstrap_handoff = Some(PackageBootstrapHandoff::Admitted(request.clone()));
        state.last_failure = None;
        return Ok(false);
    }

    let error = GaussianLodPackageError::AtomicHandoffCapacityExceeded {
        required_pages,
        limit_pages: u64::from(limits.max_pages),
        required_decoded_bytes,
        limit_decoded_bytes: limits.max_bytes,
        required_gaussians,
        limit_gaussians: limits.max_gaussians,
    };
    // The current package leases now own every visible bootstrap page. Retire
    // the runtime's startup reserve and cancel all speculative root/camera
    // demand so an impossible atomic handoff is stable and request-free.
    runtime
        .release_package_bootstrap_reserve()
        .map_err(GaussianLodPackageError::Runtime)?;
    let mut view_ids = Vec::with_capacity(views.len() + 1);
    view_ids.push(PACKAGE_ROOT_FALLBACK_VIEW);
    view_ids.extend(views.iter().map(|(view_id, _)| *view_id));
    runtime
        .cancel_package_view_work(&view_ids)
        .map_err(GaussianLodPackageError::Runtime)?;
    state.resident_pages = runtime.cache().stats().resident_pages;
    state.last_failure = Some(LodOrchestrationFailure::from(&error));
    state.bootstrap_handoff = Some(PackageBootstrapHandoff::CapacityExceeded {
        request: request.clone(),
        error,
    });
    Ok(true)
}

fn clear_package_direct_target(
    state: &mut PackageInstantiation,
) -> Result<(), GaussianLodPackageError> {
    if state.cold_direct_target.take().is_none() {
        return Ok(());
    }
    replace_package_pending_page_leases(state, &BTreeSet::new())?;
    let mut view_ids = Vec::with_capacity(state.views.len() + 1);
    view_ids.push(PACKAGE_ROOT_FALLBACK_VIEW);
    view_ids.extend(
        state
            .views
            .iter()
            .map(|entity| LodRuntimeViewId(entity.to_bits())),
    );
    state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?
        .cancel_package_view_work(&view_ids)
        .map_err(GaussianLodPackageError::Runtime)?;
    Ok(())
}

fn prepare_package_cold_direct_target(
    state: &mut PackageInstantiation,
    request: &PackageCutRequestSignature,
    camera_views: &[PackageCameraView],
    settings: &GaussianLodSettings,
    world_from_local: Mat4,
) -> Result<(), GaussianLodPackageError> {
    if state.current.is_some()
        || state.pending.is_some()
        || state.has_published_cut
        || state
            .cold_direct_target
            .as_ref()
            .is_some_and(|target| target.request == *request)
    {
        return Ok(());
    }
    let replaced_direct_target = state.cold_direct_target.is_some();
    if replaced_direct_target {
        clear_package_direct_target(state)?;
    }

    let runtime = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    if !runtime.terminal_failures().is_empty()
        || runtime.has_active_package_bootstrap()
        || runtime
            .hierarchy()
            .manifest()
            .build
            .has_bounded_refinement_amplification()
        || settings.quality_endpoint()
            != crate::gaussian::lod_settings::LodQualityEndpoint::Original
    {
        return Ok(());
    }
    let views = camera_views
        .iter()
        .map(|camera| {
            (
                LodRuntimeViewId(camera.entity.to_bits()),
                camera.view.with_world_from_local(world_from_local),
            )
        })
        .collect::<Vec<_>>();
    let plan = runtime
        .package_all_resident_target_plan(&views, settings)
        .map_err(GaussianLodPackageError::Runtime)?;
    let mut footprint = plan.pages().clone();
    footprint.extend(
        runtime
            .hierarchy()
            .roots()
            .iter()
            .filter_map(|root| runtime.hierarchy().page(*root)),
    );
    footprint.extend(runtime.active_coverage_guard_pages().iter().copied());
    let limits = runtime.cache().limits();
    let mut decoded_bytes = 0_u64;
    let mut gaussians = 0_u64;
    for &page in &footprint {
        let descriptor =
            runtime
                .hierarchy()
                .page_descriptor(page)
                .ok_or(GaussianLodPackageError::Runtime(
                    LodRuntimeError::MissingPageDescriptor(page),
                ))?;
        decoded_bytes = decoded_bytes
            .checked_add(descriptor.decoded_len)
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
        gaussians = gaussians
            .checked_add(u64::from(descriptor.gaussian_count))
            .ok_or(GaussianLodPackageError::AtlasSizeOverflow)?;
    }
    if footprint.len() > limits.max_pages as usize
        || decoded_bytes > limits.max_bytes
        || gaussians > limits.max_gaussians
    {
        return Ok(());
    }
    state.cold_direct_target = Some(PackageColdDirectTarget {
        request: request.clone(),
        plan,
    });
    Ok(())
}

/// True once every camera has reached a target/terminal fixed point or emitted
/// one explicitly bounded temporal topology step. Temporal steps are admitted
/// only after all page work is quiescent, so this exception cannot expose an
/// ordinary partial residency wave. Terminally unavailable descendants no
/// longer own transport work, so their complete resident ancestor cut is the
/// final honest result.
fn package_stream_frames_reached_publish_fixed_point<T: LodPageTransport>(
    runtime: &LodStreamingRuntime<T>,
    frames: &[(Entity, LodStreamFrame)],
) -> bool {
    // Global queue/in-flight counts include proactive ABI16 child prefetch and
    // therefore are not target fixed-point evidence. The logical requested-node
    // set plus capacity/terminal state is the exact request-scoped predicate.
    !frames.is_empty()
        && frames.iter().all(|(_, frame)| {
            (frame.selection_stable() || frame.temporal_transition_applied())
                && frame.capacity_blocked_requests() == 0
                && frame.frontier().requested_nodes.iter().all(|node| {
                    runtime
                        .hierarchy()
                        .page(*node)
                        .is_none_or(|page| runtime.is_terminal_failure(page))
                })
        })
}

fn split_cohort_capacity_error(stall: LodSplitCohortCapacityStall) -> GaussianLodPackageError {
    GaussianLodPackageError::SplitCohortCapacityExceeded {
        view: stall.view,
        parent: stall.parent,
        required_pages: stall.required_pages,
        limit_pages: stall.limit_pages,
        required_decoded_bytes: stall.required_decoded_bytes,
        limit_decoded_bytes: stall.limit_decoded_bytes,
        required_gaussians: stall.required_gaussians,
        limit_gaussians: stall.limit_gaussians,
    }
}

/// Allows only a slot-releasing complete cut through explicit retained-capacity
/// pressure. A nearly full LRU may continuously evict not-yet-selected demanded
/// pages without producing an `InsufficientEvictableCapacity` result, so a
/// missing requested page that exceeds any remaining resident dimension also
/// qualifies. Ordinary page waves remain suppressed by the fixed-point gate.
fn package_candidate_releases_retained_capacity<T: LodPageTransport>(
    runtime: &LodStreamingRuntime<T>,
    frames: &[(Entity, LodStreamFrame)],
    current_page_leases: &BTreeSet<LodPageId>,
    candidate: &LodRenderCandidates,
) -> bool {
    let explicitly_blocked = frames
        .iter()
        .any(|(_, frame)| frame.capacity_blocked_requests() > 0)
        || runtime.split_cohort_capacity_stall().is_some();
    if current_page_leases.is_empty() || !explicitly_blocked {
        return false;
    }
    let next_pages = package_candidate_pages(candidate);
    current_page_leases.difference(&next_pages).any(|page| {
        runtime
            .cache()
            .get(*page)
            .is_some_and(|resident| resident.pin_count == 1)
    })
}

fn package_candidate_set_is_complete_empty(candidates: &LodRenderCandidates) -> bool {
    !candidates.is_empty()
        && candidates
            .by_camera
            .values()
            .all(|candidate| !package_candidate_requires_atlas(candidate))
}

fn package_candidate_requires_atlas(candidate: &LodRenderCandidate) -> bool {
    candidate.frontier().candidate_count() != 0 || !candidate.required_atlas_ranges().is_empty()
}

fn enqueue_package_current_recovery(
    state: &PackageInstantiation,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), GaussianLodPackageError> {
    debug_assert!(state.current.is_some());
    debug_assert_eq!(
        state.current_page_leases,
        package_candidate_pages(state.current.as_ref().expect("checked current cut"))
    );
    for &slot in state.visible_slots.values() {
        atlas_uploads
            .enqueue_slot(state.atlas.id(), slot, state.plan.gaussians_per_slot)
            .map_err(|error| GaussianLodPackageError::AtlasUpload(error.to_string()))?;
    }
    Ok(())
}

fn enqueue_package_materialized_slots(
    state: &PackageInstantiation,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), GaussianLodPackageError> {
    for slot in state.mirror.materialized_slots() {
        atlas_uploads
            .enqueue_slot(state.atlas.id(), slot, state.plan.gaussians_per_slot)
            .map_err(|error| GaussianLodPackageError::AtlasUpload(error.to_string()))?;
    }
    Ok(())
}

fn clear_package_pending_transaction(
    state: &mut PackageInstantiation,
) -> Result<(), GaussianLodPackageError> {
    restore_package_runtime_after_pending_cancellation(state)?;
    replace_package_pending_page_leases(state, &BTreeSet::new())?;
    if let Some(pending) = state.pending.take() {
        for candidate in pending.by_camera.values() {
            candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
        }
    }
    discard_package_unpublished_staged_cut(state);
    state.pending_request = None;
    state.pending_request_fixed_point = false;
    state.pending_progressive_view_blend = false;
    state.pending_presentation_modes.clear();
    state.pending_transition_must_commit = false;
    state.pending_fallback_nodes.clear();
    Ok(())
}

fn restore_package_runtime_after_pending_cancellation(
    state: &mut PackageInstantiation,
) -> Result<(), GaussianLodPackageError> {
    let Some(pending) = state.pending.as_ref() else {
        return Ok(());
    };
    let pending_views = pending
        .by_camera
        .values()
        .map(|candidate| candidate.frontier().view())
        .collect::<BTreeSet<_>>();
    let retained = state
        .current
        .as_ref()
        .map(|current| {
            current
                .by_camera
                .values()
                .map(|candidate| {
                    (
                        candidate.frontier().view(),
                        candidate
                            .target_render_ranges()
                            .iter()
                            .map(|range| range.node)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    let retained_views = retained.keys().copied().collect::<BTreeSet<_>>();
    let live_views = state
        .views
        .iter()
        .map(|entity| LodRuntimeViewId(entity.to_bits()))
        .collect::<BTreeSet<_>>();
    let known_views = pending_views
        .union(&retained_views)
        .copied()
        .collect::<BTreeSet<_>>();
    let runtime = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    for (&view, nodes) in &retained {
        if !live_views.contains(&view) {
            continue;
        }
        runtime
            .restore_rendered_frontier(view, nodes)
            .map_err(GaussianLodPackageError::Runtime)?;
    }
    for view in known_views {
        if !live_views.contains(&view) || !retained_views.contains(&view) {
            runtime
                .remove_view(view)
                .map_err(GaussianLodPackageError::Runtime)?;
        }
    }
    Ok(())
}

fn discard_package_unpublished_staged_cut(state: &mut PackageInstantiation) {
    // Staging owns only cut publication. Materialized slots belong to the
    // bounded runtime cache and remain reusable while that allocator
    // page/generation is resident, whether or not a cut currently references
    // them. `LodPageAtlasMirror::stage_page` revokes the proof when the runtime
    // replaces a slot, and package teardown drops the complete sparse owner.
    state.staged = None;
    if let Some(debug) = state.debug.as_mut() {
        // Staged debug payloads never enter the live atlas before logical
        // publication. Cancellation therefore leaves current-cut readiness
        // unchanged; only independent lazy-enable initialization can gate it.
        debug.atlas.set_complete(debug.initialization.is_empty());
    }
}

/// Transfers the package-owned residency lease atomically: every new page is
/// retained before any old-only page is released. A failed acquisition rolls
/// back the pages acquired by this call and leaves the published cut leased.
fn replace_package_current_page_leases(
    state: &mut PackageInstantiation,
    next: &BTreeSet<LodPageId>,
) -> Result<(), GaussianLodPackageError> {
    replace_package_page_leases(&mut state.runtime, &mut state.current_page_leases, next)
}

fn replace_package_pending_page_leases(
    state: &mut PackageInstantiation,
    next: &BTreeSet<LodPageId>,
) -> Result<(), GaussianLodPackageError> {
    replace_package_page_leases(&mut state.runtime, &mut state.pending_page_leases, next)
}

fn replace_package_page_leases(
    runtime: &mut Mutex<LodStreamingRuntime<PackagePageTransport>>,
    leases: &mut BTreeSet<LodPageId>,
    next: &BTreeSet<LodPageId>,
) -> Result<(), GaussianLodPackageError> {
    if *leases == *next {
        return Ok(());
    }
    let previous = leases.clone();
    let runtime = runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    let mut acquired = Vec::new();
    for &page in next.difference(&previous) {
        if let Err(error) = runtime.retain_resident_page(page) {
            for acquired_page in acquired.into_iter().rev() {
                let _ = runtime.release_resident_page(acquired_page);
            }
            return Err(GaussianLodPackageError::Runtime(error));
        }
        acquired.push(page);
    }
    let mut released = Vec::new();
    for &page in previous.difference(next) {
        if let Err(error) = runtime.release_resident_page(page) {
            for released_page in released {
                let _ = runtime.retain_resident_page(released_page);
            }
            for acquired_page in acquired.into_iter().rev() {
                let _ = runtime.release_resident_page(acquired_page);
            }
            return Err(GaussianLodPackageError::Runtime(error));
        }
        released.push(page);
    }
    leases.clone_from(next);
    Ok(())
}

fn validate_package_pending_page_leases(
    state: &PackageInstantiation,
    next: &BTreeSet<LodPageId>,
) -> Result<(), GaussianLodPackageError> {
    if state.pending_page_leases != *next {
        return Err(GaussianLodPackageError::RenderCommitFailed {
            detail: format!(
                "pending page leases do not match the active replacement: retained={}, required={}",
                state.pending_page_leases.len(),
                next.len()
            ),
        });
    }
    Ok(())
}

/// Moves a fully leased replacement into current ownership without releasing
/// any previous retain. This is the infallible lease portion of the logical
/// cut commit; old retains are cleanup work after publication.
fn commit_package_current_page_leases(state: &mut PackageInstantiation) -> BTreeSet<LodPageId> {
    let next = std::mem::take(&mut state.pending_page_leases);
    std::mem::replace(&mut state.current_page_leases, next)
}

fn retire_package_page_leases_best_effort(
    state: &mut PackageInstantiation,
    previous: BTreeSet<LodPageId>,
) -> Option<LodOrchestrationFailure> {
    let runtime = match state.runtime.get_mut() {
        Ok(runtime) => runtime,
        Err(_) => {
            return Some(LodOrchestrationFailure::with_detail(
                LodOrchestrationFailureCode::RuntimeFailed,
                "activated package cut; previous page-lease cleanup found a poisoned runtime",
            ));
        }
    };
    let mut first_failure = None;
    for page in previous {
        if let Err(error) = runtime.release_resident_page(page)
            && first_failure.is_none()
        {
            first_failure = Some(LodOrchestrationFailure::with_detail(
                LodOrchestrationFailureCode::RuntimeFailed,
                format!(
                    "activated package cut; previous page {} lease cleanup failed: {error}",
                    page.0
                ),
            ));
        }
    }
    first_failure
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

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PackageViewBlendEdgeKey {
    parent: LodNodeId,
    children: Vec<LodNodeId>,
    parent_metric: LodViewBlendMetric,
    child_metrics: Vec<LodViewBlendMetric>,
}

impl PackageViewBlendEdgeKey {
    fn from_edge(edge: &LodViewBlendEdge) -> Self {
        Self {
            parent: edge.parent(),
            children: edge.children().to_vec(),
            parent_metric: edge.parent_metric(),
            child_metrics: edge.child_metrics().to_vec(),
        }
    }
}

fn package_node_is_descendant_or_same<H>(
    hierarchy: &H,
    mut node: LodNodeId,
    ancestor: LodNodeId,
) -> bool
where
    H: LodHierarchy<NodeId = LodNodeId>,
{
    let mut visited = HashSet::new();
    loop {
        if node == ancestor {
            return true;
        }
        if !visited.insert(node) {
            return false;
        }
        let Some(parent) = hierarchy.parent(node) else {
            return false;
        };
        node = parent;
    }
}

fn package_apply_provable_initial_view_blend_endpoint<H>(
    hierarchy: &H,
    target_nodes: &BTreeSet<LodNodeId>,
    provable_nodes: &mut BTreeSet<LodNodeId>,
    parent: LodNodeId,
    children: &[LodNodeId],
    initial_weight_bits: u32,
    inherits_render_state: bool,
) -> bool
where
    H: LodHierarchy<NodeId = LodNodeId>,
{
    if children.is_empty() || hierarchy.children(parent) != children {
        return false;
    }
    let target_has_parent = target_nodes.contains(&parent);
    let target_has_any_child = children.iter().any(|child| target_nodes.contains(child));
    let target_has_all_children = children.iter().all(|child| target_nodes.contains(child));
    let target_is_parent_only = target_has_parent && !target_has_any_child;
    let target_is_complete_children_only = !target_has_parent && target_has_all_children;
    if !target_is_parent_only && !target_is_complete_children_only {
        return false;
    }
    let initial_parent = if initial_weight_bits == 0.0_f32.to_bits() {
        true
    } else if initial_weight_bits == 1.0_f32.to_bits() {
        false
    } else {
        return false;
    };

    provable_nodes.remove(&parent);
    for child in children {
        provable_nodes.remove(child);
    }
    if inherits_render_state {
        // A common immutable edge inherits its render-owned displayed weight;
        // it does not reset to `initial_weight`. Neither side is therefore
        // categorical evidence for retiring a different edge.
        return true;
    }
    if initial_parent {
        provable_nodes.insert(parent);
    } else {
        provable_nodes.extend(children.iter().copied());
    }
    true
}

fn package_replacement_provable_initial_drawable_nodes<H>(
    hierarchy: &H,
    target_nodes: &BTreeSet<LodNodeId>,
    edges: &[LodViewBlendEdge],
    inherited_edges: &HashSet<PackageViewBlendEdgeKey>,
) -> Option<BTreeSet<LodNodeId>>
where
    H: LodHierarchy<NodeId = LodNodeId>,
{
    let mut provable_nodes = target_nodes.clone();
    let mut occupied_endpoints = BTreeSet::new();
    for edge in edges {
        if !occupied_endpoints.insert(edge.parent())
            || edge
                .children()
                .iter()
                .any(|child| !occupied_endpoints.insert(*child))
        {
            return None;
        }
        let key = PackageViewBlendEdgeKey::from_edge(edge);
        if !package_apply_provable_initial_view_blend_endpoint(
            hierarchy,
            target_nodes,
            &mut provable_nodes,
            edge.parent(),
            edge.children(),
            edge.initial_weight_bits(),
            inherited_edges.contains(&key),
        ) {
            return None;
        }
    }
    Some(provable_nodes)
}

fn package_replacement_matches_removed_blend_endpoint<H>(
    hierarchy: &H,
    parent: LodNodeId,
    children: &[LodNodeId],
    endpoint: LodViewBlendEndpoint,
    replacement_nodes: &BTreeSet<LodNodeId>,
) -> bool
where
    H: LodHierarchy<NodeId = LodNodeId>,
{
    match endpoint {
        LodViewBlendEndpoint::Fractional => false,
        LodViewBlendEndpoint::ParentExact => replacement_nodes.contains(&parent),
        LodViewBlendEndpoint::ChildrenExact => children.iter().all(|child| {
            replacement_nodes.iter().copied().any(|replacement| {
                package_node_is_descendant_or_same(hierarchy, replacement, *child)
            })
        }),
    }
}

/// A replacement may add any number of disjoint boundaries while common edge
/// keys retain their render-owned state. Removing an edge is stricter: the
/// currently ACTIVE drawable mask must be categorical on the same topology side
/// represented by the replacement cut. This serializes only overlapping
/// ancestor/descendant work, not unrelated fractional branches.
fn package_candidate_set_view_blend_retirement_attestations<H>(
    hierarchy: &H,
    current: &LodRenderCandidates,
    replacement: &LodRenderCandidates,
) -> Option<BTreeMap<Entity, LodViewBlendPredecessorAttestation>>
where
    H: LodHierarchy<NodeId = LodNodeId>,
{
    let mut attestations = BTreeMap::new();
    for (camera, candidate) in &current.by_camera {
        if candidate.view_blend_mode() != Some(LodTemporalTransitionMode::Morphing) {
            continue;
        }
        let transition = candidate.temporal_transition()?;
        let morph = transition.morph()?;
        let retirement = candidate.view_blend_retirement_snapshot()?;
        if !candidate.render_is_active()
            || retirement.endpoints.len() != morph.edges().len()
            || retirement.status.edge_count as usize != morph.edges().len()
            || retirement.status.invalid_pressure_count != 0
            || retirement.status.missing_consumer_count != 0
        {
            return None;
        }

        let next = replacement.get(*camera)?;
        let next_edges = if next.view_blend_mode() == Some(LodTemporalTransitionMode::Morphing) {
            let next_morph = next
                .temporal_transition()
                .and_then(|transition| transition.morph())?;
            next_morph.edges()
        } else {
            &[]
        };
        let next_by_key = next_edges
            .iter()
            .map(|edge| (PackageViewBlendEdgeKey::from_edge(edge), edge))
            .collect::<HashMap<_, _>>();
        let replacement_nodes = next
            .target_render_ranges()
            .iter()
            .map(|range| range.node)
            .collect::<BTreeSet<_>>();
        let current_keys = morph
            .edges()
            .iter()
            .map(PackageViewBlendEdgeKey::from_edge)
            .collect::<HashSet<_>>();
        let replacement_initial_nodes = package_replacement_provable_initial_drawable_nodes(
            hierarchy,
            &replacement_nodes,
            next_edges,
            &current_keys,
        )?;

        let mut requirements = Vec::new();
        for (edge, endpoint) in morph
            .edges()
            .iter()
            .zip(retirement.endpoints.iter().copied())
        {
            let key = PackageViewBlendEdgeKey::from_edge(edge);
            if next_by_key.contains_key(&key) {
                continue;
            }
            if !package_replacement_matches_removed_blend_endpoint(
                hierarchy,
                edge.parent(),
                edge.children(),
                endpoint,
                &replacement_initial_nodes,
            ) {
                return None;
            }
            requirements.push(LodViewBlendRetirementRequirement::new(
                edge.clone(),
                endpoint,
            ));
        }
        if !requirements.is_empty() {
            let attestation =
                candidate.view_blend_predecessor_attestation(&retirement, requirements)?;
            attestations.insert(*camera, attestation);
        }
    }
    Some(attestations)
}

fn package_pending_active_presentation_is_safe(
    state: &mut PackageInstantiation,
) -> Result<bool, GaussianLodPackageError> {
    let Some(pending) = state.pending.clone() else {
        return Ok(true);
    };
    if pending.len() != state.pending_presentation_modes.len()
        || !pending.by_camera.iter().all(|(camera, candidate)| {
            let Some(expected_mode) = state.pending_presentation_modes.get(camera) else {
                return false;
            };
            if candidate.view_blend_status().is_some_and(|status| {
                status.invalid_pressure_count != 0 || status.missing_consumer_count != 0
            }) {
                return false;
            }
            if candidate.temporal_transition_mode() != *expected_mode {
                return false;
            }
            match candidate.active_presentation() {
                Some(LodRenderActivePresentation::ViewBlend) => {
                    *expected_mode == Some(LodTemporalTransitionMode::Morphing)
                }
                Some(LodRenderActivePresentation::HardTarget) => {
                    *expected_mode != Some(LodTemporalTransitionMode::Morphing)
                        && candidate.render_ranges() == candidate.target_render_ranges()
                }
                None => false,
            }
        })
    {
        return Ok(false);
    }
    if state.pending_progressive_view_blend
        && pending.by_camera.values().any(|candidate| {
            candidate.active_presentation() != Some(LodRenderActivePresentation::ViewBlend)
        })
    {
        return Ok(false);
    }
    let Some(current) = state.current.clone() else {
        return Ok(true);
    };
    let runtime = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    let Some(expected_attestations) = package_candidate_set_view_blend_retirement_attestations(
        runtime.hierarchy(),
        &current,
        &pending,
    ) else {
        return Ok(false);
    };
    let attached_count = pending
        .by_camera
        .values()
        .filter(|candidate| candidate.predecessor_view_blend_attestation().is_some())
        .count();
    Ok(attached_count == expected_attestations.len()
        && expected_attestations.iter().all(|(camera, expected)| {
            pending.get(*camera).is_some_and(|candidate| {
                candidate.predecessor_view_blend_attestation_epoch_is_current()
                    && candidate.predecessor_view_blend_attestation_matches(expected)
            })
        }))
}

fn package_candidate_set_matches_request(
    candidates: &LodRenderCandidates,
    request: &PackageCutRequestSignature,
) -> bool {
    let frozen = request.selection_mode == LodSelectionMode::Frozen;
    candidates.len() == request.cameras.len()
        && request.cameras.iter().all(|camera| {
            candidates.get(camera.entity).is_some_and(|candidate| {
                let frontier = candidate.frontier();
                frontier.view() == LodRuntimeViewId(camera.entity.to_bits())
                    && frontier.quality_status().requested_target == request.target
                    && frontier.selection_view_frozen() == frozen
                    && u64::from(frontier.candidate_count()) <= request.max_active_gaussians
            })
        })
}

fn package_candidate_set_is_active(candidates: &LodRenderCandidates) -> bool {
    !candidates.is_empty()
        && candidates
            .by_camera
            .values()
            .all(|candidate| candidate.active_presentation().is_some())
}

fn package_candidate_set_has_invalid_view_blend_pressure(candidates: &LodRenderCandidates) -> bool {
    candidates.by_camera.values().any(|candidate| {
        package_view_blend_status_has_invalid_pressure(candidate.view_blend_status())
    })
}

fn package_candidate_set_has_missing_view_blend_consumers(
    candidates: &LodRenderCandidates,
) -> bool {
    candidates.by_camera.values().any(|candidate| {
        package_view_blend_status_has_missing_consumers(candidate.view_blend_status())
    })
}

fn package_view_blend_status_has_missing_consumers(
    status: Option<LodViewBlendStatusSnapshot>,
) -> bool {
    status.is_some_and(|status| status.missing_consumer_count != 0)
}

fn package_view_blend_status_has_invalid_pressure(
    status: Option<LodViewBlendStatusSnapshot>,
) -> bool {
    status.is_some_and(|status| status.invalid_pressure_count != 0)
}

fn invalid_view_blend_pressure_failure() -> LodOrchestrationFailure {
    LodOrchestrationFailure::with_detail(
        LodOrchestrationFailureCode::UnsupportedConfiguration,
        "camera-conditioned LoD pressure became non-finite or threshold-contradictory; retaining the last drawable blend weights until a valid view evaluation recovers",
    )
}

const fn package_current_request_can_short_circuit(
    has_current: bool,
    request_matches: bool,
    predictive_maintenance_required: bool,
) -> bool {
    has_current && request_matches && !predictive_maintenance_required
}

fn package_request_ownership_after_commit(
    committed_request: Option<PackageCutRequestSignature>,
    committed_bootstrap: bool,
    request_fixed_point: bool,
) -> Option<PackageCutRequestSignature> {
    (!committed_bootstrap && request_fixed_point)
        .then_some(committed_request)
        .flatten()
}

const fn package_same_payload_request_fixed_point(
    selector_fixed_point: bool,
    selection_view_frozen_changed: bool,
    view_blend_lagging: bool,
) -> bool {
    selector_fixed_point && !selection_view_frozen_changed && !view_blend_lagging
}

/// A categorical legacy cohort remains an intermediate topology step and does
/// not own the whole camera/policy request until selector convergence. ABI16 is
/// different: an ACTIVE fractional edge table may itself be the stationary
/// view-conditioned fixed point, provided selection is fixed, no edge is
/// recovering from residency/Frozen lag, and a selection-mode change has had a
/// render publication turn. Pending staleness therefore follows drawable phase
/// ownership rather than assuming every authored blend must settle away.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackagePendingStaleDisposition {
    CancelSafe,
    FinishAndCommit,
}

fn reconcile_stale_package_pending_transition(
    pending: &mut LodRenderCandidates,
    live_views: &BTreeSet<Entity>,
    already_latched: bool,
) -> PackagePendingStaleDisposition {
    let live_transitioning = pending.by_camera.iter().any(|(camera, candidate)| {
        live_views.contains(camera) && candidate.render_is_transitioning()
    });
    if !already_latched && !live_transitioning {
        return PackagePendingStaleDisposition::CancelSafe;
    }

    // A removed camera owns no RenderView and therefore cannot draw or
    // acknowledge this transition. Retire only that consumer; every still-live
    // companion retains the shared union until the transaction reaches ACTIVE.
    for (camera, candidate) in &pending.by_camera {
        if !live_views.contains(camera) {
            candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
        }
    }
    pending
        .by_camera
        .retain(|camera, _| live_views.contains(camera));
    if pending.is_empty() {
        PackagePendingStaleDisposition::CancelSafe
    } else {
        PackagePendingStaleDisposition::FinishAndCommit
    }
}

fn materialize_package_cut(
    state: &mut PackageInstantiation,
    cut: &LodRenderCandidates,
    fallback_nodes: &BTreeSet<LodNodeId>,
    atlas_uploads: &mut LodAtlasUploadQueue,
    staging: &mut PackageStagingPermit<'_>,
    max_gpu_bytes: u64,
    debug_work: &mut PackageDebugPreparationWork,
) -> Result<bool, GaussianLodPackageError> {
    let ranges = cut
        .by_camera
        .values()
        .flat_map(LodRenderCandidate::required_atlas_ranges)
        .copied()
        .collect::<Vec<_>>();
    let normalized_ranges = normalize_package_ranges(&ranges);
    let reusable = state.staged.as_ref().is_some_and(|staged| {
        staged.ranges == normalized_ranges && staged.fallback_nodes == *fallback_nodes
    });
    let mut staged = if reusable {
        state.staged.take().expect("checked staged package cut")
    } else {
        discard_package_unpublished_staged_cut(state);
        prepare_package_staged_cut(state, &normalized_ranges, fallback_nodes)?
    };
    advance_package_staged_cut(state, &mut staged, atlas_uploads, staging, max_gpu_bytes)?;
    advance_package_staged_debug_preparation(state, &mut staged, debug_work)?;
    if !staged.complete || !staged.debug.complete {
        state.staged = Some(staged);
        return Ok(false);
    }
    validate_package_staged_cut(state, &staged)?;
    commit_package_staged_debug_annotations(
        state,
        &staged.ranges,
        &staged.fallback_nodes,
        &staged.debug,
    )?;
    publish_package_staged_cut(state, staged);
    Ok(true)
}

fn stage_package_cut_bounded(
    state: &mut PackageInstantiation,
    cut: &LodRenderCandidates,
    fallback_nodes: &BTreeSet<LodNodeId>,
    atlas_uploads: &mut LodAtlasUploadQueue,
    staging: &mut PackageStagingPermit<'_>,
    max_gpu_bytes: u64,
    debug_work: &mut PackageDebugPreparationWork,
) -> Result<bool, GaussianLodPackageError> {
    let ranges = package_candidate_staging_ranges(state.plan, cut)?;
    materialize_package_ranges_bounded(
        state,
        &ranges,
        fallback_nodes,
        atlas_uploads,
        staging,
        max_gpu_bytes,
        debug_work,
    )
}

fn materialize_package_frontiers_bounded(
    state: &mut PackageInstantiation,
    frontiers: &[LodCandidateFrontier],
    fallback_nodes: &BTreeSet<LodNodeId>,
    atlas_uploads: &mut LodAtlasUploadQueue,
    staging: &mut PackageStagingPermit<'_>,
    max_gpu_bytes: u64,
    debug_work: &mut PackageDebugPreparationWork,
) -> Result<bool, GaussianLodPackageError> {
    let ranges = frontiers
        .iter()
        .flat_map(|frontier| frontier.physical_ranges())
        .copied()
        .collect::<Vec<_>>();
    materialize_package_ranges_bounded(
        state,
        &ranges,
        fallback_nodes,
        atlas_uploads,
        staging,
        max_gpu_bytes,
        debug_work,
    )
}

fn materialize_package_ranges_bounded(
    state: &mut PackageInstantiation,
    ranges: &[LodPhysicalRange],
    fallback_nodes: &BTreeSet<LodNodeId>,
    atlas_uploads: &mut LodAtlasUploadQueue,
    staging: &mut PackageStagingPermit<'_>,
    max_gpu_bytes: u64,
    debug_work: &mut PackageDebugPreparationWork,
) -> Result<bool, GaussianLodPackageError> {
    let normalized_ranges = normalize_package_ranges(ranges);
    let target_pages = normalized_ranges
        .iter()
        .map(|range| range.page)
        .collect::<BTreeSet<_>>();
    replace_package_pending_page_leases(state, &target_pages)?;

    let reusable = state.staged.as_ref().is_some_and(|staged| {
        staged.ranges == normalized_ranges && staged.fallback_nodes == *fallback_nodes
    });
    let mut staged = if reusable {
        state.staged.take().expect("checked staged package cut")
    } else {
        discard_package_unpublished_staged_cut(state);
        prepare_package_staged_cut(state, &normalized_ranges, fallback_nodes)?
    };
    advance_package_staged_cut(state, &mut staged, atlas_uploads, staging, max_gpu_bytes)?;
    advance_package_staged_debug_preparation(state, &mut staged, debug_work)?;
    let complete = staged.complete && staged.debug.complete;
    state.staged = Some(staged);
    Ok(complete)
}

fn prepare_package_staged_cut(
    state: &mut PackageInstantiation,
    ranges: &[LodPhysicalRange],
    fallback_nodes: &BTreeSet<LodNodeId>,
) -> Result<PackageStagedCut, GaussianLodPackageError> {
    let normalized_ranges = normalize_package_ranges(ranges);
    let selection = plan_package_atlas_selection(state.plan, &normalized_ranges)?;
    let mut materializations = Vec::new();
    materializations
        .try_reserve_exact(selection.materializations.len())
        .map_err(|_| GaussianLodPackageError::AtlasSizeOverflow)?;
    for (page_id, slot) in selection.materializations {
        if !state.mirror.is_page_current(page_id, slot) {
            materializations.push((page_id, slot));
        }
    }
    let complete = materializations.is_empty();
    let mut staged = PackageStagedCut {
        ranges: normalized_ranges,
        slots: selection.selected_slots,
        materializations,
        next_materialization: 0,
        complete,
        fallback_nodes: fallback_nodes.clone(),
        debug: PackageStagedDebugPreparation::default(),
    };
    reset_package_staged_debug_preparation(state, &mut staged);
    Ok(staged)
}

fn advance_package_staged_cut(
    state: &mut PackageInstantiation,
    staged: &mut PackageStagedCut,
    atlas_uploads: &mut LodAtlasUploadQueue,
    staging: &mut PackageStagingPermit<'_>,
    max_gpu_bytes: u64,
) -> Result<(), GaussianLodPackageError> {
    ensure_package_staged_debug_preparation(state, staged);
    let runtime = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    while let Some(&(page_id, slot)) = staged.materializations.get(staged.next_materialization) {
        let atlas_current = state.mirror.is_page_current(page_id, slot);
        if !atlas_current
            && !staging.try_consume_slot(
                state.atlas.id(),
                slot.index,
                state.plan.gaussians_per_slot,
                max_gpu_bytes,
            )?
        {
            break;
        }
        let page = runtime
            .decoded_page(page_id)
            .ok_or(GaussianLodPackageError::ResidentPageNotDecoded(page_id))?;
        if !atlas_current {
            let payload = state
                .mirror
                .materialize_page_payload(page, slot)
                .map_err(GaussianLodPackageError::RenderCommit)?;
            state
                .transient_atlas
                .write_slot(slot.index, state.plan.gaussians_per_slot, payload)
                .map_err(|error| GaussianLodPackageError::AtlasUpload(error.to_string()))?;
            atlas_uploads
                .enqueue_slot(state.atlas.id(), slot, state.plan.gaussians_per_slot)
                .map_err(|error| GaussianLodPackageError::AtlasUpload(error.to_string()))?;
        }
        if let Some(debug) = state.debug.as_ref() {
            let metadata_is_current = debug.atlas.page_matches_indexed_node_residency(
                &debug.index,
                page_id,
                slot,
                |node| {
                    if staged.fallback_nodes.contains(&node) {
                        LodDebugResidency::AncestorFallback
                    } else {
                        LodDebugResidency::Resident
                    }
                },
            );
            if !metadata_is_current && staged.debug.targets.insert((page_id, slot)) {
                // Keep pending-only data outside the live sparse atlas. The
                // retained current cut stays fully debug-ready while this Arc
                // is built under the separate CPU record budget.
                staged.debug.pending.push_back((page_id, slot));
                staged.debug.complete = false;
            }
        }
        staged.next_materialization += 1;
    }
    staged.complete = staged.next_materialization == staged.materializations.len();
    if staged.complete {
        state
            .mirror
            .validate_ranges(&staged.ranges)
            .map_err(GaussianLodPackageError::RenderCommit)?;
    }
    Ok(())
}

fn reset_package_staged_debug_preparation(
    state: &mut PackageInstantiation,
    staged: &mut PackageStagedCut,
) {
    let retained_current_targets = state
        .visible_ranges
        .iter()
        .map(|range| (range.page, range.slot))
        .collect();
    let PackageInstantiation {
        debug: Some(debug),
        mirror,
        ..
    } = state
    else {
        staged.debug = PackageStagedDebugPreparation {
            complete: true,
            ..default()
        };
        return;
    };
    let identity = debug
        .atlas
        .metadata()
        .sparse()
        .map(|sparse| sparse.identity());
    let pending = staged
        .ranges
        .iter()
        .map(|range| (range.page, range.slot))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|&(page_id, slot)| {
            mirror.is_page_current(page_id, slot)
                && !debug.atlas.page_matches_indexed_node_residency(
                    &debug.index,
                    page_id,
                    slot,
                    |node| {
                        if staged.fallback_nodes.contains(&node) {
                            LodDebugResidency::AncestorFallback
                        } else {
                            LodDebugResidency::Resident
                        }
                    },
                )
        })
        .collect::<VecDeque<_>>();
    let targets = pending.iter().copied().collect();
    let complete = pending.is_empty();
    staged.debug = PackageStagedDebugPreparation {
        sidecar_identity: identity,
        targets,
        retained_current_targets,
        pending,
        prepared: Vec::new(),
        prepublished: BTreeSet::new(),
        complete,
    };
}

fn ensure_package_staged_debug_preparation(
    state: &mut PackageInstantiation,
    staged: &mut PackageStagedCut,
) {
    let identity = state.debug.as_ref().and_then(|debug| {
        debug
            .atlas
            .metadata()
            .sparse()
            .map(|sparse| sparse.identity())
    });
    if staged.debug.sidecar_identity != identity {
        reset_package_staged_debug_preparation(state, staged);
    }
}

fn advance_package_staged_debug_preparation(
    state: &mut PackageInstantiation,
    staged: &mut PackageStagedCut,
    work: &mut PackageDebugPreparationWork,
) -> Result<(), GaussianLodPackageError> {
    ensure_package_staged_debug_preparation(state, staged);
    if staged.debug.complete {
        return Ok(());
    }
    let PackageInstantiation {
        debug: Some(debug),
        runtime,
        mirror,
        plan,
        ..
    } = state
    else {
        staged.debug.complete = true;
        return Ok(());
    };
    let runtime = runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?;
    while let Some(&(page_id, slot)) = staged.debug.pending.front() {
        if !mirror.is_page_current(page_id, slot) {
            break;
        }
        if debug
            .atlas
            .page_matches_indexed_node_residency(&debug.index, page_id, slot, |node| {
                if staged.fallback_nodes.contains(&node) {
                    LodDebugResidency::AncestorFallback
                } else {
                    LodDebugResidency::Resident
                }
            })
        {
            staged.debug.pending.pop_front();
            continue;
        }
        let record_work = debug.prepared_page_record_work(
            page_id,
            slot,
            plan.gaussians_per_slot,
            &staged.fallback_nodes,
        );
        let regenerated_records = if !debug.page_basis_is_current(page_id, slot) {
            plan.gaussians_per_slot as usize
        } else {
            0
        };
        if !work.can_consume(record_work) {
            break;
        }
        let page = runtime
            .decoded_page(page_id)
            .ok_or(GaussianLodPackageError::ResidentPageNotDecoded(page_id))?;
        let records = debug.prepared_page_records(
            page,
            slot,
            plan.gaussians_per_slot,
            &staged.fallback_nodes,
        )?;
        staged.debug.pending.pop_front();
        if !staged
            .debug
            .retained_current_targets
            .contains(&(page_id, slot))
        {
            debug
                .atlas
                .write_prepared_sparse_page(page_id, slot, Arc::clone(&records))
                .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?;
            staged.debug.prepublished.insert((page_id, slot));
        }
        staged.debug.prepared.push((page_id, slot, records));
        work.consume_prepared_page(record_work, regenerated_records);
    }
    staged.debug.complete = staged.debug.pending.is_empty();
    Ok(())
}

fn validate_package_staged_cut(
    state: &PackageInstantiation,
    staged: &PackageStagedCut,
) -> Result<(), GaussianLodPackageError> {
    if !staged.complete {
        return Err(GaussianLodPackageError::RenderCommitFailed {
            detail: format!(
                "render activated an incomplete staged cut: completed_materializations={}, total_materializations={}",
                staged.next_materialization,
                staged.materializations.len()
            ),
        });
    }
    state
        .mirror
        .validate_ranges(&staged.ranges)
        .map_err(GaussianLodPackageError::RenderCommit)
}

/// Atomically advances cut-dependent debug provenance at the same logical
/// boundary as the staged visible cut.
///
/// Both newly allocated slots and pages shared with the retained cut remain in
/// the staged Arc set until this function. Work is applied to a copy-on-write
/// atlas and swapped only after every page succeeds, so an error preserves the
/// complete current-cut snapshot and pending-only revisions never gate it.
fn commit_package_staged_debug_annotations(
    state: &mut PackageInstantiation,
    ranges: &[LodPhysicalRange],
    fallback_nodes: &BTreeSet<LodNodeId>,
    preparation: &PackageStagedDebugPreparation,
) -> Result<(), GaussianLodPackageError> {
    let PackageInstantiation {
        debug: Some(debug),
        mirror,
        ..
    } = state
    else {
        return Ok(());
    };

    let sidecar_identity = debug
        .atlas
        .metadata()
        .sparse()
        .map(|sparse| sparse.identity());
    if preparation.sidecar_identity != sidecar_identity || !preparation.complete {
        return Err(GaussianLodPackageError::DebugAnnotations(
            "staged debug provenance was not completely prepared for the current sidecar"
                .to_owned(),
        ));
    }
    let targets = ranges
        .iter()
        .map(|range| (range.page, range.slot))
        .collect::<BTreeSet<_>>();
    let mut next_atlas = debug.atlas.clone();
    next_atlas.set_complete(false);
    for (page_id, slot, records) in &preparation.prepared {
        if preparation.prepublished.contains(&(*page_id, *slot)) {
            continue;
        }
        next_atlas
            .write_prepared_sparse_page(*page_id, *slot, Arc::clone(records))
            .map_err(|error| GaussianLodPackageError::DebugAnnotations(error.to_string()))?;
    }
    for (page_id, slot) in targets {
        if !mirror.is_page_current(page_id, slot) {
            return Err(GaussianLodPackageError::RenderCommit(
                LodRenderCommitError::FrontierReferencesUnsynchronizedPage {
                    page: page_id,
                    slot,
                },
            ));
        }
        if !next_atlas.page_matches_indexed_node_residency(&debug.index, page_id, slot, |node| {
            if fallback_nodes.contains(&node) {
                LodDebugResidency::AncestorFallback
            } else {
                LodDebugResidency::Resident
            }
        }) {
            return Err(GaussianLodPackageError::DebugAnnotations(format!(
                "staged debug provenance for page {} slot {} was not prepared",
                page_id.0, slot.index
            )));
        }
    }
    next_atlas.set_complete(true);
    debug.atlas = next_atlas;
    debug.initialization.clear();
    Ok(())
}

/// Publishes all logical visible ownership without fallible cleanup work.
/// Validation and target-page leasing must already be complete.
fn publish_package_staged_cut(state: &mut PackageInstantiation, staged: PackageStagedCut) {
    debug_assert!(staged.complete);
    state.visible_slots = staged.slots;
    state.visible_ranges = staged.ranges;
    state.visible_fallback_nodes = staged.fallback_nodes;
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

fn package_gpu_staging_step_byte_limit(settings: &GaussianLodSettings) -> u64 {
    settings
        .budgets
        .max_gpu_upload_bytes_per_commit
        .min(settings.budgets.max_upload_bytes_per_frame)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
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
    #[cfg(all(test, not(target_arch = "wasm32")))]
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
        #[cfg(all(test, not(target_arch = "wasm32")))]
        intervals_by_slot,
        selected_slots,
        materializations,
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
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
#[cfg(all(test, not(target_arch = "wasm32")))]
fn rewrite_atlas_to_ranges<T: LodPageTransport>(
    runtime: &LodStreamingRuntime<T>,
    mirror: &mut LodPageAtlasMirror,
    mut debug: Option<&mut PackageDebugAnnotations>,
    plan: GaussianLodPackageAtlasPlan,
    ranges: &[LodPhysicalRange],
    fallback_nodes: &BTreeSet<LodNodeId>,
    _previous_slots: &BTreeMap<u32, AtlasSlot>,
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
        selected_slots,
        materializations,
        ..
    } = selection;
    let mut dirty_slots = BTreeSet::new();
    for (page_id, slot) in materializations {
        let already_materialized = mirror.is_page_current(page_id, slot);
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
        if !already_materialized {
            dirty_slots.insert(slot.index);
        }
    }
    mirror
        .validate_ranges(ranges)
        .map_err(GaussianLodPackageError::RenderCommit)?;

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
    let retained_current = state.current.is_some();
    let candidates_are_current = state.pending.is_none() && retained_current;
    let mut candidates = state
        .pending
        .as_ref()
        .or(state.current.as_ref())
        .cloned()
        .unwrap_or_else(LodRenderCandidates::package_required);
    candidates.candidate_draw_required = true;
    candidates.retained_current = retained_current;
    candidates.candidates_are_current = candidates_are_current;
    candidates.retained_current_is_stale = retained_current && !state.current_request_matches_live;
    candidates.transition_must_commit = state.pending_transition_must_commit;
    candidates.debug_metadata_staged = state.debug.is_none()
        || state.pending.is_none()
        || state
            .staged
            .as_ref()
            .is_some_and(|staged| staged.debug.complete);
    let mut entity_commands = commands.entity(entity);
    entity_commands.insert(candidates);
    if let Some(debug) = &state.debug {
        entity_commands.insert(debug.atlas.metadata());
    } else {
        entity_commands.remove::<LodDebugMetadata>();
    }
    #[cfg(feature = "testing")]
    {
        let (runtime_work_available, work) = match state.runtime.get_mut() {
            Ok(runtime) => (true, runtime.package_work_counts_for_testing()),
            Err(_) => (false, Default::default()),
        };
        let (queued, in_flight, preprocess, capacity_blocked, requested_pages, split_cohort) = work;
        let pending = state.pending.as_ref();
        entity_commands.insert(GaussianLodPackageTestingSnapshot {
            current_cut_request_identity_present: state.current_request.is_some(),
            current_cut_matches_live_request: state.current_request_matches_live,
            pending_present: pending.is_some(),
            pending_all_prepared_or_later: pending.is_some_and(|candidates| {
                candidates
                    .by_camera
                    .values()
                    .all(LodRenderCandidate::render_is_prepared)
            }),
            pending_all_render_active: pending.is_some_and(package_candidate_set_is_active),
            pending_any_render_transitioning: pending.is_some_and(|candidates| {
                candidates
                    .by_camera
                    .values()
                    .any(LodRenderCandidate::render_is_transitioning)
            }),
            pending_any_view_blend_replan_requested: pending.is_some_and(|candidates| {
                candidates
                    .by_camera
                    .values()
                    .any(LodRenderCandidate::view_blend_replan_requested)
            }),
            runtime_work_available,
            runtime_request_queue_len: queued,
            runtime_transport_in_flight_requests: in_flight,
            preprocess_waiting_jobs: preprocess.waiting,
            preprocess_backend_tracked_jobs: preprocess.submitted,
            preprocess_ready_pages: preprocess.ready,
            runtime_capacity_blocked_requests: capacity_blocked,
            runtime_max_last_observed_view_requested_pages: requested_pages,
            runtime_split_cohort_admitted: split_cohort,
            view_blend_retirement_attestation_retry_count: state
                .view_blend_retirement_attestation_retry_count,
        });
    }
    let phase = package_status_phase(state);
    entity_commands.insert(GaussianLodPackageStatus {
        phase,
        resident_pages: state.resident_pages,
        active_gaussians: state.active_gaussians,
        terminal_failures: state.terminal_failures,
        failure: state.last_failure.clone(),
    });
}

fn package_status_phase(state: &PackageInstantiation) -> GaussianLodPackagePhase {
    let current_active = state
        .current
        .as_ref()
        .is_some_and(package_candidate_set_is_active);
    if current_active {
        if state.terminal_failures > 0 || state.last_failure.is_some() {
            GaussianLodPackagePhase::Degraded
        } else {
            GaussianLodPackagePhase::Active
        }
    } else if state.current.is_some() {
        // A retained candidate that is WAITING/PREPARED cannot currently draw.
        // Keep recovery distinguishable from cold loading and surface the
        // diagnostic installed by `drive_package_state`.
        if state.current_recovery_queued
            || state.terminal_failures > 0
            || state.last_failure.is_some()
        {
            GaussianLodPackagePhase::Degraded
        } else {
            GaussianLodPackagePhase::Loading
        }
    } else if state.terminal_failures > 0 || state.last_failure.is_some() {
        GaussianLodPackagePhase::Failed
    } else {
        // A CPU-materialized root is not itself a draw capability: package
        // atlases require a per-camera filtered candidate. Cold/new views stay
        // Loading until that first ancestor candidate completes compaction.
        GaussianLodPackagePhase::Loading
    }
}

fn publish_package_failure(
    entity: Entity,
    state: &mut PackageInstantiation,
    error: GaussianLodPackageError,
    commands: &mut Commands,
) {
    let error = clear_package_pending_transaction(state)
        .err()
        .unwrap_or(error);
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
    MainWorldStagingBudget(LodAtlasUploadBudgetError),
    GpuUploadCommitTooLarge {
        dirty_slots: u64,
        bytes: u64,
        limit: u64,
    },
    AtomicHandoffCapacityExceeded {
        required_pages: u64,
        limit_pages: u64,
        required_decoded_bytes: u64,
        limit_decoded_bytes: u64,
        required_gaussians: u64,
        limit_gaussians: u64,
    },
    SplitCohortCapacityExceeded {
        view: LodRuntimeViewId,
        parent: LodNodeId,
        required_pages: u64,
        limit_pages: u64,
        required_decoded_bytes: u64,
        limit_decoded_bytes: u64,
        required_gaussians: u64,
        limit_gaussians: u64,
    },
    RuntimePoisoned,
    MissingAtlasAsset,
    CompletedPageNotResident(LodPageId),
    ResidentPageNotDecoded(LodPageId),
    RenderCommitFailed {
        detail: String,
    },
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
            GaussianLodPackageError::RenderCommitFailed { .. } => {
                LodOrchestrationFailureCode::RenderCommitFailed
            }
            GaussianLodPackageError::AtlasCannotFitPage { .. }
            | GaussianLodPackageError::RootFallbackExceedsAtlas { .. }
            | GaussianLodPackageError::AtlasSizeOverflow
            | GaussianLodPackageError::AtlasAllocationFailed { .. }
            | GaussianLodPackageError::MainWorldStagingBudget(_)
            | GaussianLodPackageError::GpuUploadCommitTooLarge { .. }
            | GaussianLodPackageError::AtomicHandoffCapacityExceeded { .. }
            | GaussianLodPackageError::SplitCohortCapacityExceeded { .. }
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
