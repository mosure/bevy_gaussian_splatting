//! Automatic Bevy integration for bounded Gaussian LoD streaming.
//!
//! Flat 3D clouds are converted once into an ephemeral hierarchy, streamed
//! through the same validated runtime as packaged scenes, mirrored into a
//! fixed-capacity planar atlas, and committed per camera to GPU compaction.
//! A permanently resident, globally covering guard cut makes every atlas-backed
//! transition complete under arbitrary camera motion. The immutable flat source
//! is used only during cold construction; normal motion, quality pressure, and
//! capacity recovery stay within the bounded page-cache atlas.

#[cfg(all(not(target_arch = "wasm32"), feature = "sort_rayon"))]
use std::sync::OnceLock;
use std::{
    any::TypeId,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
};

#[cfg(not(target_arch = "wasm32"))]
use bevy::tasks::{AsyncComputeTaskPool, Task, TaskPool, futures::check_ready};
use bevy::{
    asset::AssetEventSystems,
    camera::{
        CameraUpdateSystems, Projection,
        visibility::{VisibilitySystems, VisibleEntities},
    },
    prelude::*,
    transform::TransformSystems,
};
use bevy_interleave::prelude::{Planar, PlanarHandle};
#[cfg(all(not(target_arch = "wasm32"), feature = "sort_rayon"))]
use rayon::prelude::*;

#[cfg(test)]
use crate::gaussian::formats::planar_3d_lod::build_planar_3d_lod;
use crate::{
    CloudSettings, GaussianCamera,
    gaussian::{
        cloud::CloudVisibilityClass,
        formats::{
            planar_3d::{
                Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle,
                gaussian_3d_gpu_bytes_per_record,
            },
            planar_3d_chunked::{LodNodeId, LodPageId, PlanarGaussian3dPage},
            planar_3d_lod::{
                GaussianLodBuildSettings, LodBuildError, PlanarGaussian3dLod,
                build_planar_3d_lod_owned_cancelable,
            },
        },
        lod_debug::{
            LodDebugAnnotationAtlas, LodDebugManifestIndex, LodDebugMetadata, LodDebugResidency,
        },
        lod_settings::{
            GaussianLodSettings, GaussianStreamingSettings, LodDegradation, LodQualityEndpoint,
            LodSelectionMode,
        },
    },
    io::{
        lod::{GaussianLodHandle, encode_page},
        lodge::GaussianLodgeHandle,
    },
    sort::{SortedEntries, SortedEntriesHandle},
    stream::{
        LodRenderPathSupportError,
        atlas_upload::{
            GaussianLodAtlasUploadPlugin, LodAtlasUploadQueue, LodTransientAtlas,
            LodTransientAtlasRegistry,
        },
        cache::AtlasSlot,
        hierarchy::LodView,
        render_commit::{
            GaussianLodRenderCommitPlugin, LodOrchestrationFailure, LodOrchestrationFailureCode,
            LodOrchestrationSource, LodOrchestrationTransition, LodOrchestrationTransitionKind,
            LodRenderCommitError,
        },
        require_lod_render_path,
        runtime::{
            LodCandidateFrontier, LodPhysicalRange, LodRuntimeError, LodRuntimeFrameId,
            LodRuntimeViewId, LodStreamFrame, LodStreamingRuntime,
        },
        transport::{LodPageTransport, MemoryPageTransport},
    },
};

pub use crate::stream::render_commit::{
    LodPageAtlasMirror, LodRenderCandidate, LodRenderCandidates,
};

use crate::stream::render_commit::{
    LOD_RENDER_ACTIVE, LOD_RENDER_FAILED, LOD_RENDER_PREPARED, LOD_RENDER_WAITING,
};

const CAPACITY_PRESSURE_STABLE_FRAMES: u8 = 4;
const CAPACITY_PRESSURE_ESCAPE_FRAMES: u8 = 32;
#[cfg(not(target_arch = "wasm32"))]
const EPHEMERAL_SNAPSHOT_RECORDS_PER_UPDATE: usize = 32 * 1024;
#[cfg(any(test, target_arch = "wasm32"))]
const WASM_SYNCHRONOUS_EPHEMERAL_SOURCE_LIMIT: u32 = 1_024;

/// Global hard bounds for automatic flat-cloud hierarchy construction and its
/// physical GPU atlas. Existing packaged/runtime APIs remain available for
/// virtual scenes larger than the ephemeral source bound.
#[derive(Resource, Clone, Debug)]
pub struct GaussianLodBridgeConfig {
    pub auto_build_flat_clouds: bool,
    pub max_ephemeral_source_gaussians: u32,
    pub max_ephemeral_stored_gaussians: u64,
    pub max_atlas_gaussians: u32,
    /// Hard canonical-plus-derived GPU storage bound for the physical atlas.
    pub max_atlas_bytes: u64,
    pub max_views_per_cloud: u32,
    pub build_settings: GaussianLodBuildSettings,
    pub streaming_settings: GaussianStreamingSettings,
}

impl Default for GaussianLodBridgeConfig {
    fn default() -> Self {
        let streaming_settings = GaussianStreamingSettings::default();
        Self {
            auto_build_flat_clouds: true,
            max_ephemeral_source_gaussians: 262_144,
            max_ephemeral_stored_gaussians: 524_288,
            max_atlas_gaussians: 524_288,
            max_atlas_bytes: 512 * 1024 * 1024,
            max_views_per_cloud: 16,
            // The CPU default is a binary progressive MomentMerge hierarchy.
            // Its branching factor is the maximum direct record amplification,
            // while 1024-record leaves retain efficient atlas/page granularity.
            build_settings: GaussianLodBuildSettings::default(),
            streaming_settings,
        }
    }
}

impl GaussianLodBridgeConfig {
    pub fn validate(&self) -> Result<(), LodBridgeError> {
        self.validate_structure()?;
        self.streaming_settings
            .validate()
            .map_err(|error| LodBridgeError::StreamingSettings(error.to_string()))?;
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), LodBridgeError> {
        if self.max_ephemeral_source_gaussians == 0 {
            return Err(LodBridgeError::ZeroLimit("max_ephemeral_source_gaussians"));
        }
        if self.max_ephemeral_stored_gaussians == 0 {
            return Err(LodBridgeError::ZeroLimit("max_ephemeral_stored_gaussians"));
        }
        if self.max_atlas_gaussians == 0 {
            return Err(LodBridgeError::ZeroLimit("max_atlas_gaussians"));
        }
        if self.max_atlas_bytes == 0 {
            return Err(LodBridgeError::ZeroLimit("max_atlas_bytes"));
        }
        if self.max_views_per_cloud == 0 {
            return Err(LodBridgeError::ZeroLimit("max_views_per_cloud"));
        }
        self.build_settings
            .validate()
            .map_err(|error| LodBridgeError::Build(error.to_string()))?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Reflect)]
pub enum GaussianLodBridgePhase {
    Building,
    StreamingFallback,
    WaitingForRender,
    Active,
    CompleteFallback,
}

/// Per-cloud observability for automatic LoD. Errors are sticky until the
/// source changes or LoD returns to the original-quality endpoint.
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component)]
pub struct GaussianLodBridgeStatus {
    pub phase: GaussianLodBridgePhase,
    pub active_views: u32,
    pub resident_pages: u32,
    pub active_gaussians: u64,
    pub failure: Option<LodOrchestrationFailure>,
}

impl GaussianLodBridgeStatus {
    fn fallback(error: LodBridgeError) -> Self {
        Self {
            phase: GaussianLodBridgePhase::CompleteFallback,
            active_views: 0,
            resident_pages: 0,
            active_gaussians: 0,
            failure: Some(LodOrchestrationFailure::from(&error)),
        }
    }

    /// Human-readable context retained for compatibility with logging and UI
    /// code that previously consumed an untyped error string.
    pub fn error_detail(&self) -> Option<&str> {
        self.failure.as_ref().and_then(|failure| failure.detail())
    }
}

trait ErasedLodRuntime: Send + Sync {
    fn begin_frame(&mut self) -> LodRuntimeFrameId;
    fn finish_frame(&mut self, frame: LodRuntimeFrameId) -> Result<(), LodRuntimeError>;
    fn update_view_in_frame(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        view: LodView,
        settings: &GaussianLodSettings,
        streaming: &GaussianStreamingSettings,
    ) -> Result<LodStreamFrame, LodRuntimeError>;
    fn coverage_guard_candidate(
        &mut self,
        view_id: LodRuntimeViewId,
        view: LodView,
        settings: &GaussianLodSettings,
    ) -> Result<Option<LodCandidateFrontier>, LodRuntimeError>;
    fn remove_view(&mut self, view: LodRuntimeViewId) -> Result<bool, LodRuntimeError>;
    fn restore_rendered_frontier(
        &mut self,
        view: LodRuntimeViewId,
        nodes: &[LodNodeId],
    ) -> Result<(), LodRuntimeError>;
    fn retry_from_rendered_frontier(
        &mut self,
        view: LodRuntimeViewId,
        nodes: &[LodNodeId],
    ) -> Result<(), LodRuntimeError>;
    fn retain_resident_page(&mut self, page: LodPageId) -> Result<AtlasSlot, LodRuntimeError>;
    fn release_resident_page(&mut self, page: LodPageId) -> Result<(), LodRuntimeError>;
    fn resident_slot(&self, page: LodPageId) -> Option<AtlasSlot>;
    fn resident_pin_count(&self, page: LodPageId) -> Option<u32>;
    fn decoded_page(&self, page: LodPageId) -> Option<PlanarGaussian3dPage>;
    fn parent(&self, node: LodNodeId) -> Option<LodNodeId>;
}

impl<T> ErasedLodRuntime for LodStreamingRuntime<T>
where
    T: LodPageTransport + Send + Sync,
    T::Ticket: Send + Sync,
{
    fn begin_frame(&mut self) -> LodRuntimeFrameId {
        LodStreamingRuntime::begin_frame(self)
    }

    fn finish_frame(&mut self, frame: LodRuntimeFrameId) -> Result<(), LodRuntimeError> {
        LodStreamingRuntime::finish_frame(self, frame).map(|_| ())
    }

    fn update_view_in_frame(
        &mut self,
        frame: LodRuntimeFrameId,
        view_id: LodRuntimeViewId,
        view: LodView,
        settings: &GaussianLodSettings,
        streaming: &GaussianStreamingSettings,
    ) -> Result<LodStreamFrame, LodRuntimeError> {
        LodStreamingRuntime::update_view_in_frame(self, frame, view_id, view, settings, streaming)
    }

    fn coverage_guard_candidate(
        &mut self,
        view_id: LodRuntimeViewId,
        view: LodView,
        settings: &GaussianLodSettings,
    ) -> Result<Option<LodCandidateFrontier>, LodRuntimeError> {
        LodStreamingRuntime::coverage_guard_candidate(self, view_id, view, settings)
    }

    fn remove_view(&mut self, view: LodRuntimeViewId) -> Result<bool, LodRuntimeError> {
        LodStreamingRuntime::remove_view(self, view)
    }

    fn restore_rendered_frontier(
        &mut self,
        view: LodRuntimeViewId,
        nodes: &[LodNodeId],
    ) -> Result<(), LodRuntimeError> {
        LodStreamingRuntime::restore_rendered_frontier(self, view, nodes)
    }

    fn retry_from_rendered_frontier(
        &mut self,
        view: LodRuntimeViewId,
        nodes: &[LodNodeId],
    ) -> Result<(), LodRuntimeError> {
        LodStreamingRuntime::retry_from_rendered_frontier(self, view, nodes)
    }

    fn retain_resident_page(&mut self, page: LodPageId) -> Result<AtlasSlot, LodRuntimeError> {
        LodStreamingRuntime::retain_resident_page(self, page)
    }

    fn release_resident_page(&mut self, page: LodPageId) -> Result<(), LodRuntimeError> {
        LodStreamingRuntime::release_resident_page(self, page)
    }

    fn resident_slot(&self, page: LodPageId) -> Option<AtlasSlot> {
        self.cache().get(page).map(|resident| resident.slot)
    }

    fn resident_pin_count(&self, page: LodPageId) -> Option<u32> {
        self.cache().get(page).map(|resident| resident.pin_count)
    }

    fn decoded_page(&self, page: LodPageId) -> Option<PlanarGaussian3dPage> {
        LodStreamingRuntime::decoded_page(self, page).cloned()
    }

    fn parent(&self, node: LodNodeId) -> Option<LodNodeId> {
        self.hierarchy().node(node).and_then(|node| node.parent)
    }
}

#[derive(Clone, Copy, Debug)]
struct StructuralSettings {
    max_resident_gaussians: u64,
    max_resident_bytes: u64,
    max_resident_pages: u32,
    max_pending_requests: u32,
}

impl StructuralSettings {
    fn apply(self, settings: &GaussianLodSettings) -> GaussianLodSettings {
        let mut effective = settings.clone();
        effective.budgets.max_resident_gaussians = self.max_resident_gaussians;
        effective.budgets.max_resident_bytes = self.max_resident_bytes;
        effective.budgets.max_resident_pages = self.max_resident_pages;
        effective.budgets.max_pending_requests = self.max_pending_requests;
        effective
    }
}

/// Inputs whose change invalidates the runtime hierarchy, atlas layout, or
/// runtime structural contract. Visual selection and per-frame work controls
/// remain live-updateable; constructor-sized preprocess capacities do not.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BridgeStructuralSignature {
    build_settings: GaussianLodBuildSettings,
    max_ephemeral_source_gaussians: u32,
    max_ephemeral_stored_gaussians: u64,
    max_atlas_gaussians: u32,
    max_atlas_bytes: u64,
    max_resident_gaussians: u64,
    max_resident_bytes: u64,
    max_resident_pages: u32,
    max_pending_requests: u32,
    max_upload_bytes_per_frame: u64,
    max_gpu_upload_bytes_per_commit: u64,
    max_concurrent_requests: u32,
    max_encoded_page_bytes: u64,
    debug_metadata: bool,
}

impl BridgeStructuralSignature {
    fn new(
        settings: &GaussianLodSettings,
        streaming: &GaussianStreamingSettings,
        config: &GaussianLodBridgeConfig,
        debug_metadata: bool,
    ) -> Self {
        Self {
            build_settings: config.build_settings,
            max_ephemeral_source_gaussians: config.max_ephemeral_source_gaussians,
            max_ephemeral_stored_gaussians: config.max_ephemeral_stored_gaussians,
            max_atlas_gaussians: config.max_atlas_gaussians,
            max_atlas_bytes: config.max_atlas_bytes,
            max_resident_gaussians: settings.budgets.max_resident_gaussians,
            max_resident_bytes: settings.budgets.max_resident_bytes,
            max_resident_pages: settings.budgets.max_resident_pages,
            max_pending_requests: settings.budgets.max_pending_requests,
            max_upload_bytes_per_frame: settings.budgets.max_upload_bytes_per_frame,
            max_gpu_upload_bytes_per_commit: settings.budgets.max_gpu_upload_bytes_per_commit,
            max_concurrent_requests: streaming.max_concurrent_requests,
            max_encoded_page_bytes: streaming.effective_max_encoded_page_bytes(),
            debug_metadata,
        }
    }
}

struct BridgeCloudState {
    source: Handle<PlanarGaussian3d>,
    source_gaussian_count: u32,
    atlas: Handle<PlanarGaussian3d>,
    transient_atlas: Option<LodTransientAtlas>,
    /// Last GPU allocation generation whose current/guard pages were queued.
    transient_atlas_generation: Option<u64>,
    runtime: Box<dyn ErasedLodRuntime>,
    mirror: LodPageAtlasMirror,
    debug_atlas: Option<LodDebugAnnotationAtlas>,
    debug_manifest_index: Option<LodDebugManifestIndex>,
    debug_slots: Vec<Option<DebugAtlasSlotRecord>>,
    fallback_debug_metadata: LodDebugMetadata,
    source_debug_metadata: Option<LodDebugMetadata>,
    debug_revision: u64,
    published_debug: Option<(bool, u64)>,
    #[cfg(test)]
    decoded_page_acquisitions: u64,
    #[cfg(test)]
    pre_frame_pending_lease_acquisitions: u64,
    #[cfg(test)]
    pre_frame_staged_replacement_retentions: u64,
    #[cfg(test)]
    deferred_ordinary_publications: u64,
    structural: StructuralSettings,
    signature: BridgeStructuralSignature,
    streaming: GaussianStreamingSettings,
    /// Last all-camera candidate set published ACTIVE by the render world.
    /// Its physical pages remain leased while a moving camera stages a
    /// replacement, so the retained GPU output can never observe slot reuse.
    current: Option<LodRenderCandidates>,
    current_page_leases: BTreeSet<LodPageId>,
    /// Replacement pages are leased before their atlas uploads are exposed to
    /// render. ACTIVE then transfers this set to `current_page_leases` before
    /// any old-only page is released.
    pending_page_leases: BTreeSet<LodPageId>,
    pending_fallback_nodes: BTreeSet<LodNodeId>,
    /// Exact generation-safe all-camera payload observed while render leases
    /// prevent unresolved demand from entering the bounded cache, or while a
    /// drained cold start is terminally capacity-blocked. Stable frames compare
    /// against this snapshot without cloning it; a new snapshot is allocated
    /// only when the physical cut actually changes.
    capacity_pressure_payload: Option<BTreeMap<Entity, CapacityPressureCandidatePayload>>,
    capacity_pressure_stable_frames: u8,
    /// Consecutive pressure duration independent from payload stability. This
    /// guarantees changing or non-relieving cuts still reach the emergency
    /// guard within a bounded number of frames.
    capacity_pressure_total_frames: u8,
    /// Exact effective views which selected the published cut. A dynamic cut
    /// is drawable only while this all-camera snapshot still matches.
    current_views: BTreeMap<Entity, LodView>,
    handshakes: BTreeMap<Entity, BridgeHandshake>,
    /// Exact selection snapshots captured on entry to frozen mode. These are
    /// also the views supplied to the runtime, so handshake provenance tracks
    /// the selector's actual frozen input rather than the moving live camera.
    frozen_selection_views: BTreeMap<Entity, LodView>,
    views: BTreeSet<Entity>,
    flat_source_bypass: bool,
    active: bool,
}

#[derive(Debug, Eq, PartialEq)]
struct CapacityPressureCandidatePayload {
    view: LodRuntimeViewId,
    candidate_count: u32,
    physical_ranges: Vec<LodPhysicalRange>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DebugAtlasSlotRecord {
    page: LodPageId,
    slot: AtlasSlot,
    fallback_nodes: Arc<[LodNodeId]>,
}

#[derive(Clone, Debug)]
struct BridgeHandshake {
    candidate: LodRenderCandidate,
    phase: Arc<AtomicU8>,
    staged: bool,
    /// Exact effective camera/cloud view used to select `candidate`.
    selected_view: LodView,
}

impl BridgeCloudState {
    fn owns_render_handle(&self, handle: AssetId<PlanarGaussian3d>) -> bool {
        handle == self.atlas.id()
            || ((self.flat_source_bypass || self.transient_atlas.is_some())
                && handle == self.source.id())
    }

    fn invalidate_handshakes(&self) {
        for handshake in self.handshakes.values() {
            handshake.phase.store(LOD_RENDER_WAITING, Ordering::Release);
        }
    }

    fn debug_page_is_current(
        &self,
        page: LodPageId,
        slot: AtlasSlot,
        fallback_nodes: &[LodNodeId],
    ) -> bool {
        if self.debug_atlas.is_none() {
            return true;
        }
        self.debug_slots
            .get(slot.index as usize)
            .and_then(Option::as_ref)
            .is_some_and(|record| {
                record.page == page
                    && record.slot == slot
                    && record.fallback_nodes.as_ref() == fallback_nodes
            })
    }

    fn handshake_for(
        &mut self,
        camera: Entity,
        frontier: &LodCandidateFrontier,
        selected_view: LodView,
    ) -> Arc<AtomicU8> {
        if let Some(handshake) = self.handshakes.get_mut(&camera)
            && bridge_candidate_matches_frontier(&handshake.candidate, frontier)
            && (handshake.selected_view == selected_view
                || handshake.phase.load(Ordering::Acquire) != LOD_RENDER_FAILED)
        {
            if frontier.candidate_count() == 0 && frontier.physical_ranges().is_empty() {
                // Complete empty cuts have no render pass that could restore a
                // cached phase after bridge invalidation. Re-publish the shared
                // constructor's zero-work capability on the existing token.
                handshake.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
            }
            // Identical physical coverage remains a valid capability under a
            // new camera. Refresh selector/status provenance without revoking
            // the token or rebuilding page leases; render compaction already
            // keys its output on the live camera uniforms.
            handshake.candidate =
                LodRenderCandidate::with_phase(frontier.clone(), Arc::clone(&handshake.phase));
            handshake.selected_view = selected_view;
            return Arc::clone(&handshake.phase);
        }
        if let Some(previous) = self.handshakes.remove(&camera) {
            // A committed token belongs to the retained GPU output until its
            // replacement is ACTIVE. Only revoke superseded pending work.
            let is_current = self.current.as_ref().is_some_and(|current| {
                current
                    .get(camera)
                    .is_some_and(|candidate| Arc::ptr_eq(&candidate.phase, &previous.phase))
            });
            if !is_current {
                previous.phase.store(LOD_RENDER_WAITING, Ordering::Release);
            }
        }
        let phase = Arc::new(AtomicU8::new(LOD_RENDER_WAITING));
        self.handshakes.insert(
            camera,
            BridgeHandshake {
                candidate: LodRenderCandidate::with_phase(frontier.clone(), Arc::clone(&phase)),
                phase: Arc::clone(&phase),
                staged: false,
                selected_view,
            },
        );
        phase
    }

    fn sync_debug_page(
        &mut self,
        page: &PlanarGaussian3dPage,
        slot: AtlasSlot,
        fallback_nodes: &[LodNodeId],
    ) -> Result<(), LodBridgeError> {
        debug_assert!(fallback_nodes.windows(2).all(|pair| pair[0] < pair[1]));
        if self.debug_atlas.is_none() || self.debug_manifest_index.is_none() {
            debug_assert!(self.debug_atlas.is_none() && self.debug_manifest_index.is_none());
            return Ok(());
        }
        if self.debug_page_is_current(page.id, slot, fallback_nodes) {
            return Ok(());
        }
        let index = slot.index as usize;
        let previous = self
            .debug_slots
            .get(index)
            .cloned()
            .ok_or(LodBridgeError::AtlasSlotOutOfRange(slot.index))?;
        let next = DebugAtlasSlotRecord {
            page: page.id,
            slot,
            fallback_nodes: Arc::from(fallback_nodes),
        };
        let debug_atlas = self.debug_atlas.as_mut().expect("checked above");
        let debug_manifest_index = self.debug_manifest_index.as_ref().expect("checked above");
        if let Some(previous) = previous.as_ref()
            && (previous.page != page.id || previous.slot != slot)
        {
            debug_atlas
                .clear_slot(previous.slot)
                .map_err(|error| LodBridgeError::DebugAnnotations(error.to_string()))?;
        }
        debug_atlas
            .write_page_indexed_with_node_residency(debug_manifest_index, page, slot, |node| {
                if fallback_nodes.binary_search(&node).is_ok() {
                    LodDebugResidency::AncestorFallback
                } else {
                    LodDebugResidency::Resident
                }
            })
            .map_err(|error| LodBridgeError::DebugAnnotations(error.to_string()))?;
        self.debug_slots[index] = Some(next);
        self.debug_revision = self.debug_revision.wrapping_add(1).max(1);
        Ok(())
    }

    fn stage_completed_page(
        &mut self,
        page_id: LodPageId,
        slot: AtlasSlot,
    ) -> Result<(), LodBridgeError> {
        #[cfg(test)]
        {
            self.decoded_page_acquisitions = self.decoded_page_acquisitions.saturating_add(1);
        }
        let page = self
            .runtime
            .decoded_page(page_id)
            .ok_or(LodBridgeError::ResidentPageNotDecoded(page_id))?;
        self.sync_debug_page(&page, slot, &[])?;
        Ok(self.mirror.stage_page(page_id, slot)?)
    }

    fn publish_debug_metadata(&mut self, entity: Entity, commands: &mut Commands) {
        let desired = if self.active && self.debug_atlas.is_some() {
            (true, self.debug_revision)
        } else {
            (false, 0)
        };
        if self.published_debug == Some(desired) {
            return;
        }
        let metadata = if self.active {
            self.debug_atlas
                .as_ref()
                .map(LodDebugAnnotationAtlas::metadata)
                .unwrap_or_else(|| self.fallback_debug_metadata.clone())
        } else {
            self.fallback_debug_metadata.clone()
        };
        commands.entity(entity).insert(metadata);
        self.published_debug = Some(desired);
    }

    fn restore_source_debug_metadata(self, entity: Entity, commands: &mut Commands) {
        let mut target = commands.entity(entity);
        if let Some(metadata) = self.source_debug_metadata {
            target.insert(metadata);
        } else {
            target.remove::<LodDebugMetadata>();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct EphemeralBridgeRequest {
    id: u64,
    source: AssetId<PlanarGaussian3d>,
    source_revision: u64,
    source_len: usize,
    signature: BridgeStructuralSignature,
}

#[cfg(not(target_arch = "wasm32"))]
struct EphemeralBridgeSnapshot {
    source_handle: Handle<PlanarGaussian3d>,
    settings: GaussianLodSettings,
    streaming: GaussianStreamingSettings,
    config: GaussianLodBridgeConfig,
    debug_metadata: bool,
    expected_len: usize,
    next_index: usize,
    records: Vec<Gaussian3d>,
}

#[cfg(not(target_arch = "wasm32"))]
struct EphemeralBridgeCompletion {
    request: EphemeralBridgeRequest,
    result: Result<(BridgeCloudState, PlanarGaussian3d), LodBridgeError>,
}

#[cfg(not(target_arch = "wasm32"))]
enum PendingEphemeralBridgePhase {
    Snapshot(Box<EphemeralBridgeSnapshot>),
    Building(Task<EphemeralBridgeCompletion>),
}

#[cfg(not(target_arch = "wasm32"))]
struct PendingEphemeralBridge {
    request: EphemeralBridgeRequest,
    canceled: Arc<AtomicBool>,
    phase: PendingEphemeralBridgePhase,
}

#[derive(Resource, Default)]
struct GaussianLodBridgeManager {
    clouds: HashMap<Entity, BridgeCloudState>,
    blocked: HashMap<Entity, (AssetId<PlanarGaussian3d>, BridgeStructuralSignature)>,
    #[cfg(not(target_arch = "wasm32"))]
    pending: HashMap<Entity, PendingEphemeralBridge>,
    #[cfg(not(target_arch = "wasm32"))]
    retired_tasks: Vec<Task<EphemeralBridgeCompletion>>,
    #[cfg(not(target_arch = "wasm32"))]
    source_revisions: HashMap<AssetId<PlanarGaussian3d>, u64>,
    #[cfg(not(target_arch = "wasm32"))]
    next_request_id: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl GaussianLodBridgeManager {
    fn next_ephemeral_request(
        &mut self,
        source: AssetId<PlanarGaussian3d>,
        source_len: usize,
        signature: BridgeStructuralSignature,
    ) -> EphemeralBridgeRequest {
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        EphemeralBridgeRequest {
            id: self.next_request_id,
            source,
            source_revision: self.source_revisions.get(&source).copied().unwrap_or(0),
            source_len,
            signature,
        }
    }

    fn cancel_ephemeral_request(&mut self, entity: Entity) {
        let Some(pending) = self.pending.remove(&entity) else {
            return;
        };
        pending.canceled.store(true, Ordering::Release);
        if let PendingEphemeralBridgePhase::Building(task) = pending.phase {
            // The builder is a synchronous CPU closure once its first worker
            // poll begins. Dropping its Task cannot preempt that closure, so
            // retain and poll it to completion before admitting another giant
            // job. This keeps cancellation memory/concurrency charged.
            self.retired_tasks.push(task);
        }
    }

    fn reap_retired_tasks(&mut self) {
        self.retired_tasks
            .retain_mut(|task| check_ready(task).is_none());
    }

    fn worker_busy(&self) -> bool {
        !self.retired_tasks.is_empty()
            || self
                .pending
                .values()
                .any(|pending| matches!(&pending.phase, PendingEphemeralBridgePhase::Building(_)))
    }

    fn bump_source_revision(&mut self, source: AssetId<PlanarGaussian3d>) {
        let revision = self.source_revisions.entry(source).or_default();
        *revision = revision.wrapping_add(1).max(1);
    }
}

#[derive(Clone, Copy)]
struct BridgeCameraView {
    entity: Entity,
    view: LodView,
}

struct BridgeCameraObservation {
    camera: BridgeCameraView,
    /// `None` is retained for lightweight unit worlds that do not install
    /// Bevy's visibility systems; production cameras always provide the set.
    visible_clouds: Option<HashSet<Entity>>,
}

type BridgeCameraQueryItem = (
    Entity,
    &'static Camera,
    &'static Projection,
    &'static GlobalTransform,
    Option<&'static VisibleEntities>,
);

fn flat_streaming_settings(
    per_cloud: Option<&GaussianStreamingSettings>,
    config: &GaussianLodBridgeConfig,
) -> Result<GaussianStreamingSettings, LodBridgeError> {
    let streaming = per_cloud.unwrap_or(&config.streaming_settings);
    streaming
        .validate()
        .map_err(|error| LodBridgeError::StreamingSettings(error.to_string()))?;
    Ok(streaming.clone())
}

fn validate_flat_lod_render_path(settings: &GaussianLodSettings) -> Result<(), LodBridgeError> {
    if settings.quality_endpoint() == LodQualityEndpoint::Original {
        return Ok(());
    }
    require_lod_render_path().map_err(LodBridgeError::UnsupportedRenderPath)
}

const PROGRESSIVE_EPHEMERAL_PHYSICAL_PAGE_CAPACITY: u32 = 1_024;

fn ephemeral_physical_page_capacity(build: GaussianLodBuildSettings) -> u32 {
    build
        .leaf_capacity
        .min(PROGRESSIVE_EPHEMERAL_PHYSICAL_PAGE_CAPACITY)
}

fn preflight_ephemeral_source(
    source: &PlanarGaussian3d,
    settings: &GaussianLodSettings,
    config: &GaussianLodBridgeConfig,
) -> Result<u32, LodBridgeError> {
    config.validate_structure()?;
    settings
        .validate()
        .map_err(|error| LodBridgeError::Build(error.to_string()))?;
    let source_count = u32::try_from(source.len()).map_err(|_| LodBridgeError::SourceTooLarge {
        actual: source.len() as u64,
        limit: u64::from(config.max_ephemeral_source_gaussians),
    })?;
    if source_count == 0 || source_count > config.max_ephemeral_source_gaussians {
        return Err(LodBridgeError::SourceTooLarge {
            actual: u64::from(source_count),
            limit: u64::from(config.max_ephemeral_source_gaussians),
        });
    }
    if u64::from(source_count) > config.max_ephemeral_stored_gaussians {
        return Err(LodBridgeError::StoredGaussianLimit {
            actual: u64::from(source_count),
            limit: config.max_ephemeral_stored_gaussians,
        });
    }
    if source.spherical_harmonic.len() != source.len()
        || source.rotation.len() != source.len()
        || source.scale_opacity.len() != source.len()
    {
        return Err(LodBridgeError::Build(
            "flat source planes have different lengths".to_owned(),
        ));
    }
    // The physical atlas is a bounded page cache, not a second copy of the
    // virtual source. Preflight therefore proves only that one maximum-sized
    // decoded page can be addressed. Source-wide capacity is deliberately not
    // required: roots and detail pages replace one another in these slots.
    let stride = ephemeral_physical_page_capacity(config.build_settings).min(source_count);
    let page_gpu_bytes = u64::from(stride)
        .checked_mul(gaussian_3d_gpu_bytes_per_record())
        .ok_or(LodBridgeError::AtlasSizeOverflow)?;
    if settings.budgets.max_resident_pages == 0
        || stride > config.max_atlas_gaussians
        || page_gpu_bytes > config.max_atlas_bytes
    {
        return Err(LodBridgeError::AtlasCannotFitPage {
            stride,
            max_gaussians: config.max_atlas_gaussians,
            max_bytes: config.max_atlas_bytes,
        });
    }
    Ok(source_count)
}

#[cfg(any(test, target_arch = "wasm32"))]
fn wasm_synchronous_ephemeral_source_is_supported(
    source_len: usize,
    config: &GaussianLodBridgeConfig,
) -> bool {
    source_len
        <= config
            .max_ephemeral_source_gaussians
            .min(WASM_SYNCHRONOUS_EPHEMERAL_SOURCE_LIMIT) as usize
}

#[cfg(not(target_arch = "wasm32"))]
fn snapshot_gaussian(source: &PlanarGaussian3d, index: usize) -> Gaussian3d {
    Gaussian3d {
        position_visibility: source.position_visibility[index],
        spherical_harmonic: source.spherical_harmonic[index],
        rotation: source.rotation[index],
        scale_opacity: source.scale_opacity[index],
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "sort_rayon"))]
fn transient_lod_worker_threads(available: usize) -> usize {
    (available / 4).clamp(1, 4)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "sort_rayon"))]
fn transient_lod_worker_pool() -> &'static rayon::ThreadPool {
    static TRANSIENT_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    TRANSIENT_POOL.get_or_init(|| {
        let available = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(transient_lod_worker_threads(available))
            .thread_name(|index| format!("lod-transient-{index}"))
            .build()
            .expect("bounded transient LoD worker pool should build")
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "sort_rayon"))]
fn build_ephemeral_lod_owned(
    source: Vec<Gaussian3d>,
    settings: GaussianLodBuildSettings,
    canceled: Option<&AtomicBool>,
) -> Result<Option<(PlanarGaussian3dLod, Vec<Gaussian3d>)>, LodBuildError> {
    let is_canceled = || canceled.is_some_and(|canceled| canceled.load(Ordering::Acquire));
    transient_lod_worker_pool()
        .install(|| build_planar_3d_lod_owned_cancelable(source, settings, &is_canceled))
}

#[cfg(any(target_arch = "wasm32", not(feature = "sort_rayon")))]
fn build_ephemeral_lod_owned(
    source: Vec<Gaussian3d>,
    settings: GaussianLodBuildSettings,
    canceled: Option<&AtomicBool>,
) -> Result<Option<(PlanarGaussian3dLod, Vec<Gaussian3d>)>, LodBuildError> {
    let is_canceled = || canceled.is_some_and(|canceled| canceled.load(Ordering::Acquire));
    build_planar_3d_lod_owned_cancelable(source, settings, &is_canceled)
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn advance_ephemeral_bridge_build(
    manager: &mut GaussianLodBridgeManager,
    entity: Entity,
    source_handle: Handle<PlanarGaussian3d>,
    source: &PlanarGaussian3d,
    settings: &GaussianLodSettings,
    streaming: &GaussianStreamingSettings,
    config: &GaussianLodBridgeConfig,
    debug_metadata: bool,
    signature: BridgeStructuralSignature,
) -> Option<Result<(BridgeCloudState, PlanarGaussian3d), LodBridgeError>> {
    if let Err(error) = preflight_ephemeral_source(source, settings, config) {
        manager.cancel_ephemeral_request(entity);
        return Some(Err(error));
    }

    let source_revision = manager
        .source_revisions
        .get(&source_handle.id())
        .copied()
        .unwrap_or(0);
    let matches_current = manager.pending.get(&entity).is_some_and(|pending| {
        pending.request.source == source_handle.id()
            && pending.request.source_revision == source_revision
            && pending.request.source_len == source.len()
            && pending.request.signature == signature
    });
    if !matches_current {
        manager.cancel_ephemeral_request(entity);
        if manager.worker_busy() || !manager.pending.is_empty() {
            return None;
        }
        let request = manager.next_ephemeral_request(source_handle.id(), source.len(), signature);
        let mut records = Vec::new();
        if let Err(error) = records.try_reserve_exact(source.len()) {
            return Some(Err(LodBridgeError::Build(format!(
                "could not reserve transient source snapshot: {error}"
            ))));
        }
        manager.pending.insert(
            entity,
            PendingEphemeralBridge {
                request,
                canceled: Arc::new(AtomicBool::new(false)),
                phase: PendingEphemeralBridgePhase::Snapshot(Box::new(EphemeralBridgeSnapshot {
                    source_handle,
                    settings: settings.clone(),
                    streaming: streaming.clone(),
                    config: config.clone(),
                    debug_metadata,
                    expected_len: source.len(),
                    next_index: 0,
                    records,
                })),
            },
        );
    }

    let worker_busy = manager.worker_busy();
    let mut snapshot_complete = false;
    let mut completed = None;
    if let Some(pending) = manager.pending.get_mut(&entity) {
        match &mut pending.phase {
            PendingEphemeralBridgePhase::Snapshot(snapshot) => {
                if worker_busy {
                    return None;
                }
                let end = snapshot
                    .next_index
                    .saturating_add(EPHEMERAL_SNAPSHOT_RECORDS_PER_UPDATE)
                    .min(snapshot.expected_len);
                snapshot.records.extend(
                    (snapshot.next_index..end).map(|index| snapshot_gaussian(source, index)),
                );
                snapshot.next_index = end;
                snapshot_complete = end == snapshot.expected_len && !worker_busy;
            }
            PendingEphemeralBridgePhase::Building(task) => {
                if let Some(completion) = check_ready(task) {
                    completed = Some((pending.request, completion));
                }
            }
        }
    }

    if let Some((current_request, completion)) = completed {
        manager.pending.remove(&entity);
        if completion.request == current_request {
            return Some(completion.result);
        }
    }

    if snapshot_complete {
        let pending = manager
            .pending
            .remove(&entity)
            .expect("completed snapshot remains pending");
        let PendingEphemeralBridgePhase::Snapshot(snapshot) = pending.phase else {
            unreachable!("only snapshots become workers")
        };
        let snapshot = *snapshot;
        let request = pending.request;
        let canceled = Arc::clone(&pending.canceled);
        let worker_canceled = Arc::clone(&canceled);
        let task = AsyncComputeTaskPool::get_or_init(TaskPool::new).spawn(async move {
            let result = create_ephemeral_bridge_owned_cancelable(
                snapshot.source_handle,
                snapshot.records,
                None,
                &snapshot.settings,
                &snapshot.streaming,
                &snapshot.config,
                snapshot.debug_metadata,
                Some(&worker_canceled),
            );
            EphemeralBridgeCompletion { request, result }
        });
        manager.pending.insert(
            entity,
            PendingEphemeralBridge {
                request,
                canceled,
                phase: PendingEphemeralBridgePhase::Building(task),
            },
        );
    }
    None
}

/// Installs automatic flat-cloud LoD orchestration and render extraction.
#[derive(Default)]
pub struct GaussianLodBridgePlugin;

/// Main-world point after which an ephemeral bridge's final render handle is
/// stable for extraction. Sort storage performs a second sizing pass after
/// this set because its ordinary Update pass necessarily runs before bridge
/// source/atlas swaps in PostUpdate.
#[derive(SystemSet, Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GaussianLodBridgeUpdate;

impl Plugin for GaussianLodBridgePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<GaussianLodAtlasUploadPlugin>() {
            app.add_plugins(GaussianLodAtlasUploadPlugin);
        }
        if !app.is_plugin_added::<GaussianLodRenderCommitPlugin>() {
            app.add_plugins(GaussianLodRenderCommitPlugin);
        }
        app.init_resource::<GaussianLodBridgeConfig>()
            .init_resource::<GaussianLodBridgeManager>()
            .add_systems(
                PostUpdate,
                update_gaussian_lod_bridges
                    .in_set(GaussianLodBridgeUpdate)
                    .after(AssetEventSystems)
                    .after(CameraUpdateSystems)
                    .after(VisibilitySystems::CheckVisibility)
                    .after(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                (
                    prepare_transient_sorted_entry_capacity,
                    publish_bridge_status_transitions,
                )
                    .after(GaussianLodBridgeUpdate),
            );
    }
}

fn sorted_entry_capacity_for_count(count: u32) -> usize {
    let count = usize::try_from(count).unwrap_or(usize::MAX);
    let side = (count as f64).sqrt().ceil() as usize;
    side.saturating_mul(side)
}

fn prepare_transient_sorted_entry_capacity(
    manager: Res<GaussianLodBridgeManager>,
    sorted_entries: Option<ResMut<Assets<SortedEntries>>>,
    clouds: Query<(Entity, &SortedEntriesHandle)>,
    cameras: Query<Entity, (With<Camera>, With<GaussianCamera>)>,
    #[cfg(feature = "buffer_texture")] mut images: ResMut<Assets<Image>>,
) {
    let Some(mut sorted_entries) = sorted_entries else {
        return;
    };
    let camera_count = cameras.iter().len();
    if camera_count == 0 {
        return;
    }
    for (entity, sorted_handle) in &clouds {
        let Some(state) = manager
            .clouds
            .get(&entity)
            .filter(|state| state.transient_atlas.is_some())
        else {
            continue;
        };
        let required = sorted_entry_capacity_for_count(state.mirror.physical_gaussians());
        let Some(current) = sorted_entries.get(&sorted_handle.0) else {
            continue;
        };
        if current.camera_count == camera_count && current.entry_count >= required {
            continue;
        }
        let retained = current.entry_count.max(required);
        let replacement = SortedEntries::new(
            camera_count,
            retained,
            #[cfg(feature = "buffer_texture")]
            &mut images,
        );
        let _ = sorted_entries.insert(sorted_handle.0.id(), replacement);
    }
}

fn publish_bridge_status_transitions(
    statuses: Query<(Entity, &GaussianLodBridgeStatus), Changed<GaussianLodBridgeStatus>>,
    mut removed: RemovedComponents<GaussianLodBridgeStatus>,
    mut previous: Local<
        HashMap<Entity, (GaussianLodBridgePhase, Option<LodOrchestrationFailureCode>)>,
    >,
    mut recovery_pending: Local<HashSet<Entity>>,
    mut transitions: MessageWriter<LodOrchestrationTransition>,
) {
    for entity in removed.read() {
        previous.remove(&entity);
        recovery_pending.remove(&entity);
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
        let had_failure = recovery_pending.contains(&entity);
        if status.failure.is_some() {
            recovery_pending.insert(entity);
        }
        let kind =
            bridge_status_transition_kind(status.phase, status.failure.is_some(), had_failure);
        if let Some(kind) = kind {
            transitions.write(LodOrchestrationTransition {
                entity,
                source: LodOrchestrationSource::EphemeralBridge,
                kind,
                failure: status.failure.clone(),
            });
        }
        if status.failure.is_none() && status.phase == GaussianLodBridgePhase::Active {
            recovery_pending.remove(&entity);
        }
    }
}

fn bridge_status_transition_kind(
    phase: GaussianLodBridgePhase,
    has_failure: bool,
    had_failure: bool,
) -> Option<LodOrchestrationTransitionKind> {
    if has_failure {
        Some(if phase == GaussianLodBridgePhase::Active {
            LodOrchestrationTransitionKind::Degraded
        } else {
            LodOrchestrationTransitionKind::Failed
        })
    } else if had_failure && phase == GaussianLodBridgePhase::Active {
        Some(LodOrchestrationTransitionKind::Recovered)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_gaussian_lod_bridges(
    mut commands: Commands,
    config: Res<GaussianLodBridgeConfig>,
    mut manager: ResMut<GaussianLodBridgeManager>,
    mut assets: ResMut<Assets<PlanarGaussian3d>>,
    mut atlas_uploads: ResMut<LodAtlasUploadQueue>,
    mut transient_atlases: Option<ResMut<LodTransientAtlasRegistry>>,
    mut asset_events: MessageReader<AssetEvent<PlanarGaussian3d>>,
    cameras: Query<BridgeCameraQueryItem, With<GaussianCamera>>,
    lodge_targets: Query<(), With<GaussianLodgeHandle>>,
    mut clouds: Query<
        (
            Entity,
            &mut PlanarGaussian3dHandle,
            Option<&GaussianLodSettings>,
            Option<&GaussianStreamingSettings>,
            Option<&CloudSettings>,
            Option<&LodDebugMetadata>,
            &GlobalTransform,
        ),
        (Without<GaussianLodHandle>, Without<GaussianLodgeHandle>),
    >,
) {
    let config_error = config.validate_structure().err();
    let camera_views = collect_camera_views(&cameras, config.max_views_per_cloud);
    let mut changed_assets = HashSet::new();
    let mut invalidated_assets = HashSet::new();
    let mut removed_assets = HashSet::new();
    for event in asset_events.read() {
        let changed = match event {
            AssetEvent::Added { id } => Some((*id, false, false)),
            AssetEvent::Modified { id } | AssetEvent::LoadedWithDependencies { id } => {
                Some((*id, true, false))
            }
            AssetEvent::Removed { id } => Some((*id, true, true)),
            AssetEvent::Unused { .. } => None,
        };
        let Some((id, invalidated, removed)) = changed else {
            continue;
        };
        changed_assets.insert(id);
        if invalidated {
            invalidated_assets.insert(id);
        }
        if removed {
            removed_assets.insert(id);
        }
        #[cfg(not(target_arch = "wasm32"))]
        manager.bump_source_revision(id);
    }
    #[cfg(not(target_arch = "wasm32"))]
    manager.reap_retired_tasks();
    if let Some(registry) = transient_atlases.as_deref_mut() {
        let _ = registry.queue_pending_initialization(&mut atlas_uploads);
    }
    let mut seen = BTreeSet::new();

    for (
        entity,
        mut handle,
        settings,
        per_cloud_streaming,
        cloud_settings,
        source_debug_metadata,
        cloud_transform,
    ) in &mut clouds
    {
        seen.insert(entity);
        #[cfg(not(target_arch = "wasm32"))]
        if manager.pending.get(&entity).is_some_and(|pending| {
            pending.request.source != handle.handle().id()
                || changed_assets.contains(&pending.request.source)
        }) {
            manager.cancel_ephemeral_request(entity);
        }
        let bridged_source_changed = manager.clouds.get(&entity).is_some_and(|state| {
            state.owns_render_handle(handle.handle().id())
                && changed_assets.contains(&state.source.id())
        });
        if bridged_source_changed {
            let source_was_removed = manager.clouds.get(&entity).is_some_and(|state| {
                removed_assets.contains(&state.source.id()) && assets.get(&state.source).is_none()
            });
            let retirement = if source_was_removed {
                deactivate_bridge_for_missing_source(
                    entity,
                    &mut handle,
                    &mut manager,
                    &mut assets,
                    &mut commands,
                );
                Ok(())
            } else {
                deactivate_bridge_for_source_change(
                    entity,
                    &mut handle,
                    &mut manager,
                    &mut assets,
                    &mut atlas_uploads,
                    &mut commands,
                )
            };
            match retirement {
                Ok(()) => {
                    commands.entity(entity).insert(GaussianLodBridgeStatus {
                        phase: GaussianLodBridgePhase::Building,
                        active_views: 0,
                        resident_pages: 0,
                        active_gaussians: 0,
                        failure: None,
                    });
                }
                Err(error) => {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                }
            }
            continue;
        }
        let bridge_atlas_changed = manager.clouds.get(&entity).is_some_and(|state| {
            handle.handle().id() == state.atlas.id()
                && invalidated_assets.contains(&state.atlas.id())
        });
        if bridge_atlas_changed {
            let retirement = deactivate_bridge(
                entity,
                &mut handle,
                &mut manager,
                &mut assets,
                &mut atlas_uploads,
                &mut commands,
            );
            match retirement {
                Ok(()) => {
                    commands.entity(entity).insert(GaussianLodBridgeStatus {
                        phase: GaussianLodBridgePhase::Building,
                        active_views: 0,
                        resident_pages: 0,
                        active_gaussians: 0,
                        failure: None,
                    });
                }
                Err(error) => {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                }
            }
            continue;
        }
        if manager
            .blocked
            .get(&entity)
            .is_some_and(|(source, _)| changed_assets.contains(source))
        {
            manager.blocked.remove(&entity);
        }
        let Some(settings) = settings else {
            #[cfg(not(target_arch = "wasm32"))]
            manager.cancel_ephemeral_request(entity);
            match deactivate_bridge(
                entity,
                &mut handle,
                &mut manager,
                &mut assets,
                &mut atlas_uploads,
                &mut commands,
            ) {
                Ok(()) => {
                    commands.entity(entity).remove::<GaussianLodBridgeStatus>();
                }
                Err(error) => {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                }
            }
            continue;
        };
        if let Err(error) = settings.validate() {
            #[cfg(not(target_arch = "wasm32"))]
            manager.cancel_ephemeral_request(entity);
            let retirement = deactivate_bridge(
                entity,
                &mut handle,
                &mut manager,
                &mut assets,
                &mut atlas_uploads,
                &mut commands,
            );
            // Live selection settings are intentionally absent from the
            // structural signature. Never sticky-block a validation failure:
            // correcting the component must retry on the next frame.
            manager.blocked.remove(&entity);
            commands
                .entity(entity)
                .insert(GaussianLodBridgeStatus::fallback(
                    retirement
                        .err()
                        .unwrap_or_else(|| LodBridgeError::Build(error.to_string())),
                ));
            continue;
        }
        let endpoint = settings.quality_endpoint();
        if endpoint == LodQualityEndpoint::Original {
            #[cfg(not(target_arch = "wasm32"))]
            manager.cancel_ephemeral_request(entity);
            let retirement = deactivate_bridge(
                entity,
                &mut handle,
                &mut manager,
                &mut assets,
                &mut atlas_uploads,
                &mut commands,
            );
            manager.blocked.remove(&entity);
            match retirement {
                Ok(()) => {
                    commands.entity(entity).remove::<GaussianLodBridgeStatus>();
                }
                Err(error) => {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                }
            }
            continue;
        }
        if !config.auto_build_flat_clouds {
            #[cfg(not(target_arch = "wasm32"))]
            manager.cancel_ephemeral_request(entity);
            let retirement = deactivate_bridge(
                entity,
                &mut handle,
                &mut manager,
                &mut assets,
                &mut atlas_uploads,
                &mut commands,
            );
            match retirement {
                Ok(()) => {
                    commands.entity(entity).remove::<GaussianLodBridgeStatus>();
                }
                Err(error) => {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                }
            }
            continue;
        }
        let streaming = match flat_streaming_settings(per_cloud_streaming, &config) {
            Ok(streaming) => streaming,
            Err(error) => {
                #[cfg(not(target_arch = "wasm32"))]
                manager.cancel_ephemeral_request(entity);
                let retirement = deactivate_bridge(
                    entity,
                    &mut handle,
                    &mut manager,
                    &mut assets,
                    &mut atlas_uploads,
                    &mut commands,
                );
                commands
                    .entity(entity)
                    .insert(GaussianLodBridgeStatus::fallback(
                        retirement.err().unwrap_or(error),
                    ));
                continue;
            }
        };
        if let Err(error) = validate_flat_lod_render_path(settings) {
            #[cfg(not(target_arch = "wasm32"))]
            manager.cancel_ephemeral_request(entity);
            let retirement = deactivate_bridge(
                entity,
                &mut handle,
                &mut manager,
                &mut assets,
                &mut atlas_uploads,
                &mut commands,
            );
            manager.blocked.remove(&entity);
            commands
                .entity(entity)
                .remove::<LodRenderCandidates>()
                .insert(GaussianLodBridgeStatus::fallback(
                    retirement.err().unwrap_or(error),
                ));
            continue;
        }
        if let Some(error) = &config_error {
            #[cfg(not(target_arch = "wasm32"))]
            manager.cancel_ephemeral_request(entity);
            let retirement = deactivate_bridge(
                entity,
                &mut handle,
                &mut manager,
                &mut assets,
                &mut atlas_uploads,
                &mut commands,
            );
            commands
                .entity(entity)
                .insert(GaussianLodBridgeStatus::fallback(
                    retirement.err().unwrap_or_else(|| error.clone()),
                ));
            continue;
        }
        let camera_views = match &camera_views {
            Ok(camera_views) => camera_views
                .iter()
                .filter(|observation| {
                    observation
                        .visible_clouds
                        .as_ref()
                        .is_none_or(|visible| visible.contains(&entity))
                })
                .map(|observation| observation.camera)
                .collect::<Vec<_>>(),
            Err(error) => {
                #[cfg(not(target_arch = "wasm32"))]
                manager.cancel_ephemeral_request(entity);
                let retirement = deactivate_bridge(
                    entity,
                    &mut handle,
                    &mut manager,
                    &mut assets,
                    &mut atlas_uploads,
                    &mut commands,
                );
                commands
                    .entity(entity)
                    .insert(GaussianLodBridgeStatus::fallback(
                        retirement.err().unwrap_or_else(|| error.clone()),
                    ));
                continue;
            }
        };
        let debug_metadata =
            cloud_settings.is_some_and(|settings| settings.lod_debug.requires_metadata());
        let signature =
            BridgeStructuralSignature::new(settings, &streaming, &config, debug_metadata);
        #[cfg(not(target_arch = "wasm32"))]
        if manager
            .pending
            .get(&entity)
            .is_some_and(|pending| pending.request.signature != signature)
        {
            manager.cancel_ephemeral_request(entity);
        }
        let structure_changed = manager.clouds.get(&entity).is_some_and(|state| {
            state.owns_render_handle(handle.handle().id()) && state.signature != signature
        });
        if structure_changed {
            let retirement = deactivate_bridge(
                entity,
                &mut handle,
                &mut manager,
                &mut assets,
                &mut atlas_uploads,
                &mut commands,
            );
            match retirement {
                Ok(()) => {
                    commands.entity(entity).insert(GaussianLodBridgeStatus {
                        phase: GaussianLodBridgePhase::Building,
                        active_views: 0,
                        resident_pages: 0,
                        active_gaussians: 0,
                        failure: None,
                    });
                }
                Err(error) => {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                }
            }
            continue;
        }
        if !manager.clouds.contains_key(&entity) {
            if manager
                .blocked
                .get(&entity)
                .is_some_and(|(source, blocked_signature)| {
                    *source == handle.handle().id() && *blocked_signature == signature
                })
            {
                continue;
            }
            manager.blocked.remove(&entity);
            let source_handle = handle.handle().clone();
            let Some(source) = assets.get(&source_handle) else {
                continue;
            };
            commands.entity(entity).insert(GaussianLodBridgeStatus {
                phase: GaussianLodBridgePhase::Building,
                active_views: 0,
                resident_pages: 0,
                active_gaussians: 0,
                failure: None,
            });
            #[cfg(not(target_arch = "wasm32"))]
            let build = advance_ephemeral_bridge_build(
                &mut manager,
                entity,
                source_handle.clone(),
                source,
                settings,
                &streaming,
                &config,
                debug_metadata,
                signature,
            );
            #[cfg(target_arch = "wasm32")]
            let build = if !wasm_synchronous_ephemeral_source_is_supported(source.len(), &config) {
                Some(Err(LodBridgeError::SourceTooLarge {
                    actual: source.len() as u64,
                    limit: u64::from(
                        config
                            .max_ephemeral_source_gaussians
                            .min(WASM_SYNCHRONOUS_EPHEMERAL_SOURCE_LIMIT),
                    ),
                }))
            } else {
                Some(create_ephemeral_bridge(
                    source_handle.clone(),
                    source,
                    source_debug_metadata.cloned(),
                    settings,
                    &streaming,
                    &config,
                    debug_metadata,
                ))
            };
            let Some(build) = build else {
                continue;
            };
            match build {
                Ok((mut state, atlas_cloud)) => {
                    // Debug metadata belongs to the live entity, not the
                    // potentially long-running hierarchy request. Capture it
                    // at publication so a component update during the build
                    // cannot be overwritten when the bridge later retires.
                    state.source_debug_metadata = source_debug_metadata.cloned();
                    state.fallback_debug_metadata =
                        source_debug_metadata.cloned().unwrap_or_default();
                    let publication = if let Some(registry) = transient_atlases.as_deref_mut() {
                        let atlas = assets.reserve_handle();
                        match LodTransientAtlas::new_empty(state.mirror.physical_gaussians()) {
                            Err(error) => Err(LodBridgeError::AtlasUpload(error.to_string())),
                            Ok(transient) => {
                                let registration = registry
                                    .register(
                                        atlas.id(),
                                        state.source.id(),
                                        state.source_gaussian_count,
                                        state.mirror.layout().gaussians_per_slot,
                                        &transient,
                                    )
                                    .and_then(|()| {
                                        registry.queue_pending_initialization(&mut atlas_uploads)
                                    })
                                    .map_err(|error| {
                                        LodBridgeError::AtlasUpload(error.to_string())
                                    });
                                if registration.is_ok() {
                                    state.transient_atlas_generation =
                                        Some(transient.ticket().generation());
                                    state.atlas = atlas;
                                    state.transient_atlas = Some(transient);
                                }
                                registration
                            }
                        }
                    } else {
                        let atlas_cloud = if atlas_cloud.is_empty() {
                            dense_bounded_atlas(state.mirror.physical_gaussians())
                        } else {
                            atlas_cloud
                        };
                        state.atlas = assets.add(atlas_cloud);
                        *handle = PlanarGaussian3dHandle(state.atlas.clone());
                        Ok(())
                    };
                    if let Err(error) = publication {
                        manager
                            .blocked
                            .insert(entity, (source_handle.id(), signature));
                        commands
                            .entity(entity)
                            .insert(GaussianLodBridgeStatus::fallback(error));
                        continue;
                    }
                    manager.blocked.remove(&entity);
                    manager.clouds.insert(entity, state);
                }
                Err(error) => {
                    manager
                        .blocked
                        .insert(entity, (source_handle.id(), signature));
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                    continue;
                }
            }
        }

        let Some(state) = manager.clouds.get_mut(&entity) else {
            continue;
        };
        if !state.owns_render_handle(handle.handle().id()) {
            // An application replaced the cloud while it was bridged. Drop the
            // old runtime; the new source is reconsidered next frame.
            let source = handle.handle().clone();
            if let Some(mut state) = manager.clouds.remove(&entity) {
                let retirement =
                    restore_bridge_before_discard(&mut state, &mut assets, &mut atlas_uploads);
                state.invalidate_handshakes();
                state.restore_source_debug_metadata(entity, &mut commands);
                if let Err(error) = retirement {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                }
            }
            manager.blocked.remove(&entity);
            commands.entity(entity).remove::<LodRenderCandidates>();
            *handle = PlanarGaussian3dHandle(source);
            continue;
        }

        if state
            .transient_atlas
            .as_ref()
            .is_some_and(|atlas| atlas.ticket().is_failed())
        {
            let source = state.source.clone();
            let blocked = (source.id(), state.signature);
            *handle = PlanarGaussian3dHandle(source);
            commands
                .entity(entity)
                .remove::<LodRenderCandidates>()
                .insert(GaussianLodBridgeStatus::fallback(
                    LodBridgeError::AtlasUpload(
                        "transient atlas GPU initialization failed".to_owned(),
                    ),
                ));
            if let Some(state) = manager.clouds.remove(&entity) {
                state.invalidate_handshakes();
                state.restore_source_debug_metadata(entity, &mut commands);
            }
            manager.blocked.insert(entity, blocked);
            continue;
        }
        if state
            .transient_atlas
            .as_ref()
            .is_some_and(|atlas| !atlas.ticket().is_ready())
        {
            if let Some(mut current) = state.current.clone() {
                if let Err(error) = clear_bridge_pending_transaction(state) {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                    continue;
                }
                for (&camera, candidate) in &current.by_camera {
                    candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
                    let selected_view = state
                        .current_views
                        .get(&camera)
                        .copied()
                        .expect("a published bridge cut retains its selected view");
                    state.handshakes.insert(
                        camera,
                        BridgeHandshake {
                            candidate: candidate.clone(),
                            phase: Arc::clone(&candidate.phase),
                            staged: false,
                            selected_view,
                        },
                    );
                }
                state.active = false;
                current.candidate_draw_required = true;
                let active_gaussians = current
                    .by_camera
                    .values()
                    .map(|candidate| u64::from(candidate.rendered_candidate_count()))
                    .max()
                    .unwrap_or(0);
                *handle = PlanarGaussian3dHandle(state.atlas.clone());
                commands.entity(entity).insert((
                    current,
                    GaussianLodBridgeStatus {
                        phase: GaussianLodBridgePhase::WaitingForRender,
                        active_views: state
                            .current
                            .as_ref()
                            .map_or(0, |current| current.len().try_into().unwrap_or(u32::MAX)),
                        resident_pages: state
                            .current_page_leases
                            .len()
                            .try_into()
                            .unwrap_or(u32::MAX),
                        active_gaussians,
                        failure: None,
                    },
                ));
            } else {
                *handle = PlanarGaussian3dHandle(state.source.clone());
                commands
                    .entity(entity)
                    .remove::<LodRenderCandidates>()
                    .insert(GaussianLodBridgeStatus {
                        phase: GaussianLodBridgePhase::Building,
                        active_views: 0,
                        resident_pages: 0,
                        active_gaussians: state.source_gaussian_count.into(),
                        failure: None,
                    });
            }
            continue;
        }

        if let Some(generation) = state
            .transient_atlas
            .as_ref()
            .map(|atlas| atlas.ticket().generation())
            && state.transient_atlas_generation != Some(generation)
        {
            // A recreated bounded GPU atlas contains no valid page payloads.
            // Requeue the exact retained cut before it can regain ACTIVE;
            // generation proofs fail closed until every referenced slot lands.
            if let Some(current) = state.current.clone() {
                if let Err(error) =
                    requeue_bridge_candidate_pages(state, &current, &mut atlas_uploads)
                {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                    continue;
                }
                for (&camera, candidate) in &current.by_camera {
                    candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
                    let selected_view = state
                        .current_views
                        .get(&camera)
                        .copied()
                        .expect("a published bridge cut retains its selected view");
                    state.handshakes.insert(
                        camera,
                        BridgeHandshake {
                            candidate: candidate.clone(),
                            phase: Arc::clone(&candidate.phase),
                            staged: true,
                            selected_view,
                        },
                    );
                }
                state.active = false;
            } else {
                // A cold candidate may already own pending leases and be
                // marked staged even though no cut has reached ACTIVE. Atlas
                // recreation invalidates those GPU proofs too; revoke the
                // transaction and its CPU mirror proofs so the ordinary cold
                // path must rematerialize and requeue every staged page for the
                // new allocation generation.
                if let Err(error) = clear_bridge_pending_transaction(state) {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(error));
                    continue;
                }
                let materialized_slots = state.mirror.materialized_slots();
                if let Err(error) = materialized_slots
                    .into_iter()
                    .try_for_each(|slot| state.mirror.clear_materialized_slot(slot.index))
                {
                    commands
                        .entity(entity)
                        .insert(GaussianLodBridgeStatus::fallback(LodBridgeError::from(
                            error,
                        )));
                    continue;
                }
                // PREPARED lowers the cold source bypass before the first
                // ACTIVE cut exists. If that GPU allocation is then recreated,
                // restore the immutable source until the replacement atlas is
                // proved PREPARED again; an empty/reuploading page cache is not
                // a drawable fallback.
                state.flat_source_bypass = true;
                state.active = false;
            }
            state.transient_atlas_generation = Some(generation);
        }

        state.streaming = streaming;
        let effective_settings = state.structural.apply(settings);
        let result = update_bridge_cloud(
            state,
            &effective_settings,
            cloud_transform,
            &camera_views,
            &mut assets,
            &mut atlas_uploads,
        );
        match result {
            Ok((candidates, status)) => {
                *handle = PlanarGaussian3dHandle(if state.flat_source_bypass {
                    state.source.clone()
                } else {
                    state.atlas.clone()
                });
                if candidates.by_camera.is_empty() && !candidates.candidate_draw_required {
                    commands.entity(entity).remove::<LodRenderCandidates>();
                } else {
                    commands.entity(entity).insert(candidates);
                }
                commands.entity(entity).insert(status);
                state.publish_debug_metadata(entity, &mut commands);
            }
            Err(error) => {
                error!(
                    ?entity,
                    atlas = ?state.atlas.id(),
                    source = ?state.source.id(),
                    %error,
                    "LoD bridge update failed; retiring the bounded atlas and restoring its source"
                );
                let retirement =
                    restore_bridge_before_discard(state, &mut assets, &mut atlas_uploads);
                *handle = PlanarGaussian3dHandle(state.source.clone());
                commands
                    .entity(entity)
                    .remove::<LodRenderCandidates>()
                    .insert(GaussianLodBridgeStatus::fallback(
                        retirement.err().unwrap_or(error),
                    ));
                if let Some(state) = manager.clouds.remove(&entity) {
                    state.invalidate_handshakes();
                    state.restore_source_debug_metadata(entity, &mut commands);
                }
            }
        }
    }

    let stale = manager
        .clouds
        .keys()
        .filter(|entity| !seen.contains(entity))
        .copied()
        .collect::<Vec<_>>();
    for entity in stale {
        if let Some(mut state) = manager.clouds.remove(&entity) {
            let _ = restore_bridge_before_discard(&mut state, &mut assets, &mut atlas_uploads);
            state.invalidate_handshakes();
            // Adding a LODGE handle excludes this cloud from the mutable
            // hierarchy query before the resident catalog is necessarily
            // ready. Restore the immutable predecessor now so a canceled or
            // failed strategy switch can never strand the entity on the
            // retired transient atlas.
            if lodge_targets.contains(entity) && commands.get_entity(entity).is_ok() {
                commands
                    .entity(entity)
                    .insert(PlanarGaussian3dHandle(state.source.clone()))
                    .remove::<LodRenderCandidates>()
                    .remove::<GaussianLodBridgeStatus>();
                state.restore_source_debug_metadata(entity, &mut commands);
            }
        }
        manager.blocked.remove(&entity);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let stale_pending = manager
            .pending
            .keys()
            .filter(|entity| !seen.contains(entity))
            .copied()
            .collect::<Vec<_>>();
        for entity in stale_pending {
            manager.cancel_ephemeral_request(entity);
        }
    }
    manager.blocked.retain(|entity, _| seen.contains(entity));
    #[cfg(not(target_arch = "wasm32"))]
    for source in removed_assets {
        manager.source_revisions.remove(&source);
    }
}

fn deactivate_bridge(
    entity: Entity,
    handle: &mut PlanarGaussian3dHandle,
    manager: &mut GaussianLodBridgeManager,
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
    commands: &mut Commands,
) -> Result<(), LodBridgeError> {
    let mut retirement_error = None;
    if let Some(mut state) = manager.clouds.remove(&entity) {
        if let Err(error) = restore_bridge_before_discard(&mut state, assets, atlas_uploads) {
            retirement_error = Some(error);
        }
        state.invalidate_handshakes();
        *handle = PlanarGaussian3dHandle(state.source.clone());
        state.restore_source_debug_metadata(entity, commands);
    }
    manager.blocked.remove(&entity);
    commands.entity(entity).remove::<LodRenderCandidates>();
    retirement_error.map_or(Ok(()), Err)
}

fn deactivate_bridge_for_source_change(
    entity: Entity,
    handle: &mut PlanarGaussian3dHandle,
    manager: &mut GaussianLodBridgeManager,
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
    commands: &mut Commands,
) -> Result<(), LodBridgeError> {
    let mut retirement_error = None;
    if let Some(mut state) = manager.clouds.remove(&entity) {
        let retirement = restore_bridge_before_discard(&mut state, assets, atlas_uploads);
        if let Err(error) = retirement {
            retirement_error = Some(error);
        }
        state.invalidate_handshakes();
        *handle = PlanarGaussian3dHandle(state.source.clone());
        state.restore_source_debug_metadata(entity, commands);
    }
    manager.blocked.remove(&entity);
    commands.entity(entity).remove::<LodRenderCandidates>();
    retirement_error.map_or(Ok(()), Err)
}

fn deactivate_bridge_for_missing_source(
    entity: Entity,
    handle: &mut PlanarGaussian3dHandle,
    manager: &mut GaussianLodBridgeManager,
    assets: &mut Assets<PlanarGaussian3d>,
    commands: &mut Commands,
) {
    if let Some(state) = manager.clouds.remove(&entity) {
        state.invalidate_handshakes();
        *handle = PlanarGaussian3dHandle(state.source.clone());
        assets.remove(state.atlas.id());
        state.restore_source_debug_metadata(entity, commands);
    }
    manager.blocked.remove(&entity);
    commands.entity(entity).remove::<LodRenderCandidates>();
}

fn collect_camera_views(
    cameras: &Query<BridgeCameraQueryItem, With<GaussianCamera>>,
    max_views: u32,
) -> Result<Vec<BridgeCameraObservation>, LodBridgeError> {
    let active = cameras
        .iter()
        .filter(|(_, camera, _, _, _)| camera.is_active)
        .collect::<Vec<_>>();
    if active.len() > max_views as usize {
        return Err(LodBridgeError::ViewLimitExceeded {
            actual: active.len() as u64,
            limit: max_views,
        });
    }
    let mut views = Vec::with_capacity(active.len());
    for (entity, camera, projection, transform, visible_entities) in active {
        let view = lod_view_from_camera(camera, projection, transform)
            .ok_or(LodBridgeError::UnsupportedCamera(entity))?;
        let visible_clouds = visible_entities.map(|visible| {
            visible
                .iter(TypeId::of::<CloudVisibilityClass>())
                .copied()
                .collect()
        });
        views.push(BridgeCameraObservation {
            camera: BridgeCameraView { entity, view },
            visible_clouds,
        });
    }
    views.sort_by_key(|observation| observation.camera.entity);
    Ok(views)
}

fn lod_view_from_camera(
    camera: &Camera,
    projection: &Projection,
    transform: &GlobalTransform,
) -> Option<LodView> {
    let viewport_height = camera.physical_viewport_size()?.y as f32;
    if viewport_height <= 0.0 {
        return None;
    }
    let world_from_view = transform.to_matrix();
    let clip_from_world = projection.get_clip_from_view() * world_from_view.inverse();
    let camera_position = transform.translation();
    let view = match projection {
        Projection::Perspective(perspective) => LodView::perspective(
            camera_position,
            viewport_height,
            perspective.fov,
            perspective.near.max(f32::EPSILON),
        ),
        Projection::Orthographic(orthographic) => LodView::orthographic(
            camera_position,
            viewport_height,
            (orthographic.area.max.y - orthographic.area.min.y)
                .abs()
                .max(f32::EPSILON),
            orthographic.near.abs().max(f32::EPSILON),
        ),
        Projection::Custom(_) => return None,
    };
    Some(view.with_clip_from_world(clip_from_world))
}

#[cfg(any(test, target_arch = "wasm32"))]
fn create_ephemeral_bridge(
    source_handle: Handle<PlanarGaussian3d>,
    source: &PlanarGaussian3d,
    source_debug_metadata: Option<LodDebugMetadata>,
    settings: &GaussianLodSettings,
    streaming: &GaussianStreamingSettings,
    config: &GaussianLodBridgeConfig,
    debug_metadata: bool,
) -> Result<(BridgeCloudState, PlanarGaussian3d), LodBridgeError> {
    preflight_ephemeral_source(source, settings, config)?;
    let (state, atlas) = create_ephemeral_bridge_owned(
        source_handle,
        source.iter().collect(),
        source_debug_metadata,
        settings,
        streaming,
        config,
        debug_metadata,
    )?;
    // Direct/unit and the deliberately tiny synchronous Wasm path use the
    // ordinary Assets fallback. Native transient publication stays sparse and
    // never calls this helper after the worker returns.
    let atlas = if atlas.is_empty() {
        dense_bounded_atlas(state.mirror.physical_gaussians())
    } else {
        atlas
    };
    Ok((state, atlas))
}

fn dense_bounded_atlas(physical_gaussians: u32) -> PlanarGaussian3d {
    PlanarGaussian3d::from(vec![Gaussian3d::default(); physical_gaussians as usize])
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(test, target_arch = "wasm32"))]
fn create_ephemeral_bridge_owned(
    source_handle: Handle<PlanarGaussian3d>,
    source: Vec<Gaussian3d>,
    source_debug_metadata: Option<LodDebugMetadata>,
    settings: &GaussianLodSettings,
    streaming: &GaussianStreamingSettings,
    config: &GaussianLodBridgeConfig,
    debug_metadata: bool,
) -> Result<(BridgeCloudState, PlanarGaussian3d), LodBridgeError> {
    create_ephemeral_bridge_owned_cancelable(
        source_handle,
        source,
        source_debug_metadata,
        settings,
        streaming,
        config,
        debug_metadata,
        None,
    )
}

const EPHEMERAL_ENCODE_PAGES_PER_CHUNK: usize = 16;

#[cfg(all(not(target_arch = "wasm32"), feature = "sort_rayon"))]
fn encode_ephemeral_page_chunk(
    pages: &[PlanarGaussian3dPage],
) -> Result<Vec<Vec<u8>>, LodBridgeError> {
    transient_lod_worker_pool().install(|| {
        pages
            .par_iter()
            .map(|page| encode_page(page).map_err(|error| LodBridgeError::Codec(error.to_string())))
            .collect()
    })
}

#[cfg(any(target_arch = "wasm32", not(feature = "sort_rayon")))]
fn encode_ephemeral_page_chunk(
    pages: &[PlanarGaussian3dPage],
) -> Result<Vec<Vec<u8>>, LodBridgeError> {
    pages
        .iter()
        .map(|page| encode_page(page).map_err(|error| LodBridgeError::Codec(error.to_string())))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn create_ephemeral_bridge_owned_cancelable(
    source_handle: Handle<PlanarGaussian3d>,
    source: Vec<Gaussian3d>,
    source_debug_metadata: Option<LodDebugMetadata>,
    settings: &GaussianLodSettings,
    streaming: &GaussianStreamingSettings,
    config: &GaussianLodBridgeConfig,
    debug_metadata: bool,
    canceled: Option<&AtomicBool>,
) -> Result<(BridgeCloudState, PlanarGaussian3d), LodBridgeError> {
    let ensure_current = || {
        if canceled.is_some_and(|canceled| canceled.load(Ordering::Acquire)) {
            Err(LodBridgeError::Build(
                "transient hierarchy request was canceled".to_owned(),
            ))
        } else {
            Ok(())
        }
    };
    ensure_current()?;
    let source_count = u32::try_from(source.len()).map_err(|_| LodBridgeError::SourceTooLarge {
        actual: source.len() as u64,
        limit: u64::from(config.max_ephemeral_source_gaussians),
    })?;
    if source_count == 0 || source_count > config.max_ephemeral_source_gaussians {
        return Err(LodBridgeError::SourceTooLarge {
            actual: u64::from(source_count),
            limit: u64::from(config.max_ephemeral_source_gaussians),
        });
    }
    let Some((mut built, source)) =
        build_ephemeral_lod_owned(source, config.build_settings, canceled)
            .map_err(|error| LodBridgeError::Build(error.to_string()))?
    else {
        return Err(LodBridgeError::Build(
            "transient hierarchy request was canceled".to_owned(),
        ));
    };
    ensure_current()?;
    if built.manifest.header.stored_gaussian_count > config.max_ephemeral_stored_gaussians {
        return Err(LodBridgeError::StoredGaussianLimit {
            actual: built.manifest.header.stored_gaussian_count,
            limit: config.max_ephemeral_stored_gaussians,
        });
    }
    // The immutable source asset is not part of the page-cache atlas. Releasing
    // the canonical builder input before page encoding materially lowers peak
    // host memory for large transient conversions.
    drop(source);

    let mut transport = MemoryPageTransport::default();
    if built.pages.len() != built.manifest.pages.len() {
        return Err(LodBridgeError::Build(format!(
            "builder returned {} payloads for {} page descriptors",
            built.pages.len(),
            built.manifest.pages.len()
        )));
    }
    let mut page_index = 0_usize;
    let mut pages = std::mem::take(&mut built.pages).into_iter();
    loop {
        ensure_current()?;
        let chunk = pages
            .by_ref()
            .take(EPHEMERAL_ENCODE_PAGES_PER_CHUNK)
            .collect::<Vec<_>>();
        if chunk.is_empty() {
            break;
        }
        let encoded = encode_ephemeral_page_chunk(&chunk)?;
        ensure_current()?;
        for (page, encoded) in chunk.into_iter().zip(encoded) {
            let descriptor = built
                .manifest
                .pages
                .get_mut(page_index)
                .ok_or(LodBridgeError::MissingPageDescriptor(page.id))?;
            if descriptor.id != page.id {
                return Err(LodBridgeError::MissingPageDescriptor(page.id));
            }
            descriptor.storage = Some(
                crate::gaussian::formats::planar_3d_chunked::LodPageStorage {
                    uri: format!("memory://ephemeral/{}", page.id.0),
                    byte_range: None,
                    encoded_len: encoded.len() as u64,
                },
            );
            transport.insert(page.id, encoded);
            page_index += 1;
        }
    }
    debug_assert_eq!(page_index, built.manifest.pages.len());

    let stride = built
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .ok_or(LodBridgeError::Build("manifest has no pages".to_owned()))?;
    let canonical_bytes_per_slot = u64::from(stride)
        .checked_mul(size_of::<Gaussian3d>() as u64)
        .ok_or(LodBridgeError::AtlasSizeOverflow)?;
    let gpu_bytes_per_slot = u64::from(stride)
        .checked_mul(gaussian_3d_gpu_bytes_per_record())
        .ok_or(LodBridgeError::AtlasSizeOverflow)?;
    // One slot per unique manifest page is sufficient for every possible
    // camera cut: no selection can reference a page outside this manifest.
    // Capacity beyond that bound can never carry useful data. Crucially this
    // is independent of the virtual source count.
    let maximum_useful_slots = built.manifest.header.page_count;
    let slots_by_count = config.max_atlas_gaussians / stride;
    let slots_by_bytes = (config.max_atlas_bytes / gpu_bytes_per_slot)
        .try_into()
        .unwrap_or(u32::MAX);
    let slot_count = settings
        .budgets
        .max_resident_pages
        .min(slots_by_count)
        .min(slots_by_bytes)
        .min(maximum_useful_slots);
    if slot_count == 0 {
        return Err(LodBridgeError::AtlasCannotFitPage {
            stride,
            max_gaussians: config.max_atlas_gaussians,
            max_bytes: config.max_atlas_bytes,
        });
    }
    let physical_gaussians = slot_count
        .checked_mul(stride)
        .ok_or(LodBridgeError::AtlasSizeOverflow)?;
    let canonical_physical_bytes = u64::from(slot_count)
        .checked_mul(canonical_bytes_per_slot)
        .ok_or(LodBridgeError::AtlasSizeOverflow)?;
    let physical_gpu_bytes = u64::from(slot_count)
        .checked_mul(gpu_bytes_per_slot)
        .ok_or(LodBridgeError::AtlasSizeOverflow)?;

    let mut effective = settings.clone();
    effective.budgets.max_resident_pages = slot_count;
    effective.budgets.max_resident_gaussians = effective
        .budgets
        .max_resident_gaussians
        .min(u64::from(physical_gaussians));
    effective.budgets.max_resident_bytes = effective
        .budgets
        .max_resident_bytes
        .min(canonical_physical_bytes);
    let structural = StructuralSettings {
        max_resident_gaussians: effective.budgets.max_resident_gaussians,
        max_resident_bytes: effective.budgets.max_resident_bytes,
        max_resident_pages: effective.budgets.max_resident_pages,
        max_pending_requests: effective.budgets.max_pending_requests,
    };
    ensure_current()?;
    let debug_manifest_index = debug_metadata
        .then(|| LodDebugManifestIndex::new(&built.manifest))
        .transpose()
        .map_err(|error| LodBridgeError::DebugAnnotations(error.to_string()))?;
    ensure_current()?;
    let runtime = LodStreamingRuntime::new(built.manifest, transport, &effective, streaming)
        .map_err(LodBridgeError::Runtime)?;
    let mirror = LodPageAtlasMirror::new(runtime.atlas_layout(), slot_count)?;
    let debug_atlas = debug_metadata
        .then(|| LodDebugAnnotationAtlas::new(slot_count, stride))
        .transpose()
        .map_err(|error| LodBridgeError::DebugAnnotations(error.to_string()))?;
    let fallback_debug_metadata = source_debug_metadata.clone().unwrap_or_default();

    ensure_current()?;
    // Native transient publication owns sparse per-slot CPU payloads. Returning
    // an empty seed here avoids zeroing the entire physical page-cache capacity
    // on the worker; the legacy direct/Assets path densifies explicitly.
    let fallback = PlanarGaussian3d::default();
    ensure_current()?;
    debug_assert!(physical_gpu_bytes <= config.max_atlas_bytes);
    debug_assert!(slot_count <= maximum_useful_slots);
    let completed = (
        BridgeCloudState {
            source: source_handle,
            source_gaussian_count: source_count,
            atlas: Handle::default(),
            transient_atlas: None,
            transient_atlas_generation: None,
            runtime: Box::new(runtime),
            mirror,
            debug_atlas,
            debug_manifest_index,
            debug_slots: if debug_metadata {
                vec![None; slot_count as usize]
            } else {
                Vec::new()
            },
            fallback_debug_metadata,
            source_debug_metadata,
            debug_revision: 0,
            published_debug: None,
            #[cfg(test)]
            decoded_page_acquisitions: 0,
            #[cfg(test)]
            pre_frame_pending_lease_acquisitions: 0,
            #[cfg(test)]
            pre_frame_staged_replacement_retentions: 0,
            #[cfg(test)]
            deferred_ordinary_publications: 0,
            structural,
            signature: BridgeStructuralSignature::new(settings, streaming, config, debug_metadata),
            streaming: streaming.clone(),
            current: None,
            current_page_leases: BTreeSet::new(),
            pending_page_leases: BTreeSet::new(),
            pending_fallback_nodes: BTreeSet::new(),
            capacity_pressure_payload: None,
            capacity_pressure_stable_frames: 0,
            capacity_pressure_total_frames: 0,
            current_views: BTreeMap::new(),
            handshakes: BTreeMap::new(),
            frozen_selection_views: BTreeMap::new(),
            views: BTreeSet::new(),
            // Cold transient conversion keeps the already-loaded source bound
            // until a quiescent bounded cut is ready to publish. It is never
            // used again for motion, residency, or active-budget escape.
            flat_source_bypass: true,
            active: false,
        },
        fallback,
    );
    // If invalidation raced the final bounded atlas conversion, destroy the
    // stale result on this worker instead of publishing it on the main thread.
    ensure_current()?;
    Ok(completed)
}

fn bridge_candidate_set_is_render_active(candidates: &LodRenderCandidates) -> bool {
    !candidates.is_empty()
        && candidates
            .by_camera
            .values()
            .all(LodRenderCandidate::render_is_active)
}

/// True only when every camera's permanent coverage guard actually satisfies
/// the live presentation target. Completeness and residency make a guard safe
/// to address, but do not make a very coarse representative visually useful.
fn coverage_guard_frontiers_satisfy_requested_quality(
    frontiers: &[(Entity, LodCandidateFrontier)],
) -> bool {
    !frontiers.is_empty()
        && frontiers.iter().all(|(_, frontier)| {
            let quality = frontier.quality_status();
            frontier.is_coverage_guard()
                && quality.degradation == LodDegradation::None
                && quality.achieved_max_target_ratio.is_finite()
                && quality.achieved_max_target_ratio <= 1.0
        })
}

/// The exact cold source or an all-camera ACTIVE atlas cut is a known drawable
/// presentation capability. A degraded emergency guard must not displace it.
fn bridge_has_valid_drawable_fallback(state: &BridgeCloudState) -> bool {
    state.flat_source_bypass
        || state
            .current
            .as_ref()
            .is_some_and(bridge_candidate_set_is_render_active)
}

/// Commits a render-published transaction against the camera membership which
/// selected it before a newly observed membership can invalidate its token.
///
/// Render may publish ACTIVE after the preceding main-world update. An added
/// or removed camera in this update changes the next all-camera transaction,
/// but cannot roll back a complete old-membership cut which has already been
/// drawn. Its page generations are validated and leased before any runtime
/// view is removed, then the ordinary membership path can safely retarget.
fn commit_active_bridge_transaction_before_membership_change(
    state: &mut BridgeCloudState,
) -> Result<bool, LodBridgeError> {
    let complete_old_membership = !state.views.is_empty()
        && state.handshakes.len() == state.views.len()
        && state
            .views
            .iter()
            .all(|camera| state.handshakes.contains_key(camera));
    let transaction_is_current = state.current.as_ref().is_some_and(|current| {
        bridge_handshakes_match_candidates(&state.handshakes, current)
            && bridge_handshakes_match_views(&state.handshakes, &state.current_views)
    });
    let every_active = complete_old_membership
        && !transaction_is_current
        && state
            .handshakes
            .values()
            .all(|handshake| handshake.candidate.render_is_active());
    if !every_active {
        return Ok(false);
    }

    let ranges_are_current = state
        .handshakes
        .values()
        .flat_map(|handshake| handshake.candidate.render_ranges())
        .all(|range| state.runtime.resident_slot(range.page) == Some(range.slot));
    if !ranges_are_current {
        // An ACTIVE bit without its recorded page generations is no longer a
        // render capability. The normal membership invalidation below retains
        // the last valid current cut and selects a fresh transaction.
        return Ok(false);
    }

    let candidates = bridge_handshake_candidates(state);
    let pages = bridge_candidate_pages(&candidates);
    replace_bridge_pending_page_leases(state, &pages)?;
    commit_bridge_page_leases(state, &pages)?;
    state.current_views = state
        .handshakes
        .iter()
        .map(|(&camera, handshake)| (camera, handshake.selected_view))
        .collect();
    debug_assert_eq!(state.current_views.len(), candidates.len());
    state.current = Some(candidates);
    state.flat_source_bypass = false;
    state.active = true;
    clear_capacity_pressure_tracking(state);
    Ok(true)
}

fn update_bridge_cloud(
    state: &mut BridgeCloudState,
    settings: &GaussianLodSettings,
    cloud_transform: &GlobalTransform,
    camera_views: &[BridgeCameraView],
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(LodRenderCandidates, GaussianLodBridgeStatus), LodBridgeError> {
    let world_from_local = cloud_transform.to_matrix();
    let effective_views = camera_views
        .iter()
        .map(|camera| {
            (
                camera.entity,
                camera.view.with_world_from_local(world_from_local),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let next_views = camera_views
        .iter()
        .map(|view| view.entity)
        .collect::<BTreeSet<_>>();
    let view_set_changed = state.views != next_views;
    if view_set_changed {
        commit_active_bridge_transaction_before_membership_change(state)?;
    }
    let dynamic_selection = settings.selection_mode == LodSelectionMode::Dynamic;
    let selection_views = if dynamic_selection {
        state.frozen_selection_views.clear();
        effective_views.clone()
    } else {
        state
            .frozen_selection_views
            .retain(|camera, _| effective_views.contains_key(camera));
        let mut frozen_views = BTreeMap::new();
        for (&camera, &effective_view) in &effective_views {
            let selected_view = *state
                .frozen_selection_views
                .entry(camera)
                .or_insert(effective_view);
            frozen_views.insert(camera, selected_view);
        }
        frozen_views
    };
    // Every runtime frontier is a globally covering antichain. A camera change
    // can make its quality stale, but can never make it spatially incomplete.
    // Keep the last ACTIVE bounded cut drawable while a live-view replacement
    // streams; render compaction re-evaluates that cut against the new camera.

    for removed in state
        .views
        .difference(&next_views)
        .copied()
        .collect::<Vec<_>>()
    {
        state
            .runtime
            .remove_view(runtime_view_id(removed))
            .map_err(LodBridgeError::Runtime)?;
        if let Some(handshake) = state.handshakes.remove(&removed) {
            let is_current = state.current.as_ref().is_some_and(|current| {
                current
                    .get(removed)
                    .is_some_and(|candidate| Arc::ptr_eq(&candidate.phase, &handshake.phase))
            });
            if !is_current {
                handshake.phase.store(LOD_RENDER_WAITING, Ordering::Release);
            }
        }
    }
    state.views = next_views;
    if view_set_changed {
        // A pending transaction is all-camera atomic. Camera membership
        // changes invalidate that transaction, but never the published cut or
        // its leases.
        clear_bridge_pending_transaction(state)?;
    }

    // An incomplete materialization step or render PREPARED publication may
    // survive the preceding main-world update. Pin that complete all-camera
    // payload before the first sequential view update swaps its runtime pins:
    // a later camera may commit preprocessing and evict pages released by an
    // earlier camera in this same frame.
    let pre_frame_handshake_complete = state.handshakes.len() == camera_views.len()
        && camera_views
            .iter()
            .all(|camera| state.handshakes.contains_key(&camera.entity));
    let pre_frame_handshake_is_current = state.current.as_ref().is_some_and(|current| {
        bridge_handshakes_match_candidates(&state.handshakes, current)
            && bridge_handshakes_match_views(&state.handshakes, &state.current_views)
    });
    let pre_frame_handshake_is_pending = pre_frame_handshake_complete
        && !pre_frame_handshake_is_current
        && !state
            .handshakes
            .values()
            .any(|handshake| handshake.candidate.failed())
        && (!state.pending_page_leases.is_empty()
            || state
                .handshakes
                .values()
                .any(|handshake| handshake.staged || handshake.candidate.render_is_prepared()));
    let pre_frame_handshake_has_active = pre_frame_handshake_is_pending
        && state
            .handshakes
            .values()
            .any(|handshake| handshake.candidate.render_is_active());
    let pre_frame_handshake_matches_policy =
        bridge_handshakes_match_policy(&state.handshakes, settings);
    let pre_frame_handshake_needs_lease = pre_frame_handshake_is_pending
        && (pre_frame_handshake_has_active || pre_frame_handshake_matches_policy);
    if pre_frame_handshake_is_pending && !pre_frame_handshake_needs_lease {
        // A materializing/staged/PREPARED transaction selected for an obsolete
        // target, cap, or frozen-mode contract has not rendered and can be
        // revoked safely. ACTIVE is different: render has already drawn it, so
        // it must commit once before the next policy is selected to avoid
        // visual rollback.
        clear_bridge_pending_transaction(state)?;
    }
    if pre_frame_handshake_needs_lease {
        let ranges_are_current = state
            .handshakes
            .values()
            .flat_map(|handshake| handshake.candidate.render_ranges())
            .all(|range| state.runtime.resident_slot(range.page) == Some(range.slot));
        if ranges_are_current {
            let pages = state
                .handshakes
                .values()
                .flat_map(|handshake| {
                    handshake
                        .candidate
                        .render_ranges()
                        .iter()
                        .map(|range| range.page)
                })
                .collect::<BTreeSet<_>>();
            #[cfg(test)]
            let acquired_new_lease_set = state.pending_page_leases != pages;
            replace_bridge_pending_page_leases(state, &pages)?;
            #[cfg(test)]
            if acquired_new_lease_set {
                state.pre_frame_pending_lease_acquisitions =
                    state.pre_frame_pending_lease_acquisitions.saturating_add(1);
            }
        } else {
            // Generation mismatch means the extracted replacement never owned
            // a valid transaction. Retain the published cut and select afresh.
            clear_bridge_pending_transaction(state)?;
        }
    }
    let pre_frame_staged_replacement = pre_frame_handshake_needs_lease
        && state.handshakes.len() == camera_views.len()
        && camera_views
            .iter()
            .all(|camera| state.handshakes.contains_key(&camera.entity));

    let frame = state.runtime.begin_frame();
    let mut frames = Vec::with_capacity(camera_views.len());
    for camera in camera_views {
        let selected_view = selection_views[&camera.entity];
        let stream_frame = state
            .runtime
            .update_view_in_frame(
                frame,
                runtime_view_id(camera.entity),
                selected_view,
                settings,
                &state.streaming,
            )
            .map_err(LodBridgeError::Runtime)?;
        frames.push((camera.entity, stream_frame));
    }
    state
        .runtime
        .finish_frame(frame)
        .map_err(LodBridgeError::Runtime)?;
    for (_, stream_frame) in &frames {
        for &page in stream_frame.completed_pages() {
            let slot = state
                .runtime
                .resident_slot(page)
                .ok_or(LodBridgeError::CompletedPageNotResident(page))?;
            state.stage_completed_page(page, slot)?;
        }
    }

    let mut fallback_nodes = selected_ancestor_fallback_nodes(&*state.runtime, &frames);
    let mut frontiers = Vec::with_capacity(frames.len());
    for (camera, stream_frame) in &frames {
        let frontier = match stream_frame.candidate_frontier(settings.max_active_gaussians_u32()) {
            Ok(frontier) => frontier,
            Err(LodRuntimeError::NoResidentFrontier) => continue,
            Err(error) => return Err(LodBridgeError::Runtime(error)),
        };
        frontiers.push((*camera, frontier));
    }

    let resident_pages = frames
        .last()
        .map(|(_, frame)| frame.cache_stats().resident_pages)
        .unwrap_or(0);
    let mut complete_frontier_set =
        !camera_views.is_empty() && frontiers.len() == camera_views.len();
    let ordinary_frontier_publish_ready = complete_frontier_set
        && frames.iter().all(|(_, frame)| {
            frame.frontier().requested_nodes.is_empty()
                && frame.queued_requests() == 0
                && frame.in_flight_requests() == 0
                && frame.capacity_blocked_requests() == 0
        });
    let leases_hold_capacity =
        !state.current_page_leases.is_empty() || !state.pending_page_leases.is_empty();
    let explicitly_capacity_blocked = frames
        .iter()
        .any(|(_, frame)| frame.capacity_blocked_requests() > 0);
    let saturated_with_unresolved_demand = frames.iter().any(|(_, frame)| {
        let cache = frame.cache_stats();
        (cache.resident_pages >= state.structural.max_resident_pages
            || cache.resident_bytes >= state.structural.max_resident_bytes
            || cache.resident_gaussians >= state.structural.max_resident_gaussians)
            && !frame.frontier().requested_nodes.is_empty()
    });
    // Cold source fallback owns no bridge page lease, so a cache at structural
    // capacity can keep cycling evictable detail pages forever without ever
    // entering the active-cut pressure path. Explicit blocking and saturated
    // unresolved demand both start the bounded guard escape. Queue/in-flight
    // churn and an incomplete ordinary frontier must not prevent the globally
    // complete, permanently resident guard from becoming drawable.
    let cold_without_bridge_leases = state.current.is_none()
        && state.flat_source_bypass
        && state.current_page_leases.is_empty()
        && state.pending_page_leases.is_empty();
    let cold_capacity_pressure = cold_without_bridge_leases
        && (explicitly_capacity_blocked || saturated_with_unresolved_demand);
    let capacity_pressure = (leases_hold_capacity
        && (explicitly_capacity_blocked || saturated_with_unresolved_demand))
        || cold_capacity_pressure;
    if capacity_pressure {
        state.capacity_pressure_total_frames =
            state.capacity_pressure_total_frames.saturating_add(1);
        // Only a retained atlas cut can make a stable ordinary frontier useful
        // as slot relief. Cold pressure always waits for the independent escape
        // timer and then publishes the complete guard; it never publishes an
        // intermediate ancestor merely because that payload stayed unchanged.
        if leases_hold_capacity && complete_frontier_set {
            if state
                .capacity_pressure_payload
                .as_ref()
                .is_some_and(|payload| {
                    capacity_pressure_payload_matches_frontiers(payload, &frontiers)
                })
            {
                state.capacity_pressure_stable_frames =
                    state.capacity_pressure_stable_frames.saturating_add(1);
            } else {
                state.capacity_pressure_payload =
                    Some(capture_capacity_pressure_payload(&frontiers));
                state.capacity_pressure_stable_frames = 1;
            }
        } else {
            state.capacity_pressure_payload = None;
            state.capacity_pressure_stable_frames = 0;
        }
    } else {
        clear_capacity_pressure_tracking(state);
    }

    let stable_capacity_payload_ready = !pre_frame_staged_replacement
        && capacity_pressure
        && leases_hold_capacity
        && complete_frontier_set
        && state.capacity_pressure_stable_frames >= CAPACITY_PRESSURE_STABLE_FRAMES;
    let mut capacity_relief_selected = false;
    if stable_capacity_payload_ready {
        let ranges_are_current = frontiers
            .iter()
            .flat_map(|(_, frontier)| frontier.physical_ranges())
            .all(|range| state.runtime.resident_slot(range.page) == Some(range.slot));
        let next_pages = bridge_frontier_pages(&frontiers);
        // After this frame's selector pin swap, an old-only page with exactly
        // the bridge's current lease remaining becomes evictable at ACTIVE.
        // Exclude guard-pinned or otherwise shared pages: replacing a cut which
        // cannot actually free a slot would only add a visual transition.
        let releases_capacity = state
            .current_page_leases
            .difference(&next_pages)
            .any(|page| state.runtime.resident_pin_count(*page) == Some(1));
        let dirty_bytes = pending_gpu_upload_bytes_for_frontiers(state, &frontiers)?;
        let staging_step_bytes = gpu_staging_step_byte_limit(settings);
        let one_dirty_slot_fits =
            dirty_bytes == 0 || gpu_upload_bytes_per_slot(state)? <= staging_step_bytes;
        capacity_relief_selected = ranges_are_current && releases_capacity && one_dirty_slot_fits;
    }

    let capacity_guard_escape_ready = capacity_pressure
        && (stable_capacity_payload_ready
            || state.capacity_pressure_total_frames >= CAPACITY_PRESSURE_ESCAPE_FRAMES);

    let mut capacity_guard_selected = false;
    if !pre_frame_staged_replacement
        && !capacity_relief_selected
        && !camera_views.is_empty()
        && capacity_guard_escape_ready
    {
        let mut guard_frontiers = Vec::with_capacity(camera_views.len());
        for camera in camera_views {
            let selected_view = selection_views[&camera.entity];
            let Some(frontier) = state
                .runtime
                .coverage_guard_candidate(runtime_view_id(camera.entity), selected_view, settings)
                .map_err(LodBridgeError::Runtime)?
            else {
                continue;
            };
            guard_frontiers.push((camera.entity, frontier));
        }
        if guard_frontiers.len() == camera_views.len()
            && (coverage_guard_frontiers_satisfy_requested_quality(&guard_frontiers)
                || !bridge_has_valid_drawable_fallback(state))
        {
            let guard_transaction_in_progress = state.handshakes.len() == guard_frontiers.len()
                && guard_frontiers.iter().all(|(camera, _)| {
                    state
                        .handshakes
                        .get(camera)
                        .is_some_and(|handshake| handshake.candidate.frontier().is_coverage_guard())
                });
            // A guard is a globally complete, permanently resident antichain.
            // It may replace a drawable source/current capability only when it
            // also satisfies every camera's live presentation target. A
            // degraded guard remains available solely as last-resort recovery
            // when no valid drawable fallback exists.
            fallback_nodes = guard_frontiers
                .iter()
                .flat_map(|(_, frontier)| frontier.physical_ranges().iter().map(|range| range.node))
                .collect();
            frontiers = guard_frontiers;
            complete_frontier_set = true;
            capacity_guard_selected = true;
            if !guard_transaction_in_progress {
                // Drop the detail replacement once, then let the guard token
                // survive PREPARED -> ACTIVE across subsequent pressure frames.
                clear_bridge_pending_transaction(state)?;
            }
        } else {
            // Keep the previous drawable output when the guard is incomplete
            // or fails the live presentation target. Never discard an exact
            // source/ACTIVE cut merely to report capacity progress.
            clear_bridge_pending_transaction(state)?;
            frontiers.clear();
            complete_frontier_set = false;
        }
    }

    if !pre_frame_staged_replacement
        && !capacity_relief_selected
        && !capacity_guard_selected
        && !ordinary_frontier_publish_ready
    {
        // A complete resident ancestor cut can still be only an intermediate
        // page wave while its requested descendants are queued or in flight.
        // Publishing every such wave makes parent/child representatives swap
        // repeatedly. Keep the last ACTIVE token (or the exact cold source)
        // until selection reaches a drained target/terminal fixed point.
        #[cfg(test)]
        if complete_frontier_set {
            state.deferred_ordinary_publications =
                state.deferred_ordinary_publications.saturating_add(1);
        }
        clear_bridge_replacement_transaction(state)?;
        complete_frontier_set = false;
    }
    if pre_frame_staged_replacement {
        // Render has already consumed this complete replacement capability.
        // Selector demand and capacity pressure discovered later in the same
        // main-world frame may begin the next transaction, but cannot revoke
        // the staged/PREPARED/ACTIVE token before its phase is observed and its
        // leases/current ownership are transferred below.
        #[cfg(test)]
        {
            state.pre_frame_staged_replacement_retentions = state
                .pre_frame_staged_replacement_retentions
                .saturating_add(1);
        }
        complete_frontier_set = true;
    }

    if !complete_frontier_set {
        // With no complete cut, preserve the immutable source only during cold
        // start. Once an atlas cut has been published, retain that bounded
        // capability while its replacement/guard finishes.
        let mut candidates = state.current.clone().unwrap_or_default();
        candidates.candidate_draw_required = !state.flat_source_bypass
            && (state.current.is_some() || !state.mirror.materialized_slots().is_empty());
        let active_gaussians = state
            .current
            .as_ref()
            .and_then(|current| {
                current
                    .by_camera
                    .values()
                    .map(|candidate| u64::from(candidate.rendered_candidate_count()))
                    .max()
            })
            .unwrap_or(0);
        return Ok((
            candidates,
            GaussianLodBridgeStatus {
                phase: state.current.as_ref().map_or(
                    GaussianLodBridgePhase::StreamingFallback,
                    |current| {
                        if bridge_candidate_set_is_render_active(current) {
                            GaussianLodBridgePhase::Active
                        } else {
                            GaussianLodBridgePhase::WaitingForRender
                        }
                    },
                ),
                active_views: state
                    .current
                    .as_ref()
                    .map_or(0, |current| current.len().try_into().unwrap_or(u32::MAX)),
                resident_pages,
                active_gaussians,
                failure: None,
            },
        ));
    }

    let handshakes_match_current = state.current.as_ref().is_some_and(|current| {
        bridge_handshakes_match_candidates(&state.handshakes, current)
            && bridge_handshakes_match_views(&state.handshakes, &state.current_views)
    });
    if handshakes_match_current
        && state.active
        && state
            .current
            .as_ref()
            .is_some_and(|current| !bridge_candidate_set_is_render_active(current))
    {
        // Device/pipeline recreation revokes the retained render capability
        // without changing its logical frontier. Re-stage that same cut once;
        // its CPU atlas mirror and page leases remain valid.
        state.active = false;
        for handshake in state.handshakes.values_mut() {
            handshake.staged = false;
        }
    }
    let handshake_camera_set_complete = state.handshakes.len() == camera_views.len()
        && camera_views
            .iter()
            .all(|camera| state.handshakes.contains_key(&camera.entity));
    let handshake_failed = state
        .handshakes
        .values()
        .any(|handshake| handshake.candidate.failed());
    if handshake_failed {
        if state.current.is_none() || handshakes_match_current {
            return Err(LodBridgeError::RenderCommitFailed);
        }
        clear_bridge_pending_transaction(state)?;
        let mut current = state.current.clone().unwrap_or_default();
        current.candidate_draw_required = true;
        let active_gaussians = current
            .by_camera
            .values()
            .map(|candidate| u64::from(candidate.rendered_candidate_count()))
            .max()
            .unwrap_or(0);
        return Ok((
            current,
            GaussianLodBridgeStatus {
                phase: if bridge_candidate_set_is_render_active(
                    state
                        .current
                        .as_ref()
                        .expect("the retained failure path requires a current cut"),
                ) {
                    GaussianLodBridgePhase::Active
                } else {
                    GaussianLodBridgePhase::WaitingForRender
                },
                active_views: state
                    .current
                    .as_ref()
                    .map_or(0, |current| current.len().try_into().unwrap_or(u32::MAX)),
                resident_pages,
                active_gaussians,
                failure: Some(LodOrchestrationFailure::from(
                    &LodBridgeError::RenderCommitFailed,
                )),
            },
        ));
    }

    // Once materialization starts or any camera prepares a complete
    // replacement, retain the entire all-camera payload through activation.
    // Continuous camera motion may keep selecting and requesting newer pages,
    // but it cannot cancel the transaction every main/render-world round trip.
    let retain_pending = handshake_camera_set_complete
        && !handshakes_match_current
        && (!state.pending_page_leases.is_empty()
            || state
                .handshakes
                .values()
                .any(|handshake| handshake.staged || handshake.candidate.render_is_prepared()));
    let (mut render_candidates, debug_fallback_nodes) = if retain_pending {
        (
            bridge_handshake_candidates(state),
            state.pending_fallback_nodes.clone(),
        )
    } else {
        if !state.pending_page_leases.is_empty() {
            replace_bridge_pending_page_leases(state, &BTreeSet::new())?;
        }
        let mut candidates = LodRenderCandidates::default();
        for (camera, frontier) in frontiers {
            let selected_view = selection_views[&camera];
            let phase = state.handshake_for(camera, &frontier, selected_view);
            candidates
                .by_camera
                .insert(camera, LodRenderCandidate::with_phase(frontier, phase));
        }
        state.pending_fallback_nodes.clone_from(&fallback_nodes);
        (candidates, fallback_nodes)
    };
    render_candidates.staging_atlas = Some(state.atlas.id());
    render_candidates.candidate_draw_required = !state.flat_source_bypass
        && (state.current.is_some() || !state.mirror.materialized_slots().is_empty());

    let complete_camera_set =
        !camera_views.is_empty() && render_candidates.by_camera.len() == camera_views.len();
    let every_active = complete_camera_set
        && render_candidates
            .by_camera
            .values()
            .all(|candidate| candidate.phase.load(Ordering::Acquire) == LOD_RENDER_ACTIVE);
    let every_prepared = complete_camera_set
        && render_candidates.by_camera.values().all(|candidate| {
            matches!(
                candidate.phase.load(Ordering::Acquire),
                LOD_RENDER_PREPARED | LOD_RENDER_ACTIVE
            )
        });
    let every_staged = complete_camera_set
        && render_candidates.by_camera.keys().all(|camera| {
            state
                .handshakes
                .get(camera)
                .is_some_and(|handshake| handshake.staged)
        });
    let any_failed = render_candidates
        .by_camera
        .values()
        .any(|candidate| candidate.phase.load(Ordering::Acquire) == LOD_RENDER_FAILED);
    if any_failed {
        return Err(LodBridgeError::RenderCommitFailed);
    }

    if state.flat_source_bypass && every_prepared {
        // Render preparation has proved both the bounded atlas generations and
        // the compaction/radix variants while the immutable source remained
        // drawable. The next extraction can atomically bind the atlas; its
        // first candidate compacts and draws in that same render frame.
        state.flat_source_bypass = false;
    }

    let candidate_is_current = state
        .current
        .as_ref()
        .is_some_and(|current| bridge_candidate_sets_match(current, &render_candidates));
    let mut attempted_candidate_staging = false;
    if state.current.is_none() && complete_camera_set && !every_staged {
        // Cold publication has no retained atlas output to protect. Lease and
        // progressively queue the complete bounded cut while the immutable
        // source remains drawable. Render keeps this token WAITING until every
        // referenced slot generation has landed, then the existing
        // PREPARED/ACTIVE handshake performs one atomic source-to-atlas handoff.
        let pending_pages = bridge_candidate_pages(&render_candidates);
        replace_bridge_pending_page_leases(state, &pending_pages)?;
        attempted_candidate_staging = true;
        let staged = synchronize_bridge_candidate_pages_in_atlas_bounded(
            state,
            &render_candidates,
            &debug_fallback_nodes,
            assets,
            atlas_uploads,
            gpu_staging_step_byte_limit(settings),
        )?;
        if staged {
            for camera in render_candidates.by_camera.keys() {
                if let Some(handshake) = state.handshakes.get_mut(camera) {
                    handshake.staged = true;
                }
            }
        }
    }
    let any_prepared = complete_camera_set
        && render_candidates
            .by_camera
            .values()
            .any(LodRenderCandidate::render_is_prepared);
    if !candidate_is_current && any_prepared {
        // Runtime cache insertions finish before its per-view frontier pins are
        // swapped, and no insertion follows that swap in update_view_in_frame.
        // Thus the PREPARED payload is still resident here; acquiring its
        // explicit lease now closes the main/render-world activation race
        // without pinning every short-lived WAITING motion candidate.
        let pending_pages = bridge_candidate_pages(&render_candidates);
        replace_bridge_pending_page_leases(state, &pending_pages)?;
    }

    if !candidate_is_current && every_active {
        // ACTIVE is published by the render world only after every staged
        // atlas generation has landed and the candidate radix output is ready.
        // Pending pages already own independent leases, so releasing the old
        // published set cannot expose a one-frame eviction/reuse race.
        let next_pages = bridge_candidate_pages(&render_candidates);
        commit_bridge_page_leases(state, &next_pages)?;
        state.current_views = render_candidates
            .by_camera
            .keys()
            .filter_map(|camera| {
                state
                    .handshakes
                    .get(camera)
                    .map(|handshake| (*camera, handshake.selected_view))
            })
            .collect();
        debug_assert_eq!(state.current_views.len(), render_candidates.len());
        state.current = Some(render_candidates.clone());
        state.active = true;
        // Any distinct ACTIVE publication changes the render-owned lease set.
        // Start a fresh pressure epoch so released slots have a bounded chance
        // to admit the next refinement wave before guard fallback.
        clear_capacity_pressure_tracking(state);
    } else if candidate_is_current && every_active {
        state.current_views = render_candidates
            .by_camera
            .keys()
            .filter_map(|camera| {
                state
                    .handshakes
                    .get(camera)
                    .map(|handshake| (*camera, handshake.selected_view))
            })
            .collect();
        state.current = Some(render_candidates.clone());
        state.active = true;
    } else if every_prepared && !every_staged {
        if candidate_is_current {
            requeue_bridge_candidate_pages(state, &render_candidates, atlas_uploads)?;
            for camera in render_candidates.by_camera.keys() {
                if let Some(handshake) = state.handshakes.get_mut(camera) {
                    handshake.staged = true;
                }
            }
        } else if !attempted_candidate_staging {
            let staged = synchronize_bridge_candidate_pages_in_atlas_bounded(
                state,
                &render_candidates,
                &debug_fallback_nodes,
                assets,
                atlas_uploads,
                gpu_staging_step_byte_limit(settings),
            )?;
            if staged {
                for camera in render_candidates.by_camera.keys() {
                    if let Some(handshake) = state.handshakes.get_mut(camera) {
                        handshake.staged = true;
                    }
                }
            }
        }
    }
    render_candidates.candidate_draw_required = !state.flat_source_bypass
        && (render_candidates.candidate_draw_required
            || state.current.is_some()
            || !state.mirror.materialized_slots().is_empty()
            || (!candidate_is_current && every_prepared));

    let visible_candidates = state.current.as_ref().unwrap_or(&render_candidates);
    let active_gaussians = visible_candidates
        .by_camera
        .values()
        .map(|candidate| u64::from(candidate.rendered_candidate_count()))
        .max()
        .unwrap_or(0);
    let active_views = visible_candidates.len().try_into().unwrap_or(u32::MAX);
    let phase = state.current.as_ref().map_or_else(
        || {
            if complete_camera_set {
                GaussianLodBridgePhase::WaitingForRender
            } else {
                GaussianLodBridgePhase::StreamingFallback
            }
        },
        |current| {
            if bridge_candidate_set_is_render_active(current) {
                GaussianLodBridgePhase::Active
            } else {
                GaussianLodBridgePhase::WaitingForRender
            }
        },
    );
    Ok((
        render_candidates,
        GaussianLodBridgeStatus {
            phase,
            active_views,
            resident_pages,
            active_gaussians,
            failure: None,
        },
    ))
}

fn synchronize_bridge_candidate_pages_in_atlas_bounded(
    state: &mut BridgeCloudState,
    render_candidates: &LodRenderCandidates,
    debug_fallback_nodes: &BTreeSet<LodNodeId>,
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
    max_new_bytes: u64,
) -> Result<bool, LodBridgeError> {
    let atlas_id = state.atlas.id();
    if let Some(transient) = state.transient_atlas.take() {
        let result = synchronize_bridge_candidate_pages_bounded(
            state,
            render_candidates,
            debug_fallback_nodes,
            atlas_id,
            BridgeAtlasMaterialization::Sparse(&transient),
            atlas_uploads,
            max_new_bytes,
        );
        state.transient_atlas = Some(transient);
        result
    } else {
        let atlas = assets
            .get_mut_untracked(&state.atlas)
            .ok_or(LodBridgeError::MissingAtlasAsset)?;
        synchronize_bridge_candidate_pages_bounded(
            state,
            render_candidates,
            debug_fallback_nodes,
            atlas_id,
            BridgeAtlasMaterialization::Dense(atlas),
            atlas_uploads,
            max_new_bytes,
        )
    }
}

enum BridgeAtlasMaterialization<'a> {
    Dense(&'a mut PlanarGaussian3d),
    Sparse(&'a LodTransientAtlas),
}

fn requeue_bridge_candidate_pages(
    state: &BridgeCloudState,
    candidates: &LodRenderCandidates,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), LodBridgeError> {
    let atlas = state.atlas.id();
    let gaussians_per_slot = state.mirror.layout().gaussians_per_slot;
    let pages = candidates
        .by_camera
        .values()
        .flat_map(LodRenderCandidate::render_ranges)
        .map(|range| (range.page, range.slot))
        .collect::<BTreeSet<_>>();
    for (page, slot) in pages {
        if !state.mirror.is_page_current(page, slot) {
            return Err(LodBridgeError::AtlasUpload(format!(
                "retained LoD page {page:?} is not current in slot {slot:?}"
            )));
        }
        atlas_uploads
            .enqueue_slot(atlas, slot, gaussians_per_slot)
            .map_err(|error| LodBridgeError::AtlasUpload(error.to_string()))?;
    }
    for candidate in candidates.by_camera.values() {
        state.mirror.validate_ranges(candidate.render_ranges())?;
    }
    Ok(())
}

/// Synchronizes each physical page at most once even when several logical
/// sibling ranges or cameras reference the same slot.
#[cfg(test)]
fn synchronize_bridge_candidate_pages(
    state: &mut BridgeCloudState,
    candidates: &LodRenderCandidates,
    fallback_nodes: &BTreeSet<LodNodeId>,
    atlas_id: AssetId<PlanarGaussian3d>,
    atlas: BridgeAtlasMaterialization<'_>,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), LodBridgeError> {
    let complete = synchronize_bridge_candidate_pages_bounded(
        state,
        candidates,
        fallback_nodes,
        atlas_id,
        atlas,
        atlas_uploads,
        u64::MAX,
    )?;
    debug_assert!(
        complete,
        "an unbounded staging step must materialize every candidate page"
    );
    Ok(())
}

/// Materializes a deterministic prefix of the candidate's dirty physical
/// pages. The full candidate page union is leased by the caller before this
/// function runs, so returning an incomplete step never exposes slot reuse.
fn synchronize_bridge_candidate_pages_bounded(
    state: &mut BridgeCloudState,
    candidates: &LodRenderCandidates,
    fallback_nodes: &BTreeSet<LodNodeId>,
    atlas_id: AssetId<PlanarGaussian3d>,
    mut atlas: BridgeAtlasMaterialization<'_>,
    atlas_uploads: &mut LodAtlasUploadQueue,
    max_new_bytes: u64,
) -> Result<bool, LodBridgeError> {
    let pages = candidates
        .by_camera
        .values()
        .flat_map(LodRenderCandidate::render_ranges)
        .map(|range| (range.page, range.slot))
        .collect::<BTreeSet<_>>();
    let gaussians_per_slot = state.mirror.layout().gaussians_per_slot;
    let bytes_per_slot = gpu_upload_bytes_per_slot(state)?;
    if pages
        .iter()
        .any(|(page, slot)| !state.mirror.is_page_current(*page, *slot))
        && bytes_per_slot > max_new_bytes
    {
        return Err(LodBridgeError::GpuUploadCommitBudgetExceeded {
            required: bytes_per_slot,
            limit: max_new_bytes,
        });
    }
    let mut remaining_bytes = max_new_bytes;

    for (page_id, slot) in pages {
        let mut page_fallbacks = state
            .debug_manifest_index
            .as_ref()
            .and_then(|index| index.node_ids(page_id))
            .into_iter()
            .flatten()
            .filter(|node| fallback_nodes.contains(node))
            .collect::<Vec<_>>();
        page_fallbacks.sort_unstable();
        let debug_current = state.debug_page_is_current(page_id, slot, &page_fallbacks);
        let atlas_current = state.mirror.is_page_current(page_id, slot);
        if !atlas_current && remaining_bytes < bytes_per_slot {
            break;
        }
        if debug_current && atlas_current {
            continue;
        }
        #[cfg(test)]
        {
            state.decoded_page_acquisitions = state.decoded_page_acquisitions.saturating_add(1);
        }
        let page = state
            .runtime
            .decoded_page(page_id)
            .ok_or(LodBridgeError::ResidentPageNotDecoded(page_id))?;
        if !debug_current {
            state.sync_debug_page(&page, slot, &page_fallbacks)?;
        }
        if !atlas_current {
            match &mut atlas {
                BridgeAtlasMaterialization::Dense(atlas) => {
                    state.mirror.materialize_page(atlas, &page, slot)?;
                }
                BridgeAtlasMaterialization::Sparse(transient) => {
                    let payload = state.mirror.materialize_page_payload(&page, slot)?;
                    transient
                        .write_slot(slot.index, gaussians_per_slot, payload)
                        .map_err(|error| LodBridgeError::AtlasUpload(error.to_string()))?;
                }
            }
            atlas_uploads
                .enqueue_slot(atlas_id, slot, gaussians_per_slot)
                .map_err(|error| LodBridgeError::AtlasUpload(error.to_string()))?;
            remaining_bytes -= bytes_per_slot;
        }
    }
    let complete = candidates.by_camera.values().all(|candidate| {
        candidate
            .render_ranges()
            .iter()
            .all(|range| state.mirror.is_range_current(*range))
    });
    if complete {
        for candidate in candidates.by_camera.values() {
            state.mirror.validate_ranges(candidate.render_ranges())?;
        }
    }
    Ok(complete)
}

#[cfg(test)]
fn pending_gpu_upload_bytes(
    state: &BridgeCloudState,
    candidates: &LodRenderCandidates,
) -> Result<u64, LodBridgeError> {
    let dirty_slots = candidates
        .by_camera
        .values()
        .flat_map(LodRenderCandidate::render_ranges)
        .filter(|range| !state.mirror.is_range_current(**range))
        .map(|range| range.slot.index)
        .collect::<BTreeSet<_>>();
    let bytes_per_slot = gpu_upload_bytes_per_slot(state)?;
    (dirty_slots.len() as u64)
        .checked_mul(bytes_per_slot)
        .ok_or(LodBridgeError::AtlasSizeOverflow)
}

fn gpu_upload_bytes_per_slot(state: &BridgeCloudState) -> Result<u64, LodBridgeError> {
    u64::from(state.mirror.layout().gaussians_per_slot)
        .checked_mul(gaussian_3d_gpu_bytes_per_record())
        .ok_or(LodBridgeError::AtlasSizeOverflow)
}

fn gpu_staging_step_byte_limit(settings: &GaussianLodSettings) -> u64 {
    settings
        .budgets
        .max_gpu_upload_bytes_per_commit
        .min(settings.budgets.max_upload_bytes_per_frame)
}

#[cfg(test)]
fn validate_gpu_upload_commit_budget(
    state: &BridgeCloudState,
    candidates: &LodRenderCandidates,
    limit: u64,
) -> Result<u64, LodBridgeError> {
    let required = pending_gpu_upload_bytes(state, candidates)?;
    if required > limit {
        return Err(LodBridgeError::GpuUploadCommitBudgetExceeded { required, limit });
    }
    Ok(required)
}

fn selected_ancestor_fallback_nodes(
    runtime: &dyn ErasedLodRuntime,
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
            let mut cursor = runtime.parent(requested);
            while let Some(ancestor) = cursor {
                if selected.contains(&ancestor) {
                    nodes.insert(ancestor);
                    break;
                }
                cursor = runtime.parent(ancestor);
            }
        }
    }
    nodes
}

fn runtime_view_id(entity: Entity) -> LodRuntimeViewId {
    LodRuntimeViewId(entity.to_bits())
}

fn bridge_candidate_pages(candidates: &LodRenderCandidates) -> BTreeSet<LodPageId> {
    candidates
        .by_camera
        .values()
        .flat_map(|candidate| candidate.render_ranges().iter().map(|range| range.page))
        .collect()
}

fn clear_capacity_pressure_tracking(state: &mut BridgeCloudState) {
    state.capacity_pressure_payload = None;
    state.capacity_pressure_stable_frames = 0;
    state.capacity_pressure_total_frames = 0;
}

fn bridge_frontier_pages(frontiers: &[(Entity, LodCandidateFrontier)]) -> BTreeSet<LodPageId> {
    frontiers
        .iter()
        .flat_map(|(_, frontier)| frontier.physical_ranges().iter().map(|range| range.page))
        .collect()
}

fn capacity_pressure_payload_matches_frontiers(
    payload: &BTreeMap<Entity, CapacityPressureCandidatePayload>,
    frontiers: &[(Entity, LodCandidateFrontier)],
) -> bool {
    payload.len() == frontiers.len()
        && frontiers.iter().all(|(camera, frontier)| {
            payload.get(camera).is_some_and(|candidate| {
                candidate.view == frontier.view()
                    && candidate.candidate_count == frontier.candidate_count()
                    && candidate.physical_ranges.as_slice() == frontier.physical_ranges()
            })
        })
}

fn capture_capacity_pressure_payload(
    frontiers: &[(Entity, LodCandidateFrontier)],
) -> BTreeMap<Entity, CapacityPressureCandidatePayload> {
    frontiers
        .iter()
        .map(|(camera, frontier)| {
            (
                *camera,
                CapacityPressureCandidatePayload {
                    view: frontier.view(),
                    candidate_count: frontier.candidate_count(),
                    physical_ranges: frontier.physical_ranges().to_vec(),
                },
            )
        })
        .collect()
}

fn pending_gpu_upload_bytes_for_frontiers(
    state: &BridgeCloudState,
    frontiers: &[(Entity, LodCandidateFrontier)],
) -> Result<u64, LodBridgeError> {
    let dirty_slots = frontiers
        .iter()
        .flat_map(|(_, frontier)| frontier.physical_ranges())
        .filter(|range| !state.mirror.is_range_current(**range))
        .map(|range| range.slot.index)
        .collect::<BTreeSet<_>>();
    let bytes_per_slot = gpu_upload_bytes_per_slot(state)?;
    (dirty_slots.len() as u64)
        .checked_mul(bytes_per_slot)
        .ok_or(LodBridgeError::AtlasSizeOverflow)
}

fn bridge_candidate_matches_frontier(
    candidate: &LodRenderCandidate,
    frontier: &LodCandidateFrontier,
) -> bool {
    candidate.frontier().view() == frontier.view()
        && candidate.frontier().candidate_count() == frontier.candidate_count()
        && candidate.frontier().physical_ranges() == frontier.physical_ranges()
}

fn bridge_candidate_sets_match(left: &LodRenderCandidates, right: &LodRenderCandidates) -> bool {
    left.by_camera.len() == right.by_camera.len()
        && left.by_camera.iter().all(|(camera, candidate)| {
            right
                .get(*camera)
                .is_some_and(|other| candidate.same_payload(other))
        })
}

fn bridge_handshakes_match_candidates(
    handshakes: &BTreeMap<Entity, BridgeHandshake>,
    candidates: &LodRenderCandidates,
) -> bool {
    handshakes.len() == candidates.len()
        && handshakes.iter().all(|(camera, handshake)| {
            candidates
                .get(*camera)
                .is_some_and(|candidate| candidate.same_payload(&handshake.candidate))
        })
}

fn bridge_handshakes_match_policy(
    handshakes: &BTreeMap<Entity, BridgeHandshake>,
    settings: &GaussianLodSettings,
) -> bool {
    let requested_target = settings.quality_target();
    let max_active_gaussians = settings.max_active_gaussians_u32();
    let selection_view_frozen = settings.selection_mode == LodSelectionMode::Frozen;
    handshakes.values().all(|handshake| {
        let frontier = handshake.candidate.frontier();
        frontier.quality_status().requested_target == requested_target
            && frontier.candidate_count() <= max_active_gaussians
            && frontier.selection_view_frozen() == selection_view_frozen
    })
}

fn bridge_handshakes_match_views(
    handshakes: &BTreeMap<Entity, BridgeHandshake>,
    views: &BTreeMap<Entity, LodView>,
) -> bool {
    handshakes.len() == views.len()
        && handshakes.iter().all(|(camera, handshake)| {
            views
                .get(camera)
                .is_some_and(|view| *view == handshake.selected_view)
        })
}

fn bridge_handshake_candidates(state: &BridgeCloudState) -> LodRenderCandidates {
    LodRenderCandidates {
        by_camera: state
            .handshakes
            .iter()
            .map(|(&camera, handshake)| (camera, handshake.candidate.clone()))
            .collect(),
        staging_atlas: Some(state.atlas.id()),
        candidate_draw_required: true,
        retained_current: false,
        candidates_are_current: false,
        retained_current_is_stale: false,
        debug_metadata_staged: true,
        transition_must_commit: false,
    }
}

/// Replaces one independently reference-counted bridge lease set. New pages
/// are retained before old-only pages are released, and a failed operation
/// restores the previous set.
fn replace_bridge_page_leases(
    runtime: &mut dyn ErasedLodRuntime,
    leases: &mut BTreeSet<LodPageId>,
    next: &BTreeSet<LodPageId>,
) -> Result<(), LodBridgeError> {
    if *leases == *next {
        return Ok(());
    }
    let previous = leases.clone();
    let mut acquired = Vec::new();
    for &page in next.difference(&previous) {
        if let Err(error) = runtime.retain_resident_page(page) {
            for acquired_page in acquired.into_iter().rev() {
                let _ = runtime.release_resident_page(acquired_page);
            }
            return Err(LodBridgeError::Runtime(error));
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
            return Err(LodBridgeError::Runtime(error));
        }
        released.push(page);
    }
    leases.clone_from(next);
    Ok(())
}

fn replace_bridge_pending_page_leases(
    state: &mut BridgeCloudState,
    next: &BTreeSet<LodPageId>,
) -> Result<(), LodBridgeError> {
    replace_bridge_page_leases(&mut *state.runtime, &mut state.pending_page_leases, next)
}

/// Publishes a staged all-camera lease transaction. Pending and current sets
/// own distinct cache references, including on shared pages, so releasing the
/// complete old set leaves exactly one reference for every newly active page.
fn commit_bridge_page_leases(
    state: &mut BridgeCloudState,
    next: &BTreeSet<LodPageId>,
) -> Result<(), LodBridgeError> {
    if state.pending_page_leases == *next {
        let previous = state.current_page_leases.clone();
        let mut released = Vec::new();
        for &page in &previous {
            if let Err(error) = state.runtime.release_resident_page(page) {
                for released_page in released {
                    let _ = state.runtime.retain_resident_page(released_page);
                }
                return Err(LodBridgeError::Runtime(error));
            }
            released.push(page);
        }
        state.current_page_leases = std::mem::take(&mut state.pending_page_leases);
        return Ok(());
    }

    replace_bridge_page_leases(&mut *state.runtime, &mut state.current_page_leases, next)?;
    replace_bridge_pending_page_leases(state, &BTreeSet::new())
}

fn clear_bridge_pending_transaction(state: &mut BridgeCloudState) -> Result<(), LodBridgeError> {
    restore_bridge_runtime_to_current(state, true)?;
    replace_bridge_pending_page_leases(state, &BTreeSet::new())?;
    if state.current.is_none() {
        // PREPARED may already have lowered the cold source bypass even though
        // no atlas cut has reached ACTIVE. Any invalidation of that pending
        // transaction must restore the complete source before its token and
        // leases disappear, otherwise the sparse atlas would be exposed with
        // no validated per-view output.
        state.flat_source_bypass = true;
    }
    for handshake in state.handshakes.values() {
        let is_current = state.current.as_ref().is_some_and(|current| {
            current
                .by_camera
                .values()
                .any(|candidate| Arc::ptr_eq(&candidate.phase, &handshake.phase))
        });
        if !is_current {
            handshake.phase.store(LOD_RENDER_WAITING, Ordering::Release);
        }
    }
    state.handshakes.clear();
    state.pending_fallback_nodes.clear();
    Ok(())
}

fn clear_bridge_replacement_transaction(
    state: &mut BridgeCloudState,
) -> Result<(), LodBridgeError> {
    restore_bridge_runtime_to_current(state, false)?;
    replace_bridge_pending_page_leases(state, &BTreeSet::new())?;
    if state.current.is_none() {
        // See `clear_bridge_pending_transaction`: dropping a cold PREPARED
        // replacement also revokes the atlas handoff capability.
        state.flat_source_bypass = true;
    }
    let current = state.current.as_ref();
    state.handshakes.retain(|camera, handshake| {
        let is_current = current.is_some_and(|current| {
            current
                .get(*camera)
                .is_some_and(|candidate| Arc::ptr_eq(&candidate.phase, &handshake.phase))
        });
        if !is_current {
            handshake.phase.store(LOD_RENDER_WAITING, Ordering::Release);
        }
        is_current
    });
    state.pending_fallback_nodes.clear();
    Ok(())
}

fn restore_bridge_runtime_to_current(
    state: &mut BridgeCloudState,
    clear_temporal_demand: bool,
) -> Result<(), LodBridgeError> {
    let Some(current) = state.current.as_ref() else {
        return Ok(());
    };
    let live_views = state
        .views
        .iter()
        .map(|camera| runtime_view_id(*camera))
        .collect::<BTreeSet<_>>();
    let retained = current
        .by_camera
        .values()
        .filter_map(|candidate| {
            let view = candidate.frontier().view();
            live_views.contains(&view).then(|| {
                (
                    view,
                    candidate
                        .target_render_ranges()
                        .iter()
                        .map(|range| range.node)
                        .collect::<Vec<_>>(),
                )
            })
        })
        .collect::<Vec<_>>();
    for (view, nodes) in retained {
        let result = if clear_temporal_demand {
            state.runtime.restore_rendered_frontier(view, &nodes)
        } else {
            state.runtime.retry_from_rendered_frontier(view, &nodes)
        };
        result.map_err(LodBridgeError::Runtime)?;
    }
    Ok(())
}

fn restore_bridge_before_discard(
    state: &mut BridgeCloudState,
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), LodBridgeError> {
    // The entity is rebound to its immutable source by the caller. A bounded
    // page-cache atlas has no flat fallback layout to restore; simply retire
    // its upload work and owned asset without an O(source) rewrite.
    atlas_uploads.remove_atlas(state.atlas.id());
    if state.transient_atlas.is_none() {
        assets.remove(state.atlas.id());
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LodBridgeError {
    ZeroLimit(&'static str),
    Build(String),
    AtlasUpload(String),
    Codec(String),
    DebugAnnotations(String),
    StreamingSettings(String),
    UnsupportedRenderPath(LodRenderPathSupportError),
    Runtime(LodRuntimeError),
    SourceTooLarge {
        actual: u64,
        limit: u64,
    },
    StoredGaussianLimit {
        actual: u64,
        limit: u64,
    },
    AtlasCannotFitPage {
        stride: u32,
        max_gaussians: u32,
        max_bytes: u64,
    },
    AtlasCannotFitSource {
        source: u32,
        physical: u32,
    },
    AtlasSizeOverflow,
    AtlasSlotOutOfRange(u32),
    AtlasPageNotStaged(LodPageId),
    StaleAtlasSlot {
        page: LodPageId,
        slot: AtlasSlot,
    },
    PageExceedsAtlasStride {
        page: LodPageId,
        count: u32,
        stride: u32,
    },
    AtlasLengthMismatch {
        expected: u32,
        actual: usize,
    },
    MissingPageDescriptor(LodPageId),
    CompletedPageNotResident(LodPageId),
    ResidentPageNotDecoded(LodPageId),
    FrontierReferencesUnsynchronizedPage {
        page: LodPageId,
        slot: AtlasSlot,
    },
    MissingAtlasAsset,
    MissingSourceAsset,
    FallbackExceedsAtlas {
        source: u64,
        atlas: u64,
    },
    CompleteFallbackExceedsGpuUploadBudget {
        required: u64,
        limit: u64,
    },
    GpuUploadCommitBudgetExceeded {
        required: u64,
        limit: u64,
    },
    RenderCommitFailed,
    ViewLimitExceeded {
        actual: u64,
        limit: u32,
    },
    UnsupportedCamera(Entity),
}

impl fmt::Display for LodBridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LodBridgeError {}

impl From<LodRenderCommitError> for LodBridgeError {
    fn from(error: LodRenderCommitError) -> Self {
        match error {
            LodRenderCommitError::ZeroAtlasSlots => Self::ZeroLimit("atlas slot count"),
            LodRenderCommitError::AtlasSizeOverflow => Self::AtlasSizeOverflow,
            LodRenderCommitError::AtlasSlotOutOfRange(index) => Self::AtlasSlotOutOfRange(index),
            LodRenderCommitError::AtlasPageNotStaged(page) => Self::AtlasPageNotStaged(page),
            LodRenderCommitError::StaleAtlasSlot { page, slot } => {
                Self::StaleAtlasSlot { page, slot }
            }
            LodRenderCommitError::PageExceedsAtlasStride {
                page,
                count,
                stride,
            } => Self::PageExceedsAtlasStride {
                page,
                count,
                stride,
            },
            LodRenderCommitError::AtlasLengthMismatch { expected, actual } => {
                Self::AtlasLengthMismatch { expected, actual }
            }
            LodRenderCommitError::FrontierReferencesUnsynchronizedPage { page, slot } => {
                Self::FrontierReferencesUnsynchronizedPage { page, slot }
            }
        }
    }
}

impl From<&LodBridgeError> for LodOrchestrationFailure {
    fn from(error: &LodBridgeError) -> Self {
        let code = match error {
            LodBridgeError::ZeroLimit(_)
            | LodBridgeError::Build(_)
            | LodBridgeError::StreamingSettings(_) => {
                LodOrchestrationFailureCode::InvalidConfiguration
            }
            LodBridgeError::UnsupportedRenderPath(_) | LodBridgeError::UnsupportedCamera(_) => {
                LodOrchestrationFailureCode::UnsupportedConfiguration
            }
            LodBridgeError::MissingSourceAsset => LodOrchestrationFailureCode::SourceUnavailable,
            LodBridgeError::Codec(_) => LodOrchestrationFailureCode::DecodeValidationFailed,
            LodBridgeError::Runtime(_) => LodOrchestrationFailureCode::RuntimeFailed,
            LodBridgeError::AtlasUpload(_)
            | LodBridgeError::AtlasSlotOutOfRange(_)
            | LodBridgeError::AtlasPageNotStaged(_)
            | LodBridgeError::StaleAtlasSlot { .. }
            | LodBridgeError::PageExceedsAtlasStride { .. }
            | LodBridgeError::AtlasLengthMismatch { .. }
            | LodBridgeError::FrontierReferencesUnsynchronizedPage { .. }
            | LodBridgeError::MissingAtlasAsset => LodOrchestrationFailureCode::AtlasCommitFailed,
            LodBridgeError::SourceTooLarge { .. }
            | LodBridgeError::StoredGaussianLimit { .. }
            | LodBridgeError::AtlasCannotFitPage { .. }
            | LodBridgeError::AtlasCannotFitSource { .. }
            | LodBridgeError::AtlasSizeOverflow
            | LodBridgeError::FallbackExceedsAtlas { .. }
            | LodBridgeError::CompleteFallbackExceedsGpuUploadBudget { .. }
            | LodBridgeError::GpuUploadCommitBudgetExceeded { .. }
            | LodBridgeError::ViewLimitExceeded { .. } => {
                LodOrchestrationFailureCode::CapacityExceeded
            }
            LodBridgeError::RenderCommitFailed => LodOrchestrationFailureCode::RenderCommitFailed,
            LodBridgeError::DebugAnnotations(_)
            | LodBridgeError::MissingPageDescriptor(_)
            | LodBridgeError::CompletedPageNotResident(_)
            | LodBridgeError::ResidentPageNotDecoded(_) => {
                LodOrchestrationFailureCode::InternalInvariant
            }
        };
        Self::with_detail(code, error.to_string())
    }
}

#[cfg(test)]
mod tests;
