//! Automatic Bevy integration for bounded Gaussian LoD streaming.
//!
//! Flat 3D clouds are converted once into an ephemeral hierarchy, streamed
//! through the same validated runtime as packaged scenes, mirrored into a
//! fixed-capacity planar atlas, and committed per camera to GPU compaction.
//! The atlas initially contains a complete padded copy of the source. Resident
//! pages are not materialized until the render world confirms that compaction
//! pipelines and a complete candidate list are staged, so every failure path
//! retains the complete flat fallback draw.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

use bevy::{
    asset::AssetEventSystems,
    camera::{CameraUpdateSystems, Projection, visibility::VisibilitySystems},
    prelude::*,
    transform::TransformSystems,
};
use bevy_interleave::prelude::{Planar, PlanarHandle};

use crate::{
    CloudSettings, GaussianCamera,
    gaussian::{
        formats::{
            planar_3d::{
                Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle,
                gaussian_3d_gpu_bytes_per_record,
            },
            planar_3d_chunked::{LodNodeId, LodPageId, PlanarGaussian3dPage},
            planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        },
        lod_debug::{
            LodDebugAnnotationAtlas, LodDebugManifestIndex, LodDebugMetadata, LodDebugResidency,
        },
        lod_settings::{GaussianLodSettings, GaussianStreamingSettings, LodQualityEndpoint},
    },
    io::lod::{GaussianLodHandle, encode_page},
    stream::{
        LodRenderPathSupportError,
        atlas_upload::{GaussianLodAtlasUploadPlugin, LodAtlasUploadQueue},
        cache::AtlasSlot,
        hierarchy::LodView,
        render_commit::{
            GaussianLodRenderCommitPlugin, LodOrchestrationFailure, LodOrchestrationFailureCode,
            LodOrchestrationSource, LodOrchestrationTransition, LodOrchestrationTransitionKind,
            LodRenderCommitError,
        },
        require_lod_render_path,
        runtime::{
            LodCandidateFrontier, LodRuntimeError, LodRuntimeFrameId, LodRuntimeViewId,
            LodStreamFrame, LodStreamingRuntime,
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

const FLAT_SOURCE_BYPASS_MIN_SELECTED_NUMERATOR: u64 = 95;
const FLAT_SOURCE_BYPASS_MIN_SELECTED_DENOMINATOR: u64 = 100;

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
    fn remove_view(&mut self, view: LodRuntimeViewId) -> Result<bool, LodRuntimeError>;
    fn resident_slot(&self, page: LodPageId) -> Option<AtlasSlot>;
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

    fn remove_view(&mut self, view: LodRuntimeViewId) -> Result<bool, LodRuntimeError> {
        LodStreamingRuntime::remove_view(self, view)
    }

    fn resident_slot(&self, page: LodPageId) -> Option<AtlasSlot> {
        self.cache().get(page).map(|resident| resident.slot)
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
/// runtime structural contract. Visual selection controls and operational
/// request concurrency remain live-updateable.
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
    max_gpu_upload_bytes_per_commit: u64,
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
            max_gpu_upload_bytes_per_commit: settings.budgets.max_gpu_upload_bytes_per_commit,
            max_encoded_page_bytes: streaming.effective_max_encoded_page_bytes(),
            debug_metadata,
        }
    }
}

struct BridgeCloudState {
    source: Handle<PlanarGaussian3d>,
    source_gaussian_count: u32,
    atlas: Handle<PlanarGaussian3d>,
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
    structural: StructuralSettings,
    signature: BridgeStructuralSignature,
    streaming: GaussianStreamingSettings,
    handshakes: BTreeMap<Entity, BridgeHandshake>,
    views: BTreeSet<Entity>,
    flat_source_bypass: bool,
    active: bool,
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
}

impl BridgeCloudState {
    fn owns_render_handle(&self, handle: AssetId<PlanarGaussian3d>) -> bool {
        handle == self.atlas.id() || (self.flat_source_bypass && handle == self.source.id())
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

    fn handshake_for(&mut self, camera: Entity, frontier: &LodCandidateFrontier) -> Arc<AtomicU8> {
        let draft = LodRenderCandidate::with_phase(
            frontier.clone(),
            Arc::new(AtomicU8::new(LOD_RENDER_WAITING)),
        );
        if let Some(handshake) = self.handshakes.get(&camera)
            && handshake.candidate.same_payload(&draft)
        {
            return Arc::clone(&handshake.phase);
        }
        if let Some(previous) = self.handshakes.remove(&camera) {
            // Render extraction may still retain the previous capability for a
            // frame. Revoking it before publishing the replacement forces the
            // complete flat fallback path during residency/frontier churn.
            previous.phase.store(LOD_RENDER_WAITING, Ordering::Release);
        }
        let phase = Arc::new(AtomicU8::new(LOD_RENDER_WAITING));
        self.handshakes.insert(
            camera,
            BridgeHandshake {
                candidate: LodRenderCandidate::with_phase(frontier.clone(), Arc::clone(&phase)),
                phase: Arc::clone(&phase),
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

#[derive(Resource, Default)]
struct GaussianLodBridgeManager {
    clouds: HashMap<Entity, BridgeCloudState>,
    blocked: HashMap<Entity, (AssetId<PlanarGaussian3d>, BridgeStructuralSignature)>,
}

#[derive(Clone, Copy)]
struct BridgeCameraView {
    entity: Entity,
    view: LodView,
}

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
                publish_bridge_status_transitions.after(GaussianLodBridgeUpdate),
            );
    }
}

fn publish_bridge_status_transitions(
    statuses: Query<(Entity, &GaussianLodBridgeStatus), Changed<GaussianLodBridgeStatus>>,
    mut removed: RemovedComponents<GaussianLodBridgeStatus>,
    mut previous: Local<
        HashMap<Entity, (GaussianLodBridgePhase, Option<LodOrchestrationFailureCode>)>,
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
        let kind = bridge_status_transition_kind(
            status.phase,
            status.failure.is_some(),
            old.is_some_and(|(_, failure)| failure.is_some()),
        );
        if let Some(kind) = kind {
            transitions.write(LodOrchestrationTransition {
                entity,
                source: LodOrchestrationSource::EphemeralBridge,
                kind,
                failure: status.failure.clone(),
            });
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
    } else if had_failure {
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
    mut asset_events: MessageReader<AssetEvent<PlanarGaussian3d>>,
    cameras: Query<(Entity, &Camera, &Projection, &GlobalTransform), With<GaussianCamera>>,
    mut clouds: Query<
        (
            Entity,
            &mut PlanarGaussian3dHandle,
            Option<&GaussianLodSettings>,
            Option<&GaussianStreamingSettings>,
            Option<&CloudSettings>,
            Option<&LodDebugMetadata>,
            Option<&ViewVisibility>,
            &GlobalTransform,
        ),
        Without<GaussianLodHandle>,
    >,
) {
    let config_error = config.validate_structure().err();
    let camera_views = collect_camera_views(&cameras, config.max_views_per_cloud);
    let (changed_assets, invalidated_assets, removed_assets) = asset_events.read().fold(
        (HashSet::new(), HashSet::new(), HashSet::new()),
        |(mut changed, mut invalidated, mut removed), event| {
            match event {
                AssetEvent::Added { id } => {
                    changed.insert(*id);
                }
                AssetEvent::Modified { id } | AssetEvent::LoadedWithDependencies { id } => {
                    changed.insert(*id);
                    invalidated.insert(*id);
                }
                AssetEvent::Removed { id } => {
                    changed.insert(*id);
                    invalidated.insert(*id);
                    removed.insert(*id);
                }
                AssetEvent::Unused { .. } => {}
            }
            (changed, invalidated, removed)
        },
    );
    let mut seen = BTreeSet::new();

    for (
        entity,
        mut handle,
        settings,
        per_cloud_streaming,
        cloud_settings,
        source_debug_metadata,
        view_visibility,
        cloud_transform,
    ) in &mut clouds
    {
        seen.insert(entity);
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
        let endpoint = settings.quality_endpoint();
        if endpoint == LodQualityEndpoint::Original {
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
            Ok(camera_views) => camera_views,
            Err(error) => {
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
        if view_visibility.is_some_and(|visibility| !visibility.get()) {
            let suspension = if let Some(state) = manager.clouds.get_mut(&entity) {
                let result = (|| {
                    if state.active {
                        restore_bridge_flat_fallback(state, &mut assets, &mut atlas_uploads)?;
                    }
                    suspend_bridge_runtime(state)
                })();
                state.active = false;
                state.invalidate_handshakes();
                result
            } else {
                Ok(())
            };
            commands.entity(entity).remove::<LodRenderCandidates>();
            match suspension {
                Ok(()) => {
                    commands.entity(entity).insert(GaussianLodBridgeStatus {
                        phase: GaussianLodBridgePhase::StreamingFallback,
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
            match create_ephemeral_bridge(
                source_handle.clone(),
                source,
                source_debug_metadata.cloned(),
                settings,
                &streaming,
                &config,
                debug_metadata,
            ) {
                Ok((mut state, atlas_cloud)) => {
                    state.atlas = assets.add(atlas_cloud);
                    *handle = PlanarGaussian3dHandle(state.atlas.clone());
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
                    restore_bridge_flat_fallback(&mut state, &mut assets, &mut atlas_uploads);
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

        state.streaming = streaming;
        let effective_settings = state.structural.apply(settings);
        let result = update_bridge_cloud(
            state,
            &effective_settings,
            cloud_transform,
            camera_views,
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
                if candidates.by_camera.is_empty() {
                    commands.entity(entity).remove::<LodRenderCandidates>();
                } else {
                    commands.entity(entity).insert(candidates);
                }
                commands.entity(entity).insert(status);
                state.publish_debug_metadata(entity, &mut commands);
            }
            Err(error) => {
                let retirement =
                    restore_bridge_flat_fallback(state, &mut assets, &mut atlas_uploads);
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
            let _ = restore_bridge_flat_fallback(&mut state, &mut assets, &mut atlas_uploads);
            state.invalidate_handshakes();
        }
        manager.blocked.remove(&entity);
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
        if let Err(error) = restore_bridge_flat_fallback(&mut state, assets, atlas_uploads) {
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
        if let Err(error) =
            restore_bridge_flat_fallback_after_source_change(&mut state, assets, atlas_uploads)
        {
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
    cameras: &Query<(Entity, &Camera, &Projection, &GlobalTransform), With<GaussianCamera>>,
    max_views: u32,
) -> Result<Vec<BridgeCameraView>, LodBridgeError> {
    let active = cameras
        .iter()
        .filter(|(_, camera, _, _)| camera.is_active)
        .collect::<Vec<_>>();
    if active.len() > max_views as usize {
        return Err(LodBridgeError::ViewLimitExceeded {
            actual: active.len() as u64,
            limit: max_views,
        });
    }
    let mut views = Vec::with_capacity(active.len());
    for (entity, camera, projection, transform) in active {
        let view = lod_view_from_camera(camera, projection, transform)
            .ok_or(LodBridgeError::UnsupportedCamera(entity))?;
        views.push(BridgeCameraView { entity, view });
    }
    views.sort_by_key(|view| view.entity);
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

fn create_ephemeral_bridge(
    source_handle: Handle<PlanarGaussian3d>,
    source: &PlanarGaussian3d,
    source_debug_metadata: Option<LodDebugMetadata>,
    settings: &GaussianLodSettings,
    streaming: &GaussianStreamingSettings,
    config: &GaussianLodBridgeConfig,
    debug_metadata: bool,
) -> Result<(BridgeCloudState, PlanarGaussian3d), LodBridgeError> {
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
    let mut built = build_planar_3d_lod(source, config.build_settings)
        .map_err(|error| LodBridgeError::Build(error.to_string()))?;
    if built.manifest.header.stored_gaussian_count > config.max_ephemeral_stored_gaussians {
        return Err(LodBridgeError::StoredGaussianLimit {
            actual: built.manifest.header.stored_gaussian_count,
            limit: config.max_ephemeral_stored_gaussians,
        });
    }

    let mut transport = MemoryPageTransport::default();
    for page in &built.pages {
        let encoded =
            encode_page(page).map_err(|error| LodBridgeError::Codec(error.to_string()))?;
        let descriptor = built
            .manifest
            .pages
            .iter_mut()
            .find(|descriptor| descriptor.id == page.id)
            .ok_or(LodBridgeError::MissingPageDescriptor(page.id))?;
        descriptor.storage = Some(
            crate::gaussian::formats::planar_3d_chunked::LodPageStorage {
                uri: format!("memory://ephemeral/{}", page.id.0),
                byte_range: None,
                encoded_len: encoded.len() as u64,
            },
        );
        transport.insert(page.id, encoded);
    }
    built
        .validate()
        .map_err(|error| LodBridgeError::Build(error.to_string()))?;

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
    let required_fallback_slots = source_count.div_ceil(stride);
    // One slot per unique manifest page is sufficient for every possible
    // camera cut: no selection can reference a page outside this manifest.
    // The initial flat fallback has a
    // different layout, so retain enough padded slots for it independently.
    // Capacity beyond both bounds can never carry useful data and only makes
    // the source-independent hard limits turn into unnecessary allocations.
    let maximum_useful_slots = required_fallback_slots.max(built.manifest.header.page_count);
    let required_fallback_gpu_bytes = u64::from(required_fallback_slots)
        .checked_mul(gpu_bytes_per_slot)
        .ok_or(LodBridgeError::AtlasSizeOverflow)?;
    if required_fallback_gpu_bytes > settings.budgets.max_gpu_upload_bytes_per_commit {
        return Err(LodBridgeError::CompleteFallbackExceedsGpuUploadBudget {
            required: required_fallback_gpu_bytes,
            limit: settings.budgets.max_gpu_upload_bytes_per_commit,
        });
    }
    let slots_by_count = config.max_atlas_gaussians / stride;
    let slots_by_bytes = (config.max_atlas_bytes / gpu_bytes_per_slot)
        .try_into()
        .unwrap_or(u32::MAX);
    let slots_by_commit = (settings.budgets.max_gpu_upload_bytes_per_commit / gpu_bytes_per_slot)
        .try_into()
        .unwrap_or(u32::MAX);
    let slot_count = settings
        .budgets
        .max_resident_pages
        .min(slots_by_count)
        .min(slots_by_bytes)
        .min(slots_by_commit)
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
    if physical_gaussians < source_count {
        return Err(LodBridgeError::AtlasCannotFitSource {
            source: source_count,
            physical: physical_gaussians,
        });
    }
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
    let debug_manifest_index = debug_metadata
        .then(|| LodDebugManifestIndex::new(&built.manifest))
        .transpose()
        .map_err(|error| LodBridgeError::DebugAnnotations(error.to_string()))?;
    let runtime = LodStreamingRuntime::new(built.manifest, transport, &effective, streaming)
        .map_err(LodBridgeError::Runtime)?;
    let mirror = LodPageAtlasMirror::new(runtime.atlas_layout(), slot_count)?;
    let debug_atlas = debug_metadata
        .then(|| LodDebugAnnotationAtlas::new(slot_count, stride))
        .transpose()
        .map_err(|error| LodBridgeError::DebugAnnotations(error.to_string()))?;
    let fallback_debug_metadata = source_debug_metadata.clone().unwrap_or_default();

    let mut fallback = vec![Gaussian3d::default(); physical_gaussians as usize];
    for (target, gaussian) in fallback.iter_mut().zip(source.iter()) {
        *target = gaussian;
    }
    debug_assert!(physical_gpu_bytes <= config.max_atlas_bytes);
    debug_assert!(physical_gpu_bytes <= settings.budgets.max_gpu_upload_bytes_per_commit);
    debug_assert!(slot_count <= maximum_useful_slots);
    Ok((
        BridgeCloudState {
            source: source_handle,
            source_gaussian_count: source_count,
            atlas: Handle::default(),
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
            structural,
            signature: BridgeStructuralSignature::new(settings, streaming, config, debug_metadata),
            streaming: streaming.clone(),
            handshakes: BTreeMap::new(),
            views: BTreeSet::new(),
            flat_source_bypass: false,
            active: false,
        },
        PlanarGaussian3d::from(fallback),
    ))
}

fn suspend_bridge_runtime(state: &mut BridgeCloudState) -> Result<(), LodBridgeError> {
    for view in std::mem::take(&mut state.views) {
        state
            .runtime
            .remove_view(runtime_view_id(view))
            .map_err(LodBridgeError::Runtime)?;
        if let Some(handshake) = state.handshakes.remove(&view) {
            handshake.phase.store(LOD_RENDER_WAITING, Ordering::Release);
        }
    }
    let frame = state.runtime.begin_frame();
    state
        .runtime
        .finish_frame(frame)
        .map_err(LodBridgeError::Runtime)?;
    Ok(())
}

fn retained_flat_source_bypass_is_eligible(
    source_gaussian_count: u32,
    active_camera_count: usize,
    debug_metadata_requested: bool,
    mut selected_counts: impl ExactSizeIterator<Item = u32>,
) -> bool {
    if debug_metadata_requested
        || source_gaussian_count == 0
        || active_camera_count == 0
        || selected_counts.len() != active_camera_count
    {
        return false;
    }

    let source = u64::from(source_gaussian_count);
    selected_counts.all(|selected| {
        u64::from(selected) * FLAT_SOURCE_BYPASS_MIN_SELECTED_DENOMINATOR
            >= source * FLAT_SOURCE_BYPASS_MIN_SELECTED_NUMERATOR
    })
}

fn update_bridge_cloud(
    state: &mut BridgeCloudState,
    settings: &GaussianLodSettings,
    cloud_transform: &GlobalTransform,
    camera_views: &[BridgeCameraView],
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(LodRenderCandidates, GaussianLodBridgeStatus), LodBridgeError> {
    let next_views = camera_views
        .iter()
        .map(|view| view.entity)
        .collect::<BTreeSet<_>>();
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
            handshake.phase.store(LOD_RENDER_WAITING, Ordering::Release);
        }
    }
    let view_set_changed = state.views != next_views;
    state.views = next_views;

    let frame = state.runtime.begin_frame();
    let mut frames = Vec::with_capacity(camera_views.len());
    let world_from_local = cloud_transform.to_matrix();
    for camera in camera_views {
        let stream_frame = state
            .runtime
            .update_view_in_frame(
                frame,
                runtime_view_id(camera.entity),
                camera.view.with_world_from_local(world_from_local),
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

    let fallback_nodes = selected_ancestor_fallback_nodes(&*state.runtime, &frames);
    let mut frontiers = Vec::with_capacity(frames.len());
    for (camera, stream_frame) in &frames {
        let frontier = match stream_frame.candidate_frontier(settings.max_active_gaussians_u32()) {
            Ok(frontier) => frontier,
            Err(LodRuntimeError::NoResidentFrontier) => continue,
            Err(error) => return Err(LodBridgeError::Runtime(error)),
        };
        frontiers.push((*camera, frontier));
    }

    let use_retained_flat_source = retained_flat_source_bypass_is_eligible(
        state.source_gaussian_count,
        camera_views.len(),
        state.debug_atlas.is_some(),
        frontiers
            .iter()
            .map(|(_, frontier)| frontier.candidate_count()),
    );

    if use_retained_flat_source {
        // A complete cut that would remove at most five percent of the source
        // cannot justify compaction plus a second radix pass. Keep its logical
        // selection/freeze provenance, but render the exact retained source and
        // report that actual source count and quality. Debug annotations retain
        // the atlas path so every selected hierarchy record stays inspectable.
        if state.active {
            restore_bridge_flat_fallback(state, assets, atlas_uploads)?;
            state.active = false;
        }
        state.invalidate_handshakes();
        state.handshakes.clear();
        state.flat_source_bypass = true;
        let mut render_candidates = LodRenderCandidates::default();
        for (camera, frontier) in frontiers {
            render_candidates.by_camera.insert(
                camera,
                LodRenderCandidate::retained_flat_source(frontier, state.source_gaussian_count),
            );
        }
        let resident_pages = frames
            .last()
            .map(|(_, frame)| frame.cache_stats().resident_pages)
            .unwrap_or(0);
        return Ok((
            render_candidates,
            GaussianLodBridgeStatus {
                phase: GaussianLodBridgePhase::Active,
                active_views: camera_views.len().try_into().unwrap_or(u32::MAX),
                resident_pages,
                active_gaussians: u64::from(state.source_gaussian_count),
                failure: None,
            },
        ));
    }

    if state.flat_source_bypass {
        // The atlas has remained a complete flat fallback while bypassed, so
        // it is safe to expose again while a new compacted cut handshakes.
        state.flat_source_bypass = false;
        state.invalidate_handshakes();
        state.handshakes.clear();
    }

    let mut render_candidates = LodRenderCandidates::default();
    for (camera, frontier) in frontiers {
        let phase = state.handshake_for(camera, &frontier);
        render_candidates
            .by_camera
            .insert(camera, LodRenderCandidate::with_phase(frontier, phase));
    }

    let complete_camera_set =
        !camera_views.is_empty() && render_candidates.by_camera.len() == camera_views.len();
    let debug_fallback_nodes = fallback_nodes;
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
    let any_failed = render_candidates
        .by_camera
        .values()
        .any(|candidate| candidate.phase.load(Ordering::Acquire) == LOD_RENDER_FAILED);
    if any_failed {
        return Err(LodBridgeError::RenderCommitFailed);
    }

    if state.active && (view_set_changed || !complete_camera_set || !every_active) {
        restore_bridge_flat_fallback(state, assets, atlas_uploads)?;
        state.active = false;
        for handshake in state.handshakes.values() {
            handshake.phase.store(LOD_RENDER_WAITING, Ordering::Release);
        }
    }

    if state.active || every_prepared {
        validate_gpu_upload_commit_budget(
            state,
            &render_candidates,
            settings.budgets.max_gpu_upload_bytes_per_commit,
        )?;
    }

    if !state.active && every_prepared {
        let atlas_id = state.atlas.id();
        let atlas = assets
            .get_mut_untracked(&state.atlas)
            .ok_or(LodBridgeError::MissingAtlasAsset)?;
        synchronize_bridge_candidate_pages(
            state,
            &render_candidates,
            &debug_fallback_nodes,
            atlas_id,
            atlas,
            atlas_uploads,
        )?;
        for candidate in render_candidates.by_camera.values() {
            candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
        }
        state.active = true;
    } else if state.active {
        let atlas_id = state.atlas.id();
        let atlas = assets
            .get_mut_untracked(&state.atlas)
            .ok_or(LodBridgeError::MissingAtlasAsset)?;
        synchronize_bridge_candidate_pages(
            state,
            &render_candidates,
            &debug_fallback_nodes,
            atlas_id,
            atlas,
            atlas_uploads,
        )?;
    }

    let resident_pages = frames
        .last()
        .map(|(_, frame)| frame.cache_stats().resident_pages)
        .unwrap_or(0);
    let active_gaussians = render_candidates
        .by_camera
        .values()
        .map(|candidate| u64::from(candidate.rendered_candidate_count()))
        .max()
        .unwrap_or(0);
    let phase = if state.active {
        GaussianLodBridgePhase::Active
    } else if complete_camera_set {
        GaussianLodBridgePhase::WaitingForRender
    } else {
        GaussianLodBridgePhase::StreamingFallback
    };
    Ok((
        render_candidates,
        GaussianLodBridgeStatus {
            phase,
            active_views: camera_views.len().try_into().unwrap_or(u32::MAX),
            resident_pages,
            active_gaussians,
            failure: None,
        },
    ))
}

/// Synchronizes each physical page at most once even when several logical
/// sibling ranges or cameras reference the same slot.
fn synchronize_bridge_candidate_pages(
    state: &mut BridgeCloudState,
    candidates: &LodRenderCandidates,
    fallback_nodes: &BTreeSet<LodNodeId>,
    atlas_id: AssetId<PlanarGaussian3d>,
    atlas: &mut PlanarGaussian3d,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), LodBridgeError> {
    let pages = candidates
        .by_camera
        .values()
        .flat_map(LodRenderCandidate::render_ranges)
        .map(|range| (range.page, range.slot))
        .collect::<BTreeSet<_>>();
    let gaussians_per_slot = state.mirror.layout().gaussians_per_slot;

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
            state.mirror.materialize_page(atlas, &page, slot)?;
            atlas_uploads
                .enqueue_slot(atlas_id, slot, gaussians_per_slot)
                .map_err(|error| LodBridgeError::AtlasUpload(error.to_string()))?;
        }
    }
    for candidate in candidates.by_camera.values() {
        state.mirror.validate_ranges(candidate.render_ranges())?;
    }
    Ok(())
}

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
    let bytes_per_slot = u64::from(state.mirror.layout().gaussians_per_slot)
        .checked_mul(gaussian_3d_gpu_bytes_per_record())
        .ok_or(LodBridgeError::AtlasSizeOverflow)?;
    (dirty_slots.len() as u64)
        .checked_mul(bytes_per_slot)
        .ok_or(LodBridgeError::AtlasSizeOverflow)
}

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

fn restore_bridge_flat_fallback(
    state: &mut BridgeCloudState,
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), LodBridgeError> {
    let source = assets
        .get(&state.source)
        .ok_or(LodBridgeError::MissingSourceAsset)?
        .clone();
    let atlas = assets
        .get_mut_untracked(&state.atlas)
        .ok_or(LodBridgeError::MissingAtlasAsset)?;
    restore_complete_flat_fallback(atlas, &source)?;
    let gaussians_per_slot = state.mirror.layout().gaussians_per_slot;
    for slot in state.mirror.materialized_slots() {
        atlas_uploads
            .enqueue_slot(state.atlas.id(), slot, gaussians_per_slot)
            .map_err(|error| LodBridgeError::AtlasUpload(error.to_string()))?;
    }
    state.mirror.mark_fallback_materialized();
    Ok(())
}

fn restore_bridge_flat_fallback_after_source_change(
    state: &mut BridgeCloudState,
    assets: &mut Assets<PlanarGaussian3d>,
    atlas_uploads: &mut LodAtlasUploadQueue,
) -> Result<(), LodBridgeError> {
    let source = assets
        .get(&state.source)
        .ok_or(LodBridgeError::MissingSourceAsset)?
        .clone();
    let atlas = assets
        .get_mut_untracked(&state.atlas)
        .ok_or(LodBridgeError::MissingAtlasAsset)?;
    restore_complete_flat_fallback(atlas, &source)?;
    atlas_uploads
        .enqueue_complete_atlas(
            state.atlas.id(),
            state.mirror.physical_gaussians(),
            state.mirror.layout().gaussians_per_slot,
        )
        .map_err(|error| LodBridgeError::AtlasUpload(error.to_string()))?;
    state.mirror.mark_fallback_materialized();
    Ok(())
}

fn restore_complete_flat_fallback(
    atlas: &mut PlanarGaussian3d,
    source: &PlanarGaussian3d,
) -> Result<(), LodBridgeError> {
    if source.len() > atlas.len() {
        return Err(LodBridgeError::FallbackExceedsAtlas {
            source: source.len() as u64,
            atlas: atlas.len() as u64,
        });
    }
    for index in 0..atlas.len() {
        Planar::set(atlas, index, Gaussian3d::default());
    }
    for (index, gaussian) in source.iter().enumerate() {
        Planar::set(atlas, index, gaussian);
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
