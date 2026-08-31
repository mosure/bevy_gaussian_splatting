#![allow(dead_code)] // ShaderType derives emit unused check helpers
#[cfg(feature = "precompute_covariance_3d")]
use std::any::TypeId;
use std::{borrow::Cow, hash::Hash, num::NonZero};

use bevy::render::render_resource::TextureFormat;
use bevy::shader::ShaderDefVal;
use bevy::{
    asset::{AssetEvent, AssetId, load_internal_asset, uuid_handle},
    camera::primitives::Aabb,
    core_pipeline::{
        core_3d::{Transparent3d, TransparentSortingInfo3d},
        prepass::{
            MotionVectorPrepass, PreviousViewData, PreviousViewUniformOffset, PreviousViewUniforms,
        },
    },
    ecs::{
        query::ROQueryItem,
        system::{SystemParamItem, lifetimeless::*},
    },
    pbr::PrepassViewBindGroup,
    prelude::*,
    render::{
        Extract, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{ComponentUniforms, DynamicUniformIndex, UniformComponentPlugin},
        globals::{GlobalsBuffer, GlobalsUniform},
        init_gpu_resource,
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        sync_world::RenderEntity,
        view::{
            ExtractedView, RenderVisibilityRanges, RenderVisibleEntities,
            VISIBILITY_RANGES_STORAGE_BUFFER_COUNT, ViewUniform, ViewUniformOffset, ViewUniforms,
        },
    },
};
use bevy_interleave::prelude::*;

#[cfg(feature = "buffer_storage")]
use crate::sort::SortEntry;
use crate::{
    camera::GaussianCamera,
    gaussian::{
        cloud::CloudVisibilityClass,
        formats::planar_3d_chunked::LodPageId,
        interface::CommonCloud,
        lod_debug::{LodDebugMetadata, LodDebugRecord, LodDebugSettings, LodDebugSparseMetadata},
        lod_settings::{GaussianLodSettings, LodQualityTarget},
        settings::{
            CloudSettings, DrawMode, GaussianColorSpace, GaussianMode, RadixSortDepthBits,
            RasterizeMode,
        },
    },
    material::{
        spherical_harmonics::{HALF_SH_COEFF_COUNT, SH_COEFF_COUNT, SH_DEGREE, SH_VEC4_PLANES},
        spherindrical_harmonics::{SH_4D_COEFF_COUNT, SH_4D_DEGREE_TIME},
    },
    morph::MorphPlugin,
    sort::{GpuSortedEntry, SortPlugin, SortTrigger, SortedEntriesHandle, sort_entry_binding_size},
};

#[cfg(lod_render_path)]
use crate::stream::render_commit::LodRenderCandidates;

#[cfg(feature = "morph_interpolate")]
use crate::morph::interpolate::GaussianInterpolateBindGroups;
#[cfg(feature = "morph_particles")]
use crate::morph::particle::ParticleBehaviorBindGroup;
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
use crate::sort::radix::RadixBindGroup;

#[cfg(feature = "packed")]
mod packed;

#[cfg(feature = "buffer_storage")]
mod planar;

#[cfg(lod_render_path)]
pub mod lod;
#[cfg(feature = "lod")]
pub mod recovery;
#[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
mod texture;

const BINDINGS_SHADER_HANDLE: Handle<Shader> = uuid_handle!("cfd9a3d9-a0cb-40c8-ab0b-073110a02474");
const GAUSSIAN_SHADER_HANDLE: Handle<Shader> = uuid_handle!("9a18d83b-137d-4f44-9628-e2defc4b62b0");
const GAUSSIAN_2D_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("713fb941-b4f5-408e-bbde-32fb7dc447ce");
const GAUSSIAN_3D_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("b7eb322b-983b-4ce0-a5a2-3c0d6cb06d65");
const GAUSSIAN_4D_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("26234995-0932-4dfa-ab8d-53df1e779dd4");
const HELPERS_SHADER_HANDLE: Handle<Shader> = uuid_handle!("9ca57ab0-07de-4a43-94f8-547c38e292cb");
const LOD_DEBUG_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("4d449c34-2e1e-48c4-9561-d04fed7c5f2b");
const LOD_MORPH_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("872c7fd3-9f1f-4ad5-81dc-1cf33090e715");
const PACKED_SHADER_HANDLE: Handle<Shader> = uuid_handle!("5bb62086-7004-4575-9972-274dc8acccf1");
const PLANAR_SHADER_HANDLE: Handle<Shader> = uuid_handle!("d6a3f978-f795-4786-8475-26366f28d852");
const TEXTURE_SHADER_HANDLE: Handle<Shader> = uuid_handle!("500e2ebf-51a8-402e-9c88-e0d5152c3486");
const TRANSFORM_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("648516b2-87cc-4937-ae1c-d986952e9fa7");

/// Intended 3DGS/Mip-style screen-space variance in physical-pixel units. The
/// WGSL projection uses two internal coordinate units per physical pixel and
/// therefore adds four times this value before converting its bounds back to
/// NDC. That multiplication corrects the prior shader's accidental 0.075
/// physical-pixel variance; determinant normalization preserves integrated
/// alpha while the corrected footprint changes the old projected extent.
pub const GAUSSIAN_MIP_FILTER_VARIANCE_2D: f32 = 0.3;

/// Authored Gaussian support used by hierarchy representatives. A candidate
/// LoD cut must not shrink this support as opacity decreases because a merged
/// representative's low peak can still encode substantial spatial mass.
pub const GAUSSIAN_AUTHORED_SUPPORT_SIGMA: f32 = 3.0;

const GAUSSIAN_OPACITY_RADIUS_LOG_FLOOR: f32 = 0.000_001;

/// A filtered screen-space covariance and the peak-opacity multiplier that
/// preserves its integrated alpha over the plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianMipFilter2d {
    pub covariance: [f32; 3],
    pub opacity_scale: f32,
}

/// Applies the renderer's 0.3-pixel covariance footprint with determinant
/// normalization. For a valid area covariance,
/// `opacity_scale * sqrt(det(filtered)) == sqrt(det(original))`.
#[inline]
pub fn gaussian_mip_filter_covariance_2d(covariance: [f32; 3]) -> GaussianMipFilter2d {
    let filtered = [
        covariance[0] + GAUSSIAN_MIP_FILTER_VARIANCE_2D,
        covariance[1],
        covariance[2] + GAUSSIAN_MIP_FILTER_VARIANCE_2D,
    ];
    let original_determinant = covariance[0] * covariance[2] - covariance[1] * covariance[1];
    let filtered_determinant = filtered[0] * filtered[2] - filtered[1] * filtered[1];
    let determinant_ratio = original_determinant / filtered_determinant;
    let opacity_scale =
        if original_determinant > 0.0 && filtered_determinant > 0.0 && determinant_ratio >= 0.0 {
            determinant_ratio.clamp(0.0, 1.0).sqrt()
        } else {
            0.0
        };

    GaussianMipFilter2d {
        covariance: filtered,
        opacity_scale,
    }
}

/// Returns the raster support cutoff while preserving the historical adaptive
/// flat-cloud policy. LoD candidates always use their authored 3-sigma support:
/// this matches offline support accounting and GPU compaction for every finite
/// portable opacity, including values greater than one.
#[inline]
pub fn gaussian_support_cutoff(
    opacity: f32,
    opacity_adaptive_radius: bool,
    lod_candidate: bool,
) -> f32 {
    if lod_candidate || !opacity_adaptive_radius {
        return GAUSSIAN_AUTHORED_SUPPORT_SIGMA;
    }
    (GAUSSIAN_AUTHORED_SUPPORT_SIGMA * GAUSSIAN_AUTHORED_SUPPORT_SIGMA
        + 2.0 * opacity.max(GAUSSIAN_OPACITY_RADIUS_LOG_FLOOR).ln())
    .max(GAUSSIAN_OPACITY_RADIUS_LOG_FLOOR)
    .sqrt()
}

// TODO: consider refactor to bind via bevy's mesh (dynamic vertex planes) + shared batching/instancing/preprocessing
//       utilize RawBufferVec<T> for gaussian data?
pub struct RenderPipelinePlugin<R: PlanarSync> {
    _phantom: std::marker::PhantomData<R>,
}

/// Recovery-startup ordering boundary for resources whose constructors read a
/// specialized Gaussian cloud pipeline from the render world.
#[derive(SystemSet, Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CloudPipelineReady;

impl<R: PlanarSync> Default for RenderPipelinePlugin<R> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<R: PlanarSync> Plugin for RenderPipelinePlugin<R>
where
    R::PlanarType: CommonCloud,
    R::GpuPlanarType: GpuPlanarStorage,
    <R::GpuPlanarType as GpuPlanar>::PackedType: ReflectInterleaved,
{
    fn build(&self, app: &mut App) {
        debug!("building render pipeline plugin");

        app.add_plugins(MorphPlugin::<R>::default());
        app.add_plugins(SortPlugin::<R>::default());
        #[cfg(lod_render_path)]
        app.add_plugins(lod::LodCompactionPlugin::<R>::default());
        app.init_resource::<PlanarStorageRebindQueue<R>>();
        app.add_systems(PostUpdate, queue_planar_storage_rebinds::<R>);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Transparent3d, DrawGaussians<R>>()
                .init_gpu_resource::<GaussianUniformBindGroups>()
                .init_resource::<LodDebugGpuUploadStats>()
                .init_resource::<PlanarStorageRebindQueue<R>>()
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_gaussians::<R>,
                        extract_planar_storage_rebind_queue::<R>,
                    ),
                )
                .add_systems(
                    Render,
                    (
                        refresh_planar_storage_bind_groups::<R>
                            .in_set(RenderSystems::PrepareBindGroups),
                        queue_gaussian_bind_group::<R>.in_set(RenderSystems::PrepareBindGroups),
                        // Queue specializes the pipeline from debug readiness,
                        // while the draw command later reads the same binding.
                        // Prepare it after every GPU asset producer but before
                        // Queue so both observe one immutable readiness epoch.
                        prepare_lod_debug_bind_group::<R>
                            .after(RenderSystems::PrepareAssets)
                            .before(RenderSystems::Queue),
                        queue_gaussian_view_bind_groups::<R>
                            .in_set(RenderSystems::PrepareBindGroups),
                        queue_gaussian_compute_view_bind_groups::<R>
                            .in_set(RenderSystems::PrepareBindGroups),
                        queue_gaussians::<R>.in_set(RenderSystems::Queue),
                    ),
                );
        }

        // TODO: refactor common resources into a common plugin
        if app.is_plugin_added::<UniformComponentPlugin<CloudUniform>>() {
            debug!("render plugin already added");
            return;
        }

        load_internal_asset!(
            app,
            BINDINGS_SHADER_HANDLE,
            "bindings.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            GAUSSIAN_SHADER_HANDLE,
            "gaussian.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            GAUSSIAN_2D_SHADER_HANDLE,
            "gaussian_2d.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            GAUSSIAN_3D_SHADER_HANDLE,
            "gaussian_3d.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            GAUSSIAN_4D_SHADER_HANDLE,
            "gaussian_4d.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            HELPERS_SHADER_HANDLE,
            "helpers.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            LOD_DEBUG_SHADER_HANDLE,
            "lod_debug.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            LOD_MORPH_SHADER_HANDLE,
            "lod_morph.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(app, PACKED_SHADER_HANDLE, "packed.wgsl", Shader::from_wgsl);

        load_internal_asset!(app, PLANAR_SHADER_HANDLE, "planar.wgsl", Shader::from_wgsl);

        load_internal_asset!(
            app,
            TEXTURE_SHADER_HANDLE,
            "texture.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            TRANSFORM_SHADER_HANDLE,
            "transform.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(UniformComponentPlugin::<CloudUniform>::default());

        #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
        app.add_plugins(texture::BufferTexturePlugin);
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_systems(
                    RenderStartup,
                    (
                        invalidate_gaussian_gpu_component_caches::<R>
                            .before(CloudPipelineReady)
                            .ambiguous_with_all(),
                        init_gpu_resource::<PlanarStorageLayouts<R>>
                            .before(CloudPipelineReady)
                            .ambiguous_with_all(),
                        init_gpu_resource::<CloudPipeline<R>>
                            .in_set(CloudPipelineReady)
                            .ambiguous_with_all(),
                    ),
                )
                .init_gpu_resource::<SpecializedRenderPipelines<CloudPipeline<R>>>();
        }
    }
}

/// Drops every Gaussian render-world component that directly retains a device
/// object. Render entities survive Bevy's device replacement, so leaving any
/// one of these components in place can make an otherwise recovery-aware queue
/// system reuse a bind group from the old wgpu device.
fn invalidate_gaussian_gpu_component_caches<R: PlanarSync>(world: &mut World) {
    fn remove_all<T: Component>(world: &mut World) -> usize {
        let entities = {
            let mut query = world.query_filtered::<Entity, With<T>>();
            query.iter(world).collect::<Vec<_>>()
        };
        let count = entities.len();
        for entity in entities {
            world.entity_mut(entity).remove::<T>();
        }
        count
    }

    let mut removed = 0;
    removed += remove_all::<PlanarStorageBindGroup<R>>(world);
    removed += remove_all::<PlanarStorageBoundAsset<R>>(world);
    removed += remove_all::<SortBindGroup>(world);
    removed += remove_all::<LodDebugBindGroup<R>>(world);
    removed += remove_all::<GaussianViewBindGroup>(world);
    removed += remove_all::<GaussianComputeViewBindGroup>(world);

    #[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
    {
        removed += remove_all::<RadixBindGroup>(world);
    }
    #[cfg(feature = "morph_interpolate")]
    {
        removed += remove_all::<GaussianInterpolateBindGroups<R>>(world);
    }
    #[cfg(feature = "morph_particles")]
    {
        removed += remove_all::<ParticleBehaviorBindGroup>(world);
    }

    debug!(
        component_count = removed,
        planar_type = %std::any::type_name::<R::PlanarType>(),
        "invalidated Gaussian GPU component caches for render-device startup"
    );
}

#[derive(Resource)]
pub struct PlanarStorageRebindQueue<R: PlanarSync> {
    handles: Vec<AssetId<R::PlanarType>>,
    marker: std::marker::PhantomData<R>,
}

/// Asset identity of the GPU storage currently bound to one render entity.
///
/// Asset events alone cannot establish this identity: an entity may swap back
/// to an already-loaded handle without producing a new event for that asset.
/// Keeping the identity beside the bind group prevents the old asset's GPU
/// storage from surviving such a handle change.
#[derive(Component)]
struct PlanarStorageBoundAsset<R: PlanarSync> {
    id: AssetId<R::PlanarType>,
    marker: std::marker::PhantomData<fn() -> R>,
}

#[allow(type_alias_bounds)]
type PlanarStorageBindingQuery<'w, R: PlanarSync> = (
    Entity,
    &'w R::PlanarTypeHandle,
    Option<&'w PlanarStorageBindGroup<R>>,
    Option<&'w PlanarStorageBoundAsset<R>>,
);

const fn planar_storage_binding_needs_refresh(
    has_bind_group: bool,
    identity_matches: bool,
    asset_was_queued: bool,
) -> bool {
    !has_bind_group || !identity_matches || asset_was_queued
}

impl<R: PlanarSync> Default for PlanarStorageRebindQueue<R> {
    fn default() -> Self {
        Self {
            handles: Vec::new(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<R: PlanarSync> Clone for PlanarStorageRebindQueue<R> {
    fn clone(&self) -> Self {
        Self {
            handles: self.handles.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<R: PlanarSync> PlanarStorageRebindQueue<R> {
    pub fn push_unique(&mut self, id: AssetId<R::PlanarType>) {
        if !self.handles.contains(&id) {
            self.handles.push(id);
        }
    }

    pub(crate) fn contains(&self, id: AssetId<R::PlanarType>) -> bool {
        self.handles.contains(&id)
    }
}

fn queue_planar_storage_rebinds<R: PlanarSync>(
    mut events: MessageReader<AssetEvent<R::PlanarType>>,
    mut queue: ResMut<PlanarStorageRebindQueue<R>>,
) {
    for event in events.read() {
        match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id } => {
                queue.push_unique(*id);
            }
            AssetEvent::Removed { id } => {
                queue.handles.retain(|handle_id| handle_id != id);
            }
            AssetEvent::Unused { .. } => {}
        }
    }
}

fn extract_planar_storage_rebind_queue<R: PlanarSync>(
    mut commands: Commands,
    mut main_world: ResMut<bevy::render::MainWorld>,
) {
    let mut queue = main_world.resource_mut::<PlanarStorageRebindQueue<R>>();
    commands.insert_resource(queue.clone());
    queue.handles.clear();
}

fn refresh_planar_storage_bind_groups<R: PlanarSync>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    gpu_planars: Res<RenderAssets<R::GpuPlanarType>>,
    bind_group_layouts: Res<bevy_interleave::interface::storage::PlanarStorageLayouts<R>>,
    mut queue: ResMut<PlanarStorageRebindQueue<R>>,
    query: Query<PlanarStorageBindingQuery<'_, R>>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    let layout = &bind_group_layouts.bind_group_layout;
    let queued = std::mem::take(&mut queue.handles);

    for (entity, planar_handle, existing, bound_asset) in &query {
        let id = planar_handle.handle().id();
        let identity_matches = bound_asset.is_some_and(|bound| bound.id == id);
        let Some(gpu_planar) = gpu_planars.get(planar_handle.handle()) else {
            // Neither a handle swap nor an in-place asset replacement may
            // keep rendering through storage from the previous GPU asset.
            // Removing the identity also guarantees a later GPU-asset
            // appearance triggers reconstruction without another AssetEvent.
            commands
                .entity(entity)
                .remove::<PlanarStorageBindGroup<R>>()
                .remove::<PlanarStorageBoundAsset<R>>();
            continue;
        };
        if !planar_storage_binding_needs_refresh(
            existing.is_some(),
            identity_matches,
            queued.contains(&id),
        ) {
            continue;
        }
        let bind_group = gpu_planar.bind_group(render_device.as_ref(), layout);

        commands.entity(entity).insert((
            PlanarStorageBindGroup::<R> {
                bind_group,
                phantom: std::marker::PhantomData,
            },
            PlanarStorageBoundAsset::<R> {
                id,
                marker: std::marker::PhantomData,
            },
        ));
    }
}

#[derive(Bundle)]
pub struct GpuCloudBundle<R: PlanarSync> {
    pub aabb: Aabb,
    pub settings: CloudSettings,
    pub settings_uniform: CloudUniform,
    pub sorted_entries: SortedEntriesHandle,
    pub cloud_handle: R::PlanarTypeHandle,
    pub transform: GlobalTransform,
}

#[cfg(lod_render_path)]
#[allow(type_alias_bounds)]
type GpuCloudBundleQuery<R: bevy_interleave::prelude::PlanarSync> = (
    Entity,
    &'static <R as bevy_interleave::prelude::PlanarSync>::PlanarTypeHandle,
    &'static Aabb,
    &'static SortedEntriesHandle,
    &'static CloudSettings,
    Option<&'static GaussianLodSettings>,
    &'static GlobalTransform,
    Option<&'static LodDebugBindGroup<R>>,
    Option<&'static LodRenderCandidates>,
);

#[cfg(not(lod_render_path))]
#[allow(type_alias_bounds)]
type GpuCloudBundleQuery<R: bevy_interleave::prelude::PlanarSync> = (
    Entity,
    &'static <R as bevy_interleave::prelude::PlanarSync>::PlanarTypeHandle,
    &'static Aabb,
    &'static SortedEntriesHandle,
    &'static CloudSettings,
    Option<&'static GaussianLodSettings>,
    &'static GlobalTransform,
    Option<&'static LodDebugBindGroup<R>>,
);

#[allow(type_alias_bounds)]
type GpuCloudBindGroupQuery<R: bevy_interleave::prelude::PlanarSync> = (
    Entity,
    &'static <R as bevy_interleave::prelude::PlanarSync>::PlanarTypeHandle,
    &'static SortedEntriesHandle,
    Option<&'static SortBindGroup>,
);

#[derive(Component)]
pub struct LodDebugBindGroup<R: PlanarSync> {
    bind_group: BindGroup,
    _buffer: Buffer,
    _config_buffer: Buffer,
    // Retain the immutable allocation used for identity comparison. Without
    // this, an allocator could reuse a dropped metadata pointer and make a new
    // snapshot look unchanged.
    _source_metadata: Option<LodDebugMetadata>,
    source_pointer: usize,
    sparse_identity: Option<u64>,
    sparse_slot_invariant_revisions: Vec<u64>,
    sparse_slot_payload_revisions: Vec<u64>,
    record_count: usize,
    ready: bool,
    current_invariant_ready: bool,
    current_payload_ready: bool,
    pending_invariants_ready: bool,
    upload_complete: bool,
    settings: LodDebugSettings,
    lod_settings: Option<GaussianLodSettings>,
    marker: std::marker::PhantomData<fn() -> R>,
}

/// Render-frame ownership of the compaction output relative to the extracted
/// debug sidecar. LoD render preparation updates this after candidate commit
/// decisions and before annotation bind-group preparation.
#[derive(Component, Clone, Debug, Default)]
pub(crate) struct LodDebugCandidateEpoch {
    pub(crate) candidates_are_current: bool,
    pub(crate) retained_current: bool,
    pub(crate) debug_metadata_staged: bool,
    pub(crate) pending_candidate_active: bool,
    pub(crate) pending_activation_armed: bool,
    pub(crate) required_slots: Vec<(LodPageId, u32, u32)>,
}

#[cfg(feature = "testing")]
impl<R: PlanarSync> LodDebugBindGroup<R> {
    pub const fn preset_for_testing(&self) -> crate::LodDebugPreset {
        self.settings.preset
    }

    pub const fn ready_for_testing(&self) -> bool {
        self.ready
    }

    pub const fn sparse_identity_for_testing(&self) -> Option<u64> {
        self.sparse_identity
    }

    pub const fn record_count_for_testing(&self) -> usize {
        self.record_count
    }
}

#[cfg(lod_render_path)]
impl<R: PlanarSync> LodDebugBindGroup<R> {
    pub(crate) fn candidate_invariants_ready(
        &self,
        metadata: Option<&LodDebugMetadata>,
        candidates: &LodRenderCandidates,
    ) -> bool {
        let Some(metadata) = metadata else {
            return false;
        };
        if let Some(sparse) = metadata.sparse() {
            lod_debug_sparse_identity_matches(self.sparse_identity, sparse.identity())
                && lod_debug_candidate_revisions_ready(
                    sparse,
                    candidates,
                    &self.sparse_slot_invariant_revisions,
                    LodDebugRevisionKind::Invariant,
                )
        } else {
            let records = metadata.records();
            let source_pointer = if records.is_empty() {
                0
            } else {
                records.as_ptr() as usize
            };
            self.sparse_identity.is_none() && self.source_pointer == source_pointer && self.ready
        }
    }
}

#[cfg(lod_render_path)]
pub(crate) const fn lod_debug_sparse_identity_matches(
    binding_identity: Option<u64>,
    metadata_identity: u64,
) -> bool {
    matches!(binding_identity, Some(identity) if identity == metadata_identity)
}

/// Cumulative render-world counters for LoD debug allocation and upload work.
///
/// These counters are intentionally monotonic so headless qualification can
/// sample deltas around a camera, preset, or quality mutation without relying
/// on wall-clock timing. A config-only change must advance
/// [`Self::config_bytes_written`] without advancing record bytes or record
/// buffer allocations.
#[derive(Resource, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LodDebugGpuUploadStats {
    record_buffer_allocations: u64,
    record_bytes_written: u64,
    config_bytes_written: u64,
    ready_bind_group_queues: u64,
    specialized_pipeline_queues: u64,
}

impl LodDebugGpuUploadStats {
    pub const fn record_buffer_allocations(self) -> u64 {
        self.record_buffer_allocations
    }

    pub const fn record_bytes_written(self) -> u64 {
        self.record_bytes_written
    }

    pub const fn config_bytes_written(self) -> u64 {
        self.config_bytes_written
    }

    /// Visible cloud/view queues that had a requested and upload-ready debug
    /// binding. This is a render-world diagnostic, not a draw-count estimate.
    pub const fn ready_bind_group_queues(self) -> u64 {
        self.ready_bind_group_queues
    }

    /// Visible cloud/view queues whose specialized pipeline key enabled the
    /// `LOD_DEBUG` shader definition.
    pub const fn specialized_pipeline_queues(self) -> u64 {
        self.specialized_pipeline_queues
    }

    pub const fn max_sparse_record_bytes_per_frame() -> u64 {
        LOD_DEBUG_MAX_SPARSE_UPLOAD_BYTES_PER_FRAME
    }

    pub const fn max_sparse_record_slots_per_frame() -> usize {
        LOD_DEBUG_MAX_SPARSE_UPLOAD_SLOTS_PER_FRAME
    }

    pub const fn config_bytes_per_write() -> u64 {
        std::mem::size_of::<LodDebugGpuUniform>() as u64
    }
}

#[allow(type_alias_bounds)]
type LodDebugPrepareQuery<R: PlanarSync> = (
    Entity,
    &'static R::PlanarTypeHandle,
    &'static CloudSettings,
    Option<&'static GaussianLodSettings>,
    Option<&'static LodDebugMetadata>,
    Option<&'static LodDebugCandidateEpoch>,
    Option<&'static mut LodDebugBindGroup<R>>,
);

/// Group-4 uniform. Keeping it separate preserves the baseline CloudUniform
/// ABI exactly when annotations are disabled.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct LodDebugGpuUniform {
    flags: [u32; 4],
    /// Error target px, detail fraction, and two reserved lanes.
    /// A negative error target means that no LoD policy was extracted.
    quality_params: [f32; 4],
}

impl LodDebugGpuUniform {
    fn new(
        settings: &LodDebugSettings,
        lod_settings: Option<&GaussianLodSettings>,
        metadata_count: usize,
    ) -> Self {
        let quality_params = lod_settings.map_or([-1.0, 0.0, 0.0, 0.0], |lod| {
            // Endpoints use finite sentinels so the uniform remains portable:
            // coarsest is an unbounded error target with zero structural
            // demand, while original is an exact zero-error target at detail
            // one. Balanced values are the selector's authoritative fields.
            let (target_px, detail_fraction) = match lod.quality_target() {
                LodQualityTarget::Coarsest => (f32::MAX, 0.0),
                LodQualityTarget::Balanced {
                    detail_fraction,
                    max_error_px,
                } => (max_error_px, detail_fraction),
                LodQualityTarget::Original => (0.0, 1.0),
            };
            [target_px, detail_fraction, 0.0, 0.0]
        });
        Self {
            flags: [
                settings.preset.shader_code(),
                metadata_count.min(u32::MAX as usize) as u32,
                0,
                0,
            ],
            quality_params,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_lod_debug_bind_group<R: PlanarSync>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<CloudPipeline<R>>,
    gpu_clouds: Res<RenderAssets<R::GpuPlanarType>>,
    mut stats: ResMut<LodDebugGpuUploadStats>,
    mut clouds: Query<LodDebugPrepareQuery<R>>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    let pipeline_changed = pipeline.is_changed();
    let Some(layout) = pipeline.lod_debug_layout.as_ref() else {
        for (entity, _, _, _, _, _, existing) in &mut clouds {
            if existing.is_some() {
                commands.entity(entity).remove::<LodDebugBindGroup<R>>();
            }
        }
        return;
    };

    let fallback = [LodDebugRecord::default()];
    for (entity, handle, settings, lod_settings, metadata, candidate_epoch, existing) in &mut clouds
    {
        if !settings.lod_debug.requires_metadata() {
            if existing.is_some() {
                commands.entity(entity).remove::<LodDebugBindGroup<R>>();
            }
            continue;
        }

        let Some(gpu_cloud) = gpu_clouds.get(handle.handle()) else {
            continue;
        };
        let sparse = metadata.and_then(LodDebugMetadata::sparse);
        let records = metadata.map(LodDebugMetadata::records).unwrap_or_default();
        let record_count = sparse
            .map(LodDebugSparseMetadata::record_count)
            .unwrap_or(records.len())
            .min(gpu_cloud.len());
        let source_pointer = if sparse.is_none() && record_count != 0 {
            records.as_ptr() as usize
        } else {
            0
        };
        let sparse_identity = sparse.map(LodDebugSparseMetadata::identity);
        let byte_len = record_count
            .max(1)
            .checked_mul(std::mem::size_of::<LodDebugRecord>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(u64::MAX);
        let limits = render_device.limits();
        if byte_len > limits.max_storage_buffer_binding_size || byte_len > limits.max_buffer_size {
            warn!(
                "LoD debug metadata for entity {entity:?} needs {byte_len} bytes, exceeding the adapter storage-buffer limit; rendering without annotations"
            );
            if existing.is_some() {
                commands.entity(entity).remove::<LodDebugBindGroup<R>>();
            }
            continue;
        }

        if let Some(mut existing) = existing
            && existing.source_pointer == source_pointer
            && existing.sparse_identity == sparse_identity
            && existing.record_count == record_count
        {
            let existing: &mut LodDebugBindGroup<R> = existing.as_mut();
            if let Some(sparse) = sparse {
                let upload = apply_lod_debug_sparse_uploads(
                    &render_queue,
                    &existing._buffer,
                    sparse,
                    record_count,
                    candidate_epoch,
                    &mut existing.sparse_slot_invariant_revisions,
                    &mut existing.sparse_slot_payload_revisions,
                );
                stats.record_bytes_written = stats
                    .record_bytes_written
                    .saturating_add(upload.bytes_written);
                existing.upload_complete = upload.complete;
                existing.ready = update_lod_debug_sparse_binding_readiness(
                    sparse,
                    candidate_epoch,
                    &existing.sparse_slot_invariant_revisions,
                    &existing.sparse_slot_payload_revisions,
                    upload.complete,
                    &mut existing.current_invariant_ready,
                    &mut existing.current_payload_ready,
                    &mut existing.pending_invariants_ready,
                );
            } else {
                let candidate_epoch_ready = candidate_epoch.is_none_or(|epoch| {
                    lod_debug_candidate_epoch_ready(
                        epoch.candidates_are_current,
                        epoch.pending_candidate_active,
                        epoch.pending_activation_armed,
                    )
                });
                existing.ready = record_count != 0 && candidate_epoch_ready;
                existing.current_invariant_ready = existing.ready;
                existing.current_payload_ready = existing.ready;
                existing.pending_invariants_ready = existing.ready;
                existing.upload_complete = true;
            }

            if existing.settings != settings.lod_debug
                || existing.lod_settings.as_ref() != lod_settings
            {
                let config =
                    LodDebugGpuUniform::new(&settings.lod_debug, lod_settings, record_count);
                render_queue.write_buffer(&existing._config_buffer, 0, bytemuck::bytes_of(&config));
                stats.config_bytes_written = stats
                    .config_bytes_written
                    .saturating_add(std::mem::size_of::<LodDebugGpuUniform>() as u64);
                existing.settings = settings.lod_debug;
                existing.lod_settings = lod_settings.cloned();
            }
            if pipeline_changed {
                existing.bind_group = create_lod_debug_bind_group(
                    &render_device,
                    layout,
                    &existing._buffer,
                    &existing._config_buffer,
                );
            }
            existing._source_metadata = metadata.cloned();
            continue;
        }

        let (
            buffer,
            mut sparse_slot_invariant_revisions,
            mut sparse_slot_payload_revisions,
            ready,
            current_invariant_ready,
            current_payload_ready,
            pending_invariants_ready,
            upload_complete,
        ) = if let Some(sparse) = sparse {
            let buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("lod_debug_records_sparse"),
                size: byte_len,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut invariant_revisions = vec![0; sparse.slots().len()];
            let mut payload_revisions = vec![0; sparse.slots().len()];
            let upload = apply_lod_debug_sparse_uploads(
                &render_queue,
                &buffer,
                sparse,
                record_count,
                candidate_epoch,
                &mut invariant_revisions,
                &mut payload_revisions,
            );
            stats.record_bytes_written = stats
                .record_bytes_written
                .saturating_add(upload.bytes_written);
            let mut current_invariant_ready = false;
            let mut current_payload_ready = false;
            let mut pending_invariants_ready = false;
            let ready = update_lod_debug_sparse_binding_readiness(
                sparse,
                candidate_epoch,
                &invariant_revisions,
                &payload_revisions,
                upload.complete,
                &mut current_invariant_ready,
                &mut current_payload_ready,
                &mut pending_invariants_ready,
            );
            (
                buffer,
                invariant_revisions,
                payload_revisions,
                ready,
                current_invariant_ready,
                current_payload_ready,
                pending_invariants_ready,
                upload.complete,
            )
        } else {
            let upload_records = if record_count == 0 {
                &fallback[..]
            } else {
                &records[..record_count]
            };
            let contents = bytemuck::cast_slice(upload_records);
            let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("lod_debug_records"),
                contents,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            stats.record_bytes_written = stats
                .record_bytes_written
                .saturating_add(contents.len() as u64);
            let candidate_epoch_ready = candidate_epoch.is_none_or(|epoch| {
                lod_debug_candidate_epoch_ready(
                    epoch.candidates_are_current,
                    epoch.pending_candidate_active,
                    epoch.pending_activation_armed,
                )
            });
            let ready = record_count != 0 && candidate_epoch_ready;
            (
                buffer,
                Vec::new(),
                Vec::new(),
                ready,
                ready,
                ready,
                ready,
                true,
            )
        };
        stats.record_buffer_allocations = stats.record_buffer_allocations.saturating_add(1);
        let config = LodDebugGpuUniform::new(&settings.lod_debug, lod_settings, record_count);
        let config_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("lod_debug_config"),
            contents: bytemuck::bytes_of(&config),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        stats.config_bytes_written = stats
            .config_bytes_written
            .saturating_add(std::mem::size_of::<LodDebugGpuUniform>() as u64);
        let bind_group =
            create_lod_debug_bind_group(&render_device, layout, &buffer, &config_buffer);
        commands.entity(entity).insert(LodDebugBindGroup::<R> {
            bind_group,
            _buffer: buffer,
            _config_buffer: config_buffer,
            _source_metadata: metadata.cloned(),
            source_pointer,
            sparse_identity,
            sparse_slot_invariant_revisions: std::mem::take(&mut sparse_slot_invariant_revisions),
            sparse_slot_payload_revisions: std::mem::take(&mut sparse_slot_payload_revisions),
            record_count,
            ready,
            current_invariant_ready,
            current_payload_ready,
            pending_invariants_ready,
            upload_complete,
            settings: settings.lod_debug,
            lod_settings: lod_settings.cloned(),
            marker: std::marker::PhantomData,
        });
    }
}

const LOD_DEBUG_MAX_SPARSE_UPLOAD_BYTES_PER_FRAME: u64 = 64 * 1024 * 1024;
const LOD_DEBUG_MAX_SPARSE_UPLOAD_SLOTS_PER_FRAME: usize = 256;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LodDebugSparseUploadResult {
    bytes_written: u64,
    slots_written: usize,
    complete: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LodDebugRevisionKind {
    Invariant,
    Payload,
}

#[cfg(lod_render_path)]
fn lod_debug_candidate_revisions_ready(
    sparse: &LodDebugSparseMetadata,
    candidates: &LodRenderCandidates,
    applied_revisions: &[u64],
    kind: LodDebugRevisionKind,
) -> bool {
    candidates.by_camera.values().all(|candidate| {
        candidate.required_atlas_ranges().iter().all(|range| {
            let Some(slot) = sparse.slots().get(range.slot.index as usize) else {
                return false;
            };
            let revision = match kind {
                LodDebugRevisionKind::Invariant => slot.invariant_revision(),
                LodDebugRevisionKind::Payload => slot.payload_revision(),
            };
            slot.key() == Some((range.page, range.slot.generation))
                && slot.records().is_some()
                && applied_revisions.get(range.slot.index as usize).copied() == Some(revision)
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn update_lod_debug_sparse_binding_readiness(
    sparse: &LodDebugSparseMetadata,
    candidate_epoch: Option<&LodDebugCandidateEpoch>,
    applied_invariant_revisions: &[u64],
    applied_payload_revisions: &[u64],
    upload_complete: bool,
    current_invariant_ready: &mut bool,
    current_payload_ready: &mut bool,
    pending_invariants_ready: &mut bool,
) -> bool {
    let globally_ready = sparse.is_complete() && upload_complete;
    let Some(candidate_epoch) = candidate_epoch else {
        *current_invariant_ready = globally_ready;
        *current_payload_ready = globally_ready;
        *pending_invariants_ready = false;
        return globally_ready;
    };

    let target_invariants_ready = candidate_epoch.debug_metadata_staged
        && lod_debug_epoch_revisions_ready(
            sparse,
            candidate_epoch,
            applied_invariant_revisions,
            LodDebugRevisionKind::Invariant,
        );
    let target_payload_ready = candidate_epoch.debug_metadata_staged
        && lod_debug_epoch_revisions_ready(
            sparse,
            candidate_epoch,
            applied_payload_revisions,
            LodDebugRevisionKind::Payload,
        );

    if candidate_epoch.candidates_are_current {
        *current_invariant_ready = target_invariants_ready;
        *current_payload_ready = target_payload_ready;
        *pending_invariants_ready = false;
    } else {
        *pending_invariants_ready = target_invariants_ready;
        if !candidate_epoch.retained_current {
            *current_invariant_ready = false;
            *current_payload_ready = false;
        } else if globally_ready {
            // Covers enabling debug while a replacement is already pending:
            // the pending component does not duplicate retained-current ranges,
            // so a complete whole-sidecar upload is the conservative proof.
            *current_invariant_ready = true;
            *current_payload_ready = true;
        }
    }

    let replacement_may_draw = !candidate_epoch.candidates_are_current
        && (candidate_epoch.pending_candidate_active || candidate_epoch.pending_activation_armed);
    // Every compacted entry now carries the exact per-view Residency code that
    // belongs to its own candidate epoch. Consequently Residency, like the
    // invariant presets, can remain specialized while current and replacement
    // outputs overlap; mutable record payload uploads continue in the
    // background for flat/legacy entries whose packed code is zero.
    lod_debug_sparse_candidate_epoch_ready(
        candidate_epoch.candidates_are_current,
        candidate_epoch.retained_current,
        replacement_may_draw,
        *current_invariant_ready,
        *pending_invariants_ready,
    )
}

#[inline]
const fn lod_debug_sparse_candidate_epoch_ready(
    candidates_are_current: bool,
    retained_current: bool,
    replacement_may_draw: bool,
    current_invariants_ready: bool,
    pending_invariants_ready: bool,
) -> bool {
    if candidates_are_current {
        current_invariants_ready
    } else if replacement_may_draw {
        pending_invariants_ready && (!retained_current || current_invariants_ready)
    } else if retained_current {
        current_invariants_ready
    } else {
        pending_invariants_ready
    }
}

fn lod_debug_epoch_revisions_ready(
    sparse: &LodDebugSparseMetadata,
    candidate_epoch: &LodDebugCandidateEpoch,
    applied_revisions: &[u64],
    kind: LodDebugRevisionKind,
) -> bool {
    candidate_epoch
        .required_slots
        .iter()
        .all(|&(page, index, generation)| {
            let Some(slot) = sparse.slots().get(index as usize) else {
                return false;
            };
            let revision = match kind {
                LodDebugRevisionKind::Invariant => slot.invariant_revision(),
                LodDebugRevisionKind::Payload => slot.payload_revision(),
            };
            slot.key() == Some((page, generation))
                && slot.records().is_some()
                && applied_revisions.get(index as usize).copied() == Some(revision)
        })
}

fn apply_lod_debug_sparse_uploads(
    render_queue: &RenderQueue,
    buffer: &Buffer,
    sparse: &LodDebugSparseMetadata,
    record_count: usize,
    priority_epoch: Option<&LodDebugCandidateEpoch>,
    applied_invariant_revisions: &mut [u64],
    applied_payload_revisions: &mut [u64],
) -> LodDebugSparseUploadResult {
    let record_size = std::mem::size_of::<LodDebugRecord>();
    let mut result = LodDebugSparseUploadResult {
        complete: true,
        ..default()
    };

    if let Some(candidate_epoch) = priority_epoch {
        let invariant_only = !candidate_epoch.candidates_are_current;
        for &(_, slot_index, _) in &candidate_epoch.required_slots {
            let index = slot_index as usize;
            let Some(slot) = sparse.slots().get(index) else {
                continue;
            };
            let already_applied = if invariant_only {
                applied_invariant_revisions.get(index).copied() == Some(slot.invariant_revision())
            } else {
                applied_payload_revisions.get(index).copied() == Some(slot.payload_revision())
            };
            if already_applied {
                continue;
            }
            if !apply_lod_debug_sparse_slot_upload(
                render_queue,
                buffer,
                sparse,
                record_count,
                index,
                record_size,
                applied_invariant_revisions,
                applied_payload_revisions,
                &mut result,
            ) {
                result.complete = false;
                return result;
            }
        }
    }

    for index in 0..sparse.slots().len() {
        let slot = &sparse.slots()[index];
        if applied_payload_revisions.get(index).copied() == Some(slot.payload_revision()) {
            continue;
        }
        if !apply_lod_debug_sparse_slot_upload(
            render_queue,
            buffer,
            sparse,
            record_count,
            index,
            record_size,
            applied_invariant_revisions,
            applied_payload_revisions,
            &mut result,
        ) {
            result.complete = false;
            return result;
        }
    }
    result.complete = sparse.slots().iter().enumerate().all(|(index, slot)| {
        applied_payload_revisions.get(index).copied() == Some(slot.payload_revision())
    });
    result
}

#[allow(clippy::too_many_arguments)]
fn apply_lod_debug_sparse_slot_upload(
    render_queue: &RenderQueue,
    buffer: &Buffer,
    sparse: &LodDebugSparseMetadata,
    record_count: usize,
    index: usize,
    record_size: usize,
    applied_invariant_revisions: &mut [u64],
    applied_payload_revisions: &mut [u64],
    result: &mut LodDebugSparseUploadResult,
) -> bool {
    let slot = &sparse.slots()[index];
    let Some(records) = slot.records() else {
        if let Some(applied) = applied_invariant_revisions.get_mut(index) {
            *applied = slot.invariant_revision();
        }
        if let Some(applied) = applied_payload_revisions.get_mut(index) {
            *applied = slot.payload_revision();
        }
        return true;
    };
    let start = match index.checked_mul(sparse.records_per_slot()) {
        Some(start) => start,
        None => {
            return false;
        }
    };
    let end = match start.checked_add(records.len()) {
        Some(end) => end,
        None => {
            return false;
        }
    };
    if end > record_count {
        // The GPU cloud can be smaller than the metadata's physical
        // address space after adapter-side clamping. Such a tail is never
        // drawable and therefore needs no upload.
        if let Some(applied) = applied_invariant_revisions.get_mut(index) {
            *applied = slot.invariant_revision();
        }
        if let Some(applied) = applied_payload_revisions.get_mut(index) {
            *applied = slot.payload_revision();
        }
        return true;
    }
    let contents = bytemuck::cast_slice(records);
    let bytes = contents.len() as u64;
    if result.slots_written >= LOD_DEBUG_MAX_SPARSE_UPLOAD_SLOTS_PER_FRAME
        || result.bytes_written.saturating_add(bytes) > LOD_DEBUG_MAX_SPARSE_UPLOAD_BYTES_PER_FRAME
    {
        return false;
    }
    let offset = start
        .checked_mul(record_size)
        .and_then(|offset| u64::try_from(offset).ok())
        .unwrap_or(u64::MAX);
    render_queue.write_buffer(buffer, offset, contents);
    if let Some(applied) = applied_invariant_revisions.get_mut(index) {
        *applied = slot.invariant_revision();
    }
    if let Some(applied) = applied_payload_revisions.get_mut(index) {
        *applied = slot.payload_revision();
    }
    result.bytes_written = result.bytes_written.saturating_add(bytes);
    result.slots_written += 1;
    true
}

fn create_lod_debug_bind_group(
    render_device: &RenderDevice,
    layout: &BindGroupLayout,
    records: &Buffer,
    config: &Buffer,
) -> BindGroup {
    render_device.create_bind_group(
        "lod_debug_bind_group",
        layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: records.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: config.as_entire_binding(),
            },
        ],
    )
}

#[inline]
const fn lod_debug_shader_enabled(
    requested: bool,
    binding_present: bool,
    binding_ready: bool,
    layout_available: bool,
) -> bool {
    requested && binding_present && binding_ready && layout_available
}

/// A pending render candidate and the live annotation sidecar are separate
/// transactions. Once replacement output is armed (or has activated before the
/// next main-world poll), the retained sidecar must not be bound to that output.
#[inline]
const fn lod_debug_candidate_epoch_ready(
    candidates_are_current: bool,
    pending_candidate_active: bool,
    pending_activation_armed: bool,
) -> bool {
    candidates_are_current || (!pending_candidate_active && !pending_activation_armed)
}

#[allow(clippy::too_many_arguments)]
fn queue_gaussians<R: PlanarSync>(
    gaussian_cloud_uniform: Res<ComponentUniforms<CloudUniform>>,
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CloudPipeline<R>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<CloudPipeline<R>>>,
    pipeline_cache: Res<PipelineCache>,
    mut debug_stats: ResMut<LodDebugGpuUploadStats>,
    gaussian_clouds: Res<RenderAssets<R::GpuPlanarType>>,
    sorted_entries: Res<RenderAssets<GpuSortedEntry>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    mut views: Query<(
        &ExtractedView,
        &GaussianCamera,
        &RenderVisibleEntities,
        Option<&Msaa>,
    )>,
    gaussian_splatting_bundles: Query<GpuCloudBundleQuery<R>>,
) {
    debug!("queue_gaussians");

    let warmup = views.iter().any(|(_, camera, _, _)| camera.warmup);
    if warmup {
        debug!("skipping gaussian cloud render during warmup");
        return;
    }

    // TODO: condition this system based on CloudBindGroup attachment
    if gaussian_cloud_uniform.buffer().is_none() {
        debug!("uniform buffer not initialized");
        return;
    };

    let draw_custom = transparent_3d_draw_functions
        .read()
        .id::<DrawGaussians<R>>();

    for (view, _, visible_entities, msaa) in &mut views {
        debug!("queue gaussians view");
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            debug!("transparent phase not found");
            continue;
        };

        debug!("visible entities...");
        let Some(visible_class) = visible_entities.get::<CloudVisibilityClass>() else {
            continue;
        };

        for (render_entity, visible_entity) in &visible_class.entities_cpu_culling {
            if gaussian_splatting_bundles.get(*render_entity).is_err() {
                debug!("gaussian splatting bundle not found");
                continue;
            }

            #[cfg(lod_render_path)]
            let (
                _entity,
                cloud_handle,
                aabb,
                sorted_entries_handle,
                settings,
                _lod_settings,
                transform,
                lod_debug_bind_group,
                render_candidates,
            ) = gaussian_splatting_bundles.get(*render_entity).unwrap();

            #[cfg(not(lod_render_path))]
            let (
                _entity,
                cloud_handle,
                aabb,
                sorted_entries_handle,
                settings,
                _lod_settings,
                transform,
                lod_debug_bind_group,
            ) = gaussian_splatting_bundles.get(*render_entity).unwrap();

            #[cfg(lod_render_path)]
            let lod_candidate =
                render_candidates.is_some_and(|candidates| candidates.candidate_draw_required);
            #[cfg(not(lod_render_path))]
            let lod_candidate = false;

            #[cfg(lod_render_path)]
            let external_active_set = render_candidates
                .and_then(|candidates| {
                    candidates
                        .by_camera
                        .get(&view.retained_view_entity.main_entity.id())
                })
                .is_some_and(|candidate| candidate.is_external_active_set());
            #[cfg(not(lod_render_path))]
            let external_active_set = false;

            if !gaussian_rasterization_is_supported(settings.gaussian_mode, settings.rasterize_mode)
            {
                error!(
                    gaussian_mode = ?settings.gaussian_mode,
                    rasterize_mode = ?settings.rasterize_mode,
                    "unsupported Gaussian/rasterization mode pair; skipping draw"
                );
                #[cfg(lod_render_path)]
                if let Some(candidates) = render_candidates {
                    for candidate in candidates.by_camera.values() {
                        candidate.phase.store(
                            crate::stream::render_commit::LOD_RENDER_FAILED,
                            std::sync::atomic::Ordering::Release,
                        );
                    }
                }
                continue;
            }

            debug!("queue gaussians clouds");
            if gaussian_clouds.get(cloud_handle.handle()).is_none() {
                debug!("gaussian cloud asset not found");
                return;
            }

            if sorted_entries.get(sorted_entries_handle).is_none() {
                debug!("sorted entries asset not found");
                return;
            }

            let msaa = msaa.cloned().unwrap_or_default();

            // Hierarchy Page/Level/Residency records do not describe a LODGE
            // resident catalog. Even a stale binding from an earlier entity
            // state must not enable that shader permutation for an external
            // candidate.
            let lod_debug_requested =
                settings.lod_debug.requires_metadata() && !external_active_set;
            let debug_binding_ready =
                lod_debug_requested && lod_debug_bind_group.is_some_and(|debug| debug.ready);
            let debug_pipeline_active = lod_debug_shader_enabled(
                lod_debug_requested,
                lod_debug_bind_group.is_some(),
                lod_debug_bind_group.is_some_and(|debug| debug.ready),
                custom_pipeline.lod_debug_layout_desc.is_some(),
            );
            let key = cloud_pipeline_key(
                settings,
                debug_pipeline_active,
                lod_candidate,
                msaa.samples(),
                view.target_format == TextureFormat::Rgba16Float,
            );
            if debug_binding_ready {
                debug_stats.ready_bind_group_queues =
                    debug_stats.ready_bind_group_queues.saturating_add(1);
            }
            if key.lod_debug {
                debug_stats.specialized_pipeline_queues =
                    debug_stats.specialized_pipeline_queues.saturating_add(1);
            }

            let pipeline = pipelines.specialize(&pipeline_cache, &custom_pipeline, key);

            let rangefinder = view.rangefinder3d();
            let aabb_center = (aabb.min() + aabb.max()) / 2.0;
            let aabb_size = aabb.max() - aabb.min();
            let center = *transform
                * GlobalTransform::from(
                    Transform::from_translation(aabb_center.into()).with_scale(aabb_size.into()),
                );
            let distance = rangefinder.distance(&center.translation());

            transparent_phase.add_transient(Transparent3d {
                sorting_info: TransparentSortingInfo3d::Sorted {
                    mesh_center: center.translation(),
                    depth_bias: 0.0,
                },
                entity: (*render_entity, *visible_entity),
                draw_function: draw_custom,
                distance,
                pipeline,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: false,
            });
        }
    }
}

// TODO: pipeline trait
// TODO: support extentions /w ComputePipelineDescriptor builder
#[derive(Resource)]
pub struct CloudPipeline<R: PlanarSync> {
    shader: Handle<Shader>,
    pub gaussian_cloud_layout: BindGroupLayout,
    pub gaussian_cloud_layout_desc: BindGroupLayoutDescriptor,
    pub gaussian_uniform_layout: BindGroupLayout,
    pub gaussian_uniform_layout_desc: BindGroupLayoutDescriptor,
    pub view_layout: BindGroupLayout,
    pub view_layout_desc: BindGroupLayoutDescriptor,
    pub compute_view_layout: BindGroupLayout,
    pub compute_view_layout_desc: BindGroupLayoutDescriptor,
    pub sorted_layout: BindGroupLayout,
    pub sorted_layout_desc: BindGroupLayoutDescriptor,
    #[cfg(lod_render_path)]
    pub(crate) lod_sorted_layout: BindGroupLayout,
    #[cfg(lod_render_path)]
    pub(crate) lod_sorted_layout_desc: BindGroupLayoutDescriptor,
    pub lod_debug_layout: Option<BindGroupLayout>,
    pub lod_debug_layout_desc: Option<BindGroupLayoutDescriptor>,
    phantom: std::marker::PhantomData<R>,
}

fn buffer_layout(
    buffer_binding_type: BufferBindingType,
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZero<u64>>,
) -> BindGroupLayoutEntryBuilder {
    match buffer_binding_type {
        BufferBindingType::Uniform => {
            binding_types::uniform_buffer_sized(has_dynamic_offset, min_binding_size)
        }
        BufferBindingType::Storage { read_only } => {
            if read_only {
                binding_types::storage_buffer_read_only_sized(has_dynamic_offset, min_binding_size)
            } else {
                binding_types::storage_buffer_sized(has_dynamic_offset, min_binding_size)
            }
        }
    }
}

pub(crate) fn storage_layout_descriptor<P: ReflectInterleaved>(
    label: impl Into<Cow<'static, str>>,
    read_only: bool,
) -> BindGroupLayoutDescriptor {
    let entries = P::min_binding_sizes()
        .iter()
        .enumerate()
        .map(|(idx, size)| BindGroupLayoutEntry {
            binding: idx as u32,
            visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(*size as u64),
            },
            count: None,
        })
        .collect::<Vec<_>>();

    BindGroupLayoutDescriptor::new(label, &entries)
}

pub(crate) fn gaussian_storage_layout_descriptor<R: PlanarSync>(
    label: impl Into<Cow<'static, str>>,
    read_only: bool,
) -> BindGroupLayoutDescriptor
where
    <R::GpuPlanarType as GpuPlanar>::PackedType: ReflectInterleaved + 'static,
{
    #[cfg(feature = "precompute_covariance_3d")]
    if TypeId::of::<<R::GpuPlanarType as GpuPlanar>::PackedType>()
        == TypeId::of::<crate::gaussian::formats::planar_3d::Gaussian3d>()
    {
        use crate::{
            gaussian::f32::{Covariance3dOpacity, PositionVisibility, Rotation, ScaleOpacity},
            material::spherical_harmonics::SphericalHarmonicCoefficients,
        };

        let sizes = [
            std::mem::size_of::<PositionVisibility>(),
            std::mem::size_of::<SphericalHarmonicCoefficients>(),
            std::mem::size_of::<Rotation>(),
            std::mem::size_of::<ScaleOpacity>(),
            std::mem::size_of::<Covariance3dOpacity>(),
        ];
        let entries = sizes
            .iter()
            .enumerate()
            .map(|(binding, size)| BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(*size as u64),
                },
                count: None,
            })
            .collect::<Vec<_>>();
        return BindGroupLayoutDescriptor::new(label, &entries);
    }

    storage_layout_descriptor::<<R::GpuPlanarType as GpuPlanar>::PackedType>(label, read_only)
}

impl<R: PlanarSync> FromWorld for CloudPipeline<R>
where
    R::GpuPlanarType: GpuPlanarStorage,
    <R::GpuPlanarType as GpuPlanar>::PackedType: ReflectInterleaved,
{
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();

        let visibility_ranges_buffer_binding_type = render_device
            .get_supported_read_only_binding_type(VISIBILITY_RANGES_STORAGE_BUFFER_COUNT);

        let visibility_ranges_entry = buffer_layout(
            visibility_ranges_buffer_binding_type,
            false,
            Some(Vec4::min_size()),
        )
        .build(14, ShaderStages::VERTEX);

        let view_layout_entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(GlobalsUniform::min_size()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(PreviousViewData::min_size()),
                },
                count: None,
            },
            visibility_ranges_entry,
        ];

        let compute_view_layout_entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(GlobalsUniform::min_size()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(PreviousViewData::min_size()),
                },
                count: None,
            },
            visibility_ranges_entry,
        ];

        let view_layout_desc =
            BindGroupLayoutDescriptor::new("gaussian_view_layout", &view_layout_entries);
        let view_layout = render_device
            .create_bind_group_layout(Some("gaussian_view_layout"), &view_layout_entries);

        let compute_view_layout_desc = BindGroupLayoutDescriptor::new(
            "gaussian_compute_view_layout",
            &compute_view_layout_entries,
        );
        let compute_view_layout = render_device.create_bind_group_layout(
            Some("gaussian_compute_view_layout"),
            &compute_view_layout_entries,
        );

        let gaussian_uniform_layout_entries = [BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: Some(CloudUniform::min_size()),
            },
            count: None,
        }];
        let gaussian_uniform_layout_desc = BindGroupLayoutDescriptor::new(
            "gaussian_uniform_layout",
            &gaussian_uniform_layout_entries,
        );
        let gaussian_uniform_layout = render_device.create_bind_group_layout(
            Some("gaussian_uniform_layout"),
            &gaussian_uniform_layout_entries,
        );

        #[cfg(not(feature = "morph_particles"))]
        let read_only = true;
        #[cfg(feature = "morph_particles")]
        let read_only = false;

        let gaussian_cloud_layout = R::GpuPlanarType::bind_group_layout(render_device, read_only);
        let gaussian_cloud_layout_desc =
            gaussian_storage_layout_descriptor::<R>("gaussian_cloud_layout", read_only);

        #[cfg(feature = "buffer_storage")]
        let sorted_layout_entries = [BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: true,
                min_binding_size: BufferSize::new(std::mem::size_of::<SortEntry>() as u64),
            },
            count: None,
        }];
        #[cfg(feature = "buffer_storage")]
        let sorted_layout_desc =
            BindGroupLayoutDescriptor::new("sorted_layout", &sorted_layout_entries);
        #[cfg(feature = "buffer_storage")]
        let sorted_layout =
            render_device.create_bind_group_layout(Some("sorted_layout"), &sorted_layout_entries);
        #[cfg(lod_render_path)]
        let lod_sorted_layout_entries = [
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: BufferSize::new(std::mem::size_of::<SortEntry>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(8 * std::mem::size_of::<u32>() as u64),
                },
                count: None,
            },
        ];
        #[cfg(lod_render_path)]
        let lod_sorted_layout_desc = BindGroupLayoutDescriptor::new(
            "gaussian_lod_sorted_morph_layout",
            &lod_sorted_layout_entries,
        );
        #[cfg(lod_render_path)]
        let lod_sorted_layout = render_device.create_bind_group_layout(
            Some("gaussian_lod_sorted_morph_layout"),
            &lod_sorted_layout_entries,
        );
        #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
        let sorted_layout = texture::get_sorted_bind_group_layout(render_device);
        #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
        let sorted_layout_desc = BindGroupLayoutDescriptor::new(
            "texture_sorted_layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    view_dimension: TextureViewDimension::D2,
                    sample_type: TextureSampleType::Uint,
                    multisampled: false,
                },
                count: None,
            }],
        );

        // Debug annotations use a fifth bind group so ordinary clouds retain
        // the established 0..=3 layout and do not pay for a dummy metadata
        // buffer. Four bind groups is the WebGPU minimum; unsupported adapters
        // simply keep this optional diagnostic path unavailable.
        #[cfg(all(feature = "buffer_storage", not(feature = "webgl2")))]
        let (lod_debug_layout, lod_debug_layout_desc) = if render_device.limits().max_bind_groups
            >= 5
            && render_device.limits().max_storage_buffer_binding_size
                >= std::mem::size_of::<LodDebugRecord>() as u64
        {
            let entries = [
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(
                            std::mem::size_of::<LodDebugRecord>() as u64
                        ),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(
                            std::mem::size_of::<LodDebugGpuUniform>() as u64,
                        ),
                    },
                    count: None,
                },
            ];
            (
                Some(render_device.create_bind_group_layout(Some("lod_debug_layout"), &entries)),
                Some(BindGroupLayoutDescriptor::new("lod_debug_layout", &entries)),
            )
        } else {
            (None, None)
        };
        #[cfg(not(all(feature = "buffer_storage", not(feature = "webgl2"))))]
        let (lod_debug_layout, lod_debug_layout_desc) = (None, None);

        debug!("created cloud pipeline");

        Self {
            gaussian_cloud_layout,
            gaussian_cloud_layout_desc,
            gaussian_uniform_layout,
            gaussian_uniform_layout_desc,
            view_layout,
            view_layout_desc,
            compute_view_layout,
            compute_view_layout_desc,
            shader: GAUSSIAN_SHADER_HANDLE,
            sorted_layout,
            sorted_layout_desc,
            #[cfg(lod_render_path)]
            lod_sorted_layout,
            #[cfg(lod_render_path)]
            lod_sorted_layout_desc,
            lod_debug_layout,
            lod_debug_layout_desc,
            phantom: std::marker::PhantomData,
        }
    }
}

// TODO: allow setting shader defines via API
// TODO: separate shader defines for each pipeline
#[derive(Clone, Copy, Debug)]
pub struct ShaderDefines {
    pub radix_bits_per_digit: u32,
    pub radix_digit_places: u32,
    pub radix_key_shift: u32,
    pub radix_base: u32,
    pub entries_per_invocation_a: u32,
    pub entries_per_invocation_c: u32,
    pub workgroup_invocations_a: u32,
    pub workgroup_invocations_c: u32,
    pub workgroup_entries_a: u32,
    pub workgroup_entries_c: u32,
    pub sorting_buffer_size: u32,
}

impl ShaderDefines {
    pub fn for_radix_depth_bits(radix_sort_depth_bits: RadixSortDepthBits) -> Self {
        let radix_bits_per_digit = 8;
        let radix_digit_places = radix_sort_depth_bits.bits() / radix_bits_per_digit;
        let radix_key_shift = 32 - radix_sort_depth_bits.bits();
        let radix_base = 1 << radix_bits_per_digit;
        let entries_per_invocation_a = 4;
        let entries_per_invocation_c = 4;
        let workgroup_invocations_a = radix_base * radix_digit_places;
        let workgroup_invocations_c = radix_base;
        let workgroup_entries_a = workgroup_invocations_a * entries_per_invocation_a;
        let workgroup_entries_c = workgroup_invocations_c * entries_per_invocation_c;
        let sorting_buffer_size =
            radix_base * radix_digit_places * std::mem::size_of::<u32>() as u32;

        Self {
            radix_bits_per_digit,
            radix_digit_places,
            radix_key_shift,
            radix_base,
            entries_per_invocation_a,
            entries_per_invocation_c,
            workgroup_invocations_a,
            workgroup_invocations_c,
            workgroup_entries_a,
            workgroup_entries_c,
            sorting_buffer_size,
        }
    }

    pub fn max_tile_count(&self, count: usize) -> u32 {
        (count as u32).div_ceil(self.workgroup_entries_c)
    }

    pub fn sorting_status_counters_buffer_size(&self, count: usize) -> usize {
        self.radix_base as usize * self.max_tile_count(count) as usize * std::mem::size_of::<u32>()
    }

    pub fn radix_initial_parity(&self) -> usize {
        (self.radix_digit_places % 2) as usize
    }
}

impl Default for ShaderDefines {
    fn default() -> Self {
        Self::for_radix_depth_bits(RadixSortDepthBits::default())
    }
}

pub fn shader_defs(key: CloudPipelineKey) -> Vec<ShaderDefVal> {
    shader_defs_with_defines(key, ShaderDefines::default())
}

/// Returns whether the selected public raster mode has a complete shader
/// implementation for the Gaussian representation. Unsupported pairs are
/// rejected before pipeline specialization so they cannot surface later as an
/// asynchronous Naga failure (and, for LoD, cannot replace a drawable source).
pub(crate) const fn gaussian_rasterization_is_supported(
    gaussian_mode: GaussianMode,
    rasterize_mode: RasterizeMode,
) -> bool {
    match rasterize_mode {
        RasterizeMode::Normal => !matches!(gaussian_mode, GaussianMode::Gaussian4d),
        RasterizeMode::Velocity => matches!(gaussian_mode, GaussianMode::Gaussian4d),
        RasterizeMode::Classification
        | RasterizeMode::Color
        | RasterizeMode::Depth
        | RasterizeMode::OpticalFlow
        | RasterizeMode::Position => true,
    }
}

pub fn shader_defs_with_defines(
    key: CloudPipelineKey,
    defines: ShaderDefines,
) -> Vec<ShaderDefVal> {
    let mut shader_defs = vec![
        ShaderDefVal::UInt("SH_COEFF_COUNT".into(), SH_COEFF_COUNT as u32),
        ShaderDefVal::UInt("SH_4D_COEFF_COUNT".into(), SH_4D_COEFF_COUNT as u32),
        ShaderDefVal::UInt("SH_DEGREE".into(), SH_DEGREE as u32),
        ShaderDefVal::UInt("SH_DEGREE_TIME".into(), SH_4D_DEGREE_TIME as u32),
        ShaderDefVal::UInt("HALF_SH_COEFF_COUNT".into(), HALF_SH_COEFF_COUNT as u32),
        ShaderDefVal::UInt("SH_VEC4_PLANES".into(), SH_VEC4_PLANES as u32),
        ShaderDefVal::UInt("RADIX_BASE".into(), defines.radix_base),
        ShaderDefVal::UInt("RADIX_BITS_PER_DIGIT".into(), defines.radix_bits_per_digit),
        ShaderDefVal::UInt("RADIX_DIGIT_PLACES".into(), defines.radix_digit_places),
        ShaderDefVal::UInt("RADIX_KEY_SHIFT".into(), defines.radix_key_shift),
        ShaderDefVal::UInt(
            "ENTRIES_PER_INVOCATION_A".into(),
            defines.entries_per_invocation_a,
        ),
        ShaderDefVal::UInt(
            "ENTRIES_PER_INVOCATION_C".into(),
            defines.entries_per_invocation_c,
        ),
        ShaderDefVal::UInt(
            "WORKGROUP_INVOCATIONS_A".into(),
            defines.workgroup_invocations_a,
        ),
        ShaderDefVal::UInt(
            "WORKGROUP_INVOCATIONS_C".into(),
            defines.workgroup_invocations_c,
        ),
        ShaderDefVal::UInt("WORKGROUP_ENTRIES_C".into(), defines.workgroup_entries_c),
    ];

    if key.aabb {
        shader_defs.push("USE_AABB".into());
    }

    if !key.aabb {
        shader_defs.push("USE_OBB".into());
    }

    if key.binary_gaussian_op {
        shader_defs.push("BINARY_GAUSSIAN_OP".into());
    }

    if key.opacity_adaptive_radius {
        shader_defs.push("OPACITY_ADAPTIVE_RADIUS".into());
    }

    if key.visualize_bounding_box {
        shader_defs.push("VISUALIZE_BOUNDING_BOX".into());
    }

    if key.lod_debug {
        shader_defs.push("LOD_DEBUG".into());
    }

    if key.lod_candidate {
        shader_defs.push("LOD_CANDIDATE".into());
        #[cfg(lod_render_path)]
        if key.gaussian_mode == GaussianMode::Gaussian3d {
            // The ABI-16 parent map and interpolation helpers currently encode
            // the canonical planar 3D representation only. Other Gaussian
            // modes still use the exact hard-cut candidate path.
            shader_defs.push("LOD_MORPH".into());
        }
    }

    #[cfg(feature = "morph_particles")]
    shader_defs.push("READ_WRITE_POINTS".into());

    #[cfg(feature = "packed")]
    shader_defs.push("PACKED".into());

    #[cfg(feature = "buffer_storage")]
    shader_defs.push("BUFFER_STORAGE".into());

    #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
    shader_defs.push("BUFFER_TEXTURE".into());

    // #[cfg(feature = "f16")]
    // shader_defs.push("F16".into());

    shader_defs.push("F32".into());

    #[cfg(feature = "packed")]
    shader_defs.push("PACKED_F32".into());

    // #[cfg(all(feature = "f16", feature = "buffer_storage"))]
    // shader_defs.push("PLANAR_F16".into());

    #[cfg(feature = "buffer_storage")]
    shader_defs.push("PLANAR_F32".into());

    // #[cfg(all(feature = "f16", feature = "buffer_texture"))]
    // shader_defs.push("PLANAR_TEXTURE_F16".into());

    #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
    shader_defs.push("PLANAR_TEXTURE_F32".into());

    #[cfg(feature = "precompute_covariance_3d")]
    if key.gaussian_mode == GaussianMode::Gaussian3d {
        shader_defs.push("PRECOMPUTE_COVARIANCE_3D".into());
    }

    #[cfg(feature = "webgl2")]
    shader_defs.push("WEBGL2".into());

    match key.gaussian_mode {
        GaussianMode::Gaussian2d => shader_defs.push("GAUSSIAN_2D".into()),
        GaussianMode::Gaussian3d => shader_defs.push("GAUSSIAN_3D".into()),
        GaussianMode::Gaussian4d => shader_defs.push("GAUSSIAN_4D".into()),
    }

    match key.gaussian_mode {
        GaussianMode::Gaussian2d | GaussianMode::Gaussian3d => {
            shader_defs.push("GAUSSIAN_3D_STRUCTURE".into());
        }
        _ => {}
    }

    match key.rasterize_mode {
        RasterizeMode::Classification => shader_defs.push("RASTERIZE_CLASSIFICATION".into()),
        RasterizeMode::Color => shader_defs.push("RASTERIZE_COLOR".into()),
        RasterizeMode::Depth => shader_defs.push("RASTERIZE_DEPTH".into()),
        RasterizeMode::Normal => shader_defs.push("RASTERIZE_NORMAL".into()),
        RasterizeMode::OpticalFlow => shader_defs.push("RASTERIZE_OPTICAL_FLOW".into()),
        RasterizeMode::Position => shader_defs.push("RASTERIZE_POSITION".into()),
        RasterizeMode::Velocity => shader_defs.push("RASTERIZE_VELOCITY".into()),
    }

    match key.draw_mode {
        DrawMode::All => {}
        DrawMode::Selected => shader_defs.push("DRAW_SELECTED".into()),
        DrawMode::HighlightSelected => shader_defs.push("HIGHLIGHT_SELECTED".into()),
    }

    shader_defs
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Default)]
pub struct CloudPipelineKey {
    pub aabb: bool,
    pub binary_gaussian_op: bool,
    pub visualize_bounding_box: bool,
    pub opacity_adaptive_radius: bool,
    pub draw_mode: DrawMode,
    pub gaussian_mode: GaussianMode,
    pub rasterize_mode: RasterizeMode,
    pub lod_debug: bool,
    pub lod_candidate: bool,
    pub sample_count: u32,
    pub hdr: bool,
}

/// Builds the exact Gaussian raster specialization shared by draw queuing and
/// the LoD two-phase commit. A cold source-to-atlas handoff may publish
/// `PREPARED` only after this same `LOD_CANDIDATE` variant is compiled; keeping
/// the key construction centralized prevents a supposedly ready candidate
/// from switching to a different, still-queued MSAA/HDR/debug permutation.
pub(crate) fn cloud_pipeline_key(
    settings: &CloudSettings,
    lod_debug: bool,
    lod_candidate: bool,
    sample_count: u32,
    hdr: bool,
) -> CloudPipelineKey {
    CloudPipelineKey {
        aabb: settings.aabb,
        binary_gaussian_op: false,
        opacity_adaptive_radius: settings.opacity_adaptive_radius,
        visualize_bounding_box: settings.visualize_bounding_box,
        draw_mode: settings.draw_mode,
        gaussian_mode: settings.gaussian_mode,
        rasterize_mode: settings.rasterize_mode,
        lod_debug,
        lod_candidate,
        sample_count,
        hdr,
    }
}

impl<R: PlanarSync> SpecializedRenderPipeline for CloudPipeline<R> {
    type Key = CloudPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let shader_defs = shader_defs(key);

        let format = if key.hdr {
            TextureFormat::Rgba16Float
        } else {
            TextureFormat::Rgba8UnormSrgb
        };

        debug!("specializing cloud pipeline");

        #[cfg(lod_render_path)]
        let sorted_layout = if key.lod_candidate {
            self.lod_sorted_layout_desc.clone()
        } else {
            self.sorted_layout_desc.clone()
        };
        #[cfg(not(lod_render_path))]
        let sorted_layout = self.sorted_layout_desc.clone();
        let mut layout = vec![
            self.view_layout_desc.clone(),
            self.gaussian_uniform_layout_desc.clone(),
            self.gaussian_cloud_layout_desc.clone(),
            sorted_layout,
        ];
        if key.lod_debug {
            layout.push(
                self.lod_debug_layout_desc
                    .clone()
                    .expect("LoD debug key requires an available metadata layout"),
            );
        }

        RenderPipelineDescriptor {
            label: Some("gaussian cloud render pipeline".into()),
            layout,
            immediate_size: 0,
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                entry_point: Some("vs_points".into()),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: Some("fs_main".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                unclipped_depth: false,
                cull_mode: None,
                conservative: false,
                polygon_mode: PolygonMode::Fill,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            zero_initialize_workgroup_memory: true,
        }
    }
}

#[allow(type_alias_bounds)]
type DrawGaussians<R: bevy_interleave::prelude::PlanarSync> = (
    SetItemPipeline,
    // SetViewBindGroup<0>,
    SetPreviousViewBindGroup<0>,
    SetGaussianUniformBindGroup<1>,
    DrawGaussianInstanced<R>,
);

#[allow(dead_code)]
#[derive(Component, ShaderType, Clone, Copy)]
pub struct CloudUniform {
    pub transform: Mat4,
    pub global_opacity: f32,
    pub global_scale: f32,
    /// Conservative upper bound on the transform's largest singular value.
    /// Computing it once per extracted cloud avoids repeating matrix Gram work
    /// for all four vertices of every splat.
    pub transform_scale_bound: f32,
    pub count: u32,
    pub count_root_ceil: u32,
    pub time: f32,
    pub time_start: f32,
    pub time_stop: f32,
    pub num_classes: u32,
    pub color_space: u32,
    pub min: Vec4,
    pub max: Vec4,
}

pub(crate) fn gaussian_transform_scale_bound(matrix: Mat4) -> f32 {
    let transform_x = matrix.x_axis.truncate();
    let transform_y = matrix.y_axis.truncate();
    let transform_z = matrix.z_axis.truncate();
    let gram_xx = transform_x.dot(transform_x);
    let gram_xy = transform_x.dot(transform_y);
    let gram_xz = transform_x.dot(transform_z);
    let gram_yy = transform_y.dot(transform_y);
    let gram_yz = transform_y.dot(transform_z);
    let gram_zz = transform_z.dot(transform_z);
    let bound = (gram_xx + gram_xy.abs() + gram_xz.abs())
        .max(gram_yy + gram_xy.abs() + gram_yz.abs())
        .max(gram_zz + gram_xz.abs() + gram_yz.abs())
        .max(0.0)
        .sqrt();
    if bound.is_finite() { bound } else { f32::NAN }
}

#[allow(clippy::type_complexity)]
pub fn extract_gaussians<R: PlanarSync>(
    mut commands: Commands,
    mut prev_commands_len: Local<usize>,
    asset_server: Res<AssetServer>,
    gaussian_cloud_res: Res<RenderAssets<R::GpuPlanarType>>,
    gaussians_query: Extract<
        Query<(
            RenderEntity,
            &ViewVisibility,
            &R::PlanarTypeHandle,
            &Aabb,
            &SortedEntriesHandle,
            &CloudSettings,
            &GlobalTransform,
        )>,
    >,
) {
    let mut commands_list = Vec::with_capacity(*prev_commands_len);
    // let visible_gaussians = gaussians_query.iter().filter(|(_, vis, ..)| vis.is_visible());

    for (entity, visibility, cloud_handle, aabb, sorted_entries, settings, transform) in
        gaussians_query.iter()
    {
        debug!("extracting gaussian cloud entity: {:?}", entity);

        if !visibility.get() {
            debug!("gaussian cloud not visible");
            continue;
        }

        if let Some(load_state) = asset_server.get_load_state(cloud_handle.handle())
            && load_state.is_loading()
        {
            debug!("gaussian cloud asset loading");
            continue;
        }

        if gaussian_cloud_res.get(cloud_handle.handle()).is_none() {
            debug!("gaussian cloud asset not found");
            continue;
        }

        let cloud = gaussian_cloud_res.get(cloud_handle.handle()).unwrap();
        let transform_matrix = transform.to_matrix();
        let settings_uniform = CloudUniform {
            transform: transform_matrix,
            global_opacity: settings.global_opacity,
            global_scale: settings.global_scale,
            transform_scale_bound: gaussian_transform_scale_bound(transform_matrix),
            count: cloud.len() as u32,
            count_root_ceil: (cloud.len() as f32).sqrt().ceil() as u32,
            time: settings.time,
            time_start: settings.time_start,
            time_stop: settings.time_stop,
            num_classes: settings.num_classes as u32,
            color_space: match settings.color_space {
                GaussianColorSpace::SrgbRec709Display => 0,
                GaussianColorSpace::LinRec709Display => 1,
            },
            min: aabb.min().extend(1.0),
            max: aabb.max().extend(1.0),
        };

        commands_list.push((
            entity,
            GpuCloudBundle::<R> {
                aabb: *aabb,
                settings: settings.clone(),
                settings_uniform,
                sorted_entries: sorted_entries.clone(),
                cloud_handle: cloud_handle.clone(),
                transform: *transform,
            },
        ));
    }
    *prev_commands_len = commands_list.len();
    commands.insert_batch(commands_list);
}

#[derive(Resource, Default)]
pub struct GaussianUniformBindGroups {
    pub base_bind_group: Option<BindGroup>,
}

#[derive(Component)]
pub struct SortBindGroup {
    pub sorted_bind_group: BindGroup,
}

#[allow(clippy::too_many_arguments)]
fn queue_gaussian_bind_group<R: PlanarSync>(
    mut commands: Commands,
    mut groups: ResMut<GaussianUniformBindGroups>,
    gaussian_cloud_pipeline: Res<CloudPipeline<R>>,
    render_device: Res<RenderDevice>,
    gaussian_uniforms: Res<ComponentUniforms<CloudUniform>>,
    asset_server: Res<AssetServer>,
    gaussian_cloud_res: Res<RenderAssets<R::GpuPlanarType>>,
    sorted_entries_res: Res<RenderAssets<GpuSortedEntry>>,
    gaussian_clouds: Query<GpuCloudBindGroupQuery<R>>,
    #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))] gpu_images: Res<
        RenderAssets<bevy::render::texture::GpuImage>,
    >,
) {
    let Some(resource) = gaussian_uniforms.binding() else {
        return;
    };

    let pipeline_changed = gaussian_cloud_pipeline.is_changed();
    if gaussian_uniforms.is_changed() || pipeline_changed || groups.base_bind_group.is_none() {
        groups.base_bind_group = Some(render_device.create_bind_group(
            "gaussian_uniform_bind_group",
            &gaussian_cloud_pipeline.gaussian_uniform_layout,
            &[BindGroupEntry {
                binding: 0,
                resource,
            }],
        ));
    }

    let gaussian_assets_changed = gaussian_cloud_res.is_changed();
    let sorted_assets_changed = sorted_entries_res.is_changed();
    #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
    let mut should_refresh_for_assets =
        pipeline_changed || gaussian_assets_changed || sorted_assets_changed;
    #[cfg(not(all(feature = "buffer_texture", not(feature = "buffer_storage"))))]
    let should_refresh_for_assets =
        pipeline_changed || gaussian_assets_changed || sorted_assets_changed;

    #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
    {
        let textures_changed = gpu_images.is_changed();
        should_refresh_for_assets |= textures_changed;
    }

    for query in gaussian_clouds.iter() {
        let (entity, cloud_handle, sorted_entries_handle, existing_bind_group) = query;

        if let Some(load_state) = asset_server.get_load_state(cloud_handle.handle())
            && load_state.is_loading()
        {
            debug!("queue gaussian bind group: cloud asset loading");
            continue;
        }

        let Some(cloud) = gaussian_cloud_res.get(cloud_handle.handle()) else {
            debug!("queue gaussian bind group: cloud asset not found");
            continue;
        };

        if let Some(load_state) = asset_server.get_load_state(&sorted_entries_handle.0)
            && load_state.is_loading()
        {
            debug!("queue gaussian bind group: sorted entries asset loading");
            continue;
        }

        let Some(sorted_entries) = sorted_entries_res.get(&sorted_entries_handle.0) else {
            debug!("queue gaussian bind group: sorted entries asset not found");
            continue;
        };
        let Some(sorted_entry_binding_size) =
            sort_entry_binding_size(sorted_entries.entry_count, cloud.len())
        else {
            // A cloud handle can grow in PostUpdate (the LoD bridge swaps in
            // its physical atlas) one frame before the main-world sorted-entry
            // asset is resized. Drop the stale bind group instead of binding
            // beyond its buffer or allowing robust OOB reads.
            commands.entity(entity).remove::<SortBindGroup>();
            continue;
        };
        #[cfg(not(feature = "buffer_storage"))]
        let _ = sorted_entry_binding_size;

        if !should_refresh_for_assets && existing_bind_group.is_some() {
            continue;
        }

        #[cfg(feature = "buffer_storage")]
        let sorted_bind_group = render_device.create_bind_group(
            "render_sorted_bind_group",
            &gaussian_cloud_pipeline.sorted_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &sorted_entries.sorted_entry_buffer,
                    offset: 0,
                    size: BufferSize::new(sorted_entry_binding_size),
                }),
            }],
        );
        #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
        let sorted_bind_group = render_device.create_bind_group(
            Some("render_sorted_bind_group"),
            &gaussian_cloud_pipeline.sorted_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(
                    &gpu_images
                        .get(&sorted_entries.texture)
                        .unwrap()
                        .texture_view,
                ),
            }],
        );

        debug!("inserting sorted bind group");

        commands
            .entity(entity)
            .insert(SortBindGroup { sorted_bind_group });
    }
}

#[derive(Component)]
pub struct GaussianViewBindGroup {
    pub value: BindGroup,
}

#[derive(Component)]
pub struct GaussianComputeViewBindGroup {
    pub value: BindGroup,
}

// TODO: move to gaussian camera module
// TODO: remove cloud pipeline dependency by separating view layout

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn queue_gaussian_view_bind_groups<R: PlanarSync>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    gaussian_cloud_pipeline: Res<CloudPipeline<R>>,
    view_uniforms: Res<ViewUniforms>,
    previous_view_uniforms: Res<PreviousViewUniforms>,
    views: Query<
        (
            Entity,
            &ExtractedView,
            Option<&PreviousViewData>,
            Option<&GaussianViewBindGroup>,
        ),
        With<GaussianCamera>,
    >,
    visibility_ranges: Res<RenderVisibilityRanges>,
    globals_buffer: Res<GlobalsBuffer>,
) {
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };
    let Some(previous_view_binding) = previous_view_uniforms.uniforms.binding() else {
        return;
    };
    let Some(globals) = globals_buffer.buffer.binding() else {
        return;
    };
    let Some(visibility_ranges_buffer) = visibility_ranges.buffer().buffer() else {
        return;
    };

    let resources_changed = gaussian_cloud_pipeline.is_changed()
        || view_uniforms.is_changed()
        || previous_view_uniforms.is_changed()
        || globals_buffer.is_changed()
        || visibility_ranges.is_changed();

    for (entity, _extracted_view, _maybe_previous_view, existing_bind_group) in &views {
        if !resources_changed && existing_bind_group.is_some() {
            continue;
        }

        let layout = &gaussian_cloud_pipeline.view_layout;

        let entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: view_binding.clone(),
            },
            BindGroupEntry {
                binding: 1,
                resource: globals.clone(),
            },
            BindGroupEntry {
                binding: 2,
                resource: previous_view_binding.clone(),
            },
            BindGroupEntry {
                binding: 14,
                resource: visibility_ranges_buffer.as_entire_binding(),
            },
        ];

        let view_bind_group =
            render_device.create_bind_group("gaussian_view_bind_group", layout, &entries);

        debug!("inserting gaussian view bind group");

        commands.entity(entity).insert(GaussianViewBindGroup {
            value: view_bind_group,
        });
    }
}

// Prepare the compute view bind group using the compute_view_layout (for compute pipelines)
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn queue_gaussian_compute_view_bind_groups<R: PlanarSync>(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    gaussian_cloud_pipeline: Res<CloudPipeline<R>>,
    view_uniforms: Res<ViewUniforms>,
    previous_view_uniforms: Res<PreviousViewUniforms>,
    views: Query<
        (
            Entity,
            &ExtractedView,
            Option<&PreviousViewData>,
            Option<&GaussianComputeViewBindGroup>,
        ),
        With<GaussianCamera>,
    >,
    visibility_ranges: Res<RenderVisibilityRanges>,
    globals_buffer: Res<GlobalsBuffer>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };
    let Some(previous_view_binding) = previous_view_uniforms.uniforms.binding() else {
        return;
    };
    let Some(globals) = globals_buffer.buffer.binding() else {
        return;
    };
    let Some(visibility_ranges_buffer) = visibility_ranges.buffer().buffer() else {
        return;
    };

    let resources_changed = gaussian_cloud_pipeline.is_changed()
        || view_uniforms.is_changed()
        || previous_view_uniforms.is_changed()
        || globals_buffer.is_changed()
        || visibility_ranges.is_changed();

    for (entity, _extracted_view, _maybe_previous_view, existing_bind_group) in &views {
        if !resources_changed && existing_bind_group.is_some() {
            continue;
        }

        let layout = &gaussian_cloud_pipeline.compute_view_layout;

        let entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: view_binding.clone(),
            },
            BindGroupEntry {
                binding: 1,
                resource: globals.clone(),
            },
            BindGroupEntry {
                binding: 2,
                resource: previous_view_binding.clone(),
            },
            BindGroupEntry {
                binding: 14,
                resource: visibility_ranges_buffer.as_entire_binding(),
            },
        ];

        let view_bind_group =
            render_device.create_bind_group("gaussian_compute_view_bind_group", layout, &entries);

        commands
            .entity(entity)
            .insert(GaussianComputeViewBindGroup {
                value: view_bind_group,
            });
    }
}

pub struct SetViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetViewBindGroup<I> {
    type Param = ();
    type ViewQuery = (Read<GaussianViewBindGroup>, Read<ViewUniformOffset>);
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _: &P,
        (gaussian_view_bind_group, view_uniform): ROQueryItem<'w, 'w, Self::ViewQuery>,
        _entity: Option<()>,
        _: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &gaussian_view_bind_group.value, &[view_uniform.offset]);

        debug!("set view bind group");

        RenderCommandResult::Success
    }
}

pub struct SetPreviousViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetPreviousViewBindGroup<I> {
    type Param = SRes<PrepassViewBindGroup>;
    type ViewQuery = (
        Read<ViewUniformOffset>,
        Option<Has<MotionVectorPrepass>>,
        Option<Read<PreviousViewUniformOffset>>,
    );
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _: &P,
        (view_uniform_offset, has_motion_vector_prepass, previous_view_uniform_offset): ROQueryItem<
            'w,
            'w,
            Self::ViewQuery,
        >,
        _entity: Option<()>,
        prepass_view_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let prepass_view_bind_group = prepass_view_bind_group.into_inner();
        match previous_view_uniform_offset {
            Some(previous_view_uniform_offset) if has_motion_vector_prepass.unwrap_or_default() => {
                pass.set_bind_group(
                    I,
                    prepass_view_bind_group.motion_vectors.as_ref().unwrap(),
                    &[
                        view_uniform_offset.offset,
                        previous_view_uniform_offset.offset,
                    ],
                );
            }
            _ => pass.set_bind_group(
                I,
                prepass_view_bind_group.motion_vectors.as_ref().unwrap(),
                &[view_uniform_offset.offset, 0],
            ),
        }

        debug!("set previous view bind group");

        RenderCommandResult::Success
    }
}

pub struct SetGaussianUniformBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetGaussianUniformBindGroup<I> {
    type Param = SRes<GaussianUniformBindGroups>;
    type ViewQuery = ();
    type ItemQuery = Read<DynamicUniformIndex<CloudUniform>>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        gaussian_cloud_index: Option<ROQueryItem<'w, 'w, Self::ItemQuery>>,
        bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let bind_groups = bind_groups.into_inner();
        let bind_group = bind_groups
            .base_bind_group
            .as_ref()
            .expect("bind group not initialized");

        let mut set_bind_group = |indices: &[u32]| pass.set_bind_group(I, bind_group, indices);

        if gaussian_cloud_index.is_none() {
            debug!("skipping gaussian uniform bind group\n");
            return RenderCommandResult::Skip;
        }

        let gaussian_cloud_index = gaussian_cloud_index.unwrap().index();
        set_bind_group(&[gaussian_cloud_index]);

        debug!("set gaussian uniform bind group");

        RenderCommandResult::Success
    }
}

pub struct DrawGaussianInstanced<R: PlanarSync> {
    phantom: std::marker::PhantomData<R>,
}

#[allow(type_alias_bounds)]
#[cfg(lod_render_path)]
type DrawGaussianItemQuery<R: PlanarSync> = (
    Read<R::PlanarTypeHandle>,
    Read<PlanarStorageBindGroup<R>>,
    Read<SortBindGroup>,
    Read<CloudSettings>,
    Option<Read<LodDebugBindGroup<R>>>,
    Option<Read<LodRenderCandidates>>,
);

#[allow(type_alias_bounds)]
#[cfg(not(lod_render_path))]
type DrawGaussianItemQuery<R: PlanarSync> = (
    Read<R::PlanarTypeHandle>,
    Read<PlanarStorageBindGroup<R>>,
    Read<SortBindGroup>,
    Read<CloudSettings>,
    Option<Read<LodDebugBindGroup<R>>>,
);

#[allow(type_alias_bounds)]
#[cfg(lod_render_path)]
type DrawGaussianParam<R: PlanarSync> = (
    SRes<RenderAssets<R::GpuPlanarType>>,
    SRes<lod::LodCompactionBuffers<R>>,
);

#[allow(type_alias_bounds)]
#[cfg(not(lod_render_path))]
type DrawGaussianParam<R: PlanarSync> = SRes<RenderAssets<R::GpuPlanarType>>;

#[cfg(lod_render_path)]
const fn skip_unready_candidate_required_draw(
    candidate_draw_required: bool,
    lod_output_ready: bool,
) -> bool {
    candidate_draw_required && !lod_output_ready
}

impl<R: PlanarSync> Default for DrawGaussianInstanced<R> {
    fn default() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<P: PhaseItem, R: PlanarSync> RenderCommand<P> for DrawGaussianInstanced<R>
where
    R::GpuPlanarType: GpuPlanarStorage,
{
    type Param = DrawGaussianParam<R>;
    type ViewQuery = (Read<SortTrigger>, Read<ExtractedView>);
    type ItemQuery = DrawGaussianItemQuery<R>;

    #[inline]
    fn render<'w>(
        item: &P,
        (view, _extracted_view): ROQueryItem<'w, 'w, Self::ViewQuery>,
        entity: Option<ROQueryItem<'w, 'w, Self::ItemQuery>>,
        gaussian_params: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        debug!("render call");

        #[cfg(not(lod_render_path))]
        let _ = item;

        #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
        let _ = view;

        #[cfg(lod_render_path)]
        let (
            handle,
            planar_bind_groups,
            sort_bind_groups,
            _cloud_settings,
            lod_debug_bind_group,
            render_candidates,
        ) = entity.expect("gaussian cloud entity not found");
        #[cfg(not(lod_render_path))]
        let (handle, planar_bind_groups, sort_bind_groups, _cloud_settings, lod_debug_bind_group) =
            entity.expect("gaussian cloud entity not found");

        #[cfg(lod_render_path)]
        let (gaussian_clouds, lod_buffers) = gaussian_params;
        #[cfg(lod_render_path)]
        let gaussian_clouds = gaussian_clouds.into_inner();
        #[cfg(lod_render_path)]
        let lod_buffers = lod_buffers.into_inner();
        #[cfg(not(lod_render_path))]
        let gaussian_clouds = gaussian_params.into_inner();

        let gpu_gaussian_cloud = match gaussian_clouds.get(handle.handle()) {
            Some(gpu_gaussian_cloud) => gpu_gaussian_cloud,
            None => {
                debug!("gpu cloud not found");
                return RenderCommandResult::Skip;
            }
        };

        #[cfg(lod_render_path)]
        let lod_state = lod_buffers.get_ready(
            _extracted_view.retained_view_entity,
            item.entity(),
            handle.handle().id(),
        );

        #[cfg(lod_render_path)]
        let external_active_set = render_candidates
            .and_then(|candidates| {
                candidates
                    .by_camera
                    .get(&_extracted_view.retained_view_entity.main_entity.id())
            })
            .is_some_and(|candidate| candidate.is_external_active_set());
        #[cfg(not(lod_render_path))]
        let external_active_set = false;

        #[cfg(lod_render_path)]
        if skip_unready_candidate_required_draw(
            render_candidates.is_some_and(|candidates| candidates.candidate_draw_required),
            lod_state.is_some(),
        ) {
            // A package atlas is a page cache. Drawing it without a validated
            // per-view candidate would expose a parent/child union or a
            // partially staged transaction, so cold/new views stay loading.
            return RenderCommandResult::Skip;
        }

        debug!("drawing indirect");

        pass.set_bind_group(2, &planar_bind_groups.bind_group, &[]);

        #[cfg(feature = "buffer_storage")]
        {
            // TODO: align dynamic offset to `min_storage_buffer_offset_alignment`
            #[cfg(lod_render_path)]
            if let Some(state) = lod_state {
                pass.set_bind_group(
                    3,
                    state.sorted_entry_bind_group(_cloud_settings.radix_sort_depth_bits),
                    &[0],
                );
            } else {
                pass.set_bind_group(
                    3,
                    &sort_bind_groups.sorted_bind_group,
                    &[view.camera_index as u32
                        * std::mem::size_of::<SortEntry>() as u32
                        * gpu_gaussian_cloud.len() as u32],
                );
            }

            #[cfg(not(lod_render_path))]
            pass.set_bind_group(
                3,
                &sort_bind_groups.sorted_bind_group,
                &[view.camera_index as u32
                    * std::mem::size_of::<SortEntry>() as u32
                    * gpu_gaussian_cloud.len() as u32],
            );
        }

        #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
        {
            pass.set_bind_group(3, &sort_bind_groups.sorted_bind_group, &[]);
        }

        if _cloud_settings.lod_debug.requires_metadata()
            && !external_active_set
            && let Some(lod_debug_bind_group) = lod_debug_bind_group
            && lod_debug_bind_group.ready
        {
            pass.set_bind_group(4, &lod_debug_bind_group.bind_group, &[]);
        }

        #[cfg(feature = "webgl2")]
        pass.draw(0..4, 0..gpu_gaussian_cloud.len() as u32);

        #[cfg(not(feature = "webgl2"))]
        {
            #[cfg(lod_render_path)]
            let draw_indirect_buffer = lod_buffers
                .get_ready(
                    _extracted_view.retained_view_entity,
                    item.entity(),
                    handle.handle().id(),
                )
                .map(|state| &state.indirect_args_buffer)
                .unwrap_or_else(|| gpu_gaussian_cloud.draw_indirect_buffer());

            #[cfg(not(lod_render_path))]
            let draw_indirect_buffer = gpu_gaussian_cloud.draw_indirect_buffer();

            pass.draw_indirect(draw_indirect_buffer, 0);
        }

        RenderCommandResult::Success
    }
}

#[cfg(test)]
mod shader_contract_tests {
    use super::{
        CloudPipelineKey, GAUSSIAN_AUTHORED_SUPPORT_SIGMA, GAUSSIAN_MIP_FILTER_VARIANCE_2D,
        LodDebugGpuUniform, LodDebugGpuUploadStats, gaussian_mip_filter_covariance_2d,
        gaussian_rasterization_is_supported, gaussian_support_cutoff,
        lod_debug_candidate_epoch_ready, lod_debug_shader_enabled,
        lod_debug_sparse_candidate_epoch_ready, planar_storage_binding_needs_refresh, shader_defs,
    };

    #[cfg(lod_render_path)]
    use super::skip_unready_candidate_required_draw;

    #[test]
    fn lod_morph_sample_avoids_wgsl_reserved_field_names() {
        let shader = include_str!("lod_morph.wgsl");
        assert!(shader.contains("enabled: bool"));
        assert!(!shader.contains("active: bool"));
    }

    #[cfg(lod_render_path)]
    #[test]
    fn package_page_cache_never_falls_through_to_full_atlas_draw() {
        assert!(skip_unready_candidate_required_draw(true, false));
        assert!(!skip_unready_candidate_required_draw(true, true));
        assert!(!skip_unready_candidate_required_draw(false, false));
    }

    #[cfg(feature = "precompute_covariance_3d")]
    use super::gaussian_storage_layout_descriptor;
    use crate::LodPageId;
    #[cfg(feature = "precompute_covariance_3d")]
    use crate::gaussian::formats::planar_3d::Gaussian3d;
    use crate::gaussian::lod_debug::{
        LOD_DEBUG_PAGE_LINEAR_LUMINANCE, lod_debug_page_color, stable_page_color_key,
    };

    #[test]
    fn planar_storage_rebinds_handle_swap_without_an_asset_event() {
        assert!(planar_storage_binding_needs_refresh(true, false, false));
        assert!(!planar_storage_binding_needs_refresh(true, true, false));
        assert!(planar_storage_binding_needs_refresh(false, true, false));
        assert!(planar_storage_binding_needs_refresh(true, true, true));
    }

    #[test]
    fn mip_filter_preserves_integrated_alpha_while_retaining_the_authored_footprint() {
        for covariance in [
            [4.0_f32, 0.0, 9.0],
            [2.0, 0.75, 1.0],
            [0.0004, 0.0001, 0.0003],
        ] {
            let filtered = gaussian_mip_filter_covariance_2d(covariance);
            assert_eq!(
                filtered.covariance,
                [
                    covariance[0] + GAUSSIAN_MIP_FILTER_VARIANCE_2D,
                    covariance[1],
                    covariance[2] + GAUSSIAN_MIP_FILTER_VARIANCE_2D,
                ]
            );
            assert!(filtered.opacity_scale > 0.0 && filtered.opacity_scale <= 1.0);

            let original_determinant =
                covariance[0] * covariance[2] - covariance[1] * covariance[1];
            let filtered_determinant = filtered.covariance[0] * filtered.covariance[2]
                - filtered.covariance[1] * filtered.covariance[1];
            let original_integrated_alpha = original_determinant.sqrt();
            let filtered_integrated_alpha = filtered.opacity_scale * filtered_determinant.sqrt();
            let relative_error = (filtered_integrated_alpha - original_integrated_alpha).abs()
                / original_integrated_alpha.max(f32::MIN_POSITIVE);
            assert!(
                relative_error <= 2.0e-6,
                "covariance={covariance:?} filtered={filtered:?} relative_error={relative_error}"
            );
        }

        let degenerate = gaussian_mip_filter_covariance_2d([1.0, 1.0, 1.0]);
        assert_eq!(degenerate.covariance, [1.3, 1.0, 1.3]);
        assert_eq!(degenerate.opacity_scale, 0.0);
    }

    #[test]
    fn lod_candidate_support_is_authored_three_sigma_without_changing_flat_policy() {
        for opacity in [
            -4.0_f32,
            0.0,
            1.0e-8,
            1.0e-4,
            0.001,
            0.01,
            0.1,
            1.0,
            2.0,
            f32::MAX,
        ] {
            let legacy_flat = (9.0 + 2.0 * opacity.max(0.000_001_f32).ln())
                .max(0.000_001)
                .sqrt();
            assert_eq!(
                gaussian_support_cutoff(opacity, true, false).to_bits(),
                legacy_flat.to_bits(),
                "flat cutoff drifted for opacity={opacity}"
            );
            assert_eq!(
                gaussian_support_cutoff(opacity, true, true),
                GAUSSIAN_AUTHORED_SUPPORT_SIGMA,
                "LoD support diverged from authored three sigma for opacity={opacity}"
            );
            assert_eq!(
                gaussian_support_cutoff(opacity, false, false),
                GAUSSIAN_AUTHORED_SUPPORT_SIGMA
            );
        }
        let flat_cutoff = gaussian_support_cutoff(0.001, true, false);
        let lod_cutoff = gaussian_support_cutoff(0.001, true, true);
        assert!(flat_cutoff < 0.0011);
        let flat_modeled_mass = 1.0 - (-0.5 * flat_cutoff * flat_cutoff).exp();
        let lod_modeled_mass = 1.0 - (-0.5 * lod_cutoff * lod_cutoff).exp();
        assert!(flat_modeled_mass < 1.0e-6);
        assert!(lod_modeled_mass > 0.98);
    }

    #[test]
    fn mip_filter_and_lod_support_shader_contract_match_the_cpu_oracle() {
        const SHADER_COORDINATE_UNITS_PER_PIXEL: f32 = 2.0;
        const SHADER_COVARIANCE_UNITS_PER_PIXEL_SQUARED: f32 =
            SHADER_COORDINATE_UNITS_PER_PIXEL * SHADER_COORDINATE_UNITS_PER_PIXEL;
        const SHADER_FILTER_VARIANCE: f32 =
            GAUSSIAN_MIP_FILTER_VARIANCE_2D * SHADER_COVARIANCE_UNITS_PER_PIXEL_SQUARED;

        fn shader_filter_model(physical_covariance: [f32; 3]) -> ([f32; 3], f32) {
            let covariance =
                physical_covariance.map(|value| value * SHADER_COVARIANCE_UNITS_PER_PIXEL_SQUARED);
            let filtered = [
                covariance[0] + SHADER_FILTER_VARIANCE,
                covariance[1],
                covariance[2] + SHADER_FILTER_VARIANCE,
            ];
            let original_determinant =
                covariance[0] * covariance[2] - covariance[1] * covariance[1];
            let filtered_determinant = filtered[0] * filtered[2] - filtered[1] * filtered[1];
            let determinant_ratio = original_determinant / filtered_determinant;
            let opacity_scale = if original_determinant > 0.0
                && filtered_determinant > 0.0
                && determinant_ratio >= 0.0
            {
                determinant_ratio.clamp(0.0, 1.0).sqrt()
            } else {
                0.0
            };
            (
                filtered.map(|value| value / SHADER_COVARIANCE_UNITS_PER_PIXEL_SQUARED),
                opacity_scale,
            )
        }

        fn mip_support_radius_world(
            cutoff: f32,
            min_shader_focal: f32,
            projection_w: f32,
            view_z: f32,
        ) -> Option<f32> {
            if !cutoff.is_finite()
                || cutoff < 0.0
                || !min_shader_focal.is_finite()
                || min_shader_focal <= 0.0
            {
                return None;
            }
            let mut radius = cutoff * SHADER_FILTER_VARIANCE.sqrt() / min_shader_focal;
            if projection_w == 0.0 {
                let depth = view_z.abs();
                if !depth.is_finite() || depth <= 0.0 {
                    return None;
                }
                radius *= depth;
            } else if projection_w != 1.0 {
                return None;
            }
            radius.is_finite().then_some(radius)
        }

        assert_eq!(GAUSSIAN_MIP_FILTER_VARIANCE_2D, 0.3);
        assert_eq!(SHADER_FILTER_VARIANCE, 1.2);
        assert_eq!(GAUSSIAN_AUTHORED_SUPPORT_SIGMA, 3.0);
        let mip_radius_shader = GAUSSIAN_AUTHORED_SUPPORT_SIGMA * SHADER_FILTER_VARIANCE.sqrt();
        assert!((mip_radius_shader / SHADER_COORDINATE_UNITS_PER_PIXEL - 1.643_167_6).abs() < 1e-6);
        let perspective_margin = mip_support_radius_world(3.0, 300.0, 0.0, -5.0).unwrap();
        let orthographic_margin = mip_support_radius_world(3.0, 300.0, 1.0, f32::NAN).unwrap();
        assert!((perspective_margin - 0.054_772_26).abs() < 1e-7);
        assert!((orthographic_margin - 0.010_954_452).abs() < 1e-8);
        assert!(mip_support_radius_world(3.0, 0.0, 0.0, -5.0).is_none());
        assert!(mip_support_radius_world(3.0, 300.0, 0.0, 0.0).is_none());
        assert!(mip_support_radius_world(f32::NAN, 300.0, 1.0, 0.0).is_none());
        assert!(mip_support_radius_world(3.0, 300.0, 0.5, -5.0).is_none());
        assert!(mip_support_radius_world(3.0, 300.0, f32::NAN, -5.0).is_none());
        let tiny_authored_radius = 3.0 * 0.001;
        let center_distance_outside = 0.015;
        assert!(tiny_authored_radius < center_distance_outside);
        assert!(tiny_authored_radius + perspective_margin > center_distance_outside);
        for covariance in [
            [4.0_f32, 0.0, 9.0],
            [2.0, 0.75, 1.0],
            [0.0004, 0.0001, 0.0003],
            [1.0, 1.0, 1.0],
        ] {
            let cpu = gaussian_mip_filter_covariance_2d(covariance);
            let shader = shader_filter_model(covariance);
            assert_eq!(cpu.covariance, shader.0);
            assert_eq!(cpu.opacity_scale.to_bits(), shader.1.to_bits());
        }

        let helpers = include_str!("helpers.wgsl");
        assert!(helpers.contains("GAUSSIAN_SHADER_COORDINATE_UNITS_PER_PIXEL: f32 = 2.0"));
        assert!(helpers.contains("GAUSSIAN_MIP_FILTER_VARIANCE_2D_PHYSICAL: f32 = 0.3"));
        assert!(helpers.contains("GAUSSIAN_MIP_FILTER_VARIANCE_2D_SHADER"));
        assert!(helpers.contains("fn gaussian_mip_support_radius_world("));
        assert!(helpers.contains("let mip_radius_shader = cutoff * sqrt("));
        assert!(helpers.contains("let min_focal = min(focal.x, focal.y);"));
        assert!(helpers.contains("if projection_w == 0.0"));
        assert!(helpers.contains("else if projection_w != 1.0"));
        assert!(helpers.contains("radius_world = radius_world * depth;"));
        assert!(helpers.contains("return -1.0;"));
        assert!(helpers.contains("* GAUSSIAN_SHADER_COORDINATE_UNITS_PER_PIXEL"));
        assert!(helpers.contains("fn gaussian_mip_filter_covariance_2d"));
        assert!(helpers.contains("original_determinant / filtered_determinant"));
        assert!(helpers.contains("sqrt(clamp(determinant_ratio, 0.0, 1.0))"));
        assert!(!helpers.contains("cov[0][0] += 0.3f"));
        assert!(helpers.contains("if view.clip_from_view[3].w == 1.0"));
        assert!(helpers.contains(
            "focal.x, 0.0, 0.0,\n            0.0, -focal.y, 0.0,\n            0.0, 0.0, 0.0"
        ));
        assert!(helpers.contains("focal.x / t.z, 0.0, -(focal.x * t.x) * s"));
        assert!(helpers.contains("0.0, -focal.y / t.z, (focal.y * t.y) * s"));

        let gaussian_3d = include_str!("gaussian_3d.wgsl");
        assert!(gaussian_3d.contains("struct GaussianMipCovariance2d"));
        assert!(gaussian_3d.contains("filtered: vec4<f32>"));

        let gaussian = include_str!("gaussian.wgsl");
        assert!(gaussian.contains("fn gaussian_support_cutoff"));
        assert!(gaussian.contains(
            "#ifdef LOD_CANDIDATE\n    // Portable pages accept every finite authored opacity."
        ));
        assert!(gaussian.contains("return GAUSSIAN_AUTHORED_SUPPORT_SIGMA;"));
        assert!(gaussian.contains("gaussian_mip_support_radius_world("));
        assert!(gaussian.contains("authored_radius_world + mip_radius_world"));
        let compaction = include_str!("lod_compaction.wgsl");
        assert!(
            compaction.contains(
                "let local_radius = 3.0 * abs(gaussian_uniforms.global_scale) * max_scale;"
            )
        );
        assert!(compaction.contains("gaussian_mip_support_radius_world(position_world, 3.0)"));
        assert!(compaction.contains("authored_radius_world + mip_radius_world"));
        assert!(gaussian.contains("opacity = opacity * gaussian_mip.filtered.w;"));
        assert!(gaussian.contains("opacity = opacity * gaussian_mip.w;"));
        assert!(gaussian.contains("lod_morph_fragment_color("));
    }

    #[test]
    fn cov2d_projection_and_filter_units_match_physical_pixels() {
        const SHADER_COORDINATE_UNITS_PER_PIXEL: f32 = 2.0;
        const SHADER_COVARIANCE_SCALE: f32 =
            SHADER_COORDINATE_UNITS_PER_PIXEL * SHADER_COORDINATE_UNITS_PER_PIXEL;

        fn covariance_bilinear(left: [f32; 3], covariance: [f32; 6], right: [f32; 3]) -> f32 {
            left[0]
                * (covariance[0] * right[0] + covariance[1] * right[1] + covariance[2] * right[2])
                + left[1]
                    * (covariance[1] * right[0]
                        + covariance[3] * right[1]
                        + covariance[4] * right[2])
                + left[2]
                    * (covariance[2] * right[0]
                        + covariance[4] * right[1]
                        + covariance[5] * right[2])
        }

        fn projected_covariance(
            derivative_x: [f32; 3],
            derivative_y: [f32; 3],
            covariance: [f32; 6],
        ) -> [f32; 3] {
            [
                covariance_bilinear(derivative_x, covariance, derivative_x),
                covariance_bilinear(derivative_x, covariance, derivative_y),
                covariance_bilinear(derivative_y, covariance, derivative_y),
            ]
        }

        fn assert_covariance_close(actual: [f32; 3], expected: [f32; 3]) {
            for (actual, expected) in actual.into_iter().zip(expected) {
                let tolerance = 2.0e-6 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "actual={actual} expected={expected} tolerance={tolerance}"
                );
            }
        }

        let covariance_3d = [0.04_f32, 0.01, 0.005, 0.09, -0.003, 0.16];

        // Perspective uses the historical full-viewport focal. Its two shader
        // units per physical pixel scale every Jacobian component by two and
        // every projected covariance component by four.
        let shader_focal = [1.25_f32 * 800.0, 1.5 * 600.0];
        let view_position = [2.0_f32, -1.0, -10.0];
        let inverse_depth_squared = 1.0 / (view_position[2] * view_position[2]);
        let perspective_shader_j = [
            [
                shader_focal[0] / view_position[2],
                0.0,
                -(shader_focal[0] * view_position[0]) * inverse_depth_squared,
            ],
            [
                0.0,
                -shader_focal[1] / view_position[2],
                (shader_focal[1] * view_position[1]) * inverse_depth_squared,
            ],
        ];
        let perspective_physical_j = perspective_shader_j
            .map(|derivative| derivative.map(|value| value / SHADER_COORDINATE_UNITS_PER_PIXEL));
        let perspective_shader = projected_covariance(
            perspective_shader_j[0],
            perspective_shader_j[1],
            covariance_3d,
        );
        let perspective_physical = projected_covariance(
            perspective_physical_j[0],
            perspective_physical_j[1],
            covariance_3d,
        );
        assert_covariance_close(perspective_physical, [121.0, -12.825, 186.705]);
        assert_covariance_close(
            perspective_shader.map(|value| value / SHADER_COVARIANCE_SCALE),
            perspective_physical,
        );

        // Orthographic projection is affine: the same full-viewport focal is
        // used, but it has no inverse-depth or view-z coupling at any depth.
        let orthographic_shader_j = [
            [0.02_f32 * 800.0, 0.0, 0.0],
            [0.0, -(0.03_f32 * 600.0), 0.0],
        ];
        let orthographic_physical_j = orthographic_shader_j
            .map(|derivative| derivative.map(|value| value / SHADER_COORDINATE_UNITS_PER_PIXEL));
        let orthographic_shader = projected_covariance(
            orthographic_shader_j[0],
            orthographic_shader_j[1],
            covariance_3d,
        );
        let orthographic_physical = projected_covariance(
            orthographic_physical_j[0],
            orthographic_physical_j[1],
            covariance_3d,
        );
        assert_covariance_close(orthographic_physical, [2.56, -0.72, 7.29]);
        for _view_depth in [-2.0_f32, -20.0] {
            assert_covariance_close(
                orthographic_shader.map(|value| value / SHADER_COVARIANCE_SCALE),
                orthographic_physical,
            );
        }

        let shader_filtered = [
            orthographic_shader[0] + 1.2,
            orthographic_shader[1],
            orthographic_shader[2] + 1.2,
        ]
        .map(|value| value / SHADER_COVARIANCE_SCALE);
        assert_covariance_close(
            shader_filtered,
            [
                orthographic_physical[0] + GAUSSIAN_MIP_FILTER_VARIANCE_2D,
                orthographic_physical[1],
                orthographic_physical[2] + GAUSSIAN_MIP_FILTER_VARIANCE_2D,
            ],
        );
    }

    #[test]
    fn gaussian_bounds_math_is_finite_and_matches_the_covariance_inverse() {
        fn obb_axes(covariance: [f32; 3]) -> ([f32; 2], [f32; 2]) {
            let determinant = covariance[0] * covariance[2] - covariance[1] * covariance[1];
            let midpoint = 0.5 * (covariance[0] + covariance[2]);
            let discriminant = (midpoint * midpoint - determinant).max(0.0);
            let lambda1 = midpoint + discriminant.sqrt();
            let candidate = [-covariance[1], lambda1 - covariance[0]];
            let candidate_l1 = candidate[0].abs() + candidate[1].abs();
            let major = if candidate_l1 > 1.0e-12 {
                let inverse_length =
                    1.0 / (candidate[0] * candidate[0] + candidate[1] * candidate[1]).sqrt();
                [candidate[0] * inverse_length, candidate[1] * inverse_length]
            } else {
                [1.0, 0.0]
            };
            (major, [major[1], -major[0]])
        }

        for covariance in [
            [9.0_f32, 0.0, 4.0],
            [4.0, 0.0, 9.0],
            [4.0, 0.0, 4.0],
            [4.0, 1.0e-7, 4.0],
        ] {
            let (major, minor) = obb_axes(covariance);
            assert!(major.into_iter().chain(minor).all(f32::is_finite));
            assert!((major[0] * major[0] + major[1] * major[1] - 1.0).abs() <= 1.0e-6);
            assert!((minor[0] * minor[0] + minor[1] * minor[1] - 1.0).abs() <= 1.0e-6);
            assert!((major[0] * minor[0] + major[1] * minor[1]).abs() <= 1.0e-6);
            let handedness = major[0] * minor[1] - major[1] * minor[0];
            assert!((handedness + 1.0).abs() <= 1.0e-6);
        }
        assert_eq!(obb_axes([9.0, 0.0, 4.0]).0, [1.0, 0.0]);
        assert_eq!(obb_axes([4.0, 0.0, 9.0]).0, [0.0, 1.0]);

        let covariance = [4.0_f32, 1.0, 2.0];
        let delta = [2.0_f32, 1.0];
        let determinant = covariance[0] * covariance[2] - covariance[1] * covariance[1];
        let conic = [
            covariance[2] / determinant,
            -covariance[1] / determinant,
            covariance[0] / determinant,
        ];
        let shader_power = -0.5
            * (conic[0] * delta[0] * delta[0]
                + 2.0 * conic[1] * delta[0] * delta[1]
                + conic[2] * delta[1] * delta[1]);
        let inverse_covariance_power = -0.5
            * (covariance[2] * delta[0] * delta[0] - 2.0 * covariance[1] * delta[0] * delta[1]
                + covariance[0] * delta[1] * delta[1])
            / determinant;
        assert!((shader_power - inverse_covariance_power).abs() <= 1.0e-7);

        for cutoff in [0.5_f32, 1.0, 3.0] {
            let power_at_unit_radius = -0.5 * cutoff * cutoff;
            assert_eq!(
                (-0.5 * (cutoff * cutoff) * 1.0).to_bits(),
                power_at_unit_radius.to_bits()
            );
        }

        let helpers = include_str!("helpers.wgsl");
        assert!(helpers.contains("let major_axis_candidate = vec2<f32>"));
        assert!(helpers.contains("var eigvec1 = vec2<f32>(1.0, 0.0)"));
        assert!(
            helpers.contains("abs(major_axis_candidate.x) + abs(major_axis_candidate.y) > 1.0e-12")
        );
        assert!(helpers.contains("eigvec1.y,\n        -eigvec1.x"));

        let gaussian = include_str!("gaussian.wgsl");
        assert_eq!(gaussian.matches("cutoff_squared: f32").count(), 6);
        assert_eq!(
            gaussian
                .matches("output.cutoff_squared = cutoff * cutoff")
                .count(),
            1
        );
        assert!(gaussian.contains("@location(8) cutoff_squared: f32"));
        assert!(gaussian.contains("@location(8) @interpolate(flat) cutoff_squared: f32"));
        assert!(gaussian.contains("let power = -0.5 * input.cutoff_squared * distance_squared"));
        assert!(!gaussian.contains("distance_squared > 3.0 * 3.0"));
        assert_eq!(gaussian.matches("+ 2.0 * conic.y * d.x * d.y").count(), 2);
    }

    #[test]
    fn lod_candidate_pipeline_key_is_explicit_and_opt_in() {
        let disabled = shader_defs(CloudPipelineKey::default());
        assert!(
            !disabled
                .iter()
                .any(|define| format!("{define:?}").contains("LOD_CANDIDATE"))
        );

        let enabled = shader_defs(CloudPipelineKey {
            lod_candidate: true,
            ..Default::default()
        });
        assert!(
            enabled
                .iter()
                .any(|define| format!("{define:?}").contains("LOD_CANDIDATE"))
        );
        #[cfg(lod_render_path)]
        assert!(
            enabled
                .iter()
                .any(|define| format!("{define:?}").contains("LOD_MORPH"))
        );
    }

    #[test]
    fn lod_morph_fragment_radiance_preserves_both_filtered_endpoints_per_pixel() {
        const ALPHA_LIMIT: f32 = 0.999_999;
        const FRAGMENT_ALPHA_LIMIT: f32 = 0.999;

        fn optical_depth(opacity: f32) -> f32 {
            let bounded = opacity.clamp(0.0, ALPHA_LIMIT);
            -(1.0 - bounded).max(1.0 - ALPHA_LIMIT).ln()
        }

        fn fragment_color(
            parent: (f32, f32, [f32; 3]),
            child: (f32, f32, [f32; 3]),
            gaussian_weight: f32,
            morph_blend_t: f32,
        ) -> ([f32; 3], f32) {
            let (parent_peak_alpha, parent_coefficient, parent_linear_rgb) = parent;
            let (child_peak_alpha, child_coefficient, child_linear_rgb) = child;
            let parent_alpha =
                (gaussian_weight * parent_peak_alpha).clamp(0.0, FRAGMENT_ALPHA_LIMIT);
            let child_alpha = (gaussian_weight * child_peak_alpha).clamp(0.0, FRAGMENT_ALPHA_LIMIT);
            if morph_blend_t >= 1.0 && parent_coefficient <= 0.0 && child_coefficient >= 1.0 {
                return (
                    child_linear_rgb.map(|channel| channel * child_alpha),
                    child_alpha,
                );
            }
            if morph_blend_t <= 0.0 && child_coefficient <= 0.0 && parent_coefficient >= 1.0 {
                return (
                    parent_linear_rgb.map(|channel| channel * parent_alpha),
                    parent_alpha,
                );
            }
            let parent_tau = parent_coefficient.max(0.0) * optical_depth(parent_alpha);
            let child_tau = child_coefficient.max(0.0) * optical_depth(child_alpha);
            let total_tau = parent_tau + child_tau;
            if total_tau <= 0.0 {
                return ([0.0; 3], 0.0);
            }
            let alpha = (1.0 - (-total_tau).exp()).min(FRAGMENT_ALPHA_LIMIT);
            let linear_rgb = std::array::from_fn(|channel| {
                (parent_linear_rgb[channel] * parent_tau + child_linear_rgb[channel] * child_tau)
                    / total_tau
            });
            (linear_rgb.map(|channel| channel * alpha), alpha)
        }

        fn srgb_to_linear(channel: f32) -> f32 {
            if channel <= 0.040_45 {
                channel / 12.92
            } else {
                ((channel + 0.055) / 1.055).powf(2.4)
            }
        }

        let parent_opacity = 0.73_f32;
        let child_opacity = 0.19_f32;
        let parent_linear_rgb = [0.8_f32, 0.13, 0.04];
        let child_linear_rgb = [0.03_f32, 0.37, 0.91];
        let parent_mip = gaussian_mip_filter_covariance_2d([0.018, 0.004, 3.7]).opacity_scale;
        let child_mip = gaussian_mip_filter_covariance_2d([1.4, -0.12, 0.8]).opacity_scale;
        assert!(parent_mip < 1.0 && child_mip < 1.0);

        for global_opacity in [0.0_f32, 0.25, 1.0] {
            let parent_peak = (global_opacity * parent_opacity * parent_mip).clamp(0.0, 1.0);
            let child_peak = (global_opacity * child_opacity * child_mip).clamp(0.0, 1.0);
            for run_length in [1_u32, 3, 17] {
                for gaussian_weight in [1.0_f32, 0.75, 0.25, 0.01, 0.000_01] {
                    let (parent_premultiplied, parent_share) = fragment_color(
                        (parent_peak, 1.0 / run_length as f32, parent_linear_rgb),
                        (child_peak, 0.0, child_linear_rgb),
                        gaussian_weight,
                        0.0,
                    );
                    let recomposed = 1.0 - (1.0 - parent_share).powi(run_length as i32);
                    let expected_parent =
                        (gaussian_weight * parent_peak).clamp(0.0, FRAGMENT_ALPHA_LIMIT);
                    assert!(
                        (recomposed - expected_parent).abs() <= 2.0e-6,
                        "g={global_opacity} weight={gaussian_weight} run={run_length} share={parent_share} recomposed={recomposed} expected={expected_parent}"
                    );
                    if parent_share > 0.0 {
                        for channel in 0..3 {
                            assert!(
                                (parent_premultiplied[channel] / parent_share
                                    - parent_linear_rgb[channel])
                                    .abs()
                                    <= 2.0e-6,
                                "parent endpoint changed linear radiance"
                            );
                        }
                    }

                    let (child_premultiplied, child_endpoint) = fragment_color(
                        (parent_peak, 0.0, parent_linear_rgb),
                        (child_peak, 1.0, child_linear_rgb),
                        gaussian_weight,
                        1.0,
                    );
                    assert_eq!(
                        child_endpoint.to_bits(),
                        (gaussian_weight * child_peak)
                            .clamp(0.0, FRAGMENT_ALPHA_LIMIT)
                            .to_bits()
                    );
                    assert_eq!(
                        child_premultiplied.map(f32::to_bits),
                        child_linear_rgb
                            .map(|channel| channel * child_endpoint)
                            .map(f32::to_bits),
                    );
                }
            }
        }

        for (parent, child) in [(0.000_01, 0.000_001), (0.001, 0.000_1)] {
            let (premultiplied, alpha) = fragment_color(
                (parent, 0.3 / 11.0, parent_linear_rgb),
                (child, 0.7, child_linear_rgb),
                0.01,
                0.7,
            );
            assert!(alpha.is_finite() && alpha >= 0.0);
            assert!(premultiplied.into_iter().all(f32::is_finite));
        }

        // An open-interval projected-area correction may exceed one when an
        // endpoint and the interpolated proxy have substantially different
        // depths/areas. Selected can gate the other endpoint to zero; that is
        // still an interior optical-depth scale, never an endpoint fast path.
        let child_peak = 0.42_f32;
        let gaussian_weight = 0.8_f32;
        let child_area_coefficient = 3.25_f32;
        let (interior_premultiplied, interior_alpha) = fragment_color(
            (0.77, 0.0, parent_linear_rgb),
            (child_peak, child_area_coefficient, child_linear_rgb),
            gaussian_weight,
            0.25,
        );
        let child_alpha = (gaussian_weight * child_peak).clamp(0.0, FRAGMENT_ALPHA_LIMIT);
        let expected_interior_alpha = (1.0
            - (-child_area_coefficient * optical_depth(child_alpha)).exp())
        .min(FRAGMENT_ALPHA_LIMIT);
        assert!((interior_alpha - expected_interior_alpha).abs() <= 2.0e-6);
        assert!((interior_alpha - child_alpha).abs() > 0.1);
        for channel in 0..3 {
            assert!(
                (interior_premultiplied[channel] - child_linear_rgb[channel] * interior_alpha)
                    .abs()
                    <= 2.0e-6
            );
        }

        // Convert endpoint SH results to linear light independently. Applying
        // the nonlinear transfer after mixing encoded endpoint colors is not
        // equivalent and would bias interior LoD radiance.
        let parent_srgb = [0.05_f32, 0.25, 0.8];
        let child_srgb = [0.9_f32, 0.1, 0.02];
        let parent_linear = parent_srgb.map(srgb_to_linear);
        let child_linear = child_srgb.map(srgb_to_linear);
        let (premultiplied, alpha) = fragment_color(
            (0.72, 0.4, parent_linear),
            (0.31, 0.6, child_linear),
            0.63,
            0.6,
        );
        let correct_linear = premultiplied.map(|channel| channel / alpha);
        let wrong_encoded_then_linear: [f32; 3] = std::array::from_fn(|channel| {
            srgb_to_linear(parent_srgb[channel] * 0.4 + child_srgb[channel] * 0.6)
        });
        assert!(
            correct_linear
                .into_iter()
                .zip(wrong_encoded_then_linear)
                .any(|(correct, wrong)| (correct - wrong).abs() > 0.02),
            "test colors did not expose nonlinear sRGB interpolation bias"
        );

        // For a fixed normalized support, integrating optical depth over a
        // projected Gaussian contributes its sqrt(det(covariance)) area. The
        // endpoint/current area ratios therefore cancel the interpolated area
        // exactly and leave a linear parent/child mass blend.
        let parent_area = 7.0_f32;
        let child_area = 2.5_f32;
        let parent_unit_tau = 1.7_f32;
        let child_unit_tau = 0.4_f32;
        let run_length = 5.0_f32;
        for (t, current_area) in [(0.0_f32, 7.0_f32), (0.2, 5.8), (0.5, 4.0), (1.0, 2.5)] {
            let parent_coefficient = if t <= 0.0 {
                1.0 / run_length
            } else if t >= 1.0 {
                0.0
            } else {
                (1.0 - t) * parent_area / current_area / run_length
            };
            let child_coefficient = if t <= 0.0 {
                0.0
            } else if t >= 1.0 {
                1.0
            } else {
                t * child_area / current_area
            };
            let integrated_proxy_mass = current_area
                * (parent_coefficient * parent_unit_tau + child_coefficient * child_unit_tau);
            let expected = (1.0 - t) * parent_area * parent_unit_tau / run_length
                + t * child_area * child_unit_tau;
            assert!((integrated_proxy_mass - expected).abs() <= 2.0e-6);
        }

        let shader = include_str!("lod_morph.wgsl");
        assert!(shader.contains("fn lod_morph_fragment_color"));
        assert!(shader.contains("gaussian_weight * parent_peak_alpha"));
        assert!(shader.contains("gaussian_weight * child_peak_alpha"));
        assert!(shader.contains("morph_blend_t >= 1.0"));
        assert!(shader.contains("morph_blend_t <= 0.0"));
        assert!(shader.contains("parent_optical_depth_coefficient"));
        assert!(shader.contains("parent_linear_rgb * parent_tau"));
        assert!(shader.contains("child_linear_rgb * child_tau"));
        assert!(shader.contains("double-density halo"));
        assert!(!shader.contains("fn lod_morph_spherical_harmonics"));

        let gaussian = include_str!("gaussian.wgsl");
        assert!(gaussian.contains("@interpolate(flat) lod_morph_alpha: vec4<f32>"));
        assert!(
            gaussian.contains("@location(9) @interpolate(flat) lod_morph_parent_color: vec4<f32>")
        );
        assert_eq!(gaussian.matches("@location(9)").count(), 2);
        const LOD_MORPH_INTER_STAGE_LOCATION_COUNT: u32 = 10;
        const WEBGPU_MIN_MAX_INTER_STAGE_SHADER_VARIABLES: u32 = 16;
        const WEBGL2_MIN_MAX_VARYING_VECTORS: u32 = 15;
        const {
            assert!(
                LOD_MORPH_INTER_STAGE_LOCATION_COUNT <= WEBGPU_MIN_MAX_INTER_STAGE_SHADER_VARIABLES
            );
            assert!(LOD_MORPH_INTER_STAGE_LOCATION_COUNT <= WEBGL2_MIN_MAX_VARYING_VECTORS);
        }
        assert!(gaussian.contains("gaussian_mip.parent_opacity_scale"));
        assert!(gaussian.contains("gaussian_mip.child_opacity_scale"));
        assert!(gaussian.contains("gaussian_mip.parent_projected_area_ratio"));
        assert!(gaussian.contains("gaussian_mip.child_projected_area_ratio"));
        assert!(gaussian.contains("return lod_morph_fragment_color("));
        assert!(gaussian.contains("input.lod_morph_parent_color.w"));
        assert!(gaussian.contains("input.lod_morph_parent_color.rgb"));
        assert!(gaussian.contains("gaussian_render_color_at("));
        assert!(gaussian.contains("parent_transformed_position"));
        assert!(gaussian.contains("child_transformed_position"));
        assert!(!gaussian.contains("lod_morph_spherical_harmonics"));

        let gaussian_3d = include_str!("gaussian_3d.wgsl");
        assert!(gaussian_3d.contains("fn projected_area_ratio"));
        assert!(gaussian_3d.contains("endpoint_determinant / current_determinant"));
    }

    #[test]
    fn lod_morph_support_cutoff_preserves_endpoint_union_and_candidate_three_sigma_policy() {
        // Keep the standalone union helper endpoint-exact and conservative for
        // unequal inputs even though production LoD candidate inputs are both
        // the fixed authored cutoff.
        let parent = 1.75_f32;
        let child = 3.5_f32;
        assert_ne!(parent.to_bits(), child.to_bits());
        let cutoff = |t: f32| {
            if t <= 0.0 {
                parent
            } else if t >= 1.0 {
                child
            } else {
                parent.max(child)
            }
        };
        assert_eq!(cutoff(0.0).to_bits(), parent.to_bits());
        assert_eq!(cutoff(1.0).to_bits(), child.to_bits());
        for t in [f32::MIN_POSITIVE, 0.25, 0.5, 0.75, 1.0 - f32::EPSILON] {
            assert_eq!(cutoff(t).to_bits(), parent.max(child).to_bits());
        }
        for opacity in [0.08_f32, 0.91, 2.0, f32::MAX] {
            assert_eq!(
                gaussian_support_cutoff(opacity, true, true),
                GAUSSIAN_AUTHORED_SUPPORT_SIGMA
            );
        }

        let morph = include_str!("lod_morph.wgsl");
        assert!(morph.contains("fn lod_morph_support_cutoff"));
        assert!(morph.contains("return max(parent_cutoff, child_cutoff)"));
        let shader = include_str!("gaussian.wgsl");
        assert!(shader.contains("var cutoff = gaussian_support_cutoff(opacity)"));
        assert!(shader.contains("let parent_cutoff = gaussian_support_cutoff("));
        assert!(shader.contains("cutoff = lod_morph_support_cutoff("));
        assert!(!shader.contains("cutoff = mix(parent_cutoff, cutoff, morph_blend_t)"));
        assert!(shader.contains("output.cutoff_squared = cutoff * cutoff"));
        assert!(shader.contains("gaussian_support_radius_world("));
        assert!(shader.contains("get_bounding_box_clip("));
    }

    #[test]
    fn lod_morph_visibility_is_endpoint_exact_and_retains_the_open_interval_union() {
        fn visibility(parent: f32, child: f32, t: f32) -> f32 {
            if t <= 0.0 {
                parent
            } else if t >= 1.0 {
                child
            } else {
                parent.max(child)
            }
        }

        for (parent, child) in [(1.0_f32, 0.0_f32), (0.0, 1.0), (4.0, 2.0)] {
            assert_eq!(visibility(parent, child, 0.0).to_bits(), parent.to_bits());
            assert_eq!(visibility(parent, child, 1.0).to_bits(), child.to_bits());
            for t in [f32::MIN_POSITIVE, 0.25, 0.5, 0.75, 1.0 - f32::EPSILON] {
                assert_eq!(
                    visibility(parent, child, t).to_bits(),
                    parent.max(child).to_bits()
                );
            }
        }

        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum DrawPolicy {
            All,
            Selected,
            HighlightSelected,
        }
        let contributes =
            |policy: DrawPolicy, value: f32| policy != DrawPolicy::Selected || value >= 0.5;
        let base_coefficients = |t: f32| {
            if t <= 0.0 {
                (1.0_f32, 0.0_f32)
            } else if t >= 1.0 {
                (0.0, 1.0)
            } else {
                (0.4, 0.6)
            }
        };
        let coefficients = |policy: DrawPolicy, parent: f32, child: f32, t: f32| {
            let (parent_base, child_base) = base_coefficients(t);
            (
                parent_base
                    * if contributes(policy, parent) {
                        1.0
                    } else {
                        0.0
                    },
                child_base * if contributes(policy, child) { 1.0 } else { 0.0 },
            )
        };

        // All and HighlightSelected render zero-valued selection metadata at
        // both exact cuts and throughout the open interval. Highlight is a
        // color annotation only.
        for policy in [DrawPolicy::All, DrawPolicy::HighlightSelected] {
            for (parent, child) in [(0.0_f32, 0.0_f32), (0.0, 1.0), (1.0, 0.0)] {
                for t in [0.0_f32, 0.4, 1.0] {
                    assert_eq!(coefficients(policy, parent, child, t), base_coefficients(t));
                }
            }
        }

        // Selected alone gates each optical-depth endpoint at the established
        // inclusive 0.5 threshold while the open-interval proxy retains their
        // endpoint visibility union.
        for (parent, child, expected_open) in [
            (0.0_f32, 1.0_f32, (0.0_f32, 0.6_f32)),
            (1.0, 0.0, (0.4, 0.0)),
            (0.25, 0.5, (0.0, 0.6)),
            (0.5, 0.49, (0.4, 0.0)),
        ] {
            assert_eq!(
                coefficients(DrawPolicy::Selected, parent, child, 0.4),
                expected_open
            );
            assert_eq!(
                visibility(parent, child, 0.4) >= 0.5,
                parent >= 0.5 || child >= 0.5
            );
            assert_eq!(
                coefficients(DrawPolicy::Selected, parent, child, 0.0).0 > 0.0,
                parent >= 0.5
            );
            assert_eq!(
                coefficients(DrawPolicy::Selected, parent, child, 1.0).1 > 0.0,
                child >= 0.5
            );
        }

        let morph = include_str!("lod_morph.wgsl");
        assert!(morph.contains("fn lod_morph_visibility"));
        assert!(morph.contains("return max(parent_visibility, child_visibility)"));

        let compaction = include_str!("lod_compaction.wgsl");
        assert!(!compaction.contains("get_visibility"));
        assert!(!compaction.contains("visibility <= 0.0"));
        assert!(!compaction.contains("lod_morph_visibility"));

        let gaussian = include_str!("gaussian.wgsl");
        assert!(gaussian.contains("displayed_visibility = lod_morph_visibility("));
        assert!(gaussian.contains("fn lod_morph_visibility_contributes"));
        assert!(gaussian.contains("return visibility >= 0.5"));
        assert!(gaussian.contains("return true;"));
        assert!(!gaussian.contains("return visibility > 0.0"));
        assert!(gaussian.contains("lod_morph_visibility_contributes(parent_visibility)"));
        assert!(gaussian.contains("lod_morph_visibility_contributes(child_visibility)"));
        assert!(gaussian.contains("discard_quad |= displayed_visibility < 0.5"));
        assert!(gaussian.contains("if parent_visibility > 0.5"));
        assert!(gaussian.contains("if child_visibility > 0.5"));
        assert!(!gaussian.contains("if (displayed_visibility > 0.5)"));
    }

    #[test]
    fn lod_morph_covariance_and_support_are_representation_independent() {
        fn interpolate_covariance(parent: [f32; 6], child: [f32; 6], t: f32) -> [f32; 6] {
            std::array::from_fn(|index| parent[index] + (child[index] - parent[index]) * t)
        }

        // These are endpoint local covariances produced by the standard
        // scale/rotation path and persisted by the precompute path.
        let parent = [4.0_f32, 0.3, -0.2, 1.5, 0.1, 0.7];
        let child = [0.8_f32, -0.15, 0.05, 2.7, -0.4, 3.2];
        for t in [0.0_f32, 0.125, 0.5, 0.875, 1.0] {
            let standard = interpolate_covariance(parent, child, t);
            let precomputed = interpolate_covariance(parent, child, t);
            assert_eq!(standard.map(f32::to_bits), precomputed.map(f32::to_bits));
        }

        let parent_max_scale = 4.0_f32;
        let child_max_scale = 2.5_f32;
        for t in [0.0_f32, 0.2, 0.5, 0.9, 1.0] {
            let support =
                ((1.0 - t) * parent_max_scale.powi(2) + t * child_max_scale.powi(2)).sqrt();
            let covariance_spectral_bound =
                (1.0 - t) * parent_max_scale.powi(2) + t * child_max_scale.powi(2);
            assert!(support * support + 1.0e-5 >= covariance_spectral_bound);
        }

        let shader = include_str!("gaussian_3d.wgsl");
        assert!(shader.contains("let child_local_cov3d"));
        assert!(shader.contains("parent_local_cov3d"));
        assert!(shader.contains("local_cov3d = lod_morph_covariance("));
        assert!(shader.contains("transform_local_cov3d(local_cov3d)"));
        assert!(!shader.contains("scale = lod_morph_log_scale("));

        let morph = include_str!("lod_morph.wgsl");
        assert!(morph.contains("fn lod_morph_support_max_scale"));
        assert!(morph.contains("parent_max * parent_max"));
        assert!(morph.contains("child_max * child_max"));
    }

    #[test]
    fn lod_morph_table_uses_one_bounded_per_view_weight_for_compaction_and_raster() {
        let morph = include_str!("lod_morph.wgsl");
        for contract in [
            "arrayLength(&lod_morph_words)",
            "lod_presentation_mode() != LOD_PRESENTATION_MODE_MORPH",
            "descriptor_count > descriptor_capacity",
            "mapping_record_count > mapping_capacity",
            "weight_count > weight_capacity",
            "parent_physical_index >= source_count",
            "edge_index >= weight_count",
            "bitcast<f32>(lod_morph_words[weight_start + edge_index])",
        ] {
            assert!(
                morph.contains(contract),
                "missing morph contract: {contract}"
            );
        }

        let gaussian = include_str!("gaussian.wgsl");
        assert!(gaussian.contains("const LOD_ENTRY_SOURCE_INDEX_MASK: u32 = 0x0fffffffu;"));
        assert!(gaussian.contains("const LOD_ENTRY_PRESENTATION_CLASS_SHIFT: u32 = 28u;"));
        assert!(gaussian.contains("const LOD_ENTRY_PRESENTATION_CLASS_MASK: u32 = 3u << LOD_ENTRY_PRESENTATION_CLASS_SHIFT;"));
        assert!(gaussian.contains("if lod_morph_from_entry(entry)"));
        assert!(gaussian.contains("let morph = lod_morph_sample("));
        let compaction = include_str!("lod_compaction.wgsl");
        assert!(compaction.contains("const LOD_ENTRY_SOURCE_INDEX_MASK: u32 = 0x0fffffffu;"));
        assert!(compaction.contains("const LOD_ENTRY_PRESENTATION_CLASS_SHIFT: u32 = 28u;"));
        assert!(compaction.contains("let range_metadata = candidate_and_scan_words[word + 3u];"));
        assert!(compaction.contains("let residency = range_metadata & 3u;"));
        assert!(compaction.contains("let presentation_class = (range_metadata >> 2u) & 3u;"));
        assert!(compaction.contains("let morph = lod_morph_sample("));
        assert!(!morph.contains("progress_bits"));
        assert!(!morph.contains("COARSEN_BIT"));
    }

    #[test]
    fn external_active_set_presentation_scales_only_final_peak_opacity() {
        let morph = include_str!("lod_morph.wgsl");
        for contract in [
            "const LOD_PRESENTATION_MODE_MORPH: u32 = 1u;",
            "const LOD_PRESENTATION_MODE_EXTERNAL_ACTIVE_SET: u32 = 2u;",
            "fn lod_external_active_set_opacity_coefficient(active_set_class: u32) -> f32",
            "if lod_presentation_mode() != LOD_PRESENTATION_MODE_EXTERNAL_ACTIVE_SET",
            "lod_morph_words[LOD_PRESENTATION_FIRST_WEIGHT_WORD]",
            "lod_morph_words[LOD_PRESENTATION_SECOND_WEIGHT_WORD]",
        ] {
            assert!(
                morph.contains(contract),
                "missing presentation contract: {contract}"
            );
        }
        let sample = morph
            .split("fn lod_morph_sample(")
            .nth(1)
            .expect("morph sampler");
        let mode_guard = sample
            .find("lod_presentation_mode() != LOD_PRESENTATION_MODE_MORPH")
            .expect("morph mode guard");
        let table_parse = sample
            .find("let descriptor_count = lod_morph_words[0u]")
            .expect("morph descriptor parse");
        assert!(
            mode_guard < table_parse,
            "external class 1 must never parse as morph"
        );

        let gaussian = include_str!("gaussian.wgsl");
        let authored = gaussian
            .find("var output_opacity = clamp(opacity * gaussian_uniforms.global_opacity")
            .expect("final authored/global/Mip opacity");
        let external = gaussian
            .find("output_opacity = output_opacity * lod_external_active_set_opacity_coefficient(")
            .expect("external opacity coefficient");
        let output = gaussian
            .find("output.color = vec4<f32>(rgb, output_opacity)")
            .expect("vertex output opacity");
        assert!(authored < external && external < output);
        assert_eq!(
            gaussian
                .matches("lod_external_active_set_opacity_coefficient(")
                .count(),
            1,
            "the external coefficient is applied exactly once to final peak opacity"
        );

        let coefficient = |class: u32, first: f32, second: f32| match class {
            0 => 1.0,
            1 => first,
            2 => second,
            _ => 0.0,
        };
        for (first, second) in [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)] {
            assert_eq!(coefficient(0, first, second), 1.0);
            assert_eq!(coefficient(1, first, second), first);
            assert_eq!(coefficient(2, first, second), second);
            assert_eq!(coefficient(3, first, second), 0.0);
        }

        let render = include_str!("mod.rs");
        let queue = render
            .split("fn queue_gaussians")
            .nth(1)
            .and_then(|body| body.split("pub struct CloudPipeline").next())
            .expect("Gaussian queue path");
        assert!(queue.contains("candidate.is_external_active_set()"));
        assert!(queue.contains("settings.lod_debug.requires_metadata() && !external_active_set"));
        let draw = render
            .split(
                "impl<P: PhaseItem, R: PlanarSync> RenderCommand<P> for DrawGaussianInstanced<R>",
            )
            .nth(1)
            .expect("Gaussian indirect draw path");
        assert!(draw.contains("candidate.is_external_active_set()"));
        assert!(draw.contains("&& !external_active_set"));

        let compaction = include_str!("lod_compaction.wgsl");
        assert!(compaction.contains("active_entries[output_index] = evaluation.entry;"));
        assert_eq!(
            compaction.matches("pack_lod_entry_value(source)").count(),
            1
        );
    }

    #[test]
    fn precomputed_debug_morph_stays_within_webgpu_vertex_storage_minimum() {
        const PRECOMPUTED_GAUSSIAN_PLANES: u32 = 5;
        const SORTED_AND_MORPH_BINDINGS: u32 = 2;
        const DEBUG_STORAGE_BINDINGS: u32 = 1;
        const WEBGPU_MIN_STORAGE_BUFFERS_PER_SHADER_STAGE: u32 = 8;
        assert_eq!(
            PRECOMPUTED_GAUSSIAN_PLANES + SORTED_AND_MORPH_BINDINGS + DEBUG_STORAGE_BINDINGS,
            WEBGPU_MIN_STORAGE_BUFFERS_PER_SHADER_STAGE
        );
    }

    #[test]
    fn raster_frustum_culling_uses_gaussian_support_not_only_its_center() {
        let shader = include_str!("gaussian.wgsl");
        assert!(shader.contains("fn gaussian_support_radius_world"));
        assert!(shader.contains("fn gaussian_support_sphere_in_frustum"));
        assert!(shader.contains("gaussian_uniforms.transform_scale_bound"));
        assert!(shader.contains("gaussian_mip_support_radius_world("));
        assert!(shader.contains("authored_radius_world + mip_radius_world"));
        assert!(!shader.contains("let gram_xx"));
        assert!(!shader.contains("length(plane.xyz)"));
        assert!(shader.contains("let projected_position = world_to_clip(transformed_position);"));
        assert!(shader.contains("discard_quad |= !gaussian_support_sphere_in_frustum"));
        assert!(!shader.contains("discard_quad |= !in_frustum(projected_position.xyz);"));
    }

    #[test]
    fn lod_debug_page_palette_is_rec709_luma_equalized_in_cpu_and_wgsl() {
        assert_eq!(
            LOD_DEBUG_PAGE_LINEAR_LUMINANCE.to_bits(),
            0.30_f32.to_bits()
        );
        for page in [1_u64, 2, 42, 0xffff, 0x1_0000_0001, u64::MAX] {
            let color = lod_debug_page_color(stable_page_color_key(LodPageId(page)));
            let luminance = color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722;
            assert!(
                (luminance - LOD_DEBUG_PAGE_LINEAR_LUMINANCE).abs() <= 2.0e-6,
                "Page {page} has nonuniform linear luma: color={color:?}, luma={luminance}"
            );
            assert!(
                color
                    .into_iter()
                    .all(|channel| (0.0..=1.0).contains(&channel))
            );
        }

        let shader = include_str!("lod_debug.wgsl");
        for contract in [
            "const LOD_DEBUG_PAGE_LINEAR_LUMINANCE: f32 = 0.30;",
            "const REC709_LINEAR_LUMINANCE: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);",
            "let base = lod_debug_hsv_to_rgb(hue, 0.72, 0.95);",
            "let luminance = dot(base, REC709_LINEAR_LUMINANCE);",
            "base * (LOD_DEBUG_PAGE_LINEAR_LUMINANCE / luminance)",
        ] {
            assert!(
                shader.contains(contract),
                "missing Page-palette contract: {contract}"
            );
        }
    }

    #[test]
    fn lod_debug_shader_is_opt_in_and_bounds_checks_metadata() {
        let requires_authored_color_contract =
            |mode: u32, metadata_in_bounds: bool, quality_policy_valid: bool| {
                if !metadata_in_bounds || mode == 0 {
                    return true;
                }
                match mode {
                    1..=3 => false,
                    4 => true,
                    5 => !quality_policy_valid,
                    _ => true,
                }
            };
        for mode in 1..=3 {
            assert!(!requires_authored_color_contract(mode, true, true));
        }
        assert!(requires_authored_color_contract(4, true, true));
        assert!(!requires_authored_color_contract(5, true, true));
        assert!(requires_authored_color_contract(5, true, false));
        assert!(requires_authored_color_contract(1, false, true));
        assert!(requires_authored_color_contract(0, true, true));

        let disabled = shader_defs(CloudPipelineKey::default());
        assert!(
            !disabled
                .iter()
                .any(|define| format!("{define:?}").contains("LOD_DEBUG"))
        );

        let enabled = shader_defs(CloudPipelineKey {
            lod_debug: true,
            ..Default::default()
        });
        assert!(
            enabled
                .iter()
                .any(|define| format!("{define:?}").contains("LOD_DEBUG"))
        );

        let shader = include_str!("lod_debug.wgsl");
        assert!(shader.contains("@group(4) @binding(0) var<storage, read> lod_debug_records"));
        assert!(shader.contains("splat_index >= metadata_count"));
        assert!(shader.contains("LOD_DEBUG_ORIGINAL_REPRESENTATION_BIT"));
        assert!(shader.contains("LOD_DEBUG_RESIDENCY_MASK: u32 = 0x0000ffffu"));
        assert!(shader.contains("fn lod_debug_high_fidelity_certificate"));
        assert!(shader.contains("record.residency >> LOD_DEBUG_CERTIFICATE_SHIFT"));
        assert!(shader.contains("lod_debug_residency(record, entry_residency)"));
        assert!(shader.contains("let distance = bitcast<f32>(distance_bits);"));
        assert!(shader.contains("fn apply_lod_debug_annotation"));
        assert!(shader.contains("fn apply_lod_debug_morph_annotation"));
        assert!(shader.contains("fn lod_debug_requires_authored_color(splat_index: u32) -> bool"));
        assert!(shader.contains("fn lod_debug_morph_requires_authored_color"));
        assert!(shader.contains("case 1u, 2u, 3u: { return false; }"));
        assert!(shader.contains("case 4u: { return true; }"));
        assert!(shader.contains("case 5u: { return lod_debug_uniforms.quality_params.x < 0.0; }"));
        for mode_case in ["case 1u", "case 2u", "case 3u", "case 4u", "case 5u"] {
            assert!(
                shader.contains(mode_case),
                "missing shader debug preset {mode_case}"
            );
        }
        assert!(shader.contains("annotated = authored;"));
        assert!(shader.contains("LOD_DEBUG_BOUNDARY_WIDTH"));
        assert!(shader.contains("LOD_DEBUG_BOUNDARY_COLOR"));
        assert!(shader.contains("fn lod_debug_selection_pressure"));
        assert!(shader.contains("record.node_center[0]"));
        assert!(shader.contains("record.node_radius"));
        assert!(shader.contains("view.clip_from_view[3][2]"));
        assert!(shader.contains("view.clip_from_view[3][3] == 1.0"));
        assert!(shader.contains("fn lod_debug_projected_node"));
        assert!(shader.contains("2.0 * projected.support_radius_px / viewport_height_px"));
        assert!(shader.contains("HIGH_QUALITY_FIDELITY_GUARD_START: f32 = 0.90"));
        assert!(shader.contains("HIGH_QUALITY_FIDELITY_GUARD_FULL: f32 = 0.99"));
        assert!(shader.contains("HIGH_QUALITY_CERTIFICATE_GUARD_START: f32 = 0.90"));
        assert!(shader.contains("HIGH_QUALITY_CERTIFICATE_GUARD_FULL: f32 = 0.95"));
        assert!(shader.contains("PROJECTED_ERROR_AUTHORITY_FULL: f32 = 0.99"));
        assert!(shader.contains("fn lod_debug_high_quality_fidelity_guard"));
        assert!(shader.contains("fn lod_debug_high_quality_certificate_guard"));
        assert!(shader.contains("fn lod_debug_high_quality_certificate_demand"));
        assert!(shader.contains("fn lod_debug_projected_error_authority"));
        assert!(shader.contains("/ PROJECTED_ERROR_AUTHORITY_FULL"));
        assert!(shader.contains("return normalized * normalized * normalized"));
        assert!(shader.contains("let effective_coverage = projected_coverage"));
        assert!(shader.contains("+ (1.0 - projected_coverage) * fidelity_guard"));
        assert!(shader.contains("structural_demand = requested_detail * effective_coverage"));
        assert!(shader.contains("max(record.quality_threshold, 0.0)"));
        assert!(
            shader.contains("let balanced_pressure = min(structural_pressure, error_pressure)")
        );
        assert!(shader.contains("error_authority * error_pressure"));
        assert!(!shader.contains("if fidelity_guard <= 0.0"));
        assert!(shader.contains("let base_demand = detail * normalized;"));
        assert!(shader.contains("lod_debug_high_quality_certificate_guard(detail)"));
        assert!(!shader.contains("let base_demand = detail * normalized * normalized"));
        assert!(shader.contains("if certificate <= 1.0 / LOD_DEBUG_CERTIFICATE_MAX"));
        assert!(
            shader.contains(
                "requested_detail >= HIGH_QUALITY_CERTIFICATE_GUARD_FULL && !is_original"
            )
        );
        assert!(shader.contains("certificate_pressure = lod_debug_pressure_ratio"));
        assert!(shader.contains("lod_debug_high_fidelity_certificate(record)"));
        assert!(shader.contains("return max(guarded_error_pressure, certificate_pressure)"));
        assert!(shader.contains("return select(3.402823e+38, 0.0, is_original);"));
        assert!(shader.contains("fn lod_debug_level_color"));
        assert!(shader.contains("record.geometric_error"));
        assert!(!shader.contains("record.errors"));
        assert!(shader.contains("if lod_debug_uniforms.flags.x == 3u"));
        assert!(shader.contains("return mix(parent, child, blend_t);"));

        let parent_annotation = [0.9_f32, 0.1, 0.3];
        let child_annotation = [0.1_f32, 0.8, 0.6];
        let interpolate_annotation = |blend_t: f32| {
            std::array::from_fn::<_, 3, _>(|channel| {
                if blend_t <= 0.0 {
                    parent_annotation[channel]
                } else if blend_t >= 1.0 {
                    child_annotation[channel]
                } else {
                    parent_annotation[channel]
                        + (child_annotation[channel] - parent_annotation[channel]) * blend_t
                }
            })
        };
        assert_eq!(interpolate_annotation(0.0), parent_annotation);
        assert_eq!(interpolate_annotation(1.0), child_annotation);
        let midpoint = interpolate_annotation(0.5);
        for (actual, expected) in midpoint.into_iter().zip([0.5, 0.45, 0.45]) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }

        let gaussian = include_str!("gaussian.wgsl");
        assert!(gaussian.contains("#ifdef LOD_DEBUG"));
        assert!(gaussian.contains("lod_debug_requires_authored_color,"));
        assert_eq!(
            gaussian
                .matches("var debug_requires_authored_color = lod_debug_requires_authored_color")
                .count(),
            2,
            "classification and authored-color raster paths must both skip SH when debug fully overwrites it"
        );
        assert!(gaussian.contains("lod_debug_morph_requires_authored_color("));
        assert!(gaussian.contains("rgb = apply_lod_debug_morph_annotation("));
        assert!(gaussian.contains("lod_residency_from_entry(entry)"));
        assert!(gaussian.contains("const LOD_ENTRY_SOURCE_INDEX_MASK: u32 = 0x0fffffffu;"));
        assert!(gaussian.contains("const LOD_ENTRY_PRESENTATION_CLASS_SHIFT: u32 = 28u;"));
        assert!(gaussian.contains("let splat_index = source_index_from_entry(entry);"));
    }

    #[test]
    fn sparse_debug_stays_ready_across_candidate_activation_epochs() {
        assert!(lod_debug_sparse_candidate_epoch_ready(
            true, true, false, true, false,
        ));
        assert!(lod_debug_sparse_candidate_epoch_ready(
            false, true, false, true, false,
        ));
        assert!(lod_debug_sparse_candidate_epoch_ready(
            false, true, true, true, true,
        ));
        assert!(lod_debug_sparse_candidate_epoch_ready(
            false, false, true, false, true,
        ));
        assert!(!lod_debug_sparse_candidate_epoch_ready(
            false, true, true, true, false,
        ));
        assert!(!lod_debug_sparse_candidate_epoch_ready(
            false, true, true, false, true,
        ));
    }

    #[test]
    fn incomplete_sparse_debug_uses_authored_color_until_uploads_are_ready() {
        assert!(!lod_debug_shader_enabled(true, true, false, true));
        assert!(!lod_debug_shader_enabled(true, false, true, true));
        assert!(!lod_debug_shader_enabled(true, true, true, false));
        assert!(!lod_debug_shader_enabled(false, true, true, true));
        assert!(lod_debug_shader_enabled(true, true, true, true));
        assert_eq!(LodDebugGpuUploadStats::config_bytes_per_write(), 32);
        assert_eq!(
            LodDebugGpuUploadStats::max_sparse_record_bytes_per_frame(),
            64 * 1024 * 1024
        );
        assert_eq!(
            LodDebugGpuUploadStats::max_sparse_record_slots_per_frame(),
            256
        );
    }

    #[test]
    fn lod_debug_sidecar_is_never_bound_across_candidate_epochs() {
        assert!(lod_debug_candidate_epoch_ready(true, false, false));
        assert!(lod_debug_candidate_epoch_ready(true, true, true));
        assert!(lod_debug_candidate_epoch_ready(false, false, false));
        assert!(!lod_debug_candidate_epoch_ready(false, false, true));
        assert!(!lod_debug_candidate_epoch_ready(false, true, false));
        assert!(!lod_debug_candidate_epoch_ready(false, true, true));
    }

    #[test]
    fn lod_debug_uniform_is_group4_local_and_vec4_aligned() {
        assert_eq!(std::mem::size_of::<LodDebugGpuUniform>(), 32);
        assert_eq!(std::mem::offset_of!(LodDebugGpuUniform, flags), 0);
        assert_eq!(std::mem::offset_of!(LodDebugGpuUniform, quality_params), 16);
        let bindings = include_str!("bindings.wgsl");
        assert!(!bindings.contains("lod_debug_"));
        let debug_shader = include_str!("lod_debug.wgsl");
        for field in ["flags: vec4<u32>", "quality_params: vec4<f32>"] {
            assert!(
                debug_shader.contains(field),
                "missing shader uniform field {field}"
            );
        }
        for removed_field in [
            "params: vec4<f32>",
            "boundary_color: vec4<f32>",
            "resident_color: vec4<f32>",
            "fallback_color: vec4<f32>",
            "unavailable_color: vec4<f32>",
            "unknown_color: vec4<f32>",
        ] {
            assert!(
                !debug_shader
                    .lines()
                    .any(|line| line.trim() == removed_field)
            );
        }
        assert!(debug_shader.contains("@group(4) @binding(1) var<uniform>"));
    }

    #[test]
    fn lod_debug_uniform_carries_the_authoritative_quality_contract() {
        use crate::gaussian::{lod_debug::LodDebugSettings, lod_settings::GaussianLodSettings};

        let settings = LodDebugSettings::default();
        let missing = LodDebugGpuUniform::new(&settings, None, 17);
        assert_eq!(missing.flags[1], 17);
        assert_eq!(missing.quality_params, [-1.0, 0.0, 0.0, 0.0]);

        let lod = GaussianLodSettings {
            quality: 0.5,
            ..Default::default()
        };
        let continuous = LodDebugGpuUniform::new(&settings, Some(&lod), 17);
        assert_eq!(
            continuous.quality_params,
            [lod.screen_space_error_limit_px(), 0.5, 0.0, 0.0]
        );

        let coarsest = GaussianLodSettings {
            quality: 0.0,
            ..Default::default()
        };
        assert_eq!(
            LodDebugGpuUniform::new(&settings, Some(&coarsest), 0).quality_params,
            [f32::MAX, 0.0, 0.0, 0.0]
        );
        let original = GaussianLodSettings::default();
        assert_eq!(
            LodDebugGpuUniform::new(&settings, Some(&original), 0).quality_params,
            [0.0, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn lod_debug_quality_authorities_and_certificate_match_the_cpu_target() {
        use crate::gaussian::lod_settings::{
            LodQualityTarget, high_fidelity_certificate_pressure, high_quality_certificate_demand,
            high_quality_certificate_guard, high_quality_fidelity_guard, projected_error_authority,
        };

        const HIGH_QUALITY_FIDELITY_GUARD_START: f32 = 0.90;
        const HIGH_QUALITY_FIDELITY_GUARD_FULL: f32 = 0.99;
        const HIGH_QUALITY_CERTIFICATE_GUARD_START: f32 = 0.90;
        const HIGH_QUALITY_CERTIFICATE_GUARD_FULL: f32 = 0.95;
        const PROJECTED_ERROR_AUTHORITY_FULL: f32 = 0.99;
        const MIN_QUANTIZED_HIGH_FIDELITY_CERTIFICATE: f32 = 1.0 / u16::MAX as f32;

        fn shader_guard(detail: f32, start: f32, full: f32) -> f32 {
            if detail <= start {
                0.0
            } else if detail >= full {
                1.0
            } else {
                let t = (detail - start) / (full - start);
                t * t * (3.0 - 2.0 * t)
            }
        }

        fn shader_cubic_authority(detail: f32, full: f32) -> f32 {
            let normalized = (detail.clamp(0.0, 1.0) / full).clamp(0.0, 1.0);
            normalized * normalized * normalized
        }

        fn shader_error_authority(detail: f32) -> f32 {
            shader_cubic_authority(detail, PROJECTED_ERROR_AUTHORITY_FULL)
        }

        fn shader_certificate_demand(detail: f32, coverage: f32) -> f32 {
            let detail = detail.clamp(0.0, 1.0);
            let normalized = (detail / HIGH_QUALITY_CERTIFICATE_GUARD_FULL).clamp(0.0, 1.0);
            let base = detail * normalized;
            let coverage_authority =
                shader_cubic_authority(detail, HIGH_QUALITY_CERTIFICATE_GUARD_FULL);
            let gate = shader_guard(
                detail,
                HIGH_QUALITY_CERTIFICATE_GUARD_START,
                HIGH_QUALITY_CERTIFICATE_GUARD_FULL,
            );
            gate * base
                * (coverage.clamp(0.0, 1.0) + (1.0 - coverage.clamp(0.0, 1.0)) * coverage_authority)
        }

        fn shader_ratio(numerator: f32, denominator: f32) -> f32 {
            if denominator <= 0.0 {
                if numerator <= 0.0 { 0.0 } else { f32::MAX }
            } else {
                (numerator / denominator).min(f32::MAX)
            }
        }

        fn shader_certificate_pressure(
            detail: f32,
            coverage: f32,
            certificate: f32,
            is_original: bool,
        ) -> f32 {
            if certificate <= MIN_QUANTIZED_HIGH_FIDELITY_CERTIFICATE {
                if detail >= HIGH_QUALITY_CERTIFICATE_GUARD_FULL && !is_original {
                    f32::MAX
                } else {
                    0.0
                }
            } else {
                shader_ratio(shader_certificate_demand(detail, coverage), certificate)
            }
        }

        for detail in [0.0, 0.25, 0.5, 0.90, 0.945, 0.95, 0.99] {
            let target = LodQualityTarget::Balanced {
                detail_fraction: detail,
                max_error_px: 1.0,
            };
            let structural_guard = shader_guard(
                detail,
                HIGH_QUALITY_FIDELITY_GUARD_START,
                HIGH_QUALITY_FIDELITY_GUARD_FULL,
            );
            let error_authority = shader_error_authority(detail);
            let certificate_guard = shader_guard(
                detail,
                HIGH_QUALITY_CERTIFICATE_GUARD_START,
                HIGH_QUALITY_CERTIFICATE_GUARD_FULL,
            );
            assert!((structural_guard - high_quality_fidelity_guard(detail)).abs() < 1e-7);
            assert!((error_authority - projected_error_authority(detail)).abs() < 1e-7);
            assert!((certificate_guard - high_quality_certificate_guard(detail)).abs() < 1e-7);
            for coverage in [0.0, 0.25, 0.75, 1.0] {
                let certificate_demand = shader_certificate_demand(detail, coverage);
                assert!(
                    (certificate_demand - high_quality_certificate_demand(detail, coverage)).abs()
                        < 1e-7
                );
                let effective_coverage = coverage + (1.0 - coverage) * structural_guard;
                let shader_demand = detail * effective_coverage;
                assert!(
                    (shader_demand - target.structural_detail_demand(coverage)).abs() < 1e-6,
                    "detail={detail} coverage={coverage}"
                );
                for threshold in [0.6, 1.0] {
                    for error_pressure in [0.25, 2.0, 128.0] {
                        for certificate in [
                            0.0,
                            MIN_QUANTIZED_HIGH_FIDELITY_CERTIFICATE,
                            2.0 * MIN_QUANTIZED_HIGH_FIDELITY_CERTIFICATE,
                            certificate_demand.max(2.0 * MIN_QUANTIZED_HIGH_FIDELITY_CERTIFICATE),
                            1.0,
                        ] {
                            let shader_certificate =
                                shader_certificate_pressure(detail, coverage, certificate, false);
                            let cpu_certificate = high_fidelity_certificate_pressure(
                                detail,
                                coverage,
                                certificate,
                                false,
                            );
                            assert!(
                                shader_certificate == cpu_certificate
                                    || (shader_certificate - cpu_certificate).abs() < 1e-6
                            );
                            let structural_pressure = shader_demand / threshold;
                            let balanced_pressure = structural_pressure.min(error_pressure);
                            let shader_pressure = balanced_pressure
                                .max(error_authority * error_pressure)
                                .max(shader_certificate);
                            let cpu_pressure = target.node_pressure(
                                threshold,
                                error_pressure,
                                coverage,
                                certificate,
                                false,
                            );
                            assert!(
                                shader_pressure == cpu_pressure
                                    || (shader_pressure - cpu_pressure).abs() < 1e-5,
                                "detail={detail} coverage={coverage} threshold={threshold} error={error_pressure} certificate={certificate}"
                            );
                        }
                    }
                }
            }
        }

        assert_eq!(
            shader_certificate_pressure(0.95, 0.0, 0.0, true),
            high_fidelity_certificate_pressure(0.95, 0.0, 0.0, true)
        );
    }

    #[cfg(feature = "precompute_covariance_3d")]
    #[test]
    fn precomputed_covariance_is_an_additive_canonical_storage_plane() {
        let bindings = include_str!("bindings.wgsl");
        let planar = include_str!("planar.wgsl");
        let gaussian_3d = include_str!("gaussian_3d.wgsl");

        assert!(bindings.contains("@group(2) @binding(2) var<storage, read> rotation"));
        assert!(bindings.contains("@group(2) @binding(3) var<storage, read> scale_opacity"));
        assert!(
            bindings.contains("@group(2) @binding(4) var<storage, read> covariance_3d_opacity")
        );
        assert!(planar.contains("fn get_cov3d(index: u32)"));
        assert!(planar.contains("fn get_rotation(index: u32)"));
        assert!(gaussian_3d.contains("let child_local_cov3d = get_cov3d(index);"));
        assert!(gaussian_3d.contains("transform_local_cov3d(local_cov3d)"));
        assert!(gaussian_3d.contains("fn covariance_storage_scale_squared() -> f32"));
        assert!(
            gaussian_3d.contains("gaussian_uniforms.global_scale * gaussian_uniforms.global_scale")
        );
        assert_eq!(gaussian_3d.matches("* storage_scale_squared").count(), 3);
        let storage_scale_helper = gaussian_3d
            .split("fn covariance_storage_scale_squared() -> f32")
            .nth(1)
            .and_then(|source| source.split("fn transform_local_cov3d").next())
            .expect("representation-specific covariance storage scale helper");
        assert!(storage_scale_helper.contains("#ifdef PRECOMPUTE_COVARIANCE_3D"));
        assert!(storage_scale_helper.contains("return 1.0;"));

        let gaussian = include_str!("gaussian.wgsl");
        let precomputed_storage_import = gaussian
            .split("#else ifdef BUFFER_STORAGE")
            .nth(1)
            .and_then(|storage| storage.split("#else ifdef BUFFER_TEXTURE").next())
            .expect("precomputed planar storage import block");
        assert!(precomputed_storage_import.contains("get_cov3d,"));
        assert!(precomputed_storage_import.contains("get_rotation,"));

        let defs = shader_defs(CloudPipelineKey {
            gaussian_mode: crate::gaussian::settings::GaussianMode::Gaussian3d,
            rasterize_mode: crate::gaussian::settings::RasterizeMode::Normal,
            ..Default::default()
        });
        for required in [
            "PRECOMPUTE_COVARIANCE_3D",
            "GAUSSIAN_3D",
            "RASTERIZE_NORMAL",
        ] {
            assert!(
                defs.iter()
                    .any(|define| format!("{define:?}").contains(required)),
                "missing specialization definition {required}"
            );
        }
    }

    #[test]
    fn standard_and_precomputed_covariance_apply_dynamic_global_scale_once() {
        use bevy::math::{EulerRot, Mat3, Quat, Vec3};

        let local_scale = Vec3::new(0.35, 1.25, 2.75);
        let rotation = Mat3::from_quat(Quat::from_euler(EulerRot::XYZ, 0.37, -0.81, 1.13));
        let cloud_transform = Mat3::from_cols(
            Vec3::new(1.2, 0.15, -0.05),
            Vec3::new(-0.2, 0.8, 0.1),
            Vec3::new(0.3, -0.25, 1.6),
        );
        let raw_m = Mat3::from_diagonal(local_scale) * rotation;
        let raw_covariance = raw_m.transpose() * raw_m;

        for global_scale in [-2.0_f32, -0.5, 0.0, 0.5, 2.0] {
            let runtime_m = Mat3::from_diagonal(local_scale * global_scale) * rotation;
            let runtime_covariance = runtime_m.transpose() * runtime_m;
            // The standard shader branch obtains this covariance from
            // `get_scale_matrix`, which has already applied global_scale. Its
            // storage multiplier must therefore be exactly one.
            let standard_world = cloud_transform * runtime_covariance * cloud_transform.transpose();

            let scale_squared = global_scale * global_scale;
            let precomputed_local = Mat3::from_cols(
                raw_covariance.x_axis * scale_squared,
                raw_covariance.y_axis * scale_squared,
                raw_covariance.z_axis * scale_squared,
            );
            let precomputed_world =
                cloud_transform * precomputed_local * cloud_transform.transpose();

            for (standard, precomputed) in standard_world
                .to_cols_array()
                .into_iter()
                .zip(precomputed_world.to_cols_array())
            {
                let tolerance = 2.0e-5 * standard.abs().max(1.0);
                assert!(
                    (standard - precomputed).abs() <= tolerance,
                    "global_scale={global_scale} standard={standard} precomputed={precomputed}"
                );
            }

            // Applying the storage-plane multiplier to the standard branch as
            // well would produce g^4 covariance. Non-unit values must make that
            // failure observably different from the expected standard result.
            if scale_squared != 0.0 && scale_squared != 1.0 {
                let double_scaled = cloud_transform
                    * Mat3::from_cols(
                        runtime_covariance.x_axis * scale_squared,
                        runtime_covariance.y_axis * scale_squared,
                        runtime_covariance.z_axis * scale_squared,
                    )
                    * cloud_transform.transpose();
                assert!(
                    standard_world
                        .to_cols_array()
                        .into_iter()
                        .zip(double_scaled.to_cols_array())
                        .any(|(standard, wrong)| (standard - wrong).abs() > 1.0e-4)
                );
            }
        }
    }

    #[test]
    fn gaussian_and_rasterization_modes_fail_closed_before_specialization() {
        use crate::gaussian::settings::{GaussianMode, RasterizeMode};

        for gaussian_mode in [
            GaussianMode::Gaussian2d,
            GaussianMode::Gaussian3d,
            GaussianMode::Gaussian4d,
        ] {
            for rasterize_mode in [
                RasterizeMode::Classification,
                RasterizeMode::Color,
                RasterizeMode::Depth,
                RasterizeMode::Normal,
                RasterizeMode::OpticalFlow,
                RasterizeMode::Position,
                RasterizeMode::Velocity,
            ] {
                let expected = match rasterize_mode {
                    RasterizeMode::Normal => gaussian_mode != GaussianMode::Gaussian4d,
                    RasterizeMode::Velocity => gaussian_mode == GaussianMode::Gaussian4d,
                    _ => true,
                };
                assert_eq!(
                    gaussian_rasterization_is_supported(gaussian_mode, rasterize_mode),
                    expected,
                    "gaussian={gaussian_mode:?} raster={rasterize_mode:?}"
                );
            }
        }

        let lod_3d = shader_defs(CloudPipelineKey {
            gaussian_mode: GaussianMode::Gaussian3d,
            lod_candidate: true,
            ..Default::default()
        });
        let lod_4d = shader_defs(CloudPipelineKey {
            gaussian_mode: GaussianMode::Gaussian4d,
            lod_candidate: true,
            ..Default::default()
        });
        let lod_3d_has_morph = lod_3d
            .iter()
            .any(|define| format!("{define:?}").contains("LOD_MORPH"));
        let lod_4d_has_morph = lod_4d
            .iter()
            .any(|define| format!("{define:?}").contains("LOD_MORPH"));
        #[cfg(lod_render_path)]
        {
            assert!(lod_3d_has_morph);
            assert!(!lod_4d_has_morph);
        }
        #[cfg(not(lod_render_path))]
        {
            assert!(!lod_3d_has_morph);
            assert!(!lod_4d_has_morph);
        }
    }

    #[cfg(feature = "precompute_covariance_3d")]
    #[test]
    fn interpolation_uses_the_five_plane_precomputed_output_layout() {
        let descriptor = gaussian_storage_layout_descriptor::<Gaussian3d>(
            "gaussian_interpolate_output_layout_contract",
            false,
        );
        assert_eq!(descriptor.entries.len(), 5);
        assert_eq!(descriptor.entries[4].binding, 4);

        let interpolate = include_str!("../morph/interpolate.rs");
        assert!(interpolate.contains("gaussian_storage_layout_descriptor::<R>("));
        assert!(
            !interpolate
                .contains("let output_layout_desc = storage_layout_descriptor::<<R::GpuPlanarType")
        );
    }
}
