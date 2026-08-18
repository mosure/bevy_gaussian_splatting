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
        renderer::RenderDevice,
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
        interface::CommonCloud,
        lod_debug::{LodDebugMetadata, LodDebugRecord, LodDebugSettings},
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
const PACKED_SHADER_HANDLE: Handle<Shader> = uuid_handle!("5bb62086-7004-4575-9972-274dc8acccf1");
const PLANAR_SHADER_HANDLE: Handle<Shader> = uuid_handle!("d6a3f978-f795-4786-8475-26366f28d852");
const TEXTURE_SHADER_HANDLE: Handle<Shader> = uuid_handle!("500e2ebf-51a8-402e-9c88-e0d5152c3486");
const TRANSFORM_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("648516b2-87cc-4937-ae1c-d986952e9fa7");

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
                        prepare_lod_debug_bind_group::<R>.in_set(RenderSystems::PrepareBindGroups),
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
    record_count: usize,
    settings: LodDebugSettings,
    lod_settings: Option<GaussianLodSettings>,
    marker: std::marker::PhantomData<fn() -> R>,
}

#[allow(type_alias_bounds)]
type LodDebugPrepareQuery<R: PlanarSync> = (
    Entity,
    &'static R::PlanarTypeHandle,
    &'static CloudSettings,
    Option<&'static GaussianLodSettings>,
    Option<&'static LodDebugMetadata>,
    Option<&'static LodDebugBindGroup<R>>,
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
    pipeline: Res<CloudPipeline<R>>,
    gpu_clouds: Res<RenderAssets<R::GpuPlanarType>>,
    clouds: Query<LodDebugPrepareQuery<R>>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    let pipeline_changed = pipeline.is_changed();
    let Some(layout) = pipeline.lod_debug_layout.as_ref() else {
        for (entity, _, _, _, _, existing) in &clouds {
            if existing.is_some() {
                commands.entity(entity).remove::<LodDebugBindGroup<R>>();
            }
        }
        return;
    };

    let fallback = [LodDebugRecord::default()];
    for (entity, handle, settings, lod_settings, metadata, existing) in &clouds {
        if !settings.lod_debug.requires_metadata() {
            if existing.is_some() {
                commands.entity(entity).remove::<LodDebugBindGroup<R>>();
            }
            continue;
        }

        let Some(gpu_cloud) = gpu_clouds.get(handle.handle()) else {
            continue;
        };
        let records = metadata.map(LodDebugMetadata::records).unwrap_or_default();
        let record_count = records.len().min(gpu_cloud.len());
        let source_pointer = if record_count == 0 {
            0
        } else {
            records.as_ptr() as usize
        };
        if !pipeline_changed
            && existing.is_some_and(|existing| {
                existing.source_pointer == source_pointer
                    && existing.record_count == record_count
                    && existing.settings == settings.lod_debug
                    && existing.lod_settings.as_ref() == lod_settings
            })
        {
            continue;
        }

        let upload_records = if record_count == 0 {
            &fallback[..]
        } else {
            &records[..record_count]
        };
        let contents = bytemuck::cast_slice(upload_records);
        let byte_len = contents.len() as u64;
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

        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("lod_debug_records"),
            contents,
            usage: BufferUsages::STORAGE,
        });
        let config = LodDebugGpuUniform::new(&settings.lod_debug, lod_settings, record_count);
        let config_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("lod_debug_config"),
            contents: bytemuck::bytes_of(&config),
            usage: BufferUsages::UNIFORM,
        });
        let bind_group = render_device.create_bind_group(
            "lod_debug_bind_group",
            layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        );
        commands.entity(entity).insert(LodDebugBindGroup::<R> {
            bind_group,
            _buffer: buffer,
            _config_buffer: config_buffer,
            _source_metadata: metadata.cloned(),
            source_pointer,
            record_count,
            settings: settings.lod_debug,
            lod_settings: lod_settings.cloned(),
            marker: std::marker::PhantomData,
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_gaussians<R: PlanarSync>(
    gaussian_cloud_uniform: Res<ComponentUniforms<CloudUniform>>,
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CloudPipeline<R>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<CloudPipeline<R>>>,
    pipeline_cache: Res<PipelineCache>,
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

            let key = CloudPipelineKey {
                aabb: settings.aabb,
                binary_gaussian_op: false,
                opacity_adaptive_radius: settings.opacity_adaptive_radius,
                visualize_bounding_box: settings.visualize_bounding_box,
                draw_mode: settings.draw_mode,
                gaussian_mode: settings.gaussian_mode,
                rasterize_mode: settings.rasterize_mode,
                lod_debug: settings.lod_debug.requires_metadata()
                    && lod_debug_bind_group.is_some()
                    && custom_pipeline.lod_debug_layout_desc.is_some(),
                sample_count: msaa.samples(),
                hdr: view.target_format == TextureFormat::Rgba16Float,
            };

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

    pub temporal_sort_window_size: u32,
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

            temporal_sort_window_size: 16,
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
        ShaderDefVal::UInt(
            "TEMPORAL_SORT_WINDOW_SIZE".into(),
            defines.temporal_sort_window_size,
        ),
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
    pub sample_count: u32,
    pub hdr: bool,
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

        let mut layout = vec![
            self.view_layout_desc.clone(),
            self.gaussian_uniform_layout_desc.clone(),
            self.gaussian_cloud_layout_desc.clone(),
            self.sorted_layout_desc.clone(),
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
type DrawGaussianParam<R: PlanarSync> = (
    SRes<RenderAssets<R::GpuPlanarType>>,
    SRes<lod::LodCompactionBuffers<R>>,
);

#[allow(type_alias_bounds)]
#[cfg(not(lod_render_path))]
type DrawGaussianParam<R: PlanarSync> = SRes<RenderAssets<R::GpuPlanarType>>;

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
    type ItemQuery = (
        Read<R::PlanarTypeHandle>,
        Read<PlanarStorageBindGroup<R>>,
        Read<SortBindGroup>,
        Read<CloudSettings>,
        Option<Read<LodDebugBindGroup<R>>>,
    );

    #[inline]
    fn render<'w>(
        item: &P,
        (view, _extracted_view): ROQueryItem<'w, 'w, Self::ViewQuery>,
        entity: Option<(
            &'w R::PlanarTypeHandle,
            &'w PlanarStorageBindGroup<R>,
            &'w SortBindGroup,
            &'w CloudSettings,
            Option<&'w LodDebugBindGroup<R>>,
        )>,
        gaussian_params: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        debug!("render call");

        #[cfg(not(lod_render_path))]
        let _ = item;

        #[cfg(all(feature = "buffer_texture", not(feature = "buffer_storage")))]
        let _ = view;

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
            && let Some(lod_debug_bind_group) = lod_debug_bind_group
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
        CloudPipelineKey, LodDebugGpuUniform, planar_storage_binding_needs_refresh, shader_defs,
    };

    #[cfg(feature = "precompute_covariance_3d")]
    use super::gaussian_storage_layout_descriptor;
    #[cfg(feature = "precompute_covariance_3d")]
    use crate::gaussian::formats::planar_3d::Gaussian3d;

    #[test]
    fn planar_storage_rebinds_handle_swap_without_an_asset_event() {
        assert!(planar_storage_binding_needs_refresh(true, false, false));
        assert!(!planar_storage_binding_needs_refresh(true, true, false));
        assert!(planar_storage_binding_needs_refresh(false, true, false));
        assert!(planar_storage_binding_needs_refresh(true, true, true));
    }

    #[test]
    fn raster_frustum_culling_uses_gaussian_support_not_only_its_center() {
        let shader = include_str!("gaussian.wgsl");
        assert!(shader.contains("fn gaussian_support_radius_world"));
        assert!(shader.contains("fn gaussian_support_sphere_in_frustum"));
        assert!(shader.contains("gaussian_uniforms.transform_scale_bound"));
        assert!(!shader.contains("let gram_xx"));
        assert!(!shader.contains("length(plane.xyz)"));
        assert!(shader.contains("let projected_position = world_to_clip(transformed_position);"));
        assert!(shader.contains("discard_quad |= !gaussian_support_sphere_in_frustum"));
        assert!(!shader.contains("discard_quad |= !in_frustum(projected_position.xyz);"));
    }

    #[test]
    fn lod_debug_shader_is_opt_in_and_bounds_checks_metadata() {
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
        assert!(shader.contains("lod_debug_residency(record)"));
        assert!(shader.contains("let distance = bitcast<f32>(distance_bits);"));
        assert!(shader.contains("fn apply_lod_debug_annotation"));
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
        assert!(shader.contains("HIGH_QUALITY_CERTIFICATE_GUARD_FULL: f32 = 0.95"));
        assert!(shader.contains("PROJECTED_ERROR_AUTHORITY_FULL: f32 = 0.99"));
        assert!(shader.contains("fn lod_debug_high_quality_fidelity_guard"));
        assert!(shader.contains("fn lod_debug_high_quality_certificate_guard"));
        assert!(shader.contains("fn lod_debug_high_quality_certificate_demand"));
        assert!(shader.contains("fn lod_debug_projected_error_authority"));
        assert!(shader.contains("/ PROJECTED_ERROR_AUTHORITY_FULL"));
        assert!(shader.contains("normalized * normalized * normalized"));
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

        let gaussian = include_str!("gaussian.wgsl");
        assert!(gaussian.contains("#ifdef LOD_DEBUG"));
        assert!(gaussian.contains("rgb = apply_lod_debug_annotation(splat_index, rgb)"));
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
            let authority = shader_cubic_authority(detail, HIGH_QUALITY_CERTIFICATE_GUARD_FULL);
            base * (coverage.clamp(0.0, 1.0) + (1.0 - coverage.clamp(0.0, 1.0)) * authority)
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
            let certificate_guard =
                shader_cubic_authority(detail, HIGH_QUALITY_CERTIFICATE_GUARD_FULL);
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
        assert!(gaussian_3d.contains("transform_precomputed_cov3d(get_cov3d(index))"));
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
