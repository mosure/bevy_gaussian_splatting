use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
};

use bevy::{
    asset::{UntypedAssetId, load_internal_asset, uuid_handle},
    core_pipeline::{Core3d, Core3dSystems, prepass::PreviousViewUniformOffset},
    prelude::*,
    render::{
        GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::DynamicUniformIndex,
        init_gpu_resource,
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding,
            BufferBindingType, BufferDescriptor, BufferId, BufferInitDescriptor, BufferSize,
            BufferUsages, CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, ShaderStages,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        view::{ExtractedView, RenderVisibleEntities, ViewUniformOffset},
    },
};
use bevy_interleave::{interface::storage::PlanarStorageBindGroup, prelude::*};
use static_assertions::assert_cfg;

use bevy::render::view::RetainedViewEntity;

#[cfg(feature = "morph_interpolate")]
use crate::{
    gaussian::formats::planar_3d::PlanarGaussian3d,
    morph::interpolate::{GaussianInterpolate, InterpolateLabel},
};

#[cfg(feature = "morph_particles")]
use crate::morph::particle::{MorphLabel, ParticleBehaviorsHandle};

use crate::{
    CloudSettings, GaussianCamera, RadixSortDepthBits,
    gaussian::cloud::CloudVisibilityClass,
    render::{
        CloudPipeline, CloudPipelineKey, CloudPipelineReady, CloudUniform,
        GaussianUniformBindGroups, ShaderDefines, shader_defs_with_defines,
    },
    sort::{
        GpuSortedEntry, SortEntry, SortMode, SortPluginFlag, SortedEntriesHandle,
        sort_entry_binding_size,
    },
};

#[cfg(lod_render_path)]
use crate::render::lod::{
    DISPATCH_A_INDIRECT_OFFSET, DISPATCH_C_INDIRECT_OFFSET, LodCompactionBuffers,
    LodCompactionLabel, lod_view_cloud_key,
};

#[cfg(lod_render_path)]
use crate::stream::{atlas_upload::LodAtlasGpuGenerations, render_commit::LodRenderCandidates};

assert_cfg!(
    not(all(feature = "sort_radix", feature = "buffer_texture",)),
    "sort_radix and buffer_texture are incompatible",
);

const RADIX_SHADER_HANDLE: Handle<Shader> = uuid_handle!("dedb3ddf-f254-4361-8762-e221774de1ed");
const RADIX_PIPELINE_RESET: usize = 0;
const RADIX_PIPELINE_A: usize = 1;
const RADIX_PIPELINE_B: usize = 2;
const RADIX_PIPELINE_C_COUNT: usize = 3;
const RADIX_PIPELINE_C_SCAN: usize = 4;
const RADIX_PIPELINE_C_SCATTER: usize = 5;
#[cfg(lod_render_path)]
const RADIX_PIPELINE_ACTIVE_A: usize = 6;
#[cfg(lod_render_path)]
const RADIX_PIPELINE_COUNT: usize = 7;
#[cfg(not(lod_render_path))]
const RADIX_PIPELINE_COUNT: usize = 6;
const RADIX_DEPTH_BITS_VARIANT_COUNT: usize = 3;

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub struct RadixSortLabel;

#[derive(Default)]
pub struct RadixSortPlugin<R: PlanarSync> {
    phantom: std::marker::PhantomData<R>,
}

impl<R: PlanarSync> Plugin for RadixSortPlugin<R>
where
    R::GpuPlanarType: GpuPlanarStorage,
{
    fn build(&self, app: &mut App) {
        // TODO: run once
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Render,
                (
                    queue_radix_bind_group::<R>.in_set(RenderSystems::Queue),
                    #[cfg(lod_render_path)]
                    queue_lod_radix_bind_groups::<R>.in_set(RenderSystems::Queue),
                ),
            );

            render_app.init_gpu_resource::<RadixSortBuffers<R>>();
            render_app.init_resource::<RadixSortWorkCache<R>>();
            render_app.init_resource::<PendingRadixBufferChanges<R>>();
            render_app.add_systems(
                ExtractSchedule,
                (
                    invalidate_stale_radix_bind_groups::<R>,
                    flush_radix_buffer_changes::<R>,
                )
                    .chain(),
            );

            #[cfg(lod_render_path)]
            render_app
                .init_gpu_resource::<LodRadixBindGroups<R>>()
                .configure_sets(Core3d, RadixSortLabel.after(LodCompactionLabel));
            #[cfg(feature = "morph_particles")]
            render_app.configure_sets(Core3d, RadixSortLabel.after(MorphLabel));
            #[cfg(feature = "morph_interpolate")]
            if TypeId::of::<R::PlanarType>() == TypeId::of::<PlanarGaussian3d>() {
                render_app.add_systems(
                    Core3d,
                    run_radix_sort::<R>
                        .in_set(RadixSortLabel)
                        .after(InterpolateLabel)
                        .before(Core3dSystems::Prepass),
                );
            } else {
                render_app.add_systems(
                    Core3d,
                    run_radix_sort::<R>
                        .in_set(RadixSortLabel)
                        .before(Core3dSystems::Prepass),
                );
            }

            #[cfg(not(feature = "morph_interpolate"))]
            render_app.add_systems(
                Core3d,
                run_radix_sort::<R>
                    .in_set(RadixSortLabel)
                    .before(Core3dSystems::Prepass),
            );
        }

        if app.is_plugin_added::<SortPluginFlag>() {
            debug!("sort plugin already added");
            return;
        }

        load_internal_asset!(app, RADIX_SHADER_HANDLE, "radix.wgsl", Shader::from_wgsl);
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                RenderStartup,
                init_gpu_resource::<RadixSortPipeline<R>>
                    .after(CloudPipelineReady)
                    .ambiguous_with_all(),
            );
        }
    }
}

#[derive(Resource)]
struct RadixSortWorkCache<R: PlanarSync> {
    signatures: HashMap<(RetainedViewEntity, Entity, AssetId<R::PlanarType>), u64>,
    marker: PhantomData<fn() -> R>,
}

impl<R: PlanarSync> Default for RadixSortWorkCache<R> {
    fn default() -> Self {
        Self {
            signatures: HashMap::new(),
            marker: PhantomData,
        }
    }
}

fn hash_legacy_camera_sort_inputs(view: &ExtractedView, hasher: &mut impl Hasher) {
    // The vanilla key is squared world-space distance from the camera. Camera
    // rotation, projection, and viewport cannot change that global order. This
    // cache is not a per-pixel depth correction such as StopThePop.
    for value in view.world_from_view.translation().to_array() {
        value.to_bits().hash(hasher);
    }
}

#[allow(clippy::too_many_arguments)]
fn legacy_sort_signature(
    view: &ExtractedView,
    transform: &GlobalTransform,
    settings: &CloudSettings,
    buffer_generation: u64,
    sorted_entry_buffer_id: impl Hash,
    cloud_len: usize,
    storage_changed_frame: Option<u32>,
    atlas_content_revision: u64,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_legacy_camera_sort_inputs(view, &mut hasher);
    for value in transform.to_matrix().to_cols_array() {
        value.to_bits().hash(&mut hasher);
    }
    // Morph interpolation writes positions/visibility before this pass from
    // these settings, without replacing the output storage bind group.
    settings.time.to_bits().hash(&mut hasher);
    settings.time_start.to_bits().hash(&mut hasher);
    settings.time_stop.to_bits().hash(&mut hasher);
    settings.radix_sort_depth_bits.hash(&mut hasher);
    buffer_generation.hash(&mut hasher);
    sorted_entry_buffer_id.hash(&mut hasher);
    cloud_len.hash(&mut hasher);
    storage_changed_frame.hash(&mut hasher);
    atlas_content_revision.hash(&mut hasher);
    hasher.finish()
}

const fn legacy_sort_cache_allowed(has_interpolate: bool, has_particles: bool) -> bool {
    !has_interpolate && !has_particles
}

#[cfg(lod_render_path)]
const fn skip_legacy_sort_for_required_candidate(candidate_draw_required: bool) -> bool {
    candidate_draw_required
}

#[derive(Resource)]
pub struct RadixSortBuffers<R: PlanarSync> {
    // TODO: use a more ECS-friendly approach
    pub asset_map: HashMap<AssetId<R::PlanarType>, GpuRadixBuffers>,
    next_generation: u64,
}

impl<R: PlanarSync> Default for RadixSortBuffers<R> {
    fn default() -> Self {
        RadixSortBuffers {
            asset_map: HashMap::new(),
            next_generation: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuRadixBuffers {
    pub sorting_global_buffer: Buffer,
    pub sorting_status_counter_buffer: Buffer,
    pub sorting_pass_buffers: [Buffer; 4],
    pub entry_buffer_b: Buffer,
    pub capacity: usize,
    pub generation: u64,
}

impl GpuRadixBuffers {
    pub fn new(count: usize, generation: u64, render_device: &RenderDevice) -> Self {
        let allocation_count = count.max(1);
        let sorting_global_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("sorting global buffer"),
            size: ShaderDefines::default().sorting_buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sorting_status_counter_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("status counters buffer"),
            size: ShaderDefines::default().sorting_status_counters_buffer_size(allocation_count)
                as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sorting_pass_buffers = (0..4)
            .map(|idx| {
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: format!("sorting pass buffer {idx}").as_str().into(),
                    contents: &[idx as u8, 0, 0, 0],
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                })
            })
            .collect::<Vec<Buffer>>()
            .try_into()
            .unwrap();

        let entry_buffer_b = render_device.create_buffer(&BufferDescriptor {
            label: Some("entry buffer b"),
            size: (allocation_count * std::mem::size_of::<SortEntry>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        GpuRadixBuffers {
            sorting_global_buffer,
            sorting_status_counter_buffer,
            sorting_pass_buffers,
            entry_buffer_b,
            capacity: count,
            generation,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RadixAllocationDecision {
    Remove,
    Reuse,
    Replace,
}

fn radix_allocation_decision(
    current_capacity: Option<usize>,
    live_capacity: Option<usize>,
) -> RadixAllocationDecision {
    match (current_capacity, live_capacity) {
        (_, None) => RadixAllocationDecision::Remove,
        (Some(current), Some(live)) if current == live => RadixAllocationDecision::Reuse,
        (_, Some(_)) => RadixAllocationDecision::Replace,
    }
}

#[derive(Resource)]
struct PendingRadixBufferChanges<R: PlanarSync> {
    invalidated_assets: HashSet<UntypedAssetId>,
    marker: PhantomData<fn() -> R>,
}

impl<R: PlanarSync> Default for PendingRadixBufferChanges<R> {
    fn default() -> Self {
        Self {
            invalidated_assets: HashSet::new(),
            marker: PhantomData,
        }
    }
}

/// Phase one of a resize/removal transaction. Commands are deliberately
/// deferred: chaining this system before `flush_radix_buffer_changes` inserts
/// an ApplyDeferred edge, dropping every dependent bind group before any old
/// backing buffer is released or any replacement is allocated.
fn invalidate_stale_radix_bind_groups<R: PlanarSync>(
    mut commands: Commands,
    gpu_gaussian_clouds: Res<RenderAssets<R::GpuPlanarType>>,
    sort_buffers: Res<RadixSortBuffers<R>>,
    mut pending: ResMut<PendingRadixBufferChanges<R>>,
    mut work_cache: ResMut<RadixSortWorkCache<R>>,
    clouds: Query<(Entity, &RadixBindGroup, Option<&R::PlanarTypeHandle>)>,
) {
    let live_assets = gpu_gaussian_clouds
        .iter()
        .map(|(asset_id, cloud)| (asset_id, cloud.len()))
        .collect::<HashMap<_, _>>();
    pending.invalidated_assets.clear();
    for (asset_id, buffers) in &sort_buffers.asset_map {
        if radix_allocation_decision(Some(buffers.capacity), live_assets.get(asset_id).copied())
            != RadixAllocationDecision::Reuse
        {
            pending.invalidated_assets.insert(asset_id.untyped());
        }
    }

    for (entity, bind_group, handle) in &clouds {
        if bind_group.cloud_asset.type_id() != TypeId::of::<R::PlanarType>() {
            continue;
        }
        let current_asset = handle.map(|handle| handle.handle().id().untyped());
        if Some(bind_group.cloud_asset) != current_asset
            || pending.invalidated_assets.contains(&bind_group.cloud_asset)
        {
            commands.entity(entity).remove::<RadixBindGroup>();
        }
    }

    work_cache.signatures.retain(|(_, _, asset_id), _| {
        live_assets.contains_key(asset_id)
            && !pending.invalidated_assets.contains(&asset_id.untyped())
    });
}

/// Phase two runs only after the chain's ApplyDeferred sync point. It first
/// drops every invalidated old buffer set, then allocates all replacements, so
/// a cross-asset resize cannot transiently retain both generations.
fn flush_radix_buffer_changes<R: PlanarSync>(
    gpu_gaussian_clouds: Res<RenderAssets<R::GpuPlanarType>>,
    mut sort_buffers: ResMut<RadixSortBuffers<R>>,
    mut pending: ResMut<PendingRadixBufferChanges<R>>,
    render_device: Res<RenderDevice>,
) {
    let invalidated_assets = std::mem::take(&mut pending.invalidated_assets);
    sort_buffers
        .asset_map
        .retain(|asset_id, _| !invalidated_assets.contains(&asset_id.untyped()));

    // Defensive removal for a first-frame disappearance that had no dependent
    // component. This still precedes the allocation loop below.
    let live_assets = gpu_gaussian_clouds
        .iter()
        .map(|(asset_id, cloud)| (asset_id, cloud.len()))
        .collect::<HashMap<_, _>>();
    sort_buffers
        .asset_map
        .retain(|asset_id, _| live_assets.contains_key(asset_id));

    for (asset_id, cloud) in gpu_gaussian_clouds.iter() {
        let decision = radix_allocation_decision(
            sort_buffers
                .asset_map
                .get(&asset_id)
                .map(|buffers| buffers.capacity),
            Some(cloud.len()),
        );
        if decision == RadixAllocationDecision::Reuse {
            continue;
        }
        debug_assert_eq!(decision, RadixAllocationDecision::Replace);

        let generation = sort_buffers.next_generation;
        sort_buffers.next_generation = sort_buffers.next_generation.wrapping_add(1).max(1);
        let gpu_radix_buffers = GpuRadixBuffers::new(cloud.len(), generation, &render_device);
        sort_buffers.asset_map.insert(asset_id, gpu_radix_buffers);
    }
}

#[derive(Resource)]
pub struct RadixSortPipeline<R: PlanarSync> {
    pub radix_sort_layout: BindGroupLayout,
    pub variants: [Option<RadixSortPipelineVariant>; RADIX_DEPTH_BITS_VARIANT_COUNT],
    sorting_layout: Vec<BindGroupLayoutDescriptor>,
    phantom: std::marker::PhantomData<R>,
}

#[derive(Clone, Copy)]
pub struct RadixSortPipelineVariant {
    pub shader_defines: ShaderDefines,
    pub radix_sort_pipelines: [CachedComputePipelineId; RADIX_PIPELINE_COUNT],
}

impl RadixSortPipelineVariant {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.radix_sort_pipelines.iter().all(|sort_pipeline| {
            matches!(
                pipeline_cache.get_compute_pipeline_state(*sort_pipeline),
                CachedPipelineState::Ok(_)
            )
        })
    }
}

impl<R: PlanarSync> RadixSortPipeline<R> {
    fn variant(
        &self,
        radix_sort_depth_bits: RadixSortDepthBits,
    ) -> Option<&RadixSortPipelineVariant> {
        self.variants[radix_sort_depth_bits.pipeline_index()].as_ref()
    }

    pub(crate) fn queue_variant(
        &mut self,
        pipeline_cache: &PipelineCache,
        radix_sort_depth_bits: RadixSortDepthBits,
    ) {
        let index = radix_sort_depth_bits.pipeline_index();
        if self.variants[index].is_some() {
            return;
        }

        self.variants[index] = Some(queue_radix_sort_pipeline_variant(
            pipeline_cache,
            self.sorting_layout.clone(),
            radix_sort_depth_bits,
        ));
    }

    pub(crate) fn variant_is_loaded(
        &self,
        pipeline_cache: &PipelineCache,
        radix_sort_depth_bits: RadixSortDepthBits,
    ) -> bool {
        self.variant(radix_sort_depth_bits)
            .is_some_and(|variant| variant.is_loaded(pipeline_cache))
    }
}

impl<R: PlanarSync> FromWorld for RadixSortPipeline<R> {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();
        let gaussian_cloud_pipeline = render_world.resource::<CloudPipeline<R>>();

        let sorting_buffer_entry = BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(
                    ShaderDefines::default().sorting_buffer_size as u64,
                ),
            },
            count: None,
        };

        let sorting_status_counters_buffer_entry = BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(
                    ShaderDefines::default().sorting_status_counters_buffer_size(1) as u64,
                ),
            },
            count: None,
        };

        let draw_indirect_buffer_entry = BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                // Radix never mutates the exact instance count. Keeping this
                // binding read-only permits the same buffer to supply an
                // indirect dispatch in the active LoD path; read-write storage
                // is exclusive and conflicts with `BufferUses::INDIRECT`.
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(
                    std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64,
                ),
            },
            count: None,
        };

        let radix_sort_layout_entries = [
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<u32>() as u64),
                },
                count: None,
            },
            sorting_buffer_entry,
            sorting_status_counters_buffer_entry,
            draw_indirect_buffer_entry,
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<SortEntry>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<SortEntry>() as u64),
                },
                count: None,
            },
        ];
        let radix_sort_layout_desc =
            BindGroupLayoutDescriptor::new("radix_sort_layout", &radix_sort_layout_entries);
        let radix_sort_layout = render_device
            .create_bind_group_layout(Some("radix_sort_layout"), &radix_sort_layout_entries);

        let sorting_layout = vec![
            gaussian_cloud_pipeline.compute_view_layout_desc.clone(),
            gaussian_cloud_pipeline.gaussian_uniform_layout_desc.clone(),
            gaussian_cloud_pipeline.gaussian_cloud_layout_desc.clone(),
            radix_sort_layout_desc.clone(),
        ];

        let variants = [None; RADIX_DEPTH_BITS_VARIANT_COUNT];

        RadixSortPipeline {
            radix_sort_layout,
            variants,
            sorting_layout,
            phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extracted_view(
        world_from_view: GlobalTransform,
        clip_from_view: Mat4,
        viewport: UVec4,
    ) -> ExtractedView {
        ExtractedView {
            retained_view_entity: RetainedViewEntity::new(Entity::PLACEHOLDER.into(), None, 0),
            clip_from_view,
            world_from_view,
            clip_from_world: None,
            target_format: bevy::render::render_resource::TextureFormat::Rgba8Unorm,
            viewport,
            color_grading: bevy::render::view::ColorGrading::default(),
            invert_culling: false,
        }
    }

    fn legacy_camera_sort_signature(view: &ExtractedView) -> u64 {
        let mut hasher = DefaultHasher::new();
        hash_legacy_camera_sort_inputs(view, &mut hasher);
        hasher.finish()
    }

    fn blelloch_exclusive_scan(values: &mut [usize]) -> usize {
        assert!(values.len().is_power_of_two());
        let mut stride = 1;
        while stride < values.len() {
            for index in ((stride * 2 - 1)..values.len()).step_by(stride * 2) {
                values[index] += values[index - stride];
            }
            stride *= 2;
        }
        let total = *values.last().unwrap();
        *values.last_mut().unwrap() = 0;
        stride = values.len() / 2;
        loop {
            for index in ((stride * 2 - 1)..values.len()).step_by(stride * 2) {
                let left = index - stride;
                let left_value = values[left];
                let parent_value = values[index];
                values[left] = parent_value;
                values[index] = parent_value + left_value;
            }
            if stride == 1 {
                break;
            }
            stride /= 2;
        }
        total
    }

    fn stable_binary_digit_partition(entries: &[SortEntry], shift: u32) -> Vec<SortEntry> {
        let mut current = entries.to_vec();
        let mut next = vec![SortEntry::default(); entries.len()];
        for bit in 0..8 {
            let mut zero_prefix = vec![0usize; 1_024];
            for (index, entry) in current.iter().enumerate() {
                zero_prefix[index] = usize::from(((entry.key >> (shift + bit)) & 1) == 0);
            }
            let zero_count = blelloch_exclusive_scan(&mut zero_prefix);
            for (index, entry) in current.iter().enumerate() {
                let zeros_before = zero_prefix[index];
                let destination = if ((entry.key >> (shift + bit)) & 1) == 0 {
                    zeros_before
                } else {
                    zero_count + index - zeros_before
                };
                next[destination] = *entry;
            }
            std::mem::swap(&mut current, &mut next);
        }
        current
    }

    fn shader_radix_pass(input: &[SortEntry], shift: u32) -> Vec<SortEntry> {
        let mut global_counts = [0usize; 256];
        for entry in input {
            global_counts[((entry.key >> shift) & 0xff) as usize] += 1;
        }
        let mut global_offsets = global_counts;
        blelloch_exclusive_scan(&mut global_offsets);
        let mut earlier_tile_counts = [0usize; 256];
        let mut output = vec![SortEntry::default(); input.len()];
        for tile in input.chunks(1_024) {
            let sorted_tile = stable_binary_digit_partition(tile, shift);
            let mut local_rank = [0usize; 256];
            for entry in sorted_tile {
                let digit = ((entry.key >> shift) & 0xff) as usize;
                let destination =
                    global_offsets[digit] + earlier_tile_counts[digit] + local_rank[digit];
                output[destination] = entry;
                local_rank[digit] += 1;
            }
            for digit in 0..256 {
                earlier_tile_counts[digit] += local_rank[digit];
            }
        }
        output
    }

    #[test]
    fn radix_count_binding_is_read_only_for_indirect_dispatch_compatibility() {
        let host = include_str!("radix.rs");
        let shader = include_str!("radix.wgsl");
        assert!(host.contains("ty: BufferBindingType::Storage { read_only: true }"));
        assert!(shader.contains("var<storage, read> draw_indirect: RadixDrawIndirect"));
        assert!(shader.contains("return draw_indirect.instance_count"));
        assert!(!shader.contains("atomicLoad(&draw_indirect.instance_count)"));
    }

    #[test]
    fn legacy_camera_sort_signature_ignores_rotation_projection_and_viewport() {
        let position = Vec3::new(1.25, -3.5, 8.0);
        let base = extracted_view(
            GlobalTransform::from(Transform::from_translation(position)),
            Mat4::IDENTITY,
            UVec4::new(0, 0, 1280, 720),
        );
        let changed_view = extracted_view(
            GlobalTransform::from(
                Transform::from_translation(position).with_rotation(Quat::from_rotation_y(1.25)),
            ),
            Mat4::from_scale(Vec3::new(2.0, 3.0, 1.0)),
            UVec4::new(20, 40, 3840, 2160),
        );

        assert_eq!(
            legacy_camera_sort_signature(&base),
            legacy_camera_sort_signature(&changed_view)
        );
    }

    #[test]
    fn legacy_camera_sort_signature_invalidates_on_translation() {
        let base = extracted_view(
            GlobalTransform::from(Transform::from_xyz(1.25, -3.5, 8.0)),
            Mat4::IDENTITY,
            UVec4::new(0, 0, 1280, 720),
        );
        let translated = extracted_view(
            GlobalTransform::from(Transform::from_xyz(1.25, -3.5, 8.001)),
            Mat4::IDENTITY,
            UVec4::new(0, 0, 1280, 720),
        );

        assert_ne!(
            legacy_camera_sort_signature(&base),
            legacy_camera_sort_signature(&translated)
        );
    }

    #[test]
    fn legacy_sort_signature_invalidates_on_atlas_content_revision() {
        let view = extracted_view(
            GlobalTransform::from(Transform::from_xyz(1.25, -3.5, 8.0)),
            Mat4::IDENTITY,
            UVec4::new(0, 0, 1280, 720),
        );
        let transform = GlobalTransform::IDENTITY;
        let settings = CloudSettings::default();
        let signature = |revision| {
            legacy_sort_signature(
                &view,
                &transform,
                &settings,
                7,
                11_u64,
                1024,
                Some(13),
                revision,
            )
        };

        assert_eq!(signature(0), signature(0));
        assert_ne!(signature(0), signature(1));
        assert_ne!(signature(1), signature(2));
    }

    #[test]
    fn legacy_sort_cache_is_disabled_for_live_gpu_writers() {
        assert!(legacy_sort_cache_allowed(false, false));
        assert!(!legacy_sort_cache_allowed(true, false));
        assert!(!legacy_sort_cache_allowed(false, true));
        assert!(!legacy_sort_cache_allowed(true, true));
    }

    #[cfg(lod_render_path)]
    #[test]
    fn required_package_without_a_usable_lod_path_skips_legacy_sort() {
        assert!(!skip_legacy_sort_for_required_candidate(false));
        assert!(skip_legacy_sort_for_required_candidate(true));

        let host = include_str!("radix.rs");
        let run = host
            .rsplit("fn run_radix_sort")
            .next()
            .expect("radix runner");
        assert!(host.contains("Option<&'static LodRenderCandidates>"));
        let lod_path = run
            .find("lod_buffers.get_ready_mut")
            .expect("usable LoD radix path");
        let package_guard = run
            .find("skip_legacy_sort_for_required_candidate")
            .expect("candidate-required fallback guard");
        let legacy_lookup = run
            .find("sort_buffers.asset_map.get")
            .expect("legacy radix allocation lookup");
        let stale_cache_removal = run
            .find("work_cache.signatures.remove(&legacy_key)")
            .expect("required-package stale cache removal");
        assert!(
            lod_path < package_guard
                && package_guard < stale_cache_removal
                && stale_cache_removal < legacy_lookup
        );
        assert!(run[stale_cache_removal..legacy_lookup].contains("continue;"));
    }

    #[test]
    fn vanilla_radix_key_has_no_view_direction_or_projection_dependency() {
        let shader = include_str!("radix.wgsl");
        assert!(shader.contains("let diff = transformed_position - view.world_position"));
        assert!(shader.contains("Rotation-stable global pre-sort only"));
        assert!(!shader.contains("world_to_clip"));
        assert!(!shader.contains("in_frustum"));
        assert!(!shader.contains("view.clip_from_view"));
        assert!(!shader.contains("view.viewport"));
    }

    #[test]
    fn portable_radix_entry_points_never_exceed_256_invocations() {
        let shader = include_str!("radix.wgsl");
        assert!(shader.contains("@compute @workgroup_size(#{RADIX_BASE})\nfn radix_reset"));
        assert!(shader.contains("@compute @workgroup_size(#{RADIX_BASE})\nfn radix_sort_a"));
        assert!(shader.contains("@compute @workgroup_size(#{RADIX_BASE})\nfn radix_sort_active_a"));
        assert!(shader.contains("@compute @workgroup_size(#{RADIX_BASE})\nfn radix_sort_b"));
        assert!(!shader.contains("#{RADIX_BASE}, #{RADIX_DIGIT_PLACES}"));
        assert!(!shader.contains("@compute @workgroup_size(1)\nfn radix_sort_b"));
        assert!(shader.contains("@compute @workgroup_size(#{WORKGROUP_INVOCATIONS_C})"));
        assert_eq!(crate::render::ShaderDefines::default().radix_base, 256);
        assert_eq!(
            crate::render::ShaderDefines::default().workgroup_invocations_c,
            256
        );
    }

    #[test]
    fn radix_uses_parallel_stable_scans_without_capacity_clears() {
        let host = include_str!("radix.rs");
        let shader = include_str!("radix.wgsl");
        assert!(shader.contains("Eight stable binary partitions"));
        assert!(shader.contains("tile_prefix_values"));
        assert!(shader.contains("workgroup_digit_histogram"));
        assert!(shader.contains("base < tile_count; base += #{RADIX_BASE}u"));
        assert!(shader.contains("fn exclusive_radix_scan"));
        assert!(shader.contains("fn exclusive_tile_scan"));
        assert!(shader.contains("Work-efficient exclusive Blelloch scan"));
        assert!(shader.contains("radix_scan_values[index - stride]"));
        assert!(shader.contains("tile_prefix_values[left_index] = parent_value"));
        assert!(!shader.contains("local_digit_counts"));
        assert!(!shader.contains("while offset < #{RADIX_BASE}u"));
        assert!(!shader.contains("while offset < tile_size"));
        assert!(!shader.contains("zeros_inclusive"));
        assert!(!shader.contains("var addends:"));
        assert!(!shader.contains("var pass ="), "`pass` is reserved by WGSL");
        assert!(!shader.contains("let pass ="), "`pass` is reserved by WGSL");
        assert!(
            !shader.contains("let entry = select("),
            "WGSL select cannot choose between Entry structures"
        );
        let full_capacity_clear = ["clear", "_buffer"].concat();
        assert!(!host.contains(&full_capacity_clear));
        assert!(host.contains("dispatch_workgroups(1, shader_defines.radix_base, 1)"));
    }

    #[test]
    fn stale_radix_assets_are_removed_and_resizes_replace_exact_capacity() {
        assert_eq!(
            radix_allocation_decision(Some(1_000_000), None),
            RadixAllocationDecision::Remove
        );
        assert_eq!(
            radix_allocation_decision(Some(1_000_000), Some(1_000_000)),
            RadixAllocationDecision::Reuse
        );
        assert_eq!(
            radix_allocation_decision(Some(1_000_000), Some(32_000)),
            RadixAllocationDecision::Replace
        );
        assert_eq!(
            radix_allocation_decision(Some(32_000), Some(2_000_000)),
            RadixAllocationDecision::Replace
        );
        assert_eq!(
            radix_allocation_decision(None, Some(1_000)),
            RadixAllocationDecision::Replace
        );

        let host = include_str!("radix.rs");
        assert!(host.contains("commands.entity(entity).remove::<RadixBindGroup>()"));
        assert!(host.contains("work_cache.signatures"));
        assert!(host.contains("live_assets.contains_key(asset_id)"));
        assert!(host.contains("cloud_asset: UntypedAssetId"));

        let chained_transaction = host
            .find("invalidate_stale_radix_bind_groups::<R>,\n                    flush_radix_buffer_changes::<R>,\n                )\n                    .chain()")
            .expect("resize invalidation and flush must have an ApplyDeferred chain edge");
        let invalidate_phase = host
            .find("fn invalidate_stale_radix_bind_groups")
            .expect("legacy invalidation phase");
        let flush_phase = host
            .find("fn flush_radix_buffer_changes")
            .expect("legacy flush phase");
        assert!(chained_transaction < invalidate_phase && invalidate_phase < flush_phase);
        let flush_source = &host[flush_phase..];
        let old_buffer_drop = flush_source
            .find("!invalidated_assets.contains(&asset_id.untyped())")
            .expect("all invalidated old buffers are dropped together");
        let replacement_allocation = flush_source
            .find("GpuRadixBuffers::new")
            .expect("replacement allocation");
        assert!(
            old_buffer_drop < replacement_allocation,
            "old buffers must drop after dependent bind groups and before any replacement allocation"
        );
    }

    #[cfg(lod_render_path)]
    #[test]
    fn stable_lsd_passes_match_reference_for_every_depth_and_parity() {
        let source = (0..2_053u32)
            .map(|index| SortEntry {
                // Stay within 16 bits so all supported depths have one exact
                // reference order, with many equal keys spanning tile edges.
                key: index.wrapping_mul(73).wrapping_add(index / 17) % 257,
                index,
            })
            .rev()
            .collect::<Vec<_>>();
        let mut reference = source.clone();
        reference.sort_by_key(|entry| entry.key);

        for bits in RadixSortDepthBits::VARIANTS {
            let mut input = source.clone();
            for pass in 0..bits.bits() / 8 {
                input = shader_radix_pass(&input, pass * 8);
            }
            assert_eq!(input, reference, "{}-bit stable LSD parity", bits.bits());
            assert_eq!(
                (bits.bits() / 8) % 2,
                crate::render::lod::radix_sorted_output_buffer_index(bits) as u32
            );
        }
    }
}

fn queue_radix_sort_pipeline_variant(
    pipeline_cache: &PipelineCache,
    sorting_layout: Vec<BindGroupLayoutDescriptor>,
    radix_sort_depth_bits: RadixSortDepthBits,
) -> RadixSortPipelineVariant {
    let shader_defines = ShaderDefines::for_radix_depth_bits(radix_sort_depth_bits);
    let shader_defs = shader_defs_with_defines(CloudPipelineKey::default(), shader_defines);
    let label_suffix = radix_sort_depth_bits.bits();

    let radix_reset = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(format!("radix_sort_reset_{label_suffix}bit").into()),
        layout: sorting_layout.clone(),
        immediate_size: 0,
        shader: RADIX_SHADER_HANDLE,
        shader_defs: shader_defs.clone(),
        entry_point: Some("radix_reset".into()),
        zero_initialize_workgroup_memory: true,
    });

    let radix_sort_a = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(format!("radix_sort_a_{label_suffix}bit").into()),
        layout: sorting_layout.clone(),
        immediate_size: 0,
        shader: RADIX_SHADER_HANDLE,
        shader_defs: shader_defs.clone(),
        entry_point: Some("radix_sort_a".into()),
        zero_initialize_workgroup_memory: true,
    });

    let radix_sort_b = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(format!("radix_sort_b_{label_suffix}bit").into()),
        layout: sorting_layout.clone(),
        immediate_size: 0,
        shader: RADIX_SHADER_HANDLE,
        shader_defs: shader_defs.clone(),
        entry_point: Some("radix_sort_b".into()),
        zero_initialize_workgroup_memory: true,
    });

    let radix_sort_c_count = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(format!("radix_sort_c_count_tiles_{label_suffix}bit").into()),
        layout: sorting_layout.clone(),
        immediate_size: 0,
        shader: RADIX_SHADER_HANDLE,
        shader_defs: shader_defs.clone(),
        entry_point: Some("radix_sort_c_count_tiles".into()),
        zero_initialize_workgroup_memory: true,
    });

    let radix_sort_c_scan = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(format!("radix_sort_c_scan_tiles_{label_suffix}bit").into()),
        layout: sorting_layout.clone(),
        immediate_size: 0,
        shader: RADIX_SHADER_HANDLE,
        shader_defs: shader_defs.clone(),
        entry_point: Some("radix_sort_c_scan_tiles".into()),
        zero_initialize_workgroup_memory: true,
    });

    let radix_sort_c_scatter = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(format!("radix_sort_c_scatter_{label_suffix}bit").into()),
        layout: sorting_layout.clone(),
        immediate_size: 0,
        shader: RADIX_SHADER_HANDLE,
        shader_defs: shader_defs.clone(),
        entry_point: Some("radix_sort_c_scatter".into()),
        zero_initialize_workgroup_memory: true,
    });

    #[cfg(lod_render_path)]
    let radix_sort_active_a = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(format!("radix_sort_active_a_{label_suffix}bit").into()),
        layout: sorting_layout,
        immediate_size: 0,
        shader: RADIX_SHADER_HANDLE,
        shader_defs,
        entry_point: Some("radix_sort_active_a".into()),
        zero_initialize_workgroup_memory: true,
    });

    RadixSortPipelineVariant {
        shader_defines,
        radix_sort_pipelines: [
            radix_reset,
            radix_sort_a,
            radix_sort_b,
            radix_sort_c_count,
            radix_sort_c_scan,
            radix_sort_c_scatter,
            #[cfg(lod_render_path)]
            radix_sort_active_a,
        ],
    }
}

#[derive(Component)]
pub struct RadixBindGroup {
    // For each digit pass idx in 0..RADIX_DIGIT_PLACES, we create 2 bind groups (parity 0/1):
    // index = pass_idx * 2 + parity (parity 0: input=sorted_entries, output=entry_buffer_b; parity 1: input=entry_buffer_b, output=sorted_entries)
    pub radix_sort_bind_groups: [BindGroup; 8],
    cloud_asset: UntypedAssetId,
    buffer_generation: u64,
    sorted_entry_buffer_id: BufferId,
    sorted_entry_buffer_size: u64,
    radix_depth_bits: RadixSortDepthBits,
}

#[cfg(lod_render_path)]
mod lod_bind_groups {
    use super::*;

    pub(super) struct LodRadixBindGroup {
        pub(super) generation: u64,
        pub(super) groups: [BindGroup; 4],
    }

    #[derive(Resource)]
    pub(crate) struct LodRadixBindGroups<R: PlanarSync> {
        pub(super) entries:
            HashMap<(RetainedViewEntity, Entity, AssetId<R::PlanarType>), LodRadixBindGroup>,
    }

    impl<R: PlanarSync> Default for LodRadixBindGroups<R> {
        fn default() -> Self {
            Self {
                entries: HashMap::new(),
            }
        }
    }

    impl<R: PlanarSync> LodRadixBindGroups<R> {
        /// Release dependent bind groups before their compaction buffers are dropped or replaced.
        /// A bind group retains its bound buffers, so ordering this ahead of allocation is required
        /// for the compaction aggregate budget to remain a peak-live-memory bound.
        pub(crate) fn retain_keys(
            &mut self,
            active: &HashSet<(RetainedViewEntity, Entity, AssetId<R::PlanarType>)>,
        ) {
            self.entries.retain(|key, _| active.contains(key));
        }

        pub(crate) fn remove(
            &mut self,
            key: &(RetainedViewEntity, Entity, AssetId<R::PlanarType>),
        ) {
            self.entries.remove(key);
        }
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn queue_lod_radix_bind_groups<R: PlanarSync>(
        mut groups: ResMut<LodRadixBindGroups<R>>,
        mut radix_pipeline: ResMut<RadixSortPipeline<R>>,
        pipeline_cache: Res<PipelineCache>,
        render_device: Res<RenderDevice>,
        lod_buffers: Res<LodCompactionBuffers<R>>,
        views: Query<(&ExtractedView, &RenderVisibleEntities), With<GaussianCamera>>,
        clouds: Query<(Entity, &R::PlanarTypeHandle, &CloudSettings)>,
    ) where
        R::GpuPlanarType: GpuPlanarStorage,
    {
        let mut active = HashSet::new();
        for (view, visible_entities) in &views {
            let Some(visible_clouds) = visible_entities.get::<CloudVisibilityClass>() else {
                continue;
            };
            for (render_entity, _) in &visible_clouds.entities_cpu_culling {
                let Ok((entity, handle, settings)) = clouds.get(*render_entity) else {
                    continue;
                };
                if settings.sort_mode != SortMode::Radix {
                    continue;
                }
                let key =
                    lod_view_cloud_key(view.retained_view_entity, entity, handle.handle().id());
                let Some(state) = lod_buffers
                    .get(key.0, key.1, key.2)
                    .filter(|state| state.has_staged_candidates())
                else {
                    continue;
                };
                active.insert(key);
                radix_pipeline.queue_variant(&pipeline_cache, settings.radix_sort_depth_bits);
                if groups
                    .entries
                    .get(&key)
                    .is_some_and(|group| group.generation == state.generation())
                {
                    continue;
                }

                let radix_groups = std::array::from_fn(|pass_index| {
                    let (input, output) = if pass_index % 2 == 0 {
                        (&state.active_entries_buffer, &state.radix_scratch_buffer)
                    } else {
                        (&state.radix_scratch_buffer, &state.active_entries_buffer)
                    };
                    render_device.create_bind_group(
                        "gaussian_lod_radix_bind_group",
                        &radix_pipeline.radix_sort_layout,
                        &[
                            BindGroupEntry {
                                binding: 0,
                                resource: state.sorting_pass_buffers[pass_index]
                                    .as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: state.sorting_global_buffer.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 2,
                                resource: state.sorting_status_counter_buffer.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 3,
                                resource: state.indirect_args_buffer.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 4,
                                resource: input.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 5,
                                resource: output.as_entire_binding(),
                            },
                        ],
                    )
                });
                groups.entries.insert(
                    key,
                    LodRadixBindGroup {
                        generation: state.generation(),
                        groups: radix_groups,
                    },
                );
            }
        }
        groups.entries.retain(|key, _| active.contains(key));
    }
}

#[cfg(lod_render_path)]
pub(crate) use lod_bind_groups::{LodRadixBindGroups, queue_lod_radix_bind_groups};

type RadixViewQueryItem = (
    &'static GaussianCamera,
    &'static ExtractedView,
    &'static RenderVisibleEntities,
    &'static crate::render::GaussianComputeViewBindGroup,
    &'static ViewUniformOffset,
    &'static PreviousViewUniformOffset,
);

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn queue_radix_bind_group<R: PlanarSync>(
    mut commands: Commands,
    mut radix_pipeline: ResMut<RadixSortPipeline<R>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    gaussian_cloud_res: Res<RenderAssets<R::GpuPlanarType>>,
    sorted_entries_res: Res<RenderAssets<GpuSortedEntry>>,
    gaussian_clouds: Query<(
        Entity,
        &R::PlanarTypeHandle,
        &SortedEntriesHandle,
        &CloudSettings,
        Option<&RadixBindGroup>,
    )>,
    sort_buffers: Res<RadixSortBuffers<R>>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    for (entity, cloud_handle, sorted_entries_handle, settings, existing) in gaussian_clouds.iter()
    {
        if settings.sort_mode != SortMode::Radix {
            commands.entity(entity).remove::<RadixBindGroup>();
            continue;
        }

        radix_pipeline.queue_variant(&pipeline_cache, settings.radix_sort_depth_bits);

        // TODO: deduplicate asset load checks
        if let Some(load_state) = asset_server.get_load_state(cloud_handle.handle())
            && load_state.is_loading()
        {
            continue;
        }

        let Some(cloud) = gaussian_cloud_res.get(cloud_handle.handle()) else {
            commands.entity(entity).remove::<RadixBindGroup>();
            continue;
        };

        if let Some(load_state) = asset_server.get_load_state(&sorted_entries_handle.0)
            && load_state.is_loading()
        {
            continue;
        }

        let Some(sorted_entries) = sorted_entries_res.get(sorted_entries_handle) else {
            continue;
        };

        let Some(sorted_entry_binding_size) =
            sort_entry_binding_size(sorted_entries.entry_count, cloud.len())
        else {
            // The LoD bridge can publish a larger atlas handle one frame
            // before `SortedEntries` is resized. Never create a legacy radix
            // bind group whose declared range exceeds the old GPU buffer.
            commands.entity(entity).remove::<RadixBindGroup>();
            continue;
        };

        if !sort_buffers
            .asset_map
            .contains_key(&cloud_handle.handle().id())
        {
            commands.entity(entity).remove::<RadixBindGroup>();
            continue;
        }

        let sorting_assets = &sort_buffers.asset_map[&cloud_handle.handle().id()];
        if existing.is_some_and(|existing| {
            existing.cloud_asset == cloud_handle.handle().id().untyped()
                && existing.buffer_generation == sorting_assets.generation
                && existing.sorted_entry_buffer_id == sorted_entries.sorted_entry_buffer.id()
                && existing.sorted_entry_buffer_size == sorted_entries.sorted_entry_buffer.size()
                && existing.radix_depth_bits == settings.radix_sort_depth_bits
        }) {
            continue;
        }

        let sorting_global_entry = BindGroupEntry {
            binding: 1,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &sorting_assets.sorting_global_buffer,
                offset: 0,
                size: BufferSize::new(sorting_assets.sorting_global_buffer.size()),
            }),
        };

        let sorting_status_counters_entry = BindGroupEntry {
            binding: 2,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &sorting_assets.sorting_status_counter_buffer,
                offset: 0,
                size: BufferSize::new(sorting_assets.sorting_status_counter_buffer.size()),
            }),
        };

        let draw_indirect_entry = BindGroupEntry {
            binding: 3,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: cloud.draw_indirect_buffer(),
                offset: 0,
                size: BufferSize::new(cloud.draw_indirect_buffer().size()),
            }),
        };

        let radix_sort_bind_groups: [BindGroup; 8] = {
            let mut groups: Vec<BindGroup> = Vec::with_capacity(8);
            for pass_idx in 0..4 {
                for parity in 0..=1 {
                    let (input_buf, output_buf) = if parity == 0 {
                        (
                            &sorted_entries.sorted_entry_buffer,
                            &sorting_assets.entry_buffer_b,
                        )
                    } else {
                        (
                            &sorting_assets.entry_buffer_b,
                            &sorted_entries.sorted_entry_buffer,
                        )
                    };

                    let group = render_device.create_bind_group(
                        format!("radix_sort_bind_group pass={pass_idx} parity={parity}").as_str(),
                        &radix_pipeline.radix_sort_layout,
                        &[
                            // sorting_pass_index (u32) == pass_idx regardless of parity
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: &sorting_assets.sorting_pass_buffers[pass_idx],
                                    offset: 0,
                                    size: BufferSize::new(std::mem::size_of::<u32>() as u64),
                                }),
                            },
                            sorting_global_entry.clone(),
                            sorting_status_counters_entry.clone(),
                            draw_indirect_entry.clone(),
                            // input_entries
                            BindGroupEntry {
                                binding: 4,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: input_buf,
                                    offset: 0,
                                    size: BufferSize::new(sorted_entry_binding_size),
                                }),
                            },
                            // output_entries
                            BindGroupEntry {
                                binding: 5,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: output_buf,
                                    offset: 0,
                                    size: BufferSize::new(sorted_entry_binding_size),
                                }),
                            },
                        ],
                    );
                    groups.push(group);
                }
            }
            groups.try_into().unwrap()
        };

        commands.entity(entity).insert(RadixBindGroup {
            radix_sort_bind_groups,
            cloud_asset: cloud_handle.handle().id().untyped(),
            buffer_generation: sorting_assets.generation,
            sorted_entry_buffer_id: sorted_entries.sorted_entry_buffer.id(),
            sorted_entry_buffer_size: sorted_entries.sorted_entry_buffer.size(),
            radix_depth_bits: settings.radix_sort_depth_bits,
        });
    }
}

#[allow(type_alias_bounds)]
#[cfg(lod_render_path)]
type RadixCloudQueryItem<R: PlanarSync> = (
    Entity,
    &'static R::PlanarTypeHandle,
    Ref<'static, PlanarStorageBindGroup<R>>,
    &'static RadixBindGroup,
    &'static DynamicUniformIndex<CloudUniform>,
    &'static CloudSettings,
    &'static GlobalTransform,
    Option<&'static LodRenderCandidates>,
);

#[allow(type_alias_bounds)]
#[cfg(not(lod_render_path))]
type RadixCloudQueryItem<R: PlanarSync> = (
    Entity,
    &'static R::PlanarTypeHandle,
    Ref<'static, PlanarStorageBindGroup<R>>,
    &'static RadixBindGroup,
    &'static DynamicUniformIndex<CloudUniform>,
    &'static CloudSettings,
    &'static GlobalTransform,
);

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_radix_sort<R: PlanarSync>(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<RadixSortPipeline<R>>,
    gaussian_uniforms: Res<GaussianUniformBindGroups>,
    sort_buffers: Res<RadixSortBuffers<R>>,
    mut work_cache: ResMut<RadixSortWorkCache<R>>,
    #[cfg(lod_render_path)] mut lod_buffers: ResMut<LodCompactionBuffers<R>>,
    #[cfg(lod_render_path)] lod_radix_groups: Res<LodRadixBindGroups<R>>,
    #[cfg(lod_render_path)] atlas_generations: Res<LodAtlasGpuGenerations>,
    gpu_planars: Res<RenderAssets<R::GpuPlanarType>>,
    view_bind_group: ViewQuery<RadixViewQueryItem>,
    #[cfg(feature = "morph_interpolate")] interpolate_writers: Query<
        (),
        With<GaussianInterpolate<R>>,
    >,
    #[cfg(feature = "morph_particles")] particle_writers: Query<(), With<ParticleBehaviorsHandle>>,
    gaussian_clouds: Query<RadixCloudQueryItem<R>>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    let (
        _camera,
        _extracted_view,
        visible_entities,
        view_bind_group,
        view_uniform_offset,
        previous_view_uniform_offset,
    ) = view_bind_group.into_inner();

    let Some(uniform_bind_group) = gaussian_uniforms.base_bind_group.as_ref() else {
        debug!("RadixSort run skipped: GaussianUniform base bind group missing");
        return;
    };

    let Some(visible_clouds) = visible_entities.get::<CloudVisibilityClass>() else {
        return;
    };

    for (render_entity, _) in &visible_clouds.entities_cpu_culling {
        let Ok(cloud_item) = gaussian_clouds.get(*render_entity) else {
            continue;
        };
        #[cfg(lod_render_path)]
        let (
            _cloud_entity,
            cloud_handle,
            cloud_bind_group,
            radix_bind_group,
            cloud_uniform_index,
            cloud_settings,
            transform,
            render_candidates,
        ) = cloud_item;
        #[cfg(not(lod_render_path))]
        let (
            _cloud_entity,
            cloud_handle,
            cloud_bind_group,
            radix_bind_group,
            cloud_uniform_index,
            cloud_settings,
            transform,
        ) = cloud_item;
        let Some(cloud) = gpu_planars.get(cloud_handle.handle()) else {
            continue;
        };

        let Some(pipeline_variant) = pipeline.variant(cloud_settings.radix_sort_depth_bits) else {
            continue;
        };
        if !pipeline_variant.is_loaded(&pipeline_cache) {
            continue;
        }

        let shader_defines = pipeline_variant.shader_defines;
        let radix_digit_places = shader_defines.radix_digit_places;
        let initial_parity = shader_defines.radix_initial_parity();
        let workgroup_entries_a =
            shader_defines.radix_base * shader_defines.entries_per_invocation_a;
        let workgroup_entries_c = shader_defines.workgroup_entries_c;

        #[cfg(lod_render_path)]
        if let (Some(state), Some(radix_groups)) = (
            lod_buffers.get_ready_mut(
                _extracted_view.retained_view_entity,
                _cloud_entity,
                cloud_handle.handle().id(),
            ),
            lod_radix_groups.entries.get(&lod_view_cloud_key(
                _extracted_view.retained_view_entity,
                _cloud_entity,
                cloud_handle.handle().id(),
            )),
        ) {
            if state.radix_sort_is_current() {
                continue;
            }

            macro_rules! radix_direct_stage {
                ($label:literal, $pipeline_index:expr, $group:expr, $x:expr, $y:expr, $z:expr) => {{
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some($label),
                            ..default()
                        },
                    );
                    pass.set_bind_group(
                        0,
                        &view_bind_group.value,
                        &[
                            view_uniform_offset.offset,
                            previous_view_uniform_offset.offset,
                        ],
                    );
                    pass.set_bind_group(1, uniform_bind_group, &[cloud_uniform_index.index()]);
                    pass.set_bind_group(2, &cloud_bind_group.bind_group, &[]);
                    pass.set_bind_group(3, $group, &[]);
                    pass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(
                                pipeline_variant.radix_sort_pipelines[$pipeline_index],
                            )
                            .expect("loaded radix pipeline"),
                    );
                    pass.dispatch_workgroups($x, $y, $z);
                }};
            }

            macro_rules! radix_indirect_stage {
                ($label:literal, $pipeline_index:expr, $group:expr, $offset:expr) => {{
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some($label),
                            ..default()
                        },
                    );
                    pass.set_bind_group(
                        0,
                        &view_bind_group.value,
                        &[
                            view_uniform_offset.offset,
                            previous_view_uniform_offset.offset,
                        ],
                    );
                    pass.set_bind_group(1, uniform_bind_group, &[cloud_uniform_index.index()]);
                    pass.set_bind_group(2, &cloud_bind_group.bind_group, &[]);
                    pass.set_bind_group(3, $group, &[]);
                    pass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(
                                pipeline_variant.radix_sort_pipelines[$pipeline_index],
                            )
                            .expect("loaded radix pipeline"),
                    );
                    pass.dispatch_workgroups_indirect(&state.indirect_args_buffer, $offset);
                }};
            }

            radix_direct_stage!(
                "lod_radix_reset",
                RADIX_PIPELINE_RESET,
                &radix_groups.groups[0],
                1,
                1,
                1
            );
            radix_indirect_stage!(
                "lod_radix_histogram",
                RADIX_PIPELINE_ACTIVE_A,
                &radix_groups.groups[0],
                DISPATCH_A_INDIRECT_OFFSET
            );
            radix_direct_stage!(
                "lod_radix_histogram_scan",
                RADIX_PIPELINE_B,
                &radix_groups.groups[0],
                1,
                radix_digit_places,
                1
            );

            for pass_idx in 0..radix_digit_places {
                let group = &radix_groups.groups[pass_idx as usize];
                radix_indirect_stage!(
                    "lod_radix_tile_count",
                    RADIX_PIPELINE_C_COUNT,
                    group,
                    DISPATCH_C_INDIRECT_OFFSET
                );
                radix_direct_stage!(
                    "lod_radix_tile_scan",
                    RADIX_PIPELINE_C_SCAN,
                    group,
                    1,
                    shader_defines.radix_base,
                    1
                );
                radix_indirect_stage!(
                    "lod_radix_scatter",
                    RADIX_PIPELINE_C_SCATTER,
                    group,
                    DISPATCH_C_INDIRECT_OFFSET
                );
            }
            state.mark_radix_sorted();
            continue;
        }

        let legacy_key = (
            _extracted_view.retained_view_entity,
            _cloud_entity,
            cloud_handle.handle().id(),
        );
        #[cfg(lod_render_path)]
        if skip_legacy_sort_for_required_candidate(
            render_candidates.is_some_and(|candidates| candidates.candidate_draw_required),
        ) {
            // The draw command rejects an unfiltered package atlas whenever no
            // usable per-view LoD output exists. Sorting that same full atlas
            // cannot produce a draw and is especially costly during cold load
            // or device recovery.
            work_cache.signatures.remove(&legacy_key);
            continue;
        }
        let Some(sorting_assets) = sort_buffers.asset_map.get(&cloud_handle.handle().id()) else {
            continue;
        };
        #[cfg(feature = "morph_interpolate")]
        let has_interpolate = interpolate_writers.get(_cloud_entity).is_ok();
        #[cfg(not(feature = "morph_interpolate"))]
        let has_interpolate = false;
        #[cfg(feature = "morph_particles")]
        let has_particles = particle_writers.get(_cloud_entity).is_ok();
        #[cfg(not(feature = "morph_particles"))]
        let has_particles = false;
        let cache_allowed = legacy_sort_cache_allowed(has_interpolate, has_particles);
        #[cfg(lod_render_path)]
        let atlas_content_revision =
            atlas_generations.content_revision(cloud_handle.handle().id().untyped());
        #[cfg(not(lod_render_path))]
        let atlas_content_revision = 0;
        let signature = legacy_sort_signature(
            _extracted_view,
            transform,
            cloud_settings,
            sorting_assets.generation,
            radix_bind_group.sorted_entry_buffer_id,
            cloud.len(),
            Some(cloud_bind_group.last_changed().get()),
            atlas_content_revision,
        );
        if cache_allowed && work_cache.signatures.get(&legacy_key) == Some(&signature) {
            continue;
        }
        if cache_allowed && work_cache.signatures.len() >= 65_536 {
            work_cache.signatures.clear();
        }
        let tile_workgroups = (cloud.len() as u32).div_ceil(workgroup_entries_c);
        let command_encoder = render_context.command_encoder();
        // Draw counts are initialized with the flat cloud and, when LoD is
        // active, owned by a distinct per-view compaction buffer. Radix only
        // sorts entries and must never replace an exact post-filter count.

        {
            let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

            // Reset per-frame counters/histograms
            let radix_reset = pipeline_cache
                .get_compute_pipeline(pipeline_variant.radix_sort_pipelines[RADIX_PIPELINE_RESET])
                .unwrap();
            pass.set_pipeline(radix_reset);
            pass.set_bind_group(
                0,
                &view_bind_group.value,
                &[
                    view_uniform_offset.offset,
                    previous_view_uniform_offset.offset,
                ],
            );
            pass.set_bind_group(1, uniform_bind_group, &[cloud_uniform_index.index()]);
            pass.set_bind_group(2, &cloud_bind_group.bind_group, &[]);
            pass.set_bind_group(
                3,
                &radix_bind_group.radix_sort_bind_groups[initial_parity],
                &[],
            );
            pass.dispatch_workgroups(1, 1, 1);

            let radix_sort_a = pipeline_cache
                .get_compute_pipeline(pipeline_variant.radix_sort_pipelines[RADIX_PIPELINE_A])
                .unwrap();
            pass.set_pipeline(radix_sort_a);

            pass.dispatch_workgroups((cloud.len() as u32).div_ceil(workgroup_entries_a), 1, 1);

            let radix_sort_b = pipeline_cache
                .get_compute_pipeline(pipeline_variant.radix_sort_pipelines[RADIX_PIPELINE_B])
                .unwrap();
            pass.set_pipeline(radix_sort_b);

            pass.dispatch_workgroups(1, radix_digit_places, 1);
        }

        // TODO: add options to only complete a fraction of the sorting process
        for pass_idx in 0..radix_digit_places {
            let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

            // Set common bind groups for view/uniforms and cloud storage
            pass.set_bind_group(
                0,
                &view_bind_group.value,
                &[
                    view_uniform_offset.offset,
                    previous_view_uniform_offset.offset,
                ],
            );
            pass.set_bind_group(1, uniform_bind_group, &[cloud_uniform_index.index()]);
            pass.set_bind_group(2, &cloud_bind_group.bind_group, &[]);

            // Choose the initial parity so the final pass writes to sorted_entries.
            let parity = ((pass_idx as usize) + initial_parity) % 2;
            let bg_index = (pass_idx as usize) * 2 + parity;
            pass.set_bind_group(3, &radix_bind_group.radix_sort_bind_groups[bg_index], &[]);

            let radix_sort_c_count = pipeline_cache
                .get_compute_pipeline(pipeline_variant.radix_sort_pipelines[RADIX_PIPELINE_C_COUNT])
                .unwrap();
            pass.set_pipeline(radix_sort_c_count);
            pass.dispatch_workgroups(1, tile_workgroups, 1);

            let radix_sort_c_scan = pipeline_cache
                .get_compute_pipeline(pipeline_variant.radix_sort_pipelines[RADIX_PIPELINE_C_SCAN])
                .unwrap();
            pass.set_pipeline(radix_sort_c_scan);
            // One 256-lane workgroup per digit scans tiles in 256-wide chunks.
            pass.dispatch_workgroups(1, shader_defines.radix_base, 1);

            let radix_sort_c_scatter = pipeline_cache
                .get_compute_pipeline(
                    pipeline_variant.radix_sort_pipelines[RADIX_PIPELINE_C_SCATTER],
                )
                .unwrap();
            pass.set_pipeline(radix_sort_c_scatter);
            pass.dispatch_workgroups(1, tile_workgroups, 1);
        }
        if cache_allowed {
            work_cache.signatures.insert(legacy_key, signature);
        } else {
            work_cache.signatures.remove(&legacy_key);
        }
    }
}
