use std::collections::HashMap;

use bevy::{
    prelude::*,
    asset::load_internal_asset,
    core_pipeline::core_3d::graph::{
        Core3d,
        Node3d,
    },
    render::{
        render_asset::RenderAssets,
        render_resource::{
            BindGroup,
            BindGroupEntry,
            BindGroupLayout,
            BindGroupLayoutEntry,
            BindingResource,
            BindingType,
            Buffer,
            BufferBindingType,
            BufferDescriptor,
            BufferInitDescriptor,
            BufferBinding,
            BufferSize,
            BufferUsages,
            CachedComputePipelineId,
            CachedPipelineState,
            ComputePassDescriptor,
            ComputePipelineDescriptor,
            PipelineCache,
            ShaderStages,
        },
        renderer::{
            RenderContext,
            RenderDevice,
        },
        render_graph::{
            Node,
            NodeRunError,
            RenderGraphApp,
            RenderGraphContext,
            RenderLabel,
        },
        Render,
        RenderApp,
        RenderSet,
        view::ViewUniformOffset,
    },
};
use bevy_interleave::{
    prelude::*,
    interface::storage::PlanarStorageBindGroup,
};
use static_assertions::assert_cfg;

use crate::{
    CloudSettings,
    GaussianCamera,
    render::{
        CloudPipeline,
        CloudPipelineKey,
        GaussianUniformBindGroups,
        GaussianViewBindGroup,
        ShaderDefines,
        shader_defs,
    },
    sort::{
        GpuSortedEntry,
        SortEntry,
        SortedEntriesHandle,
        SortMode,
        SortPluginFlag,
    },
};


assert_cfg!(
    not(all(
        feature = "sort_radix",
        feature = "buffer_texture",
    )),
    "sort_radix and buffer_texture are incompatible",
);


const RADIX_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(6234673214);
const TEMPORAL_SORT_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(1634543224);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct RadixSortLabel;


#[derive(Default)]
pub struct RadixSortPlugin<R: PlanarStorage> {
    phantom: std::marker::PhantomData<R>,
}

impl<R: PlanarStorage> Plugin for RadixSortPlugin<R> {
    fn build(&self, app: &mut App) {
        // TODO: run once
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_systems(
                    Render,
                    (
                        queue_radix_bind_group::<R>.in_set(RenderSet::Queue),
                    ),
                );

            render_app.init_resource::<RadixSortBuffers<R>>();
            render_app.add_systems(
                ExtractSchedule,
                update_sort_buffers::<R>,
            );
        }

        if app.is_plugin_added::<SortPluginFlag>() {
            debug!("sort plugin already added");
            return;
        }

        load_internal_asset!(
            app,
            RADIX_SHADER_HANDLE,
            "radix.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            TEMPORAL_SORT_SHADER_HANDLE,
            "temporal.wgsl",
            Shader::from_wgsl
        );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_graph_node::<RadixSortNode<R>>(
                    Core3d,
                    RadixSortLabel,
                )
                .add_render_graph_edge(
                    Core3d,
                    RadixSortLabel,
                    Node3d::Prepass,
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<RadixSortPipeline<R>>();
        }
    }
}

#[derive(Resource)]
pub struct RadixSortBuffers<R: PlanarStorage> {
    // TODO: use a more ECS-friendly approach
    pub asset_map: HashMap<
        AssetId<R::PlanarType>,
        GpuRadixBuffers,
    >,
}

impl<R: PlanarStorage> Default for RadixSortBuffers<R> {
    fn default() -> Self {
        RadixSortBuffers {
            asset_map: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuRadixBuffers {
    pub sorting_global_buffer: Buffer,
    pub sorting_status_counter_buffer: Buffer,
    pub sorting_pass_buffers: [Buffer; 4],
    pub entry_buffer_b: Buffer,
}
impl GpuRadixBuffers {
    pub fn new(
        count: usize,
        render_device: &RenderDevice,
    ) -> Self {
        let sorting_global_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("sorting global buffer"),
            size: ShaderDefines::default().sorting_buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sorting_status_counter_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("status counters buffer"),
            size: ShaderDefines::default().sorting_status_counters_buffer_size(count) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sorting_pass_buffers = (0..4)
            .map(|idx| {
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: format!("sorting pass buffer {}", idx).as_str().into(),
                    contents: &[idx as u8, 0, 0, 0],
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                })
            })
            .collect::<Vec<Buffer>>()
            .try_into()
            .unwrap();

        let entry_buffer_b = render_device.create_buffer(&BufferDescriptor {
            label: Some("entry buffer b"),
            size: (count * std::mem::size_of::<SortEntry>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        GpuRadixBuffers {
            sorting_global_buffer,
            sorting_status_counter_buffer,
            sorting_pass_buffers,
            entry_buffer_b,
        }
    }
}


fn update_sort_buffers<R: PlanarStorage>(
    gpu_gaussian_clouds: Res<RenderAssets<R::GpuPlanarType>>,
    mut sort_buffers: ResMut<RadixSortBuffers<R>>,
    render_device: Res<RenderDevice>,
) {
    for (asset_id, cloud,) in gpu_gaussian_clouds.iter() {
        // TODO: handle cloud resize operations and resolve leaked stale buffers
        if sort_buffers.asset_map.contains_key(&asset_id) {
            continue;
        }

        let gpu_radix_buffers = GpuRadixBuffers::new(cloud.len(), &render_device);
        sort_buffers.asset_map.insert(asset_id, gpu_radix_buffers);
    }
}


#[derive(Resource)]
pub struct RadixSortPipeline<R: PlanarStorage> {
    pub radix_sort_layout: BindGroupLayout,
    pub radix_sort_pipelines: [CachedComputePipelineId; 3],
    phantom: std::marker::PhantomData<R>,
}

impl<R: PlanarStorage> FromWorld for RadixSortPipeline<R> {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();
        let gaussian_cloud_pipeline = render_world.resource::<CloudPipeline<R>>();

        let sorting_buffer_entry = BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(ShaderDefines::default().sorting_buffer_size as u64),
            },
            count: None,
        };

        let sorting_status_counters_buffer_entry = BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(ShaderDefines::default().sorting_status_counters_buffer_size(1) as u64),
            },
            count: None,
        };

        let draw_indirect_buffer_entry = BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64),
            },
            count: None,
        };

        let radix_sort_layout = render_device.create_bind_group_layout(
            Some("radix_sort_layout"),
            &[
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
            ],
        );

        let sorting_layout = vec![
            gaussian_cloud_pipeline.view_layout.clone(),
            gaussian_cloud_pipeline.gaussian_uniform_layout.clone(),
            gaussian_cloud_pipeline.gaussian_cloud_layout.clone(),
            radix_sort_layout.clone(),
        ];
        let shader_defs = shader_defs(CloudPipelineKey::default());

        let pipeline_cache = render_world.resource::<PipelineCache>();
        let radix_sort_a = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radix_sort_a".into()),
            layout: sorting_layout.clone(),
            push_constant_ranges: vec![],
            shader: RADIX_SHADER_HANDLE,
            shader_defs: shader_defs.clone(),
            entry_point: "radix_sort_a".into(),
            zero_initialize_workgroup_memory: true,
        });

        let radix_sort_b = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radix_sort_b".into()),
            layout: sorting_layout.clone(),
            push_constant_ranges: vec![],
            shader: RADIX_SHADER_HANDLE,
            shader_defs: shader_defs.clone(),
            entry_point: "radix_sort_b".into(),
            zero_initialize_workgroup_memory: true,
        });

        let radix_sort_c = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radix_sort_c".into()),
            layout: sorting_layout.clone(),
            push_constant_ranges: vec![],
            shader: RADIX_SHADER_HANDLE,
            shader_defs: shader_defs.clone(),
            entry_point: "radix_sort_c".into(),
            zero_initialize_workgroup_memory: true,
        });

        RadixSortPipeline {
            radix_sort_layout,
            radix_sort_pipelines: [
                radix_sort_a,
                radix_sort_b,
                radix_sort_c,
            ],
            phantom: std::marker::PhantomData,
        }
    }
}



#[derive(Component)]
pub struct RadixBindGroup {
    pub radix_sort_bind_groups: [BindGroup; 4],
}

#[allow(clippy::too_many_arguments)]
pub fn queue_radix_bind_group<R: PlanarStorage>(
    mut commands: Commands,
    radix_pipeline: Res<RadixSortPipeline<R>>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    gaussian_cloud_res: Res<RenderAssets<R::GpuPlanarType>>,
    sorted_entries_res: Res<RenderAssets<GpuSortedEntry>>,
    gaussian_clouds: Query<(
        Entity,
        &R::PlanarTypeHandle,
        &SortedEntriesHandle,
        &CloudSettings,
    )>,
    sort_buffers: Res<RadixSortBuffers<R>>,
) {
    for (
        entity,
        cloud_handle,
        sorted_entries_handle,
        settings,
    ) in gaussian_clouds.iter() {
        if settings.sort_mode != SortMode::Radix {
            continue;
        }

        // TODO: deduplicate asset load checks
        if let Some(load_state) = asset_server.get_load_state(cloud_handle.handle()) {
            if load_state.is_loading() {
                continue;
            }
        }

        if gaussian_cloud_res.get(cloud_handle.handle()).is_none() {
            continue;
        }

        if let Some(load_state) = asset_server.get_load_state(&sorted_entries_handle.0) {
            if load_state.is_loading() {
                continue;
            }
        }

        if sorted_entries_res.get(sorted_entries_handle).is_none() {
            continue;
        }

        if !sort_buffers.asset_map.contains_key(&cloud_handle.handle().id()) {
            continue;
        }

        let cloud = gaussian_cloud_res.get(cloud_handle.handle()).unwrap();
        let sorted_entries = sorted_entries_res.get(sorted_entries_handle).unwrap();
        let sorting_assets = &sort_buffers.asset_map[&cloud_handle.handle().id()];

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

        let radix_sort_bind_groups: [BindGroup; 4] = (0..4)
            .map(|idx| {
                render_device.create_bind_group(
                    format!("radix_sort_bind_group {}", idx).as_str(),
                    &radix_pipeline.radix_sort_layout,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: &sorting_assets.sorting_pass_buffers[idx],
                                offset: 0,
                                size: BufferSize::new(std::mem::size_of::<u32>() as u64),
                            }),
                        },
                        sorting_global_entry.clone(),
                        sorting_status_counters_entry.clone(),
                        draw_indirect_entry.clone(),
                        BindGroupEntry {
                            binding: 4,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: if idx % 2 == 0 {
                                    &sorted_entries.sorted_entry_buffer
                                } else {
                                    &sorting_assets.entry_buffer_b
                                },
                                offset: 0,
                                size: BufferSize::new((cloud.len() * std::mem::size_of::<SortEntry>()) as u64),
                            }),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: if idx % 2 == 0 {
                                    &sorting_assets.entry_buffer_b
                                } else {
                                    &sorted_entries.sorted_entry_buffer
                                },
                                offset: 0,
                                size: BufferSize::new((cloud.len() * std::mem::size_of::<SortEntry>()) as u64),
                            }),
                        },
                    ],
                )
            })
            .collect::<Vec<BindGroup>>()
            .try_into()
            .unwrap();

        commands.entity(entity).insert(RadixBindGroup {
            radix_sort_bind_groups,
        });
    }
}


pub struct RadixSortNode<R: PlanarStorage> {
    gaussian_clouds: QueryState<(
        &'static R::PlanarTypeHandle,
        &'static PlanarStorageBindGroup<R>,
        &'static RadixBindGroup,
    )>,
    initialized: bool,
    view_bind_group: QueryState<(
        &'static GaussianCamera,
        &'static GaussianViewBindGroup,
        &'static ViewUniformOffset,
    )>,
}

impl<R: PlanarStorage> FromWorld for RadixSortNode<R> {
    fn from_world(world: &mut World) -> Self {
        Self {
            gaussian_clouds: world.query(),
            initialized: false,
            view_bind_group: world.query(),
        }
    }
}

impl<R: PlanarStorage> Node for RadixSortNode<R> {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<RadixSortPipeline<R>>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if !self.initialized {
            let mut pipelines_loaded = true;
            for sort_pipeline in pipeline.radix_sort_pipelines.iter() {
                if let CachedPipelineState::Ok(_) =
                        pipeline_cache.get_compute_pipeline_state(*sort_pipeline)
                {
                    continue;
                }

                pipelines_loaded = false;
            }

            self.initialized = pipelines_loaded;

            if !self.initialized {
                return;
            }
        }

        self.gaussian_clouds.update_archetypes(world);
        self.view_bind_group.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if !self.initialized {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<RadixSortPipeline<R>>();
        let gaussian_uniforms = world.resource::<GaussianUniformBindGroups>();
        let sort_buffers = world.resource::<RadixSortBuffers<R>>();

        for (
            _camera,
            view_bind_group,
            view_uniform_offset,
        ) in self.view_bind_group.iter_manual(world) {
            for (
                cloud_handle,
                cloud_bind_group,
                radix_bind_group,
            ) in self.gaussian_clouds.iter_manual(world) {
                let cloud = world
                    .get_resource::<RenderAssets<R::GpuPlanarType>>()
                    .unwrap()
                    .get(cloud_handle.handle())
                    .unwrap();

                assert!(sort_buffers.asset_map.contains_key(&cloud_handle.handle().id()));
                let sorting_assets = &sort_buffers.asset_map[&cloud_handle.handle().id()];

                {
                    let command_encoder = render_context.command_encoder();
                    let radix_digit_places = ShaderDefines::default().radix_digit_places;

                    {
                        command_encoder.clear_buffer(
                            &sorting_assets.sorting_global_buffer,
                            0,
                            None,
                        );

                        command_encoder.clear_buffer(
                            &sorting_assets.sorting_status_counter_buffer,
                            0,
                            None,
                        );

                        command_encoder.clear_buffer(
                            cloud.draw_indirect_buffer(),
                            0,
                            None,
                        );
                    }

                    {
                        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

                        pass.set_bind_group(
                            0,
                            &view_bind_group.value,
                            &[view_uniform_offset.offset],
                        );
                        pass.set_bind_group(
                            1,
                            gaussian_uniforms.base_bind_group.as_ref().unwrap(),
                            &[0],
                        );
                        pass.set_bind_group(
                            2,
                            &cloud_bind_group.bind_group,
                            &[]
                        );
                        pass.set_bind_group(
                            3,
                            &radix_bind_group.radix_sort_bind_groups[1],
                            &[],
                        );

                        let radix_sort_a = pipeline_cache.get_compute_pipeline(pipeline.radix_sort_pipelines[0]).unwrap();
                        pass.set_pipeline(radix_sort_a);

                        let workgroup_entries_a = ShaderDefines::default().workgroup_entries_a;
                        pass.dispatch_workgroups((cloud.len() as u32).div_ceil(workgroup_entries_a), 1, 1);


                        let radix_sort_b = pipeline_cache.get_compute_pipeline(pipeline.radix_sort_pipelines[1]).unwrap();
                        pass.set_pipeline(radix_sort_b);

                        pass.dispatch_workgroups(1, radix_digit_places, 1);
                    }

                    // TODO: add options to only complete a fraction of the sorting process
                    for pass_idx in 0..radix_digit_places {
                        if pass_idx > 0 {
                            command_encoder.clear_buffer(
                                &sorting_assets.sorting_status_counter_buffer,
                                0,
                                None,
                            );
                        }

                        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

                        let radix_sort_c = pipeline_cache.get_compute_pipeline(pipeline.radix_sort_pipelines[2]).unwrap();
                        pass.set_pipeline(radix_sort_c);

                        pass.set_bind_group(
                            0,
                            &view_bind_group.value,
                            &[view_uniform_offset.offset],
                        );
                        pass.set_bind_group(
                            1,
                            gaussian_uniforms.base_bind_group.as_ref().unwrap(),
                            &[0],
                        );
                        pass.set_bind_group(
                            2,
                            &cloud_bind_group.bind_group,
                            &[]
                        );
                        pass.set_bind_group(
                            3,
                            &radix_bind_group.radix_sort_bind_groups[pass_idx as usize],
                            &[],
                        );

                        let workgroup_entries_c = ShaderDefines::default().workgroup_entries_c;
                        pass.dispatch_workgroups(1, (cloud.len() as u32).div_ceil(workgroup_entries_c), 1);
                    }
                }
            }
        }

        Ok(())
    }
}
