use std::collections::HashMap;

use bevy::{
    prelude::*,
    asset::{
        load_internal_asset,
        LoadState,
    },
    core_pipeline::core_3d::CORE_3D,
    render::{
        render_asset::RenderAssets,
        render_resource::{
            BindGroup,
            BindGroupEntry,
            BindGroupLayout,
            BindGroupLayoutDescriptor,
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
        },
        Render,
        RenderApp,
        RenderSet,
        view::ViewUniformOffset,
    },
};
use static_assertions::assert_cfg;

use crate::{
    gaussian::cloud::Cloud,
    GaussianCloudSettings,
    render::{
        GaussianCloudBindGroup,
        GaussianCloudPipeline,
        GaussianCloudPipelineKey,
        GaussianUniformBindGroups,
        GaussianViewBindGroup,
        ShaderDefines,
        shader_defs,
    },
    sort::{
        SortEntry,
        SortedEntries,
        SortMode,
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

pub mod node {
    pub const RADIX_SORT: &str = "radix_sort";
}

#[derive(Default)]
pub struct RadixSortPlugin;

impl Plugin for RadixSortPlugin {
    fn build(&self, app: &mut App) {
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

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_graph_node::<RadixSortNode>(
                    CORE_3D,
                    node::RADIX_SORT,
                )
                .add_render_graph_edge(
                    CORE_3D,
                    node::RADIX_SORT,
                     bevy::core_pipeline::core_3d::graph::node::PREPASS,
                );

            render_app
                .add_systems(
                    Render,
                    (
                        queue_radix_bind_group.in_set(RenderSet::QueueMeshes),
                    ),
                );

            render_app.init_resource::<RadixSortBuffers>();
            render_app.add_systems(ExtractSchedule, update_sort_buffers);
        }
    }

    fn finish(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<RadixSortPipeline>();
        }
    }
}

#[derive(Resource, Default)]
pub struct RadixSortBuffers {
    pub asset_map: HashMap<
        AssetId<Cloud>,
        GpuRadixBuffers,
    >,
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


fn update_sort_buffers(
    gpu_gaussian_clouds: Res<RenderAssets<Cloud>>,
    mut sort_buffers: ResMut<RadixSortBuffers>,
    render_device: Res<RenderDevice>,
) {
    for (asset_id, cloud,) in gpu_gaussian_clouds.iter() {
        // TODO: handle cloud resize operations and resolve leaked stale buffers
        if sort_buffers.asset_map.contains_key(&asset_id) {
            continue;
        }

        let gpu_radix_buffers = GpuRadixBuffers::new(cloud.count, &render_device);
        sort_buffers.asset_map.insert(asset_id, gpu_radix_buffers);
    }
}


#[derive(Resource)]
pub struct RadixSortPipeline {
    pub radix_sort_layout: BindGroupLayout,
    pub radix_sort_pipelines: [CachedComputePipelineId; 3],
}

impl FromWorld for RadixSortPipeline {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();
        let gaussian_cloud_pipeline = render_world.resource::<GaussianCloudPipeline>();

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
                min_binding_size: BufferSize::new(std::mem::size_of::<wgpu::util::DrawIndirect>() as u64),
            },
            count: None,
        };

        let radix_sort_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("radix_sort_layout"),
            entries: &[
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
        });

        let sorting_layout = vec![
            gaussian_cloud_pipeline.view_layout.clone(),
            gaussian_cloud_pipeline.gaussian_uniform_layout.clone(),
            gaussian_cloud_pipeline.gaussian_cloud_layout.clone(),
            radix_sort_layout.clone(),
        ];
        let shader_defs = shader_defs(GaussianCloudPipelineKey::default());

        let pipeline_cache = render_world.resource::<PipelineCache>();
        let radix_sort_a = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radix_sort_a".into()),
            layout: sorting_layout.clone(),
            push_constant_ranges: vec![],
            shader: RADIX_SHADER_HANDLE,
            shader_defs: shader_defs.clone(),
            entry_point: "radix_sort_a".into(),
        });

        let radix_sort_b = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radix_sort_b".into()),
            layout: sorting_layout.clone(),
            push_constant_ranges: vec![],
            shader: RADIX_SHADER_HANDLE,
            shader_defs: shader_defs.clone(),
            entry_point: "radix_sort_b".into(),
        });

        let radix_sort_c = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radix_sort_c".into()),
            layout: sorting_layout.clone(),
            push_constant_ranges: vec![],
            shader: RADIX_SHADER_HANDLE,
            shader_defs: shader_defs.clone(),
            entry_point: "radix_sort_c".into(),
        });

        RadixSortPipeline {
            radix_sort_layout,
            radix_sort_pipelines: [
                radix_sort_a,
                radix_sort_b,
                radix_sort_c,
            ],
        }
    }
}



#[derive(Component)]
pub struct RadixBindGroup {
    pub radix_sort_bind_groups: [BindGroup; 4],
}

#[allow(clippy::too_many_arguments)]
pub fn queue_radix_bind_group(
    mut commands: Commands,
    radix_pipeline: Res<RadixSortPipeline>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    gaussian_cloud_res: Res<RenderAssets<Cloud>>,
    sorted_entries_res: Res<RenderAssets<SortedEntries>>,
    gaussian_clouds: Query<(
        Entity,
        &Handle<Cloud>,
        &Handle<SortedEntries>,
        &GaussianCloudSettings,
    )>,
    sort_buffers: Res<RadixSortBuffers>,
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
        if Some(LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        if gaussian_cloud_res.get(cloud_handle).is_none() {
            continue;
        }

        if Some(LoadState::Loading) == asset_server.get_load_state(sorted_entries_handle) {
            continue;
        }

        if sorted_entries_res.get(sorted_entries_handle).is_none() {
            continue;
        }

        if !sort_buffers.asset_map.contains_key(&cloud_handle.id()) {
            continue;
        }

        let cloud = gaussian_cloud_res.get(cloud_handle).unwrap();
        let sorted_entries = sorted_entries_res.get(sorted_entries_handle).unwrap();
        let sorting_assets = &sort_buffers.asset_map[&cloud_handle.id()];

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
                buffer: &cloud.draw_indirect_buffer,
                offset: 0,
                size: BufferSize::new(cloud.draw_indirect_buffer.size()),
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
                                size: BufferSize::new((cloud.count * std::mem::size_of::<SortEntry>()) as u64),
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
                                size: BufferSize::new((cloud.count * std::mem::size_of::<SortEntry>()) as u64),
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


pub struct RadixSortNode {
    gaussian_clouds: QueryState<(
        &'static Handle<Cloud>,
        &'static GaussianCloudBindGroup,
        &'static RadixBindGroup,
    )>,
    initialized: bool,
    view_bind_group: QueryState<(
        &'static GaussianViewBindGroup,
        &'static ViewUniformOffset,
    )>,
}

impl FromWorld for RadixSortNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            gaussian_clouds: world.query(),
            initialized: false,
            view_bind_group: world.query(),
        }
    }
}

impl Node for RadixSortNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<RadixSortPipeline>();
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
        let pipeline = world.resource::<RadixSortPipeline>();
        let gaussian_uniforms = world.resource::<GaussianUniformBindGroups>();
        let sort_buffers = world.resource::<RadixSortBuffers>();

        for (
            view_bind_group,
            view_uniform_offset,
        ) in self.view_bind_group.iter_manual(world) {
            for (
                cloud_handle,
                cloud_bind_group,
                radix_bind_group,
            ) in self.gaussian_clouds.iter_manual(world) {
                let cloud = world.get_resource::<RenderAssets<Cloud>>().unwrap().get(cloud_handle).unwrap();

                assert!(sort_buffers.asset_map.contains_key(&cloud_handle.id()));
                let sorting_assets = &sort_buffers.asset_map[&cloud_handle.id()];

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
                            &cloud.draw_indirect_buffer,
                            0,
                            None,
                        );
                    }

                    // TODO: add options to only complete a fraction of the sorting process
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
                            &[0], // TODO: fix transforms - dynamic offset using DynamicUniformIndex
                        );
                        pass.set_bind_group(
                            2,
                            &cloud_bind_group.cloud_bind_group,
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
                        pass.dispatch_workgroups((cloud.count as u32 + workgroup_entries_a - 1) / workgroup_entries_a, 1, 1);


                        let radix_sort_b = pipeline_cache.get_compute_pipeline(pipeline.radix_sort_pipelines[1]).unwrap();
                        pass.set_pipeline(radix_sort_b);

                        pass.dispatch_workgroups(1, radix_digit_places, 1);
                    }

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
                            &[0], // TODO: fix transforms - dynamic offset using DynamicUniformIndex
                        );
                        pass.set_bind_group(
                            2,
                            &cloud_bind_group.cloud_bind_group,
                            &[]
                        );
                        pass.set_bind_group(
                            3,
                            &radix_bind_group.radix_sort_bind_groups[pass_idx as usize],
                            &[],
                        );

                        let workgroup_entries_c = ShaderDefines::default().workgroup_entries_c;
                        pass.dispatch_workgroups(1, (cloud.count as u32 + workgroup_entries_c - 1) / workgroup_entries_c, 1);
                    }
                }
            }
        }

        Ok(())
    }
}
