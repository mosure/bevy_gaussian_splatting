use std::hash::Hash;

use bevy::{
    prelude::*,
    asset::{
        load_internal_asset,
        LoadState,
    },
    core_pipeline::core_3d::{
        Transparent3d,
        CORE_3D,
    },
    ecs::{
        system::{
            lifetimeless::*,
            SystemParamItem,
        },
        query::ROQueryItem,
    },
    render::{
        Extract,
        extract_component::{
            DynamicUniformIndex,
            UniformComponentPlugin,
            ComponentUniforms,
        },
        globals::{
            GlobalsUniform,
            GlobalsBuffer,
        },
        render_asset::{
            PrepareAssetError,
            RenderAsset,
            RenderAssets,
            RenderAssetPlugin,
        },
        render_phase::{
            AddRenderCommand,
            DrawFunctions,
            PhaseItem,
            RenderCommand,
            RenderCommandResult,
            RenderPhase,
            SetItemPipeline,
            TrackedRenderPass,
        },
        render_resource::*,
        renderer::{
            RenderDevice,
            RenderContext,
        },
        Render,
        RenderApp,
        RenderSet,
        view::{
            ExtractedView,
            ViewUniform,
            ViewUniforms,
            ViewUniformOffset,
        },
        render_graph::{
            self,
            RenderGraphApp,
        },
    },
};

use crate::gaussian::{
    Gaussian,
    GaussianCloud,
    GaussianCloudSettings,
    MAX_SH_COEFF_COUNT,
};


const GAUSSIAN_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(68294581);
const SPHERICAL_HARMONICS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(834667312);

pub mod node {
    pub const RADIX_SORT: &str = "radix_sort";
}


#[derive(Default)]
pub struct RenderPipelinePlugin;

impl Plugin for RenderPipelinePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            GAUSSIAN_SHADER_HANDLE,
            "gaussian.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            SPHERICAL_HARMONICS_SHADER_HANDLE,
            "spherical_harmonics.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(RenderAssetPlugin::<GaussianCloud>::default());
        app.add_plugins(UniformComponentPlugin::<GaussianCloudUniform>::default());

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
                .add_render_command::<Transparent3d, DrawGaussians>()
                .init_resource::<GaussianUniformBindGroups>()
                .add_systems(ExtractSchedule, extract_gaussians)
                .add_systems(
                    Render,
                    (
                        queue_gaussian_bind_group.in_set(RenderSet::Queue),
                        queue_gaussian_view_bind_groups.in_set(RenderSet::Queue),
                        queue_gaussians.in_set(RenderSet::Queue),
                    ),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<GaussianCloudPipeline>()
                .init_resource::<SpecializedRenderPipelines<GaussianCloudPipeline>>();
        }
    }
}


#[derive(Bundle)]
pub struct GpuGaussianSplattingBundle {
    pub settings: GaussianCloudSettings,
    pub settings_uniform: GaussianCloudUniform,
    pub verticies: Handle<GaussianCloud>,
}

#[derive(Debug, Clone)]
pub struct GpuGaussianCloud {
    pub gaussian_buffer: Buffer,
    pub count: u32,

    pub draw_indirect_buffer: Buffer,
    pub sorting_global_buffer: Buffer,
    pub sorting_pass_buffers: [Buffer; 4],
    pub entry_buffer_a: Buffer,
    pub entry_buffer_b: Buffer,
}
impl RenderAsset for GaussianCloud {
    type ExtractedAsset = GaussianCloud;
    type PreparedAsset = GpuGaussianCloud;
    type Param = SRes<RenderDevice>;

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        gaussian_cloud: Self::ExtractedAsset,
        render_device: &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let gaussian_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian cloud buffer"),
            contents: bytemuck::cast_slice(gaussian_cloud.0.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        });

        let count = gaussian_cloud.0.len() as u32;

        // TODO: derive sorting_buffer_size from cloud count (with possible rounding to next power of 2)
        let sorting_global_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("sorting global buffer"),
            size: ShaderDefines::default().sorting_buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let draw_indirect_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("draw indirect buffer"),
            size: std::mem::size_of::<wgpu::util::DrawIndirect>() as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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

        let entry_buffer_a = render_device.create_buffer(&BufferDescriptor {
            label: Some("entry buffer a"),
            size: (count as usize * std::mem::size_of::<(u32, u32)>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let entry_buffer_b = render_device.create_buffer(&BufferDescriptor {
            label: Some("entry buffer b"),
            size: (count as usize * std::mem::size_of::<(u32, u32)>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok(GpuGaussianCloud {
            gaussian_buffer,
            count,
            draw_indirect_buffer,
            sorting_global_buffer,
            sorting_pass_buffers,
            entry_buffer_a,
            entry_buffer_b,
        })
    }
}


#[allow(clippy::too_many_arguments)]
fn queue_gaussians(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<GaussianCloudPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<GaussianCloudPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    gaussian_clouds: Res<RenderAssets<GaussianCloud>>,
    gaussian_splatting_bundles: Query<(
        Entity,
        &Handle<GaussianCloud>,
        &GaussianCloudSettings,
    )>,
    mut views: Query<(
        &ExtractedView,
        &mut RenderPhase<Transparent3d>,
    )>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawGaussians>();

    for (_view, mut transparent_phase) in &mut views {
        for (entity, cloud, settings) in &gaussian_splatting_bundles {
            if let Some(_cloud) = gaussian_clouds.get(cloud) {
                let key = GaussianCloudPipelineKey {
                    aabb: settings.aabb,
                    visualize_bounding_box: settings.visualize_bounding_box,
                };

                let pipeline = pipelines.specialize(&pipeline_cache, &custom_pipeline, key);

                transparent_phase.add(Transparent3d {
                    entity,
                    draw_function: draw_custom,
                    distance: 0.0,
                    pipeline,
                    batch_range: 0..1,
                    dynamic_offset: None,
                });
            }
        }
    }
}




#[derive(Resource)]
pub struct GaussianCloudPipeline {
    shader: Handle<Shader>,
    pub gaussian_cloud_layout: BindGroupLayout,
    pub gaussian_uniform_layout: BindGroupLayout,
    pub view_layout: BindGroupLayout,
    pub radix_sort_layout: BindGroupLayout,
    pub radix_sort_pipelines: [CachedComputePipelineId; 3],
    pub temporal_sort_pipelines: [CachedComputePipelineId; 2],
    pub sorted_layout: BindGroupLayout,
}

impl FromWorld for GaussianCloudPipeline {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();

        let view_layout_entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(GlobalsUniform::min_size()),
                },
                count: None,
            },
        ];

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_view_layout"),
            entries: &view_layout_entries,
        });

        let gaussian_uniform_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_uniform_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(GaussianCloudUniform::min_size()),
                    },
                    count: None,
                },
            ],
        });

        let gaussian_cloud_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_cloud_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(std::mem::size_of::<Gaussian>() as u64),
                    },
                    count: None,
                },
            ],
        });

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

        let draw_indirect_buffer_entry = BindGroupLayoutEntry {
            binding: 2,
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
                draw_indirect_buffer_entry,
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(std::mem::size_of::<(u32, u32)>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(std::mem::size_of::<(u32, u32)>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let sorted_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sorted_layout"),
            entries: &vec![
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(std::mem::size_of::<(u32, u32)>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let compute_layout = vec![
            view_layout.clone(),
            gaussian_uniform_layout.clone(),
            gaussian_cloud_layout.clone(),
            radix_sort_layout.clone(),
        ];
        let shader = GAUSSIAN_SHADER_HANDLE;
        let shader_defs = shader_defs(false, false);

        let pipeline_cache = render_world.resource::<PipelineCache>();
        let radix_sort_a = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radix_sort_a".into()),
            layout: compute_layout.clone(),
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: shader_defs.clone(),
            entry_point: "radix_sort_a".into(),
        });

        let radix_sort_b = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radix_sort_b".into()),
            layout: compute_layout.clone(),
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: shader_defs.clone(),
            entry_point: "radix_sort_b".into(),
        });

        let radix_sort_c = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radix_sort_c".into()),
            layout: compute_layout.clone(),
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: shader_defs.clone(),
            entry_point: "radix_sort_c".into(),
        });


        let temporal_sort_flip = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("temporal_sort_flip".into()),
            layout: compute_layout.clone(),
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: shader_defs.clone(),
            entry_point: "temporal_sort_flip".into(),
        });

        let temporal_sort_flop = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("temporal_sort_flop".into()),
            layout: compute_layout.clone(),
            push_constant_ranges: vec![],
            shader: shader.clone(),
            shader_defs: shader_defs.clone(),
            entry_point: "temporal_sort_flop".into(),
        });

        GaussianCloudPipeline {
            gaussian_cloud_layout,
            gaussian_uniform_layout,
            view_layout,
            shader: shader.clone(),
            radix_sort_layout,
            radix_sort_pipelines: [
                radix_sort_a,
                radix_sort_b,
                radix_sort_c,
            ],
            temporal_sort_pipelines: [
                temporal_sort_flip,
                temporal_sort_flop,
            ],
            sorted_layout,
        }
    }
}

// TODO: allow setting shader defines via API
struct ShaderDefines {
    radix_bits_per_digit: u32,
    radix_digit_places: u32,
    radix_base: u32,
    entries_per_invocation_a: u32,
    entries_per_invocation_c: u32,
    workgroup_invocations_a: u32,
    workgroup_invocations_c: u32,
    workgroup_entries_a: u32,
    workgroup_entries_c: u32,
    max_tile_count_c: u32,
    sorting_buffer_size: usize,

    temporal_sort_window_size: u32,
}

impl Default for ShaderDefines {
    fn default() -> Self {
        let radix_bits_per_digit = 8;
        let radix_digit_places = 32 / radix_bits_per_digit;
        let radix_base = 1 << radix_bits_per_digit;
        let entries_per_invocation_a = 4;
        let entries_per_invocation_c = 4;
        let workgroup_invocations_a = radix_base * radix_digit_places;
        let workgroup_invocations_c = radix_base;
        let workgroup_entries_a = workgroup_invocations_a * entries_per_invocation_a;
        let workgroup_entries_c = workgroup_invocations_c * entries_per_invocation_c;
        let max_tile_count_c = (10000000 + workgroup_entries_c - 1) / workgroup_entries_c;
        let sorting_buffer_size = (
            radix_base as usize *
            (radix_digit_places as usize + max_tile_count_c as usize) *
            std::mem::size_of::<u32>()
        ) + std::mem::size_of::<u32>() * 5;

        Self {
            radix_bits_per_digit,
            radix_digit_places,
            radix_base,
            entries_per_invocation_a,
            entries_per_invocation_c,
            workgroup_invocations_a,
            workgroup_invocations_c,
            workgroup_entries_a,
            workgroup_entries_c,
            max_tile_count_c,
            sorting_buffer_size,

            temporal_sort_window_size: 16,
        }
    }
}

fn shader_defs(
    aabb: bool,
    visualize_bounding_box: bool,
) -> Vec<ShaderDefVal> {
    let defines = ShaderDefines::default();
    let mut shader_defs = vec![
        ShaderDefVal::UInt("MAX_SH_COEFF_COUNT".into(), MAX_SH_COEFF_COUNT as u32),
        ShaderDefVal::UInt("RADIX_BASE".into(), defines.radix_base),
        ShaderDefVal::UInt("RADIX_BITS_PER_DIGIT".into(), defines.radix_bits_per_digit),
        ShaderDefVal::UInt("RADIX_DIGIT_PLACES".into(), defines.radix_digit_places),
        ShaderDefVal::UInt("ENTRIES_PER_INVOCATION_A".into(), defines.entries_per_invocation_a),
        ShaderDefVal::UInt("ENTRIES_PER_INVOCATION_C".into(), defines.entries_per_invocation_c),
        ShaderDefVal::UInt("WORKGROUP_INVOCATIONS_A".into(), defines.workgroup_invocations_a),
        ShaderDefVal::UInt("WORKGROUP_INVOCATIONS_C".into(), defines.workgroup_invocations_c),
        ShaderDefVal::UInt("WORKGROUP_ENTRIES_C".into(), defines.workgroup_entries_c),
        ShaderDefVal::UInt("MAX_TILE_COUNT_C".into(), defines.max_tile_count_c),

        ShaderDefVal::UInt("TEMPORAL_SORT_WINDOW_SIZE".into(), defines.temporal_sort_window_size),
    ];

    if aabb {
        shader_defs.push("USE_AABB".into());
    }

    if !aabb {
        shader_defs.push("USE_OBB".into());
    }

    if visualize_bounding_box {
        shader_defs.push("VISUALIZE_BOUNDING_BOX".into());
    }

    shader_defs
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct GaussianCloudPipelineKey {
    pub aabb: bool,
    pub visualize_bounding_box: bool,
}

impl SpecializedRenderPipeline for GaussianCloudPipeline {
    type Key = GaussianCloudPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let shader_defs = shader_defs(
            key.aabb,
            key.visualize_bounding_box,
        );

        RenderPipelineDescriptor {
            label: Some("gaussian cloud render pipeline".into()),
            layout: vec![
                self.view_layout.clone(),
                self.gaussian_uniform_layout.clone(),
                self.gaussian_cloud_layout.clone(),
                self.sorted_layout.clone(),
            ],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                entry_point: "vs_points".into(),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: "fs_main".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::DstAlpha,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
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
                depth_write_enabled: false,
                depth_compare: CompareFunction::GreaterEqual,
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
                count: 4,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            push_constant_ranges: Vec::new(),
        }
    }
}

type DrawGaussians = (
    SetItemPipeline,
    SetGaussianViewBindGroup<0>,
    SetGaussianUniformBindGroup<1>,
    DrawGaussianInstanced,
);


#[derive(Component, ShaderType, Clone)]
pub struct GaussianCloudUniform {
    pub transform: Mat4,
    pub global_scale: f32,
}

pub fn extract_gaussians(
    mut commands: Commands,
    mut prev_commands_len: Local<usize>,
    gaussians_query: Extract<
        Query<(
            Entity,
            // &ComputedVisibility,
            &Handle<GaussianCloud>,
            &GaussianCloudSettings,
        )>,
    >,
) {
    let mut commands_list = Vec::with_capacity(*prev_commands_len);
    // let visible_gaussians = gaussians_query.iter().filter(|(_, vis, ..)| vis.is_visible());

    for (entity, verticies, settings) in gaussians_query.iter() {
        let settings_uniform = GaussianCloudUniform {
            transform: settings.global_transform.compute_matrix(),
            global_scale: settings.global_scale,
        };
        commands_list.push((
            entity,
            GpuGaussianSplattingBundle {
                settings: settings.clone(),
                settings_uniform,
                verticies: verticies.clone(),
            },
        ));
    }
    *prev_commands_len = commands_list.len();
    commands.insert_or_spawn_batch(commands_list);
}


#[derive(Resource, Default)]
pub struct GaussianUniformBindGroups {
    base_bind_group: Option<BindGroup>,
}

#[derive(Component)]
pub struct GaussianCloudBindGroup {
    pub cloud_bind_group: BindGroup,
    pub radix_sort_bind_groups: [BindGroup; 4],
    pub sorted_bind_group: BindGroup,
}

pub fn queue_gaussian_bind_group(
    mut commands: Commands,
    mut groups: ResMut<GaussianUniformBindGroups>,
    gaussian_cloud_pipeline: Res<GaussianCloudPipeline>,
    render_device: Res<RenderDevice>,
    gaussian_uniforms: Res<ComponentUniforms<GaussianCloudUniform>>,
    asset_server: Res<AssetServer>,
    gaussian_cloud_res: Res<RenderAssets<GaussianCloud>>,
    gaussian_clouds: Query<(
        Entity,
        &Handle<GaussianCloud>,
    )>,
) {
    let Some(model) = gaussian_uniforms.buffer() else {
        return;
    };

    assert!(model.size() == std::mem::size_of::<GaussianCloudUniform>() as u64);

    groups.base_bind_group = Some(render_device.create_bind_group(
        "gaussian_uniform_bind_group",
        &gaussian_cloud_pipeline.gaussian_uniform_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: model,
                    offset: 0,
                    size: BufferSize::new(model.size()),
                }),
            },
        ],
    ));

    for (entity, cloud_handle) in gaussian_clouds.iter() {
        if Some(LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        if gaussian_cloud_res.get(cloud_handle).is_none() {
            continue;
        }

        let cloud = gaussian_cloud_res.get(cloud_handle).unwrap();

        let sorting_global_entry = BindGroupEntry {
            binding: 1,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &cloud.sorting_global_buffer,
                offset: 0,
                size: BufferSize::new(cloud.sorting_global_buffer.size()),
            }),
        };

        let draw_indirect_entry = BindGroupEntry {
            binding: 2,
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
                    &gaussian_cloud_pipeline.radix_sort_layout,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: &cloud.sorting_pass_buffers[idx],
                                offset: 0,
                                size: BufferSize::new(std::mem::size_of::<u32>() as u64),
                            }),
                        },
                        sorting_global_entry.clone(),
                        draw_indirect_entry.clone(),
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: if idx % 2 == 0 {
                                    &cloud.entry_buffer_a
                                } else {
                                    &cloud.entry_buffer_b
                                },
                                offset: 0,
                                size: BufferSize::new((cloud.count as usize * std::mem::size_of::<(u32, u32)>()) as u64),
                            }),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: if idx % 2 == 0 {
                                    &cloud.entry_buffer_b
                                } else {
                                    &cloud.entry_buffer_a
                                },
                                offset: 0,
                                size: BufferSize::new((cloud.count as usize * std::mem::size_of::<(u32, u32)>()) as u64),
                            }),
                        },
                    ],
                )
            })
            .collect::<Vec<BindGroup>>()
            .try_into()
            .unwrap();

        commands.entity(entity).insert(GaussianCloudBindGroup {
            cloud_bind_group: render_device.create_bind_group(
                "gaussian_cloud_bind_group",
                &gaussian_cloud_pipeline.gaussian_cloud_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &cloud.gaussian_buffer,
                            offset: 0,
                            size: BufferSize::new(cloud.gaussian_buffer.size()),
                        }),
                    },
                ],
            ),
            radix_sort_bind_groups,
            sorted_bind_group: render_device.create_bind_group(
                "render_sorted_bind_group",
                &gaussian_cloud_pipeline.sorted_layout,
                &[
                    BindGroupEntry {
                        binding: 5,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &cloud.entry_buffer_a,
                            offset: 0,
                            size: BufferSize::new((cloud.count as usize * std::mem::size_of::<(u32, u32)>()) as u64),
                        }),
                    },
                ],
            ),
        });
    }
}


#[derive(Component)]
pub struct GaussianViewBindGroup {
    pub value: BindGroup,
}

pub fn queue_gaussian_view_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    gaussian_cloud_pipeline: Res<GaussianCloudPipeline>,
    view_uniforms: Res<ViewUniforms>,
    views: Query<(
        Entity,
        &ExtractedView,
        &mut RenderPhase<Transparent3d>,
    )>,
    globals_buffer: Res<GlobalsBuffer>,
) {
    if let (
        Some(view_binding),
        Some(globals),
    ) = (
        view_uniforms.uniforms.binding(),
        globals_buffer.buffer.binding(),
    ) {
        for (
            entity,
            _extracted_view,
            _render_phase,
        ) in &views
        {
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
            ];

            let view_bind_group = render_device.create_bind_group(
                "gaussian_view_bind_group",
                layout,
                &entries,
            );


            commands.entity(entity).insert(GaussianViewBindGroup {
                value: view_bind_group,
            });
        }
    }
}

pub struct SetGaussianViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetGaussianViewBindGroup<I> {
    type Param = ();
    type ViewWorldQuery = (
        Read<ViewUniformOffset>,
        Read<GaussianViewBindGroup>,
    );
    type ItemWorldQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        (view_uniform, gaussian_view_bind_group): ROQueryItem<
            'w,
            Self::ViewWorldQuery,
        >,
        _entity: (),
        _: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(
            I,
            &gaussian_view_bind_group.value,
            &[view_uniform.offset],
        );

        RenderCommandResult::Success
    }
}


pub struct SetGaussianUniformBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetGaussianUniformBindGroup<I> {
    type Param = SRes<GaussianUniformBindGroups>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<DynamicUniformIndex<GaussianCloudUniform>>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        gaussian_cloud_index: ROQueryItem<Self::ItemWorldQuery>,
        bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let bind_groups = bind_groups.into_inner();
        let bind_group = bind_groups.base_bind_group.as_ref().expect("bind group not initialized");

        let mut set_bind_group = |indices: &[u32]| pass.set_bind_group(I, bind_group, indices);
        let gaussian_cloud_index = gaussian_cloud_index.index();
        set_bind_group(&[gaussian_cloud_index]);

        RenderCommandResult::Success
    }
}

pub struct DrawGaussianInstanced;
impl<P: PhaseItem> RenderCommand<P> for DrawGaussianInstanced {
    type Param = SRes<RenderAssets<GaussianCloud>>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = (
        Read<Handle<GaussianCloud>>,
        Read<GaussianCloudBindGroup>,
    );

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        (
            handle,
            bind_groups,
        ): (
            &'w Handle<GaussianCloud>,
            &'w GaussianCloudBindGroup,
        ),
        gaussian_clouds: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_gaussian_cloud = match gaussian_clouds.into_inner().get(handle) {
            Some(gpu_gaussian_cloud) => gpu_gaussian_cloud,
            None => return RenderCommandResult::Failure,
        };

        pass.set_bind_group(2, &bind_groups.cloud_bind_group, &[]);
        pass.set_bind_group(3, &bind_groups.sorted_bind_group, &[]);

        pass.draw_indirect(&gpu_gaussian_cloud.draw_indirect_buffer, 0);

        RenderCommandResult::Success
    }
}





struct RadixSortNode {
    gaussian_clouds: QueryState<(
        &'static Handle<GaussianCloud>,
        &'static GaussianCloudBindGroup
    )>,
    initialized: bool,
    pipeline_idx: Option<u32>,
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
            pipeline_idx: None,
            view_bind_group: world.query(),
        }
    }
}

impl render_graph::Node for RadixSortNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GaussianCloudPipeline>();
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

        if self.pipeline_idx.is_none() {
            self.pipeline_idx = Some(0);
        } else {
            self.pipeline_idx = Some((self.pipeline_idx.unwrap() + 1) % pipeline.radix_sort_pipelines.len() as u32);
        }

        self.gaussian_clouds.update_archetypes(world);
        self.view_bind_group.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if !self.initialized || self.pipeline_idx.is_none() {
            return Ok(());
        }

        let _idx = self.pipeline_idx.unwrap() as usize; // TODO: temporal sort

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GaussianCloudPipeline>();
        let gaussian_uniforms = world.resource::<GaussianUniformBindGroups>();

        let command_encoder = render_context.command_encoder();

        for (
            view_bind_group,
            view_uniform_offset,
        ) in self.view_bind_group.iter_manual(world) {
            for (
                cloud_handle,
                cloud_bind_group
            ) in self.gaussian_clouds.iter_manual(world) {
                let cloud = world.get_resource::<RenderAssets<GaussianCloud>>().unwrap().get(cloud_handle).unwrap();

                let radix_digit_places = ShaderDefines::default().radix_digit_places;

                command_encoder.clear_buffer(
                    &cloud.sorting_global_buffer,
                    0,
                    None,
                );

                {
                    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

                    // TODO: view/global
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
                        &cloud_bind_group.radix_sort_bind_groups[1],
                        &[],
                    );

                    let radix_sort_a = pipeline_cache.get_compute_pipeline(pipeline.radix_sort_pipelines[0]).unwrap();
                    pass.set_pipeline(radix_sort_a);

                    let workgroup_entries_a = ShaderDefines::default().workgroup_entries_a;
                    pass.dispatch_workgroups((cloud.count + workgroup_entries_a - 1) / workgroup_entries_a, 1, 1);


                    let radix_sort_b = pipeline_cache.get_compute_pipeline(pipeline.radix_sort_pipelines[1]).unwrap();
                    pass.set_pipeline(radix_sort_b);

                    pass.dispatch_workgroups(1, radix_digit_places, 1);
                }

                for pass_idx in 0..radix_digit_places {
                    if pass_idx > 0 {
                        let size = ShaderDefines::default().radix_base * ShaderDefines::default().max_tile_count_c * std::mem::size_of::<u32>() as u32;
                        command_encoder.clear_buffer(
                            &cloud.sorting_global_buffer,
                            0,
                            std::num::NonZeroU64::new(size as u64).unwrap().into()
                        );
                    }

                    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

                    let radix_sort_c = pipeline_cache.get_compute_pipeline(pipeline.radix_sort_pipelines[2]).unwrap();
                    pass.set_pipeline(&radix_sort_c);

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
                        &cloud_bind_group.radix_sort_bind_groups[pass_idx as usize],
                        &[],
                    );

                    let workgroup_entries_c = ShaderDefines::default().workgroup_entries_c;
                    pass.dispatch_workgroups(1, (cloud.count + workgroup_entries_c - 1) / workgroup_entries_c, 1);
                }
            }
        }


        Ok(())
    }
}
