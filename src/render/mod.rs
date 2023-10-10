use std::hash::Hash;

use bevy::{
    prelude::*,
    asset::{
        load_internal_asset,
        HandleUntyped,
    },
    core_pipeline::core_3d::Transparent3d,
    ecs::{
        system::{
            lifetimeless::*,
            SystemParamItem,
        },
        query::ROQueryItem,
    },
    reflect::TypeUuid,
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
        mesh::GpuBufferInfo,
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
        renderer::RenderDevice,
        Render,
        RenderApp,
        RenderSet,
        view::{
            ExtractedView,
            ViewUniform,
            ViewUniforms,
            ViewUniformOffset,
        },
    },
};

use crate::gaussian::{
    Gaussian,
    GaussianCloud,
    GaussianCloudSettings,
    MAX_SH_COEFF_COUNT,
};


const GAUSSIAN_SHADER_HANDLE: HandleUntyped = HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 68294581);
const SPHERICAL_HARMONICS_SHADER_HANDLE: HandleUntyped = HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 834667312);


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
    pub settings_uniform: GaussianCloudUniform,
    pub verticies: Handle<GaussianCloud>,
}

#[derive(Debug, Clone)]
pub struct GpuGaussianCloud {
    pub buffer: Buffer,
    pub count: u32,
    pub buffer_info: GpuBufferInfo,
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
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian cloud buffer"),
            contents: bytemuck::cast_slice(gaussian_cloud.0.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        });

        Ok(GpuGaussianCloud {
            buffer,
            count: gaussian_cloud.0.len() as u32,
            buffer_info: GpuBufferInfo::NonIndexed,
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
    )>,
    mut views: Query<(&ExtractedView, &mut RenderPhase<Transparent3d>)>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawGaussians>();

    for (_view, mut transparent_phase) in &mut views {
        for (entity, verticies) in &gaussian_splatting_bundles {
            if let Some(_cloud) = gaussian_clouds.get(verticies) {
                let key = GaussianCloudPipelineKey {

                };

                let pipeline = pipelines.specialize(&pipeline_cache, &custom_pipeline, key);

                transparent_phase.add(Transparent3d {
                    entity,
                    draw_function: draw_custom,
                    distance: 0.0,
                    pipeline,
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
}

impl FromWorld for GaussianCloudPipeline {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();

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
        ];

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_view_layout"),
            entries: &view_layout_entries,
        });

        let gaussian_uniform_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_uniform_layout"),
            entries: &vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
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
            entries: &vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        GaussianCloudPipeline {
            gaussian_cloud_layout,
            gaussian_uniform_layout,
            view_layout,
            shader: GAUSSIAN_SHADER_HANDLE.typed(),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct GaussianCloudPipelineKey {

}

impl SpecializedRenderPipeline for GaussianCloudPipeline {
    type Key = GaussianCloudPipelineKey;

    fn specialize(&self, _key: Self::Key) -> RenderPipelineDescriptor {
        let shader_defs = vec![
            ShaderDefVal::UInt("MAX_SH_COEFF_COUNT".into(), MAX_SH_COEFF_COUNT as u32),
        ];

        RenderPipelineDescriptor {
            label: Some("gaussian cloud pipeline".into()),
            layout: vec![
                self.view_layout.clone(),
                self.gaussian_uniform_layout.clone(),
                self.gaussian_cloud_layout.clone(),
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
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
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
    pub global_scale: f32,
    pub transform: Mat4,
}

pub fn extract_gaussians(
    mut commands: Commands,
    mut prev_commands_len: Local<usize>,
    gaussians_query: Extract<
        Query<(
            Entity,
            // &ComputedVisibility,
            &GaussianCloudSettings,
            &Handle<GaussianCloud>,
        )>,
    >,
) {
    let mut commands_list = Vec::with_capacity(*prev_commands_len);
    // let visible_gaussians = gaussians_query.iter().filter(|(_, vis, ..)| vis.is_visible());

    for (entity, settings, verticies) in gaussians_query.iter() {
        let settings_uniform = GaussianCloudUniform {
            global_scale: settings.global_scale,
            transform: settings.global_transform.compute_matrix(),
        };
        commands_list.push((
            entity,
            GpuGaussianSplattingBundle {
                settings_uniform,
                verticies: verticies.clone_weak(),
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
    pub bind_group: BindGroup,
}

pub fn queue_gaussian_bind_group(
    mut commands: Commands,
    mut groups: ResMut<GaussianUniformBindGroups>,
    gaussian_cloud_pipeline: Res<GaussianCloudPipeline>,
    render_device: Res<RenderDevice>,
    gaussian_uniforms: Res<ComponentUniforms<GaussianCloudUniform>>,
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

    for (entity, cloud_handle) in gaussian_clouds.iter() {
        let cloud = gaussian_cloud_res.get(cloud_handle).unwrap();

        groups.base_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: model,
                        offset: 0,
                        size: BufferSize::new(model.size()),
                    }),
                },
            ],
            layout: &gaussian_cloud_pipeline.gaussian_uniform_layout,
            label: Some("gaussian_uniform_bind_group"),
        }));

        commands.entity(entity).insert(GaussianCloudBindGroup {
            bind_group: render_device.create_bind_group(&BindGroupDescriptor {
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &cloud.buffer,
                            offset: 0,
                            size: BufferSize::new(cloud.buffer.size()),
                        }),
                    },
                ],
                layout: &gaussian_cloud_pipeline.gaussian_cloud_layout,
                label: Some("gaussian_cloud_bind_group"),
            }),
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

            let view_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                entries: &entries,
                label: Some("gaussian_view_bind_group"),
                layout,
            });


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
        (handle, bind_group): (&'w Handle<GaussianCloud>, &'w GaussianCloudBindGroup),
        gaussian_clouds: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_gaussian_cloud = match gaussian_clouds.into_inner().get(handle) {
            Some(gpu_gaussian_cloud) => gpu_gaussian_cloud,
            None => return RenderCommandResult::Failure,
        };

        pass.set_bind_group(2, &bind_group.bind_group, &[]);

        match &gpu_gaussian_cloud.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                pass.draw_indexed(0..*count, 0, 0..gpu_gaussian_cloud.count as u32);
            }
            GpuBufferInfo::NonIndexed => {
                pass.draw(0..4, 0..gpu_gaussian_cloud.count as u32);
            }

            // TODO: add support for indirect draw and match over sort methods
        }
        RenderCommandResult::Success
    }
}
