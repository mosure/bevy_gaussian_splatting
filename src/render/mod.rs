use std::{hash::Hash, num::NonZeroU64};

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
        query::{
            QueryItem,
            ROQueryItem,
        },
    },
    reflect::TypeUuid,
    render::{
        Extract,
        extract_component::{
            DynamicUniformIndex,
            UniformComponentPlugin,
            ComponentUniforms,
            ExtractComponent,
            ExtractComponentPlugin,
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

use crate::{GaussianSplattingBundle, gaussian::GaussianCloudSettings};
use crate::gaussian::{
    Gaussian,
    GaussianCloud,
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

        // TODO: either use extract_gaussians OR ExtractComponentPlugin (extract_gaussians allows for earlier visibility culling)
        app.add_plugins(ExtractComponentPlugin::<GaussianSplattingBundle>::default());

        // TODO(future): pre-pass filter using output from core 3d render pipeline

        // TODO: gaussian splatting render pipeline
        // TODO: add a gaussian splatting render pass
        // TODO: add a gaussian splatting camera component
        // TODO: add a gaussian cloud sorting system

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Transparent3d, DrawGaussians>()
                .init_resource::<GaussianBindGroups>()
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


#[derive(Component, Clone)]
pub struct GpuGaussianSplattingBundle {
    settings_uniform: GaussianCloudUniform,
    verticies: Handle<GaussianCloud>,
}

impl ExtractComponent for GaussianSplattingBundle {
    type Query = &'static GaussianSplattingBundle;
    type Filter = ();
    type Out = GpuGaussianSplattingBundle;

    fn extract_component(item: QueryItem<'_, Self::Query>) -> Option<GpuGaussianSplattingBundle> {
        Some(GpuGaussianSplattingBundle {
            settings_uniform: GaussianCloudUniform {
                global_scale: item.settings.global_scale,
                transform: item.settings.global_transform.compute_matrix(),
            },
            verticies: item.verticies.clone(),
        })
    }
}



// TODO: use point mesh pipeline instead of custom pipeline?
#[derive(Debug, Clone)]
pub struct GpuGaussianCloud {
    pub vertex_buffer: Buffer, //TODO: add this buffer to group 1 layout (and move gaussian uniforms to group 0, binding 2)
    pub vertex_count: u32,
    pub buffer_info: GpuBufferInfo,
}
impl RenderAsset for GaussianCloud {
    type ExtractedAsset = GaussianCloud;
    type PreparedAsset = GpuGaussianCloud;
    type Param = SRes<RenderDevice>;

    /// clones the gaussian cloud
    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    /// converts the extracted gaussian cloud a into [`GpuGaussianCloud`].
    fn prepare_asset(
        gaussian_cloud: Self::ExtractedAsset,
        render_device: &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let vertex_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("gaussian cloud vertex buffer"),
            contents: bytemuck::cast_slice(gaussian_cloud.0.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        Ok(GpuGaussianCloud {
            vertex_buffer,
            vertex_count: gaussian_cloud.0.len() as u32,
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
    gaussian_splatting_bundles: Query<(Entity, &GpuGaussianSplattingBundle)>,
    mut views: Query<(&ExtractedView, &mut RenderPhase<Transparent3d>)>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawGaussians>();

    for (_view, mut transparent_phase) in &mut views {
        for (entity, bundle) in &gaussian_splatting_bundles {
            if let Some(_cloud) = gaussian_clouds.get(&bundle.verticies) {
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
    pub gaussian_layout: BindGroupLayout,
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
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(GaussianCloudUniform::min_size()),
                },
                count: None,
            }
        ];

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_view_layout"),
            entries: &view_layout_entries,
        });


        let gaussian_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_layout"),
            entries: &vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<Gaussian>() as u64)
                    },
                    count: None,
                },
            ],
        });

        GaussianCloudPipeline {
            gaussian_layout,
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
            "MESH_BINDGROUP_1".into(),
            ShaderDefVal::UInt("MAX_SH_COEFF_COUNT".into(), MAX_SH_COEFF_COUNT as u32),
        ];

        RenderPipelineDescriptor {
            label: Some("gaussian cloud pipeline".into()),
            layout: vec![
                self.view_layout.clone(),
                self.gaussian_layout.clone(),
            ],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                entry_point: "vs_points".into(),
                buffers: vec![
                    VertexBufferLayout {
                        array_stride: std::mem::size_of::<Gaussian>() as u64,
                        step_mode: VertexStepMode::Instance,
                        attributes: vec![
                            // position
                            VertexAttribute {
                                format: VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            // log_scale
                            VertexAttribute {
                                format: VertexFormat::Float32x3,
                                offset: VertexFormat::Float32x3.size(),
                                shader_location: 1,
                            },
                            // rotation
                            VertexAttribute {
                                format: VertexFormat::Float32x4,
                                offset: 2 * VertexFormat::Float32x3.size(),
                                shader_location: 2,
                            },
                            // opacity
                            VertexAttribute {
                                format: VertexFormat::Float32,
                                offset: 2 * VertexFormat::Float32x3.size() + VertexFormat::Float32x4.size(),
                                shader_location: 3,
                            },
                            // spherical_harmonic array...
                        ],
                    }
                ],
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
            multisample: MultisampleState::default(),
            push_constant_ranges: Vec::new(),
        }
    }
}

type DrawGaussians = (
    SetItemPipeline,
    // TODO: convert to gaussian bind group, use native globals and view uniforms: https://github.com/bevyengine/bevy/blob/0d23d71c19c784ceb1acfbb134dda9ce0c2adc61/crates/bevy_render/src/view/view.wgsl#L10
    //          also see: https://github.com/bevyengine/bevy/blob/0d23d71c19c784ceb1acfbb134dda9ce0c2adc61/crates/bevy_pbr/src/render/mesh.rs#L1006
    SetGaussianViewBindGroup<0>,
    SetGaussianBindGroup<1>,
    DrawGaussianInstanced,
);


#[derive(Component, ShaderType, Clone)]
pub struct GaussianCloudUniform {
    pub global_scale: f32,
    pub transform: Mat4,
}

// TODO: this is redundant with ExtractComponent for GaussianSplattingBundle
pub fn extract_gaussians(
    mut commands: Commands,
    mut prev_commands_len: Local<usize>,
    gaussians_query: Extract<
        Query<(
            Entity,
            &ComputedVisibility,
            &GaussianCloudSettings,
            &Handle<GaussianCloud>,
        )>,
    >,
) {
    let mut commands_list = Vec::with_capacity(*prev_commands_len);
    let visible_gaussians = gaussians_query.iter().filter(|(_, vis, ..)| vis.is_visible());

    for (entity, _, settings, handle) in
        visible_gaussians
    {
        let uniform = GaussianCloudUniform {
            global_scale: settings.global_scale,
            transform: settings.global_transform.compute_matrix(),
        };
        commands_list.push((entity, (handle.clone_weak(), uniform)));
    }
    *prev_commands_len = commands_list.len();
    commands.insert_or_spawn_batch(commands_list);
}


#[derive(Resource, Default)]
pub struct GaussianBindGroups {
    base_bind_group: Option<BindGroup>,
}

pub fn queue_gaussian_bind_group(
    mut groups: ResMut<GaussianBindGroups>,
    gaussian_cloud_pipeline: Res<GaussianCloudPipeline>,
    render_device: Res<RenderDevice>,
    gaussian_uniforms: Res<ComponentUniforms<GaussianCloudUniform>>,
) {
    let layout: &BindGroupLayout = &gaussian_cloud_pipeline.gaussian_layout;
    let Some(model) = gaussian_uniforms.buffer() else {
        return;
    };

    groups.base_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: model,
                    offset: 0,
                    size: Some(BufferSize::new(GaussianCloudUniform::min_size().get()).unwrap()),
                }),
            }
        ],
        layout,
        label: Some("gaussian_bind_group"),
    }));
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
    gaussian_cloud_uniforms: Res<ComponentUniforms<GaussianCloudUniform>>,
) {
    if let (
        Some(view_binding),
        Some(globals),
        Some(gaussian_cloud_uniform),
    ) = (
        view_uniforms.uniforms.binding(),
        globals_buffer.buffer.binding(),
        gaussian_cloud_uniforms.binding(),
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
                BindGroupEntry {
                    binding: 2,
                    resource: gaussian_cloud_uniform.clone(),
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


pub struct SetGaussianBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetGaussianBindGroup<I> {
    type Param = SRes<GaussianBindGroups>;
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
    type ItemWorldQuery = Read<Handle<GaussianCloud>>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        gaussian_cloud_handle: &'w Handle<GaussianCloud>,
        gaussian_clouds: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_gaussian_cloud = match gaussian_clouds.into_inner().get(gaussian_cloud_handle) {
            Some(gpu_gaussian_cloud) => gpu_gaussian_cloud,
            None => return RenderCommandResult::Failure,
        };

        pass.set_vertex_buffer(0, gpu_gaussian_cloud.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, gpu_gaussian_cloud.vertex_buffer.slice(..));

        match &gpu_gaussian_cloud.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                pass.draw_indexed(0..*count, 0, 0..gpu_gaussian_cloud.vertex_count as u32);
            }
            GpuBufferInfo::NonIndexed => {
                pass.draw(0..4, 0..gpu_gaussian_cloud.vertex_count as u32);
            }

            // TODO: add support for indirect draw and match over sort methods
        }
        RenderCommandResult::Success
    }
}
