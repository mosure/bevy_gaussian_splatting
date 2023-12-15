use bevy::{
    prelude::*,
    asset::{
        load_internal_asset,
        LoadState,
    },
    core_pipeline::core_3d::CORE_3D,
    ecs::system::{
        lifetimeless::SRes,
        SystemParamItem,
    },
    render::{
        Extract,
        render_asset::{
            PrepareAssetError,
            RenderAsset,
            RenderAssets,
            RenderAssetPlugin,
        },
        render_resource::*,
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

use crate::render::{
    GaussianCloudBindGroup,
    GaussianCloudPipeline,
    GaussianUniformBindGroups,
    GaussianViewBindGroup,
    shader_defs,
};

pub use particle::{
    ParticleBehavior,
    ParticleBehaviors,
    random_particle_behaviors,
};

pub mod particle;


const PARTICLE_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(234553453455);
pub mod node {
    pub const MORPH: &str = "gaussian_cloud_morph";
}


#[derive(Default)]
pub struct MorphPlugin;

impl Plugin for MorphPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PARTICLE_SHADER_HANDLE,
            "particle.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<ParticleBehaviors>();
        app.init_asset::<ParticleBehaviors>();
        app.register_asset_reflect::<ParticleBehaviors>();
        app.add_plugins(RenderAssetPlugin::<ParticleBehaviors>::default());

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_graph_node::<MorphNode>(
                    CORE_3D,
                    node::MORPH,
                )
                .add_render_graph_edge(
                    CORE_3D,
                    node::MORPH,
                    bevy::core_pipeline::core_3d::graph::node::PREPASS,
                );

            render_app
                .add_systems(ExtractSchedule, extract_particle_behaviors)
                .add_systems(
                    Render,
                    (
                        queue_morph_bind_group.in_set(RenderSet::QueueMeshes),
                    ),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<MorphPipeline>();
        }
    }
}


pub fn extract_particle_behaviors(
    mut commands: Commands,
    mut prev_commands_len: Local<usize>,
    gaussians_query: Extract<
        Query<(
            Entity,
            &Handle<ParticleBehaviors>,
        )>,
    >,
) {
    let mut commands_list = Vec::with_capacity(*prev_commands_len);

    for (entity, behaviors) in gaussians_query.iter() {
        commands_list.push((
            entity,
            behaviors.clone(),
        ));
    }
    *prev_commands_len = commands_list.len();
    commands.insert_or_spawn_batch(commands_list);
}


#[derive(Debug, Clone)]
pub struct GpuMorphBuffers {
    pub morph_count: u32,
    pub particle_behavior_buffer: Buffer,
}

impl RenderAsset for ParticleBehaviors {
    type ExtractedAsset = ParticleBehaviors;
    type PreparedAsset = GpuMorphBuffers;
    type Param = SRes<RenderDevice>;

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        particle_behaviors: Self::ExtractedAsset,
        render_device: &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let morph_count = particle_behaviors.0.len() as u32;

        let particle_behavior_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("particle behavior buffer"),
            contents: bytemuck::cast_slice(
                particle_behaviors.0.as_slice()
            ),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        });

        Ok(GpuMorphBuffers {
            morph_count,
            particle_behavior_buffer,
        })
    }
}


#[derive(Resource)]
pub struct MorphPipeline {
    pub morph_layout: BindGroupLayout,
    pub particle_behavior_pipeline: CachedComputePipelineId,
}

impl FromWorld for MorphPipeline {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();
        let gaussian_cloud_pipeline = render_world.resource::<GaussianCloudPipeline>();

        let morph_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gaussian_cloud_morph_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(std::mem::size_of::<ParticleBehavior>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let shader_defs = shader_defs(false, false);
        let pipeline_cache = render_world.resource::<PipelineCache>();

        let particle_behavior_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("particle_behavior_pipeline".into()),
            layout: vec![
                gaussian_cloud_pipeline.view_layout.clone(),
                gaussian_cloud_pipeline.gaussian_uniform_layout.clone(),
                gaussian_cloud_pipeline.gaussian_cloud_layout.clone(),
                morph_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: PARTICLE_SHADER_HANDLE,
            shader_defs: shader_defs.clone(),
            entry_point: "apply_particle_behaviors".into(),
        });

        MorphPipeline {
            morph_layout,
            particle_behavior_pipeline,
        }
    }
}



#[derive(Component)]
pub struct MorphBindGroup {
    pub morph_bindgroup: BindGroup,
}

pub fn queue_morph_bind_group(
    mut commands: Commands,
    morph_pipeline: Res<MorphPipeline>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    particle_behaviors_res: Res<RenderAssets<ParticleBehaviors>>,
    particle_behaviors: Query<(
        Entity,
        &Handle<ParticleBehaviors>,
    )>,
) {
    for (entity, behaviors_handle) in particle_behaviors.iter() {
        if Some(LoadState::Loading) == asset_server.get_load_state(behaviors_handle) {
            continue;
        }

        if particle_behaviors_res.get(behaviors_handle).is_none() {
            continue;
        }

        let behaviors = particle_behaviors_res.get(behaviors_handle).unwrap();

        let morph_bindgroup = render_device.create_bind_group(
            "morph_bind_group",
            &morph_pipeline.morph_layout,
            &[
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &behaviors.particle_behavior_buffer,
                        offset: 0,
                        size: BufferSize::new(behaviors.particle_behavior_buffer.size()),
                    }),
                },
            ],
        );

        commands.entity(entity).insert(MorphBindGroup {
            morph_bindgroup,
        });
    }
}



pub struct MorphNode {
    gaussian_clouds: QueryState<(
        &'static GaussianCloudBindGroup,
        &'static Handle<ParticleBehaviors>,
        &'static MorphBindGroup,
    )>,
    initialized: bool,
    view_bind_group: QueryState<(
        &'static GaussianViewBindGroup,
        &'static ViewUniformOffset,
    )>,
}


impl FromWorld for MorphNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            gaussian_clouds: world.query(),
            initialized: false,
            view_bind_group: world.query(),
        }
    }
}

impl Node for MorphNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<MorphPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if !self.initialized {
            if let CachedPipelineState::Ok(_) =
                pipeline_cache.get_compute_pipeline_state(pipeline.particle_behavior_pipeline)
            {
                self.initialized = true;
            }

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
        let pipeline = world.resource::<MorphPipeline>();

        let command_encoder = render_context.command_encoder();

        for (
            view_bind_group,
            view_uniform_offset,
        ) in self.view_bind_group.iter_manual(world) {
            for (
                cloud_bind_group,
                behaviors_handle,
                morph_bind_group,
            ) in self.gaussian_clouds.iter_manual(world) {
                let behaviors = world.get_resource::<RenderAssets<ParticleBehaviors>>().unwrap().get(behaviors_handle).unwrap();
                let gaussian_uniforms = world.resource::<GaussianUniformBindGroups>();

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
                        &cloud_bind_group.cloud_bind_group,
                        &[]
                    );
                    pass.set_bind_group(
                        3,
                        &morph_bind_group.morph_bindgroup,
                        &[],
                    );

                    let particle_behavior = pipeline_cache.get_compute_pipeline(pipeline.particle_behavior_pipeline).unwrap();
                    pass.set_pipeline(particle_behavior);
                    pass.dispatch_workgroups(behaviors.morph_count / 32, 32, 1);
                }
            }
        }

        Ok(())
    }
}
