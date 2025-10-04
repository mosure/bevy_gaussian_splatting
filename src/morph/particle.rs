use rand::{Rng, prelude::Distribution};
use std::marker::Copy;

#[allow(unused_imports)]
use bevy::{
    asset::{LoadState, load_internal_asset, uuid_handle},
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    ecs::system::{SystemParamItem, lifetimeless::SRes},
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSystems,
        render_asset::{
            PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssetUsages, RenderAssets,
        },
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel},
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, Buffer, BufferBinding, BufferBindingType, BufferInitDescriptor,
            BufferSize, BufferUsages, CachedComputePipelineId, CachedPipelineState,
            ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, PipelineCache,
            ShaderStages, ShaderType, TextureDimension, TextureFormat,
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewUniformOffset,
    },
};
use bevy_interleave::prelude::*;
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::{
    camera::GaussianCamera,
    render::{
        CloudPipeline, CloudPipelineKey, GaussianUniformBindGroups, GaussianViewBindGroup,
        shader_defs,
    },
};

const PARTICLE_SHADER_HANDLE: Handle<Shader> = uuid_handle!("00000000-0000-0000-0000-00369c79ab8f");

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct MorphLabel;

pub struct ParticleBehaviorPlugin<R: PlanarSync> {
    phantom: std::marker::PhantomData<R>,
}
impl<R: PlanarSync> Default for ParticleBehaviorPlugin<R> {
    fn default() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<R: PlanarSync> Plugin for ParticleBehaviorPlugin<R> {
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_render_graph_node::<ParticleBehaviorNode<R>>(Core3d, MorphLabel);

            // TODO: avoid duplicating the extract system
            render_app.add_systems(
                Render,
                (queue_particle_behavior_bind_group::<R>.in_set(RenderSystems::Queue),),
            );
        }

        if app.is_plugin_added::<RenderAssetPlugin<GpuParticleBehaviorBuffers>>() {
            return;
        }

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_render_graph_edge(Core3d, MorphLabel, Node3d::LatePrepass);
        }

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(ExtractSchedule, extract_particle_behaviors);
        }

        load_internal_asset!(
            app,
            PARTICLE_SHADER_HANDLE,
            "particle.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<ParticleBehaviors>();
        app.register_type::<ParticleBehaviorsHandle>();
        app.init_asset::<ParticleBehaviors>();
        app.init_asset::<ParticleBehaviors>();
        app.register_asset_reflect::<ParticleBehaviors>();
        app.add_plugins(RenderAssetPlugin::<GpuParticleBehaviorBuffers>::default());
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<ParticleBehaviorPipeline<R>>();
        }
    }
}

pub fn extract_particle_behaviors(
    mut commands: Commands,
    mut prev_commands_len: Local<usize>,
    gaussians_query: Extract<Query<(Entity, &ParticleBehaviorsHandle)>>,
) {
    let mut commands_list = Vec::with_capacity(*prev_commands_len);

    for (entity, behaviors) in gaussians_query.iter() {
        commands_list.push((entity, behaviors.clone()));
    }
    *prev_commands_len = commands_list.len();
    commands.insert_or_spawn_batch(commands_list);
}

#[derive(Debug, Clone)]
pub struct GpuParticleBehaviorBuffers {
    pub particle_behavior_count: u32,
    pub particle_behavior_buffer: Buffer,
}

impl RenderAsset for GpuParticleBehaviorBuffers {
    type SourceAsset = ParticleBehaviors;
    type Param = SRes<RenderDevice>;

    fn prepare_asset(
        source: Self::SourceAsset,
        _: AssetId<Self::SourceAsset>,
        render_device: &mut SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let particle_behavior_count = source.0.len() as u32;

        let particle_behavior_buffer =
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("particle behavior buffer"),
                contents: bytemuck::cast_slice(source.0.as_slice()),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::STORAGE,
            });

        Ok(GpuParticleBehaviorBuffers {
            particle_behavior_count,
            particle_behavior_buffer,
        })
    }

    fn asset_usage(_: &Self::SourceAsset) -> RenderAssetUsages {
        RenderAssetUsages::default()
    }
}

#[derive(Resource)]
pub struct ParticleBehaviorPipeline<R: PlanarSync> {
    pub particle_behavior_layout: BindGroupLayout,
    pub particle_behavior_pipeline: CachedComputePipelineId,
    phantom: std::marker::PhantomData<R>,
}

impl<R: PlanarSync> FromWorld for ParticleBehaviorPipeline<R> {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();
        let gaussian_cloud_pipeline = render_world.resource::<CloudPipeline<R>>();

        let particle_behavior_layout = render_device.create_bind_group_layout(
            Some("gaussian_cloud_particle_behavior_layout"),
            &[BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(
                        std::mem::size_of::<ParticleBehavior>() as u64
                    ),
                },
                count: None,
            }],
        );

        let shader_defs = shader_defs(CloudPipelineKey::default());
        let pipeline_cache = render_world.resource::<PipelineCache>();

        let particle_behavior_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("particle_behavior_pipeline".into()),
                layout: vec![
                    gaussian_cloud_pipeline.compute_view_layout.clone(),
                    gaussian_cloud_pipeline.gaussian_uniform_layout.clone(),
                    gaussian_cloud_pipeline.gaussian_cloud_layout.clone(),
                    particle_behavior_layout.clone(),
                ],
                push_constant_ranges: vec![],
                shader: PARTICLE_SHADER_HANDLE,
                shader_defs: shader_defs.clone(),
                entry_point: Some("apply_particle_behaviors".into()),
                zero_initialize_workgroup_memory: true,
            });

        Self {
            particle_behavior_layout,
            particle_behavior_pipeline,
            phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Component)]
pub struct ParticleBehaviorBindGroup {
    pub particle_behavior_bindgroup: BindGroup,
}

pub fn queue_particle_behavior_bind_group<R: PlanarSync>(
    mut commands: Commands,
    particle_behavior_pipeline: Res<ParticleBehaviorPipeline<R>>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    particle_behaviors_res: Res<RenderAssets<GpuParticleBehaviorBuffers>>,
    particle_behaviors: Query<(Entity, &ParticleBehaviorsHandle)>,
) {
    for (entity, behaviors_handle) in particle_behaviors.iter() {
        if let Some(load_state) = asset_server.get_load_state(&behaviors_handle.0) {
            if load_state.is_loading() {
                continue;
            }
        }

        if particle_behaviors_res.get(&behaviors_handle.0).is_none() {
            continue;
        }

        let behaviors = particle_behaviors_res.get(&behaviors_handle.0).unwrap();

        let particle_behavior_bindgroup = render_device.create_bind_group(
            "particle_behavior_bind_group",
            &particle_behavior_pipeline.particle_behavior_layout,
            &[BindGroupEntry {
                binding: 7,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &behaviors.particle_behavior_buffer,
                    offset: 0,
                    size: BufferSize::new(behaviors.particle_behavior_buffer.size()),
                }),
            }],
        );

        commands.entity(entity).insert(ParticleBehaviorBindGroup {
            particle_behavior_bindgroup,
        });
    }
}

pub struct ParticleBehaviorNode<R: PlanarSync> {
    gaussian_clouds: QueryState<(
        &'static PlanarStorageBindGroup<R>,
        &'static ParticleBehaviorsHandle,
        &'static ParticleBehaviorBindGroup,
    )>,
    initialized: bool,
    view_bind_group: QueryState<(
        &'static GaussianCamera,
        &'static GaussianViewBindGroup,
        &'static ViewUniformOffset,
    )>,
    phantom: std::marker::PhantomData<R>,
}

impl<R: PlanarSync> FromWorld for ParticleBehaviorNode<R> {
    fn from_world(world: &mut World) -> Self {
        Self {
            gaussian_clouds: world.query(),
            initialized: false,
            view_bind_group: world.query(),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<R: PlanarSync> Node for ParticleBehaviorNode<R> {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<ParticleBehaviorPipeline<R>>();
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
        let pipeline = world.resource::<ParticleBehaviorPipeline<R>>();

        let command_encoder = render_context.command_encoder();

        for (_gaussian_camera, view_bind_group, view_uniform_offset) in
            self.view_bind_group.iter_manual(world)
        {
            for (planar_storage_bind_group, behaviors_handle, particle_behavior_bind_group) in
                self.gaussian_clouds.iter_manual(world)
            {
                let behaviors = world
                    .get_resource::<RenderAssets<GpuParticleBehaviorBuffers>>()
                    .unwrap()
                    .get(behaviors_handle.0.id())
                    .unwrap();
                let gaussian_uniforms = world.resource::<GaussianUniformBindGroups>();

                {
                    let mut pass =
                        command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

                    pass.set_bind_group(0, &view_bind_group.value, &[view_uniform_offset.offset]);
                    pass.set_bind_group(
                        1,
                        gaussian_uniforms.base_bind_group.as_ref().unwrap(),
                        &[0],
                    );
                    pass.set_bind_group(2, &planar_storage_bind_group.bind_group, &[]);
                    pass.set_bind_group(
                        3,
                        &particle_behavior_bind_group.particle_behavior_bindgroup,
                        &[],
                    );

                    let particle_behavior = pipeline_cache
                        .get_compute_pipeline(pipeline.particle_behavior_pipeline)
                        .unwrap();
                    pass.set_pipeline(particle_behavior);
                    pass.dispatch_workgroups(behaviors.particle_behavior_count / 32, 32, 1);
                }
            }
        }

        Ok(())
    }
}

#[derive(Component, Clone, Debug, Default, PartialEq, Reflect)]
#[reflect(Component, Default)]
pub struct ParticleBehaviorsHandle(pub Handle<ParticleBehaviors>);

impl From<Handle<ParticleBehaviors>> for ParticleBehaviorsHandle {
    fn from(handle: Handle<ParticleBehaviors>) -> Self {
        Self(handle)
    }
}

impl From<ParticleBehaviorsHandle> for AssetId<ParticleBehaviors> {
    fn from(handle: ParticleBehaviorsHandle) -> Self {
        handle.0.id()
    }
}

impl From<&ParticleBehaviorsHandle> for AssetId<ParticleBehaviors> {
    fn from(handle: &ParticleBehaviorsHandle) -> Self {
        handle.0.id()
    }
}

// TODO: add more particle system functionality (e.g. lifetime, color)
#[derive(
    Clone, Debug, Copy, PartialEq, Reflect, ShaderType, Pod, Zeroable, Serialize, Deserialize,
)]
#[repr(C)]
pub struct ParticleBehavior {
    pub indicies: [u32; 4],
    pub velocity: [f32; 4],
    pub acceleration: [f32; 4],
    pub jerk: [f32; 4],
}

impl Default for ParticleBehavior {
    fn default() -> Self {
        Self {
            indicies: [0, 0, 0, 0],
            velocity: [0.0, 0.0, 0.0, 0.0],
            acceleration: [0.0, 0.0, 0.0, 0.0],
            jerk: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

#[derive(Asset, Clone, Debug, Default, PartialEq, Reflect, Serialize, Deserialize)]
pub struct ParticleBehaviors(pub Vec<ParticleBehavior>);

impl Distribution<ParticleBehavior> for rand::distributions::Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ParticleBehavior {
        ParticleBehavior {
            acceleration: [
                rng.gen_range(-0.01..0.01),
                rng.gen_range(-0.01..0.01),
                rng.gen_range(-0.01..0.01),
                rng.gen_range(-0.01..0.01),
            ],
            jerk: [
                rng.gen_range(-0.0001..0.0001),
                rng.gen_range(-0.0001..0.0001),
                rng.gen_range(-0.0001..0.0001),
                rng.gen_range(-0.0001..0.0001),
            ],
            velocity: [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ],
            ..Default::default()
        }
    }
}

pub fn random_particle_behaviors(n: usize) -> ParticleBehaviors {
    let mut rng = rand::thread_rng();
    let mut behaviors = Vec::with_capacity(n);
    for i in 0..n {
        let mut behavior: ParticleBehavior = rng.r#gen();
        behavior.indicies[0] = i as u32;
        behaviors.push(behavior);
    }

    ParticleBehaviors(behaviors)
}
