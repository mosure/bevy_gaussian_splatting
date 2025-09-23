use std::{any::TypeId, marker::PhantomData};

use bevy::{
    asset::{LoadState, load_internal_asset, weak_handle},
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        prepass::PreviousViewUniformOffset,
    },
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderSet,
        extract_component::DynamicUniformIndex,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel},
        render_resource::{
            BindGroup, BindGroupLayout, CachedComputePipelineId, CachedPipelineState,
            ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
        },
        renderer::{RenderContext, RenderDevice},
        sync_world::RenderEntity,
        view::ViewUniformOffset,
    },
};
use bevy_interleave::prelude::*;

use crate::{
    camera::GaussianCamera,
    gaussian::formats::planar_3d::PlanarGaussian3d,
    render::{
        CloudPipeline, CloudPipelineKey, CloudUniform, GaussianComputeViewBindGroup,
        GaussianUniformBindGroups, PlanarStorageRebindQueue, shader_defs,
    },
};

const INTERPOLATE_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("b0b03f7e-9ec2-4e7d-bc96-3ddc1a8c5942");
const WORKGROUP_SIZE: u32 = 256;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct InterpolateLabel;

pub struct InterpolatePlugin<R: PlanarSync> {
    phantom: PhantomData<fn() -> R>,
}

impl<R: PlanarSync> Default for InterpolatePlugin<R> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<R> Plugin for InterpolatePlugin<R>
where
    R: PlanarSync + Send + Sync + 'static,
    R::GpuPlanarType: GpuPlanarStorage,
{
    fn build(&self, app: &mut App) {
        if TypeId::of::<R::PlanarType>() != TypeId::of::<PlanarGaussian3d>() {
            return;
        }

        load_internal_asset!(
            app,
            INTERPOLATE_SHADER_HANDLE,
            "interpolate.wgsl",
            Shader::from_wgsl
        );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_graph_node::<GaussianInterpolateNode<R>>(Core3d, InterpolateLabel)
                .add_render_graph_edge(Core3d, InterpolateLabel, Node3d::LatePrepass)
                .add_systems(ExtractSchedule, extract_gaussian_interpolate::<R>)
                .add_systems(
                    Render,
                    (queue_gaussian_interpolate_bind_groups::<R>.in_set(RenderSet::Queue),),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<GaussianInterpolatePipeline<R>>();
        }
    }
}

#[derive(Component, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct GaussianInterpolate<R: PlanarSync> {
    pub lhs: R::PlanarTypeHandle,
    pub rhs: R::PlanarTypeHandle,
}

impl<R: PlanarSync> Clone for GaussianInterpolate<R>
where
    R::PlanarTypeHandle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
        }
    }
}
#[derive(Component)]
pub struct GaussianInterpolateBindGroups<R: PlanarSync> {
    pub lhs: BindGroup,
    pub rhs: BindGroup,
    pub output: BindGroup,
    phantom: PhantomData<fn() -> R>,
}

#[derive(Resource)]
pub struct GaussianInterpolatePipeline<R: PlanarSync> {
    pub output_layout: BindGroupLayout,
    pub interpolate_pipeline: CachedComputePipelineId,
    phantom: PhantomData<fn() -> R>,
}

impl<R: PlanarSync> FromWorld for GaussianInterpolatePipeline<R>
where
    R::GpuPlanarType: GpuPlanarStorage,
{
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();
        let gaussian_cloud_pipeline = render_world.resource::<CloudPipeline<R>>();
        let pipeline_cache = render_world.resource::<PipelineCache>();

        let output_layout = R::GpuPlanarType::bind_group_layout(render_device, false);

        let mut key = CloudPipelineKey::default();
        key.binary_gaussian_op = true;
        let shader_defs = shader_defs(key);

        let interpolate_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("gaussian_interpolate_pipeline".into()),
                layout: vec![
                    gaussian_cloud_pipeline.compute_view_layout.clone(),
                    gaussian_cloud_pipeline.gaussian_uniform_layout.clone(),
                    gaussian_cloud_pipeline.gaussian_cloud_layout.clone(),
                    gaussian_cloud_pipeline.gaussian_cloud_layout.clone(),
                    output_layout.clone(),
                ],
                push_constant_ranges: vec![],
                shader: INTERPOLATE_SHADER_HANDLE,
                shader_defs,
                entry_point: "interpolate_gaussians".into(),
                zero_initialize_workgroup_memory: true,
            });

        Self {
            output_layout,
            interpolate_pipeline,
            phantom: PhantomData,
        }
    }
}

pub fn extract_gaussian_interpolate<R>(
    mut commands: Commands,
    query: Extract<Query<(Entity, &RenderEntity, &GaussianInterpolate<R>)>>,
) where
    R: PlanarSync,
    R::PlanarTypeHandle: Clone,
{
    let mut extracted: Vec<(Entity, (RenderEntity, GaussianInterpolate<R>))> = Vec::new();

    for (_entity, render_entity, component) in query.iter() {
        let render_entity = *render_entity;
        extracted.push((render_entity.id(), (render_entity, component.clone())));
    }

    if !extracted.is_empty() {
        commands.try_insert_batch(extracted);
    }
}

pub fn queue_gaussian_interpolate_bind_groups<R: PlanarSync>(
    mut commands: Commands,
    interpolate_pipeline: Res<GaussianInterpolatePipeline<R>>,
    gaussian_cloud_pipeline: Res<CloudPipeline<R>>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    gpu_planars: Res<RenderAssets<R::GpuPlanarType>>,
    mut rebind_queue: ResMut<PlanarStorageRebindQueue<R>>,
    mut query: Query<(
        Entity,
        Ref<GaussianInterpolate<R>>,
        &R::PlanarTypeHandle,
        Option<&GaussianInterpolateBindGroups<R>>,
    )>,
) where
    R::GpuPlanarType: GpuPlanarStorage,
{
    let inputs_changed = gaussian_cloud_pipeline.is_changed() || gpu_planars.is_changed();
    let mut pending_inserts: Vec<(Entity, GaussianInterpolateBindGroups<R>)> = Vec::new();

    for (entity, interpolate, output_handle, existing) in query.iter_mut() {
        let mut rebuild = inputs_changed || interpolate.is_changed();
        if existing.is_none() {
            rebuild = true;
        }

        if !rebuild {
            continue;
        }

        let lhs_handle = interpolate.lhs.handle().clone();
        let rhs_handle = interpolate.rhs.handle().clone();
        let output_asset_handle = output_handle.handle().clone();

        let mut ready = true;
        for handle in [&lhs_handle, &rhs_handle, &output_asset_handle] {
            let Some(load_state) = asset_server.get_load_state(handle.id()) else {
                ready = false;
                break;
            };
            if !matches!(load_state, LoadState::Loaded) {
                ready = false;
                break;
            }

            if gpu_planars.get(handle.id()).is_none() {
                ready = false;
                break;
            }
        }

        if !ready {
            continue;
        }

        rebind_queue.push_unique(output_asset_handle.id());

        let lhs_gpu = gpu_planars.get(lhs_handle.id()).unwrap();
        let rhs_gpu = gpu_planars.get(rhs_handle.id()).unwrap();
        let output_gpu = gpu_planars.get(output_asset_handle.id()).unwrap();

        let lhs_bind_group = lhs_gpu.bind_group(
            render_device.as_ref(),
            &gaussian_cloud_pipeline.gaussian_cloud_layout,
        );
        let rhs_bind_group = rhs_gpu.bind_group(
            render_device.as_ref(),
            &gaussian_cloud_pipeline.gaussian_cloud_layout,
        );
        let output_bind_group =
            output_gpu.bind_group(render_device.as_ref(), &interpolate_pipeline.output_layout);

        pending_inserts.push((
            entity,
            GaussianInterpolateBindGroups::<R> {
                lhs: lhs_bind_group,
                rhs: rhs_bind_group,
                output: output_bind_group,
                phantom: PhantomData,
            },
        ));
    }

    if !pending_inserts.is_empty() {
        commands.try_insert_batch(pending_inserts);
    }
}

pub struct GaussianInterpolateNode<R: PlanarSync> {
    gaussian_clouds: QueryState<(
        &'static GaussianInterpolate<R>,
        &'static GaussianInterpolateBindGroups<R>,
        &'static DynamicUniformIndex<CloudUniform>,
        &'static R::PlanarTypeHandle,
    )>,
    view_bind_group: QueryState<(
        &'static GaussianCamera,
        &'static GaussianComputeViewBindGroup,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
    )>,
    initialized: bool,
    phantom: PhantomData<fn() -> R>,
}

impl<R: PlanarSync> FromWorld for GaussianInterpolateNode<R> {
    fn from_world(world: &mut World) -> Self {
        Self {
            gaussian_clouds: world.query(),
            view_bind_group: world.query(),
            initialized: false,
            phantom: PhantomData,
        }
    }
}

impl<R: PlanarSync> Node for GaussianInterpolateNode<R>
where
    R::GpuPlanarType: GpuPlanarStorage,
{
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GaussianInterpolatePipeline<R>>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if !self.initialized {
            if let CachedPipelineState::Ok(_) =
                pipeline_cache.get_compute_pipeline_state(pipeline.interpolate_pipeline)
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
        let pipeline = world.resource::<GaussianInterpolatePipeline<R>>();
        let gaussian_uniforms = world.resource::<GaussianUniformBindGroups>();
        let Some(uniform_bind_group) = gaussian_uniforms.base_bind_group.as_ref() else {
            return Ok(());
        };

        let gpu_planars = world.resource::<RenderAssets<R::GpuPlanarType>>();

        let command_encoder = render_context.command_encoder();

        for (_camera, view_bind_group, view_uniform_offset, previous_view_uniform_offset) in
            self.view_bind_group.iter_manual(world)
        {
            for (_interpolate, bind_groups, cloud_uniform_index, output_handle) in
                self.gaussian_clouds.iter_manual(world)
            {
                let Some(output_gpu) = gpu_planars.get(output_handle.handle()) else {
                    continue;
                };

                let gaussian_count = output_gpu.len() as u32;
                if gaussian_count == 0 {
                    continue;
                }

                let workgroups = (gaussian_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                let pipeline_id = pipeline_cache
                    .get_compute_pipeline(pipeline.interpolate_pipeline)
                    .unwrap();

                let mut pass =
                    command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

                pass.set_pipeline(pipeline_id);
                pass.set_bind_group(
                    0,
                    &view_bind_group.value,
                    &[
                        view_uniform_offset.offset,
                        previous_view_uniform_offset.offset,
                    ],
                );
                pass.set_bind_group(1, uniform_bind_group, &[cloud_uniform_index.index()]);
                pass.set_bind_group(2, &bind_groups.lhs, &[]);
                pass.set_bind_group(3, &bind_groups.rhs, &[]);
                pass.set_bind_group(4, &bind_groups.output, &[]);

                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        Ok(())
    }
}

