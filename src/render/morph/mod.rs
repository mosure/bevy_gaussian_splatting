use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::*,
        renderer::RenderContext,
        view::ViewUniformOffset,
        render_graph::{
            Node,
            NodeRunError,
            RenderGraphContext,
        },
    },
};

use crate::{
    gaussian::GaussianCloud,
    render::{
        GaussianCloudBindGroup,
        GaussianCloudPipeline,
        GaussianViewBindGroup,
    },
};


#[derive(Resource)]
pub struct MorphPipeline {
    pub morph_layout: BindGroupLayout,
    pub morph_pipeline: CachedComputePipelineId,
}


pub struct MorphNode {
    gaussian_clouds: QueryState<(
        &'static Handle<GaussianCloud>,
        &'static GaussianCloudBindGroup,
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
        let pipeline = world.resource::<GaussianCloudPipeline>();
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
        let pipeline = world.resource::<GaussianCloudPipeline>();

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

                {
                    let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

                    pass.set_bind_group(
                        0,
                        &view_bind_group.value,
                        &[view_uniform_offset.offset],
                    );
                    pass.set_bind_group(
                        2,
                        &cloud_bind_group.cloud_bind_group,
                        &[]
                    );
                    pass.set_bind_group(
                        4,
                        &cloud_bind_group.morph_bindgroup,
                        &[],
                    );

                    let particle_behavior = pipeline_cache.get_compute_pipeline(pipeline.particle_behavior_pipeline).unwrap();
                    pass.set_pipeline(particle_behavior);
                    pass.dispatch_workgroups(cloud.morph_count, 1, 1);
                }
            }
        }

        Ok(())
    }
}
