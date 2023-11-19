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
        GaussianUniformBindGroups,
        GaussianViewBindGroup,
        ShaderDefines,
    },
};


pub struct RadixSortNode {
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

                {
                    command_encoder.clear_buffer(
                        &cloud.sorting_global_buffer,
                        0,
                        None,
                    );

                    command_encoder.clear_buffer(
                        &cloud.draw_indirect_buffer,
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
                        // clear SortingGlobal.status_counters
                        let size = (ShaderDefines::default().radix_base * ShaderDefines::default().max_tile_count_c) as u64 * std::mem::size_of::<u32>() as u64;
                        command_encoder.clear_buffer(
                            &cloud.sorting_global_buffer,
                            0,
                            std::num::NonZeroU64::new(size).unwrap().into()
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
