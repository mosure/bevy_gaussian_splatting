use std::{
    process::exit,
    sync::{
        Arc,
        Mutex,
    },
};

use bevy::{
    prelude::*,
    core::FrameCount,
    core_pipeline::core_3d::CORE_3D,
    render::{
        RenderApp,
        renderer::{
            RenderContext,
            RenderQueue,
        },
        render_graph::{
            Node,
            NodeRunError,
            RenderGraphApp,
            RenderGraphContext,
        },
        render_asset::RenderAssets, view::ExtractedView,
    },
};

use bevy_gaussian_splatting::{
    GaussianCloud,
    GaussianSplattingBundle,
    random_gaussians,
};

use _harness::{
    TestHarness,
    test_harness_app,
    TestState,
    TestStateArc,
};

mod _harness;


pub mod node {
    pub const RADIX_SORT_TEST: &str = "radix_sort_test";
}


fn main() {
    let mut app = test_harness_app(TestHarness {
        resolution: (512.0, 512.0),
    });

    app.add_systems(Startup, setup);

    if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .add_render_graph_node::<RadixTestNode>(
                CORE_3D,
                node::RADIX_SORT_TEST,
            )
            .add_render_graph_edge(
                CORE_3D,
                node::RADIX_SORT_TEST,
                 bevy::core_pipeline::core_3d::graph::node::END_MAIN_PASS,
            );
    }

    app.run();
}

fn setup(
    mut commands: Commands,
    mut gaussian_assets: ResMut<Assets<GaussianCloud>>,
) {
    let cloud = gaussian_assets.add(random_gaussians(10000));

    commands.spawn((
        GaussianSplattingBundle {
            cloud,
            ..default()
        },
        Name::new("gaussian_cloud"),
    ));

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            ..default()
        },
    ));
}


pub struct RadixTestNode {
    gaussian_clouds: QueryState<(
        &'static Handle<GaussianCloud>,
    )>,
    state: TestStateArc,
    views: QueryState<(
        &'static ExtractedView,
    )>,
    start_frame: u32,
}

impl FromWorld for RadixTestNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            gaussian_clouds: world.query(),
            state: Arc::new(Mutex::new(TestState::default())),
            views: world.query(),
            start_frame: 0,
        }
    }
}


impl Node for RadixTestNode {
    fn update(
        &mut self,
        world: &mut World,
    ) {
        let mut state = self.state.lock().unwrap();
        if state.test_completed {
            exit(0);
        }

        if state.test_loaded && self.start_frame == 0 {
            self.start_frame = world.get_resource::<FrameCount>().unwrap().0;
        }

        let frame_count = world.get_resource::<FrameCount>().unwrap().0;
        const FRAME_LIMIT: u32 = 10;
        if state.test_loaded && frame_count >= self.start_frame + FRAME_LIMIT {
            state.test_completed = true;
        }

        self.gaussian_clouds.update_archetypes(world);
        self.views.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        println!("radix sort test: running frame {}...", world.get_resource::<FrameCount>().unwrap().0);

        for (view,) in self.views.iter_manual(world) {
            let camera_position = view.transform.translation();
            println!("radix sort test: camera position: {:?}", camera_position);

            for (cloud_handle,) in self.gaussian_clouds.iter_manual(world) {
                let gaussian_cloud_res = world.get_resource::<RenderAssets<GaussianCloud>>().unwrap();

                let mut state = self.state.lock().unwrap();
                if gaussian_cloud_res.get(cloud_handle).is_none() {
                    continue;
                } else if !state.test_loaded {
                    state.test_loaded = true;
                }

                let cloud = gaussian_cloud_res.get(cloud_handle).unwrap();
                let gaussians = cloud.debug_gpu.gaussians.clone();

                println!("radix sort test: {} gaussians", gaussians.len());

                wgpu::util::DownloadBuffer::read_buffer(
                    render_context.render_device().wgpu_device(),
                    world.get_resource::<RenderQueue>().unwrap().0.as_ref(),
                    &cloud.radix_sort_buffers.entry_buffer_a.slice(
                        0..cloud.radix_sort_buffers.entry_buffer_a.size()
                    ),
                    move |buffer: Result<wgpu::util::DownloadBuffer, wgpu::BufferAsyncError>| {
                        let binding = buffer.unwrap();
                        let u32_muck = bytemuck::cast_slice::<u8, u32>(&*binding);

                        let mut radix_sorted_indices = Vec::new();
                        for i in (1..u32_muck.len()).step_by(2) {
                            radix_sorted_indices.push(u32_muck[i]);
                        }

                        let max_depth = radix_sorted_indices.iter()
                            .fold(0.0, |depth_acc, &idx| {
                                let position = gaussians[idx as usize].position;
                                let position_vec3 = Vec3::new(position[0], position[1], position[2]);
                                let depth = (position_vec3 - camera_position).length();

                                assert!(depth_acc <= depth, "radix sort, non-decreasing check failed: {} > {}", depth_acc, depth);

                                depth
                            });

                        assert!(max_depth > 0.0, "radix sort, max depth check failed: {}", max_depth);

                        // TODO: analyze incorrectly sorted gaussian positions or upstream buffers (e.g. histogram sort error vs. position of gaussian distance from correctly sorted index)
                    }
                );
            }
        }

        Ok(())
    }
}
