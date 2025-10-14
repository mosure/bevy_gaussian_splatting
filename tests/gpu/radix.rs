use std::{
    process::exit,
    sync::{Arc, Mutex},
};

use bevy::{
    core::FrameCount,
    core_pipeline::{
        Transparent3d,
        core_3d::graph::{Core3d, Node3d},
        tonemapping::Tonemapping,
    },
    prelude::*,
    render::{
        RenderApp,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_phase::SortedRenderPhase,
        renderer::{RenderContext, RenderQueue},
        view::ExtractedView,
    },
};

use bevy_gaussian_splatting::{
    GaussianCamera, PlanarGaussian3d, PlanarGaussian3dHandle, random_gaussians_3d,
    sort::SortedEntries,
};

use _harness::{TestHarness, TestState, TestStateArc, test_harness_app};

mod _harness;

pub mod node {
    pub const RADIX_SORT_TEST: &str = "radix_sort_test";
}

// run with `cargo run --bin test_gaussian --features="debug_gpu"`
fn main() {
    let mut app = test_harness_app(TestHarness {
        resolution: (512.0, 512.0),
    });

    app.add_systems(Startup, setup);

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .add_render_graph_node::<RadixTestNode>(CORE_3D, node::RADIX_SORT_TEST)
            .add_render_graph_edge(
                CORE_3D,
                node::RADIX_SORT_TEST,
                bevy::core_pipeline::core_3d::graph::node::END_MAIN_PASS,
            );
    }

    app.run();
}

fn setup(mut commands: Commands, mut gaussian_assets: ResMut<Assets<Cloud>>) {
    let cloud = gaussian_assets.add(random_gaussians_3d(10000));

    commands.spawn((
        PlanarGaussian3dHandle(cloud),
        CloudSettings {
            sort_mode: SortMode::Radix,
            ..default()
        },
        Name::new("gaussian_cloud"),
    ));

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            tonemapping: Tonemapping::None,
            ..default()
        },
        GaussianCamera,
    ));
}

pub struct RadixTestNode {
    gaussian_clouds: QueryState<(
        &'static PlanarGaussian3dHandle,
        &'static SortedEntriesHandle,
    )>,
    state: TestStateArc,
    views: QueryState<(
        &'static ExtractedView,
        &'static SortedRenderPhase<Transparent3d>,
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

// TODO: update radix sort to latest paradigm
impl Node for RadixTestNode {
    fn update(&mut self, world: &mut World) {
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
        for (view, _phase) in self.views.iter_manual(world) {
            let camera_position = view.transform.translation();

            for (cloud_handle, sorted_entries_handle) in self.gaussian_clouds.iter_manual(world) {
                let gaussian_cloud_res = world.get_resource::<RenderAssets<GpuCloud>>().unwrap();
                let sorted_entries_res = world
                    .get_resource::<RenderAssets<GpuSortedEntry>>()
                    .unwrap();

                let mut state = self.state.lock().unwrap();
                if gaussian_cloud_res.get(cloud_handle).is_none()
                    || sorted_entries_res.get(sorted_entries_handle).is_none()
                {
                    continue;
                } else if !state.test_loaded {
                    state.test_loaded = true;
                }

                let cloud = gaussian_cloud_res.get(cloud_handle).unwrap();
                let sorted_entries = sorted_entries_res.get(sorted_entries_handle).unwrap();
                let gaussians = cloud.debug_gpu.gaussians.clone();

                move |buffer: Result<wgpu::util::DownloadBuffer, wgpu::BufferAsyncError>| {
                    let binding = buffer.unwrap();
                    let u32_muck = bytemuck::cast_slice::<u8, u32>(&*binding);
                
                    let mut radix_sorted_indices = Vec::new();
                    for i in (1..u32_muck.len()).step_by(2) {
                        radix_sorted_indices.push((i, u32_muck[i] as usize));
                    }
                
                    // ✅ Project point into NDC space using view-projection matrix
                    let view_proj = view.projection * view.transform.compute_matrix().inverse();
                
                    radix_sorted_indices.iter().fold(f32::NEG_INFINITY, |prev_depth, &(entry_idx, idx)| {
                        if idx == 0 || u32_muck[entry_idx - 1] == 0xffffffff || u32_muck[entry_idx - 1] == 0x0 {
                            return prev_depth;
                        }
                
                        let position = gaussians[idx].position_visibility;
                        let position_vec3 = Vec3::new(position[0], position[1], position[2]);
                
                        // ✅ Project into clip space and normalize to NDC
                        let clip_pos = view_proj.project_point3(position_vec3);
                        let ndc_z = clip_pos.z;
                
                        // ✅ Assert non-decreasing NDC Z for back-to-front sorting
                        if ndc_z < prev_depth {
                            println!(
                                "Depth order error at idx {idx} → NDC z: {:.6} < {:.6}",
                                ndc_z, prev_depth
                            );
                            println!(
                                "Radix keys: [..., {:#010x}, {:#010x}, {:#010x}, ...]",
                                u32_muck[entry_idx - 1 - 2],
                                u32_muck[entry_idx - 1],
                                u32_muck[entry_idx - 1 + 2],
                            );
                        }
                
                        assert!(
                            ndc_z >= prev_depth,
                            "❌ Radix sort NDC-Z check failed: depth order incorrect ({} < {})",
                            ndc_z, prev_depth
                        );
                
                        ndc_z
                    });
                }
            }
        }

        Ok(())
    }
}
