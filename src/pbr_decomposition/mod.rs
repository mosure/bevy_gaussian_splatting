use bevy::prelude::*;
use bevy::render::{RenderApp, Render};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::render::renderer::RenderDevice;
use bevy::prelude::Res;
use bevy::prelude::Commands;

pub mod settings;
pub mod types;
pub mod spatial_hash;
pub mod normal_estimation;
pub mod synthetic_views;
pub mod material_separation;
pub mod energy_calibration;
pub mod orchestrator;

pub use settings::{PbrDecompositionSettings, SHCoordinateFrame};
pub use types::*;
pub use orchestrator::*;

use spatial_hash::load_spatial_hash_shader;
use normal_estimation::load_normal_estimation_shader;
use synthetic_views::load_synthetic_views_shader;
use material_separation::load_material_separation_shader;

pub struct PbrDecompositionPlugin;

impl Plugin for PbrDecompositionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PbrDecompositionSettings>();

        app.register_type::<PbrDecompositionSettings>();
        app.register_type::<PbrMaterialData>();
        app.register_type::<NormalData>();
        app.register_type::<SHCoordinateFrame>();
        app.register_type::<DecompositionStatus>();
        app.register_type::<DecomposedPbrMaterial>();

        load_spatial_hash_shader(app);
        load_normal_estimation_shader(app);
        load_synthetic_views_shader(app);
        load_material_separation_shader(app);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            use orchestrator::{PbrDecompositionLabel, PbrDecompositionNode3d, update_pbr_buffers, extract_settings_to_render};
            use bevy::render::render_graph::RenderGraphExt;
            use crate::sort::radix::RadixSortLabel;

            // Mirror settings into render world each frame
            render_app.add_systems(bevy::render::ExtractSchedule, extract_settings_to_render);

            // GPU buffer alloc
            render_app
                .init_resource::<normal_estimation::NormalEstimationBuffers>()
                .init_resource::<synthetic_views::SyntheticViewsBuffers>()
                .init_resource::<material_separation::MaterialSeparationBuffers>()
                .add_systems(Render, (ensure_pipelines, update_pbr_buffers));

            // Render graph node
            render_app.add_render_graph_node::<PbrDecompositionNode3d>(Core3d, PbrDecompositionLabel);
            render_app.add_render_graph_edge(Core3d, RadixSortLabel, PbrDecompositionLabel);
            render_app.add_render_graph_edge(Core3d, PbrDecompositionLabel, Node3d::LatePrepass);
        }
    }
}

fn ensure_pipelines(
    mut commands: Commands,
    device: Option<Res<RenderDevice>>,
    has_normals: Option<Res<normal_estimation::NormalEstimationPipeline>>,
    has_synth: Option<Res<synthetic_views::SyntheticViewsPipeline>>,
    has_mats: Option<Res<material_separation::MaterialSeparationPipeline>>,
) {
    if device.is_none() { return; }
    if has_normals.is_none() { commands.init_resource::<normal_estimation::NormalEstimationPipeline>(); }
    if has_synth.is_none() { commands.init_resource::<synthetic_views::SyntheticViewsPipeline>(); }
    if has_mats.is_none() { commands.init_resource::<material_separation::MaterialSeparationPipeline>(); }
}
