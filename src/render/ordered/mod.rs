//! One stable depth order and one alpha-over draw across eligible 3D clouds.
//! Flat, discrete CPU-selected and GPU-traversed sources share authored
//! projection helpers with GPS. Authored GPU hierarchy edges optionally use
//! current-camera spatial transitions. CPU candidate morphs and additive
//! rendering are excluded from this shared stream.

mod gpu;
mod pipeline;

#[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
pub use gpu::GlobalOrderDispatchTestLimit;
pub(crate) use gpu::GlobalOrderRender;
#[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
pub(crate) use gpu::Views as OrderedViews;
#[cfg(all(feature = "testing", feature = "headless", not(target_arch = "wasm32")))]
pub(crate) use gpu::preflight_global_order;
pub use gpu::{GaussianGlobalOrderDiagnostics, GaussianGlobalOrderFrame};
pub(crate) use gpu::{GlobalOrderPrepare, GlobalOrderReadiness};

use crate::{
    CloudSettings, GaussianMode, RasterizeMode, stream::render_commit::LodRenderCandidate,
};
use bevy::{
    asset::{load_internal_asset, uuid_handle},
    core_pipeline::{
        Core3d, Core3dSystems,
        core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
    },
    prelude::*,
    render::{
        GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
    },
};
use bevy_args::{Deserialize, Serialize};

const TYPES_SHADER: Handle<Shader> = uuid_handle!("0be8e3f0-88cc-46f1-8603-265cd0589bfb");
const PROJECT_SHADER: Handle<Shader> = uuid_handle!("17e64c75-1a89-4500-af97-b42356a19e52");
const GATHER_SHADER: Handle<Shader> = uuid_handle!("0b2ce4df-fb81-4934-94c3-cd85a00dd61b");
const DRAW_SHADER: Handle<Shader> = uuid_handle!("166cc70e-f234-43e2-9bb2-c6ab9467d92e");
// Radix still dispatches one 1,024-record tile along a single dimension.
pub(crate) const MAX_ORDERED_GAUSSIANS: u32 = 65_535 * 1_024;

/// Explicit camera selection for a bounded global order across ordinary quads.
#[derive(Component, Clone, Debug, PartialEq, ExtractComponent, Reflect, Serialize, Deserialize)]
#[reflect(Component)]
#[serde(default)]
pub struct GaussianGlobalOrderSettings {
    pub max_projected_gaussians: u32,
    pub max_gpu_bytes: u64,
}

impl Default for GaussianGlobalOrderSettings {
    fn default() -> Self {
        Self {
            max_projected_gaussians: 1_048_576,
            max_gpu_bytes: 256 * 1024 * 1024,
        }
    }
}

impl GaussianGlobalOrderSettings {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_projected_gaussians == 0
            || self.max_projected_gaussians > MAX_ORDERED_GAUSSIANS
            || self.max_gpu_bytes == 0
        {
            return Err(
                "global quad order requires nonzero memory and 1..=67107840 projected records",
            );
        }
        Ok(())
    }
}

pub(crate) fn global_order_for_cloud(
    camera: Option<&GaussianGlobalOrderSettings>,
    cloud: &CloudSettings,
) -> bool {
    camera.is_some()
        && cloud.gaussian_mode == GaussianMode::Gaussian3d
        && cloud.rasterize_mode == RasterizeMode::Color
        && !cloud.additive
        && cloud.lod_debug == Default::default()
        && !cloud.visualize_bounding_box
        && matches!(
            cloud.sort_mode,
            crate::sort::SortMode::Radix | crate::sort::SortMode::None
        )
}

pub(crate) fn supports_candidate(candidate: &LodRenderCandidate) -> bool {
    !candidate.is_external_active_set() && candidate.temporal_transition().is_none()
}

pub struct GaussianGlobalOrderPlugin;

impl Plugin for GaussianGlobalOrderPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, TYPES_SHADER, "types.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, PROJECT_SHADER, "project.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, GATHER_SHADER, "gather.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, DRAW_SHADER, "draw.wgsl", Shader::from_wgsl);
        app.register_type::<GaussianGlobalOrderSettings>()
            .add_plugins(ExtractComponentPlugin::<GaussianGlobalOrderSettings>::default())
            .init_resource::<GaussianGlobalOrderDiagnostics>();
        let diagnostics = app
            .world()
            .resource::<GaussianGlobalOrderDiagnostics>()
            .clone();
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .insert_resource(diagnostics)
                .init_gpu_resource::<pipeline::Pipelines>()
                .init_gpu_resource::<gpu::Views>()
                .init_resource::<GlobalOrderReadiness>()
                .add_systems(RenderStartup, gpu::reset)
                .add_systems(
                    Render,
                    gpu::prepare
                        .in_set(RenderSystems::PrepareResources)
                        .in_set(GlobalOrderPrepare)
                        .after(super::lod::LodCompactionPrepare)
                        .after(super::traversal::GpuLodTraversalPrepare),
                )
                .add_systems(
                    Core3d,
                    gpu::render
                        .in_set(Core3dSystems::MainPass)
                        .in_set(GlobalOrderRender)
                        .after(super::traversal::GpuLodTraversalRender)
                        .after(main_opaque_pass_3d)
                        .before(main_transparent_pass_3d),
                )
                .add_systems(
                    Render,
                    gpu::collect
                        .in_set(RenderSystems::Cleanup)
                        .after(RenderSystems::Render),
                );
        }
    }
}
