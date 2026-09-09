//! Order-independent Gaussian point splatting for planar 3D clouds.
//!
//! This opt-in camera backend samples the opacity-corrected Poisson process of
//! Rijsdijk et al., *Gaussian Point Splatting* (2026). All participating clouds
//! share one visibility target. Portable WGSL uses separate depth and winner
//! reductions instead of a 64-bit atomic; no depth sort is dispatched.
//!
//! See `docs/gaussian_point_splatting.md` for supported presentations, memory
//! limits and the distinction between sample noise and LoD approximation error.

mod budget;
mod gpu;
#[cfg(any(test, feature = "testing"))]
pub mod math;
mod pipeline;
mod settings;

pub use budget::{
    GaussianPointSplattingViewBudget, GaussianPointSplattingViewBudgetAction,
    GaussianPointSplattingViewBudgetDiagnostics, GaussianPointSplattingViewBudgetFrame,
};
#[cfg(all(
    feature = "testing",
    any(
        all(feature = "headless", not(target_arch = "wasm32")),
        all(target_arch = "wasm32", feature = "webgpu", feature = "web_asset")
    )
))]
pub(crate) use gpu::PointViews;
pub use gpu::{GaussianPointSplattingDiagnostics, GaussianPointSplattingFrame};
pub use settings::*;

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    core_pipeline::{
        Core3d, Core3dSystems,
        core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
    },
    prelude::*,
    render::{
        GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::ExtractComponentPlugin, init_gpu_resource, view::RetainedViewEntity,
    },
};
use std::collections::HashSet;

use crate::{
    CloudSettings, GaussianMode, RasterizeMode, stream::render_commit::LodRenderCandidate,
};

const SAMPLING_SHADER: Handle<Shader> = uuid_handle!("8d375b19-47d3-4e8c-ade5-dd133614e521");
const TYPES_SHADER: Handle<Shader> = uuid_handle!("abbdbb4b-daa2-4578-b501-2194588599cd");
const PROJECT_SHADER: Handle<Shader> = uuid_handle!("fdf88a13-9137-40e2-b8c6-10283abb5ca5");
const WORK_SHADER: Handle<Shader> = uuid_handle!("31660f21-9fc0-4594-bc62-b35e9e18ed11");
const COMPOSITE_SHADER: Handle<Shader> = uuid_handle!("f01d873d-85e2-4416-ac0d-ab8993b997bc");

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PointSplattingPrepare;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PointSplattingRender;

/// Admission/pipeline readiness; successful rendering is acknowledged separately
/// from asynchronous GPU feedback, including overflow and candidate identity.
#[derive(Resource, Default)]
pub(crate) struct PointSplattingPipelineReadiness {
    ready: HashSet<RetainedViewEntity>,
    prepared: HashSet<RetainedViewEntity>,
    claimed: HashSet<RetainedViewEntity>,
    retained: HashSet<RetainedViewEntity>,
}

impl PointSplattingPipelineReadiness {
    pub(crate) fn is_ready(&self, view: RetainedViewEntity) -> bool {
        self.ready.contains(&view)
    }

    /// Pipelines and bounded workspace can consume the selected atlas once its
    /// source handle is published. This is preflight, never a draw receipt.
    pub(crate) fn is_prepared(&self, view: RetainedViewEntity) -> bool {
        self.prepared.contains(&view)
    }

    /// Explicit GPS selection never falls back to unbounded raster/sort work
    /// when its bounded workspace cannot be admitted or is still compiling.
    pub(crate) fn suppresses_per_cloud_pass(&self, view: RetainedViewEntity) -> bool {
        self.claimed.contains(&view)
    }
}

pub(crate) fn point_splatting_for_cloud(
    camera: Option<&GaussianPointSplattingSettings>,
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

/// Installed by [`crate::GaussianSplattingPlugin`]; inserting the settings
/// component on a Gaussian camera enables this backend for that view.
pub struct GaussianPointSplattingPlugin;

impl Plugin for GaussianPointSplattingPlugin {
    fn build(&self, app: &mut App) {
        budget::install(app);
        app.register_type::<GaussianPointSplattingSettings>()
            .add_plugins(ExtractComponentPlugin::<GaussianPointSplattingSettings>::default())
            .init_resource::<GaussianPointSplattingDiagnostics>()
            .add_systems(PostUpdate, enable_depth_sampling);
        load_internal_asset!(app, SAMPLING_SHADER, "sampling.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, TYPES_SHADER, "types.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, PROJECT_SHADER, "projection.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, WORK_SHADER, "work.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, COMPOSITE_SHADER, "composite.wgsl", Shader::from_wgsl);
        let diagnostics = app
            .world()
            .resource::<GaussianPointSplattingDiagnostics>()
            .clone();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(diagnostics)
                .init_resource::<PointSplattingPipelineReadiness>()
                .init_gpu_resource::<gpu::PointViews>()
                .add_systems(RenderStartup, gpu::reset_shared_state)
                .add_systems(
                    Render,
                    gpu::prepare
                        .in_set(RenderSystems::PrepareResources)
                        .in_set(PointSplattingPrepare)
                        .after(super::lod::LodCompactionPrepare)
                        .after(super::traversal::GpuLodTraversalPrepare),
                )
                .add_systems(
                    Render,
                    gpu::collect
                        .in_set(RenderSystems::Cleanup)
                        .after(RenderSystems::Render),
                )
                .add_systems(
                    Core3d,
                    gpu::render
                        .in_set(Core3dSystems::MainPass)
                        .in_set(PointSplattingRender)
                        .after(super::traversal::GpuLodTraversalRender)
                        .after(main_opaque_pass_3d)
                        .before(main_transparent_pass_3d),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                RenderStartup,
                init_gpu_resource::<pipeline::PointPipeline>.after(super::CloudPipelineReady),
            );
        }
    }
}

fn enable_depth_sampling(mut cameras: Query<&mut Camera3d, With<GaussianPointSplattingSettings>>) {
    use bevy::render::render_resource::TextureUsages;
    for mut camera in &mut cameras {
        let usage = TextureUsages::from(camera.depth_texture_usages);
        if !usage.contains(TextureUsages::TEXTURE_BINDING) {
            camera.depth_texture_usages = (usage | TextureUsages::TEXTURE_BINDING).into();
        }
    }
}
