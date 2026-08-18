//! Deterministic fixtures and metrics shared by LoD unit, GPU, and render tests.

pub mod image_metrics;
pub mod lod_scenes;
pub mod render_oracle;

pub use image_metrics::{ImageMetrics, ImageMetricsError, compare_linear_rgba};
pub use lod_scenes::{
    LodProjection, LodScenePattern, LodTestCamera, LodTestGaussian, LodTestScene, VirtualCityScene,
    all_small_lod_scenes,
};
pub use render_oracle::{
    LodQualitySample, LodRenderOracleError, gather_frontier_gaussians, render_linear_gaussians,
    render_quality_sweep,
};
