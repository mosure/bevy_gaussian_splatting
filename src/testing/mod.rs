//! Deterministic fixtures and metrics shared by LoD unit, GPU, and render tests.

pub mod image_metrics;
mod lod_manifest;
pub mod lod_scenes;
#[cfg(feature = "lod")]
pub mod render_oracle;

pub use image_metrics::{
    BoundaryBandMetrics, BoundaryMetricsError, ImageMetrics, ImageMetricsError,
    SpatialResidualMetrics, TemporalResidualMetrics, compare_linear_rgba,
    compare_node_boundary_bands, compare_temporal_deltas, compare_temporal_second_differences,
};
pub use lod_manifest::upgrade_manifest_to_synthetic_abi16_lifecycle_fixture;
pub use lod_scenes::{
    LodProjection, LodScenePattern, LodTestCamera, LodTestGaussian, LodTestScene, VirtualCityScene,
    all_small_lod_scenes,
};
#[cfg(feature = "lod")]
pub use render_oracle::{
    LodAttributedImage, LodOracleSupport, LodQualitySample, LodRenderOracleError,
    gather_frontier_gaussians, gather_frontier_gaussians_with_nodes, render_flat_linear_gaussians,
    render_linear_gaussians, render_linear_gaussians_with_nodes,
    render_lod_linear_gaussians_with_nodes, render_production_flat_linear_gaussians,
    render_production_flat_linear_gaussians_with_owners, render_production_lod_linear_gaussians,
    render_production_lod_linear_gaussians_with_nodes, render_quality_sweep,
};
