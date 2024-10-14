use bevy::prelude::*;

use crate::sort::SortMode;


#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    PartialEq,
    Reflect,
)]
pub enum GaussianCloudDrawMode {
    #[default]
    All,
    Selected,
    HighlightSelected,
}


#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    PartialEq,
    Reflect,
)]
pub enum GaussianMode {
    #[default]
    Gaussian3d,
    GaussianSurfel,
}


#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    PartialEq,
    Reflect,
)]
pub enum GaussianCloudRasterize {
    #[default]
    Color,
    Depth,
    Normal,
}


#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct GaussianCloudSettings {
    pub aabb: bool,
    pub global_scale: f32,
    pub transform: Transform,
    pub opacity_adaptive_radius: bool,
    pub visualize_bounding_box: bool,
    pub sort_mode: SortMode,
    pub draw_mode: GaussianCloudDrawMode,
    pub gaussian_mode: GaussianMode,
    pub rasterize_mode: GaussianCloudRasterize,
}

impl Default for GaussianCloudSettings {
    fn default() -> Self {
        Self {
            aabb: false,
            global_scale: 1.0,
            transform: Transform::IDENTITY,
            opacity_adaptive_radius: true,
            visualize_bounding_box: false,
            sort_mode: SortMode::default(),
            draw_mode: GaussianCloudDrawMode::default(),
            gaussian_mode: GaussianMode::default(),
            rasterize_mode: GaussianCloudRasterize::default(),
        }
    }
}
