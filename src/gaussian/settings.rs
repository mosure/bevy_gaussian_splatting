use bevy::prelude::*;
use bevy_args::{
    Deserialize,
    Serialize,
    ValueEnum,
};

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
    Serialize,
    Deserialize,
    ValueEnum,
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


// TODO: breakdown into components
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct GaussianCloudSettings {
    pub aabb: bool,
    pub global_opacity: f32,
    pub global_scale: f32,
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
            global_opacity: 1.0,
            global_scale: 1.0,
            opacity_adaptive_radius: true,
            visualize_bounding_box: false,
            sort_mode: SortMode::default(),
            draw_mode: GaussianCloudDrawMode::default(),
            gaussian_mode: GaussianMode::default(),
            rasterize_mode: GaussianCloudRasterize::default(),
        }
    }
}
