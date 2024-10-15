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
    pub global_opacity: f32,
    pub global_scale: f32,
    pub transform: Transform,
    pub visualize_bounding_box: bool,
    pub sort_mode: SortMode,
    pub draw_mode: GaussianCloudDrawMode,
    pub rasterize_mode: GaussianCloudRasterize,
}

impl Default for GaussianCloudSettings {
    fn default() -> Self {
        Self {
            aabb: false,
            global_opacity: 1.0,
            global_scale: 1.0,
            transform: Transform::IDENTITY,
            visualize_bounding_box: false,
            sort_mode: SortMode::default(),
            draw_mode: GaussianCloudDrawMode::default(),
            rasterize_mode: GaussianCloudRasterize::default(),
        }
    }
}
