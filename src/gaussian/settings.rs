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

#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct GaussianCloudSettings {
    pub aabb: bool,
    pub global_scale: f32,
    pub global_transform: GlobalTransform,
    pub visualize_bounding_box: bool,
    pub visualize_depth: bool,
    pub sort_mode: SortMode,
    pub draw_mode: GaussianCloudDrawMode,
}

impl Default for GaussianCloudSettings {
    fn default() -> Self {
        Self {
            aabb: false,
            global_scale: 2.0,
            global_transform: Transform::IDENTITY.into(),
            visualize_bounding_box: false,
            visualize_depth: false,
            sort_mode: SortMode::default(),
            draw_mode: GaussianCloudDrawMode::default(),
        }
    }
}
