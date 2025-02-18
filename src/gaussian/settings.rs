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
pub enum DrawMode {
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
    Gaussian2d,
    #[default]
    Gaussian3d,
    Gaussian4d,
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
pub enum PlaybackMode {
    #[default]
    Loop,
    Once,
    Sin,
    Still,
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
pub enum RasterizeMode {
    Classification,
    #[default]
    Color,
    Depth,
    OpticalFlow,
    Normal,
    Velocity,
}


// TODO: breakdown into components
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct CloudSettings {
    pub aabb: bool,
    pub global_opacity: f32,
    pub global_scale: f32,
    pub opacity_adaptive_radius: bool,
    pub visualize_bounding_box: bool,
    pub sort_mode: SortMode,
    pub draw_mode: DrawMode,
    pub gaussian_mode: GaussianMode,
    pub playback_mode: PlaybackMode,
    pub rasterize_mode: RasterizeMode,
    pub num_classes: usize,
    pub time: f32,
    pub time_scale: f32,
    pub time_start: f32,
    pub time_stop: f32,
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self {
            aabb: false,
            global_opacity: 1.0,
            global_scale: 1.0,
            opacity_adaptive_radius: true,
            visualize_bounding_box: false,
            sort_mode: SortMode::default(),
            draw_mode: DrawMode::default(),
            gaussian_mode: GaussianMode::default(),
            rasterize_mode: RasterizeMode::default(),
            num_classes: 1,
            playback_mode: PlaybackMode::default(),
            time: 0.0,
            time_scale: 1.0,
            time_start: 0.0,
            time_stop: 1.0,
        }
    }
}


#[derive(Default)]
pub struct SettingsPlugin;
impl Plugin for SettingsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<CloudSettings>();

        app.add_systems(
            Update,
            (
                playback_update,
            )
        );
    }
}


fn playback_update(
    time: Res<Time>,
    mut query: Query<(&mut CloudSettings,)>,
) {
    for (mut settings,) in query.iter_mut() {
        if settings.time_scale == 0.0 {
            continue;
        }

        // bail condition
        match settings.playback_mode {
            PlaybackMode::Loop => {}
            PlaybackMode::Once => {
                if settings.time >= settings.time_stop {
                    continue;
                }
            }
            PlaybackMode::Sin => {}
            PlaybackMode::Still => {
                continue;
            }
        }

        // forward condition
        match settings.playback_mode {
            PlaybackMode::Loop | PlaybackMode::Once => {
                settings.time += time.delta_secs() * settings.time_scale;
            }
            PlaybackMode::Sin => {
                let theta = settings.time_scale * time.elapsed_secs();
                let y = (theta * 2.0 * std::f32::consts::PI).sin();
                settings.time = settings.time_start + (settings.time_stop - settings.time_start) * (y + 1.0) / 2.0;
            }
            PlaybackMode::Still => {}
        }

        // reset condition
        match settings.playback_mode {
            PlaybackMode::Loop => {
                if settings.time > settings.time_stop {
                    settings.time = settings.time_start;
                }
            }
            PlaybackMode::Once => {}
            PlaybackMode::Sin => {}
            PlaybackMode::Still => {}
        }
    }
}
