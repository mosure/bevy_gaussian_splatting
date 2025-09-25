use bevy::prelude::*;
use bevy_args::{Deserialize, Parser, Serialize};

use crate::gaussian::settings::{GaussianMode, PlaybackMode, RasterizeMode};

#[derive(Debug, Resource, Serialize, Deserialize, Parser)]
#[command(about = "bevy_gaussian_splatting viewer", version, long_about = None)]
pub struct GaussianSplattingViewer {
    #[arg(long, default_value = "true")]
    pub editor: bool,

    #[arg(long, default_value = "true")]
    pub press_esc_close: bool,

    #[arg(long, default_value = "true")]
    pub press_s_screenshot: bool,

    #[arg(long, default_value = "false")]
    pub show_axes: bool,

    #[arg(long, default_value = "true")]
    pub show_fps: bool,

    #[arg(long, default_value = "1920.0")]
    pub width: f32,

    #[arg(long, default_value = "1080.0")]
    pub height: f32,

    #[arg(long, default_value = "bevy_gaussian_splatting")]
    pub name: String,

    #[arg(long, default_value = "1")]
    pub msaa_samples: u8,

    #[arg(long, default_value = None, help = "input file path (or url/base64_url if web_asset feature is enabled)")]
    pub input_cloud: Option<String>,

    #[arg(
        long,
        default_value = None,
        help = "secondary input file used when morph_interpolate is enabled",
    )]
    pub input_cloud_target: Option<String>,

    #[arg(long, default_value = None, help = "input file path (or url/base64_url if web_asset feature is enabled)")]
    pub input_scene: Option<String>,

    #[arg(long, default_value = "0")]
    pub gaussian_count: usize,

    #[arg(long, value_enum, default_value_t = GaussianMode::Gaussian3d)]
    pub gaussian_mode: GaussianMode,

    #[arg(long, value_enum, default_value_t = PlaybackMode::Still)]
    pub playback_mode: PlaybackMode,

    #[arg(long, value_enum, default_value_t = RasterizeMode::Color)]
    pub rasterization_mode: RasterizeMode,

    #[arg(long, default_value = "0")]
    pub particle_count: usize,
}

impl Default for GaussianSplattingViewer {
    fn default() -> GaussianSplattingViewer {
        GaussianSplattingViewer {
            editor: true,
            press_esc_close: true,
            press_s_screenshot: true,
            show_axes: false,
            show_fps: true,
            width: 1920.0,
            height: 1080.0,
            name: "bevy_gaussian_splatting".to_string(),
            msaa_samples: 1,
            input_cloud: None,
            input_cloud_target: None,
            input_scene: None,
            gaussian_count: 0,
            gaussian_mode: GaussianMode::Gaussian3d,
            playback_mode: PlaybackMode::Still,
            rasterization_mode: RasterizeMode::Color,
            particle_count: 0,
        }
    }
}

pub fn setup_hooks() {
    #[cfg(debug_assertions)]
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
    }
}

pub fn log(_msg: &str) {
    #[cfg(debug_assertions)]
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::log_1(&_msg.into());
    }
    #[cfg(debug_assertions)]
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("{_msg}");
    }
}
