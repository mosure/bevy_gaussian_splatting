use bevy::prelude::*;
use bevy_args::{
    Deserialize,
    Parser,
    Serialize,
};

use crate::gaussian::settings::GaussianMode;


#[derive(
    Debug,
    Resource,
    Serialize,
    Deserialize,
    Parser,
)]
#[command(about = "bevy_gaussian_splatting viewer", version, long_about = None)]
pub struct GaussianSplattingViewer {
    #[arg(long, default_value = "true")]
    pub editor: bool,

    #[arg(long, default_value = "true")]
    pub press_esc_close: bool,

    #[arg(long, default_value = "true")]
    pub press_s_screenshot: bool,

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

    #[arg(long, default_value = "")]
    pub input_file: String,

    #[arg(long, default_value = "0")]
    pub gaussian_count: usize,

    #[arg(long, value_enum, default_value_t = GaussianMode::Gaussian3d)]
    pub gaussian_mode: GaussianMode,

    #[arg(long, default_value = "0")]
    pub particle_count: usize,
}

impl Default for GaussianSplattingViewer {
    fn default() -> GaussianSplattingViewer {
        GaussianSplattingViewer {
            editor: true,
            press_esc_close: true,
            press_s_screenshot: true,
            show_fps: true,
            width: 1920.0,
            height: 1080.0,
            name: "bevy_gaussian_splatting".to_string(),
            msaa_samples: 1,
            input_file: "".to_string(),
            gaussian_count: 0,
            gaussian_mode: GaussianMode::Gaussian3d,
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
        println!("{}", _msg);
    }
}
