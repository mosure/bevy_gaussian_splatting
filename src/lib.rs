use bevy::prelude::*;

pub use camera::GaussianCamera;

pub use gaussian::{
    packed::Gaussian,
    cloud::{
        Cloud,
        CloudHandle,
    },
    rand::random_gaussians,
    settings::{
        RasterizeMode,
        CloudSettings,
        GaussianMode,
    },
};

pub use material::spherical_harmonics::SphericalHarmonicCoefficients;

use io::loader::CloudLoader;

pub mod camera;
pub mod gaussian;
pub mod io;
pub mod material;
pub mod math;
pub mod morph;
pub mod query;
pub mod render;
pub mod sort;
pub mod stream;
pub mod utils;

#[cfg(feature = "noise")]
pub mod noise;


pub struct GaussianSplattingPlugin;

impl Plugin for GaussianSplattingPlugin {
    fn build(&self, app: &mut App) {
        // TODO: allow hot reloading of Cloud handle through inspector UI
        app.register_type::<SphericalHarmonicCoefficients>();

        app.init_asset_loader::<CloudLoader>();

        app.add_plugins((
            camera::GaussianCameraPlugin,
            gaussian::cloud::CloudPlugin,
            render::RenderPipelinePlugin,
            material::MaterialPlugin,
            query::QueryPlugin,
        ));

        #[cfg(feature = "noise")]
        app.add_plugins(noise::NoisePlugin);
    }
}
