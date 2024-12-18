use bevy::prelude::*;

pub use camera::GaussianCamera;

pub use gaussian::{
    packed::Gaussian,
    cloud::{
        GaussianCloud,
        GaussianCloudHandle,
    },
    rand::random_gaussians,
    settings::{
        GaussianCloudRasterize,
        GaussianCloudSettings,
        GaussianMode,
    },
};

pub use material::spherical_harmonics::SphericalHarmonicCoefficients;

use io::loader::GaussianCloudLoader;

pub mod camera;
pub mod gaussian;
pub mod io;
pub mod material;
pub mod morph;
pub mod query;
pub mod render;
pub mod sort;
pub mod utils;

#[cfg(feature = "noise")]
pub mod noise;


pub struct GaussianSplattingPlugin;

impl Plugin for GaussianSplattingPlugin {
    fn build(&self, app: &mut App) {
        // TODO: allow hot reloading of GaussianCloud handle through inspector UI
        app.register_type::<SphericalHarmonicCoefficients>();
        app.register_type::<GaussianCloud>();
        app.register_type::<GaussianCloudHandle>();
        app.init_asset::<GaussianCloud>();
        app.register_asset_reflect::<GaussianCloud>();

        app.init_asset_loader::<GaussianCloudLoader>();

        app.register_type::<GaussianCloudSettings>();

        app.add_plugins((
            camera::GaussianCameraPlugin,
            gaussian::cloud::GaussianCloudPlugin,
            render::RenderPipelinePlugin,
            material::MaterialPlugin,
            query::QueryPlugin,
        ));

        #[cfg(feature = "noise")]
        app.add_plugins(noise::NoisePlugin);
    }
}
