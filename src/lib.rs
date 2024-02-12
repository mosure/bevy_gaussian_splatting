use bevy::prelude::*;

pub use gaussian::{
    packed::Gaussian,
    cloud::Cloud,
    rand::random_gaussians,
    settings::GaussianCloudSettings,
};

pub use material::spherical_harmonics::SphericalHarmonicCoefficients;

use io::loader::GaussianCloudLoader;

use render::RenderPipelinePlugin;

pub mod gaussian;
pub mod io;
pub mod material;
pub mod math;
pub mod morph;
pub mod query;
pub mod render;
pub mod sort;
pub mod utils;

#[cfg(feature = "noise")]
pub mod noise;


#[derive(Bundle, Default, Reflect)]
pub struct GaussianSplattingBundle {
    pub settings: GaussianCloudSettings,
    pub cloud: Handle<Cloud>,
    pub visibility: Visibility,
}


#[derive(Component, Default)]
struct GaussianSplattingCamera;
// TODO: filter camera 3D entities

pub struct GaussianSplattingPlugin;

impl Plugin for GaussianSplattingPlugin {
    fn build(&self, app: &mut App) {
        // TODO: allow hot reloading of GaussianCloud handle through inspector UI
        app.register_type::<SphericalHarmonicCoefficients>();
        app.register_type::<Cloud>();
        app.init_asset::<Cloud>();
        app.register_asset_reflect::<Cloud>();

        app.init_asset_loader::<GaussianCloudLoader>();

        app.register_type::<GaussianCloudSettings>();
        app.register_type::<GaussianSplattingBundle>();

        app.add_plugins((
            RenderPipelinePlugin,
            material::MaterialPlugin,
            query::QueryPlugin,
        ));

        #[cfg(feature = "noise")]
        app.add_plugins(noise::NoisePlugin);
    }
}
