#![allow(incomplete_features)]
#![feature(lazy_type_alias)]

use bevy::prelude::*;
use bevy_interleave::prelude::*;

pub use camera::GaussianCamera;

pub use gaussian::{
    packed::{
        Gaussian3d,
        Gaussian4d,
        PlanarGaussian3d,
        PlanarGaussian4d,
        PlanarGaussian3dHandle,
        PlanarGaussian4dHandle,
    },
    rand::random_gaussians_3d,
    rand::random_gaussians_4d,
    settings::{
        RasterizeMode,
        CloudSettings,
        GaussianMode,
    },
};

pub use material::spherical_harmonics::SphericalHarmonicCoefficients;

use io::loader::Gaussian3dLoader;

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

        app.init_asset_loader::<Gaussian3dLoader>();
        // TODO: Gaussian4dLoader asset loader

        app.register_type::<CloudSettings>();

        app.add_plugins((
            camera::GaussianCameraPlugin,
            gaussian::cloud::CloudPlugin::<Gaussian3d>::default(),
            gaussian::cloud::CloudPlugin::<Gaussian4d>::default(),
        ));

        // TODO: add half types
        app.add_plugins((
            PlanarStoragePlugin::<Gaussian3d>::default(),
            PlanarStoragePlugin::<Gaussian4d>::default(),
        ));

        app.add_plugins((
            render::RenderPipelinePlugin::<Gaussian3d>::default(),
            render::RenderPipelinePlugin::<Gaussian4d>::default(),
        ));

        app.add_plugins((
            material::MaterialPlugin,
            query::QueryPlugin,
        ));

        #[cfg(feature = "noise")]
        app.add_plugins(noise::NoisePlugin);
    }
}
