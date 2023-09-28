use std::io::{
    BufReader,
    Cursor,
};

use bevy::{
    prelude::*,
    asset::{
        AssetLoader,
        LoadContext,
        LoadedAsset,
    },
    reflect::TypeUuid,
    utils::BoxedFuture,
};

use crate::ply::parse_ply;


#[derive(Clone, Debug, Default, Reflect)]
pub struct AnisotropicCovariance {
    pub mean: Vec3,
    pub covariance: Mat3,
}

const fn num_sh_coefficients(degree: usize) -> usize {
    if degree == 0 {
        1
    } else {
        2 * degree + 1 + num_sh_coefficients(degree - 1)
    }
}
const SH_DEGREE: usize = 3;
pub const SH_COEFF_COUNT: usize = num_sh_coefficients(SH_DEGREE) * 3;
#[derive(Clone, Debug, Reflect)]
pub struct SphericalHarmonicCoefficients {
    pub coefficients: [Vec3; SH_COEFF_COUNT],
}
impl Default for SphericalHarmonicCoefficients {
    fn default() -> Self {
        Self {
            coefficients: [Vec3::ZERO; SH_COEFF_COUNT],
        }
    }
}

#[derive(Clone, Debug, Default, Reflect)]
pub struct Gaussian {
    pub normal: Vec3,
    pub opacity: f32,
    pub transform: Transform,
    pub anisotropic_covariance: AnisotropicCovariance,
    pub spherical_harmonic: SphericalHarmonicCoefficients,
}

#[derive(Clone, Debug, Reflect, TypeUuid)]
#[uuid = "ac2f08eb-bc32-aabb-ff21-51571ea332d5"]
pub struct GaussianCloud(Vec<Gaussian>);


#[derive(Default)]
pub struct GaussianCloudLoader;

impl AssetLoader for GaussianCloudLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), bevy::asset::Error>> {
        Box::pin(async move {
            let cursor = Cursor::new(bytes);
            let mut f = BufReader::new(cursor);

            let ply_cloud = parse_ply(&mut f)?;
            let cloud = GaussianCloud(ply_cloud);

            println!("loaded {} gaussians", cloud.0.len());
            println!("first gaussian: {:?}", cloud.0[1000]);

            load_context.set_default_asset(LoadedAsset::new(cloud));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["ply"]
    }
}
