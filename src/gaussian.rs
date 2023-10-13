use std::{
    io::{
        BufReader,
        Cursor,
    },
    marker::Copy,
};

use bevy::{
    prelude::*,
    asset::{
        AssetLoader,
        LoadContext,
        LoadedAsset,
    },
    reflect::{
        TypePath,
        TypeUuid,
    },
    render::render_resource::ShaderType,
    utils::BoxedFuture,
};
use bytemuck::{
    Pod,
    Zeroable,
};

use crate::ply::parse_ply;


const fn num_sh_coefficients(degree: usize) -> usize {
    if degree == 0 {
        1
    } else {
        2 * degree + 1 + num_sh_coefficients(degree - 1)
    }
}
const SH_DEGREE: usize = 3;
pub const MAX_SH_COEFF_COUNT: usize = num_sh_coefficients(SH_DEGREE) * 3;
#[derive(Clone, Copy, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct SphericalHarmonicCoefficients {
    pub coefficients: [f32; MAX_SH_COEFF_COUNT],
}
impl Default for SphericalHarmonicCoefficients {
    fn default() -> Self {
        Self {
            coefficients: [0.0; MAX_SH_COEFF_COUNT],
        }
    }
}

#[derive(Clone, Default, Copy, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct Gaussian {
    //pub anisotropic_covariance: AnisotropicCovariance,
    //pub normal: Vec3,
    pub rotation: [f32; 4],
    pub position: Vec3,
    pub scale: Vec3,
    pub opacity: f32,
    pub spherical_harmonic: SphericalHarmonicCoefficients,
    padding: f32,
}

#[derive(Clone, TypeUuid, TypePath)]
#[uuid = "ac2f08eb-bc32-aabb-ff21-51571ea332d5"]
pub struct GaussianCloud(pub Vec<Gaussian>);

impl GaussianCloud {
    pub fn test_model() -> Self {
        let origin = Gaussian {
            rotation: [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            position: Vec3::new(0.0, 0.0, 0.0),
            scale: Vec3::new(0.5, 0.5, 0.5),
            opacity: 0.8,
            spherical_harmonic: SphericalHarmonicCoefficients{
                coefficients: [
                    1.0, 0.0, 1.0,
                    0.0, 0.5, 0.0,
                    0.3, 0.2, 0.0,
                    0.4, 0.0, 0.2,
                    0.1, 0.0, 0.0,
                    0.0, 0.3, 0.3,
                    0.0, 1.0, 1.0,
                    0.3, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.3, 1.0,
                    0.5, 0.3, 0.0,
                    0.2, 0.3, 0.1,
                    0.6, 0.3, 0.1,
                    0.0, 0.3, 0.2,
                    0.0, 0.5, 0.3,
                    0.6, 0.1, 0.2,
                ],
            },
            padding: 0.0,
        };
        let mut cloud = GaussianCloud(Vec::new());

        for &x in [-1.0, 1.0].iter() {
            for &y in [-1.0, 1.0].iter() {
                for &z in [-1.0, 1.0].iter() {
                    let mut g = origin.clone();
                    g.position = Vec3::new(x, y, z);
                    cloud.0.push(g);
                }
            }
        }

        cloud
    }
}


#[derive(Component, Reflect, Clone)]
pub struct GaussianCloudSettings {
    pub global_scale: f32,
    pub global_transform: GlobalTransform,
}

impl Default for GaussianCloudSettings {
    fn default() -> Self {
        Self {
            global_scale: 1.0,
            global_transform: Transform::IDENTITY.into(),
        }
    }
}


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

            load_context.set_default_asset(LoadedAsset::new(cloud));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["ply"]
    }
}
