use rand::{
    seq::SliceRandom,
    Rng,
};
use std::iter::FromIterator;

use bevy::prelude::*;
use serde::{
    Deserialize,
    Serialize,
};

#[cfg(feature = "sort_rayon")]
use rayon::prelude::*;

#[allow(unused_imports)]
use crate::{
    gaussian::{
        cloud::{
            Cloud,
            GaussianCloud,
        },
        f32::{
            Covariance3dOpacity,
            Position,
            PositionVisibility,
            Rotation,
            ScaleOpacity,
        },
        packed::Gaussian4d,
    },
    material::spherical_harmonics::{
        HALF_SH_COEFF_COUNT,
        SH_COEFF_COUNT,
        SphericalHarmonicCoefficients,
    },
};

#[allow(unused_imports)]
#[cfg(feature = "f16")]
use crate::gaussian::f16::{
    Covariance3dOpacityPacked128,
    RotationScaleOpacityPacked128,
    pack_f32s_to_u32,
};



#[cfg(feature = "f16")]
#[derive(
    Debug,
    Default,
    PartialEq,
    Reflect,
    Serialize,
    Deserialize,
)]
pub struct GaussianCloud4d {
    pub position_visibility: Vec<PositionVisibility>,

    pub spherical_harmonic: Vec<SphericalHarmonicCoefficients>,

    #[cfg(not(feature = "precompute_covariance_3d"))]
    pub rotation_scale_opacity_packed128: Vec<RotationScaleOpacityPacked128>,

    #[cfg(feature = "precompute_covariance_3d")]
    pub covariance_3d_opacity_packed128: Vec<Covariance3dOpacityPacked128>,
}

#[cfg(feature = "f32")]
#[derive(
    Asset,
    Clone,
    Debug,
    Default,
    PartialEq,
    Reflect,
    TypeUuid,
    Serialize,
    Deserialize,
)]
#[uuid = "ac2f08eb-bc32-aabb-ff21-51571ea332d5"]
pub struct GaussianCloud4d {
    pub position_visibility: Vec<PositionVisibility>,

    pub spherical_harmonic: Vec<SphericalHarmonicCoefficients>,

    #[cfg(feature = "precompute_covariance_3d")]
    pub covariance_3d: Vec<Covariance3dOpacity>,

    #[cfg(not(feature = "precompute_covariance_3d"))]
    pub rotation: Vec<Rotation>,
    #[cfg(not(feature = "precompute_covariance_3d"))]
    pub scale_opacity: Vec<ScaleOpacity>,
}

impl GaussianCloud<Gaussian4d> for GaussianCloud4d {
    fn is_empty(&self) -> bool {
        self.position_visibility.is_empty()
    }

    fn len(&self) -> usize {
        self.position_visibility.len()
    }

    fn len_sqrt_ceil(&self) -> usize {
        (self.len() as f32).sqrt().ceil() as usize
    }

    fn square_len(&self) -> usize {
        self.len_sqrt_ceil().pow(2)
    }


    #[cfg(all(
        not(feature = "precompute_covariance_3d"),
        feature = "f16",
    ))]
    fn packed(&self, index: usize) -> Gaussian4d {
        let rso = self.rotation_scale_opacity_packed128[index];

        let rotation = rso.rotation();
        let scale_opacity = rso.scale_opacity();

        Gaussian4d {
            isomorphic_rotations: todo!(),
            position_opacity: todo!(),
            scale: todo!(),
            spherindrical_harmonic: todo!(),
        }
    }

    #[cfg(feature = "f32")]
    fn packed(&self, index: usize) -> Gaussian4d {
        Gaussian4d {
            position_visibility: self.position_visibility[index],
            spherical_harmonic: self.spherical_harmonic[index],
            rotation: self.rotation[index],
            scale_opacity: self.scale_opacity[index],
        }
    }

    #[cfg(all(
        not(feature = "precompute_covariance_3d"),
        feature = "f16",
    ))]
    fn iter(&self) -> std::vec::IntoIter<Gaussian4d> {
        self.position_visibility.iter()
            .zip(self.spherical_harmonic.iter())
            .zip(self.rotation_scale_opacity_packed128.iter())
            .map(|((position_visibility, spherical_harmonic), rotation_scale_opacity)| {
                Gaussian4d {
                    isomorphic_rotations: todo!(),
                    position_opacity: todo!(),
                    scale: todo!(),
                    spherindrical_harmonic: todo!(),
                }
            })
    }

    #[cfg(feature = "f32")]
    fn iter(&self) -> dyn Iterator<Item=Gaussian4d> {
        self.position_visibility.iter()
            .zip(self.spherical_harmonic.iter())
            .zip(self.rotation.iter())
            .zip(self.scale_opacity.iter())
            .map(|(((position_visibility, spherical_harmonic), rotation), scale_opacity)| {
                Gaussian {
                    position_visibility: *position_visibility,
                    spherical_harmonic: *spherical_harmonic,

                    rotation: *rotation,
                    scale_opacity: *scale_opacity,
                }
            })
    }

    fn to_packed(&self) -> Vec<Gaussian4d> {
        let mut gaussians = Vec::with_capacity(self.len());

        for index in 0..self.len() {
            gaussians.push(self.packed(index));
        }

        gaussians
    }

    fn test_model() -> Self {
        let mut rng = rand::thread_rng();

        let origin = Gaussian4d {
            isomorphic_rotations: todo!(),
            position_opacity: todo!(),
            scale: todo!(),
            spherindrical_harmonic: todo!(),
        };
        let mut gaussians: Vec<Gaussian4d> = Vec::new();

        for &x in [-0.5, 0.5].iter() {
            for &y in [-0.5, 0.5].iter() {
                for &z in [-0.5, 0.5].iter() {
                    let mut g = origin;
                    g.position_opacity = [x, y, z, 0.5].into();
                    gaussians.push(g);

                    gaussians.last_mut().unwrap().spherindrical_harmonic.coefficients.shuffle(&mut rng);
                }
            }
        }

        gaussians.push(gaussians[0]);

        GaussianCloud4d::from_packed(gaussians)
    }
}

impl GaussianCloud4d {
    pub fn position(&self, index: usize) -> &[f32; 3] {
        &self.position_visibility[index].position
    }

    pub fn position_mut(&mut self, index: usize) -> &mut [f32; 3] {
        &mut self.position_visibility[index].position
    }

    pub fn position_iter(&self) -> impl Iterator<Item = &Position> + '_ {
        self.position_visibility.iter()
            .map(|position_visibility| &position_visibility.position)
    }

    #[cfg(feature = "sort_rayon")]
    pub fn position_par_iter(&self) -> impl IndexedParallelIterator<Item = &Position> {
        self.position_visibility.par_iter()
            .map(|position_visibility| &position_visibility.position)
    }


    pub fn visibility(&self, index: usize) -> f32 {
        self.position_visibility[index].visibility
    }

    pub fn visibility_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.position_visibility[index].visibility
    }


    pub fn spherical_harmonic(&self, index: usize) -> &SphericalHarmonicCoefficients {
        &self.spherical_harmonic[index]
    }

    pub fn spherical_harmonic_mut(&mut self, index: usize) -> &mut SphericalHarmonicCoefficients {
        &mut self.spherical_harmonic[index]
    }

    pub fn resize_to_square(&mut self) {
        #[cfg(all(feature = "buffer_texture", feature = "f16"))]
        {
            self.position_visibility.resize(self.square_len(), PositionVisibility::default());
            self.spherical_harmonic.resize(self.square_len(), SphericalHarmonicCoefficients::default());

            #[cfg(feature = "precompute_covariance_3d")]
            self.covariance_3d_opacity_packed128.resize(self.square_len(), Covariance3dOpacityPacked128::default());
            #[cfg(not(feature = "precompute_covariance_3d"))]
            self.rotation_scale_opacity_packed128.resize(self.square_len(), RotationScaleOpacityPacked128::default());
        }

        #[cfg(all(feature = "buffer_texture", feature = "f32"))]
        {
            self.position_visibility.resize(self.square_len(), PositionVisibility::default());
            self.spherical_harmonic.resize(self.square_len(), SphericalHarmonicCoefficients::default());
            self.rotation.resize(self.square_len(), Rotation::default());
            self.scale_opacity.resize(self.square_len(), ScaleOpacity::default());
            self.covariance_3d.resize(self.square_len(), Covariance3dOpacity::default());
        }
    }
}


impl GaussianCloud4d {
    #[cfg(feature = "f16")]
    pub fn subset(&self, indicies: &[usize]) -> Self {
        let mut position_visibility = Vec::with_capacity(indicies.len());
        let mut spherical_harmonic = Vec::with_capacity(indicies.len());

        #[cfg(feature = "precompute_covariance_3d")]
        let mut covariance_3d_opacity_packed128 = Vec::with_capacity(indicies.len());

        #[cfg(not(feature = "precompute_covariance_3d"))]
        let mut rotation_scale_opacity_packed128 = Vec::with_capacity(indicies.len());

        for &index in indicies.iter() {
            position_visibility.push(self.position_visibility[index]);
            spherical_harmonic.push(self.spherical_harmonic[index]);

            #[cfg(feature = "precompute_covariance_3d")]
            covariance_3d_opacity_packed128.push(self.covariance_3d_opacity_packed128[index]);

            #[cfg(not(feature = "precompute_covariance_3d"))]
            rotation_scale_opacity_packed128.push(self.rotation_scale_opacity_packed128[index]);
        }

        Self {
            position_visibility,
            spherical_harmonic,

            #[cfg(feature = "precompute_covariance_3d")]
            covariance_3d_opacity_packed128,
            #[cfg(not(feature = "precompute_covariance_3d"))]
            rotation_scale_opacity_packed128,
        }
    }

    #[cfg(feature = "f32")]
    pub fn subset(&self, indicies: &[usize]) -> Self {
        let mut position_visibility = Vec::with_capacity(indicies.len());
        let mut spherical_harmonic = Vec::with_capacity(indicies.len());
        let mut rotation = Vec::with_capacity(indicies.len());
        let mut scale_opacity = Vec::with_capacity(indicies.len());

        for &index in indicies.iter() {
            position_visibility.push(self.position_visibility[index]);
            spherical_harmonic.push(self.spherical_harmonic[index]);
            rotation.push(self.rotation[index]);
            scale_opacity.push(self.scale_opacity[index]);
        }

        Self {
            position_visibility,
            spherical_harmonic,
            rotation,
            scale_opacity,
        }
    }


    #[cfg(feature = "f16")]
    fn from_packed(gaussians: Vec<Gaussian4d>) -> Self {
        let mut position_visibility = Vec::with_capacity(gaussians.len());
        let mut spherical_harmonic = Vec::with_capacity(gaussians.len());

        #[cfg(feature = "precompute_covariance_3d")]
        let mut covariance_3d_opacity_packed128 = Vec::with_capacity(gaussians.len());

        #[cfg(not(feature = "precompute_covariance_3d"))]
        let mut rotation_scale_opacity_packed128 = Vec::with_capacity(gaussians.len());

        for gaussian in gaussians {
            position_visibility.push(gaussian.position_visibility);
            spherical_harmonic.push(gaussian.spherical_harmonic);

            #[cfg(feature = "precompute_covariance_3d")]
            covariance_3d_opacity_packed128.push(Covariance3dOpacityPacked128::from_gaussian(&gaussian));

            #[cfg(not(feature = "precompute_covariance_3d"))]
            rotation_scale_opacity_packed128.push(RotationScaleOpacityPacked128::from_gaussian(&gaussian));
        }

        #[allow(unused_mut)]
        let mut cloud = GaussianCloud4d {
            position_visibility,
            spherical_harmonic,

            #[cfg(feature = "precompute_covariance_3d")]
            covariance_3d_opacity_packed128,
            #[cfg(not(feature = "precompute_covariance_3d"))]
            rotation_scale_opacity_packed128,
        };

        cloud.resize_to_square();

        cloud
    }

    #[cfg(feature = "f32")]
    fn from_packed(gaussians: Vec<Gaussian>) -> Self {
        let mut position_visibility = Vec::with_capacity(gaussians.len());
        let mut spherical_harmonic = Vec::with_capacity(gaussians.len());
        let mut rotation = Vec::with_capacity(gaussians.len());
        let mut scale_opacity = Vec::with_capacity(gaussians.len());

        for gaussian in gaussians {
            position_visibility.push(gaussian.position_visibility);
            spherical_harmonic.push(gaussian.spherical_harmonic);

            rotation.push(gaussian.rotation);
            scale_opacity.push(gaussian.scale_opacity);
        }

        Self {
            position_visibility,
            spherical_harmonic,
            rotation,
            scale_opacity,
        }
    }
}

impl FromIterator<Gaussian4d> for GaussianCloud4d {
    fn from_iter<I: IntoIterator<Item=Gaussian4d>>(iter: I) -> Self {
        let gaussians = iter.into_iter().collect::<Vec<Gaussian4d>>();
        GaussianCloud4d::from_packed(gaussians)
    }
}
