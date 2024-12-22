use rand::{
    seq::SliceRandom,
    Rng,
};

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
        f32::{
            Covariance3dOpacity,
            PositionVisibility,
            Rotation,
            ScaleOpacity,
        },
        interface::{
            CommonCloud,
            TestCloud,
        },
        iter::{
            PositionIter,
            PositionParIter,
        },
        packed::Gaussian,
        settings::CloudSettings,
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
pub struct Cloud3d {
    pub position_visibility: Vec<PositionVisibility>,

    pub spherical_harmonic: Vec<SphericalHarmonicCoefficients>,

    #[cfg(not(feature = "precompute_covariance_3d"))]
    pub rotation_scale_opacity_packed128: Vec<RotationScaleOpacityPacked128>,

    #[cfg(feature = "precompute_covariance_3d")]
    pub covariance_3d_opacity_packed128: Vec<Covariance3dOpacityPacked128>,
}

#[derive(
    Clone,
    Debug,
    Default,
    PartialEq,
    Reflect,
    Serialize,
    Deserialize,
)]
pub struct Cloud3d {
    pub position_visibility: Vec<PositionVisibility>,

    pub spherical_harmonic: Vec<SphericalHarmonicCoefficients>,

    #[cfg(feature = "precompute_covariance_3d")]
    pub covariance_3d: Vec<Covariance3dOpacity>,

    #[cfg(not(feature = "precompute_covariance_3d"))]
    pub rotation: Vec<Rotation>,
    #[cfg(not(feature = "precompute_covariance_3d"))]
    pub scale_opacity: Vec<ScaleOpacity>,
}

impl CommonCloud for Cloud3d {
    type PackedType = Gaussian;

    fn len(&self) -> usize {
        self.position_visibility.len()
    }

    #[cfg(feature = "f16")]
    fn subset(&self, indicies: &[usize]) -> Self {
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

    fn subset(&self, indicies: &[usize]) -> Self {
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

    fn from_packed(gaussians: Vec<Self::PackedType>) -> Self {
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

    fn visibility(&self, index: usize) -> f32 {
        self.position_visibility[index].visibility
    }

    fn visibility_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.position_visibility[index].visibility
    }

    fn resize_to_square(&mut self) {
        #[cfg(all(feature = "buffer_texture", feature = "f16"))]
        {
            self.position_visibility.resize(self.square_len(), PositionVisibility::default());
            self.spherical_harmonic.resize(self.square_len(), SphericalHarmonicCoefficients::default());

            #[cfg(feature = "precompute_covariance_3d")]
            self.covariance_3d_opacity_packed128.resize(self.square_len(), Covariance3dOpacityPacked128::default());
            #[cfg(not(feature = "precompute_covariance_3d"))]
            self.rotation_scale_opacity_packed128.resize(self.square_len(), RotationScaleOpacityPacked128::default());
        }

        #[cfg(all(feature = "buffer_texture"))]
        {
            self.position_visibility.resize(self.square_len(), PositionVisibility::default());
            self.spherical_harmonic.resize(self.square_len(), SphericalHarmonicCoefficients::default());
            self.rotation.resize(self.square_len(), Rotation::default());
            self.scale_opacity.resize(self.square_len(), ScaleOpacity::default());
            self.covariance_3d.resize(self.square_len(), Covariance3dOpacity::default());
        }
    }


    fn position_iter(&self) -> PositionIter<'_> {
        PositionIter::new(&self.position_visibility)
    }

    #[cfg(feature = "sort_rayon")]
    fn position_par_iter(&self) -> PositionParIter<'_> {
        PositionParIter::new(&self.position_visibility)
    }
}

impl FromIterator<Gaussian> for Cloud3d {
    fn from_iter<I: IntoIterator<Item = Gaussian>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<Gaussian>>().into()
    }
}

impl From<Vec<Gaussian>> for Cloud3d {
    fn from(packed: Vec<Gaussian>) -> Self {
        Self::from_packed(packed)
    }
}


impl TestCloud for Cloud3d {
    fn test_model() -> Self {
        let mut rng = rand::thread_rng();

        let origin = Gaussian {
            rotation: [
                1.0,
                0.0,
                0.0,
                0.0,
            ].into(),
            position_visibility: [
                0.0,
                0.0,
                0.0,
                1.0,
            ].into(),
            scale_opacity: [
                0.5,
                0.5,
                0.5,
                0.5,
            ].into(),
            spherical_harmonic: SphericalHarmonicCoefficients {
                coefficients: {
                    #[cfg(feature = "f16")]
                    {
                        let mut coefficients = [0_u32; HALF_SH_COEFF_COUNT];

                        for coefficient in coefficients.iter_mut() {
                            let upper = rng.gen_range(-1.0..1.0);
                            let lower = rng.gen_range(-1.0..1.0);

                            *coefficient = pack_f32s_to_u32(upper, lower);
                        }

                        coefficients
                    }

                    {
                        let mut coefficients = [0.0; SH_COEFF_COUNT];

                        for coefficient in coefficients.iter_mut() {
                            *coefficient = rng.gen_range(-1.0..1.0);
                        }

                        coefficients
                    }
                },
            },
        };
        let mut gaussians: Vec<Gaussian> = Vec::new();

        for &x in [-0.5, 0.5].iter() {
            for &y in [-0.5, 0.5].iter() {
                for &z in [-0.5, 0.5].iter() {
                    let mut g = origin;
                    g.position_visibility = [x, y, z, 1.0].into();
                    gaussians.push(g);

                    gaussians.last_mut().unwrap().spherical_harmonic.coefficients.shuffle(&mut rng);
                }
            }
        }

        gaussians.push(gaussians[0]);
        gaussians.into()
    }
}


impl Cloud3d {
    #[cfg(all(
        not(feature = "precompute_covariance_3d"),
        feature = "f16",
    ))]
    pub fn gaussian(&self, index: usize) -> Gaussian {
        let rso = self.rotation_scale_opacity_packed128[index];

        let rotation = rso.rotation();
        let scale_opacity = rso.scale_opacity();

        Gaussian {
            position_visibility: self.position_visibility[index],
            spherical_harmonic: self.spherical_harmonic[index],
            rotation,
            scale_opacity,
        }
    }

    pub fn gaussian(&self, index: usize) -> Gaussian {
        Gaussian {
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
    pub fn gaussian_iter(&self) -> impl Iterator<Item=Gaussian> + '_ {
        self.position_visibility.iter()
            .zip(self.spherical_harmonic.iter())
            .zip(self.rotation_scale_opacity_packed128.iter())
            .map(|((position_visibility, spherical_harmonic), rotation_scale_opacity)| {
                Gaussian {
                    position_visibility: *position_visibility,
                    spherical_harmonic: *spherical_harmonic,

                    rotation: rotation_scale_opacity.rotation(),
                    scale_opacity: rotation_scale_opacity.scale_opacity(),
                }
            })
    }

    pub fn gaussian_iter(&self) -> impl Iterator<Item=Gaussian> + '_ {
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
}
