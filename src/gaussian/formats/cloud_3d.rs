use rand::{
    seq::SliceRandom,
    Rng,
};

use bevy_interleave::prelude::*;

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
        packed::{Gaussian3d, PlanarGaussian3d},
        settings::CloudSettings,
    },
    material::spherical_harmonics::{
        HALF_SH_COEFF_COUNT,
        SH_COEFF_COUNT,
        SphericalHarmonicCoefficients,
    },
};

// #[allow(unused_imports)]
// #[cfg(feature = "f16")]
// use crate::gaussian::f16::{
//     Covariance3dOpacityPacked128,
//     RotationScaleOpacityPacked128,
//     pack_f32s_to_u32,
// };


// #[cfg(feature = "f16")]
// #[derive(
//     Debug,
//     Default,
//     PartialEq,
//     Reflect,
//     Serialize,
//     Deserialize,
// )]
// pub struct Cloud3d {
//     pub position_visibility: Vec<PositionVisibility>,

//     pub spherical_harmonic: Vec<SphericalHarmonicCoefficients>,

//     #[cfg(not(feature = "precompute_covariance_3d"))]
//     pub rotation_scale_opacity_packed128: Vec<RotationScaleOpacityPacked128>,

//     #[cfg(feature = "precompute_covariance_3d")]
//     pub covariance_3d_opacity_packed128: Vec<Covariance3dOpacityPacked128>,
// }


impl CommonCloud for PlanarGaussian3d {
    type PackedType = Gaussian3d;

    fn visibility(&self, index: usize) -> f32 {
        self.position_visibility[index].visibility
    }

    fn visibility_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.position_visibility[index].visibility
    }

    fn position_iter(&self) -> PositionIter<'_> {
        PositionIter::new(&self.position_visibility)
    }

    #[cfg(feature = "sort_rayon")]
    fn position_par_iter(&self) -> PositionParIter<'_> {
        PositionParIter::new(&self.position_visibility)
    }
}

impl FromIterator<Gaussian3d> for PlanarGaussian3d {
    fn from_iter<I: IntoIterator<Item = Gaussian3d>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<Gaussian3d>>().into()
    }
}

impl From<Vec<Gaussian3d>> for PlanarGaussian3d {
    fn from(packed: Vec<Gaussian3d>) -> Self {
        Self::from_interleaved(packed)
    }
}


impl TestCloud for PlanarGaussian3d {
    fn test_model() -> Self {
        let mut rng = rand::thread_rng();

        let origin = Gaussian3d {
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
                    // #[cfg(feature = "f16")]
                    // {
                    //     let mut coefficients = [0_u32; HALF_SH_COEFF_COUNT];

                    //     for coefficient in coefficients.iter_mut() {
                    //         let upper = rng.gen_range(-1.0..1.0);
                    //         let lower = rng.gen_range(-1.0..1.0);

                    //         *coefficient = pack_f32s_to_u32(upper, lower);
                    //     }

                    //     coefficients
                    // }

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
        let mut gaussians: Vec<Gaussian3d> = Vec::new();

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


// TODO: attempt iter() on the Planar trait
impl PlanarGaussian3d {
    pub fn iter(&self) -> impl Iterator<Item=Gaussian3d> + '_ {
        self.position_visibility.iter()
            .zip(self.spherical_harmonic.iter())
            .zip(self.rotation.iter())
            .zip(self.scale_opacity.iter())
            .map(|(((position_visibility, spherical_harmonic), rotation), scale_opacity)| {
                Gaussian3d {
                    position_visibility: *position_visibility,
                    spherical_harmonic: *spherical_harmonic,

                    rotation: *rotation,
                    scale_opacity: *scale_opacity,
                }
            })
    }
}
