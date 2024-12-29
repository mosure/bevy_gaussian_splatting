use bevy_interleave::prelude::*;

use crate::{
    gaussian::{
        interface::{
            CommonCloud,
            TestCloud,
        },
        iter::{
            PositionIter,
            PositionParIter,
        },
        packed::{Gaussian4d, PlanarGaussian4d},
    },
    random_gaussians_4d,
};

// TODO: quantize 4d representation
// #[derive(
//     Debug,
//     Default,
//     PartialEq,
//     Reflect,
//     Serialize,
//     Deserialize,
// )]
// pub struct HalfCloud4d {
//     pub isotropic_rotations: Vec<IsotropicRotations>,
//     pub position_visibility: Vec<PositionVisibility>,
//     pub scale_opacity: Vec<ScaleOpacity>,
//     pub spherindrical_harmonic: Vec<SpherindricalHarmonicCoefficients>,
//     pub timestamp_timescale: Vec<TimestampTimescale>,
// }

// impl CommonCloud for HalfCloud4d {
//     fn len(&self) -> usize {
//         self.position_visibility.len()
//     }

//     fn position_iter(&self) -> impl Iterator<Item = &Position> {
//         self.position_visibility.iter()
//             .map(|position_visibility| &position_visibility.position)
//     }

//     #[cfg(feature = "sort_rayon")]
//     fn position_par_iter(&self) -> impl IndexedParallelIterator<Item = &Position> + '_ {
//         self.position_visibility.par_iter()
//             .map(|position_visibility| &position_visibility.position)
//     }

//     fn subset(&self, indicies: &[usize]) -> Self {
//         let mut isotropic_rotations = Vec::with_capacity(indicies.len());
//         let mut position_visibility = Vec::with_capacity(indicies.len());
//         let mut scale_opacity = Vec::with_capacity(indicies.len());
//         let mut spherindrical_harmonic = Vec::with_capacity(indicies.len());
//         let mut timestamp_timescale = Vec::with_capacity(indicies.len());

//         for &index in indicies.iter() {
//             position_visibility.push(self.position_visibility[index]);
//             spherindrical_harmonic.push(self.spherindrical_harmonic[index]);
//             rotation.push(self.rotation[index]);
//             scale_opacity.push(self.scale_opacity[index]);
//             timestamp_timescale.push(self.timestamp_timescale[index]);
//         }

//         Self {
//             isotropic_rotations,
//             position_visibility,
//             spherindrical_harmonic,
//             scale_opacity,
//             timestamp_timescale,
//         }
//     }
// }

// impl TestCloud for HalfCloud4d {
//     fn test_model() -> Self {
//         let mut rng = rand::thread_rng();

//         let origin = Gaussian {
//             isotropic_rotations: [
//                 1.0,
//                 0.0,
//                 0.0,
//                 0.0,
//                 1.0,
//                 0.0,
//                 0.0,
//                 0.0,
//             ].into(),
//             position_visibility: [
//                 0.0,
//                 0.0,
//                 0.0,
//                 1.0,
//             ].into(),
//             scale_opacity: [
//                 0.5,
//                 0.5,
//                 0.5,
//                 0.5,
//             ].into(),
//             spherindrical_harmonic: SpherindricalHarmonicCoefficients {
//                 coefficients: {
//                     let mut coefficients = [0.0; SH_4D_COEFF_COUNT];

//                     for coefficient in coefficients.iter_mut() {
//                         *coefficient = rng.gen_range(-1.0..1.0);
//                     }

//                     coefficients
//                 },
//             },
//         };
//         let mut gaussians: Vec<Gaussian4d> = Vec::new();

//         for &x in [-0.5, 0.5].iter() {
//             for &y in [-0.5, 0.5].iter() {
//                 for &z in [-0.5, 0.5].iter() {
//                     let mut g = origin;
//                     g.position_visibility = [x, y, z, 0.5].into();
//                     gaussians.push(g);

//                     gaussians.last_mut().unwrap().spherindrical_harmonic.coefficients.shuffle(&mut rng);
//                 }
//             }
//         }

//         gaussians.push(gaussians[0]);

//         Cloud4d::from_packed(gaussians)
//     }
// }

// impl HalfCloud4d {
//     fn from_packed(gaussians: Vec<Gaussian4d>) -> Self {
//         let mut isotropic_rotations = Vec::with_capacity(gaussians.len());
//         let mut position_visibility = Vec::with_capacity(gaussians.len());
//         let mut scale_opacity = Vec::with_capacity(gaussians.len());
//         let mut spherindrical_harmonic = Vec::with_capacity(gaussians.len());
//         let mut timestamp_timescale = Vec::with_capacity(gaussians.len());

//         for gaussian in gaussians {
//             isotropic_rotations.push(gaussian.isotropic_rotations);
//             position_visibility.push(gaussian.position_visibility);
//             scale_opacity.push(gaussian.scale_opacity);
//             spherindrical_harmonic.push(gaussian.spherindrical_harmonic);
//             timestamp_timescale.push(gaussian.timestamp_timescale);
//         }

//         Self {
//             isotropic_rotations,
//             position_visibility,
//             scale_opacity,
//             spherindrical_harmonic,
//             timestamp_timescale,
//         }
//     }
// }

// impl FromIterator<Gaussian4d> for HalfCloud4d {
//     fn from_iter<I: IntoIterator<Item=Gaussian4d>>(iter: I) -> Self {
//         let gaussians = iter.into_iter().collect::<Vec<Gaussian4d>>();
//         HalfCloud4d::from_packed(gaussians)
//     }
// }


impl CommonCloud for PlanarGaussian4d {
    type PackedType = Gaussian4d;

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

impl FromIterator<Gaussian4d> for PlanarGaussian4d {
    fn from_iter<I: IntoIterator<Item = Gaussian4d>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<Gaussian4d>>().into()
    }
}

impl From<Vec<Gaussian4d>> for PlanarGaussian4d {
    fn from(packed: Vec<Gaussian4d>) -> Self {
        Self::from_interleaved(packed)
    }
}


impl TestCloud for PlanarGaussian4d {
    fn test_model() -> Self {
        random_gaussians_4d(512)
    }
}
