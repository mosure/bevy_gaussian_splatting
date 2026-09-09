use std::marker::Copy;

use bevy::prelude::*;
use bevy_interleave::prelude::*;
use bytemuck::{Pod, Zeroable};
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, StandardUniform},
    rng,
    rngs::StdRng,
};
use serde::{Deserialize, Serialize};

use crate::{
    gaussian::{
        f32::{IsotropicRotations, PositionVisibility, ScaleOpacity, TimestampTimescale},
        interface::{CommonCloud, TestCloud},
        iter::PositionIter,
    },
    material::spherindrical_harmonics::{SH_4D_COEFF_COUNT, SpherindricalHarmonicCoefficients},
};

#[derive(
    Clone,
    Debug,
    Default,
    Copy,
    PartialEq,
    Planar,
    ReflectInterleaved,
    StorageBindings,
    Reflect,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[serde(default)]
#[repr(C)]
pub struct Gaussian4d {
    #[serde(default)]
    pub position_visibility: PositionVisibility,
    #[serde(default)]
    pub spherindrical_harmonic: SpherindricalHarmonicCoefficients,
    #[serde(default)]
    pub isotropic_rotations: IsotropicRotations,
    #[serde(default)]
    pub scale_opacity: ScaleOpacity,
    #[serde(default)]
    pub timestamp_timescale: TimestampTimescale,
}

impl CommonCloud for PlanarGaussian4d {
    type PackedType = Gaussian4d;

    fn visibility(&self, index: usize) -> f32 {
        self.position_visibility[index].visibility
    }

    fn visibility_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.position_visibility[index].visibility
    }

    fn support_radius(&self, index: usize) -> Vec3 {
        let scale = Vec3::from_array(self.scale_opacity[index].scale).abs();
        if scale.is_finite() {
            Vec3::splat(scale.max_element() * 3.0)
        } else {
            Vec3::splat(0.1)
        }
    }

    fn position_iter(&self) -> PositionIter<'_> {
        PositionIter::new(&self.position_visibility)
    }

    #[cfg(feature = "sort_rayon")]
    fn position_par_iter(&self) -> crate::gaussian::iter::PositionParIter<'_> {
        crate::gaussian::iter::PositionParIter::new(&self.position_visibility)
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

impl Distribution<Gaussian4d> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gaussian4d {
        let mut coefficients = [0.0; SH_4D_COEFF_COUNT];
        for coefficient in coefficients.iter_mut() {
            *coefficient = rng.random_range(-1.0..1.0);
        }

        Gaussian4d {
            isotropic_rotations: [
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ]
            .into(),
            position_visibility: [
                rng.random_range(-20.0..20.0),
                rng.random_range(-20.0..20.0),
                rng.random_range(-20.0..20.0),
                1.0,
            ]
            .into(),
            scale_opacity: [
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..0.8),
            ]
            .into(),
            spherindrical_harmonic: coefficients.into(),
            timestamp_timescale: [
                rng.random_range(0.0..1.0),
                rng.random_range(-1.0..1.0),
                0.0,
                0.0,
            ]
            .into(),
        }
    }
}

pub fn random_gaussians_4d(n: usize) -> PlanarGaussian4d {
    let mut rng = rng();
    let mut gaussians: Vec<Gaussian4d> = Vec::with_capacity(n);

    for _ in 0..n {
        gaussians.push(rng.random());
    }

    PlanarGaussian4d::from_interleaved(gaussians)
}

pub fn random_gaussians_4d_seeded(n: usize, seed: u64) -> PlanarGaussian4d {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut gaussians: Vec<Gaussian4d> = Vec::with_capacity(n);

    for _ in 0..n {
        gaussians.push(StandardUniform.sample(&mut rng));
    }

    PlanarGaussian4d::from_interleaved(gaussians)
}

impl TestCloud for PlanarGaussian4d {
    fn test_model() -> Self {
        random_gaussians_4d(512)
    }
}
