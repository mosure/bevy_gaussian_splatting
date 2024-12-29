use bevy_interleave::prelude::Planar;
use rand::{
    prelude::Distribution,
    Rng,
};

// #[cfg(feature = "f16")]
// use crate::gaussian::f16::pack_f32s_to_u32;

#[allow(unused_imports)]
use crate::{
    gaussian::packed::{
        Gaussian3d,
        Gaussian4d,
        PlanarGaussian3d,
        PlanarGaussian4d,
    },
    material::{
        spherical_harmonics::{
            HALF_SH_COEFF_COUNT,
            SH_COEFF_COUNT,
            SphericalHarmonicCoefficients,
        },
        spherindrical_harmonics::{
            HALF_SH_4D_COEFF_COUNT,
            SH_4D_COEFF_COUNT,
            SpherindricalHarmonicCoefficients,
        },
    },
};


impl Distribution<Gaussian3d> for rand::distributions::Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gaussian3d {
        Gaussian3d {
            rotation: [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ].into(),
            position_visibility: [
                rng.gen_range(-20.0..20.0),
                rng.gen_range(-20.0..20.0),
                rng.gen_range(-20.0..20.0),
                1.0,
            ].into(),
            scale_opacity: [
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..0.8),
            ].into(),
            spherical_harmonic: SphericalHarmonicCoefficients {
                coefficients: {
                    // #[cfg(feature = "f16")]
                    // {
                    //     let mut coefficients: [u32; HALF_SH_COEFF_COUNT] = [0; HALF_SH_COEFF_COUNT];
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
        }
    }
}


impl Distribution<Gaussian4d> for rand::distributions::Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gaussian4d {
        let mut coefficients = [0.0; SH_4D_COEFF_COUNT];
        for coefficient in coefficients.iter_mut() {
            *coefficient = rng.gen_range(-1.0..1.0);
        }

        Gaussian4d {
            isotropic_rotations: [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ].into(),
            position_visibility: [
                rng.gen_range(-20.0..20.0),
                rng.gen_range(-20.0..20.0),
                rng.gen_range(-20.0..20.0),
                1.0,
            ].into(),
            scale_opacity: [
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..0.8),
            ].into(),
            spherindrical_harmonic: coefficients.into(),
            timestamp_timescale: [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                0.0,
                0.0,
            ].into(),
        }
    }
}



pub fn random_gaussians_3d(n: usize) -> PlanarGaussian3d {
    let mut rng = rand::thread_rng();
    let mut gaussians: Vec<Gaussian3d> = Vec::with_capacity(n);

    for _ in 0..n {
        gaussians.push(rng.gen());
    }

    PlanarGaussian3d::from_interleaved(gaussians)
}

pub fn random_gaussians_4d(n: usize) -> PlanarGaussian4d {
    let mut rng = rand::thread_rng();
    let mut gaussians: Vec<Gaussian4d> = Vec::with_capacity(n);

    for _ in 0..n {
        gaussians.push(rng.gen());
    }

    PlanarGaussian4d::from_interleaved(gaussians)
}
