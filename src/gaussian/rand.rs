use rand::{
    prelude::Distribution,
    Rng,
};

#[cfg(feature = "f16")]
use crate::gaussian::f16::pack_f32s_to_u32;

#[allow(unused_imports)]
use crate::{
    gaussian::{
        cloud::Cloud,
        formats::{
            cloud_3d::Cloud3d,
            cloud_4d::Cloud4d,
        },
        packed::{
            Gaussian,
            Gaussian4d,
        },
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


impl Distribution<Gaussian> for rand::distributions::Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gaussian {
        Gaussian {
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
                    #[cfg(feature = "f16")]
                    {
                        let mut coefficients: [u32; HALF_SH_COEFF_COUNT] = [0; HALF_SH_COEFF_COUNT];
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
        }
    }
}


impl Distribution<Gaussian4d> for rand::distributions::Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gaussian4d {
        Gaussian4d {
            isomorphic_rotations: [
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
            spherindrical_harmonic: SpherindricalHarmonicCoefficients {
                coefficients: {
                    #[cfg(feature = "f16")]
                    {
                        let mut coefficients: [u32; HALF_SH_4D_COEFF_COUNT] = [0; HALF_SH_4D_COEFF_COUNT];
                        for coefficient in coefficients.iter_mut() {
                            let upper = rng.gen_range(-1.0..1.0);
                            let lower = rng.gen_range(-1.0..1.0);

                            *coefficient = pack_f32s_to_u32(upper, lower);
                        }
                        coefficients
                    }

                    {
                        let mut coefficients = [0.0; SH_4D_COEFF_COUNT];
                        for coefficient in coefficients.iter_mut() {
                            *coefficient = rng.gen_range(-1.0..1.0);
                        }
                        coefficients
                    }
                },
            },
            timestamp_timescale: [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                0.0,
                0.0,
            ].into(),
        }
    }
}



pub fn random_gaussians(n: usize) -> Cloud {
    let mut rng = rand::thread_rng();
    let mut gaussians: Vec<Gaussian> = Vec::with_capacity(n);

    for _ in 0..n {
        gaussians.push(rng.gen());
    }

    Cloud::Gaussian3d(gaussians.into())
}

pub fn random_gaussians_4d(n: usize) -> Cloud {
    let mut rng = rand::thread_rng();
    let mut gaussians: Vec<Gaussian4d> = Vec::with_capacity(n);

    for _ in 0..n {
        gaussians.push(rng.gen());
    }

    Cloud::Gaussian4d(gaussians.into())
}
