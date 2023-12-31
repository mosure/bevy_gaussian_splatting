use rand::{
    prelude::Distribution,
    Rng,
};

#[cfg(feature = "f16")]
use crate::gaussian::f16::pack_f32s_to_u32;

use crate::{
    gaussian::{
        cloud::GaussianCloud,
        packed::Gaussian,
    },
    material::spherical_harmonics::{
        HALF_SH_COEFF_COUNT,
        SphericalHarmonicCoefficients,
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

                    #[cfg(feature = "f32")]
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

pub fn random_gaussians(n: usize) -> GaussianCloud {
    let mut rng = rand::thread_rng();
    let mut gaussians: Vec<Gaussian> = Vec::with_capacity(n);

    for _ in 0..n {
        gaussians.push(rng.gen());
    }

    GaussianCloud::from_gaussians(gaussians)
}

