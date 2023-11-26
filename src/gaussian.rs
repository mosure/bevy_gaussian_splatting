use rand::{
    seq::SliceRandom,
    prelude::Distribution,
    Rng,
};
use std::marker::Copy;

use bevy::{
    prelude::*,
    reflect::TypeUuid,
    render::render_resource::ShaderType,
};
use bytemuck::{
    Pod,
    Zeroable,
};
use serde::{
    Deserialize,
    Serialize,
    Serializer,
    ser::SerializeTuple,
};


const fn num_sh_coefficients(degree: usize) -> usize {
    if degree == 0 {
        1
    } else {
        2 * degree + 1 + num_sh_coefficients(degree - 1)
    }
}
const SH_DEGREE: usize = 3;
pub const SH_CHANNELS: usize = 3;
pub const MAX_SH_COEFF_COUNT_PER_CHANNEL: usize = num_sh_coefficients(SH_DEGREE);
pub const MAX_SH_COEFF_COUNT: usize = MAX_SH_COEFF_COUNT_PER_CHANNEL * SH_CHANNELS;
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Reflect,
    ShaderType,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct SphericalHarmonicCoefficients {
    #[serde(serialize_with = "coefficients_serializer", deserialize_with = "coefficients_deserializer")]
    pub coefficients: [f32; MAX_SH_COEFF_COUNT],
}
impl Default for SphericalHarmonicCoefficients {
    fn default() -> Self {
        Self {
            coefficients: [0.0; MAX_SH_COEFF_COUNT],
        }
    }
}
fn coefficients_serializer<S>(n: &[f32; MAX_SH_COEFF_COUNT], s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut tup = s.serialize_tuple(MAX_SH_COEFF_COUNT)?;
    for &x in n.iter() {
        tup.serialize_element(&x)?;
    }

    tup.end()
}

fn coefficients_deserializer<'de, D>(d: D) -> Result<[f32; MAX_SH_COEFF_COUNT], D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct CoefficientsVisitor;

    impl<'de> serde::de::Visitor<'de> for CoefficientsVisitor {
        type Value = [f32; MAX_SH_COEFF_COUNT];

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("an array of floats")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<[f32; MAX_SH_COEFF_COUNT], A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut coefficients = [0.0; MAX_SH_COEFF_COUNT];

            for (i, coefficient) in coefficients.iter_mut().enumerate().take(MAX_SH_COEFF_COUNT) {
                *coefficient = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
            }
            Ok(coefficients)
        }
    }

    d.deserialize_tuple(MAX_SH_COEFF_COUNT, CoefficientsVisitor)
}


#[derive(
    Clone,
    Debug,
    Default,
    Copy,
    PartialEq,
    Reflect,
    ShaderType,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[repr(C)]
// TODO: support f16 gaussian clouds (shader and asset loader)
pub struct Gaussian {
    pub rotation: [f32; 4],
    pub position: [f32; 4],
    pub scale_opacity: [f32; 4],
    pub spherical_harmonic: SphericalHarmonicCoefficients,
}


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
pub struct GaussianCloud {
    pub gaussians: Vec<Gaussian>,
}

impl GaussianCloud {
    pub fn test_model() -> Self {
        let origin = Gaussian {
            rotation: [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            position: [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            scale_opacity: [
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            spherical_harmonic: SphericalHarmonicCoefficients {
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
        };
        let mut cloud = GaussianCloud {
            gaussians: Vec::new(),
            ..default()
        };

        for &x in [-0.5, 0.5].iter() {
            for &y in [-0.5, 0.5].iter() {
                for &z in [-0.5, 0.5].iter() {
                    let mut g = origin;
                    g.position = [x, y, z, 1.0];
                    cloud.gaussians.push(g);

                    let mut rng = rand::thread_rng();
                    cloud.gaussians.last_mut().unwrap().spherical_harmonic.coefficients.shuffle(&mut rng);
                }
            }
        }

        cloud.gaussians.push(cloud.gaussians[0]);

        cloud
    }
}


#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct GaussianCloudSettings {
    pub aabb: bool,
    pub global_scale: f32,
    pub global_transform: GlobalTransform,
    pub visualize_bounding_box: bool,
}

impl Default for GaussianCloudSettings {
    fn default() -> Self {
        Self {
            aabb: false,
            global_scale: 2.0,
            global_transform: Transform::IDENTITY.into(),
            visualize_bounding_box: false,
        }
    }
}

impl Distribution<Gaussian> for rand::distributions::Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gaussian {
        Gaussian {
            rotation: [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ],
            position: [
                rng.gen_range(-20.0..20.0),
                rng.gen_range(-20.0..20.0),
                rng.gen_range(-20.0..20.0),
                rng.gen_range(-1.0..1.0),
            ],
            scale_opacity: [
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..0.8),
            ],
            spherical_harmonic: SphericalHarmonicCoefficients {
                coefficients: {
                    let mut coefficients = [0.0; MAX_SH_COEFF_COUNT];
                    for coefficient in coefficients.iter_mut() {
                        *coefficient = rng.gen_range(-1.0..1.0);
                    }
                    coefficients
                },
            },
        }
    }
}

pub fn random_gaussians(n: usize) -> GaussianCloud {
    let mut rng = rand::thread_rng();
    let mut gaussians = Vec::with_capacity(n);
    for _ in 0..n {
        gaussians.push(rng.gen());
    }
    GaussianCloud {
        gaussians,
        ..default()
    }
}
