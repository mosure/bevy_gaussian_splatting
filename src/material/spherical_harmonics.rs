use std::marker::Copy;

use bevy::{
    prelude::*,
    asset::load_internal_asset,
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


const SPHERICAL_HARMONICS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(834667312);


pub struct SphericalHarmonicCoefficientsPlugin;

impl Plugin for SphericalHarmonicCoefficientsPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SPHERICAL_HARMONICS_SHADER_HANDLE,
            "spherical_harmonics.wgsl",
            Shader::from_wgsl
        );
    }
}


const fn num_sh_coefficients(degree: usize) -> usize {
    if degree == 0 {
        1
    } else {
        2 * degree + 1 + num_sh_coefficients(degree - 1)
    }
}
const SH_DEGREE: usize = 2;  // TODO: fix SH_DEGREE 3 - no default array >32 elements
pub const SH_CHANNELS: usize = 3;
pub const MAX_SH_COEFF_COUNT_PER_CHANNEL: usize = num_sh_coefficients(SH_DEGREE);
pub const MAX_SH_COEFF_COUNT: usize = (MAX_SH_COEFF_COUNT_PER_CHANNEL * SH_CHANNELS + 3) & !3;


#[cfg(feature = "precision_half")]
type f16_pod_t = [u8; 2];


#[cfg(feature = "precision_half")]
#[derive(
    Clone,
    Copy,
    Debug,
    Reflect,
    PartialEq,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct SphericalHarmonicCoefficients {
    #[reflect(ignore)]
    #[serde(serialize_with = "coefficients_serializer", deserialize_with = "coefficients_deserializer")]
    pub coefficients: [f16_pod_t; MAX_SH_COEFF_COUNT],
}


#[cfg(not(feature = "precision_half"))]
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
