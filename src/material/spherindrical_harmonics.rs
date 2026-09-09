#![allow(dead_code)] // ShaderType derives emit unused check helpers
use std::marker::Copy;

use bevy::{
    asset::{load_internal_asset, uuid_handle},
    prelude::*,
    render::render_resource::ShaderType,
};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize, Serializer, ser::SerializeTuple};

use crate::{
    material::spherical_harmonics::{SH_CHANNELS, SH_DEGREE},
    math::{gcd, pad_4},
};

pub const SH_4D_DEGREE_TIME: usize = 2;

pub const SH_4D_COEFF_COUNT_PER_CHANNEL: usize = (SH_DEGREE + 1).pow(2) * (SH_4D_DEGREE_TIME + 1);
pub const SH_4D_COEFF_COUNT: usize = pad_4(SH_4D_COEFF_COUNT_PER_CHANNEL * SH_CHANNELS);

pub const HALF_SH_4D_COEFF_COUNT: usize = pad_4(SH_4D_COEFF_COUNT / 2);

// TODO: calculate POD_PLANE_COUNT for both f16 and f32
pub const MAX_POD_U32_ARRAY_SIZE: usize = 32;
pub const POD_ARRAY_SIZE: usize = gcd(SH_4D_COEFF_COUNT, MAX_POD_U32_ARRAY_SIZE);
pub const POD_PLANE_COUNT: usize = SH_4D_COEFF_COUNT / POD_ARRAY_SIZE;

pub const WASTE: usize = POD_PLANE_COUNT * POD_ARRAY_SIZE - SH_4D_COEFF_COUNT;
static_assertions::const_assert_eq!(WASTE, 0);

pub const SH_4D_VEC4_PLANES: usize = SH_4D_COEFF_COUNT / 4;

const SPHERINDRICAL_HARMONICS_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("0b379c3c-daa3-48c5-bf4b-0262b9941a0a");

pub struct SpherindricalHarmonicCoefficientsPlugin;
impl Plugin for SpherindricalHarmonicCoefficientsPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SPHERINDRICAL_HARMONICS_SHADER_HANDLE,
            "spherindrical_harmonics.wgsl",
            Shader::from_wgsl
        );
    }
}

#[allow(dead_code)]
#[derive(
    Clone, Copy, Debug, PartialEq, Reflect, ShaderType, Pod, Zeroable, Serialize, Deserialize,
)]
#[repr(C)]
pub struct SpherindricalHarmonicCoefficients {
    #[serde(
        serialize_with = "coefficients_serializer",
        deserialize_with = "coefficients_deserializer"
    )]
    pub coefficients: [[f32; POD_ARRAY_SIZE]; POD_PLANE_COUNT],
}

impl Default for SpherindricalHarmonicCoefficients {
    fn default() -> Self {
        Self {
            coefficients: [[0.0; POD_ARRAY_SIZE]; POD_PLANE_COUNT],
        }
    }
}

impl From<[f32; SH_4D_COEFF_COUNT]> for SpherindricalHarmonicCoefficients {
    fn from(flat_coefficients: [f32; SH_4D_COEFF_COUNT]) -> Self {
        let mut coefficients = [[0.0; POD_ARRAY_SIZE]; POD_PLANE_COUNT];

        for (i, coefficient) in flat_coefficients.iter().enumerate() {
            coefficients[i / POD_ARRAY_SIZE][i % POD_ARRAY_SIZE] = *coefficient;
        }

        Self { coefficients }
    }
}

impl SpherindricalHarmonicCoefficients {
    pub fn set(&mut self, index: usize, value: f32) {
        let pod_index = index / POD_ARRAY_SIZE;
        let pod_offset = index % POD_ARRAY_SIZE;

        self.coefficients[pod_index][pod_offset] = value;
    }
}

fn coefficients_serializer<S>(
    n: &[[f32; POD_ARRAY_SIZE]; POD_PLANE_COUNT],
    s: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut tup = s.serialize_tuple(SH_4D_COEFF_COUNT)?;
    for &x in n.iter() {
        tup.serialize_element(&x)?;
    }

    tup.end()
}

fn coefficients_deserializer<'de, D>(
    d: D,
) -> Result<[[f32; POD_ARRAY_SIZE]; POD_PLANE_COUNT], D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct CoefficientsVisitor;

    impl<'de> serde::de::Visitor<'de> for CoefficientsVisitor {
        type Value = [[f32; POD_ARRAY_SIZE]; POD_PLANE_COUNT];

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("an array of floats")
        }

        fn visit_seq<A>(
            self,
            mut seq: A,
        ) -> Result<[[f32; POD_ARRAY_SIZE]; POD_PLANE_COUNT], A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut coefficients = [[0.0; POD_ARRAY_SIZE]; POD_PLANE_COUNT];

            for (i, coefficient) in coefficients.iter_mut().enumerate().take(SH_4D_COEFF_COUNT) {
                *coefficient = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
            }
            Ok(coefficients)
        }
    }

    d.deserialize_tuple(SH_4D_COEFF_COUNT, CoefficientsVisitor)
}
