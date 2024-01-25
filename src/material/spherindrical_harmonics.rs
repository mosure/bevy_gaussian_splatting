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

#[cfg(feature = "f16")]
use half::f16;

use crate::material::spherical_harmonics::{
    SH_DEGREE,
};


const SH_4D_DEGREE_TIME: usize = 0;


#[cfg(feature = "f16")]
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
pub struct SpherindricalHarmonicCoefficients {
    #[reflect(ignore)]
    #[serde(serialize_with = "coefficients_serializer", deserialize_with = "coefficients_deserializer")]
    pub coefficients: [u32; HALF_SH_COEFF_COUNT],
}
