use std::marker::Copy;

use half::f16;

use bevy::{
    prelude::*,
    render::render_resource::ShaderType,
};
use bytemuck::{
    Pod,
    Zeroable,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::gaussian::{
    f32::{
        Rotation,
        ScaleOpacity,
    },
    packed::Gaussian,
};


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
pub struct RotationScaleOpacityPacked128 {
    #[reflect(ignore)]
    pub rotation: [u32; 2],
    #[reflect(ignore)]
    pub scale_opacity: [u32; 2],
}

impl RotationScaleOpacityPacked128 {
    pub fn from_gaussian(gaussian: &Gaussian) -> Self {
        Self {
            rotation: [
                pack_f32s_to_u32(gaussian.rotation.rotation[0], gaussian.rotation.rotation[1]),
                pack_f32s_to_u32(gaussian.rotation.rotation[2], gaussian.rotation.rotation[3]),
            ],
            scale_opacity: [
                pack_f32s_to_u32(gaussian.scale_opacity.scale[0], gaussian.scale_opacity.scale[1]),
                pack_f32s_to_u32(gaussian.scale_opacity.scale[2], gaussian.scale_opacity.opacity),
            ],
        }
    }

    pub fn rotation(&self) -> Rotation {
        let (u0, l0) = unpack_u32_to_f32s(self.rotation[0]);
        let (u1, l1) = unpack_u32_to_f32s(self.rotation[1]);

        Rotation {
            rotation: [
                u0,
                l0,
                u1,
                l1,
            ],
        }
    }

    pub fn scale_opacity(&self) -> ScaleOpacity {
        let (u0, l0) = unpack_u32_to_f32s(self.scale_opacity[0]);
        let (u1, l1) = unpack_u32_to_f32s(self.scale_opacity[1]);

        ScaleOpacity {
            scale: [
                u0,
                l0,
                u1,
            ],
            opacity: l1,
        }
    }
}

impl From<[f32; 8]> for RotationScaleOpacityPacked128 {
    fn from(rotation_scale_opacity: [f32; 8]) -> Self {
        Self {
            rotation: [
                pack_f32s_to_u32(rotation_scale_opacity[0], rotation_scale_opacity[1]),
                pack_f32s_to_u32(rotation_scale_opacity[2], rotation_scale_opacity[3]),
            ],
            scale_opacity: [
                pack_f32s_to_u32(rotation_scale_opacity[4], rotation_scale_opacity[5]),
                pack_f32s_to_u32(rotation_scale_opacity[6], rotation_scale_opacity[7]),
            ],
        }
    }
}

impl From<[f16; 8]> for RotationScaleOpacityPacked128 {
    fn from(rotation_scale_opacity: [f16; 8]) -> Self {
        Self {
            rotation: [
                pack_f16s_to_u32(rotation_scale_opacity[0], rotation_scale_opacity[1]),
                pack_f16s_to_u32(rotation_scale_opacity[2], rotation_scale_opacity[3]),
            ],
            scale_opacity: [
                pack_f16s_to_u32(rotation_scale_opacity[4], rotation_scale_opacity[5]),
                pack_f16s_to_u32(rotation_scale_opacity[6], rotation_scale_opacity[7]),
            ],
        }
    }
}

impl From<[u32; 4]> for RotationScaleOpacityPacked128 {
    fn from(rotation_scale_opacity: [u32; 4]) -> Self {
        Self {
            rotation: [rotation_scale_opacity[0], rotation_scale_opacity[1]],
            scale_opacity: [rotation_scale_opacity[2], rotation_scale_opacity[3]],
        }
    }
}


pub fn pack_f32s_to_u32(upper: f32, lower: f32) -> u32 {
    pack_f16s_to_u32(
        f16::from_f32(upper),
        f16::from_f32(lower),
    )
}

pub fn pack_f16s_to_u32(upper: f16, lower: f16) -> u32 {
    let upper_bits = (upper.to_bits() as u32) << 16;
    let lower_bits = lower.to_bits() as u32;
    upper_bits | lower_bits
}

pub fn unpack_u32_to_f16s(value: u32) -> (f16, f16) {
    let upper = f16::from_bits((value >> 16) as u16);
    let lower = f16::from_bits((value & 0xFFFF) as u16);
    (upper, lower)
}

pub fn unpack_u32_to_f32s(value: u32) -> (f32, f32) {
    let (upper, lower) = unpack_u32_to_f16s(value);
    (upper.to_f32(), lower.to_f32())
}
