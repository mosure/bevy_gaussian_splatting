use rand::prelude::Distribution;
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


type f16_pod_t = [u8; 2];

#[derive(
    Clone,
    Debug,
    Default,
    Copy,
    PartialEq,
    Reflect,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct RotationScaleOpacityPacked128 {
    #[reflect(ignore)]
    pub rotation: [f16_pod_t; 4],
    #[reflect(ignore)]
    pub scale: [f16_pod_t; 3],
    #[reflect(ignore)]
    pub opacity: f16_pod_t,
}

impl From<[f32; 8]> for RotationScaleOpacityPacked128 {
    fn from(rotation_scale_opacity: [f32; 8]) -> Self {
        Self {
            rotation: [
                f16::from_f32(rotation_scale_opacity[0]).to_bits().to_be_bytes(),
                f16::from_f32(rotation_scale_opacity[1]).to_bits().to_be_bytes(),
                f16::from_f32(rotation_scale_opacity[2]).to_bits().to_be_bytes(),
                f16::from_f32(rotation_scale_opacity[3]).to_bits().to_be_bytes(),
            ],
            scale: [
                f16::from_f32(rotation_scale_opacity[4]).to_bits().to_be_bytes(),
                f16::from_f32(rotation_scale_opacity[5]).to_bits().to_be_bytes(),
                f16::from_f32(rotation_scale_opacity[6]).to_bits().to_be_bytes(),
            ],
            opacity: f16::from_f32(rotation_scale_opacity[7]).to_bits().to_be_bytes(),
        }
    }
}

impl From<[f16; 8]> for RotationScaleOpacityPacked128 {
    fn from(rotation_scale_opacity: [f16; 8]) -> Self {
        Self {
            rotation: [
                rotation_scale_opacity[0].to_bits().to_be_bytes(),
                rotation_scale_opacity[1].to_bits().to_be_bytes(),
                rotation_scale_opacity[2].to_bits().to_be_bytes(),
                rotation_scale_opacity[3].to_bits().to_be_bytes(),
            ],
            scale: [
                rotation_scale_opacity[4].to_bits().to_be_bytes(),
                rotation_scale_opacity[5].to_bits().to_be_bytes(),
                rotation_scale_opacity[6].to_bits().to_be_bytes(),
            ],
            opacity: rotation_scale_opacity[7].to_bits().to_be_bytes(),
        }
    }
}
