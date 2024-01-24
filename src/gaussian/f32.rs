use std::marker::Copy;

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
    covariance::compute_covariance_3d,
    packed::Gaussian,
};


pub type Position = [f32; 3];

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
pub struct PositionVisibility {
    pub position: Position,
    pub visibility: f32,
}

impl From<[f32; 4]> for PositionVisibility {
    fn from(position_visibility: [f32; 4]) -> Self {
        Self {
            position: [
                position_visibility[0],
                position_visibility[1],
                position_visibility[2],
            ],
            visibility: position_visibility[3],
        }
    }
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
pub struct PositionOpacity {
    pub position: Position,
    pub opacity: f32,
}

impl From<[f32; 4]> for PositionOpacity {
    fn from(position_visibility: [f32; 4]) -> Self {
        Self {
            position: [
                position_visibility[0],
                position_visibility[1],
                position_visibility[2],
            ],
            opacity: position_visibility[3],
        }
    }
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
pub struct Rotation {
    pub rotation: [f32; 4],
}

impl From<[f32; 4]> for Rotation {
    fn from(rotation: [f32; 4]) -> Self {
        Self { rotation }
    }
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
pub struct Scale4d {
    pub scale: [f32; 4],
}

impl From<[f32; 4]> for Scale4d {
    fn from(scale: [f32; 4]) -> Self {
        Self { scale }
    }
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
pub struct ScaleOpacity {
    pub scale: [f32; 3],
    pub opacity: f32,
}

impl From<[f32; 4]> for ScaleOpacity {
    fn from(scale_opacity: [f32; 4]) -> Self {
        Self {
            scale: [
                scale_opacity[0],
                scale_opacity[1],
                scale_opacity[2],
            ],
            opacity: scale_opacity[3],
        }
    }
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
pub struct Covariance3dOpacity {
    pub cov3d: [f32; 6],
    pub opacity: f32,
    pub pad: f32,
}

impl From<&Gaussian> for Covariance3dOpacity {
    fn from(gaussian: &Gaussian) -> Self {
        let cov3d = compute_covariance_3d(
            Vec4::from_slice(gaussian.rotation.rotation.as_slice()),
            Vec3::from_slice(gaussian.scale_opacity.scale.as_slice()),
        );

        Covariance3dOpacity {
            cov3d,
            opacity: gaussian.scale_opacity.opacity,
            pad: 0.0,
        }
    }
}
