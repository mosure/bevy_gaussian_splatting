use std::marker::Copy;

use bevy::prelude::*;
use bytemuck::{
    Pod,
    Zeroable,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    gaussian::f32::{
        PositionOpacity,
        PositionVisibility,
        Rotation,
        Scale4d,
        ScaleOpacity,
    },
    material::{
        spherical_harmonics::SphericalHarmonicCoefficients,
        spherindrical_harmonics::SpherindricalHarmonicCoefficients,
    },
};

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
pub struct Gaussian {
    pub rotation: Rotation,
    pub position_visibility: PositionVisibility,
    pub scale_opacity: ScaleOpacity,
    pub spherical_harmonic: SphericalHarmonicCoefficients,
}


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
pub struct Gaussian4d {
    pub isomorphic_rotations: [Rotation; 2],
    pub position_opacity: PositionOpacity,
    pub scale: Scale4d,
    pub spherindrical_harmonic: SpherindricalHarmonicCoefficients,
}
