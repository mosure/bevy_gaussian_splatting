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
        PositionVisibility,
        Rotation,
        ScaleOpacity,
    },
    material::spherical_harmonics::SphericalHarmonicCoefficients,
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
