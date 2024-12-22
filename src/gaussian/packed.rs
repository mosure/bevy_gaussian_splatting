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
        IsotropicRotations,
        PositionVisibility,
        Rotation,
        ScaleOpacity,
        TimestampTimescale,
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
    pub isomorphic_rotations: IsotropicRotations,
    pub position_visibility: PositionVisibility,
    pub scale_opacity: ScaleOpacity,
    #[reflect(ignore)]
    pub spherindrical_harmonic: SpherindricalHarmonicCoefficients,
    pub timestamp_timescale: TimestampTimescale,
}
