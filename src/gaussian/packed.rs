use std::marker::Copy;

use bevy::prelude::*;
use bevy_interleave::prelude::*;
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
    Planar,
    ReflectInterleaved,
    StorageBindings,
    Reflect,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct Gaussian3d {
    pub position_visibility: PositionVisibility,
    pub spherical_harmonic: SphericalHarmonicCoefficients,
    pub rotation: Rotation,
    pub scale_opacity: ScaleOpacity,
}

// GaussianMode::Gaussian2d /w Gaussian3d structure
pub type Gaussian2d = Gaussian3d;


#[derive(
    Clone,
    Debug,
    Default,
    Copy,
    PartialEq,
    Planar,
    ReflectInterleaved,
    StorageBindings,
    Reflect,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct Gaussian4d {
    pub position_visibility: PositionVisibility,
    pub spherindrical_harmonic: SpherindricalHarmonicCoefficients,
    pub timestamp_timescale: TimestampTimescale,
    pub isomorphic_rotations: IsotropicRotations,
    pub scale_opacity: ScaleOpacity,
}
