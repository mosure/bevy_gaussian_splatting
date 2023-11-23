use std::marker::Copy;

use bevy::{
    prelude::*,
    reflect::TypeUuid,
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


#[derive(
    Clone,
    Debug,
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
pub struct ParticleBehavior {
    pub indicies: [i32; 4],
    pub velocity: [f32; 4],
    pub acceleration: [f32; 4],
    pub jerk: [f32; 4],
}

impl Default for ParticleBehavior {
    fn default() -> Self {
        Self {
            indicies: [-1, -1, -1, -1],
            velocity: [0.0, 0.0, 0.0, 0.0],
            acceleration: [0.0, 0.0, 0.0, 0.0],
            jerk: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

#[derive(
    Asset,
    Clone,
    Debug,
    Default,
    PartialEq,
    Reflect,
    TypeUuid,
    Serialize,
    Deserialize,
)]
#[uuid = "ac2f08eb-6463-7542-6772-51571ea332d5"]
pub struct ParticleBehaviors(pub Vec<ParticleBehavior>);
