use rand::{
    prelude::Distribution,
    Rng,
};
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
    pub indicies: [u32; 4],
    pub velocity: [f32; 4],
    pub acceleration: [f32; 4],
    pub jerk: [f32; 4],
}

impl Default for ParticleBehavior {
    fn default() -> Self {
        Self {
            indicies: [0, 0, 0, 0],
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
#[uuid = "ac2f08eb-6463-2131-6772-51571ea332d5"]
pub struct ParticleBehaviors(pub Vec<ParticleBehavior>);


impl Distribution<ParticleBehavior> for rand::distributions::Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ParticleBehavior {
        ParticleBehavior {
            acceleration: [
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
            ],
            jerk: [
                rng.gen_range(-0.01..0.01),
                rng.gen_range(-0.01..0.01),
                rng.gen_range(-0.01..0.01),
                rng.gen_range(-0.01..0.01),
            ],
            velocity: [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ],
            ..Default::default()
        }
    }
}

pub fn random_particle_behaviors(n: usize) -> ParticleBehaviors {
    let mut rng = rand::thread_rng();
    let mut behaviors = Vec::with_capacity(n);
    for i in 0..n {
        let mut behavior: ParticleBehavior = rng.gen();
        behavior.indicies[0] = i as u32;
        behaviors.push(behavior);
    }

    ParticleBehaviors(behaviors)
}
