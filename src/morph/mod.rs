use bevy::prelude::*;


#[cfg(feature = "morph_particles")]
pub mod particle;


#[derive(Default)]
pub struct MorphPlugin;

impl Plugin for MorphPlugin {
    #[allow(unused)]
    fn build(&self, app: &mut App) {
        #[cfg(feature = "morph_particles")]
        app.add_plugins(particle::ParticleBehaviorPlugin);
    }
}
