use bevy::prelude::*;
use bevy_interleave::prelude::*;

#[cfg(feature = "morph_particles")]
pub mod particle;

pub struct MorphPlugin<R: PlanarSync> {
    _phantom: std::marker::PhantomData<R>,
}
impl<R: PlanarSync> Default for MorphPlugin<R> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<R: PlanarSync> Plugin for MorphPlugin<R> {
    #[allow(unused)]
    fn build(&self, app: &mut App) {
        #[cfg(feature = "morph_particles")]
        app.add_plugins(particle::ParticleBehaviorPlugin::<R>::default());
    }
}
