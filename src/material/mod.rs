use bevy::prelude::*;


#[cfg(feature = "material_noise")]
pub mod noise;


#[derive(Default)]
pub struct MaterialPlugin;

impl Plugin for MaterialPlugin {
    #[allow(unused)]
    fn build(&self, app: &mut App) {
        #[cfg(feature = "material_noise")]
        app.add_plugins(noise::NoiseMaterialPlugin);
    }
}
