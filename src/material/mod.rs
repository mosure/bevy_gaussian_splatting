use bevy::prelude::*;

pub mod depth;
pub mod spherical_harmonics;

#[cfg(feature = "material_noise")]
pub mod noise;


#[derive(Default)]
pub struct MaterialPlugin;

impl Plugin for MaterialPlugin {
    #[allow(unused)]
    fn build(&self, app: &mut App) {
        #[cfg(feature = "material_noise")]
        app.add_plugins(noise::NoiseMaterialPlugin);

        app.add_plugins((
            depth::DepthMaterialPlugin,
            spherical_harmonics::SphericalHarmonicCoefficientsPlugin,
        ));
    }
}
