use bevy::prelude::*;

pub mod classification;
pub mod depth;
pub mod optical_flow;
pub mod position;
pub mod spherical_harmonics;
pub mod spherindrical_harmonics;

#[cfg(feature = "material_noise")]
pub mod noise;

#[cfg(feature = "solari")]
pub mod solari;

#[derive(Default)]
pub struct MaterialPlugin;

impl Plugin for MaterialPlugin {
    #[allow(unused)]
    fn build(&self, app: &mut App) {
        #[cfg(feature = "material_noise")]
        app.add_plugins(noise::NoiseMaterialPlugin);

        #[cfg(feature = "solari")]
        app.add_plugins(solari::SolariMaterialPlugin);

        app.add_plugins((
            classification::ClassificationMaterialPlugin,
            depth::DepthMaterialPlugin,
            optical_flow::OpticalFlowMaterialPlugin,
            position::PositionMaterialPlugin,
            spherical_harmonics::SphericalHarmonicCoefficientsPlugin,
            spherindrical_harmonics::SpherindricalHarmonicCoefficientsPlugin,
        ));
    }
}
