use bevy::prelude::*;
use noise::{NoiseFn, RidgedMulti, Simplex};

use crate::{Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle};

#[derive(Component, Debug, Reflect)]
pub struct NoiseMaterial {
    pub scale: f32,
}
impl Default for NoiseMaterial {
    fn default() -> Self {
        NoiseMaterial { scale: 1.0 }
    }
}

#[derive(Default)]
pub struct NoiseMaterialPlugin;

impl Plugin for NoiseMaterialPlugin {
    #[allow(unused)]
    fn build(&self, app: &mut App) {
        app.register_type::<NoiseMaterial>();

        app.add_systems(Update, apply_noise_cpu);
    }
}

fn apply_noise_cpu(
    mut gaussian_clouds_res: ResMut<Assets<Cloud>>,
    mut selections: Query<(
        Entity,
        &PlanarGaussian3dHandle,
        &NoiseMaterial,
        Changed<NoiseMaterial>,
    )>,
) {
    for (_entity, cloud_handle, noise_material, changed) in selections.iter_mut() {
        if !changed {
            continue;
        }

        let mut rigid_multi = RidgedMulti::<Simplex>::default();
        rigid_multi.frequency = noise_material.scale as f64;

        let cloud = gaussian_clouds_res.get_mut(cloud_handle).unwrap();

        cloud.gaussians.iter_mut().for_each(|gaussian| {
            let point = |gaussian: &Gaussian3d, idx| {
                let x = gaussian.position_visibility[0];
                let y = gaussian.position_visibility[1];
                let z = gaussian.position_visibility[2];

                [x as f64, y as f64, z as f64, idx as f64]
            };

            for i in 0..gaussian.spherical_harmonic.coefficients.len() {
                let noise = rigid_multi.get(point(&gaussian, i));
                gaussian.spherical_harmonic.coefficients[i] = noise as f32;
            }
        });
    }
}
