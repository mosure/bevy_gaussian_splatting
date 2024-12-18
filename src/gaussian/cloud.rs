use rand::{
    seq::SliceRandom,
    Rng,
};

use bevy::{
    prelude::*,
    render::{
        primitives::Aabb,
        sync_world::SyncToRenderWorld,
        view::visibility::{
            check_visibility,
            NoFrustumCulling,
            VisibilitySystems,
        },
    },
};
use serde::{
    Deserialize,
    Serialize,
};

#[cfg(feature = "sort_rayon")]
use rayon::prelude::*;

#[allow(unused_imports)]
use crate::{
    gaussian::{
        f32::{
            Covariance3dOpacity,
            Position,
            PositionVisibility,
            Rotation,
            ScaleOpacity,
        },
        packed::Gaussian,
        settings::GaussianCloudSettings,
    },
    material::spherical_harmonics::{
        HALF_SH_COEFF_COUNT,
        SH_COEFF_COUNT,
        SphericalHarmonicCoefficients,
    },
};

#[allow(unused_imports)]
#[cfg(feature = "f16")]
use crate::gaussian::f16::{
    Covariance3dOpacityPacked128,
    RotationScaleOpacityPacked128,
    pack_f32s_to_u32,
};


#[derive(Default)]
pub struct GaussianCloudPlugin;

impl Plugin for GaussianCloudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            (
                calculate_bounds.in_set(VisibilitySystems::CalculateBounds),
                check_visibility::<With<GaussianCloudHandle>>.in_set(VisibilitySystems::CheckVisibility),
            )
        );
    }
}


// TODO: handle aabb updates (e.g. gaussian particle movements)
#[allow(clippy::type_complexity)]
pub fn calculate_bounds(
    mut commands: Commands,
    gaussian_clouds: Res<Assets<GaussianCloud>>,
    without_aabb: Query<
        (
            Entity,
            &GaussianCloudHandle,
        ),
        (
            Without<Aabb>,
            Without<NoFrustumCulling>,
        ),
    >,
) {
    for (entity, cloud_handle) in &without_aabb {
        if let Some(cloud) = gaussian_clouds.get(cloud_handle) {
            if let Some(aabb) = cloud.compute_aabb() {
                commands.entity(entity).try_insert(aabb);
            }
        }
    }
}


#[derive(
    Component,
    Clone,
    Debug,
    Default,
    PartialEq,
    Reflect,
)]
#[reflect(Component, Default)]
#[require(
    GaussianCloudSettings,
    SyncToRenderWorld,
    Transform,
    Visibility,
)]
pub struct GaussianCloudHandle(pub Handle<GaussianCloud>);

impl From<Handle<GaussianCloud>> for GaussianCloudHandle {
    fn from(handle: Handle<GaussianCloud>) -> Self {
        Self(handle)
    }
}

impl From<GaussianCloudHandle> for AssetId<GaussianCloud> {
    fn from(handle: GaussianCloudHandle) -> Self {
        handle.0.id()
    }
}

impl From<&GaussianCloudHandle> for AssetId<GaussianCloud> {
    fn from(handle: &GaussianCloudHandle) -> Self {
        handle.0.id()
    }
}


#[cfg(feature = "f16")]
#[derive(
    Asset,
    Clone,
    Debug,
    Default,
    PartialEq,
    Reflect,
    Serialize,
    Deserialize,
)]
pub struct GaussianCloud {
    pub position_visibility: Vec<PositionVisibility>,

    pub spherical_harmonic: Vec<SphericalHarmonicCoefficients>,

    #[cfg(not(feature = "precompute_covariance_3d"))]
    pub rotation_scale_opacity_packed128: Vec<RotationScaleOpacityPacked128>,

    #[cfg(feature = "precompute_covariance_3d")]
    pub covariance_3d_opacity_packed128: Vec<Covariance3dOpacityPacked128>,
}

#[cfg(feature = "f32")]
#[derive(
    Asset,
    Clone,
    Debug,
    Default,
    PartialEq,
    Reflect,
    Serialize,
    Deserialize,
)]
pub struct GaussianCloud {
    pub position_visibility: Vec<PositionVisibility>,

    pub spherical_harmonic: Vec<SphericalHarmonicCoefficients>,

    #[cfg(feature = "precompute_covariance_3d")]
    pub covariance_3d: Vec<Covariance3dOpacity>,

    #[cfg(not(feature = "precompute_covariance_3d"))]
    pub rotation: Vec<Rotation>,
    #[cfg(not(feature = "precompute_covariance_3d"))]
    pub scale_opacity: Vec<ScaleOpacity>,
}

impl GaussianCloud {
    pub fn is_empty(&self) -> bool {
        self.position_visibility.is_empty()
    }

    pub fn len(&self) -> usize {
        self.position_visibility.len()
    }

    pub fn len_sqrt_ceil(&self) -> usize {
        (self.len() as f32).sqrt().ceil() as usize
    }

    pub fn square_len(&self) -> usize {
        self.len_sqrt_ceil().pow(2)
    }

    pub fn position(&self, index: usize) -> &[f32; 3] {
        &self.position_visibility[index].position
    }

    pub fn position_mut(&mut self, index: usize) -> &mut [f32; 3] {
        &mut self.position_visibility[index].position
    }

    pub fn position_iter(&self) -> impl Iterator<Item = &Position> + '_ {
        self.position_visibility.iter()
            .map(|position_visibility| &position_visibility.position)
    }

    #[cfg(feature = "sort_rayon")]
    pub fn position_par_iter(&self) -> impl IndexedParallelIterator<Item = &Position> {
        self.position_visibility.par_iter()
            .map(|position_visibility| &position_visibility.position)
    }


    pub fn visibility(&self, index: usize) -> f32 {
        self.position_visibility[index].visibility
    }

    pub fn visibility_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.position_visibility[index].visibility
    }

    pub fn compute_aabb(&self) -> Option<Aabb> {
        if self.is_empty() {
            return None;
        }

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        // TODO: find a more correct aabb bound derived from scalar max gaussian scale
        let max_scale = 0.1;

        for position in self.position_iter() {
            min = min.min(Vec3::from(*position) - Vec3::splat(max_scale));
            max = max.max(Vec3::from(*position) + Vec3::splat(max_scale));
        }

        Aabb::from_min_max(min, max).into()
    }


    // pub fn rotation(&self, index: usize) -> &[f32; 4] {
    //     #[cfg(feature = "f16")]
    //     return &self.rotation_scale_opacity_packed128[index].rotation;

    //     #[cfg(feature = "f32")]
    //     return &self.rotation[index].rotation;
    // }

    // pub fn rotation_mut(&mut self, index: usize) -> &mut [f32; 4] {
    //     #[cfg(feature = "f16")]
    //     return &mut self.rotation_scale_opacity_packed128[index].rotation;

    //     #[cfg(feature = "f32")]
    //     return &mut self.rotation[index].rotation;
    // }


    // pub fn scale(&self, index: usize) -> &[f32; 3] {
    //     #[cfg(feature = "f16")]
    //     return &self.rotation_scale_opacity_packed128[index].scale;

    //     #[cfg(feature = "f32")]
    //     return &self.scale_opacity[index].scale;
    // }

    // pub fn scale_mut(&mut self, index: usize) -> &mut [f32; 3] {
    //     #[cfg(feature = "f16")]
    //     return &mut self.rotation_scale_opacity_packed128[index].scale;

    //     #[cfg(feature = "f32")]
    //     return &mut self.scale_opacity[index].scale;
    // }

    #[cfg(all(
        not(feature = "precompute_covariance_3d"),
        feature = "f16",
    ))]
    pub fn gaussian(&self, index: usize) -> Gaussian {
        let rso = self.rotation_scale_opacity_packed128[index];

        let rotation = rso.rotation();
        let scale_opacity = rso.scale_opacity();

        Gaussian {
            position_visibility: self.position_visibility[index],
            spherical_harmonic: self.spherical_harmonic[index],
            rotation,
            scale_opacity,
        }
    }

    #[cfg(feature = "f32")]
    pub fn gaussian(&self, index: usize) -> Gaussian {
        Gaussian {
            position_visibility: self.position_visibility[index],
            spherical_harmonic: self.spherical_harmonic[index],
            rotation: self.rotation[index],
            scale_opacity: self.scale_opacity[index],
        }
    }

    #[cfg(all(
        not(feature = "precompute_covariance_3d"),
        feature = "f16",
    ))]
    pub fn gaussian_iter(&self) -> impl Iterator<Item=Gaussian> + '_ {
        self.position_visibility.iter()
            .zip(self.spherical_harmonic.iter())
            .zip(self.rotation_scale_opacity_packed128.iter())
            .map(|((position_visibility, spherical_harmonic), rotation_scale_opacity)| {
                Gaussian {
                    position_visibility: *position_visibility,
                    spherical_harmonic: *spherical_harmonic,

                    rotation: rotation_scale_opacity.rotation(),
                    scale_opacity: rotation_scale_opacity.scale_opacity(),
                }
            })
    }

    #[cfg(feature = "f32")]
    pub fn gaussian_iter(&self) -> impl Iterator<Item=Gaussian> + '_ {
        self.position_visibility.iter()
            .zip(self.spherical_harmonic.iter())
            .zip(self.rotation.iter())
            .zip(self.scale_opacity.iter())
            .map(|(((position_visibility, spherical_harmonic), rotation), scale_opacity)| {
                Gaussian {
                    position_visibility: *position_visibility,
                    spherical_harmonic: *spherical_harmonic,

                    rotation: *rotation,
                    scale_opacity: *scale_opacity,
                }
            })
    }


    pub fn spherical_harmonic(&self, index: usize) -> &SphericalHarmonicCoefficients {
        &self.spherical_harmonic[index]
    }

    pub fn spherical_harmonic_mut(&mut self, index: usize) -> &mut SphericalHarmonicCoefficients {
        &mut self.spherical_harmonic[index]
    }

    pub fn resize_to_square(&mut self) {
        #[cfg(all(feature = "buffer_texture", feature = "f16"))]
        {
            self.position_visibility.resize(self.square_len(), PositionVisibility::default());
            self.spherical_harmonic.resize(self.square_len(), SphericalHarmonicCoefficients::default());

            #[cfg(feature = "precompute_covariance_3d")]
            self.covariance_3d_opacity_packed128.resize(self.square_len(), Covariance3dOpacityPacked128::default());
            #[cfg(not(feature = "precompute_covariance_3d"))]
            self.rotation_scale_opacity_packed128.resize(self.square_len(), RotationScaleOpacityPacked128::default());
        }

        #[cfg(all(feature = "buffer_texture", feature = "f32"))]
        {
            self.position_visibility.resize(self.square_len(), PositionVisibility::default());
            self.spherical_harmonic.resize(self.square_len(), SphericalHarmonicCoefficients::default());
            self.rotation.resize(self.square_len(), Rotation::default());
            self.scale_opacity.resize(self.square_len(), ScaleOpacity::default());
            self.covariance_3d.resize(self.square_len(), Covariance3dOpacity::default());
        }
    }
}


impl GaussianCloud {
    #[cfg(feature = "f16")]
    pub fn subset(&self, indicies: &[usize]) -> Self {
        let mut position_visibility = Vec::with_capacity(indicies.len());
        let mut spherical_harmonic = Vec::with_capacity(indicies.len());

        #[cfg(feature = "precompute_covariance_3d")]
        let mut covariance_3d_opacity_packed128 = Vec::with_capacity(indicies.len());

        #[cfg(not(feature = "precompute_covariance_3d"))]
        let mut rotation_scale_opacity_packed128 = Vec::with_capacity(indicies.len());

        for &index in indicies.iter() {
            position_visibility.push(self.position_visibility[index]);
            spherical_harmonic.push(self.spherical_harmonic[index]);

            #[cfg(feature = "precompute_covariance_3d")]
            covariance_3d_opacity_packed128.push(self.covariance_3d_opacity_packed128[index]);

            #[cfg(not(feature = "precompute_covariance_3d"))]
            rotation_scale_opacity_packed128.push(self.rotation_scale_opacity_packed128[index]);
        }

        Self {
            position_visibility,
            spherical_harmonic,

            #[cfg(feature = "precompute_covariance_3d")]
            covariance_3d_opacity_packed128,
            #[cfg(not(feature = "precompute_covariance_3d"))]
            rotation_scale_opacity_packed128,
        }
    }

    #[cfg(feature = "f32")]
    pub fn subset(&self, indicies: &[usize]) -> Self {
        let mut position_visibility = Vec::with_capacity(indicies.len());
        let mut spherical_harmonic = Vec::with_capacity(indicies.len());
        let mut rotation = Vec::with_capacity(indicies.len());
        let mut scale_opacity = Vec::with_capacity(indicies.len());

        for &index in indicies.iter() {
            position_visibility.push(self.position_visibility[index]);
            spherical_harmonic.push(self.spherical_harmonic[index]);
            rotation.push(self.rotation[index]);
            scale_opacity.push(self.scale_opacity[index]);
        }

        Self {
            position_visibility,
            spherical_harmonic,
            rotation,
            scale_opacity,
        }
    }

    #[cfg(feature = "f32")]
    pub fn to_packed(&self) -> Vec<Gaussian> {
        let mut gaussians = Vec::with_capacity(self.len());

        for index in 0..self.len() {
            gaussians.push(self.gaussian(index));
        }

        gaussians
    }
}


impl GaussianCloud {
    #[cfg(feature = "f16")]
    pub fn from_gaussians(gaussians: Vec<Gaussian>) -> Self {
        let mut position_visibility = Vec::with_capacity(gaussians.len());
        let mut spherical_harmonic = Vec::with_capacity(gaussians.len());

        #[cfg(feature = "precompute_covariance_3d")]
        let mut covariance_3d_opacity_packed128 = Vec::with_capacity(gaussians.len());

        #[cfg(not(feature = "precompute_covariance_3d"))]
        let mut rotation_scale_opacity_packed128 = Vec::with_capacity(gaussians.len());

        for gaussian in gaussians {
            position_visibility.push(gaussian.position_visibility);
            spherical_harmonic.push(gaussian.spherical_harmonic);

            #[cfg(feature = "precompute_covariance_3d")]
            covariance_3d_opacity_packed128.push(Covariance3dOpacityPacked128::from_gaussian(&gaussian));

            #[cfg(not(feature = "precompute_covariance_3d"))]
            rotation_scale_opacity_packed128.push(RotationScaleOpacityPacked128::from_gaussian(&gaussian));
        }

        #[allow(unused_mut)]
        let mut cloud = GaussianCloud {
            position_visibility,
            spherical_harmonic,

            #[cfg(feature = "precompute_covariance_3d")]
            covariance_3d_opacity_packed128,
            #[cfg(not(feature = "precompute_covariance_3d"))]
            rotation_scale_opacity_packed128,
        };

        cloud.resize_to_square();

        cloud
    }

    #[cfg(feature = "f32")]
    pub fn from_gaussians(gaussians: Vec<Gaussian>) -> Self {
        let mut position_visibility = Vec::with_capacity(gaussians.len());
        let mut spherical_harmonic = Vec::with_capacity(gaussians.len());
        let mut rotation = Vec::with_capacity(gaussians.len());
        let mut scale_opacity = Vec::with_capacity(gaussians.len());

        for gaussian in gaussians {
            position_visibility.push(gaussian.position_visibility);
            spherical_harmonic.push(gaussian.spherical_harmonic);

            rotation.push(gaussian.rotation);
            scale_opacity.push(gaussian.scale_opacity);
        }

        Self {
            position_visibility,
            spherical_harmonic,
            rotation,
            scale_opacity,
        }
    }

    pub fn test_model() -> Self {
        let mut rng = rand::thread_rng();

        let origin = Gaussian {
            rotation: [
                1.0,
                0.0,
                0.0,
                0.0,
            ].into(),
            position_visibility: [
                0.0,
                0.0,
                0.0,
                1.0,
            ].into(),
            scale_opacity: [
                0.5,
                0.5,
                0.5,
                0.5,
            ].into(),
            spherical_harmonic: SphericalHarmonicCoefficients {
                coefficients: {
                    #[cfg(feature = "f16")]
                    {
                        let mut coefficients = [0_u32; HALF_SH_COEFF_COUNT];

                        for coefficient in coefficients.iter_mut() {
                            let upper = rng.gen_range(-1.0..1.0);
                            let lower = rng.gen_range(-1.0..1.0);

                            *coefficient = pack_f32s_to_u32(upper, lower);
                        }

                        coefficients
                    }

                    #[cfg(feature = "f32")]
                    {
                        let mut coefficients = [0.0; SH_COEFF_COUNT];

                        for coefficient in coefficients.iter_mut() {
                            *coefficient = rng.gen_range(-1.0..1.0);
                        }

                        coefficients
                    }
                },
            },
        };
        let mut gaussians: Vec<Gaussian> = Vec::new();

        for &x in [-0.5, 0.5].iter() {
            for &y in [-0.5, 0.5].iter() {
                for &z in [-0.5, 0.5].iter() {
                    let mut g = origin;
                    g.position_visibility = [x, y, z, 1.0].into();
                    gaussians.push(g);

                    gaussians.last_mut().unwrap().spherical_harmonic.coefficients.shuffle(&mut rng);
                }
            }
        }

        gaussians.push(gaussians[0]);

        GaussianCloud::from_gaussians(gaussians)
    }
}

impl FromIterator<Gaussian> for GaussianCloud {
    fn from_iter<I: IntoIterator<Item=Gaussian>>(iter: I) -> Self {
        let gaussians = iter.into_iter().collect::<Vec<Gaussian>>();
        GaussianCloud::from_gaussians(gaussians)
    }
}
