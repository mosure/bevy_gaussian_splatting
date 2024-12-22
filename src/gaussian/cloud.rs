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

use crate::gaussian::{
    f32::Positions,
    formats::{
        cloud_3d::Cloud3d,
        cloud_4d::Cloud4d,
    },
    interface::CommonCloud,
    packed::Gaussian,
    settings::CloudSettings,
};


// TODO: support packed vs. planar switch at runtime
#[derive(
    Asset,
    Debug,
    PartialEq,
    Reflect,
    Serialize,
    Deserialize,
)]
pub enum Cloud {
    Gaussian2d(Cloud3d),
    // QuantizedGaussian2d(HalfCloud3d),
    Gaussian3d(Cloud3d),
    // QuantizedGaussian3d(HalfCloud3d),
    Gaussian4d(Cloud4d),
    // QuantizedGaussian4d(HalfCloud4d),
}

// TODO: enum macro /w support for associated type
impl CommonCloud for Cloud {
    // default to gaussian 3d
    type PackedType = Gaussian;

    fn len(&self) -> usize {
        match self {
            Self::Gaussian2d(cloud) => cloud.len(),
            Self::Gaussian3d(cloud) => cloud.len(),
            Self::Gaussian4d(cloud) => cloud.len(),
        }
    }

    fn position_iter(&self) -> Positions<'_> {
        match self {
            Self::Gaussian2d(cloud) => cloud.position_iter(),
            Self::Gaussian3d(cloud) => cloud.position_iter(),
            Self::Gaussian4d(cloud) => cloud.position_iter(),
        }
    }

    #[cfg(feature = "sort_rayon")]
    fn position_par_iter(&self) -> impl rayon::prelude::IndexedParallelIterator<Item = &super::f32::Position> + '_ {
        match self {
            Self::Gaussian2d(cloud) => cloud.position_par_iter(),
            Self::Gaussian3d(cloud) => cloud.position_par_iter(),
            Self::Gaussian4d(cloud) => cloud.position_par_iter(),
        }
    }

    fn subset(&self, indicies: &[usize]) -> Self {
        match self {
            Self::Gaussian2d(cloud) => Self::Gaussian2d(cloud.subset(indicies)),
            Self::Gaussian3d(cloud) => Self::Gaussian3d(cloud.subset(indicies)),
            Self::Gaussian4d(cloud) => Self::Gaussian4d(cloud.subset(indicies)),
        }
    }

    fn from_packed(packed_array: Vec<Self::PackedType>) -> Self {
        Self::Gaussian3d(Cloud3d::from_packed(packed_array))
    }

    fn visibility(&self, index: usize) -> f32 {
        match self {
            Self::Gaussian2d(cloud) => cloud.visibility(index),
            Self::Gaussian3d(cloud) => cloud.visibility(index),
            Self::Gaussian4d(cloud) => cloud.visibility(index),
        }
    }

    fn visibility_mut(&mut self, index: usize) -> &mut f32 {
        match self {
            Self::Gaussian2d(cloud) => cloud.visibility_mut(index),
            Self::Gaussian3d(cloud) => cloud.visibility_mut(index),
            Self::Gaussian4d(cloud) => cloud.visibility_mut(index),
        }
    }

    fn resize_to_square(&mut self) {
        match self {
            Self::Gaussian2d(cloud) => cloud.resize_to_square(),
            Self::Gaussian3d(cloud) => cloud.resize_to_square(),
            Self::Gaussian4d(cloud) => cloud.resize_to_square(),
        }
    }
}


#[derive(Default)]
pub struct CloudPlugin;

impl Plugin for CloudPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Cloud>();
        app.register_asset_reflect::<Cloud>();
        app.register_type::<Cloud>();

        app.register_type::<CloudHandle>();
        app.register_type::<CloudSettings>();

        app.add_systems(
            PostUpdate,
            (
                calculate_bounds.in_set(VisibilitySystems::CalculateBounds),
                check_visibility::<With<CloudHandle>>.in_set(VisibilitySystems::CheckVisibility),
            )
        );
    }
}


// TODO: handle aabb updates (e.g. gaussian particle movements)
#[allow(clippy::type_complexity)]
pub fn calculate_bounds(
    mut commands: Commands,
    gaussian_clouds: Res<Assets<Cloud>>,
    without_aabb: Query<
        (
            Entity,
            &CloudHandle,
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
    CloudSettings,
    SyncToRenderWorld,
    Transform,
    Visibility,
)]
pub struct CloudHandle(pub Handle<Cloud>);

impl From<Handle<Cloud>> for CloudHandle {
    fn from(handle: Handle<Cloud>) -> Self {
        Self(handle)
    }
}

impl From<CloudHandle> for AssetId<Cloud> {
    fn from(handle: CloudHandle) -> Self {
        handle.0.id()
    }
}

impl From<&CloudHandle> for AssetId<Cloud> {
    fn from(handle: &CloudHandle) -> Self {
        handle.0.id()
    }
}
