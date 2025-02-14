use bevy::{
    prelude::*,
    render::{
        primitives::Aabb,
        view::visibility::{
            check_visibility,
            NoFrustumCulling,
            VisibilitySystems,
        },
    },
};
use bevy_interleave::prelude::*;

use crate::gaussian::interface::CommonCloud;


#[derive(Default)]
pub struct CloudPlugin<R: PlanarStorage> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: PlanarStorage + Reflect + TypePath> Plugin for CloudPlugin<R>
where
    R::PlanarType: CommonCloud,
    R::PlanarTypeHandle: FromReflect + bevy::reflect::Typed,
{
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            (
                calculate_bounds::<R>.in_set(VisibilitySystems::CalculateBounds),
                check_visibility::<With<R::PlanarTypeHandle>>.in_set(VisibilitySystems::CheckVisibility),
            )
        );
    }
}


// TODO: handle aabb updates (e.g. gaussian particle movements)
#[allow(clippy::type_complexity)]
pub fn calculate_bounds<R: PlanarStorage>(
    mut commands: Commands,
    gaussian_clouds: Res<Assets<R::PlanarType>>,
    without_aabb: Query<
        (
            Entity,
            &R::PlanarTypeHandle,
        ),
        (
            Without<Aabb>,
            Without<NoFrustumCulling>,
        ),
    >,
)
where
    R::PlanarType: CommonCloud,
{
    for (entity, cloud_handle) in &without_aabb {
        if let Some(cloud) = gaussian_clouds.get(cloud_handle.handle()) {
            if let Some(aabb) = cloud.compute_aabb() {
                commands.entity(entity).try_insert(aabb);
            }
        }
    }
}
