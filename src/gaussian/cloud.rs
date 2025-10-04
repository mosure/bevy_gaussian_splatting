use bevy::{
    camera::visibility::{NoFrustumCulling, VisibilitySystems, VisibilityClass, add_visibility_class},
    ecs::{lifecycle::HookContext, world::DeferredWorld},
    math::bounding::Aabb3d,
    prelude::*,
};
use bevy_interleave::prelude::*;

use crate::gaussian::interface::CommonCloud;

#[derive(Component, Clone, Debug)]
pub struct GaussianCloudAabb(pub Aabb3d);

impl From<Aabb3d> for GaussianCloudAabb {
    fn from(aabb: Aabb3d) -> Self {
        GaussianCloudAabb(aabb)
    }
}

impl std::ops::Deref for GaussianCloudAabb {
    type Target = Aabb3d;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Default)]
pub struct CloudPlugin<R: PlanarSync> {
    _phantom: std::marker::PhantomData<R>,
}

pub struct CloudVisibilityClass;

fn add_planar_class(world: DeferredWorld, ctx: HookContext) {
    add_visibility_class::<CloudVisibilityClass>(world, ctx);
}

impl<R: PlanarSync + Reflect + TypePath> Plugin for CloudPlugin<R>
where
    R::PlanarType: CommonCloud,
    R::PlanarTypeHandle: FromReflect + bevy::reflect::Typed,
{
    fn build(&self, app: &mut App) {
        app.register_required_components::<R::PlanarTypeHandle, VisibilityClass>();
        app.world_mut()
            .register_component_hooks::<R::PlanarTypeHandle>()
            .on_add(add_planar_class);

        app.add_systems(
            PostUpdate,
            (calculate_bounds::<R>.in_set(VisibilitySystems::CalculateBounds),),
        );
    }
}

// TODO: handle aabb updates (e.g. gaussian particle movements)
#[allow(clippy::type_complexity)]
pub fn calculate_bounds<R: PlanarSync>(
    mut commands: Commands,
    gaussian_clouds: Res<Assets<R::PlanarType>>,
    without_aabb: Query<(Entity, &R::PlanarTypeHandle), (Without<GaussianCloudAabb>, Without<NoFrustumCulling>)>,
) where
    R::PlanarType: CommonCloud,
{
    for (entity, cloud_handle) in &without_aabb {
        if let Some(cloud) = gaussian_clouds.get(cloud_handle.handle()) {
            if let Some(aabb) = cloud.compute_aabb() {
                commands.entity(entity).try_insert(GaussianCloudAabb(aabb));
            }
        }
    }
}
