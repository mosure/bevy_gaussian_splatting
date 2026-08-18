use bevy::{
    camera::{
        primitives::Aabb,
        visibility::{VisibilityClass, VisibilitySystems, add_visibility_class},
    },
    ecs::{lifecycle::HookContext, world::DeferredWorld},
    math::bounding::BoundingVolume,
    prelude::*,
};
use bevy_interleave::prelude::*;

use crate::gaussian::interface::CommonCloud;

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
    without_aabb: Query<(Entity, &R::PlanarTypeHandle), Without<Aabb>>,
) where
    R::PlanarType: CommonCloud,
{
    for (entity, cloud_handle) in &without_aabb {
        if let Some(cloud) = gaussian_clouds.get(cloud_handle.handle())
            && let Some(aabb3d) = cloud.compute_aabb()
        {
            commands.entity(entity).try_insert(Aabb {
                center: aabb3d.center(),
                half_extents: aabb3d.half_size(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{asset::AssetPlugin, camera::visibility::NoFrustumCulling};

    use crate::gaussian::formats::planar_3d::{
        Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle,
    };

    #[test]
    fn no_frustum_culling_cloud_still_gets_renderer_bounds() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(AssetPlugin::default())
            .init_asset::<PlanarGaussian3d>()
            .add_plugins(CloudPlugin::<Gaussian3d>::default());

        let cloud = PlanarGaussian3d::from(vec![Gaussian3d {
            position_visibility: [1.0, 2.0, 3.0, 1.0].into(),
            scale_opacity: [0.5, 0.25, 0.125, 1.0].into(),
            ..default()
        }]);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(cloud);
        let entity = app
            .world_mut()
            .spawn((PlanarGaussian3dHandle(handle), NoFrustumCulling))
            .id();

        app.update();

        assert!(app.world().entity(entity).contains::<NoFrustumCulling>());
        assert!(
            app.world().entity(entity).contains::<Aabb>(),
            "NoFrustumCulling must skip rejection without suppressing the Aabb required by Gaussian extraction and phase sorting"
        );
    }
}
