use bevy::{
    prelude::*,
    asset::LoadState,
};

use crate::GaussianCloud;


#[derive(Component, Debug, Default, Reflect)]
pub struct Select {
    pub indicies: Vec<usize>,
}


#[derive(Default)]
pub struct SelectPlugin;

impl Plugin for SelectPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, apply_selection);
    }
}


fn apply_selection(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut gaussian_clouds_res: ResMut<Assets<GaussianCloud>>,
    mut selections: Query<(
        Entity,
        &Handle<GaussianCloud>,
        &Select,
    )>,
) {
    for (
        entity,
        cloud_handle,
        select,
    ) in selections.iter_mut() {
        if select.indicies.is_empty() {
            continue;
        }

        if Some(LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        if Some(LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        let cloud = gaussian_clouds_res.get_mut(cloud_handle).unwrap();

        cloud.gaussians.iter_mut()
            .for_each(|gaussian| {
                gaussian.position_visibility[3] = 0.0;
            });

        select.indicies.iter()
            .for_each(|index| {
                cloud.gaussians[*index].position_visibility[3] = 1.0;
            });

        commands.entity(entity).remove::<Select>();
    }
}
