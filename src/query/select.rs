use std::iter::FromIterator;

use bevy::{
    prelude::*,
    asset::LoadState,
};

use crate::{Cloud, io::writer::write_gaussian_cloud_to_file};


#[derive(Component, Debug, Default, Reflect)]
pub struct Select {
    pub indicies: Vec<usize>,
    pub completed: bool,
}

impl FromIterator<usize> for Select {
    fn from_iter<I: IntoIterator<Item=usize>>(iter: I) -> Self {
        let indicies = iter.into_iter().collect::<Vec<usize>>();
        Select { indicies, ..Default::default() }
    }
}

impl Select {
    pub fn invert(&mut self, cloud_size: usize) -> Select {
        let inverted = (0..cloud_size)
            .filter(|index| !self.indicies.contains(index))
            .collect::<Vec<usize>>();

        Select {
            indicies: inverted,
            completed: self.completed,
        }
    }
}


#[derive(Default)]
pub struct SelectPlugin;

impl Plugin for SelectPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Select>();

        app.add_event::<InvertSelectionEvent>();
        app.add_event::<SaveSelectionEvent>();

        app.add_systems(Update, (
            apply_selection,
            invert_selection,
            save_selection,
        ));
    }
}


fn apply_selection(
    asset_server: Res<AssetServer>,
    mut gaussian_clouds_res: ResMut<Assets<Cloud>>,
    mut selections: Query<(
        Entity,
        &Handle<Cloud>,
        &mut Select,
    )>,
) {
    for (
        _entity,
        cloud_handle,
        mut select,
    ) in selections.iter_mut() {
        if select.indicies.is_empty() || select.completed {
            continue;
        }

        if Some(LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        if Some(LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        let cloud = gaussian_clouds_res.get_mut(cloud_handle).unwrap();

        (0..cloud.len())
            .for_each(|index| {
                *cloud.visibility_mut(index) = 0.0;
            });

        select.indicies.iter()
            .for_each(|index| {
                *cloud.visibility_mut(*index) = 1.0;
            });

        select.completed = true;
    }
}



#[derive(Event, Debug, Reflect)]
pub struct InvertSelectionEvent;

fn invert_selection(
    mut events: EventReader<InvertSelectionEvent>,
    mut gaussian_clouds_res: ResMut<Assets<Cloud>>,
    mut selections: Query<(
        Entity,
        &Handle<Cloud>,
        &mut Select,
    )>,
) {
    if events.is_empty() {
        return;
    }
    events.clear();

    for (
        _entity,
        cloud_handle,
        mut select,
    ) in selections.iter_mut() {
        if select.indicies.is_empty() {
            continue;
        }

        let cloud = gaussian_clouds_res.get_mut(cloud_handle).unwrap();

        let mut new_indicies = Vec::with_capacity(cloud.len() - select.indicies.len());

        (0..cloud.len())
            .for_each(|index| {
                if cloud.visibility(index) == 0.0 {
                    new_indicies.push(index);
                }

                *cloud.visibility_mut(index) = 1.0;
            });

        select.indicies.iter()
            .for_each(|index| {
                *cloud.visibility_mut(*index) = 0.0;
            });

        select.indicies = new_indicies;
    }
}


#[derive(Event, Debug, Reflect)]
pub struct SaveSelectionEvent;

pub fn save_selection(
    mut events: EventReader<SaveSelectionEvent>,
    mut gaussian_clouds_res: ResMut<Assets<Cloud>>,
    mut selections: Query<(
        Entity,
        &Handle<Cloud>,
        &Select,
    )>,
) {
    if events.is_empty() {
        return;
    }
    events.clear();

    for (
        _entity,
        cloud_handle,
        select,
    ) in selections.iter_mut() {
        let cloud = gaussian_clouds_res.get_mut(cloud_handle).unwrap();

        let selected = cloud.subset(select.indicies.as_slice());

        write_gaussian_cloud_to_file(&selected, "live_output.gcloud");
    }
}
