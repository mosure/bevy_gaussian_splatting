use bevy::{
    prelude::*,
    asset::LoadState,
    utils::Instant,
};

use crate::{
    GaussianCloud,
    GaussianCloudSettings,
    sort::{
        SortedEntries,
        SortMode,
    },
};


#[derive(Default)]
pub struct StdSortPlugin;

impl Plugin for StdSortPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, std_sort);
    }
}

// TODO: async CPU sort to prevent frame drops on large clouds
#[allow(clippy::too_many_arguments)]
pub fn std_sort(
    asset_server: Res<AssetServer>,
    gaussian_clouds_res: Res<Assets<GaussianCloud>>,
    mut sorted_entries_res: ResMut<Assets<SortedEntries>>,
    gaussian_clouds: Query<(
        &Handle<GaussianCloud>,
        &Handle<SortedEntries>,
        &GaussianCloudSettings,
    )>,
    cameras: Query<(
        &GlobalTransform,
        &Camera3d,
    )>,
    mut last_camera_position: Local<Vec3>,
    mut last_sort_time: Local<Option<Instant>>,
    mut period: Local<std::time::Duration>,
    mut camera_debounce: Local<bool>,
    mut sort_done: Local<bool>,
) {
    if last_sort_time.is_none() {
        *period = std::time::Duration::from_millis(100);
    }

    if let Some(last_sort_time) = last_sort_time.as_ref() {
        if last_sort_time.elapsed() < *period {
            return;
        }
    }

    let sort_start_time = Instant::now();
    let mut performed_sort = false;

    for (
        camera_transform,
        _camera,
    ) in cameras.iter() {
        let camera_position = camera_transform.compute_transform().translation;
        let camera_movement = *last_camera_position != camera_position;

        if camera_movement {
            *sort_done = false;
            *camera_debounce = true;
        } else if *sort_done {
            return;
        }

        if *camera_debounce {
            *last_camera_position = camera_position;
            *camera_debounce = false;
            return;
        }

        for (
            gaussian_cloud_handle,
            sorted_entries_handle,
            settings,
        ) in gaussian_clouds.iter() {
            if settings.sort_mode != SortMode::Std {
                continue;
            }

            if Some(LoadState::Loading) == asset_server.get_load_state(gaussian_cloud_handle) {
                continue;
            }

            if Some(LoadState::Loading) == asset_server.get_load_state(sorted_entries_handle) {
                continue;
            }

            if let Some(gaussian_cloud) = gaussian_clouds_res.get(gaussian_cloud_handle) {
                if let Some(sorted_entries) = sorted_entries_res.get_mut(sorted_entries_handle) {
                    assert_eq!(gaussian_cloud.len(), sorted_entries.sorted.len());

                    *sort_done = true;
                    *last_sort_time = Some(Instant::now());

                    performed_sort = true;

                    gaussian_cloud.position_iter()
                        .zip(sorted_entries.sorted.iter_mut())
                        .enumerate()
                        .for_each(|(idx, (position, sort_entry))| {
                            let position = Vec3::from_slice(position.as_ref());
                            let delta = camera_position - position;

                            sort_entry.key = bytemuck::cast(delta.length_squared());
                            sort_entry.index = idx as u32;
                        });

                    sorted_entries.sorted.sort_unstable_by(|a, b| {
                        bytemuck::cast::<u32, f32>(b.key).partial_cmp(&bytemuck::cast::<u32, f32>(a.key)).unwrap()
                    });

                    // TODO: update DrawIndirect buffer during sort phase (GPU sort will override default DrawIndirect)
                }
            }
        }
    }

    let sort_end_time = Instant::now();
    let delta = sort_end_time - sort_start_time;

    if performed_sort {
        *period = std::time::Duration::from_millis(
            100
                .max(period.as_millis() as u64 * 4 / 5)
                .max(10 * delta.as_millis() as u64)
        );
    }
}
