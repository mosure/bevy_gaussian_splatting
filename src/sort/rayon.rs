use bevy::{
    prelude::*,
    asset::LoadState,
    utils::Instant,
};

use rayon::prelude::*;

use crate::{
    GaussianCloud,
    GaussianCloudSettings,
    sort::{
        SortedEntries,
        SortMode,
    },
};


#[derive(Default)]
pub struct RayonSortPlugin;

impl Plugin for RayonSortPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, rayon_sort);
    }
}

pub fn rayon_sort(
    asset_server: Res<AssetServer>,
    mut gaussian_clouds_res: ResMut<Assets<GaussianCloud>>,
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
) {
    let period = std::time::Duration::from_millis(1000);
    if let Some(last_sort_time) = last_sort_time.as_ref() {
        if last_sort_time.elapsed() < period {
            return;
        }
    }

    *last_sort_time = Some(Instant::now());

    // TODO: move sort to render world, use extracted views and update the existing buffer instead of creating new

    for (
        camera_transform,
        _camera,
    ) in cameras.iter() {
        let camera_position = camera_transform.compute_transform().translation;
        if *last_camera_position == camera_position {
            return;
        }
        *last_camera_position = camera_position;

        for (
            gaussian_cloud_handle,
            sorted_entries_handle,
            settings,
        ) in gaussian_clouds.iter() {
            if settings.sort_mode != SortMode::Rayon {
                continue;
            }

            if Some(LoadState::Loading) == asset_server.get_load_state(gaussian_cloud_handle) {
                continue;
            }

            if Some(LoadState::Loading) == asset_server.get_load_state(sorted_entries_handle) {
                continue;
            }

            if let Some(gaussian_cloud) = gaussian_clouds_res.get_mut(gaussian_cloud_handle) {
                if let Some(sorted_entries) = sorted_entries_res.get_mut(sorted_entries_handle) {
                    assert_eq!(gaussian_cloud.gaussians.len(), sorted_entries.sorted.len());

                    gaussian_cloud.gaussians.par_iter()
                        .zip(sorted_entries.sorted.par_iter_mut())
                        .enumerate()
                        .for_each(|(idx, (gaussian, sort_entry))| {
                            let position = Vec3::from_slice(gaussian.position.as_ref());
                            let delta = camera_position - position;

                            sort_entry.key = bytemuck::cast(delta.length_squared());
                            sort_entry.index = idx as u32;
                        });

                    sorted_entries.sorted.par_sort_unstable_by(|a, b| {
                        bytemuck::cast::<u32, f32>(b.key).partial_cmp(&bytemuck::cast::<u32, f32>(a.key)).unwrap()
                    });
                }
            }
        }
    }
}
