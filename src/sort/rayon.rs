use bevy::{
    prelude::*,
    asset::LoadState,
    math::Vec3A,
    utils::Instant,
};
use rayon::prelude::*;

use crate::{
    camera::GaussianCamera,
    GaussianCloud,
    GaussianCloudSettings,
    sort::{
        SortConfig,
        SortMode,
        SortTrigger,
        SortedEntries,
    },
};


#[derive(Default)]
pub struct RayonSortPlugin;

impl Plugin for RayonSortPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, rayon_sort);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rayon_sort(
    asset_server: Res<AssetServer>,
    gaussian_clouds_res: Res<Assets<GaussianCloud>>,
    gaussian_clouds: Query<(
        &Handle<GaussianCloud>,
        &Handle<SortedEntries>,
        &GaussianCloudSettings,
    )>,
    mut sorted_entries_res: ResMut<Assets<SortedEntries>>,
    mut cameras: Query<
        &mut SortTrigger,
        With<GaussianCamera>,
    >,
    mut sort_config: ResMut<SortConfig>,
) {
    // TODO: move sort to render world, use extracted views and update the existing buffer instead of creating new

    let sort_start_time = Instant::now();
    let mut performed_sort = false;

    for mut trigger in cameras.iter_mut() {
        if !trigger.needs_sort {
            continue;
        }

        for (
            gaussian_cloud_handle,
            sorted_entries_handle,
            settings,
        ) in gaussian_clouds.iter() {
            if settings.sort_mode != SortMode::Rayon {
                continue;
            }

            trigger.needs_sort = false;
            performed_sort = true;

            if Some(LoadState::Loading) == asset_server.get_load_state(gaussian_cloud_handle) {
                continue;
            }

            if Some(LoadState::Loading) == asset_server.get_load_state(sorted_entries_handle) {
                continue;
            }

            if let Some(gaussian_cloud) = gaussian_clouds_res.get(gaussian_cloud_handle) {
                if let Some(sorted_entries) = sorted_entries_res.get_mut(sorted_entries_handle) {
                    let gaussians = gaussian_cloud.len();
                    let mut chunks = sorted_entries.sorted.chunks_mut(gaussians);
                    let chunk = chunks.nth(trigger.camera_index).unwrap();

                    gaussian_cloud.position_par_iter()
                        .zip(chunk.par_iter_mut())
                        .enumerate()
                        .for_each(|(idx, (position, sort_entry))| {
                            let position = Vec3A::from_slice(position.as_ref());
                            let position = settings.transform.compute_affine().transform_point3a(position);

                            let delta = trigger.last_camera_position - position;

                            sort_entry.key = bytemuck::cast(delta.length_squared());
                            sort_entry.index = idx as u32;
                        });

                    chunk.par_sort_unstable_by(|a, b| {
                        bytemuck::cast::<u32, f32>(b.key).partial_cmp(&bytemuck::cast::<u32, f32>(a.key)).unwrap_or(std::cmp::Ordering::Equal)
                    });

                    // TODO: update DrawIndirect buffer during sort phase (GPU sort will override default DrawIndirect)
                }
            }
        }
    }

    let sort_end_time = Instant::now();
    let delta = sort_end_time - sort_start_time;

    if performed_sort {
        sort_config.period_ms = sort_config.period_ms
            .max(sort_config.period_ms * 4 / 5)
            .max(4 * delta.as_millis() as usize);
    }
}
