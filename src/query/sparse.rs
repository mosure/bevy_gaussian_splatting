use bevy::{asset::LoadState, prelude::*};
use kd_tree::{KdPoint, KdTree};
use static_assertions::assert_cfg;
use typenum::consts::U3;

use crate::{Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle, query::select::Select};

assert_cfg!(
    all(
        feature = "query_sparse",
        not(feature = "precompute_covariance_3d"),
    ),
    "sparse queries and precomputed covariance are not implemented",
);

#[derive(Component, Debug, Reflect)]
pub struct SparseSelect {
    pub radius: f32,
    pub neighbor_threshold: usize,
    pub completed: bool,
}

impl Default for SparseSelect {
    fn default() -> Self {
        Self {
            radius: 0.05,
            neighbor_threshold: 3,
            completed: false,
        }
    }
}

impl SparseSelect {
    pub fn select(&self, cloud: &PlanarGaussian3d) -> Select {
        let tree = KdTree::build_by_ordered_float(cloud.gaussian_iter().collect());

        cloud
            .gaussian_iter()
            .enumerate()
            .filter(|(_idx, gaussian)| {
                let neighbors = tree.within_radius(gaussian, self.radius);

                neighbors.len() < self.neighbor_threshold
            })
            .map(|(idx, _gaussian)| idx)
            .collect::<Select>()
    }
}

#[derive(Default)]
pub struct SparsePlugin;

impl Plugin for SparsePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<SparseSelect>();

        app.add_systems(Update, select_sparse_handler);
    }
}

impl KdPoint for Gaussian {
    type Scalar = f32;
    type Dim = U3;

    fn at(&self, i: usize) -> Self::Scalar {
        self.position_visibility.position[i]
    }
}

fn select_sparse_handler(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_clouds_res: Res<Assets<PlanarGaussian3d>>,
    mut selections: Query<(Entity, &PlanarGaussian3dHandle, &mut SparseSelect)>,
) {
    for (entity, cloud_handle, mut select) in selections.iter_mut() {
        if Some(LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        if Some(LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        if select.completed {
            continue;
        }
        select.completed = true;

        let cloud = gaussian_clouds_res.get(cloud_handle).unwrap();
        let tree = KdTree::build_by_ordered_float(cloud.gaussian_iter().collect());

        let new_selection = cloud
            .gaussian_iter()
            .enumerate()
            .filter(|(_idx, gaussian)| {
                let neighbors = tree.within_radius(gaussian, select.radius);

                neighbors.len() < select.neighbor_threshold
            })
            .map(|(idx, _gaussian)| idx)
            .collect::<Select>();

        commands
            .entity(entity)
            .remove::<Select>()
            .insert(new_selection);
    }
}
