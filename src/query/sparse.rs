use bevy::prelude::*;

use kd_tree::KdTree;


// TODO: select sparse points based on radius
// use kdtree - https://crates.io/crates/kd-treeuse bevy::prelude::*;


#[derive(Component, Debug, Default, Reflect)]
pub struct Sparse {
    pub enabled: bool,
}


#[derive(Default)]
pub struct SparsePlugin;

impl Plugin for SparsePlugin {
    fn build(&self, app: &mut App) {

    }
}


fn select_sparse(

) {

}
