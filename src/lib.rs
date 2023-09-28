use bevy::prelude::*;

use gaussian::{
    GaussianCloud,
    GaussianCloudLoader,
};

pub mod gaussian;
pub mod ply;
pub mod utils;


#[derive(Component, Default)]
pub struct GaussianSplattingBundle {
    pub transform: Transform,
    pub verticies: Handle<GaussianCloud>,
}

// TODO: add render pipeline config
pub struct GaussianSplattingPlugin;

impl Plugin for GaussianSplattingPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<GaussianCloud>();
        app.init_asset_loader::<GaussianCloudLoader>();

        // TODO: setup render pipeline and add GaussianSplattingBundle system
    }
}
