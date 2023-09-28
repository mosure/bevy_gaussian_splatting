use bevy::prelude::*;

use gaussian::{
    GaussianCloud,
    GaussianCloudLoader,
};

use render::RenderPipelinePlugin;

pub mod gaussian;
pub mod ply;
pub mod render;
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

        app.add_plugins(RenderPipelinePlugin);

        // TODO: add GaussianSplattingBundle system
    }
}
