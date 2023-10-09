use bevy::prelude::*;

use gaussian::{
    GaussianCloud,
    GaussianCloudLoader,
    GaussianCloudSettings,
};

use render::RenderPipelinePlugin;

pub mod gaussian;
pub mod ply;
pub mod render;
pub mod utils;


#[derive(Bundle, Default, Reflect)]
pub struct GaussianSplattingBundle {
    pub settings: GaussianCloudSettings, // TODO: implement global transform
    pub verticies: Handle<GaussianCloud>,
}


#[derive(Component, Default)]
struct GaussianSplattingCamera;
// TODO: filter camera 3D entities

pub struct GaussianSplattingPlugin;

impl Plugin for GaussianSplattingPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<GaussianCloud>();
        app.init_asset_loader::<GaussianCloudLoader>();

        app.register_type::<GaussianSplattingBundle>();

        app.add_plugins((
            RenderPipelinePlugin,
        ));
    }
}
