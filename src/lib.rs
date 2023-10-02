use bevy::{
    prelude::*,
    ecs::query::QueryItem,
    render::extract_component::ExtractComponent,
};

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
    pub transform: Transform, // TODO: implement global transform
    pub verticies: Handle<GaussianCloud>,
}

impl ExtractComponent for GaussianSplattingBundle {
    type Query = &'static GaussianSplattingBundle;
    type Filter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<'_, Self::Query>) -> Option<Self> {
        Some(GaussianSplattingBundle {
            transform: item.transform,
            verticies: item.verticies.clone(),
        })
    }
}

#[derive(Component, Default)]
struct GaussianSplattingCamera;

// TODO: add render pipeline config
pub struct GaussianSplattingPlugin;

impl Plugin for GaussianSplattingPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<GaussianCloud>();
        app.init_asset_loader::<GaussianCloudLoader>();

        app.add_plugins(RenderPipelinePlugin);
    }
}
