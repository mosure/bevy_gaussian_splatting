use bevy::{
    prelude::*,
    reflect::TypeUuid,
};

pub mod utils;


#[derive(Clone, Debug, Reflect)]
pub struct Gaussian {
    pub color: Color,
    pub transform: Transform,
    // TODO: support gaussian animations (e.g. switching between different times of day, in different regions of a scene)
}

#[derive(Clone, Debug, Reflect, TypeUuid)]
#[uuid = "ac2f08eb-bc32-aabb-ff21-51571ea332d5"]
pub struct GaussianCloud(Vec<Gaussian>);

#[derive(Component, Default)]
pub struct GaussianSplattingBundle {
    pub transform: Transform,
    pub verticies: Handle<GaussianCloud>,
}

// TODO: add render pipeline config
pub struct GaussianSplattingPlugin;

impl Plugin for GaussianSplattingPlugin {
    fn build(&self, _app: &mut App) {
        // TODO: setup render pipeline and add GaussianSplattingBundle system
    }
}
