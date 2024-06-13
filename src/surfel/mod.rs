use bevy::{
    prelude::*,
    render::render_resource::TextureFormat,
};
use bevy_interleave::prelude::*;


// TODO: automate extraction and asset loading via derive macro
// TODO: automate plugin & render pipeline
#[derive(
    Debug,
    Planar,
    ReflectInterleaved,
    StorageBindings,
    TextureBindings,
)]
pub struct SurfelGaussian {
    #[texture_format(TextureFormat::Rgba32Uint)]
    pub position_opacity: [f32; 4],

    #[texture_format(TextureFormat::Rgba32Uint)]
    pub tangent_s: [f32; 4],

    #[texture_format(TextureFormat::Rgba32Uint)]
    pub tangent_t: [f32; 4],

    #[texture_format(TextureFormat::Rgba32Uint)]
    pub scale: [f32; 4],
}


pub struct SurfelRenderPipelinePlugin;
impl Plugin for SurfelRenderPipelinePlugin {
    fn build(&self, _app: &mut App) {

    }

    fn finish(&self, _app: &mut App) {

    }
}


// TODO: extract surfel asset
// TODO: queue surfel clouds
// TODO: render surfel node
