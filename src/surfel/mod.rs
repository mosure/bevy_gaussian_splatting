use bevy::{
    prelude::*,
    render::render_resource::TextureFormat,
};
use bevy_interleave::prelude::*;


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


// TODO: render pipeline for surfel rendering
