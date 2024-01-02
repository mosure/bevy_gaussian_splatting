use bevy::{
    asset::load_internal_asset,
    prelude::*,
};


const NOISE_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(125722721);


#[derive(Default)]
pub struct NoisePlugin;

impl Plugin for NoisePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            NOISE_SHADER_HANDLE,
            "noise.wgsl",
            Shader::from_wgsl
        );
    }
}
