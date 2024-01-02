use bevy::{
    prelude::*,
    asset::load_internal_asset,
};


const DEPTH_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(51234253);


pub struct DepthMaterialPlugin;

impl Plugin for DepthMaterialPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            DEPTH_SHADER_HANDLE,
            "depth.wgsl",
            Shader::from_wgsl
        );
    }
}
