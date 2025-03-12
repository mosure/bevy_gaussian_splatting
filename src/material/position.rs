use bevy::{
    prelude::*,
    asset::load_internal_asset,
};


const POSITION_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(62346645534);


pub struct PositionMaterialPlugin;

impl Plugin for PositionMaterialPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            POSITION_SHADER_HANDLE,
            "position.wgsl",
            Shader::from_wgsl
        );
    }
}
