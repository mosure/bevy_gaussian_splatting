use bevy::{
    prelude::*,
    asset::load_internal_asset,
};


const CLASSIFICATION_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(61436234324);


pub struct ClassificationMaterialPlugin;

impl Plugin for ClassificationMaterialPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            CLASSIFICATION_SHADER_HANDLE,
            "classification.wgsl",
            Shader::from_wgsl
        );
    }
}
