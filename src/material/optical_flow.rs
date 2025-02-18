use bevy::{
    prelude::*,
    asset::load_internal_asset,
};


const OPTICAL_FLOW_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(1451151234);


pub struct OpticalFlowMaterialPlugin;

impl Plugin for OpticalFlowMaterialPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            OPTICAL_FLOW_SHADER_HANDLE,
            "optical_flow.wgsl",
            Shader::from_wgsl
        );
    }
}
