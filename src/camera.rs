use bevy::{
    prelude::*,
    render::extract_component::{
        ExtractComponent,
        ExtractComponentPlugin,
    },
};


#[derive(
    Clone,
    Component,
    Debug,
    Default,
    ExtractComponent,
    Reflect,
)]
pub struct GaussianCamera {
    pub warmup: bool,
}


#[derive(Default)]
pub struct GaussianCameraPlugin;

impl Plugin for GaussianCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<GaussianCamera>::default());

        app.add_systems(Update, apply_camera_warmup);
    }
}


fn apply_camera_warmup(
    mut cameras: Query<&mut GaussianCamera>,
) {
    for mut camera in cameras.iter_mut() {
        if camera.warmup {
            info!("camera warmup...");
            camera.warmup = false;
        }
    }
}
