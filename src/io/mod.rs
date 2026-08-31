use bevy::prelude::*;

pub mod codec;
pub mod gcloud;
pub mod loader;
#[cfg(any(feature = "lod", feature = "lod_build"))]
pub mod lod;
#[cfg(all(feature = "lod_build", not(target_arch = "wasm32")))]
pub mod lod_build_external;
#[cfg(any(feature = "lod", feature = "lod_build"))]
pub mod lodge;
#[cfg(all(feature = "lod_build", not(target_arch = "wasm32")))]
pub mod lodge_build_external;
pub mod scene;

#[cfg(feature = "io_ply")]
pub mod ply;

#[derive(Default)]
pub struct IoPlugin;
impl Plugin for IoPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset_loader::<loader::Gaussian3dLoader>();
        app.init_asset_loader::<loader::Gaussian4dLoader>();
        #[cfg(feature = "lod")]
        {
            app.init_asset::<lod::GaussianLodAsset>();
            app.init_asset_loader::<lod::GaussianLodManifestLoader>();
            app.init_asset::<lodge::GaussianLodgeAsset>();
            app.init_asset_loader::<lodge::GaussianLodgeManifestLoader>();
        }

        app.add_plugins(scene::GaussianScenePlugin);
    }
}
