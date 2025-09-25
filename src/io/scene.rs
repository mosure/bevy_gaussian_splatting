#[allow(unused_imports)]
use std::io::{BufReader, Cursor, ErrorKind};

use bevy::{
    asset::{AssetLoader, LoadContext, io::Reader},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::gaussian::{
    formats::planar_3d::{PlanarGaussian3d, PlanarGaussian3dHandle},
    settings::CloudSettings,
};

#[derive(Default)]
pub struct GaussianScenePlugin;
impl Plugin for GaussianScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<GaussianScene>();
        app.init_asset::<GaussianScene>();

        app.init_asset_loader::<GaussianSceneLoader>();

        app.add_systems(Update, (spawn_scene,));
    }
}

#[derive(Clone, Debug, Default, Reflect, Serialize, Deserialize)]
pub struct CloudBundle {
    pub asset_path: String,
    pub name: String,
    pub settings: CloudSettings,
    pub transform: Transform,
}

// TODO: support scene hierarchy with gaussian gltf extension
#[derive(Asset, Clone, Debug, Default, Reflect, Serialize, Deserialize)]
pub struct GaussianScene {
    pub bundles: Vec<CloudBundle>,
    pub root: Option<String>,
}

#[derive(Component, Clone, Debug, Default, Reflect)]
#[require(Transform, Visibility)]
pub struct GaussianSceneHandle(pub Handle<GaussianScene>);

#[derive(Component, Clone, Debug, Default, Reflect)]
pub struct GaussianSceneLoaded;

fn spawn_scene(
    mut commands: Commands,
    scene_handles: Query<(Entity, &GaussianSceneHandle), Without<GaussianSceneLoaded>>,
    asset_server: Res<AssetServer>,
    scenes: Res<Assets<GaussianScene>>,
) {
    for (entity, scene_handle) in scene_handles.iter() {
        if let Some(load_state) = &asset_server.get_load_state(&scene_handle.0) {
            if !load_state.is_loaded() {
                continue;
            }
        }

        if scenes.get(&scene_handle.0).is_none() {
            continue;
        }

        let scene = scenes.get(&scene_handle.0).unwrap();

        let bundles = scene
            .bundles
            .iter()
            .map(|bundle| {
                let root = scene.root.clone().unwrap_or_default();

                // TODO: switch between 3d and 4d clouds based on settings
                (
                    PlanarGaussian3dHandle(asset_server.load::<PlanarGaussian3d>(
                        bundle.asset_path.clone().replace("{root}", &root),
                    )),
                    Name::new(bundle.name.clone()),
                    bundle.settings.clone(),
                    bundle.transform,
                )
            })
            .collect::<Vec<_>>();

        commands
            .entity(entity)
            .with_children(move |builder| {
                for bundle in bundles {
                    builder.spawn(bundle);
                }
            })
            .insert(GaussianSceneLoaded);
    }
}

#[derive(Default)]
pub struct GaussianSceneLoader;

impl AssetLoader for GaussianSceneLoader {
    type Asset = GaussianScene;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;

        match load_context.path().extension() {
            Some(ext) if ext == "json" => {
                let mut scene: GaussianScene = serde_json::from_slice(&bytes)
                    .map_err(|err| std::io::Error::new(ErrorKind::InvalidData, err))?;

                scene.root = load_context
                    .path()
                    .parent()
                    .expect("invalid scene path")
                    .to_string_lossy()
                    .to_string()
                    .into();

                Ok(scene)
            }
            _ => Err(std::io::Error::other("only .json supported")),
        }
    }

    fn extensions(&self) -> &[&str] {
        &["json"]
    }
}
