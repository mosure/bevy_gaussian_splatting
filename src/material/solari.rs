use bevy::{
    asset::{load_internal_asset, weak_handle},
    ecs::query::Or,
    pbr::MeshMaterial3d,
    prelude::*,
    solari::prelude::*,
};

use crate::mesh::proxy::{ProxyMesh, ProxyMeshSettings, ProxyParams};

const SOLARI_SHADER_HANDLE: Handle<Shader> = weak_handle!("6beaa8d4-026e-43a3-8f6b-13213e63ed05");

pub struct SolariMaterialPlugin;

impl Plugin for SolariMaterialPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SOLARI_SHADER_HANDLE, "solari.wgsl", Shader::from_wgsl);

        app.register_type::<GaussianCloudSolari>();

        app.add_systems(
            PostUpdate,
            (
                ensure_proxy_settings,
                attach_solari_proxies.after(ensure_proxy_settings),
            ),
        );
    }
}

#[derive(Component, Default, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct GaussianCloudSolari;

fn ensure_proxy_settings(
    mut commands: Commands,
    query: Query<Entity, (With<GaussianCloudSolari>, Without<ProxyMeshSettings>)>,
) {
    for entity in &query {
        commands
            .entity(entity)
            .insert(ProxyMeshSettings {
                params: ProxyParams::default(),
            });
    }
}

fn attach_solari_proxies(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<
        (
            Entity,
            &ProxyMesh,
            Option<&RaytracingMesh3d>,
            Option<&MeshMaterial3d<StandardMaterial>>,
        ),
        (With<GaussianCloudSolari>, Or<(Changed<ProxyMesh>, Added<GaussianCloudSolari>)>),
    >,
) {
    for (entity, mesh_proxy, raytracing_mesh, mesh_material) in &query {
        if raytracing_mesh.is_none() {
            commands
                .entity(entity)
                .insert(RaytracingMesh3d(mesh_proxy.0.clone()));
        }

        if mesh_material.is_none() {
            let material_handle = materials.add(StandardMaterial {
                base_color: Color::WHITE.into(),
                perceptual_roughness: 0.9,
                metallic: 0.0,
                ..default()
            });

            commands
                .entity(entity)
                .insert(MeshMaterial3d::<StandardMaterial>(material_handle));
        }
    }
}
