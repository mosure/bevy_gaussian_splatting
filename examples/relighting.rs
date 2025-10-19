//! Demonstrates realtime dynamic raytraced lighting with Gaussian Splatting.

#[path = "relighting/camera_controller.rs"]
mod camera_controller;

use bevy::{
    camera::CameraMainTextureUsages, gltf::GltfMaterialName, log::LogPlugin, prelude::*,
    render::render_resource::TextureUsages, scene::SceneInstanceReady
};
use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, GaussianSplattingPlugin, PlanarGaussian3dHandle, RasterizeMode,
};
use bevy_gaussian_splatting::material::gaussian_material::{
    GaussianMaterial, GaussianMaterialHandle, GaussianTextureProjection,
};
use camera_controller::{CameraController, CameraControllerPlugin};
use std::f32::consts::PI;

#[derive(Component)]
struct DioramaTag;

#[derive(Resource, Clone, Copy)]
struct IcecreamPos(pub Vec3);

#[derive(Resource)]
struct GaussianMaterialCycle {
    material_handle: Handle<GaussianMaterial>,
    textures: Vec<Handle<Image>>,
    current: usize,
    timer: Timer,
}

const TEXTURE_SWAP_SECONDS: f32 = 10.0;

fn main() {
    let mut app = App::new();

    app.add_plugins((
        DefaultPlugins.set(LogPlugin{
            filter: "wgpu=error,naga=warn,bevy_gaussian_splatting=debug,bevy_render=info,bevy_asset=info".to_string(),
            ..default()
        }),
        GaussianSplattingPlugin,
        CameraControllerPlugin
    ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                pause_scene,
                toggle_lights,
                patrol_path,
                cycle_gaussian_material_textures,
            ),
        )
        .add_systems(PostUpdate, update_text)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut gaussian_materials: ResMut<Assets<GaussianMaterial>>,
) {
    let diorama_scene: Handle<Scene> = asset_server.load(
        GltfAssetLabel::Scene(0).from_asset(
            "https://github.com/bevyengine/bevy_asset_files/raw/2a5950295a8b6d9d051d59c0df69e87abcda58c3/pica_pica/mini_diorama_01.glb",
        ),
    );
    commands
        .spawn((SceneRoot(diorama_scene), Transform::from_scale(Vec3::splat(10.0)), DioramaTag))
        .observe(add_mesh_processing_on_scene_load);

    commands
        .spawn((
            SceneRoot(asset_server.load(
                GltfAssetLabel::Scene(0).from_asset("https://github.com/bevyengine/bevy_asset_files/raw/2a5950295a8b6d9d051d59c0df69e87abcda58c3/pica_pica/robot_01.glb")
            )),
            Transform::from_scale(Vec3::splat(2.0))
                .with_translation(Vec3::new(-2.0, 0.05, -2.1))
                .with_rotation(Quat::from_rotation_y(PI / 2.0)),
            PatrolPath {
                path: vec![
                    (Vec3::new(-2.0, 0.05, -2.1), Quat::from_rotation_y(PI / 2.0)),
                    (Vec3::new(2.2, 0.05, -2.1), Quat::from_rotation_y(0.0)),
                    (
                        Vec3::new(2.2, 0.05, 2.1),
                        Quat::from_rotation_y(3.0 * PI / 2.0),
                    ),
                    (Vec3::new(-2.0, 0.05, 2.1), Quat::from_rotation_y(PI)),
                ],
                i: 0,
            },
        ))
        .observe(add_mesh_processing_on_scene_load);

    let icecream_pos = Vec3::new(0.0, 1.0, 0.0);

    let gaussian_textures = vec![
        asset_server.load("pbr/Poliigon_BrickWallReclaimed_8320/Poliigon_BrickWallReclaimed_8320_BaseColor.png"),
        asset_server.load("pbr/Poliigon_GrassPatchyGround_4585/Poliigon_GrassPatchyGround_4585_BaseColor.png"),
    ];

    let gaussian_material_handle = gaussian_materials.add(GaussianMaterial {
        base_color: LinearRgba::WHITE,
        base_color_texture: gaussian_textures.first().cloned(),
        texture_projection: GaussianTextureProjection::Xz,
        bounds: None,
    });

    commands.spawn((
        PlanarGaussian3dHandle(
            asset_server.load(
                "https://raw.githubusercontent.com/mosure/bevy_gaussian_splatting/main/assets/scenes/icecream.ply",
            ),
        ),
        CloudSettings {
            aabb: true,
            visualize_bounding_box: false,
            global_opacity: 6.0,
            rasterize_mode: RasterizeMode::Color,
            sort_mode: bevy_gaussian_splatting::sort::SortMode::Std,
            ..default()
        },
        Transform::from_translation(icecream_pos).with_scale(Vec3::splat(1.5)),
        GaussianMaterialHandle(gaussian_material_handle.clone()),
    ));

    commands.insert_resource(IcecreamPos(icecream_pos));
    commands.insert_resource(GaussianMaterialCycle {
        material_handle: gaussian_material_handle,
        textures: gaussian_textures,
        current: 0,
        timer: Timer::from_seconds(TEXTURE_SWAP_SECONDS, TimerMode::Repeating),
    });
    let marker_mesh = meshes.add(Cuboid::new(0.2, 0.2, 0.2));
    let marker_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.2, 0.2),
        emissive: LinearRgba::from(Color::srgb(1.0, 0.1, 0.1)) * 50.0,
        ..default()
    });
    commands.spawn((Mesh3d(marker_mesh), MeshMaterial3d(marker_mat), Transform::from_translation(icecream_pos)));

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_xyzw(
            -0.13334629,
            -0.86597735,
            -0.3586996,
            0.3219264,
        )),
    ));

    let eye = Vec3::new(0.0, 2.0, 5.0);
    commands.spawn((
        GaussianCamera { warmup: false },
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        CameraController {
            walk_speed: 3.0,
            run_speed: 10.0,
            ..Default::default()
        },
        Transform::from_translation(eye).looking_at(icecream_pos, Vec3::Y),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
    ));

    commands.spawn((
        Text::new("Loading..."),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn add_mesh_processing_on_scene_load(
    scene_ready: On<SceneInstanceReady>,
    children: Query<&Children>,
    mesh_query: Query<(
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
        Option<&GltfMaterialName>,
    )>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    for descendant in children.iter_descendants(scene_ready.entity) {
        if let Ok((Mesh3d(mesh_handle), MeshMaterial3d(material_handle), material_name)) =
            mesh_query.get(descendant)
        {
            let mesh = meshes.get_mut(mesh_handle).unwrap();
            if !mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0) {
                let vertex_count = mesh.count_vertices();
                mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; vertex_count]);
                mesh.insert_attribute(
                    Mesh::ATTRIBUTE_TANGENT,
                    vec![[0.0, 0.0, 0.0, 0.0]; vertex_count],
                );
            }
            if !mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT) {
                mesh.generate_tangents().unwrap();
            }
            if mesh.contains_attribute(Mesh::ATTRIBUTE_UV_1) {
                mesh.remove_attribute(Mesh::ATTRIBUTE_UV_1);
            }

            if material_name.map(|s| s.0.as_str()) == Some("material") {
                let material = materials.get_mut(material_handle).unwrap();
                material.emissive = LinearRgba::BLACK;
            }
            if material_name.map(|s| s.0.as_str()) == Some("Lights") {
                let material = materials.get_mut(material_handle).unwrap();
                material.emissive =
                    LinearRgba::from(Color::srgb(0.941, 0.714, 0.043)) * 1_000_000.0;
                material.alpha_mode = AlphaMode::Opaque;
                material.specular_transmission = 0.0;

                commands.insert_resource(RobotLightMaterial(material_handle.clone()));
            }
            if material_name.map(|s| s.0.as_str()) == Some("Glass_Dark_01") {
                let material = materials.get_mut(material_handle).unwrap();
                material.alpha_mode = AlphaMode::Opaque;
                material.specular_transmission = 0.0;
            }
        }
    }
}

fn pause_scene(mut time: ResMut<Time<Virtual>>, key_input: Res<ButtonInput<KeyCode>>) {
    if key_input.just_pressed(KeyCode::Space) {
        if time.is_paused() {
            time.unpause();
        } else {
            time.pause();
        }
    }
}

#[derive(Resource)]
struct RobotLightMaterial(Handle<StandardMaterial>);

fn toggle_lights(
    key_input: Res<ButtonInput<KeyCode>>,
    robot_light_material: Option<Res<RobotLightMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    directional_light: Query<Entity, With<DirectionalLight>>,
    diorama: Query<Entity, With<DioramaTag>>,
    icecream_pos: Option<Res<IcecreamPos>>,
    mut cam_q: Query<&mut Transform, With<Camera3d>>,
    mut commands: Commands,
) {
    if key_input.just_pressed(KeyCode::Digit1) {
        if let Ok(directional_light) = directional_light.single() {
            commands.entity(directional_light).despawn();
        } else {
            commands.spawn((
                DirectionalLight {
                    illuminance: light_consts::lux::FULL_DAYLIGHT,
                    shadows_enabled: false,
                    ..default()
                },
                Transform::from_rotation(Quat::from_xyzw(
                    -0.13334629,
                    -0.86597735,
                    -0.3586996,
                    0.3219264,
                )),
            ));
        }
    }

    if key_input.just_pressed(KeyCode::Digit3) {
        if let Ok(entity) = diorama.single() {
            commands.entity(entity).despawn();
        }
    }

    if key_input.just_pressed(KeyCode::Digit4) {
        if let (Some(pos), Ok(mut cam_tf)) = (icecream_pos, cam_q.single_mut()) {
            let target = pos.0;
            let eye = target + Vec3::new(0.0, 0.6, 2.6);
            *cam_tf = Transform::from_translation(eye).looking_at(target, Vec3::Y);
        }
    }

    if key_input.just_pressed(KeyCode::Digit2)
        && let Some(robot_light_material) = robot_light_material
    {
        let material = materials.get_mut(&robot_light_material.0).unwrap();
        if material.emissive == LinearRgba::BLACK {
            material.emissive = LinearRgba::from(Color::srgb(0.941, 0.714, 0.043)) * 1_000_000.0;
        } else {
            material.emissive = LinearRgba::BLACK;
        }
    }
}

#[derive(Component)]
struct PatrolPath {
    path: Vec<(Vec3, Quat)>,
    i: usize,
}

fn patrol_path(mut query: Query<(&mut PatrolPath, &mut Transform)>, time: Res<Time<Virtual>>) {
    for (mut path, mut transform) in query.iter_mut() {
        let (mut target_position, mut target_rotation) = path.path[path.i];
        let mut distance_to_target = transform.translation.distance(target_position);
        if distance_to_target < 0.01 {
            transform.translation = target_position;
            transform.rotation = target_rotation;

            path.i = (path.i + 1) % path.path.len();
            (target_position, target_rotation) = path.path[path.i];
            distance_to_target = transform.translation.distance(target_position);
        }

        let direction = (target_position - transform.translation).normalize();
        let movement = direction * time.delta_secs();

        if movement.length() > distance_to_target {
            transform.translation = target_position;
            transform.rotation = target_rotation;
        } else {
            transform.translation += movement;
        }
    }
}

fn update_text(
    mut text: Single<&mut Text>,
    robot_light_material: Option<Res<RobotLightMaterial>>,
    materials: Res<Assets<StandardMaterial>>,
    directional_light: Query<Entity, With<DirectionalLight>>,
    time: Res<Time<Virtual>>,
) {
    let mut content = String::new();

    if time.is_paused() {
        content.push_str("(Space): Resume");
    } else {
        content.push_str("(Space): Pause");
    }

    if directional_light.single().is_ok() {
        content.push_str("\n(1): Disable directional light");
    } else {
        content.push_str("\n(1): Enable directional light");
    }

    content.push_str("\n(3): Despawn diorama (unhide icecream)");
    content.push_str("\n(4): Teleport camera to icecream");

    match robot_light_material.and_then(|m| materials.get(&m.0)) {
        Some(robot_light_material) if robot_light_material.emissive != LinearRgba::BLACK => {
            content.push_str("\n(2): Disable robot emissive light");
        }
        _ => {
            content.push_str("\n(2): Enable robot emissive light");
        }
    }

    text.0 = content;
}

fn cycle_gaussian_material_textures(
    time: Res<Time<Virtual>>,
    mut gaussian_materials: ResMut<Assets<GaussianMaterial>>,
    mut cycle: ResMut<GaussianMaterialCycle>,
) {
    if cycle.textures.is_empty() {
        return;
    }

    if !cycle.timer.tick(time.delta()).just_finished() {
        return;
    }

    cycle.current = (cycle.current + 1) % cycle.textures.len();

    if let Some(material) = gaussian_materials.get_mut(&cycle.material_handle) {
        material.base_color_texture = Some(cycle.textures[cycle.current].clone());
    }
}
