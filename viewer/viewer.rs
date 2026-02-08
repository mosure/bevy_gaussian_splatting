// TODO: move to editor crate
use std::path::PathBuf;

use bevy::{
    app::AppExit,
    camera::primitives::Aabb,
    color::palettes::css::GOLD,
    core_pipeline::{prepass::MotionVectorPrepass, tonemapping::Tonemapping},
    diagnostic::{DiagnosticsStore, FrameCount, FrameTimeDiagnosticsPlugin},
    gizmos::config::GizmoConfigStore,
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
};

#[cfg(all(feature = "file_asset", not(target_arch = "wasm32")))]
use bevy::asset::{
    AssetApp,
    io::{AssetSourceBuilder, file::FileAssetReader},
};

#[cfg(feature = "web_asset")]
use bevy::asset::io::web::WebAssetPlugin;
use bevy_args::{BevyArgsPlugin, parse_args};
use bevy_inspector_egui::{
    DefaultInspectorConfigPlugin, bevy_egui::EguiPlugin, quick::WorldInspectorPlugin,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

#[cfg(feature = "web_asset")]
use base64::{Engine as _, engine::general_purpose::URL_SAFE};
use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, GaussianMode, GaussianPrimitiveMetadata, GaussianScene,
    GaussianSceneHandle, GaussianSplattingPlugin, PlanarGaussian3d, PlanarGaussian3dHandle,
    PlanarGaussian4d, PlanarGaussian4dHandle, SceneExportCamera, SceneExportCloud,
    gaussian::interface::TestCloud,
    io::scene::GaussianSceneLoaded,
    random_gaussians_3d, random_gaussians_3d_seeded, random_gaussians_4d,
    random_gaussians_4d_seeded,
    utils::{GaussianSplattingViewer, log, setup_hooks},
    write_khr_gaussian_scene_glb,
};

#[cfg(feature = "morph_interpolate")]
use bevy_gaussian_splatting::{Gaussian3d, morph::interpolate::GaussianInterpolate};

#[cfg(feature = "material_noise")]
use bevy_gaussian_splatting::material::noise::NoiseMaterial;

#[cfg(feature = "morph_particles")]
use bevy_gaussian_splatting::morph::particle::{
    ParticleBehaviors, ParticleBehaviorsHandle, random_particle_behaviors,
};

#[cfg(feature = "query_select")]
use bevy_gaussian_splatting::query::select::{InvertSelectionEvent, SaveSelectionEvent};

#[cfg(feature = "query_sparse")]
use bevy_gaussian_splatting::query::sparse::SparseSelect;

#[derive(Component, Debug, Default)]
struct ViewerMainCamera;

#[derive(Component, Debug, Default)]
struct SceneCameraApplied;

#[derive(Component, Debug, Default)]
struct SceneRenderModeApplied;

type ExportCloudQuery = (
    &'static PlanarGaussian3dHandle,
    &'static GlobalTransform,
    Option<&'static Name>,
    Option<&'static CloudSettings>,
    Option<&'static GaussianPrimitiveMetadata>,
);

type ExportCameraQuery = (&'static GlobalTransform, Option<&'static Name>);
type SceneCameraApplyQuery = (Entity, &'static mut Transform, &'static mut PanOrbitCamera);
type SceneRenderModeQuery = (Entity, &'static Children);
type SceneRenderModeFilter = (With<GaussianSceneLoaded>, Without<SceneRenderModeApplied>);

fn parse_input_file(input_file: &str) -> String {
    #[cfg(feature = "web_asset")]
    let input_uri = match URL_SAFE.decode(input_file.as_bytes()) {
        Ok(data) => match String::from_utf8(data) {
            Ok(decoded) => decoded,
            Err(_) => input_file.to_string(),
        },
        Err(err) => {
            if let Some(decoded) = decode_percent_encoded(input_file) {
                return decoded;
            }

            // Leave as-is for regular relative paths and already-decoded URLs.
            debug!("failed to decode base64 input: {:?}", err);
            input_file.to_string()
        }
    };

    #[cfg(not(feature = "web_asset"))]
    let input_uri = input_file.to_string();

    input_uri
}

#[cfg(feature = "web_asset")]
fn decode_percent_encoded(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    let mut changed = false;

    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return None;
            }

            let high = decode_hex(bytes[index + 1])?;
            let low = decode_hex(bytes[index + 2])?;
            decoded.push((high << 4) | low);
            index += 3;
            changed = true;
            continue;
        }

        decoded.push(bytes[index]);
        index += 1;
    }

    if !changed {
        return None;
    }

    String::from_utf8(decoded).ok()
}

#[cfg(feature = "web_asset")]
fn decode_hex(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn setup_gaussian_cloud(
    mut commands: Commands,
    args: Res<GaussianSplattingViewer>,
    asset_server: Res<AssetServer>,
    mut gaussian_3d_assets: ResMut<Assets<PlanarGaussian3d>>,
    mut gaussian_4d_assets: ResMut<Assets<PlanarGaussian4d>>,
) {
    debug!("spawning camera...");
    let cloud_transform = args.cloud_transform();
    commands
        .spawn(Camera3d::default())
        .insert(Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)))
        .insert(Tonemapping::None)
        .insert(MotionVectorPrepass)
        .insert(PanOrbitCamera {
            allow_upside_down: true,
            orbit_smoothness: 0.1,
            pan_smoothness: 0.1,
            zoom_smoothness: 0.1,
            ..default()
        })
        .insert(ViewerMainCamera)
        .insert(GaussianCamera::default());

    if let Some(input_scene) = &args.input_scene {
        let input_uri = parse_input_file(input_scene.as_str());
        log(&format!("loading {input_uri}"));
        let scene: Handle<GaussianScene> = asset_server.load(&input_uri);
        commands.spawn((
            GaussianSceneHandle(scene),
            Name::new("gaussian_scene"),
            cloud_transform,
        ));
        return;
    }

    match args.gaussian_mode {
        GaussianMode::Gaussian2d | GaussianMode::Gaussian3d => {
            let cloud: Handle<PlanarGaussian3d>;
            if args.gaussian_count > 0 {
                log(&format!("generating {} gaussians", args.gaussian_count));
                cloud = if let Some(seed) = args.gaussian_seed {
                    gaussian_3d_assets.add(random_gaussians_3d_seeded(args.gaussian_count, seed))
                } else {
                    gaussian_3d_assets.add(random_gaussians_3d(args.gaussian_count))
                };
            } else if let Some(input_cloud) = &args.input_cloud {
                let input_uri = parse_input_file(input_cloud.as_str());
                log(&format!("loading {input_uri}"));
                cloud = asset_server.load(&input_uri);
            } else {
                cloud = gaussian_3d_assets.add(PlanarGaussian3d::test_model());
            }

            #[cfg(feature = "morph_interpolate")]
            {
                if let Some(input_cloud_target) = &args.input_cloud_target {
                    let input_uri = parse_input_file(input_cloud_target.as_str());
                    log(&format!("loading {input_uri}"));
                    let binary_cloud: Handle<PlanarGaussian3d> = asset_server.load(&input_uri);

                    commands.spawn((
                        CloudSettings {
                            gaussian_mode: args.gaussian_mode,
                            playback_mode: args.playback_mode,
                            rasterize_mode: args.rasterization_mode,
                            ..default()
                        },
                        GaussianInterpolate::<Gaussian3d> {
                            lhs: PlanarGaussian3dHandle(cloud),
                            rhs: PlanarGaussian3dHandle(binary_cloud),
                        },
                        Name::new("gaussian_cloud_3d_binary"),
                        ShowAxes,
                        cloud_transform,
                    ));
                } else {
                    commands.spawn((
                        CloudSettings {
                            gaussian_mode: args.gaussian_mode,
                            playback_mode: args.playback_mode,
                            rasterize_mode: args.rasterization_mode,
                            ..default()
                        },
                        PlanarGaussian3dHandle(cloud.clone()),
                        Name::new("gaussian_cloud_3d"),
                        ShowAxes,
                        cloud_transform,
                    ));
                }
            }

            #[cfg(not(feature = "morph_interpolate"))]
            {
                commands.spawn((
                    CloudSettings {
                        gaussian_mode: args.gaussian_mode,
                        playback_mode: args.playback_mode,
                        rasterize_mode: args.rasterization_mode,
                        ..default()
                    },
                    PlanarGaussian3dHandle(cloud.clone()),
                    Name::new("gaussian_cloud_3d"),
                    ShowAxes,
                    cloud_transform,
                ));
            }
        }
        GaussianMode::Gaussian4d => {
            let cloud: Handle<PlanarGaussian4d>;
            if args.gaussian_count > 0 {
                log(&format!("generating {} gaussians", args.gaussian_count));
                cloud = if let Some(seed) = args.gaussian_seed {
                    gaussian_4d_assets.add(random_gaussians_4d_seeded(args.gaussian_count, seed))
                } else {
                    gaussian_4d_assets.add(random_gaussians_4d(args.gaussian_count))
                };
            } else if let Some(input_cloud) = &args.input_cloud {
                let input_uri = parse_input_file(input_cloud.as_str());
                log(&format!("loading {input_uri}"));
                cloud = asset_server.load(&input_uri);
            } else {
                cloud = gaussian_4d_assets.add(PlanarGaussian4d::test_model());
            }

            commands.spawn((
                PlanarGaussian4dHandle(cloud),
                CloudSettings {
                    gaussian_mode: args.gaussian_mode,
                    playback_mode: args.playback_mode,
                    rasterize_mode: args.rasterization_mode,
                    ..default()
                },
                Name::new("gaussian_cloud_4d"),
                ShowAxes,
                cloud_transform,
            ));
        }
    }
}

fn apply_scene_camera_spawn(
    mut commands: Commands,
    scene_handles: Query<(Entity, &GaussianSceneHandle), Without<SceneCameraApplied>>,
    asset_server: Res<AssetServer>,
    scenes: Res<Assets<GaussianScene>>,
    mut cameras: Query<SceneCameraApplyQuery, (With<GaussianCamera>, With<ViewerMainCamera>)>,
) {
    for (entity, scene_handle) in scene_handles.iter() {
        if let Some(load_state) = asset_server.get_load_state(&scene_handle.0)
            && !load_state.is_loaded()
        {
            continue;
        }

        let Some(scene) = scenes.get(&scene_handle.0) else {
            continue;
        };

        if let Some(scene_camera) = scene.cameras.first()
            && let Ok((camera_entity, mut camera_transform, mut pan_orbit_camera)) =
                cameras.single_mut()
        {
            let orbit_radius = pan_orbit_camera
                .target_radius
                .max(pan_orbit_camera.zoom_lower_limit);
            let scene_translation = scene_camera.transform.translation;
            let scene_forward = scene_camera.transform.forward().as_vec3();
            let world_up = pan_orbit_camera.axis[1];
            let mut corrected_rotation = scene_camera.transform.rotation;

            // Imported camera can legitimately be upside-down (roll ~= PI) which makes orbit input
            // feel inverted. Flip it upright while keeping the same look direction.
            if scene_camera.transform.up().dot(world_up) < 0.0 {
                corrected_rotation =
                    Quat::from_axis_angle(scene_forward, std::f32::consts::PI) * corrected_rotation;
            }

            let corrected_transform = Transform {
                translation: scene_translation,
                rotation: corrected_rotation,
                scale: Vec3::ONE,
            };
            *camera_transform = corrected_transform;

            let focus = scene_translation + camera_transform.forward() * orbit_radius;

            let (yaw, pitch, radius) = orbit_from_translation_and_focus(
                camera_transform.translation,
                focus,
                pan_orbit_camera.axis,
            );

            pan_orbit_camera.focus = focus;
            pan_orbit_camera.target_focus = focus;
            pan_orbit_camera.yaw = Some(yaw);
            pan_orbit_camera.pitch = Some(pitch);
            pan_orbit_camera.radius = Some(radius);
            pan_orbit_camera.target_yaw = yaw;
            pan_orbit_camera.target_pitch = pitch;
            pan_orbit_camera.target_radius = radius;
            pan_orbit_camera.allow_upside_down = false;
            pan_orbit_camera.initialized = true;
            pan_orbit_camera.force_update = true;
            let _ = camera_entity;
        }

        commands.entity(entity).insert(SceneCameraApplied);
    }
}

fn apply_scene_render_mode_override(
    mut commands: Commands,
    args: Res<GaussianSplattingViewer>,
    scenes: Query<SceneRenderModeQuery, SceneRenderModeFilter>,
    mut cloud_settings: Query<&mut CloudSettings>,
) {
    if args.input_scene.is_none() {
        return;
    }

    for (entity, children) in scenes.iter() {
        for child in children.iter() {
            let child: Entity = child;
            if let Ok(mut settings) = cloud_settings.get_mut(child) {
                settings.rasterize_mode = args.rasterization_mode;
            }
        }

        commands.entity(entity).insert(SceneRenderModeApplied);
    }
}

fn orbit_from_translation_and_focus(
    translation: Vec3,
    focus: Vec3,
    axis: [Vec3; 3],
) -> (f32, f32, f32) {
    let axis = Mat3::from_cols(axis[0], axis[1], axis[2]);
    let offset = translation - focus;

    // Radius of exactly zero creates unstable orbit behavior.
    let mut radius = offset.length();
    if radius <= f32::EPSILON {
        radius = 0.05;
    }

    let offset = axis * offset;
    let yaw = offset.x.atan2(offset.z);
    let pitch = (offset.y / radius).asin();
    (yaw, pitch, radius)
}

#[cfg(feature = "morph_particles")]
fn setup_particle_behavior(
    mut commands: Commands,
    gaussian_splatting_viewer: Res<GaussianSplattingViewer>,
    mut particle_behavior_assets: ResMut<Assets<ParticleBehaviors>>,
    gaussian_cloud: Query<(Entity, &PlanarGaussian3dHandle), Without<ParticleBehaviorsHandle>>,
) {
    if gaussian_cloud.is_empty() {
        return;
    }

    let mut particle_behaviors = None;
    if gaussian_splatting_viewer.particle_count > 0 {
        log(&format!(
            "generating {} particle behaviors",
            gaussian_splatting_viewer.particle_count
        ));
        particle_behaviors = particle_behavior_assets
            .add(random_particle_behaviors(
                gaussian_splatting_viewer.particle_count,
            ))
            .into();
    }

    if let Some(particle_behaviors) = particle_behaviors
        && let Ok((entity, _)) = gaussian_cloud.single()
    {
        commands
            .entity(entity)
            .insert(ParticleBehaviorsHandle(particle_behaviors));
    }
}

#[cfg(feature = "material_noise")]
fn setup_noise_material(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_clouds: Query<(Entity, &PlanarGaussian3dHandle), Without<NoiseMaterial>>,
) {
    if gaussian_clouds.is_empty() {
        return;
    }

    for (entity, cloud_handle) in gaussian_clouds.iter() {
        if let Some(load_state) = asset_server.get_load_state(cloud_handle.0.id())
            && load_state.is_loading()
        {
            continue;
        }

        commands.entity(entity).insert(NoiseMaterial::default());
    }
}

#[cfg(feature = "query_sparse")]
fn setup_sparse_select(
    mut commands: Commands,
    gaussian_cloud: Query<(Entity, &PlanarGaussian3dHandle), Without<SparseSelect>>,
) {
    if gaussian_cloud.is_empty() {
        return;
    }

    if let Ok((entity, _)) = gaussian_cloud.single() {
        commands.entity(entity).insert(SparseSelect {
            completed: true,
            ..default()
        });
    }
}

fn viewer_app() {
    let config = parse_args::<GaussianSplattingViewer>();
    log(&format!("{config:?}"));

    #[cfg(not(feature = "morph_interpolate"))]
    if config.input_cloud_target.is_some() {
        panic!("`--input-cloud-target` requires the `morph_interpolate` feature");
    }

    let mut app = App::new();
    app.register_type::<GizmoConfigStore>();

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        // fit_canvas_to_parent: true,
        canvas: Some("#bevy".to_string()),
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: true,
        title: config.name.clone(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    #[cfg(not(target_arch = "wasm32"))]
    let primary_window = Some(Window {
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: false,
        resolution: bevy::window::WindowResolution::new(config.width as u32, config.height as u32),
        title: config.name.clone(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    #[cfg(all(feature = "file_asset", not(target_arch = "wasm32")))]
    app.register_asset_source(
        "file",
        AssetSourceBuilder::new(|| Box::new(FileAssetReader::new("")))
            .with_processed_reader(|| Box::new(FileAssetReader::new(""))),
    );

    // setup for gaussian viewer app
    app.insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)));
    let default_plugins = DefaultPlugins
        .set(AssetPlugin {
            meta_check: bevy::asset::AssetMetaCheck::Never,
            unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
            ..default()
        })
        .set(ImagePlugin::default_nearest())
        .set(WindowPlugin {
            primary_window,
            ..default()
        });

    #[cfg(feature = "web_asset")]
    let default_plugins = default_plugins.set(WebAssetPlugin {
        silence_startup_warning: true,
    });

    app.add_plugins(default_plugins);
    app.add_plugins(BevyArgsPlugin::<GaussianSplattingViewer>::default());
    app.add_plugins(PanOrbitCameraPlugin);

    if config.editor {
        app.add_plugins(EguiPlugin::default());
        app.add_plugins(DefaultInspectorConfigPlugin);
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if config.press_esc_close {
        app.add_systems(Update, press_esc_close);
    }

    if config.press_s_screenshot {
        app.add_systems(Update, press_s_screenshot);
    }

    if config.show_axes {
        app.add_systems(Update, draw_axes);
    }

    if config.show_fps {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        app.add_systems(Startup, fps_display_setup);
        app.add_systems(Update, fps_update_system);
    }

    // setup for gaussian splatting
    app.add_plugins(GaussianSplattingPlugin);
    app.add_systems(Startup, setup_gaussian_cloud);
    app.add_systems(Update, apply_scene_camera_spawn);
    app.add_systems(Update, apply_scene_render_mode_override);
    app.add_systems(Update, press_g_save_gltf_scene);

    #[cfg(feature = "material_noise")]
    app.add_systems(Update, setup_noise_material);

    #[cfg(feature = "morph_particles")]
    app.add_systems(Update, setup_particle_behavior);

    #[cfg(feature = "query_select")]
    {
        app.add_systems(Update, press_i_invert_selection);
        app.add_systems(Update, press_o_save_selection);
    }

    #[cfg(feature = "query_sparse")]
    app.add_systems(Update, setup_sparse_select);

    app.run();
}

pub fn press_s_screenshot(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    current_frame: Res<FrameCount>,
) {
    if keys.just_pressed(KeyCode::KeyS) {
        let images_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("screenshots");
        std::fs::create_dir_all(&images_dir).unwrap();
        let output_path = images_dir.join(format!("output_{}.png", current_frame.0));

        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(output_path));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn press_g_save_gltf_scene(
    keys: Res<ButtonInput<KeyCode>>,
    current_frame: Res<FrameCount>,
    gaussian_cloud_assets: Res<Assets<PlanarGaussian3d>>,
    gaussian_clouds: Query<ExportCloudQuery>,
    cameras: Query<ExportCameraQuery, (With<GaussianCamera>, With<ViewerMainCamera>)>,
) {
    if !keys.just_pressed(KeyCode::KeyG) {
        return;
    }

    let mut export_clouds = Vec::new();
    for (index, (cloud_handle, global_transform, name, settings, metadata)) in
        gaussian_clouds.iter().enumerate()
    {
        let Some(cloud) = gaussian_cloud_assets.get(&cloud_handle.0) else {
            continue;
        };

        export_clouds.push(SceneExportCloud {
            cloud: cloud.clone(),
            name: name
                .map(|value| value.as_str().to_owned())
                .unwrap_or_else(|| format!("gaussian_cloud_{index}")),
            settings: settings.cloned().unwrap_or_default(),
            transform: Transform::from_matrix(global_transform.to_matrix()),
            metadata: metadata.cloned().unwrap_or_default(),
        });
    }

    if export_clouds.is_empty() {
        log("no gaussian clouds available to export");
        return;
    }

    let export_camera = cameras
        .iter()
        .next()
        .map(|(global_transform, name)| SceneExportCamera {
            name: name
                .map(|value| value.as_str().to_owned())
                .unwrap_or_else(|| "viewer_camera".to_owned()),
            transform: Transform::from_matrix(global_transform.to_matrix()),
            ..default()
        });

    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("exports");
    if let Err(err) = std::fs::create_dir_all(&output_dir) {
        log(&format!(
            "failed to create export directory '{}': {err}",
            output_dir.display()
        ));
        return;
    }

    let output_path = output_dir.join(format!("gaussian_scene_{}.glb", current_frame.0));
    match write_khr_gaussian_scene_glb(&output_path, &export_clouds, export_camera.as_ref()) {
        Ok(()) => log(&format!(
            "saved gaussian scene to {}",
            output_path.display()
        )),
        Err(err) => log(&format!(
            "failed to save gaussian scene '{}': {err}",
            output_path.display()
        )),
    }
}

#[cfg(target_arch = "wasm32")]
fn press_g_save_gltf_scene(keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::KeyG) {
        log("GLB scene export is not supported on wasm32");
    }
}

#[derive(Component, Debug, Default, Reflect)]
pub struct ShowAxes;

fn draw_axes(mut gizmos: Gizmos, query: Query<(&Transform, &Aabb), With<ShowAxes>>) {
    for (&transform, aabb) in &query {
        let length = aabb.half_extents.length();
        gizmos.axes(transform, length);
    }
}

pub fn press_esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

#[cfg(feature = "query_select")]
fn press_i_invert_selection(
    keys: Res<ButtonInput<KeyCode>>,
    mut select_inverse_events: MessageWriter<InvertSelectionEvent>,
) {
    if keys.just_pressed(KeyCode::KeyI) {
        log("inverting selection");
        select_inverse_events.write(InvertSelectionEvent);
    }
}

#[cfg(feature = "query_select")]
fn press_o_save_selection(
    keys: Res<ButtonInput<KeyCode>>,
    mut select_inverse_events: MessageWriter<SaveSelectionEvent>,
) {
    if keys.just_pressed(KeyCode::KeyO) {
        log("saving selection");
        select_inverse_events.write(SaveSelectionEvent);
    }
}

fn fps_display_setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn((
            Text("fps: ".to_string()),
            TextFont {
                font: asset_server.load("fonts/Caveat-Bold.ttf"),
                font_size: 60.0,
                ..Default::default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(5.0),
                left: Val::Px(15.0),
                ..default()
            },
            ZIndex(2),
        ))
        .with_child((
            FpsText,
            TextColor(Color::Srgba(GOLD)),
            TextFont {
                font: asset_server.load("fonts/Caveat-Bold.ttf"),
                font_size: 60.0,
                ..Default::default()
            },
            TextSpan::default(),
        ));
}

#[derive(Component)]
struct FpsText;

fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut TextSpan, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS)
            && let Some(value) = fps.smoothed()
        {
            **text = format!("{value:.2}");
        }
    }
}

#[cfg(all(test, feature = "web_asset"))]
mod tests {
    use super::parse_input_file;

    #[test]
    fn decodes_percent_encoded_input_url() {
        let encoded = "https%3A%2F%2Fmitchell.mosure.me%2Ftrellis.glb";
        let decoded = parse_input_file(encoded);
        assert_eq!(decoded, "https://mitchell.mosure.me/trellis.glb");
    }

    #[test]
    fn keeps_plain_relative_path() {
        let input = "trellis.glb";
        let parsed = parse_input_file(input);
        assert_eq!(parsed, "trellis.glb");
    }
}

pub fn main() {
    setup_hooks();
    viewer_app();
}
