// TODO: move to editor crate
use std::path::{Path, PathBuf};

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
#[cfg(feature = "lod")]
use bevy_inspector_egui::bevy_egui::input::EguiWantsInput;
#[cfg(feature = "lod")]
use bevy_inspector_egui::bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, quick::WorldInspectorPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

#[cfg(feature = "web_asset")]
use base64::{Engine as _, engine::general_purpose::URL_SAFE};
use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, GaussianMode, GaussianScene, GaussianSceneHandle,
    GaussianSplattingPlugin, PlanarGaussian3d, PlanarGaussian3dHandle, PlanarGaussian4d,
    PlanarGaussian4dHandle,
    gaussian::interface::TestCloud,
    io::scene::GaussianSceneLoaded,
    random_gaussians_3d, random_gaussians_3d_seeded, random_gaussians_4d,
    random_gaussians_4d_seeded,
    utils::{GaussianSplattingViewer, log, setup_hooks},
};

#[cfg(feature = "lod")]
use bevy_gaussian_splatting::{
    GaussianLodBridgeConfig, GaussianLodDebugAvailability, GaussianLodHandle, GaussianLodLifecycle,
    GaussianLodPackageSource, GaussianLodSettings, GaussianLodSourceKind, GaussianLodStatus,
    LodDebugPreset, LodQualityTarget, gaussian::lod_settings::LodSelectionMode,
};

#[cfg(all(test, feature = "lod"))]
use bevy_gaussian_splatting::utils::GaussianLodViewerArgs;

#[cfg(not(target_arch = "wasm32"))]
use bevy_gaussian_splatting::{
    GaussianPrimitiveMetadata, SceneExportCamera, SceneExportCloud, write_khr_gaussian_scene_glb,
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

// Bevy's perspective matrix uses infinite reverse-Z, but its extracted shader
// frustum still applies `PerspectiveProjection::far`. The default 1,000-unit
// plane is close enough to cut off an otherwise visible Gaussian scene during
// an ordinary orbit-camera zoom. Keep enough headroom that the scene becomes
// subpixel before the viewer reaches its zoom limit.
const VIEWER_CAMERA_VISIBILITY_FAR: f32 = 1_000_000.0;
const VIEWER_CAMERA_MAX_ORBIT_RADIUS: f32 = 100_000.0;

fn viewer_perspective_projection() -> PerspectiveProjection {
    PerspectiveProjection {
        far: VIEWER_CAMERA_VISIBILITY_FAR,
        ..default()
    }
}

fn viewer_pan_orbit_camera() -> PanOrbitCamera {
    PanOrbitCamera {
        allow_upside_down: true,
        orbit_smoothness: 0.1,
        pan_smoothness: 0.1,
        zoom_smoothness: 0.1,
        zoom_upper_limit: Some(VIEWER_CAMERA_MAX_ORBIT_RADIUS),
        ..default()
    }
}

#[cfg(feature = "lod")]
#[derive(Clone, Debug, Resource)]
struct ViewerLodPolicy(GaussianLodSettings);

#[cfg(feature = "lod")]
type ViewerLodDiagnosticsQuery = (
    Entity,
    Option<&'static Name>,
    &'static mut GaussianLodSettings,
    &'static mut CloudSettings,
    Option<&'static GaussianLodPackageSource>,
    Option<&'static GaussianLodStatus>,
);

#[cfg(not(target_arch = "wasm32"))]
type ExportCloudQuery = (
    &'static PlanarGaussian3dHandle,
    &'static GlobalTransform,
    Option<&'static Name>,
    Option<&'static CloudSettings>,
    Option<&'static GaussianPrimitiveMetadata>,
);

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(feature = "lod")]
fn lod_package_source(manifest_uri: &str) -> GaussianLodPackageSource {
    if manifest_uri.starts_with("https://") || manifest_uri.starts_with("http://") {
        let base_url = manifest_uri
            .rsplit_once('/')
            .map_or(manifest_uri, |(base, _)| base);
        GaussianLodPackageSource::url(format!("{base_url}/"))
    } else {
        let native_path = manifest_uri.strip_prefix("file://").unwrap_or(manifest_uri);
        let root = Path::new(native_path)
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_string_lossy()
            .into_owned();
        GaussianLodPackageSource::native_directory(root)
    }
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
    #[cfg(feature = "lod")] lod_policy: Res<ViewerLodPolicy>,
    asset_server: Res<AssetServer>,
    mut gaussian_3d_assets: ResMut<Assets<PlanarGaussian3d>>,
    mut gaussian_4d_assets: ResMut<Assets<PlanarGaussian4d>>,
) {
    debug!("spawning camera...");
    let cloud_transform = args.cloud_transform();
    #[cfg(feature = "lod")]
    let lod_debug = args.lod_debug_settings();
    commands
        .spawn(Camera3d::default())
        .insert(Projection::Perspective(viewer_perspective_projection()))
        .insert(Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)))
        .insert(Tonemapping::None)
        .insert(MotionVectorPrepass)
        .insert(viewer_pan_orbit_camera())
        .insert(ViewerMainCamera)
        .insert(GaussianCamera::default());

    #[cfg(feature = "lod")]
    if let Some(input_lod) = &args.input_lod {
        let input_uri = parse_input_file(input_lod);
        log(&format!("loading LoD package {input_uri}"));
        let manifest = asset_server.load(&input_uri);
        commands.spawn((
            GaussianLodHandle(manifest),
            lod_package_source(&input_uri),
            CloudSettings {
                gaussian_mode: GaussianMode::Gaussian3d,
                playback_mode: args.playback_mode,
                rasterize_mode: args.rasterization_mode,
                radix_sort_depth_bits: args.radix_sort_depth_bits,
                lod_debug,
                ..default()
            },
            lod_policy.0.clone(),
            Name::new("gaussian_lod_package"),
            ShowAxes,
            cloud_transform,
        ));
        return;
    }

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
                            radix_sort_depth_bits: args.radix_sort_depth_bits,
                            #[cfg(feature = "lod")]
                            lod_debug,
                            ..default()
                        },
                        GaussianInterpolate::<Gaussian3d> {
                            lhs: PlanarGaussian3dHandle(cloud),
                            rhs: PlanarGaussian3dHandle(binary_cloud),
                        },
                        Name::new("gaussian_cloud_3d_binary"),
                        ShowAxes,
                        cloud_transform,
                        #[cfg(feature = "lod")]
                        lod_policy.0.clone(),
                    ));
                } else {
                    commands.spawn((
                        CloudSettings {
                            gaussian_mode: args.gaussian_mode,
                            playback_mode: args.playback_mode,
                            rasterize_mode: args.rasterization_mode,
                            radix_sort_depth_bits: args.radix_sort_depth_bits,
                            #[cfg(feature = "lod")]
                            lod_debug,
                            ..default()
                        },
                        PlanarGaussian3dHandle(cloud.clone()),
                        Name::new("gaussian_cloud_3d"),
                        ShowAxes,
                        cloud_transform,
                        #[cfg(feature = "lod")]
                        lod_policy.0.clone(),
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
                        radix_sort_depth_bits: args.radix_sort_depth_bits,
                        #[cfg(feature = "lod")]
                        lod_debug: lod_debug.clone(),
                        ..default()
                    },
                    PlanarGaussian3dHandle(cloud.clone()),
                    Name::new("gaussian_cloud_3d"),
                    ShowAxes,
                    cloud_transform,
                    #[cfg(feature = "lod")]
                    lod_policy.0.clone(),
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
                    radix_sort_depth_bits: args.radix_sort_depth_bits,
                    #[cfg(feature = "lod")]
                    lod_debug,
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
    #[cfg(feature = "lod")] lod_policy: Res<ViewerLodPolicy>,
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
                settings.radix_sort_depth_bits = args.radix_sort_depth_bits;
                #[cfg(feature = "lod")]
                {
                    settings.lod_debug = args.lod_debug_settings();
                }
                #[cfg(feature = "lod")]
                commands.entity(child).insert(lod_policy.0.clone());
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

#[cfg(feature = "lod")]
fn toggle_lod_freeze(
    keys: Res<ButtonInput<KeyCode>>,
    egui_wants_input: Option<Res<EguiWantsInput>>,
    mut clouds: Query<&mut GaussianLodSettings>,
) {
    if !keys.just_pressed(KeyCode::KeyF)
        || egui_wants_input.is_some_and(|input| input.wants_keyboard_input())
    {
        return;
    }

    for mut settings in &mut clouds {
        settings.selection_mode = match settings.selection_mode {
            LodSelectionMode::Dynamic => LodSelectionMode::Frozen,
            LodSelectionMode::Frozen => LodSelectionMode::Dynamic,
        };
    }
}

#[cfg(feature = "lod")]
fn lod_diagnostics_panel(mut contexts: EguiContexts, mut clouds: Query<ViewerLodDiagnosticsQuery>) {
    let Ok(context) = contexts.ctx_mut() else {
        return;
    };

    egui::Window::new("Gaussian LoD")
        .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
        .default_width(310.0)
        .resizable(false)
        .show(context, |ui| {
            let mut found_cloud = false;
            for (entity, name, mut settings, mut cloud, package, status) in &mut clouds {
                found_cloud = true;
                ui.push_id(entity, |ui| {
                    show_lod_cloud_panel(
                        ui,
                        entity,
                        name,
                        &mut settings,
                        &mut cloud,
                        package.is_some(),
                        status,
                    );
                });
                ui.separator();
            }

            if !found_cloud {
                ui.label("Waiting for a Gaussian cloud…");
            } else {
                ui.small("F: freeze / resume camera-driven LoD selection");
                ui.small("Advanced: select this named cloud in World Inspector.");
            }
        });
}

#[cfg(feature = "lod")]
fn show_lod_cloud_panel(
    ui: &mut egui::Ui,
    entity: Entity,
    name: Option<&Name>,
    settings: &mut GaussianLodSettings,
    cloud: &mut CloudSettings,
    has_package_source: bool,
    status: Option<&GaussianLodStatus>,
) {
    let cloud_name = name
        .map(|name| name.as_str().to_owned())
        .unwrap_or_else(|| format!("cloud {}", entity.index()));
    ui.strong(&cloud_name);

    let target = settings.quality_target();
    let original_endpoint = matches!(target, LodQualityTarget::Original);
    let flat_original_path = original_endpoint
        && !has_package_source
        && status.is_none_or(|status| status.source == GaussianLodSourceKind::Original);
    let selection_mode = status.map_or(settings.selection_mode, |status| status.selection_mode);

    let mut detail_quality = viewer_detail_quality(settings);
    let quality_response = ui.add(
        egui::Slider::new(&mut detail_quality, 0.0..=1.0)
            .text("Detail quality")
            .fixed_decimals(3),
    );
    if quality_response.changed() {
        apply_viewer_detail_quality(settings, detail_quality);
    }
    quality_response.on_hover_text(
        "Higher values request more hierarchy detail and lower projected error. Above 0.90, large error and builder-detected unsafe merged representatives progressively force refinement.",
    );
    ui.horizontal(|ui| {
        let mut frozen = settings.selection_mode == LodSelectionMode::Frozen;
        if ui
            .checkbox(&mut frozen, "Freeze camera selection")
            .changed()
        {
            settings.selection_mode = if frozen {
                LodSelectionMode::Frozen
            } else {
                LodSelectionMode::Dynamic
            };
        }
    });

    ui.horizontal(|ui| {
        ui.label("Debug");
        let active_preset = cloud.lod_debug.preset;
        egui::ComboBox::from_id_salt("lod_debug_preset")
            .selected_text(lod_debug_preset_label(active_preset))
            .show_ui(ui, |ui| {
                for preset in [
                    LodDebugPreset::Off,
                    LodDebugPreset::Level,
                    LodDebugPreset::Page,
                    LodDebugPreset::Residency,
                    LodDebugPreset::Boundaries,
                    LodDebugPreset::SelectionPressure,
                ] {
                    if ui
                        .selectable_label(active_preset == preset, lod_debug_preset_label(preset))
                        .clicked()
                    {
                        cloud.lod_debug.apply_preset(preset);
                    }
                }
            });
    });
    let debug_requested = cloud.lod_debug.requires_metadata();

    egui::Grid::new("lod_diagnostics")
        .num_columns(2)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            ui.label("Detail target").on_hover_text(
                "Requested structural detail and the mandatory screen-space error cap.",
            );
            ui.strong(format_lod_target(target));
            ui.end_row();

            ui.label("Target outcome");
            ui.monospace(format_lod_target_outcome(
                status.and_then(|status| status.target_satisfied),
                status.map(|status| status.degradation),
                flat_original_path,
            ));
            ui.end_row();

            ui.label("Selection mode");
            ui.monospace(match selection_mode {
                LodSelectionMode::Dynamic => "dynamic",
                LodSelectionMode::Frozen => "frozen",
            });
            ui.end_row();

            ui.label("Frozen views");
            ui.monospace(format_lod_frozen_views(
                status.map(|status| (status.frozen_views, status.active_views)),
                selection_mode,
                flat_original_path,
            ));
            ui.end_row();

            ui.label("Lifecycle");
            ui.monospace(
                status
                    .map(|status| format_lod_lifecycle(status.lifecycle))
                    .unwrap_or(if flat_original_path {
                        "original"
                    } else {
                        "initializing"
                    }),
            );
            ui.end_row();

            ui.label("Selected splats");
            ui.monospace(format_lod_runtime_count(
                status.map(|status| status.selected_gaussians),
                flat_original_path,
            ));
            ui.end_row();

            ui.label("GPU candidates");
            ui.monospace(format_lod_runtime_count(
                status.map(|status| u64::from(status.submitted_candidates)),
                flat_original_path,
            ));
            ui.end_row();

            ui.label("Resident pages");
            ui.monospace(format_lod_runtime_count(
                status.map(|status| u64::from(status.resident_pages)),
                flat_original_path,
            ));
            ui.end_row();

            ui.label("Quality pressure");
            ui.monospace(format_lod_quality_pressure(
                status.and_then(|status| status.achieved_max_target_ratio),
                flat_original_path,
            ));
            ui.end_row();

            ui.label("Achieved error");
            ui.monospace(format_lod_achieved_error(
                status.and_then(|status| status.achieved_max_error_px),
                flat_original_path,
            ));
            ui.end_row();

            if debug_requested {
                ui.label("Debug status");
                ui.monospace(format_lod_debug_availability(
                    status.map(|status| status.debug_availability),
                    flat_original_path,
                ));
                ui.end_row();
            }
        });

    if let Some(failure) = status.and_then(|status| status.failure.as_ref()) {
        ui.colored_label(egui::Color32::LIGHT_RED, format!("LoD fallback: {failure}"));
    }
    ui.small(format!("World Inspector cloud: {cloud_name}"));
}

#[cfg(feature = "lod")]
const fn lod_debug_preset_label(preset: LodDebugPreset) -> &'static str {
    match preset {
        LodDebugPreset::Off => "off",
        LodDebugPreset::Level => "hierarchy level",
        LodDebugPreset::Page => "page",
        LodDebugPreset::Residency => "residency / fallback",
        LodDebugPreset::Boundaries => "chunk boundaries",
        LodDebugPreset::SelectionPressure => "selection pressure",
    }
}

#[cfg(feature = "lod")]
fn format_lod_target(target: LodQualityTarget) -> String {
    match target {
        LodQualityTarget::Coarsest => "coarsest".to_owned(),
        LodQualityTarget::Balanced {
            detail_fraction, ..
        } => target.effective_max_screen_space_error_px().map_or_else(
            || format!("{:.0}% detail", detail_fraction * 100.0),
            |cap| format!("{:.0}% detail · ≤{cap:.2} px", detail_fraction * 100.0),
        ),
        LodQualityTarget::Original => "exact original".to_owned(),
    }
}

#[cfg(feature = "lod")]
fn format_lod_target_outcome(
    satisfied: Option<bool>,
    degradation: Option<bevy_gaussian_splatting::LodDegradation>,
    original_endpoint: bool,
) -> String {
    if original_endpoint {
        return "exact by contract".to_owned();
    }
    match satisfied {
        Some(true) => "met".to_owned(),
        Some(false) => match degradation.unwrap_or_default() {
            bevy_gaussian_splatting::LodDegradation::None => "over target (hysteresis)".to_owned(),
            bevy_gaussian_splatting::LodDegradation::ActiveBudget => {
                "degraded: active budget".to_owned()
            }
            bevy_gaussian_splatting::LodDegradation::Residency => "degraded: residency".to_owned(),
            bevy_gaussian_splatting::LodDegradation::TraversalBudget => {
                "degraded: traversal budget".to_owned()
            }
            bevy_gaussian_splatting::LodDegradation::Multiple => {
                "degraded: multiple constraints".to_owned()
            }
        },
        None => "waiting".to_owned(),
    }
}

#[cfg(feature = "lod")]
const fn format_lod_lifecycle(lifecycle: GaussianLodLifecycle) -> &'static str {
    match lifecycle {
        GaussianLodLifecycle::Original => "original",
        GaussianLodLifecycle::Building => "building",
        GaussianLodLifecycle::Streaming => "streaming",
        GaussianLodLifecycle::WaitingForRender => "waiting for render",
        GaussianLodLifecycle::Active => "active",
        GaussianLodLifecycle::Degraded => "degraded",
        GaussianLodLifecycle::Fallback => "fallback",
        GaussianLodLifecycle::Failed => "failed",
    }
}

#[cfg(feature = "lod")]
fn format_lod_quality_pressure(ratio: Option<f32>, original_endpoint: bool) -> String {
    if original_endpoint {
        return "exact by contract".to_owned();
    }
    ratio.map_or_else(
        || "waiting".to_owned(),
        |ratio| {
            let outcome = if ratio <= 1.0 { "met" } else { "over" };
            format!("{ratio:.2}× ({outcome})")
        },
    )
}

#[cfg(feature = "lod")]
fn format_lod_achieved_error(error_px: Option<f32>, original_endpoint: bool) -> String {
    if original_endpoint {
        return "exact by contract".to_owned();
    }
    error_px.map_or_else(|| "waiting".to_owned(), |error| format!("{error:.2} px"))
}

#[cfg(feature = "lod")]
fn viewer_detail_quality(settings: &GaussianLodSettings) -> f32 {
    settings.quality_clamped()
}

#[cfg(feature = "lod")]
fn apply_viewer_detail_quality(settings: &mut GaussianLodSettings, quality: f32) {
    settings.quality = quality.clamp(0.0, 1.0);
}

#[cfg(feature = "lod")]
fn viewer_lod_bridge_config() -> GaussianLodBridgeConfig {
    // The crate default is deliberately conservative for library users. The
    // viewer already owns an explicit resident-memory budget, so let its
    // ephemeral bridge admit real review assets up to that same bound. The 2x
    // stored/atlas headroom covers progressive hierarchy overhead; byte,
    // page, and commit budgets still cap physical storage.
    let budgets = GaussianLodSettings::default().budgets;
    let source_gaussians = budgets
        .max_resident_gaussians
        .try_into()
        .unwrap_or(u32::MAX);
    GaussianLodBridgeConfig {
        max_ephemeral_source_gaussians: source_gaussians,
        max_ephemeral_stored_gaussians: u64::from(source_gaussians).saturating_mul(2),
        max_atlas_gaussians: source_gaussians.saturating_mul(2),
        max_atlas_bytes: budgets.max_resident_bytes,
        ..Default::default()
    }
}

#[cfg(feature = "lod")]
fn format_lod_count(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, byte) in digits.bytes().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(char::from(byte));
    }
    grouped
}

#[cfg(feature = "lod")]
fn format_lod_runtime_count(value: Option<u64>, original_endpoint: bool) -> String {
    if original_endpoint {
        value
            .filter(|value| *value > 0)
            .map_or_else(|| "original path".to_owned(), format_lod_count)
    } else {
        value.map_or_else(|| "—".to_owned(), format_lod_count)
    }
}

#[cfg(feature = "lod")]
fn format_lod_frozen_views(
    views: Option<(u32, u32)>,
    selection_mode: LodSelectionMode,
    original_endpoint: bool,
) -> String {
    if original_endpoint {
        return "original path".to_owned();
    }
    if selection_mode == LodSelectionMode::Dynamic {
        return "not frozen".to_owned();
    }
    views.map_or_else(
        || "waiting for view".to_owned(),
        |(frozen, active)| format!("{frozen} / {active}"),
    )
}

#[cfg(feature = "lod")]
fn format_lod_debug_availability(
    availability: Option<GaussianLodDebugAvailability>,
    flat_original_path: bool,
) -> &'static str {
    if flat_original_path {
        return "unavailable at exact original";
    }
    match availability {
        None => "waiting for status",
        Some(GaussianLodDebugAvailability::Disabled) => "off",
        Some(GaussianLodDebugAvailability::UnavailableOriginalEndpoint) => {
            "unavailable at exact original"
        }
        Some(GaussianLodDebugAvailability::WaitingForMetadata) => "waiting for metadata",
        Some(GaussianLodDebugAvailability::MetadataReady) => "metadata ready",
    }
}

fn viewer_app() {
    let config = parse_args::<GaussianSplattingViewer>();
    log(&format!("{config:?}"));

    #[cfg(feature = "lod")]
    let lod_policy = ViewerLodPolicy(
        config
            .lod_settings()
            .unwrap_or_else(|error| panic!("invalid viewer LoD configuration: {error}")),
    );
    #[cfg(feature = "lod")]
    let lod_bridge_config = viewer_lod_bridge_config();
    #[cfg(feature = "lod")]
    lod_bridge_config
        .validate()
        .unwrap_or_else(|error| panic!("invalid viewer LoD bridge configuration: {error}"));

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

        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    #[cfg(not(target_arch = "wasm32"))]
    let primary_window = Some(Window {
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: false,
        resolution: bevy::window::WindowResolution::new(config.width as u32, config.height as u32),
        title: config.name.clone(),

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
    #[cfg(feature = "lod")]
    app.insert_resource(lod_policy)
        .insert_resource(lod_bridge_config);
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
    add_editor_plugins(&mut app, config.editor);
    app.add_plugins(PanOrbitCameraPlugin);

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

    #[cfg(feature = "lod")]
    app.add_systems(Update, toggle_lod_freeze);

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

fn add_editor_plugins(app: &mut App, enabled: bool) {
    add_plugins_when(app, enabled, |app| {
        app.add_plugins(EguiPlugin::default());
        app.add_plugins(WorldInspectorPlugin::new());
        #[cfg(feature = "lod")]
        app.add_systems(EguiPrimaryContextPass, lod_diagnostics_panel);
    });
}

fn add_plugins_when(app: &mut App, enabled: bool, install: impl FnOnce(&mut App)) {
    if enabled {
        install(app);
    }
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
                font: FontSource::Handle(asset_server.load("fonts/Caveat-Bold.ttf")),
                font_size: FontSize::Px(60.0),
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
                font: FontSource::Handle(asset_server.load("fonts/Caveat-Bold.ttf")),
                font_size: FontSize::Px(60.0),
                ..Default::default()
            },
            TextSpan::default(),
        ));
}

#[derive(Component)]
struct FpsText;

#[derive(Default)]
struct FpsDisplayState {
    smoothed_fps: Option<f64>,
    update_elapsed_secs: f32,
}

fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time>,
    mut state: Local<FpsDisplayState>,
    mut query: Query<&mut TextSpan, With<FpsText>>,
) {
    let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) else {
        return;
    };
    let Some(value) = fps.smoothed() else {
        return;
    };

    const SMOOTHING_ALPHA: f64 = 0.08;
    const DISPLAY_UPDATE_INTERVAL_SECS: f32 = 0.5;

    let smoothed_fps = state.smoothed_fps.map_or(value, |current| {
        current + (value - current) * SMOOTHING_ALPHA
    });
    state.smoothed_fps = Some(smoothed_fps);

    state.update_elapsed_secs += time.delta_secs();
    if state.update_elapsed_secs < DISPLAY_UPDATE_INTERVAL_SECS {
        return;
    }
    state.update_elapsed_secs = 0.0;

    let display_fps = smoothed_fps.round() as u32;
    for mut text in &mut query {
        **text = display_fps.to_string();
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

#[cfg(test)]
mod editor_tests {
    use super::*;

    struct EditorPluginProbe;

    impl Plugin for EditorPluginProbe {
        fn build(&self, _app: &mut App) {}
    }

    #[test]
    fn editor_plugin_gate_follows_runtime_setting() {
        let mut enabled = App::new();
        add_plugins_when(&mut enabled, true, |app| {
            app.add_plugins(EditorPluginProbe);
        });
        assert!(enabled.is_plugin_added::<EditorPluginProbe>());

        let mut disabled = App::new();
        add_plugins_when(&mut disabled, false, |app| {
            app.add_plugins(EditorPluginProbe);
        });
        assert!(!disabled.is_plugin_added::<EditorPluginProbe>());
    }
}

#[cfg(all(test, feature = "lod"))]
mod lod_tests {
    use super::*;

    #[test]
    fn package_manifest_paths_resolve_pages_beside_the_manifest() {
        assert_eq!(
            lod_package_source("https://cdn.example/scene/model.gsplatlod"),
            GaussianLodPackageSource::url("https://cdn.example/scene/")
        );
        assert_eq!(
            lod_package_source("assets/scene/model.gsplatlod"),
            GaussianLodPackageSource::native_directory("assets/scene")
        );
        assert_eq!(
            lod_package_source("model.gsplatlod"),
            GaussianLodPackageSource::native_directory(".")
        );
    }

    #[test]
    fn viewer_bridge_budget_admits_documented_large_review_assets() {
        const TRELLIS_GAUSSIANS: u32 = 478_368;
        const LOCAL_BONSAI_GAUSSIANS: u32 = 1_071_766;

        let bridge = viewer_lod_bridge_config();
        bridge.validate().unwrap();
        assert!(bridge.max_ephemeral_source_gaussians >= TRELLIS_GAUSSIANS);
        assert!(bridge.max_ephemeral_source_gaussians >= LOCAL_BONSAI_GAUSSIANS);
        assert_eq!(
            u64::from(bridge.max_ephemeral_source_gaussians),
            GaussianLodSettings::default()
                .budgets
                .max_resident_gaussians
        );
        assert_eq!(
            bridge.max_ephemeral_stored_gaussians,
            u64::from(bridge.max_ephemeral_source_gaussians) * 2
        );
    }

    #[test]
    fn diagnostics_format_resolved_targets_counts_and_pressure() {
        assert_eq!(
            format_lod_target(LodQualityTarget::Balanced {
                detail_fraction: 0.5,
                max_error_px: 2.0,
            }),
            "50% detail · ≤15.52 px"
        );
        assert_eq!(
            format_lod_target(LodQualityTarget::Original),
            "exact original"
        );
        assert_eq!(format_lod_target_outcome(Some(true), None, false), "met");
        assert_eq!(
            format_lod_target_outcome(
                Some(false),
                Some(bevy_gaussian_splatting::LodDegradation::None),
                false,
            ),
            "over target (hysteresis)"
        );
        assert_eq!(
            format_lod_target_outcome(
                Some(false),
                Some(bevy_gaussian_splatting::LodDegradation::Residency),
                false,
            ),
            "degraded: residency"
        );
        assert_eq!(format_lod_count(1_234_567), "1,234,567");
        assert_eq!(
            format_lod_quality_pressure(Some(0.875), false),
            "0.88× (met)"
        );
        assert_eq!(
            format_lod_quality_pressure(Some(1.25), false),
            "1.25× (over)"
        );
        assert_eq!(format_lod_quality_pressure(None, false), "waiting");
        assert_eq!(
            format_lod_quality_pressure(Some(0.0), true),
            "exact by contract"
        );
        assert_eq!(format_lod_runtime_count(Some(123), true), "123");
        assert_eq!(format_lod_runtime_count(Some(0), true), "original path");
        assert_eq!(format_lod_achieved_error(Some(0.527), false), "0.53 px");
        assert_eq!(format_lod_achieved_error(None, false), "waiting");
        assert_eq!(
            format_lod_achieved_error(Some(0.0), true),
            "exact by contract"
        );
        assert_eq!(
            format_lod_frozen_views(None, LodSelectionMode::Dynamic, true),
            "original path"
        );
        assert_eq!(
            format_lod_frozen_views(Some((0, 1)), LodSelectionMode::Dynamic, false),
            "not frozen"
        );
    }

    #[test]
    fn detail_quality_is_the_single_selection_control() {
        let mut settings = GaussianLodSettings {
            quality: 0.25,
            ..Default::default()
        };
        assert_eq!(viewer_detail_quality(&settings), 0.25);

        apply_viewer_detail_quality(&mut settings, 0.5);
        assert_eq!(settings.quality, 0.5);
        assert_eq!(viewer_detail_quality(&settings), 0.5);
    }

    #[test]
    fn diagnostics_explain_debug_availability_and_presets() {
        assert_eq!(
            format_lod_debug_availability(
                Some(GaussianLodDebugAvailability::UnavailableOriginalEndpoint),
                false
            ),
            "unavailable at exact original"
        );
        assert_eq!(
            format_lod_debug_availability(Some(GaussianLodDebugAvailability::MetadataReady), false,),
            "metadata ready"
        );
        assert_eq!(format_lod_runtime_count(Some(478_368), false), "478,368");
        assert_eq!(
            format_lod_debug_availability(None, true),
            "unavailable at exact original"
        );

        let page = bevy_gaussian_splatting::LodDebugSettings::from_preset(
            bevy_gaussian_splatting::LodDebugPreset::Page,
        );
        assert_eq!(page.preset, LodDebugPreset::Page);
    }

    #[test]
    fn freeze_hotkey_toggles_cloud_selection_without_window_automation() {
        let mut app = App::new();
        app.init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Update, toggle_lod_freeze);
        let entity = app.world_mut().spawn(GaussianLodSettings::default()).id();

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyF);
        app.update();
        assert_eq!(
            app.world()
                .get::<GaussianLodSettings>(entity)
                .unwrap()
                .selection_mode,
            LodSelectionMode::Frozen
        );

        {
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.release(KeyCode::KeyF);
            keys.clear();
        }
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyF);
        app.update();
        assert_eq!(
            app.world()
                .get::<GaussianLodSettings>(entity)
                .unwrap()
                .selection_mode,
            LodSelectionMode::Dynamic
        );
    }

    #[test]
    fn viewer_zoom_range_stays_inside_the_gaussian_shader_frustum() {
        const {
            assert!(VIEWER_CAMERA_MAX_ORBIT_RADIUS > 1_000.0);
            assert!(VIEWER_CAMERA_MAX_ORBIT_RADIUS < VIEWER_CAMERA_VISIBILITY_FAR);
        }

        let camera_transform = GlobalTransform::from(
            Transform::from_xyz(0.0, 0.0, VIEWER_CAMERA_MAX_ORBIT_RADIUS)
                .looking_at(Vec3::ZERO, Vec3::Y),
        );
        let cloud_bounds = Aabb::from_min_max(Vec3::splat(-32.0), Vec3::splat(32.0));
        let default_frustum = bevy::camera::CameraProjection::compute_frustum(
            &PerspectiveProjection::default(),
            &camera_transform,
        );
        assert!(
            !default_frustum.intersects_obb(
                &cloud_bounds,
                &bevy::math::Affine3A::IDENTITY,
                true,
                true,
            ),
            "the regression probe must remain beyond Bevy's default 1,000-unit far plane"
        );

        let viewer_frustum = bevy::camera::CameraProjection::compute_frustum(
            &viewer_perspective_projection(),
            &camera_transform,
        );
        assert!(viewer_frustum.intersects_obb(
            &cloud_bounds,
            &bevy::math::Affine3A::IDENTITY,
            true,
            true,
        ));
        assert_eq!(
            viewer_pan_orbit_camera().zoom_upper_limit,
            Some(VIEWER_CAMERA_MAX_ORBIT_RADIUS)
        );
    }

    #[test]
    fn scene_children_receive_the_validated_viewer_lod_policy() {
        let args = GaussianSplattingViewer {
            input_scene: Some("scene.glb".to_owned()),
            lod: GaussianLodViewerArgs {
                lod_quality: 0.4,
                lod_debug: Some(LodDebugPreset::Page),
                ..Default::default()
            },
            ..Default::default()
        };
        let expected = args.lod_settings().expect("test viewer policy is valid");
        let expected_debug = args.lod_debug_settings();

        let mut app = App::new();
        app.insert_resource(args)
            .insert_resource(ViewerLodPolicy(expected.clone()))
            .add_systems(Update, apply_scene_render_mode_override);

        let child = app.world_mut().spawn(CloudSettings::default()).id();
        let parent = app.world_mut().spawn(GaussianSceneLoaded).id();
        app.world_mut().entity_mut(parent).add_child(child);

        app.update();

        assert_eq!(
            app.world().get::<GaussianLodSettings>(child),
            Some(&expected)
        );
        assert_eq!(
            app.world()
                .get::<CloudSettings>(child)
                .map(|settings| &settings.lod_debug),
            Some(&expected_debug)
        );
        assert!(app.world().get::<SceneRenderModeApplied>(parent).is_some());
    }
}

pub fn main() {
    setup_hooks();
    viewer_app();
}
