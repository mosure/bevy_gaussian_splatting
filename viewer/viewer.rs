// TODO: move to editor crate
use std::path::PathBuf;

use bevy::{
    prelude::*,
    app::AppExit,
    core::{
        Name,
        FrameCount,
    },
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{
        DiagnosticsStore,
        FrameTimeDiagnosticsPlugin,
    },
    render::view::screenshot::ScreenshotManager,
    window::PrimaryWindow,
};
use bevy_args::{
    BevyArgsPlugin,
    parse_args,
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_panorbit_camera::{
    PanOrbitCamera,
    PanOrbitCameraPlugin,
};

use bevy_gaussian_splatting::{
    GaussianCloud,
    GaussianSplattingBundle,
    GaussianSplattingPlugin,
    random_gaussians,
    utils::{
        GaussianSplattingViewer,
        log,
        setup_hooks,
    },
};

#[cfg(feature = "material_noise")]
use bevy_gaussian_splatting::material::noise::NoiseMaterial;

#[cfg(feature = "morph_particles")]
use bevy_gaussian_splatting::morph::particle::{
    ParticleBehaviors,
    random_particle_behaviors,
};

#[cfg(feature = "query_select")]
use bevy_gaussian_splatting::query::select::{
    InvertSelectionEvent,
    SaveSelectionEvent,
};

#[cfg(feature = "query_sparse")]
use bevy_gaussian_splatting::query::sparse::SparseSelect;


fn setup_gaussian_cloud(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_splatting_viewer: Res<GaussianSplattingViewer>,
    mut gaussian_assets: ResMut<Assets<GaussianCloud>>,
) {
    let cloud: Handle<GaussianCloud>;
    if gaussian_splatting_viewer.gaussian_count > 0 {
        log(&format!("generating {} gaussians", gaussian_splatting_viewer.gaussian_count));
        cloud = gaussian_assets.add(random_gaussians(gaussian_splatting_viewer.gaussian_count));
    } else if !gaussian_splatting_viewer.input_file.is_empty() {
        log(&format!("loading {}", gaussian_splatting_viewer.input_file));
        cloud = asset_server.load(&gaussian_splatting_viewer.input_file);
    } else {
        cloud = gaussian_assets.add(GaussianCloud::test_model());
    }

    commands.spawn((
        GaussianSplattingBundle {
            cloud,
            ..default()
        },
        Name::new("gaussian_cloud"),
    ));

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            tonemapping: Tonemapping::None,
            ..default()
        },
        PanOrbitCamera {
            allow_upside_down: true,
            orbit_smoothness: 0.0,
            pan_smoothness: 0.0,
            zoom_smoothness: 0.0,
            ..default()
        },
    ));
}


#[cfg(feature = "morph_particles")]
fn setup_particle_behavior(
    mut commands: Commands,
    gaussian_splatting_viewer: Res<GaussianSplattingViewer>,
    mut particle_behavior_assets: ResMut<Assets<ParticleBehaviors>>,
    gaussian_cloud: Query<(
        Entity,
        &Handle<GaussianCloud>,
        Without<Handle<ParticleBehaviors>>,
    )>,
) {
    if gaussian_cloud.is_empty() {
        return;
    }

    let mut particle_behaviors = None;
    if gaussian_splatting_viewer.particle_count > 0 {
        log(&format!("generating {} particle behaviors", gaussian_splatting_viewer.particle_count));
        particle_behaviors = particle_behavior_assets.add(random_particle_behaviors(gaussian_splatting_viewer.particle_count)).into();
    }

    if let Some(particle_behaviors) = particle_behaviors {
        commands.entity(gaussian_cloud.single().0)
            .insert(particle_behaviors);
    }
}

#[cfg(feature = "material_noise")]
fn setup_noise_material(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_clouds: Query<(
        Entity,
        &Handle<GaussianCloud>,
        Without<NoiseMaterial>,
    )>,
) {
    if gaussian_clouds.is_empty() {
        return;
    }

    for (
        entity,
        cloud_handle,
        _
    ) in gaussian_clouds.iter() {
        if Some(bevy::asset::LoadState::Loading) == asset_server.get_load_state(cloud_handle) {
            continue;
        }

        commands.entity(entity)
            .insert(NoiseMaterial::default());
    }
}

#[cfg(feature = "query_sparse")]
fn setup_sparse_select(
    mut commands: Commands,
    gaussian_cloud: Query<(
        Entity,
        &Handle<GaussianCloud>,
        Without<SparseSelect>,
    )>,
) {
    if gaussian_cloud.is_empty() {
        return;
    }

    commands.entity(gaussian_cloud.single().0)
        .insert(SparseSelect {
            completed: true,
            ..default()
        });
}


fn example_app() {
    let config = parse_args::<GaussianSplattingViewer>();
    log(&format!("{:?}", config));

    let mut app = App::new();

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        fit_canvas_to_parent: true,
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
        canvas: Some("#bevy".to_string()),
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: false,
        resolution: (config.width, config.height).into(),
        title: config.name.clone(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    // setup for gaussian viewer app
    app.insert_resource(ClearColor(Color::rgb_u8(0, 0, 0)));
    app.add_plugins(
        DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(WindowPlugin {
            primary_window,
            ..default()
        })
    );
    app.add_plugins(BevyArgsPlugin::<GaussianSplattingViewer>::default());
    app.add_plugins(PanOrbitCameraPlugin);

    app.insert_resource(match config.msaa_samples {
        1 => Msaa::Off,
        2 => Msaa::Sample2,
        4 => Msaa::Sample4,
        8 => Msaa::Sample8,
        _ => Msaa::default(),
    });

    if config.editor {
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if config.press_esc_close {
        app.add_systems(Update, press_esc_close);
    }

    if config.press_s_screenshot {
        app.add_systems(Update, press_s_screenshot);
    }

    if config.show_fps {
        app.add_plugins(FrameTimeDiagnosticsPlugin);
        app.add_systems(Startup, fps_display_setup);
        app.add_systems(Update, fps_update_system);
    }


    // setup for gaussian splatting
    app.add_plugins(GaussianSplattingPlugin);
    app.add_systems(Startup, setup_gaussian_cloud);

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
    keys: Res<ButtonInput<KeyCode>>,
    main_window: Query<Entity, With<PrimaryWindow>>,
    mut screenshot_manager: ResMut<ScreenshotManager>,
    current_frame: Res<FrameCount>,
) {
    if keys.just_pressed(KeyCode::KeyS) {
        if let Ok(window_entity) = main_window.get_single() {
            let images_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("screenshots");
            std::fs::create_dir_all(&images_dir).unwrap();
            let output_path = images_dir.join(format!("output_{}.png", current_frame.0));

            screenshot_manager.take_screenshot(window_entity, move |image: Image| {
                let dyn_img = image.clone().try_into_dynamic().unwrap();
                let img = dyn_img.to_rgba8();
                img.save(&output_path).unwrap();

                log(&format!("saved screenshot to {}", output_path.display()));
            }).unwrap();
        }
    }
}

pub fn press_esc_close(
    keys: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.send(AppExit);
    }
}

#[cfg(feature = "query_select")]
fn press_i_invert_selection(
    keys: Res<ButtonInput<KeyCode>>,
    mut select_inverse_events: EventWriter<InvertSelectionEvent>,
) {
    if keys.just_pressed(KeyCode::KeyI) {
        log("inverting selection");
        select_inverse_events.send(InvertSelectionEvent);
    }
}

#[cfg(feature = "query_select")]
fn press_o_save_selection(
    keys: Res<ButtonInput<KeyCode>>,
    mut select_inverse_events: EventWriter<SaveSelectionEvent>,
) {
    if keys.just_pressed(KeyCode::KeyO) {
        log("saving selection");
        select_inverse_events.send(SaveSelectionEvent);
    }
}

fn fps_display_setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    commands.spawn((
        TextBundle::from_sections([
            TextSection::new(
                "fps: ",
                TextStyle {
                    font: asset_server.load("fonts/Caveat-Bold.ttf"),
                    font_size: 60.0,
                    color: Color::WHITE,
                },
            ),
            TextSection::from_style(TextStyle {
                font: asset_server.load("fonts/Caveat-Medium.ttf"),
                font_size: 60.0,
                color: Color::GOLD,
            }),
        ]).with_style(Style {
            position_type: PositionType::Absolute,
            bottom: Val::Px(5.0),
            left: Val::Px(15.0),
            ..default()
        }),
        FpsText,
    ));
}

#[derive(Component)]
struct FpsText;

fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                text.sections[1].value = format!("{value:.2}");
            }
        }
    }
}


pub fn main() {
    setup_hooks();
    example_app();
}
