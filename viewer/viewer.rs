use bevy::{
    prelude::*,
    app::AppExit,
    core::Name,
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{
        DiagnosticsStore,
        FrameTimeDiagnosticsPlugin,
    },
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
        get_arg,
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


pub struct GaussianSplattingViewer {
    pub editor: bool,
    pub esc_close: bool,
    pub show_fps: bool,
    pub width: f32,
    pub height: f32,
    pub name: String,
}

impl Default for GaussianSplattingViewer {
    fn default() -> GaussianSplattingViewer {
        GaussianSplattingViewer {
            editor: true,
            esc_close: true,
            show_fps: true,
            width: 1920.0,
            height: 1080.0,
            name: "bevy_gaussian_splatting".to_string(),
        }
    }
}


fn setup_gaussian_cloud(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut gaussian_assets: ResMut<Assets<GaussianCloud>>,
) {
    let cloud: Handle<GaussianCloud>;

    // TODO: add proper GaussianSplattingViewer argument parsing
    let file_arg = get_arg(1);
    if let Some(n) = file_arg.clone().and_then(|s| s.parse::<usize>().ok()) {
        println!("generating {} gaussians", n);
        cloud = gaussian_assets.add(random_gaussians(n));
    } else if let Some(filename) = file_arg {
        if filename == "--help" {
            println!("usage: cargo run -- [filename | n]");
            return;
        }

        println!("loading {}", filename);
        cloud = asset_server.load(filename.to_string());
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
        PanOrbitCamera{
            allow_upside_down: true,
            ..default()
        },
    ));
}


#[cfg(feature = "morph_particles")]
fn setup_particle_behavior(
    mut commands: Commands,
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

    let file_arg = get_arg(1);
    if let Some(_n) = file_arg.clone().and_then(|s| s.parse::<usize>().ok()) {
        let behavior_arg = get_arg(2);
        if let Some(k) = behavior_arg.clone().and_then(|s| s.parse::<usize>().ok()) {
            println!("generating {} particle behaviors", k);
            particle_behaviors = particle_behavior_assets.add(random_particle_behaviors(k)).into();
        }
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


#[cfg(feature = "query_select")]
fn press_i_invert_selection(
    keys: Res<Input<KeyCode>>,
    mut select_inverse_events: EventWriter<InvertSelectionEvent>,
) {
    if keys.just_pressed(KeyCode::I) {
        println!("inverting selection");
        select_inverse_events.send(InvertSelectionEvent);
    }
}

#[cfg(feature = "query_select")]
fn press_o_save_selection(
    keys: Res<Input<KeyCode>>,
    mut select_inverse_events: EventWriter<SaveSelectionEvent>,
) {
    if keys.just_pressed(KeyCode::O) {
        println!("saving selection");
        select_inverse_events.send(SaveSelectionEvent);
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
    let config = GaussianSplattingViewer::default();
    let mut app = App::new();

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        fit_canvas_to_parent: true,
        mode: bevy::window::WindowMode::Windowed,
        present_mode: bevy::window::PresentMode::AutoVsync,
        prevent_default_event_handling: true,
        title: config.name.clone(),
        ..default()
    });

    #[cfg(not(target_arch = "wasm32"))]
    let primary_window = Some(Window {
        fit_canvas_to_parent: true,
        mode: bevy::window::WindowMode::Windowed,
        present_mode: bevy::window::PresentMode::AutoVsync,
        prevent_default_event_handling: false,
        resolution: (config.width, config.height).into(),
        title: config.name.clone(),
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
        }),
    );
    app.add_plugins((
        PanOrbitCameraPlugin,
    ));

    if config.editor {
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if config.esc_close {
        app.add_systems(Update, esc_close);
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


pub fn esc_close(
    keys: Res<Input<KeyCode>>,
    mut exit: EventWriter<AppExit>
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.send(AppExit);
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
        if let Some(fps) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
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
