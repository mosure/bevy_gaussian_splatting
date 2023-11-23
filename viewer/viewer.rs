use bevy::{
    prelude::*,
    app::AppExit,
    core::Name,
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
    GaussianCloudSettings,
    GaussianSplattingBundle,
    GaussianSplattingPlugin,
    render::morph::{
        ParticleBehaviors,
        random_particle_behaviors,
    },
    random_gaussians,
    utils::setup_hooks,
};


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
    mut particle_behavior_assets: ResMut<Assets<ParticleBehaviors>>,
) {
    let cloud: Handle<GaussianCloud>;
    let settings = GaussianCloudSettings {
        aabb: true,
        visualize_bounding_box: false,
        ..default()
    };

    let mut particle_behaviors = None;

    // TODO: add proper GaussianSplattingViewer argument parsing
    let file_arg = std::env::args().nth(1);
    if let Some(n) = file_arg.clone().and_then(|s| s.parse::<usize>().ok()) {
        println!("generating {} gaussians", n);
        cloud = gaussian_assets.add(random_gaussians(n));

        let behavior_arg = std::env::args().nth(2);
        if let Some(k) = behavior_arg.clone().and_then(|s| s.parse::<usize>().ok()) {
            println!("generating {} particle behaviors", k);
            particle_behaviors = particle_behavior_assets.add(random_particle_behaviors(k)).into();
        }
    } else if let Some(filename) = file_arg {
        println!("loading {}", filename);
        cloud = asset_server.load(filename.to_string());
    } else {
        cloud = gaussian_assets.add(GaussianCloud::test_model());
    }

    let mut entity = commands.spawn((
        GaussianSplattingBundle {
            cloud,
            settings,
            ..default()
        },
        Name::new("gaussian_cloud"),
    ));

    if let Some(particle_behaviors) = particle_behaviors {
        entity.insert(particle_behaviors);
    }

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            ..default()
        },
        PanOrbitCamera{
            allow_upside_down: true,
            ..default()
        },
    ));
}


fn example_app() {
    let config = GaussianSplattingViewer::default();
    let mut app = App::new();

    // setup for gaussian viewer app
    app.insert_resource(ClearColor(Color::rgb_u8(0, 0, 0)));
    app.add_plugins(
        DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(WindowPlugin {
            primary_window: Some(Window {
                fit_canvas_to_parent: false,
                mode: bevy::window::WindowMode::Windowed,
                present_mode: bevy::window::PresentMode::AutoVsync,
                prevent_default_event_handling: false,
                resolution: (config.width, config.height).into(),
                title: config.name.clone(),
                ..default()
            }),
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
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        app.add_systems(Startup, fps_display_setup);
        app.add_systems(Update, fps_update_system);
    }


    // setup for gaussian splatting
    app.add_plugins(GaussianSplattingPlugin);
    app.add_systems(Startup, setup_gaussian_cloud);

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
