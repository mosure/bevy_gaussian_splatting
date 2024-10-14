use bevy::{
    prelude::*,
    app::AppExit,
    core_pipeline::tonemapping::Tonemapping,
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
    Gaussian,
    GaussianCamera,
    GaussianCloud,
    GaussianMode,
    GaussianCloudSettings,
    GaussianSplattingBundle,
    GaussianSplattingPlugin,
    utils::{
        setup_hooks,
        GaussianSplattingViewer,
    },
    SphericalHarmonicCoefficients,
};


pub fn setup_surfel_compare(
    mut commands: Commands,
    mut gaussian_assets: ResMut<Assets<GaussianCloud>>,
) {
    let grid_size_x = 10;
    let grid_size_y = 10;
    let spacing = 5.0;
    let visualize_bounding_box = false;

    let mut blue_gaussians = Vec::new();
    let mut blue_sh = SphericalHarmonicCoefficients::default();
    blue_sh.set(2, 5.0);

    for i in 0..grid_size_x {
        for j in 0..grid_size_y {
            let x = i as f32 * spacing - (grid_size_x as f32 * spacing) / 2.0;
            let y = j as f32 * spacing - (grid_size_y as f32 * spacing) / 2.0;
            let position = [x, y, 0.0, 1.0];
            let scale = [1.0, 1.0, 0.1, 0.5];

            let gaussian = Gaussian {
                position_visibility: position.into(),
                rotation: [0.0, 0.0, 0.0, 1.0].into(),
                scale_opacity: scale.into(),
                spherical_harmonic: blue_sh.clone(),
            };
            blue_gaussians.push(gaussian);
        }
    }

    commands.spawn((
        GaussianSplattingBundle {
            cloud: gaussian_assets.add(GaussianCloud::from_gaussians(blue_gaussians)),
            settings: GaussianCloudSettings {
                visualize_bounding_box,
                ..default()
            },
            ..default()
        },
        Name::new("gaussian_cloud_3dgs"),
    ));

    let mut red_gaussians = Vec::new();
    let mut red_sh = SphericalHarmonicCoefficients::default();
    red_sh.set(0, 5.0);

    for i in 0..grid_size_x {
        for j in 0..grid_size_y {
            let x = i as f32 * spacing - (grid_size_x as f32 * spacing) / 2.0;
            let y = j as f32 * spacing - (grid_size_y as f32 * spacing) / 2.0;
            let position = [x, y, 0.0, 1.0];
            let scale = [1.0, 1.0, 0.0, 0.3];

            // let angle = std::f32::consts::PI / 2.0;
            // let rotation = Quat::from_rotation_y(angle).to_array().into();

            let gaussian = Gaussian {
                position_visibility: position.into(),
                rotation: [0.0, 0.0, 0.0, 1.0].into(),
                scale_opacity: scale.into(),
                spherical_harmonic: red_sh.clone(),
            };
            red_gaussians.push(gaussian);
        }
    }

    commands.spawn((
        GaussianSplattingBundle {
            cloud: gaussian_assets.add(GaussianCloud::from_gaussians(red_gaussians)),
            settings: GaussianCloudSettings {
                visualize_bounding_box,
                transform: Transform::from_translation(Vec3::new(spacing, spacing, 0.0)),
                gaussian_mode: GaussianMode::GaussianSurfel,
                ..default()
            },
            ..default()
        },
        Name::new("gaussian_cloud_2dgs"),
    ));

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 20.0)),
            tonemapping: Tonemapping::None,
            ..default()
        },
        PanOrbitCamera {
            allow_upside_down: true,
            ..default()
        },
        GaussianCamera,
    ));
}

fn compare_surfel_app() {
    let config = parse_args::<GaussianSplattingViewer>();
    let mut app = App::new();

    // setup for gaussian viewer app
    app.insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)));
    app.add_plugins(
        DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(WindowPlugin {
            primary_window: Some(Window {
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
    app.add_plugins(BevyArgsPlugin::<GaussianSplattingViewer>::default());
    app.add_plugins(PanOrbitCameraPlugin);

    if config.editor {
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if config.press_esc_close {
        app.add_systems(Update, esc_close);
    }

    // setup for gaussian splatting
    app.add_plugins(GaussianSplattingPlugin);
    app.add_systems(Startup, setup_surfel_compare);

    app.run();
}

pub fn esc_close(
    keys: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}

pub fn main() {
    setup_hooks();
    compare_surfel_app();
}
