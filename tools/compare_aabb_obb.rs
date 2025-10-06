use bevy::{app::AppExit, core_pipeline::tonemapping::Tonemapping, prelude::*};
use bevy_args::{BevyArgsPlugin, parse_args};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, quick::WorldInspectorPlugin};
use bevy_interleave::prelude::Planar;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianSplattingPlugin, PlanarGaussian3d,
    PlanarGaussian3dHandle, SphericalHarmonicCoefficients,
    utils::{GaussianSplattingViewer, setup_hooks},
};

pub fn setup_aabb_obb_compare(
    mut commands: Commands,
    mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
) {
    let mut blue_sh = SphericalHarmonicCoefficients::default();
    blue_sh.set(2, 5.0);

    let blue_aabb_gaussian = Gaussian3d {
        position_visibility: [0.0, 0.0, 0.0, 1.0].into(),
        rotation: [0.89, 0.0, -0.432, 0.144].into(),
        scale_opacity: [10.0, 1.0, 1.0, 0.5].into(),
        spherical_harmonic: blue_sh,
    };

    commands.spawn((
        PlanarGaussian3dHandle(gaussian_assets.add(PlanarGaussian3d::from_interleaved(vec![
            blue_aabb_gaussian,
            blue_aabb_gaussian,
        ]))),
        CloudSettings {
            aabb: true,
            visualize_bounding_box: true,
            ..default()
        },
        Name::new("gaussian_cloud_aabb"),
    ));

    let mut red_sh = SphericalHarmonicCoefficients::default();
    red_sh.set(0, 5.0);

    let red_obb_gaussian = Gaussian3d {
        position_visibility: [0.0, 0.0, 0.0, 1.0].into(),
        rotation: [0.89, 0.0, -0.432, 0.144].into(),
        scale_opacity: [10.0, 1.0, 1.0, 0.5].into(),
        spherical_harmonic: red_sh,
    };

    commands.spawn((
        PlanarGaussian3dHandle(gaussian_assets.add(PlanarGaussian3d::from_interleaved(vec![
            red_obb_gaussian,
            red_obb_gaussian,
        ]))),
        CloudSettings {
            aabb: false,
            visualize_bounding_box: true,
            ..default()
        },
        Name::new("gaussian_cloud_obb"),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        Tonemapping::None,
        PanOrbitCamera {
            allow_upside_down: true,
            ..default()
        },
        GaussianCamera::default(),
    ));
}

fn compare_aabb_obb_app() {
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
        app.add_plugins(EguiPlugin::default());
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if config.press_esc_close {
        app.add_systems(Update, esc_close);
    }

    // setup for gaussian splatting
    app.add_plugins(GaussianSplattingPlugin);
    app.add_systems(Startup, setup_aabb_obb_compare);

    app.run();
}

pub fn esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: EventWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

pub fn main() {
    setup_hooks();
    compare_aabb_obb_app();
}
