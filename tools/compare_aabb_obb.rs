use bevy::{
    prelude::*,
    app::AppExit,
    core::Name,
    core_pipeline::tonemapping::Tonemapping,
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_panorbit_camera::{
    PanOrbitCamera,
    PanOrbitCameraPlugin,
};

use bevy_gaussian_splatting::{
    Gaussian,
    GaussianCloud,
    GaussianCloudSettings,
    GaussianSplattingBundle,
    GaussianSplattingPlugin,
    utils::setup_hooks, SphericalHarmonicCoefficients,
};


// TODO: move to editor crate
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


pub fn setup_aabb_obb_compare(
    mut commands: Commands,
    mut gaussian_assets: ResMut<Assets<GaussianCloud>>,
) {
    let mut blue_sh = SphericalHarmonicCoefficients::default();
    blue_sh.set(2, 5.0);

    let blue_aabb_gaussian = Gaussian {
        position_visibility: [0.0, 0.0, 0.0, 1.0].into(),
        rotation: [0.89, 0.0, -0.432, 0.144].into(),
        scale_opacity: [10.0, 1.0, 1.0, 0.5].into(),
        spherical_harmonic: blue_sh,
    };

    commands.spawn((
        GaussianSplattingBundle {
            cloud: gaussian_assets.add(
            GaussianCloud::from_gaussians(vec![
                    blue_aabb_gaussian,
                    blue_aabb_gaussian,
                ])
            ),
            settings: GaussianCloudSettings {
                aabb: true,
                visualize_bounding_box: true,
                ..default()
            },
            ..default()
        },
        Name::new("gaussian_cloud_aabb"),
    ));

    let mut red_sh = SphericalHarmonicCoefficients::default();
    red_sh.set(0, 5.0);

    let red_obb_gaussian = Gaussian {
        position_visibility: [0.0, 0.0, 0.0, 1.0].into(),
        rotation: [0.89, 0.0, -0.432, 0.144].into(),
        scale_opacity: [10.0, 1.0, 1.0, 0.5].into(),
        spherical_harmonic: red_sh,
    };

    commands.spawn((
        GaussianSplattingBundle {
            cloud: gaussian_assets.add(
            GaussianCloud::from_gaussians(vec![
                    red_obb_gaussian,
                    red_obb_gaussian,
                ])
            ),
            settings: GaussianCloudSettings {
                aabb: false,
                visualize_bounding_box: true,
                ..default()
            },
            ..default()
        },
        Name::new("gaussian_cloud_obb"),
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

fn compare_aabb_obb_app() {
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

    // setup for gaussian splatting
    app.add_plugins(GaussianSplattingPlugin);
    app.add_systems(Startup, setup_aabb_obb_compare);

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

pub fn main() {
    setup_hooks();
    compare_aabb_obb_app();
}
