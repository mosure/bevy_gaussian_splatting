use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use bevy_gaussian_splatting::GaussianSplattingPlugin;

// scraping this in CI for now until bevy ci testing is more stable
// #[test] + main thread, and windowless screenshot limitations exist
// see: https://github.com/anchpop/endless-sea/blob/3b8481f1152293907794d60e920d4cc5a7ca8f40/src/tests/helpers.rs#L69-L83

#[derive(Resource)]
pub struct TestHarness {
    pub resolution: (f32, f32),
}

pub fn test_harness_app(harness: TestHarness) -> App {
    let mut app = App::new();

    app.insert_resource(ClearColor(Color::rgb_u8(0, 0, 0)));
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            mode: bevy::window::WindowMode::Windowed,
            present_mode: bevy::window::PresentMode::AutoVsync,
            prevent_default_event_handling: false,
            resolution: harness.resolution.into(),
            title: "bevy_gaussian_splatting pipeline test".to_string(),
            ..default()
        }),
        ..default()
    }));

    app.add_plugins(GaussianSplattingPlugin);

    app.insert_resource(harness);

    app
}

#[derive(Default)]
pub struct TestState {
    pub test_loaded: bool,
    pub test_completed: bool,
}

pub type TestStateArc = Arc<Mutex<TestState>>;
