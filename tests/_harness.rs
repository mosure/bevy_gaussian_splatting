use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use bevy_gaussian_splatting::GaussianSplattingPlugin;


// scraping this for now until bevy ci testing is more stable
// #[test] and main thread limitations exist
// see: https://github.com/anchpop/endless-sea/blob/3b8481f1152293907794d60e920d4cc5a7ca8f40/src/tests/helpers.rs#L69-L83



#[derive(Resource)]
pub struct TestHarness {
    pub resolution: (f32, f32),
}

pub fn test_harness_app(
    harness: TestHarness,
) -> App {
    let mut app = App::new();

    app.insert_resource(ClearColor(Color::rgb_u8(0, 0, 0)));
    app.add_plugins(
        DefaultPlugins
        .set(WindowPlugin {
            primary_window: Some(Window {
                fit_canvas_to_parent: false,
                mode: bevy::window::WindowMode::Windowed,
                present_mode: bevy::window::PresentMode::AutoVsync,
                prevent_default_event_handling: false,
                resolution: harness.resolution.into(),
                title: "bevy_gaussian_splatting pipeline test".to_string(),
                ..default()
            }),
            ..default()
        }),
    );

    app.add_plugins(GaussianSplattingPlugin);

    app.insert_resource(harness);

    app
}



pub struct TestState {
    pub test_completed: bool,
}

impl Default for TestState {
    fn default() -> Self {
        TestState {
            test_completed: false,
        }
    }
}

pub type TestStateArc = Arc<Mutex<TestState>>;


// use bevy::{
//     render::view::screenshot::ScreenshotManager,
//     window::PrimaryWindow,
// };

// pub fn capture_example(
//     main_window: Query<Entity, With<PrimaryWindow>>,
//     mut screenshot_manager: ResMut<ScreenshotManager>,
// ) {
//     if let Ok(window_entity) = main_window.get_single() {
//         screenshot_manager.take_screenshot(window_entity, move |image: Image| {
//             // TODO: assert that the image is correct
//         }).unwrap();
//     }
// }
