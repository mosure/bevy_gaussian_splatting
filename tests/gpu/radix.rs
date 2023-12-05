use std::sync::{
    Arc,
    Mutex,
};

use bevy::{
    prelude::*,
    app::AppExit,
    core::FrameCount,
    render::view::screenshot::ScreenshotManager,
    window::PrimaryWindow,
};

use bevy_gaussian_splatting::{
    GaussianCloud,
    GaussianSplattingBundle,
    random_gaussians,
};

use _harness::{
    TestHarness,
    test_harness_app,
    TestStateArc,
};

mod _harness;


fn main() {
    // let mut app = test_harness_app(TestHarness {
    //     resolution: (512.0, 512.0),
    // });

    // app.add_systems(Startup, setup);
    // app.add_systems(Update, capture_ready);

    // app.run();
}
