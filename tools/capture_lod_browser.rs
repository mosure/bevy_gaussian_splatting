//! Opt-in browser runner using the ordinary package and renderer plugins.
#[cfg(target_arch = "wasm32")]
fn main() {
    bevy_gaussian_splatting::utils::setup_hooks();
    bevy_gaussian_splatting::testing::lod_browser_capture::run();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("capture_lod_browser requires wasm32; see tools/build_lod_browser_capture.sh");
    std::process::exit(2);
}
