const LOD_RENDER_FEATURES: [&str; 5] = [
    "CARGO_FEATURE_LOD",
    "CARGO_FEATURE_SORT_RADIX",
    "CARGO_FEATURE_BUFFER_STORAGE",
    "CARGO_FEATURE_BUFFER_TEXTURE",
    "CARGO_FEATURE_WEBGL2",
];

fn feature_enabled(name: &str) -> bool {
    // This build script reads only Cargo-provided feature inputs.
    std::env::var_os(name).is_some()
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(lod_render_path)");
    for feature in LOD_RENDER_FEATURES {
        println!("cargo::rerun-if-env-changed={feature}");
    }

    if feature_enabled("CARGO_FEATURE_LOD")
        && feature_enabled("CARGO_FEATURE_SORT_RADIX")
        && feature_enabled("CARGO_FEATURE_BUFFER_STORAGE")
        && !feature_enabled("CARGO_FEATURE_BUFFER_TEXTURE")
        && !feature_enabled("CARGO_FEATURE_WEBGL2")
    {
        println!("cargo::rustc-cfg=lod_render_path");
    }
}
