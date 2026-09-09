const LOD_RENDER_FEATURES: [&str; 3] = [
    "CARGO_FEATURE_LOD",
    "CARGO_FEATURE_SORT_RADIX",
    "CARGO_FEATURE_BUFFER_STORAGE",
];

fn feature_enabled(name: &str) -> bool {
    // This build script reads only Cargo-provided feature inputs.
    std::env::var_os(name).is_some()
}

fn main() {
    // Only this script and feature inputs affect its output. Avoid Cargo's
    // default package-wide file scan invalidating builds after doc/artifact edits.
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(lod_render_path)");
    for feature in LOD_RENDER_FEATURES {
        println!("cargo::rerun-if-env-changed={feature}");
    }

    if feature_enabled("CARGO_FEATURE_LOD")
        && feature_enabled("CARGO_FEATURE_SORT_RADIX")
        && feature_enabled("CARGO_FEATURE_BUFFER_STORAGE")
    {
        println!("cargo::rustc-cfg=lod_render_path");
    }
}
