#define_import_path bevy_gaussian_splatting::ordered_types

struct OrderedGaussian {
    mean: vec2<f32>,
    axis_x: vec2<f32>,
    axis_y: vec2<f32>,
    conic_xy: vec2<f32>,
    color: vec4<f32>,
    conic_z: f32,
    depth: f32,
    key: u32,
    cutoff_squared: f32,
    #ifdef LOD_SPATIAL_MORPH
        parent_color: vec4<f32>,
        // Parent/child optical-depth coefficients, spatial weight, active flag.
        morph: vec4<f32>,
    #endif
}

struct OrderedEntry { key: u32, value: u32 }

struct OrderedConfig {
    viewport: vec4<u32>,
    capacity: u32,
    groups: u32,
    padding: vec2<u32>,
}
