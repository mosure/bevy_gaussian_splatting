#define_import_path bevy_gaussian_splatting::point_types

// One projection per candidate, shared by all samples and both visibility passes.
struct PointGaussian {
    mean: vec2<f32>,
    basis: vec2<f32>,
    basis_y: f32,
    opacity: f32,
    dilog: f32,
    depth: f32,
    color: vec3<f32>,
    seed: u32,
    epoch: u32,
    failure: u32,
    proposal_rate: f32,
    // Zero selects the full radial process; positive selects rectangle thinning.
    proposal_peak: f32,
}

struct PointConfig {
    width: u32,
    height: u32,
    samples: u32,
    record_count: u32,
    item_count: u32,
    group_count: u32,
    block_count: u32,
    max_points: u32,
    frame: u32,
    seed: u32,
    noise_frame: u32,
    padding: u32,
    viewport_origin: vec2<u32>,
    padding_tail: vec2<u32>,
}
