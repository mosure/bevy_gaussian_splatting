#define_import_path bevy_gaussian_splatting::bindings

#import bevy_render::globals::Globals
#import bevy_render::view::View


@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;

struct GaussianUniforms {
    global_transform: mat4x4<f32>,
    global_scale: f32,
};
@group(1) @binding(0) var<uniform> gaussian_uniforms: GaussianUniforms;

struct Gaussian {
    @location(0) rotation: vec4<f32>,
    @location(1) position: vec4<f32>,
    @location(2) scale_opacity: vec4<f32>,
    sh: array<f32, #{MAX_SH_COEFF_COUNT}>,
};
@group(2) @binding(0) var<storage, read_write> points: array<Gaussian>;


struct DrawIndirect {
    vertex_count: u32,
    instance_count: atomic<u32>,
    base_vertex: u32,
    base_instance: u32,
}
struct SortingGlobal {
    digit_histogram: array<array<atomic<u32>, #{RADIX_BASE}>, #{RADIX_DIGIT_PLACES}>,
    assignment_counter: atomic<u32>,
}
struct Entry {
    key: u32,
    value: u32,
}
@group(3) @binding(0) var<uniform> sorting_pass_index: u32;
@group(3) @binding(1) var<storage, read_write> sorting: SortingGlobal;
@group(3) @binding(2) var<storage, read_write> status_counters: array<array<atomic<u32>, #{RADIX_BASE}>>;
@group(3) @binding(3) var<storage, read_write> draw_indirect: DrawIndirect;
@group(3) @binding(4) var<storage, read_write> input_entries: array<Entry>;
@group(3) @binding(5) var<storage, read_write> output_entries: array<Entry>;
@group(3) @binding(6) var<storage, read> sorted_entries: array<Entry>;


struct ParticleBehavior {
    @location(0) indicies: vec4<i32>,
    @location(1) velocity: vec4<f32>,
    @location(2) acceleration: vec4<f32>,
}

// struct WaveletBehavior {
//     @location(0) index: u32,
//     @location(1) wavelet: array<f32, #{MAX_WAVELET_COEFF_COUNT}>,
// }

// newtonian behavior
@group(4) @binding(0) var<storage, read_write> particle_behaviors: array<ParticleBehavior>;

// // wavelet behavior
// @group(4) @binding(1) var<storage, read_write> morph_wavelets: array<WaveletInput>;


struct GaussianOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(linear) uv: vec2<f32>,
    @location(3) @interpolate(linear) major_minor: vec2<f32>,
};
