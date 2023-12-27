#define_import_path bevy_gaussian_splatting::bindings

#import bevy_render::globals::Globals
#import bevy_render::view::View


@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;

struct GaussianUniforms {
    global_transform: mat4x4<f32>,
    global_scale: f32,
    count: u32,
};
@group(1) @binding(0) var<uniform> gaussian_uniforms: GaussianUniforms;

struct Gaussian {
    @location(0) rotation: vec4<f32>,
    @location(1) position_visibility: vec4<f32>,
    @location(2) scale_opacity: vec4<f32>,
    sh: array<f32, #{SH_COEFF_COUNT}>,
};

#ifdef READ_WRITE_POINTS
@group(2) @binding(0) var<storage, read_write> points: array<Gaussian>;
#else
@group(2) @binding(0) var<storage, read> points: array<Gaussian>;
#endif


struct DrawIndirect {
    vertex_count: u32,
    instance_count: atomic<u32>,
    base_vertex: u32,
    base_instance: u32,
}

struct Entry {
    key: u32,
    value: u32,
}
