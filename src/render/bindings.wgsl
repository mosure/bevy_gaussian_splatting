#define_import_path bevy_gaussian_splatting::bindings

#import bevy_render::globals::Globals
#import bevy_render::view::View


@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;

struct GaussianUniforms {
    global_transform: mat4x4<f32>,
    global_scale: f32,
    count: u32,
    count_root_ceil: u32,
};
@group(1) @binding(0) var<uniform> gaussian_uniforms: GaussianUniforms;


// TODO: move these bindings to packed vs. planar
#ifdef PACKED_F32
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
#endif


#ifdef PLANAR_F32
#ifdef READ_WRITE_POINTS
@group(2) @binding(0) var<storage, read_write> position_visibility: array<vec4<f32>>;
#else
@group(2) @binding(0) var<storage, read> position_visibility: array<vec4<f32>>;
#endif

@group(2) @binding(1) var<storage, read> spherical_harmonics: array<array<f32, #{SH_COEFF_COUNT}>>;
@group(2) @binding(2) var<storage, read> rotation: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read> scale_opacity: array<vec4<f32>>;
#endif


#ifdef PLANAR_F16
#ifdef READ_WRITE_POINTS
@group(2) @binding(0) var<storage, read_write> position_visibility: array<vec4<f32>>;
#else
@group(2) @binding(0) var<storage, read> position_visibility: array<vec4<f32>>;
#endif

@group(2) @binding(1) var<storage, read> spherical_harmonics: array<array<u32, #{HALF_SH_COEFF_COUNT}>>;
@group(2) @binding(2) var<storage, read> rotation_scale_opacity: array<vec4<u32>>;
#endif


#ifdef PLANAR_TEXTURE_F16
@group(2) @binding(0) var position_visibility: texture_2d<f32>;

#if SH_VEC4_PLANES == 1
@group(2) @binding(1) var spherical_harmonics: texture_2d<u32>;
#else
@group(2) @binding(1) var spherical_harmonics: texture_2d_array<u32>;
#endif

@group(2) @binding(2) var rotation_scale_opacity: texture_2d<u32>;
#endif


#ifdef PLANAR_TEXTURE_F32
@group(2) @binding(0) var position_visibility: texture_2d<f32>;

#if SH_VEC4_PLANES == 1
@group(2) @binding(1) var spherical_harmonics: texture_2d<f32>;
#else
@group(2) @binding(1) var spherical_harmonics: texture_2d_array<f32>;
#endif

@group(2) @binding(2) var rotation_scale_opacity: texture_2d<f32>;
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
