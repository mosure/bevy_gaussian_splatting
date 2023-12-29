#define_import_path bevy_gaussian_splatting::planar

#import bevy_gaussian_splatting::bindings::{
    position_visibility,
    spherical_harmonics,
    rotation,
    scale_opacity,
};

// TODO: type alias for results (e.g. downstream doesn't need to switch on f16/f32)


#ifdef PLANAR_F16
fn get_position(index: u32) -> vec3<f32> {
    return position_visibility[index].xyz;
}

// TODO: unpack u32 to f16 (or f32 if downstream doesn't support f16)
fn get_spherical_harmonics(index: u32) -> array<f16, #{SH_COEFF_COUNT}> {
    return spherical_harmonics[index];
}
#endif


#ifdef PLANAR_F32
fn get_position(index: u32) -> vec3<f32> {
    return position_visibility[index].xyz;
}

fn get_spherical_harmonics(index: u32) -> array<f32, #{SH_COEFF_COUNT}> {
    return spherical_harmonics[index];
}

fn get_rotation(index: u32) -> vec4<f32> {
    return rotation[index];
}

fn get_scale(index: u32) -> vec3<f32> {
    return scale_opacity[index].xyz;
}

fn get_opacity(index: u32) -> f32 {
    return scale_opacity[index].w;
}

fn get_visibility(index: u32) -> f32 {
    return position_visibility[index].w;
}
#endif
