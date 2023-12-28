#define_import_path bevy_gaussian_splatting::planar

#import bevy_gaussian_splatting::bindings::{
    position_visibility,
    spherical_harmonics,
    rotation,
    scale_opacity,
};


#ifdef PLANAR_F32

fn get_position(index: u32) -> vec3<f32> {
    return position_visibility[index].position;
}

fn get_spherical_harmonics(index: u32) -> array<f32, #{SH_COEFF_COUNT}> {
    return spherical_harmonics[index];
}

fn get_rotation(index: u32) -> vec4<f32> {
    return rotation[index].rotation;
}

fn get_scale(index: u32) -> vec3<f32> {
    return scale_opacity[index].scale;
}

fn get_opacity(index: u32) -> f32 {
    return scale_opacity[index].opacity;
}

fn get_visibility(index: u32) -> f32 {
    return position_visibility[index].visibility;
}

#endif
