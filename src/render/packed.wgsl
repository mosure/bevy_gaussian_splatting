#define_import_path bevy_gaussian_splatting::packed

#import bevy_gaussian_splatting::bindings::points


#ifdef PACKED_F32

fn get_position(index: u32) -> vec3<f32> {
    return points[index].position_visibility.xyz;
}

fn get_spherical_harmonics(index: u32) -> array<f32, #{SH_COEFF_COUNT}> {
    return points[index].sh;
}

fn get_rotation(index: u32) -> vec4<f32> {
    return points[index].rotation;
}

fn get_scale(index: u32) -> vec3<f32> {
    return points[index].scale_opacity.xyz;
}

fn get_opacity(index: u32) -> f32 {
    return points[index].scale_opacity.w;
}

fn get_visibility(index: u32) -> f32 {
    return points[index].position_visibility.w;
}

#endif
