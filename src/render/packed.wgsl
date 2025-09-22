#define_import_path bevy_gaussian_splatting::packed

#import bevy_gaussian_splatting::bindings::points
#ifdef BINARY_GAUSSIAN_OP
    #import bevy_gaussian_splatting::bindings::rhs_points
#endif

#import bevy_gaussian_splatting::spherical_harmonics::{
    spherical_harmonics_lookup,
    srgb_to_linear,
}

#ifdef PACKED_F32

fn get_position(index: u32) -> vec3<f32> {
    return points[index].position_visibility.xyz;
}

fn get_color(
    index: u32,
    ray_direction: vec3<f32>,
) -> vec3<f32> {
    let sh = get_spherical_harmonics(index);
    let color = spherical_harmonics_lookup(ray_direction, sh);
    return srgb_to_linear(color);
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

#ifdef BINARY_GAUSSIAN_OP

fn get_rhs_position(index: u32) -> vec3<f32> {
    return rhs_points[index].position_visibility.xyz;
}

fn get_rhs_color(
    index: u32,
    ray_direction: vec3<f32>,
) -> vec3<f32> {
    let sh = get_rhs_spherical_harmonics(index);
    let color = spherical_harmonics_lookup(ray_direction, sh);
    return srgb_to_linear(color);
}

fn get_rhs_spherical_harmonics(index: u32) -> array<f32, #{SH_COEFF_COUNT}> {
    return rhs_points[index].sh;
}

fn get_rhs_rotation(index: u32) -> vec4<f32> {
    return rhs_points[index].rotation;
}

fn get_rhs_scale(index: u32) -> vec3<f32> {
    return rhs_points[index].scale_opacity.xyz;
}

fn get_rhs_opacity(index: u32) -> f32 {
    return rhs_points[index].scale_opacity.w;
}

fn get_rhs_visibility(index: u32) -> f32 {
    return rhs_points[index].position_visibility.w;
}

#endif

#endif
