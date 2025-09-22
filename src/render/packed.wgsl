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

fn gaussian_position(point: Gaussian) -> vec3<f32> {
    return point.position_visibility.xyz;
}

fn gaussian_color(point: Gaussian, ray_direction: vec3<f32>) -> vec3<f32> {
    let sh = gaussian_spherical_harmonics(point);
    let color = spherical_harmonics_lookup(ray_direction, sh);
    return srgb_to_linear(color);
}

fn gaussian_spherical_harmonics(point: Gaussian) -> array<f32, #{SH_COEFF_COUNT}> {
    return point.sh;
}

fn gaussian_rotation(point: Gaussian) -> vec4<f32> {
    return point.rotation;
}

fn gaussian_scale(point: Gaussian) -> vec3<f32> {
    return point.scale_opacity.xyz;
}

fn gaussian_opacity(point: Gaussian) -> f32 {
    return point.scale_opacity.w;
}

fn gaussian_visibility(point: Gaussian) -> f32 {
    return point.position_visibility.w;
}

fn get_position(index: u32) -> vec3<f32> {
    return gaussian_position(points[index]);
}

fn get_color(
    index: u32,
    ray_direction: vec3<f32>,
) -> vec3<f32> {
    return gaussian_color(points[index], ray_direction);
}

fn get_spherical_harmonics(index: u32) -> array<f32, #{SH_COEFF_COUNT}> {
    return gaussian_spherical_harmonics(points[index]);
}

fn get_rotation(index: u32) -> vec4<f32> {
    return gaussian_rotation(points[index]);
}

fn get_scale(index: u32) -> vec3<f32> {
    return gaussian_scale(points[index]);
}

fn get_opacity(index: u32) -> f32 {
    return gaussian_opacity(points[index]);
}

fn get_visibility(index: u32) -> f32 {
    return gaussian_visibility(points[index]);
}

#ifdef BINARY_GAUSSIAN_OP

fn get_rhs_position(index: u32) -> vec3<f32> {
    return gaussian_position(rhs_points[index]);
}

fn get_rhs_color(
    index: u32,
    ray_direction: vec3<f32>,
) -> vec3<f32> {
    return gaussian_color(rhs_points[index], ray_direction);
}

fn get_rhs_spherical_harmonics(index: u32) -> array<f32, #{SH_COEFF_COUNT}> {
    return gaussian_spherical_harmonics(rhs_points[index]);
}

fn get_rhs_rotation(index: u32) -> vec4<f32> {
    return gaussian_rotation(rhs_points[index]);
}

fn get_rhs_scale(index: u32) -> vec3<f32> {
    return gaussian_scale(rhs_points[index]);
}

fn get_rhs_opacity(index: u32) -> f32 {
    return gaussian_opacity(rhs_points[index]);
}

fn get_rhs_visibility(index: u32) -> f32 {
    return gaussian_visibility(rhs_points[index]);
}

#endif


#endif
