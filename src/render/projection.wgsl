#define_import_path bevy_gaussian_splatting::projection

#import bevy_gaussian_splatting::bindings::{view, gaussian_uniforms}
#import bevy_gaussian_splatting::gaussian_3d::compute_cov2d_3dgs
#import bevy_gaussian_splatting::planar::{get_position, get_color, get_scale}
#import bevy_gaussian_splatting::helpers::gaussian_mip_support_radius_world

// Shared authored projection operations. Each consumer retains its own support,
// opacity and visibility semantics rather than reinterpreting sampling settings.
fn projected_world_position(index: u32) -> vec3<f32> {
    return (gaussian_uniforms.transform * vec4<f32>(get_position(index), 1.0)).xyz;
}

// Match the per-cloud quad renderer's conservative authored-support policy.
// A far off-axis perspective Jacobian can produce a viewport-sized ellipse
// even when the Gaussian's entire world-space support misses the frustum.
fn projected_support_in_frustum(index: u32, position: vec3<f32>, cutoff: f32) -> bool {
    let mip_radius = gaussian_mip_support_radius_world(position, cutoff);
    if !(mip_radius >= 0.0) { return true; }
    let scale = abs(get_scale(index));
    let radius = cutoff * abs(gaussian_uniforms.global_scale) * max(scale.x, max(scale.y, scale.z))
        * gaussian_uniforms.transform_scale_bound + mip_radius;
    if !(radius >= 0.0) { return true; }
    for (var plane = 0u; plane < 6u; plane += 1u) {
        if dot(view.frustum[plane].xyz, position) + view.frustum[plane].w < -radius {
            return false;
        }
    }
    return true;
}

fn projected_covariance(index: u32, position: vec3<f32>) -> vec4<f32> {
    return compute_cov2d_3dgs(position, position, position, index, index, 1.0, false).filtered;
}

fn projected_color(index: u32, position: vec3<f32>) -> vec3<f32> {
    let ray = normalize(position - view.world_position);
    let transform = gaussian_uniforms.transform;
    let local_ray = normalize(vec3<f32>(
        dot(normalize(transform[0].xyz), ray),
        dot(normalize(transform[1].xyz), ray),
        dot(normalize(transform[2].xyz), ray)));
    return get_color(index, local_ray);
}
