#define_import_path bevy_gaussian_splatting::surfel

#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
}
#import bevy_gaussian_splatting::helpers::{
    get_rotation_matrix,
    get_scale_matrix,
    intrinsic_matrix,
}

#ifdef PACKED
#ifdef PRECOMPUTE_COVARIANCE_3D
#import bevy_gaussian_splatting::packed::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_cov3d,
}
#else
#import bevy_gaussian_splatting::packed::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_rotation,
    get_scale,
}
#endif
#else

#ifdef BUFFER_STORAGE
#ifdef PRECOMPUTE_COVARIANCE_3D
#import bevy_gaussian_splatting::planar::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_cov3d,
}
#else
#import bevy_gaussian_splatting::planar::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_rotation,
    get_scale,
}
#endif
#endif

#endif


#ifdef BUFFER_TEXTURE
#ifdef PRECOMPUTE_COVARIANCE_3D
#import bevy_gaussian_splatting::texture::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_cov3d,
    location,
}
#else
#import bevy_gaussian_splatting::texture::{
    get_position,
    get_color,
    get_visibility,
    get_opacity,
    get_rotation,
    get_scale,
    location,
}
#endif
#endif


// TODO: analytic projection
fn get_bounding_box_cov2d(
    cov2d: vec3<f32>,
    direction: vec2<f32>,
    cutoff: f32,
) -> vec4<f32> {
    let fitler_size = 0.707106;

    let extent = sqrt(max(
        vec2<f32>(1.e-4, 1.e-4),
        vec2<f32>(cov2d.x, cov2d.z),
    ));
    let radius = ceil(max(max(extent.x, extent.y), cutoff * fitler_size));

    // TODO: verify OBB capability
    let radius_ndc = vec2<f32>(
        vec2<f32>(radius) / view.viewport.zw,
    );

    return vec4<f32>(
        radius_ndc * direction,
        radius * direction,
    );
}


fn compute_cov2d_surfel(
    gaussian_position: vec3<f32>,
    index: u32,
    cutoff: f32,
) -> vec3<f32> {
    let rotation = get_rotation(index);
    let scale = get_scale(index);

    let T_r = mat3x3<f32>(
        gaussian_uniforms.transform[0].xyz,
        gaussian_uniforms.transform[1].xyz,
        gaussian_uniforms.transform[2].xyz,
    );

    let S = get_scale_matrix(scale);
    let R = get_rotation_matrix(rotation);

    let L = R * S;// * transpose(T_r);

    let world_from_local = mat3x4<f32>(
        vec4<f32>(L.x, 0.0),
        vec4<f32>(L.y, 0.0),
        vec4<f32>(gaussian_position, 1.0),
    );

    let ndc_from_world = transpose(view.clip_from_world);
    let pixels_from_ndc = intrinsic_matrix();

    let T = transpose(world_from_local) * ndc_from_world * pixels_from_ndc;

    let test = vec3<f32>(cutoff * cutoff, cutoff * cutoff, -1.0);
    let d = dot(test * T[2], T[2]);
    if abs(d) < 1.0e-6 {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let f = (1.0 / d) * test;
    let means2d = vec2<f32>(
        dot(f * T[0], T[2]),
        dot(f * T[1], T[2]),
    );

    let t = vec2<f32>(
        dot(f * T[0], T[0]),
        dot(f * T[1], T[1]),
    );
    let extent = means2d * means2d - t;
    let covariance = means2d.x * means2d.y - dot(f * T[0], T[1]);

    return vec3<f32>(extent.x, covariance, extent.y);
}
