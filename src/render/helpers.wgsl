#define_import_path bevy_gaussian_splatting::helpers

#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
}


fn intrinsic_matrix() -> mat3x4<f32> {
    let focal = vec2<f32>(
        view.clip_from_view.x.x * view.viewport.z / 2.0,
        view.clip_from_view.y.y * view.viewport.w / 2.0,
    );

    let Ks = mat3x4<f32>(
        vec4<f32>(focal.x, 0.0, 0.0, (view.viewport.z - 1.0) / 2.0),
        vec4<f32>(0.0, focal.y, 0.0, (view.viewport.w - 1.0) / 2.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );

    return Ks;
}


fn get_rotation_matrix(
    rotation: vec4<f32>,
) -> mat3x3<f32> {
    let r = rotation.x;
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;

    return mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - r * z),
        2.0 * (x * z + r * y),

        2.0 * (x * y + r * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - r * x),

        2.0 * (x * z - r * y),
        2.0 * (y * z + r * x),
        1.0 - 2.0 * (x * x + y * y),
    );
}


fn get_scale_matrix(
    scale: vec3<f32>,
) -> mat3x3<f32> {
    return mat3x3<f32>(
        scale.x * gaussian_uniforms.global_scale, 0.0, 0.0,
        0.0, scale.y * gaussian_uniforms.global_scale, 0.0,
        0.0, 0.0, scale.z * gaussian_uniforms.global_scale,
    );
}
