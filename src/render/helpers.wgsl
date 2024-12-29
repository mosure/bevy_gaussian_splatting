#define_import_path bevy_gaussian_splatting::helpers

#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
}


fn cov2d(
    position: vec3<f32>,
    cov3d: array<f32, 6>,
) -> vec3<f32> {
    let Vrk = mat3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    );

    var t = view.view_from_world * vec4<f32>(position, 1.0);

    let focal = vec2<f32>(
        view.clip_from_view.x.x * view.viewport.z,
        view.clip_from_view.y.y * view.viewport.w,
    );

    let s = 1.0 / (t.z * t.z);
    let J = mat3x3(
        focal.x / t.z, 0.0, -(focal.x * t.x) * s,
        0.0, -focal.y / t.z, (focal.y * t.y) * s,
        0.0, 0.0, 0.0,
    );

    let W = transpose(
        mat3x3<f32>(
            view.view_from_world.x.xyz,
            view.view_from_world.y.xyz,
            view.view_from_world.z.xyz,
        )
    );

    let T = W * J;

    var cov = transpose(T) * transpose(Vrk) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
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
