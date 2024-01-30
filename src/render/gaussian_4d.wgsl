#define_import_path bevy_gaussian_splatting::gaussian_4d

#import bevy_gaussian_splatting::bindings::{
    view,
    globals,
    gaussian_uniforms,
}


struct DecomposedGaussian4d {
    cov3d: array<f32, 6>,
    delta_mean: vec3<f32>,
    opacity_modifier: f32,
    mask: bool,
}


fn compute_cov3d_conditional(
    position: vec3<f32>,
    scale: vec4<f32>,
    rotation: vec4<f32>,
    rotation_r: vec4<f32>,
) -> DecomposedGaussian4d {
    let dt = globals.delta_time;

    let S = mat4x4<f32>(
        gaussian_uniforms.global_scale * scale.x, 0.0, 0.0, 0.0,
        0.0, gaussian_uniforms.global_scale * scale.y, 0.0, 0.0,
        0.0, 0.0, gaussian_uniforms.global_scale * scale.z, 0.0,
        0.0, 0.0, 0.0, gaussian_uniforms.global_scale * scale.w,  // TODO: separate spatial and time scale uniforms
    );

    let a = rotation.x;
    let b = rotation.y;
    let c = rotation.z;
    let d = rotation.w;

    let p = rotation_r.x;
    let q = rotation_r.y;
    let r = rotation_r.z;
    let s = rotation_r.w;

    let M_l = mat4x4<f32>(
        a, -b, -c, -d,
        b, a, -d, c,
        c, d, a, -b,
        d, -c, b, a,
    );

    let M_r = mat4x4<f32>(
        p, q, r, s,
        -q, p, -s, r,
        -r, s, p, -q,
        -s, -r, q, p,
    );

    let R = M_r * M_l;
    let M = S * R;
    let Sigma = transpose(M) * M;

    let cov_t = Sigma[3][3];
    let marginal_t = exp(-0.5 * dt * dt / cov_t);

    let mask = marginal_t > 0.05;
    if (!mask) {
        return DecomposedGaussian4d(
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            vec3<f32>(0.0, 0.0, 0.0),
            0.0,
            false,
        );
    }

    let opacity_modifier = marginal_t;

    let cov11 = mat3x3<f32>(
        Sigma[0][0], Sigma[0][1], Sigma[0][2],
        Sigma[1][0], Sigma[1][1], Sigma[1][2],
        Sigma[2][0], Sigma[2][1], Sigma[2][2],
    );
    let cov12 = vec3<f32>(Sigma[0][3], Sigma[1][3], Sigma[2][3]);
    let cov3d_condition = cov11 - outerProduct(cov12, cov12) / cov_t;

    let delta_mean = cov12 / cov_t * dt;

    return DecomposedGaussian4d(
        cov3d_condition[0][0],
        cov3d_condition[0][1],
        cov3d_condition[0][2],
        cov3d_condition[1][1],
        cov3d_condition[1][2],
        cov3d_condition[2][2],
        delta_mean,
        opacity_modifier,
        mask,
    );
}
