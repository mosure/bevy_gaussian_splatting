#define_import_path bevy_gaussian_splatting::gaussian_4d

#import bevy_gaussian_splatting::bindings::{
    view,
    globals,
    gaussian_uniforms,
}

#ifdef BUFFER_STORAGE
    #import bevy_gaussian_splatting::planar::{
        get_isotropic_rotations,
        get_scale,
        get_timestamp,
        get_time_scale,
    }
#endif


struct DecomposedGaussian4d {
    cov3d: array<f32, 6>,
    delta_mean: vec3<f32>,
    opacity_modifier: f32,
    dir_t: f32,
    mask: bool,
}


fn outer_product(
    a: vec3<f32>,
    b: vec3<f32>,
) -> mat3x3<f32> {
    return mat3x3<f32>(
        a.x * b.x, a.x * b.y, a.x * b.z,
        a.y * b.x, a.y * b.y, a.y * b.z,
        a.z * b.x, a.z * b.y, a.z * b.z,
    );
}

fn conditional_cov3d(
    position: vec3<f32>,
    index: u32,
) -> DecomposedGaussian4d {
    let isotropic_rotations = get_isotropic_rotations(index);
    let rotation = normalize(isotropic_rotations[0]);
    let rotation_r = normalize(isotropic_rotations[1]);
    let scale = get_scale(index);

    let dt = gaussian_uniforms.time - get_timestamp(index);

    let S = mat4x4<f32>(
        gaussian_uniforms.global_scale * scale.x, 0.0, 0.0, 0.0,
        0.0, gaussian_uniforms.global_scale * scale.y, 0.0, 0.0,
        0.0, 0.0, gaussian_uniforms.global_scale * scale.z, 0.0,
        0.0, 0.0, 0.0, gaussian_uniforms.global_scale * get_time_scale(index),
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

    // let mask = marginal_t > 0.05;
    let mask = true;
    if !mask {
        return DecomposedGaussian4d(
            array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            vec3<f32>(0.0, 0.0, 0.0),
            0.0,
            dt,
            mask,
        );
    }

    let opacity_modifier = marginal_t;

    let cov11 = mat3x3<f32>(
        Sigma[0][0], Sigma[0][1], Sigma[0][2],
        Sigma[1][0], Sigma[1][1], Sigma[1][2],
        Sigma[2][0], Sigma[2][1], Sigma[2][2],
    );
    let cov12 = vec3<f32>(Sigma[0][3], Sigma[1][3], Sigma[2][3]);
    let cov_op = outer_product(cov12, cov12);
    let cov_op_t = mat3x3<f32>(
        cov_op.x.x / cov_t, cov_op.x.y / cov_t, cov_op.x.z / cov_t,
        cov_op.y.x / cov_t, cov_op.y.y / cov_t, cov_op.y.z / cov_t,
        cov_op.z.x / cov_t, cov_op.z.y / cov_t, cov_op.z.z / cov_t,
    );
    let cov3d_condition = cov11 - cov_op_t;

    let delta_mean = cov12 / cov_t * dt;

    return DecomposedGaussian4d(
        array<f32, 6>(
            cov3d_condition[0][0],
            cov3d_condition[0][1],
            cov3d_condition[0][2],
            cov3d_condition[1][1],
            cov3d_condition[1][2],
            cov3d_condition[2][2]
        ),
        delta_mean,
        opacity_modifier,
        dt,
        mask,
    );
}
