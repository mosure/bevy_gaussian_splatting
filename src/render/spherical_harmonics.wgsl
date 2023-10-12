#define_import_path bevy_gaussian_splatting::spherical_harmonics


const shc = array<f32, 16>(
    0.28209479177387814,
    -0.4886025119029199,
    0.4886025119029199,
    -0.4886025119029199,
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
);

fn spherical_harmonics_lookup(
    ray_direction: vec3<f32>,
    sh: array<vec3<f32>, #{MAX_SH_COEFF_COUNT}>,
) -> vec3<f32> {
    var rds = ray_direction * ray_direction;
    var color = vec3<f32>(0.5);

    color += shc[ 0] * sh[ 0];

    color += shc[ 1] * sh[ 1] * ray_direction.y;
    color += shc[ 2] * sh[ 2] * ray_direction.z;
    color += shc[ 3] * sh[ 3] * ray_direction.x;

    color += shc[ 4] * sh[4] * ray_direction.x * ray_direction.y;
    color += shc[ 5] * sh[5] * ray_direction.y * ray_direction.z;
    color += shc[ 6] * sh[6] * (2.0 * rds.z - rds.x - rds.y);
    color += shc[ 7] * sh[7] * ray_direction.x * ray_direction.z;
    color += shc[ 8] * sh[8] * (rds.x - rds.y);

    color += shc[ 9] * sh[9] * ray_direction.y * (3.0 * rds.x - rds.y);
    color += shc[10] * sh[10] * ray_direction.x * ray_direction.y * ray_direction.z;
    color += shc[11] * sh[11] * ray_direction.y * (4.0 * rds.z - rds.x - rds.y);
    color += shc[12] * sh[12] * ray_direction.z * (2.0 * rds.z - 3.0 * rds.x - 3.0 * rds.y);
    color += shc[13] * sh[13] * ray_direction.x * (4.0 * rds.z - rds.x - rds.y);
    color += shc[14] * sh[14] * ray_direction.z * (rds.x - rds.y);
    color += shc[15] * sh[15] * ray_direction.x * (rds.x - 3.0 * rds.y);

    return color;
}
