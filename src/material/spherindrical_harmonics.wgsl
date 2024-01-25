#define_import_path bevy_gaussian_splatting::spherindrical_harmonics
#import bevy_gaussian_splatting::bindings::globals
#import bevy_gaussian_splatting::spherical_harmonics::{
    shc,
    spherical_harmonics_lookup,
}


fn spherindrical_harmonics_lookup(
    ray_direction: vec3<f32>,
    sh: array<f32, #{SH_COEFF_COUNT}>,
) -> vec3<f32> {
    let rds = ray_direction * ray_direction;
    let dir_t = globals.time;

    var color = vec3<f32>(0.5);

    color += shc[ 0] * vec3<f32>(sh[0], sh[1], sh[2]);

#if SH_DEG > 0
    color += shc[ 1] * vec3<f32>(sh[ 3], sh[ 4], sh[ 5]) * ray_direction.y;
    color += shc[ 2] * vec3<f32>(sh[ 6], sh[ 7], sh[ 8]) * ray_direction.z;
    color += shc[ 3] * vec3<f32>(sh[ 9], sh[10], sh[11]) * ray_direction.x;
#endif

#if SH_DEG > 1
    color += shc[ 4] * vec3<f32>(sh[12], sh[13], sh[14]) * ray_direction.x * ray_direction.y;
    color += shc[ 5] * vec3<f32>(sh[15], sh[16], sh[17]) * ray_direction.y * ray_direction.z;
    color += shc[ 6] * vec3<f32>(sh[18], sh[19], sh[20]) * (2.0 * rds.z - rds.x - rds.y);
    color += shc[ 7] * vec3<f32>(sh[21], sh[22], sh[23]) * ray_direction.x * ray_direction.z;
    color += shc[ 8] * vec3<f32>(sh[24], sh[25], sh[26]) * (rds.x - rds.y);
#endif

#if SH_DEG > 2
    color += shc[ 9] * vec3<f32>(sh[27], sh[28], sh[29]) * ray_direction.y * (3.0 * rds.x - rds.y);
    color += shc[10] * vec3<f32>(sh[30], sh[31], sh[32]) * ray_direction.x * ray_direction.y * ray_direction.z;
    color += shc[11] * vec3<f32>(sh[33], sh[34], sh[35]) * ray_direction.y * (4.0 * rds.z - rds.x - rds.y);
    color += shc[12] * vec3<f32>(sh[36], sh[37], sh[38]) * ray_direction.z * (2.0 * rds.z - 3.0 * rds.x - 3.0 * rds.y);
    color += shc[13] * vec3<f32>(sh[39], sh[40], sh[41]) * ray_direction.x * (4.0 * rds.z - rds.x - rds.y);
    color += shc[14] * vec3<f32>(sh[42], sh[43], sh[44]) * ray_direction.z * (rds.x - rds.y);
    color += shc[15] * vec3<f32>(sh[45], sh[46], sh[47]) * ray_direction.x * (rds.x - 3.0 * rds.y);
#endif

// TODO: add SH_DEG and SH_DEG_T shader defines
#if SH_DEG_T > 0

#endif

#if SH_DEG_T > 1

#endif

    return color;
}
