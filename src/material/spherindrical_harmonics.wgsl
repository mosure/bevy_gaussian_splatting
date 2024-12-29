#define_import_path bevy_gaussian_splatting::spherindrical_harmonics

#import bevy_gaussian_splatting::bindings::gaussian_uniforms
#import bevy_gaussian_splatting::spherical_harmonics::{
    shc,
    spherical_harmonics_lookup,
}


const PI = radians(180.0);


fn spherindrical_harmonics_lookup(
    ray_direction: vec3<f32>,
    dir_t: f32,
    sh: array<f32, #{SH_COEFF_COUNT}>,
) -> vec3<f32> {
    let rds = ray_direction * ray_direction;

    var color = vec3<f32>(0.5);

    color += shc[ 0] * vec3<f32>(sh[0], sh[1], sh[2]);

#if SH_DEGREE > 0
    color += shc[ 1] * vec3<f32>(sh[ 3], sh[ 4], sh[ 5]) * ray_direction.y;
    color += shc[ 2] * vec3<f32>(sh[ 6], sh[ 7], sh[ 8]) * ray_direction.z;
    color += shc[ 3] * vec3<f32>(sh[ 9], sh[10], sh[11]) * ray_direction.x;
#endif

#if SH_DEGREE > 1
    let x = ray_direction.x;
    let y = ray_direction.y;
    let z = ray_direction.z;

    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;

    let l2m2 = shc[4] * xy;
    let l2m1 = shc[5] * yz;
    let l2m0 = shc[6] * (2.0 * zz - xx - yy);
    let l2p1 = shc[7] * xz;
    let l2p2 = shc[8] * (xx - yy);

    color += l2m2 * vec3<f32>(sh[12], sh[13], sh[14]);
    color += l2m1 * vec3<f32>(sh[15], sh[16], sh[17]);
    color += l2m0 * vec3<f32>(sh[18], sh[19], sh[20]);
    color += l2p1 * vec3<f32>(sh[21], sh[22], sh[23]);
    color += l2p2 * vec3<f32>(sh[24], sh[25], sh[26]);
#endif

#if SH_DEGREE > 2
    let l3m3 = shc[9] * y * (3.0 * xx - yy);
    let l3m2 = shc[10] * z * xy;
    let l3m1 = shc[11] * y * (4.0 * zz - xx - yy);
    let l3m0 = shc[12] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    let l3p1 = shc[13] * x * (4.0 * zz - xx - yy);
    let l3p2 = shc[14] * z * (xx - yy);
    let l3p3 = shc[15] * x * (xx - 3.0 * yy);

    color += l3m3 * vec3<f32>(sh[27], sh[28], sh[29]);
    color += l3m2 * vec3<f32>(sh[30], sh[31], sh[32]);
    color += l3m1 * vec3<f32>(sh[33], sh[34], sh[35]);
    color += l3m0 * vec3<f32>(sh[36], sh[37], sh[38]);
    color += l3p1 * vec3<f32>(sh[39], sh[40], sh[41]);
    color += l3p2 * vec3<f32>(sh[42], sh[43], sh[44]);
    color += l3p3 * vec3<f32>(sh[45], sh[46], sh[47]);
#endif

#if SH_DEGREE_TIME > 0
    let duration = gaussian_uniforms.time_end - gaussian_uniforms.time_start;
    let theta = dir_t / duration;

    let t1 = cos(2.0 * PI * theta);

    color += t1 * (
        l0m0 * sh[16] +
        l1m1 * sh[17] +
        l1p1 * sh[18] +
        l2m2 * sh[20] +
        l2m1 * sh[21] +
        l2m0 * sh[22] +
        l2p1 * sh[23] +
        l2p2 * sh[24] +
        l3m3 * sh[25] +
        l3m2 * sh[26] +
        l3m1 * sh[27] +
        l3m0 * sh[28] +
        l3p1 * sh[29] +
        l3p2 * sh[30] +
        l3p3 * sh[31]
    );

    #if SH_DEGREE_TIME > 1
        let t2 = cos(4.0 * PI * theta);

        color += t1 * (
            l0m0 * shc[16] +
            l1m1 * shc[17] +
            l1p1 * shc[18] +
            l2m2 * shc[20] +
            l2m1 * shc[21] +
            l2m0 * shc[22] +
            l2p1 * shc[23] +
            l2p2 * shc[24] +
            l3m3 * shc[25] +
            l3m2 * shc[26] +
            l3m1 * shc[27] +
            l3m0 * shc[28] +
            l3p1 * shc[29] +
            l3p2 * shc[30] +
            l3p3 * shc[31]
        );
    #endif
#endif

    return color;
}
