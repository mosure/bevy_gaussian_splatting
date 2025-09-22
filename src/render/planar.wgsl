#define_import_path bevy_gaussian_splatting::planar

#ifdef GAUSSIAN_3D_STRUCTURE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::bindings::{
            position_visibility,
            spherical_harmonics,
            covariance_3d_opacity,
        }
    #else
        #import bevy_gaussian_splatting::bindings::{
            position_visibility,
            spherical_harmonics,
            rotation,
            rotation_scale_opacity,
            scale_opacity,
        }
    #endif

    #import bevy_gaussian_splatting::spherical_harmonics::{
        spherical_harmonics_lookup,
        srgb_to_linear,
    }
#else ifdef GAUSSIAN_4D
    #import bevy_gaussian_splatting::bindings::{
        position_visibility,
        spherindrical_harmonics,
        isotropic_rotations,
        scale_opacity,
        timestamp_timescale,
    }

    #import bevy_gaussian_splatting::spherical_harmonics::srgb_to_linear
    #import bevy_gaussian_splatting::spherindrical_harmonics::spherindrical_harmonics_lookup
#endif

#ifdef GAUSSIAN_3D_STRUCTURE
    #ifdef PLANAR_F16
        fn get_color(
            index: u32,
            ray_direction: vec3<f32>,
        ) -> vec3<f32> {
            let sh = get_spherical_harmonics(index);
            let color = spherical_harmonics_lookup(ray_direction, sh);
            return srgb_to_linear(color);
        }

        fn get_position(index: u32) -> vec3<f32> {
            return position_visibility[index].xyz;
        }

        fn get_spherical_harmonics(index: u32) -> array<f32, #{SH_COEFF_COUNT}> {
            var coefficients: array<f32, #{SH_COEFF_COUNT}>;

            for (var i = 0u; i < #{HALF_SH_COEFF_COUNT}u; i = i + 1u) {
                let values = unpack2x16float(spherical_harmonics[index][i]);

                coefficients[i * 2u] = values[0];
                coefficients[i * 2u + 1u] = values[1];
            }

            return coefficients;
        }

        #ifdef PRECOMPUTE_COVARIANCE_3D
            fn get_cov3d(index: u32) -> array<f32, 6> {
                let c0 = unpack2x16float(covariance_3d_opacity[index].x);
                let c1 = unpack2x16float(covariance_3d_opacity[index].y);
                let c2 = unpack2x16float(covariance_3d_opacity[index].z);

                var cov3d: array<f32, 6>;

                cov3d[0] = c0.y;
                cov3d[1] = c0.x;
                cov3d[2] = c1.y;
                cov3d[3] = c1.x;
                cov3d[4] = c2.y;
                cov3d[5] = c2.x;

                return cov3d;
            }
        #else
            fn get_rotation(index: u32) -> vec4<f32> {
                let q0 = unpack2x16float(rotation_scale_opacity[index].x);
                let q1 = unpack2x16float(rotation_scale_opacity[index].y);

                return vec4<f32>(
                    q0.yx,
                    q1.yx,
                );
            }

            fn get_scale(index: u32) -> vec3<f32> {
                let s0 = unpack2x16float(rotation_scale_opacity[index].z);
                let s1 = unpack2x16float(rotation_scale_opacity[index].w);

                return vec3<f32>(
                    s0.yx,
                    s1.y,
                );
            }
        #endif

        fn get_opacity(index: u32) -> f32 {
            #ifdef PRECOMPUTE_COVARIANCE_3D
                return unpack2x16float(covariance_3d_opacity[index].w).y;
            #else
                return unpack2x16float(rotation_scale_opacity[index].w).x;
            #endif
        }

        fn get_visibility(index: u32) -> f32 {
            return position_visibility[index].w;
        }
    #else ifdef PLANAR_F32
        fn get_color(
            index: u32,
            ray_direction: vec3<f32>,
        ) -> vec3<f32> {
            let sh = get_spherical_harmonics(index);
            let color = spherical_harmonics_lookup(ray_direction, sh);
            return srgb_to_linear(color);
        }

        fn get_position(index: u32) -> vec3<f32> {
            return position_visibility[index].xyz;
        }

        fn get_spherical_harmonics(index: u32) -> array<f32, #{SH_COEFF_COUNT}> {
            return spherical_harmonics[index];
        }

        fn get_rotation(index: u32) -> vec4<f32> {
            return rotation[index];
        }

        fn get_scale(index: u32) -> vec3<f32> {
            return scale_opacity[index].xyz;
        }

        fn get_opacity(index: u32) -> f32 {
            return scale_opacity[index].w;
        }

        fn get_visibility(index: u32) -> f32 {
            return position_visibility[index].w;
        }
    #endif
#else ifdef GAUSSIAN_4D
    #ifdef PLANAR_F32
        fn get_color(
            index: u32,
            dir_t: f32,
            ray_direction: vec3<f32>,
        ) -> vec3<f32> {
            let sh = get_spherindrical_harmonics(index);
            let color = spherindrical_harmonics_lookup(ray_direction, dir_t, sh);
            return srgb_to_linear(color);
        }

        fn get_isotropic_rotations(index: u32) -> mat2x4<f32> {
            let r1x = isotropic_rotations[index][0];
            let r1y = isotropic_rotations[index][1];
            let r1z = isotropic_rotations[index][2];
            let r1w = isotropic_rotations[index][3];

            let r2x = isotropic_rotations[index][4];
            let r2y = isotropic_rotations[index][5];
            let r2z = isotropic_rotations[index][6];
            let r2w = isotropic_rotations[index][7];

            return mat2x4<f32>(
                r1x, r1y, r1z, r1w,
                r2x, r2y, r2z, r2w,
            );
        }

        fn get_scale(index: u32) -> vec3<f32> {
            return scale_opacity[index].xyz;
        }

        fn get_opacity(index: u32) -> f32 {
            return scale_opacity[index].w;
        }

        fn get_position(index: u32) -> vec3<f32> {
            return position_visibility[index].xyz;
        }

        fn get_visibility(index: u32) -> f32 {
            return position_visibility[index].w;
        }

        fn get_spherindrical_harmonics(index: u32) -> array<f32, #{SH_4D_COEFF_COUNT}> {
            return spherindrical_harmonics[index];
        }

        fn get_timestamp(index: u32) -> f32 {
            return timestamp_timescale[index].x;
        }

        fn get_time_scale(index: u32) -> f32 {
            return timestamp_timescale[index].y;
        }
    #endif

    // TODO: PLANAR_F16 for GAUSSIAN_4D
#endif
