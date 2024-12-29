#define_import_path bevy_gaussian_splatting::planar

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


#if defined(GAUSSIAN_2D) || defined(GAUSSIAN_3D)
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

    #endif

    // TODO: PLANAR_F16 for GAUSSIAN_4D
#endif
