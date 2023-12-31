#define_import_path bevy_gaussian_splatting::texture

#import bevy_gaussian_splatting::bindings::{
    gaussian_uniforms,
    position_visibility,
    spherical_harmonics,
    rotation,
    rotation_scale_opacity,
    scale_opacity,
};


fn location(index: u32) -> vec2<i32> {
    return vec2<i32>(
        i32(index) % i32(gaussian_uniforms.count_root_ceil),
        i32(index) / i32(gaussian_uniforms.count_root_ceil),
    );
}


#ifdef PLANAR_TEXTURE_F16

fn get_position(index: u32) -> vec3<f32> {
    let sample = textureLoad(
        position_visibility,
        location(index),
    );

    return sample.xyz;
}

fn get_spherical_harmonics(index: u32) -> array<f32, #{SH_COEFF_COUNT}> {
    var coefficients: array<f32, #{SH_COEFF_COUNT}>;

    for (var i = 0u; i < #{SH_VEC4_PLANES}u; i = i + 1u) {
        let sample = textureLoad(
            spherical_harmonics,
            location(index),
            i,
        );

        let v0 = unpack2x16float(sample.x);
        let v1 = unpack2x16float(sample.y);
        let v2 = unpack2x16float(sample.z);
        let v3 = unpack2x16float(sample.w);

        let base_index = i * 8u;
        coefficients[base_index     ] = v0.x;
        coefficients[base_index + 1u] = v0.y;

        coefficients[base_index + 2u] = v1.x;
        coefficients[base_index + 3u] = v1.y;

        coefficients[base_index + 4u] = v2.x;
        coefficients[base_index + 5u] = v2.y;

        coefficients[base_index + 6u] = v3.x;
        coefficients[base_index + 7u] = v3.y;
    }

    return coefficients;
}

fn get_rotation(index: u32) -> vec4<f32> {
    let sample = textureLoad(
        rotation_scale_opacity,
        location(index),
    );

    let q0 = unpack2x16float(sample.x);
    let q1 = unpack2x16float(sample.y);

    return vec4<f32>(
        q0.yx,
        q1.yx,
    );
}

fn get_scale(index: u32) -> vec3<f32> {
    let sample = textureLoad(
        rotation_scale_opacity,
        location(index),
    );

    let s0 = unpack2x16float(sample.z);
    let s1 = unpack2x16float(sample.w);

    return vec3<f32>(
        s0.yx,
        s1.y,
    );
}

fn get_opacity(index: u32) -> f32 {
    let sample = textureLoad(
        rotation_scale_opacity,
        location(index),
    );

    return unpack2x16float(sample.w).x;
}

fn get_visibility(index: u32) -> f32 {
    let sample = textureLoad(
        position_visibility,
        location(index),
    );

    return sample.w;
}
#endif


#ifdef PLANAR_TEXTURE_F32
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
