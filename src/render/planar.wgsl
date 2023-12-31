#define_import_path bevy_gaussian_splatting::planar

#import bevy_gaussian_splatting::bindings::{
    position_visibility,
    spherical_harmonics,
    rotation,
    rotation_scale_opacity,
    scale_opacity,
};


#ifdef PLANAR_F16

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

fn get_opacity(index: u32) -> f32 {
    return unpack2x16float(rotation_scale_opacity[index].w).x;
}

fn get_visibility(index: u32) -> f32 {
    return position_visibility[index].w;
}
#endif


#ifdef PLANAR_F32
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
