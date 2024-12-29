#define_import_path bevy_gaussian_splatting::gaussian_3d


#ifdef GAUSSIAN_3D
#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
}
#import bevy_gaussian_splatting::helpers::{
    cov2d,
    get_rotation_matrix,
    get_scale_matrix,
}


#ifdef PACKED
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::packed::{
            get_cov3d,
        }
    #else
        #import bevy_gaussian_splatting::packed::{
            get_rotation,
            get_scale,
        }
    #endif
#else ifdef BUFFER_STORAGE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::planar::{
            get_cov3d,
        }
    #else
        #import bevy_gaussian_splatting::planar::{
            get_rotation,
            get_scale,
        }
    #endif
#else ifdef BUFFER_TEXTURE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::texture::{
            get_cov3d,
        }
    #else
        #import bevy_gaussian_splatting::texture::{
            get_rotation,
            get_scale,
        }
    #endif
#endif


fn compute_cov3d(scale: vec3<f32>, rotation: vec4<f32>) -> array<f32, 6> {
    let S = get_scale_matrix(scale);

    let T = mat3x3<f32>(
        gaussian_uniforms.transform[0].xyz,
        gaussian_uniforms.transform[1].xyz,
        gaussian_uniforms.transform[2].xyz,
    );

    let R = get_rotation_matrix(rotation);

    let M = S * R;
    let Sigma = transpose(M) * M;
    let TS = T * Sigma * transpose(T);

    return array<f32, 6>(
        TS[0][0],
        TS[0][1],
        TS[0][2],
        TS[1][1],
        TS[1][2],
        TS[2][2],
    );
}

fn compute_cov2d_3dgs(
    position: vec3<f32>,
    index: u32,
) -> vec3<f32> {
#ifdef PRECOMPUTE_COVARIANCE_3D
    let cov3d = get_cov3d(index);
#else
    let rotation = get_rotation(index);
    let scale = get_scale(index);

    let cov3d = compute_cov3d(scale, rotation);
#endif

    return cov2d(position, cov3d);
}


fn get_bounding_box_clip(
    cov2d: vec3<f32>,
    direction: vec2<f32>,
    cutoff: f32,
) -> vec4<f32> {
    // return vec4<f32>(offset, uv);

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let trace = cov2d.x + cov2d.z;
    let mid = 0.5 * trace;
    let discriminant = max(0.0, mid * mid - det);

    let term = sqrt(discriminant);

    let lambda1 = mid + term;
    let lambda2 = max(mid - term, 0.0);

    let x_axis_length = sqrt(lambda1);
    let y_axis_length = sqrt(lambda2);


#ifdef USE_AABB
    let radius_px = cutoff * max(x_axis_length, y_axis_length);
    let radius_ndc = vec2<f32>(
        radius_px / view.viewport.zw,
    );

    return vec4<f32>(
        radius_ndc * direction,
        radius_px * direction,
    );
#endif

#ifdef USE_OBB

    let a = (cov2d.x - cov2d.z) * (cov2d.x - cov2d.z);
    let b = sqrt(a + 4.0 * cov2d.y * cov2d.y);
    let major_radius = sqrt((cov2d.x + cov2d.z + b) * 0.5);
    let minor_radius = sqrt((cov2d.x + cov2d.z - b) * 0.5);

    let bounds = cutoff * vec2<f32>(
        major_radius,
        minor_radius,
    );

    let eigvec1 = normalize(vec2<f32>(
        -cov2d.y,
        lambda1 - cov2d.x,
    ));
    let eigvec2 = vec2<f32>(
        eigvec1.y,
        -eigvec1.x
    );

    let rotation_matrix = transpose(
        mat2x2(
            eigvec1,
            eigvec2,
        )
    );

    let scaled_vertex = direction * bounds;
    let rotated_vertex = scaled_vertex * rotation_matrix;

    let scaling_factor = 1.0 / view.viewport.zw;
    let ndc_vertex = rotated_vertex * scaling_factor;

    return vec4<f32>(
        ndc_vertex,
        rotated_vertex,
    );
#endif
}


#endif  // GAUSSIAN_3D
