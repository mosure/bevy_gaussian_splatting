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
#ifdef LOD_MORPH
    #import bevy_gaussian_splatting::lod_morph::{
        lod_morph_covariance,
    }
#endif

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

fn compute_local_cov3d(scale: vec3<f32>, rotation: vec4<f32>) -> array<f32, 6> {
    let S = get_scale_matrix(scale);
    let R = get_rotation_matrix(rotation);

    let M = S * R;
    let Sigma = transpose(M) * M;

    return array<f32, 6>(
        Sigma[0][0],
        Sigma[0][1],
        Sigma[0][2],
        Sigma[1][1],
        Sigma[1][2],
        Sigma[2][2],
    );
}

// Runtime scale/rotation covariance already contains `global_scale^2` because
// `compute_local_cov3d` calls `get_scale_matrix`. The precomputed plane is the
// one representation authored before that dynamic uniform is applied.
fn covariance_storage_scale_squared() -> f32 {
    #ifdef PRECOMPUTE_COVARIANCE_3D
        return gaussian_uniforms.global_scale * gaussian_uniforms.global_scale;
    #else
        return 1.0;
    #endif
}

fn transform_local_cov3d(covariance: array<f32, 6>) -> array<f32, 6> {
    let T = mat3x3<f32>(
        gaussian_uniforms.transform[0].xyz,
        gaussian_uniforms.transform[1].xyz,
        gaussian_uniforms.transform[2].xyz,
    );
    let storage_scale_squared = covariance_storage_scale_squared();
    let Sigma = mat3x3<f32>(
        vec3<f32>(covariance[0], covariance[1], covariance[2]) * storage_scale_squared,
        vec3<f32>(covariance[1], covariance[3], covariance[4]) * storage_scale_squared,
        vec3<f32>(covariance[2], covariance[4], covariance[5]) * storage_scale_squared,
    );
    let transformed = T * Sigma * transpose(T);

    return array<f32, 6>(
        transformed[0][0],
        transformed[0][1],
        transformed[0][2],
        transformed[1][1],
        transformed[1][2],
        transformed[2][2],
    );
}

struct GaussianMipCovariance2d {
    filtered: vec4<f32>,
    parent_opacity_scale: f32,
    child_opacity_scale: f32,
    parent_projected_area_ratio: f32,
    child_projected_area_ratio: f32,
}

const LOD_MORPH_PROJECTED_DETERMINANT_FLOOR: f32 = 0.00000000000000000001;

fn projected_area_ratio(endpoint: vec3<f32>, current: vec3<f32>) -> f32 {
    let endpoint_determinant = endpoint.x * endpoint.z - endpoint.y * endpoint.y;
    let current_determinant = current.x * current.z - current.y * current.y;
    if !(endpoint_determinant > LOD_MORPH_PROJECTED_DETERMINANT_FLOOR)
        || !(current_determinant > LOD_MORPH_PROJECTED_DETERMINANT_FLOOR)
    {
        // The mip filter ordinarily makes both matrices positive definite.
        // Preserve a finite conservative blend if malformed source data reaches
        // this point instead of amplifying it through a zero-area division.
        return 1.0;
    }
    return sqrt(endpoint_determinant / current_determinant);
}

fn compute_cov2d_3dgs(
    position: vec3<f32>,
    parent_position: vec3<f32>,
    child_position: vec3<f32>,
    index: u32,
    morph_parent_index: u32,
    morph_blend_t: f32,
    morph_active: bool,
) -> GaussianMipCovariance2d {
#ifdef PRECOMPUTE_COVARIANCE_3D
    let child_local_cov3d = get_cov3d(index);
    var parent_local_cov3d = child_local_cov3d;
#else
    let child_rotation = get_rotation(index);
    let child_scale = get_scale(index);
    let child_local_cov3d = compute_local_cov3d(child_scale, child_rotation);
    var parent_local_cov3d = child_local_cov3d;
#endif

    var local_cov3d = child_local_cov3d;
    #ifdef LOD_MORPH
        if morph_active {
            #ifdef PRECOMPUTE_COVARIANCE_3D
                parent_local_cov3d = get_cov3d(morph_parent_index);
            #else
                parent_local_cov3d = compute_local_cov3d(
                get_scale(morph_parent_index),
                get_rotation(morph_parent_index),
            );
            #endif
            local_cov3d = lod_morph_covariance(
                parent_local_cov3d,
                child_local_cov3d,
                morph_blend_t,
            );
        }
    #endif

    let filtered = cov2d(position, transform_local_cov3d(local_cov3d));
    var parent_opacity_scale = filtered.w;
    var child_opacity_scale = filtered.w;
    var parent_projected_area_ratio = 1.0;
    var child_projected_area_ratio = 1.0;
    #ifdef LOD_MORPH
        if morph_active {
            let parent_filtered = cov2d(
                parent_position,
                transform_local_cov3d(parent_local_cov3d),
            );
            let child_filtered = cov2d(
                child_position,
                transform_local_cov3d(child_local_cov3d),
            );
            parent_opacity_scale = parent_filtered.w;
            child_opacity_scale = child_filtered.w;
            parent_projected_area_ratio = projected_area_ratio(
                parent_filtered.xyz,
                filtered.xyz,
            );
            child_projected_area_ratio = projected_area_ratio(
                child_filtered.xyz,
                filtered.xyz,
            );
        }
    #endif

    return GaussianMipCovariance2d(
        filtered,
        parent_opacity_scale,
        child_opacity_scale,
        parent_projected_area_ratio,
        child_projected_area_ratio,
    );
}

#endif  // GAUSSIAN_3D
