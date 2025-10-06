#define_import_path bevy_gaussian_splatting::pbr_decomposition::normal_estimation

#import bevy_gaussian_splatting::bindings::{
    position_visibility,
    rotation,
    scale_opacity,
}

struct NormalData {
    normal: vec3<f32>,
    confidence: f32,
}

struct NormalSettings {
    spatial_sigma: f32,
    color_sigma: f32,
    confidence_threshold: f32,
    _pad: f32,
}

// Gaussian inputs come from the engine's planar storage bind group (group 2).
// Outputs and settings use dedicated groups (3 and 4) for this pipeline.
@group(3) @binding(0) var<storage, read_write> normals: array<NormalData>;

@group(4) @binding(0) var<uniform> settings: NormalSettings;

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let qx = q.x;
    let qy = q.y;
    let qz = q.z;
    let qw = q.w;

    let x2 = qx * qx;
    let y2 = qy * qy;
    let z2 = qz * qz;
    let xy = qx * qy;
    let xz = qx * qz;
    let yz = qy * qz;
    let wx = qw * qx;
    let wy = qw * qy;
    let wz = qw * qz;

    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)),
        vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2))
    );
}

fn extract_normal_from_gaussian(rotation: vec4<f32>, scale: vec3<f32>) -> vec3<f32> {
    var min_scale_idx = 0u;
    var min_scale = scale.x;

    if (scale.y < min_scale) {
        min_scale = scale.y;
        min_scale_idx = 1u;
    }

    if (scale.z < min_scale) {
        min_scale = scale.z;
        min_scale_idx = 2u;
    }

    let R = quat_to_mat3(rotation);

    var normal: vec3<f32>;
    switch min_scale_idx {
        case 0u: { normal = vec3<f32>(R[0][0], R[1][0], R[2][0]); }
        case 1u: { normal = vec3<f32>(R[0][1], R[1][1], R[2][1]); }
        case 2u: { normal = vec3<f32>(R[0][2], R[1][2], R[2][2]); }
        default: { normal = vec3<f32>(0.0, 0.0, 1.0); }
    }

    return normalize(normal);
}

@compute @workgroup_size(256)
fn estimate_normals(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let gaussian_count = arrayLength(&position_visibility);
    if (idx >= gaussian_count) { return; }

    let q = rotation[idx];
    let s = scale_opacity[idx].xyz;

    let normal = extract_normal_from_gaussian(q, s);

    // Minimal confidence until neighbor smoothing is wired with spatial hash
    let confidence = 1.0;

    normals[idx] = NormalData(normal, confidence);
}
