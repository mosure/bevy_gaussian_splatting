#define_import_path bevy_gaussian_splatting::pbr_decomposition::synthetic_views

#import bevy_gaussian_splatting::bindings::{
    spherical_harmonics,
    rotation,
}
#import bevy_gaussian_splatting::spherical_harmonics::{
    spherical_harmonics_lookup,
    srgb_to_linear,
}

const PI: f32 = 3.14159265359;
const TOPK_RESIDUALS: u32 = 8u;

struct StreamingStats {
    mean_rgb: vec3<f32>,
    count: u32,

    M2_rgb: vec3<f32>,
    near_normal_count: u32,

    near_normal_mean: vec3<f32>,
    topk_count: u32,

    topk_directions: array<vec3<f32>, 8>,
    topk_intensities: array<f32, 8>,

    residual_direction_sum: vec3<f32>,
    residual_direction_M2: f32,

    _pad: array<f32, 3>,
}

struct SyntheticViewSettings {
    num_views: u32,
    near_normal_angle_cos: f32,
    sh_frame: u32,
    _pad: u32,
}

struct NormalData {
    normal: vec3<f32>,
    confidence: f32,
}

// Group 2: gaussian inputs via engine bindings (spherical_harmonics, rotation)
// Group 3: this pipeline IO: normals (read) + stats (read_write)
// Group 4: settings
@group(3) @binding(0) var<storage, read> normals: array<NormalData>;
@group(3) @binding(1) var<storage, read_write> stats: array<StreamingStats>;

@group(4) @binding(0) var<uniform> settings: SyntheticViewSettings;

const SH_FRAME_WORLD: u32 = 0u;
const SH_FRAME_LOCAL: u32 = 1u;

fn fibonacci_sphere(i: u32, n: u32) -> vec3<f32> {
    let phi = PI * (sqrt(5.0) - 1.0);
    let y = 1.0 - (f32(i) / f32(n - 1u)) * 2.0;
    let radius = sqrt(1.0 - y * y);
    let theta = phi * f32(i);

    return vec3<f32>(
        cos(theta) * radius,
        y,
        sin(theta) * radius
    );
}

fn quat_to_mat3_transpose(q: vec4<f32>) -> mat3x3<f32> {
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

    return transpose(mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)),
        vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2))
    ));
}

fn update_topk_residuals(
    stats_ptr: ptr<function, StreamingStats>,
    direction: vec3<f32>,
    intensity: f32
) {
    if ((*stats_ptr).topk_count < TOPK_RESIDUALS) {
        let idx = (*stats_ptr).topk_count;
        (*stats_ptr).topk_directions[idx] = direction;
        (*stats_ptr).topk_intensities[idx] = intensity;
        (*stats_ptr).topk_count++;
    } else {
        var min_idx = 0u;
        var min_val = (*stats_ptr).topk_intensities[0];
        for (var i = 1u; i < TOPK_RESIDUALS; i++) {
            if ((*stats_ptr).topk_intensities[i] < min_val) {
                min_val = (*stats_ptr).topk_intensities[i];
                min_idx = i;
            }
        }
        if (intensity > min_val) {
            (*stats_ptr).topk_directions[min_idx] = direction;
            (*stats_ptr).topk_intensities[min_idx] = intensity;
        }
    }
}

@compute @workgroup_size(256)
fn evaluate_synthetic_views(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let gaussian_count = arrayLength(&normals);
    if (idx >= gaussian_count) { return; }

    let sh = spherical_harmonics[idx];
    let normal_data = normals[idx];
    let normal = normal_data.normal;
    let rotation = rotation[idx];

    var local_stats: StreamingStats;
    local_stats.count = 0u;
    local_stats.mean_rgb = vec3<f32>(0.0);
    local_stats.M2_rgb = vec3<f32>(0.0);
    local_stats.near_normal_count = 0u;
    local_stats.near_normal_mean = vec3<f32>(0.0);
    local_stats.topk_count = 0u;
    local_stats.residual_direction_sum = vec3<f32>(0.0);
    local_stats.residual_direction_M2 = 0.0;

    for (var i = 0u; i < settings.num_views; i++) {
        let view_dir_world = fibonacci_sphere(i, settings.num_views);

        var view_dir_eval = view_dir_world;
        if (settings.sh_frame == SH_FRAME_LOCAL) {
            let R_inv = quat_to_mat3_transpose(rotation);
            view_dir_eval = R_inv * view_dir_world;
        }

        let color_srgb = spherical_harmonics_lookup(view_dir_eval, sh);
        let color = max(vec3<f32>(0.0), srgb_to_linear(color_srgb));

        local_stats.count += 1u;
        let delta = color - local_stats.mean_rgb;
        local_stats.mean_rgb += delta / f32(local_stats.count);
        let delta2 = color - local_stats.mean_rgb;
        local_stats.M2_rgb += delta * delta2;

        let ndotv = dot(normal, view_dir_world);
        if (ndotv > settings.near_normal_angle_cos) {
            local_stats.near_normal_count += 1u;
            let near_delta = color - local_stats.near_normal_mean;
            local_stats.near_normal_mean += near_delta / f32(local_stats.near_normal_count);
        }

        if (local_stats.near_normal_count > 0u) {
            let base_estimate = local_stats.near_normal_mean;
            let lambertian_prediction = base_estimate * max(0.0, ndotv);

            let residual = color - lambertian_prediction;
            let residual_intensity = length(residual);

            if (residual_intensity > 0.01) {
                update_topk_residuals(&local_stats, view_dir_world, residual_intensity);
                local_stats.residual_direction_sum += view_dir_world;
            }
        }
    }

    if (local_stats.topk_count > 3u) {
        let mean_direction = normalize(local_stats.residual_direction_sum);

        var angular_variance = 0.0;
        for (var i = 0u; i < local_stats.topk_count; i++) {
            let dir = local_stats.topk_directions[i];
            let angle = acos(clamp(dot(dir, mean_direction), -1.0, 1.0));
            angular_variance += angle * angle;
        }
        local_stats.residual_direction_M2 = angular_variance / f32(local_stats.topk_count);
    }

    stats[idx] = local_stats;
}
