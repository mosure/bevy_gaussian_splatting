struct Rotation {
    value: vec4<f32>,
};

struct ScaleOpacity {
    value: vec4<f32>,
};

struct CovarianceOpacity {
    first: vec4<f32>,
    second: vec4<f32>,
};

struct UploadRange {
    start: u32,
    count: u32,
};

@group(0) @binding(0) var<storage, read> rotations: array<Rotation>;
@group(0) @binding(1) var<storage, read> scales: array<ScaleOpacity>;
@group(0) @binding(2) var<storage, read_write> covariances: array<CovarianceOpacity>;
@group(0) @binding(3) var<storage, read> ranges: array<UploadRange>;

@compute @workgroup_size(64, 1, 1)
fn derive_covariance(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let range = ranges[invocation.y];
    if invocation.x >= range.count {
        return;
    }
    let index = range.start + invocation.x;
    let q = rotations[index].value;
    let scale_opacity = scales[index].value;
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    let rotation = mat3x3<f32>(
        vec3<f32>(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - r * z),
            2.0 * (x * z + r * y),
        ),
        vec3<f32>(
            2.0 * (x * y + r * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - r * x),
        ),
        vec3<f32>(
            2.0 * (x * z - r * y),
            2.0 * (y * z + r * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    );
    let scale = mat3x3<f32>(
        vec3<f32>(scale_opacity.x, 0.0, 0.0),
        vec3<f32>(0.0, scale_opacity.y, 0.0),
        vec3<f32>(0.0, 0.0, scale_opacity.z),
    );
    let transform = scale * rotation;
    let sigma = transpose(transform) * transform;
    covariances[index] = CovarianceOpacity(
        vec4<f32>(sigma[0][0], sigma[0][1], sigma[0][2], sigma[1][1]),
        vec4<f32>(sigma[1][2], sigma[2][2], scale_opacity.w, 0.0),
    );
}
