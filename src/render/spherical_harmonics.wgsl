#define_import_path bevy_gaussian_splatting::spherical_harmonics


const SH_C0_ = 0.28209479177387814f;
const SH_C1_ = 0.4886025119029199f;
const SH_C2_ = array<f32, 5>(
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
);
const SH_C3_ = array<f32, 7>(
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
);

// consider moving to this format: https://github.com/Lichtso/splatter/blob/9a7b2c1946c0b009874b4cc1ab6a5b5586fdf8e5/src/shaders.wgsl#L236-L254s


fn compute_color_from_sh_3_degree(
    position: vec3<f32>,
    sh: array<vec3<f32>, #{MAX_SH_COEFF_COUNT}>,
    camera_position: vec3<f32>,
) -> vec3<f32> {
    let dir = normalize(position - camera_position);
    var result = SH_C0_ * sh[0];

    // if deg > 0
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    result = result + SH_C1_ * (-y * sh[1] + z * sh[2] - x * sh[3]);

    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;

    // if (sh_degree > 1) {
    result = result +
        SH_C2_[0] * xy * sh[4] +
        SH_C2_[1] * yz * sh[5] +
        SH_C2_[2] * (2. * zz - xx - yy) * sh[6] +
        SH_C2_[3] * xz * sh[7] +
        SH_C2_[4] * (xx - yy) * sh[8];

    // if (sh_degree > 2) {
    result = result +
        SH_C3_[0] * y * (3. * xx - yy) * sh[9] +
        SH_C3_[1] * xy * z * sh[10] +
        SH_C3_[2] * y * (4. * zz - xx - yy) * sh[11] +
        SH_C3_[3] * z * (2. * zz - 3. * xx - 3. * yy) * sh[12] +
        SH_C3_[4] * x * (4. * zz - xx - yy) * sh[13] +
        SH_C3_[5] * z * (xx - yy) * sh[14] +
        SH_C3_[6] * x * (xx - 3. * yy) * sh[15];

    // unconditional
    result = result + 0.5;

    return max(result, vec3<f32>(0.));
}
