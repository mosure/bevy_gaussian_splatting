#define_import_path bevy_gaussian_splatting::spherical_harmonics


const SH_C0 = 0.28209479177387814f;
const SH_C1 = 0.4886025119029199f;
const SH_C2 = array(
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
);
const SH_C3 = array(
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
);

fn compute_color_from_sh_3_degree(position: vec3<f32>, sh: array<vec3<f32>, 16>) -> vec3<f32> {
    let dir = normalize(position - uniforms.camera_position);
    var result = SH_C0 * sh[0];

    // if deg > 0
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    result = result + SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);

    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;

    // if (sh_degree > 1) {
    result = result +
        SH_C2[0] * xy * sh[4] +
        SH_C2[1] * yz * sh[5] +
        SH_C2[2] * (2. * zz - xx - yy) * sh[6] +
        SH_C2[3] * xz * sh[7] +
        SH_C2[4] * (xx - yy) * sh[8];

    // if (sh_degree > 2) {
    result = result +
        SH_C3[0] * y * (3. * xx - yy) * sh[9] +
        SH_C3[1] * xy * z * sh[10] +
        SH_C3[2] * y * (4. * zz - xx - yy) * sh[11] +
        SH_C3[3] * z * (2. * zz - 3. * xx - 3. * yy) * sh[12] +
        SH_C3[4] * x * (4. * zz - xx - yy) * sh[13] +
        SH_C3[5] * z * (xx - yy) * sh[14] +
        SH_C3[6] * x * (xx - 3. * yy) * sh[15];

    // unconditional
    result = result + 0.5;

    return max(result, vec3<f32>(0.));
}
