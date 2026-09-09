#define_import_path bevy_gaussian_splatting::support

// A rotation-invariant C1 tail inside the existing support. For a 3-sigma
// cutoff the inner q<=8 density is unchanged and q>=9 is exactly zero.
// No mass normalization: all rendering paths use this same finite kernel.
fn gaussian_support_weight(q: f32, cutoff_squared: f32) -> f32 {
    if !(q >= 0.0) || !(cutoff_squared > 0.0) || q >= cutoff_squared { return 0.0; }
    let u = clamp(9.0 * (q / cutoff_squared) - 8.0, 0.0, 1.0);
    let remaining = 1.0 - u;
    return exp(-0.5 * q) * remaining * remaining * (1.0 + 2.0 * u);
}
