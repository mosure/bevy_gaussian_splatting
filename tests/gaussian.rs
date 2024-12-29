use bevy_gaussian_splatting::{
    io::codec::CloudCodec,
    PlanarGaussian3d,
    PlanarGaussian4d,
    random_gaussians_3d,
    random_gaussians_4d,
};


#[test]
fn test_codec_3d() {
    let count = 100;

    let gaussians = random_gaussians_3d(count);
    let encoded = gaussians.encode();
    let decoded = PlanarGaussian3d::decode(encoded.as_slice());

    assert_eq!(gaussians, decoded);
}

#[test]
fn test_codec_4d() {
    let count = 100;

    let gaussians = random_gaussians_4d(count);
    let encoded = gaussians.encode();
    let decoded = PlanarGaussian4d::decode(encoded.as_slice());

    assert_eq!(gaussians, decoded);
}
