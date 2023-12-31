use bevy_gaussian_splatting::{
    GaussianCloud,
    io::codec::GaussianCloudCodec,
    random_gaussians,
};


#[test]
fn test_codec() {
    let count = 10000;

    let gaussians = random_gaussians(count);
    let encoded = gaussians.encode();
    let decoded = GaussianCloud::decode(encoded.as_slice());

    assert_eq!(gaussians, decoded);
}
