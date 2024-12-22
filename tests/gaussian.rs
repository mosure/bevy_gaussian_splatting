use bevy_gaussian_splatting::{
    Cloud,
    io::codec::CloudCodec,
    random_gaussians,
};


#[test]
fn test_codec() {
    let count = 100;

    let gaussians = random_gaussians(count);
    let encoded = gaussians.encode();
    let decoded = Cloud::decode(encoded.as_slice());

    assert_eq!(gaussians, decoded);
}
