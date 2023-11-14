use bevy_gaussian_splatting::{
    Gaussian,
    io,
    random_gaussians,
};


#[test]
fn test_encode() {
    let gaussians = random_gaussians(100);
    let encoded = gaussians.encode();
    assert_eq!(encoded.len(), 100 * std::mem::size_of::<Gaussian>());
}

#[test]
fn test_decode() {

}
