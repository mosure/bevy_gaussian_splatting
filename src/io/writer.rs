use std::io::Write;

use crate::{
    GaussianCloud,
    io::codec::GaussianCloudCodec,
};


pub fn write_gaussian_cloud_to_file(
    cloud: &GaussianCloud,
    path: &str,
) {
    let gcloud_file = std::fs::File::create(path).expect("failed to create file");
    let mut gcloud_writer = std::io::BufWriter::new(gcloud_file);

    let data = cloud.encode();
    gcloud_writer.write_all(data.as_slice()).expect("failed to write to gcloud file");
}
