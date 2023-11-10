use std::io::Write;

use byte_unit::Byte;
use flexbuffers::FlexbufferSerializer;
use serde::Serialize;

use bevy_gaussian_splatting::{
    GaussianCloud,
    ply::parse_ply,
};


fn main() {
    let filename = std::env::args().nth(1).expect("no filename given");

    println!("converting {}", filename);

    let file = std::fs::File::open(&filename).expect("failed to open file");
    let mut reader = std::io::BufReader::new(file);

    let cloud = GaussianCloud(parse_ply(&mut reader).expect("failed to parse ply file"));

    let base_filename = filename.split('.').next().expect("no extension").to_string();
    let gcloud_filename = base_filename + ".gcloud";

    let gcloud_file = std::fs::File::create(&gcloud_filename).expect("failed to create file");
    let mut gcloud_writer = std::io::BufWriter::new(gcloud_file);

    let mut serializer = FlexbufferSerializer::new();
    cloud.serialize(&mut serializer).expect("failed to serialize cloud");
    gcloud_writer.write_all(serializer.view()).expect("failed to write to gcloud file");

    let post_encode_bytes = Byte::from_bytes(std::fs::metadata(&gcloud_filename).expect("failed to get metadata").len() as u128);
    println!("output file size: {}", post_encode_bytes.get_appropriate_unit(true).to_string());
}
