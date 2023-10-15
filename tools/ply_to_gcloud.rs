use bincode2::serialize_into;
use flate2::{
    Compression,
    write::GzEncoder,
};

use bevy_gaussian_splatting::{
    GaussianCloud,
    ply::parse_ply,
};


fn main() {
    let filename = std::env::args().nth(1).expect("no filename given");

    println!("converting {}", filename);

    // filepath to BufRead
    let file = std::fs::File::open(&filename).expect("failed to open file");
    let mut reader = std::io::BufReader::new(file);

    let cloud = GaussianCloud(parse_ply(&mut reader).expect("failed to parse ply file"));

    // write cloud to .gcloud file (remove .ply)
    let base_filename = filename.split('.').next().expect("no extension").to_string();
    let gcloud_filename = base_filename + ".gcloud";
    // let gcloud_file = std::fs::File::create(&gcloud_filename).expect("failed to create file");
    // let mut writer = std::io::BufWriter::new(gcloud_file);

    // serialize_into(&mut writer, &cloud).expect("failed to encode cloud");

    // write gloud.gz
    let gz_file = std::fs::File::create(&gcloud_filename).expect("failed to create file");
    let mut gz_writer = std::io::BufWriter::new(gz_file);
    let mut gz_encoder = GzEncoder::new(&mut gz_writer, Compression::default());  // TODO: consider switching to fast (or support multiple options), default is a bit slow
    serialize_into(&mut gz_encoder, &cloud).expect("failed to encode cloud");
}
