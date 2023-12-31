use byte_unit::{
    Byte,
    UnitType,
};

use bevy_gaussian_splatting::{
    GaussianCloud,
    io::{
        ply::parse_ply,
        writer::write_gaussian_cloud_to_file,
    },
};

#[cfg(feature = "query_sparse")]
use bevy_gaussian_splatting::query::sparse::SparseSelect;


fn main() {
    let filename = std::env::args().nth(1).expect("no filename given");

    println!("converting {}", filename);

    let file = std::fs::File::open(&filename).expect("failed to open file");
    let mut reader = std::io::BufReader::new(file);

    let mut cloud = GaussianCloud::from_gaussians(
        parse_ply(&mut reader).expect("failed to parse ply file"),
    );

    #[cfg(feature = "query_sparse")]
    {
        let sparse_selection = SparseSelect::default().select(&cloud);

        cloud = sparse_selection.indicies.iter()
            .map(|idx| cloud.gaussian(*idx))
            .collect();
    }

    let base_filename = filename.split('.').next().expect("no extension").to_string();
    let gcloud_filename = base_filename + ".gcloud";

    write_gaussian_cloud_to_file(&cloud, &gcloud_filename);

    let post_encode_bytes = Byte::from_u64(std::fs::metadata(&gcloud_filename).expect("failed to get metadata").len());
    println!("output file size: {}", post_encode_bytes.get_appropriate_unit(UnitType::Decimal));
}
