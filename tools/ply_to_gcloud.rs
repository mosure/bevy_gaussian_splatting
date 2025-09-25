use byte_unit::{Byte, UnitType};

use bevy_gaussian_splatting::io::{codec::CloudCodec, ply::parse_ply_3d};

#[cfg(feature = "query_sparse")]
use bevy_gaussian_splatting::query::sparse::SparseSelect;

#[allow(dead_code)]
fn is_point_in_transformed_sphere(pos: &[f32; 3]) -> bool {
    let inv_scale_x = 1.0 / 1.75;
    let inv_scale_y = 1.0 / 1.75;
    let inv_scale_z = 1.0 / 1.75;

    let inv_trans_x = 1.7;
    let inv_trans_y = -0.5;
    let inv_trans_z = -3.8;

    let transformed_x = (pos[0] + inv_trans_x) * inv_scale_x;
    let transformed_y = (pos[1] + inv_trans_y) * inv_scale_y;
    let transformed_z = (pos[2] + inv_trans_z) * inv_scale_z;

    transformed_x.powi(2) + transformed_y.powi(2) + transformed_z.powi(2) <= 1.0
}

// TODO: add better argument parsing
#[allow(unused_mut)]
fn main() {
    let filename = std::env::args().nth(1).expect("no filename given");

    println!("converting `{filename}` file to gcloud");

    let file = std::fs::File::open(&filename).expect("failed to open file");
    let mut reader = std::io::BufReader::new(file);

    // TODO: support 4d gaussian -> .gc4d
    let mut cloud = parse_ply_3d(&mut reader).expect("failed to parse ply file");

    // TODO: prioritize mesh selection over export filter
    // println!("initial cloud size: {}", cloud.len());
    // cloud = (0..cloud.len())
    //     .filter(|&idx| {
    //         is_point_in_transformed_sphere(
    //             cloud.position(idx),
    //         )
    //     })
    //     .map(|idx| cloud.gaussian(idx))
    //     .collect();
    // println!("filtered position cloud size: {}", cloud.len());

    #[cfg(feature = "query_sparse")]
    {
        let sparse_selection = SparseSelect::default().select(&cloud).invert(cloud.len());

        cloud = sparse_selection
            .indicies
            .iter()
            .map(|idx| cloud.gaussian(*idx))
            .collect();
        println!("sparsity filtered cloud size: {cloud.len()}");
    }

    let base_filename = filename
        .split('.')
        .next()
        .expect("no extension")
        .to_string();
    let gcloud_filename = base_filename + ".gcloud";

    cloud.write_to_file(&gcloud_filename);

    let post_encode_bytes = Byte::from_u64(
        std::fs::metadata(&gcloud_filename)
            .expect("failed to get metadata")
            .len(),
    );
    println!(
        "output file size: {}",
        post_encode_bytes.get_appropriate_unit(UnitType::Decimal)
    );
}
