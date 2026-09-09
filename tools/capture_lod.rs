//! Actual native LoD/flat capture and bounded source conversion.
use bevy_gaussian_splatting::testing::lod_runtime_capture::{
    attribute_cut, convert_glb_to_ply, export_cohort, export_rung, fit_rung, run_capture,
};
use bevy_gaussian_splatting::testing::point_capture::run_point_comparison;
use std::{path::Path, process::ExitCode};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.as_slice() {
        [flag, config] if flag == "--config" => run_capture(Path::new(config)),
        [flag, config] if flag == "--fit-rung" => fit_rung(Path::new(config)),
        [flag, config] if flag == "--export-cohort" => export_cohort(Path::new(config)),
        [flag, config] if flag == "--attribute-cut" => attribute_cut(Path::new(config)),
        [flag, config] if flag == "--compare-points" => run_point_comparison(Path::new(config)),
        [flag, input, output_flag, output, max_flag, limit]
            if flag == "--convert-glb"
                && output_flag == "--output"
                && max_flag == "--max-gaussians" =>
        {
            match limit.parse() {
                Ok(limit) => convert_glb_to_ply(Path::new(input), Path::new(output), limit),
                Err(error) => Err(error.into()),
            }
        }
        [
            flag,
            input,
            output_flag,
            output,
            depth_flag,
            depth,
            max_flag,
            limit,
        ] if flag == "--export-rung"
            && output_flag == "--output"
            && depth_flag == "--depth"
            && max_flag == "--max-gaussians" =>
        {
            match (depth.parse(), limit.parse()) {
                (Ok(depth), Ok(limit)) => {
                    export_rung(Path::new(input), Path::new(output), depth, limit)
                }
                (Err(error), _) | (_, Err(error)) => Err(error.into()),
            }
        }
        _ => {
            eprintln!(
                "usage: capture_lod --config PATH.json\n       capture_lod --fit-rung FIT.json\n       capture_lod --export-cohort CONFIG.json\n       capture_lod --attribute-cut CONFIG.json\n       capture_lod --compare-points CONFIG.json\n       capture_lod --convert-glb INPUT.glb --output OUTPUT.ply --max-gaussians N\n       capture_lod --export-rung MANIFEST --output OUTPUT.ply --depth D --max-gaussians N"
            );
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("capture_lod: {error}");
            ExitCode::FAILURE
        }
    }
}
