//! Explicitly separated CPU preparation and GPU lifecycle qualification.
use bevy_gaussian_splatting::testing::lod_runtime_capture::virtual_city;
use std::{path::Path, process::ExitCode};

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let result = match args.as_slice() {
        [command, flag, path] if flag == "--config" && command == "prepare" => {
            virtual_city::prepare(Path::new(path))
        }
        [command, flag, path] if flag == "--config" && command == "run" => {
            virtual_city::run(Path::new(path))
        }
        _ => {
            eprintln!("usage: capture_lod_virtual_city prepare|run --config CONFIG.json");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("capture_lod_virtual_city: {error}");
            ExitCode::FAILURE
        }
    }
}
