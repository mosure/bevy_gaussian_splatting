#![no_main]

use bevy_gaussian_splatting::io::lod::{LodCodecLimits, decode_manifest};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let _ = decode_manifest(bytes, LodCodecLimits::default());
});
