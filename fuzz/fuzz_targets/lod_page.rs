#![no_main]

use bevy_gaussian_splatting::io::lod::{LodCodecLimits, decode_page};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let _ = decode_page(bytes, LodCodecLimits::default());
});
