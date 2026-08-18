#![no_main]

use bevy_gaussian_splatting::io::lod::decode_lod_shard_index;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let _ = decode_lod_shard_index(bytes, bytes.len() as u64, 1 << 20);
});
