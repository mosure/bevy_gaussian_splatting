use static_assertions::assert_cfg;

pub mod cloud;
pub mod covariance;
pub mod f16;
pub mod f32;
pub mod formats;
pub mod interface;
pub mod iter;
#[cfg(feature = "lod_build")]
// The bounded GPU sort/reduction contracts remain available on Wasm, while
// blocking readback is rejected through a typed unsupported error.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub mod lod_build_gpu;
pub mod lod_debug;
pub mod lod_settings;
pub mod settings;

assert_cfg!(
    any(feature = "packed", feature = "planar",),
    "specify one of the following features: packed, planar",
);
