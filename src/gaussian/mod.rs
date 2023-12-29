use static_assertions::assert_cfg;

pub mod cloud;
pub mod f32;
pub mod packed;
pub mod rand;
pub mod settings;

#[cfg(feature = "f16")]
pub mod f16;


assert_cfg!(
    any(
        feature = "f16",
        feature = "f32",
    ),
    "specify one of the following features: f16, f32",
);

assert_cfg!(
    any(
        feature = "packed",
        feature = "planar",
    ),
    "specify one of the following features: packed, planar",
);


// PACKED_f16 is not supported
assert_cfg!(
    not(all(
        feature = "f16",
        feature = "packed",
    )),
    "f16 and packed are incompatible",
);
