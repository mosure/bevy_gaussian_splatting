use static_assertions::assert_cfg;


#[cfg(feature = "io_bincode2")]
pub mod bincode2;

#[cfg(feature = "io_flexbuffers")]
pub mod flexbuffers;

#[cfg(feature = "io_rkyv")]
pub mod rkyv;


assert_cfg!(
    any(
        feature = "io_bincode2",
        feature = "io_flexbuffers",
        feature = "io_rkyv",
    ),
    "no gcloud io enabled",
);
