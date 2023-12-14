use bevy::prelude::*;

use static_assertions::assert_cfg;


#[cfg(feature = "sort_radix")]
pub mod radix;

#[cfg(feature = "sort_rayon")]
pub mod rayon;


assert_cfg!(
    any(
        feature = "sort_radix",
        feature = "sort_rayon",
    ),
    "no sort mode enabled",
);


#[derive(Component, Debug, Clone)]
enum SortMode {
    None,

    #[cfg(feature = "sort_radix")]
    Radix,

    #[cfg(feature = "sort_rayon")]
    Rayon,
}

impl Default for SortMode {
    #[allow(unreachable_code)]
    fn default() -> Self {
        #[cfg(feature = "sort_rayon")]
        { return Self::Rayon; }

        #[cfg(feature = "sort_radix")]
        { return Self::Radix; }

        Self::None
    }
}


// TODO: add SortPlugin to manage sub-sort plugin registration
