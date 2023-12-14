use bevy::{
    prelude::*,
    ecs::system::{
        lifetimeless::SRes,
        SystemParamItem,
    },
    render::{
        render_resource::*,
        renderer::RenderDevice,
    },
};

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
        return Self::Rayon;

        #[cfg(feature = "sort_radix")]
        return Self::Radix;

        Self::None
    }
}


#[derive(Default)]
pub struct SortPlugin;

impl Plugin for SortPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(feature = "sort_radix")]
        app.add_plugins(radix::RadixSortPlugin);

        #[cfg(feature = "sort_rayon")]
        app.add_plugin(rayon::RayonSortPlugin);
    }
}


pub struct SortEntry {
    pub key: u32,  // TODO: CPU sort doesn't require radix keys, figure out how to efficiently remove to save VRAM in CPU sort mode
    pub index: u32,
}

#[derive(Debug, Clone)]
pub struct GpuSortedEntry {
    pub sorted_entry_buffer: Buffer,
}
impl GpuSortedEntry {
    // TODO: move into a 2nd order asset system
    pub fn new(
        count: usize,
        render_device: &mut SystemParamItem<SRes<RenderDevice>>,
    ) -> Self {
        let sorted_entry_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("sorted_entry_buffer"),
            size: (count * std::mem::size_of::<SortEntry>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        GpuSortedEntry {
            sorted_entry_buffer,
        }
    }
}
