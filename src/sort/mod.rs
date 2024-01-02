use bevy::{
    prelude::*,
    asset::LoadState,
    ecs::system::{
        lifetimeless::SRes,
        SystemParamItem,
    },
    reflect::TypeUuid,
    render::{
        render_resource::{
            Buffer,
            BufferInitDescriptor,
            BufferUsages,
            Extent3d,
            ShaderType,
            TextureDimension,
            TextureFormat,
        },
        render_asset::{
            RenderAsset,
            RenderAssetPlugin,
            PrepareAssetError,
        },
        renderer::RenderDevice,
    },
};
use bytemuck::{
    Pod,
    Zeroable,
};
use static_assertions::assert_cfg;

use crate::{
    GaussianCloud,
    GaussianCloudSettings,
};


#[cfg(feature = "sort_radix")]
pub mod radix;

#[cfg(feature = "sort_rayon")]
pub mod rayon;

#[cfg(feature = "sort_std")]
pub mod std; // rename to std_sort.rs to avoid name conflict with std crate


assert_cfg!(
    any(
        feature = "sort_radix",
        feature = "sort_rayon",
        feature = "sort_std",
    ),
    "no sort mode enabled",
);


#[derive(
    Component,
    Debug,
    Clone,
    PartialEq,
    Reflect,
)]
pub enum SortMode {
    None,

    #[cfg(feature = "sort_radix")]
    Radix,

    #[cfg(feature = "sort_rayon")]
    Rayon,

    #[cfg(feature = "sort_std")]
    Std,
}

impl Default for SortMode {
    #[allow(unreachable_code)]
    fn default() -> Self {
        #[cfg(feature = "sort_rayon")]
        return Self::Rayon;

        #[cfg(feature = "sort_radix")]
        return Self::Radix;

        #[cfg(feature = "sort_std")]
        return Self::Std;

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
        app.add_plugins(rayon::RayonSortPlugin);

        #[cfg(feature = "sort_std")]
        app.add_plugins(std::StdSortPlugin);


        app.register_type::<SortedEntries>();
        app.init_asset::<SortedEntries>();
        app.register_asset_reflect::<SortedEntries>();

        app.add_plugins(RenderAssetPlugin::<SortedEntries>::default());

        app.add_systems(Update, auto_insert_sorted_entries);

        #[cfg(feature = "buffer_texture")]
        app.add_systems(PostUpdate, update_textures_on_change);
    }
}


#[cfg(feature = "buffer_texture")]
fn update_textures_on_change(
    mut images: ResMut<Assets<Image>>,
    mut ev_asset: EventReader<AssetEvent<SortedEntries>>,
    sorted_entries_res: Res<Assets<SortedEntries>>,
) {
    for ev in ev_asset.read() {
        match ev {
            AssetEvent::Modified { id } => {
                let sorted_entries = sorted_entries_res.get(*id).unwrap();
                let image = images.get_mut(&sorted_entries.texture).unwrap();

                image.data = bytemuck::cast_slice(sorted_entries.sorted.as_slice()).to_vec();
            },
            AssetEvent::Added { id: _ } => {},
            AssetEvent::Removed { id: _ } => {},
            AssetEvent::LoadedWithDependencies { id: _ } => {},
        }
    }
}


#[allow(clippy::type_complexity)]
fn auto_insert_sorted_entries(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_clouds_res: Res<Assets<GaussianCloud>>,
    mut sorted_entries_res: ResMut<Assets<SortedEntries>>,
    gaussian_clouds: Query<(
        Entity,
        &Handle<GaussianCloud>,
        &GaussianCloudSettings,
        Without<Handle<SortedEntries>>,
    )>,

    #[cfg(feature = "buffer_texture")]
    mut images: ResMut<Assets<Image>>,
) {
    for (
        entity,
        gaussian_cloud_handle,
        _settings,
        _,
    ) in gaussian_clouds.iter() {
        // // TODO: specialize vertex shader for sort mode (e.g. draw_indirect but no sort indirection)
        // if settings.sort_mode == SortMode::None {
        //     continue;
        // }

        if Some(LoadState::Loading) == asset_server.get_load_state(gaussian_cloud_handle) {
            continue;
        }

        let cloud = gaussian_clouds_res.get(gaussian_cloud_handle);
        if cloud.is_none() {
            continue;
        }
        let cloud = cloud.unwrap();

        let sorted: Vec<SortEntry> = (0..cloud.len())
            .map(|idx| {
                SortEntry {
                    key: 1,
                    index: idx as u32,
                }
            })
            .collect();

        // TODO: move gaussian_cloud and sorted_entry assets into an asset bundle
        #[cfg(feature = "buffer_storage")]
        let sorted_entries = sorted_entries_res.add(SortedEntries {
            sorted,
        });

        #[cfg(feature = "buffer_texture")]
        let sorted_entries = sorted_entries_res.add(SortedEntries {
            texture: images.add(Image::new(
                Extent3d {
                    width: cloud.len_sqrt_ceil() as u32,
                    height: cloud.len_sqrt_ceil() as u32,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                bytemuck::cast_slice(sorted.as_slice()).to_vec(),
                TextureFormat::Rg32Uint,
            )),
            sorted,
        });

        commands.entity(entity)
            .insert(sorted_entries);
    }
}


#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Reflect,
    ShaderType,
    Pod,
    Zeroable,
)]
#[repr(C)]
pub struct SortEntry {
    pub key: u32,
    pub index: u32,
}

#[derive(
    Clone,
    Asset,
    Debug,
    Default,
    PartialEq,
    Reflect,
    TypeUuid,
)]
#[uuid = "ac2f08eb-fa13-ccdd-ea11-51571ea332d5"]
pub struct SortedEntries {
    pub sorted: Vec<SortEntry>,

    #[cfg(feature = "buffer_texture")]
    pub texture: Handle<Image>,
}

impl RenderAsset for SortedEntries {
    type ExtractedAsset = SortedEntries;
    type PreparedAsset = GpuSortedEntry;
    type Param = SRes<RenderDevice>;

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        sorted_entries: Self::ExtractedAsset,
        render_device: &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let sorted_entry_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("sorted_entry_buffer"),
            contents: bytemuck::cast_slice(sorted_entries.sorted.as_slice()),
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        });

        let count = sorted_entries.sorted.len();

        Ok(GpuSortedEntry {
            sorted_entry_buffer,
            count,

            #[cfg(feature = "buffer_texture")]
            texture: sorted_entries.texture,
        })
    }
}


// TODO: support instancing and multiple cameras
//       separate entry_buffer_a binding into unique a bind group to optimize buffer updates
#[derive(Debug, Clone)]
pub struct GpuSortedEntry {
    pub sorted_entry_buffer: Buffer,
    pub count: usize,

    #[cfg(feature = "buffer_texture")]
    pub texture: Handle<Image>,
}
