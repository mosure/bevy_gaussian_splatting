#![allow(dead_code)] // ShaderType derives emit unused check helpers
use core::time::Duration;
use std::marker::PhantomData;

use bevy::{
    asset::RenderAssetUsages,
    ecs::system::{SystemParamItem, lifetimeless::SRes},
    math::Vec3A,
    platform::time::Instant,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::*,
        renderer::RenderDevice,
    },
};
use bevy_interleave::prelude::*;
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use static_assertions::assert_cfg;

use crate::{CloudSettings, camera::GaussianCamera, gaussian::interface::CommonCloud};

#[cfg(feature = "lod")]
use crate::stream::bridge::GaussianLodBridgeUpdate;

#[cfg(feature = "sort_bitonic")]
pub mod bitonic;

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
pub mod radix;

#[cfg(feature = "sort_rayon")]
pub mod rayon;

#[cfg(feature = "sort_std")]
pub mod std_sort; // rename to std_sort.rs to avoid name conflict with std crate

assert_cfg!(
    any(
        feature = "sort_radix",
        feature = "sort_rayon",
        feature = "sort_std",
    ),
    "no sort mode enabled",
);

#[derive(Component, Debug, Clone, PartialEq, Reflect, Serialize, Deserialize)]
pub enum SortMode {
    None,

    #[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
    Radix,

    #[cfg(feature = "sort_rayon")]
    Rayon,

    #[cfg(feature = "sort_std")]
    Std,
}

impl Default for SortMode {
    #[allow(unreachable_code)]
    fn default() -> Self {
        #[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
        return Self::Radix;

        #[cfg(feature = "sort_rayon")]
        return Self::Rayon;

        #[cfg(feature = "sort_std")]
        return Self::Std;

        Self::None
    }
}

#[derive(Resource, Debug, Clone, PartialEq, Reflect)]
#[reflect(Resource)]
pub struct SortConfig {
    pub period_ms: usize,
}

impl Default for SortConfig {
    fn default() -> Self {
        Self { period_ms: 1000 }
    }
}

#[derive(Default)]
pub struct SortPluginFlag;
impl Plugin for SortPluginFlag {
    fn build(&self, _app: &mut App) {}
}

// TODO: make this generic /w shared components
#[derive(Default)]
pub struct SortPlugin<R: PlanarSync> {
    phantom: PhantomData<R>,
}

impl<R: PlanarSync> Plugin for SortPlugin<R>
where
    R::PlanarType: CommonCloud,
    R::GpuPlanarType: GpuPlanarStorage,
{
    fn build(&self, app: &mut App) {
        #[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
        app.add_plugins(radix::RadixSortPlugin::<R>::default());

        #[cfg(feature = "sort_rayon")]
        app.add_plugins(rayon::RayonSortPlugin::<R>::default());

        #[cfg(feature = "sort_std")]
        app.add_plugins(std_sort::StdSortPlugin::<R>::default());

        app.add_systems(Update, auto_insert_sorted_entries::<R>);

        if app.is_plugin_added::<SortPluginFlag>() {
            debug!("sort plugin flag already added");
            return;
        }
        app.add_plugins(SortPluginFlag);

        app.register_type::<SortConfig>();
        app.init_resource::<SortConfig>();

        app.register_type::<SortedEntries>();
        app.register_type::<SortedEntriesHandle>();
        app.init_asset::<SortedEntries>();
        app.register_asset_reflect::<SortedEntries>();

        app.register_type::<SortTrigger>();
        app.add_plugins(ExtractComponentPlugin::<SortTrigger>::default());

        app.add_plugins(RenderAssetPlugin::<GpuSortedEntry>::default());

        app.add_systems(
            Update,
            (update_sort_trigger, update_sorted_entries_sizes::<R>),
        );

        #[cfg(feature = "lod")]
        app.add_systems(
            PostUpdate,
            update_sorted_entries_sizes::<R>.after(GaussianLodBridgeUpdate),
        );

        #[cfg(feature = "buffer_texture")]
        app.add_systems(PostUpdate, update_textures_on_change);
    }
}

#[derive(Component, ExtractComponent, Debug, Default, Clone, PartialEq, Reflect)]
#[reflect(Component)]
pub struct SortTrigger {
    pub camera_index: usize,
    pub needs_sort: bool,
    pub last_camera_position: Vec3A,
    pub last_sort_time: Option<Instant>,
}

#[allow(clippy::type_complexity)]
fn update_sort_trigger(
    mut commands: Commands,
    new_gaussian_cameras: Query<Entity, (With<Camera>, With<GaussianCamera>, Without<SortTrigger>)>,
    mut existing_sort_triggers: Query<(&GlobalTransform, &Camera, &mut SortTrigger)>,
    sort_config: Res<SortConfig>,
) {
    for entity in new_gaussian_cameras.iter() {
        commands.entity(entity).insert(SortTrigger::default());
    }

    for (camera_transform, camera, mut sort_trigger) in existing_sort_triggers.iter_mut() {
        match sort_trigger.last_sort_time.as_ref() {
            None => {
                assert!(
                    camera.order >= 0,
                    "camera order must be a non-negative index into gaussian cameras"
                );

                sort_trigger.camera_index = camera.order as usize;
                sort_trigger.needs_sort = true;
                sort_trigger.last_sort_time = Some(Instant::now());
                continue;
            }
            Some(last_sort_time)
                if last_sort_time.elapsed()
                    < Duration::from_millis(sort_config.period_ms as u64) =>
            {
                continue;
            }
            Some(_) => {}
        }

        let camera_position = camera_transform.affine().translation;
        let camera_movement = sort_trigger.last_camera_position != camera_position;

        if camera_movement {
            sort_trigger.needs_sort = true;
            sort_trigger.last_sort_time = Some(Instant::now());
            sort_trigger.last_camera_position = camera_position;
        }
    }
}

#[cfg(feature = "buffer_texture")]
fn update_textures_on_change(
    mut images: ResMut<Assets<Image>>,
    mut ev_asset: MessageReader<AssetEvent<SortedEntries>>,
    sorted_entries_res: Res<Assets<SortedEntries>>,
) {
    for ev in ev_asset.read() {
        match ev {
            AssetEvent::Modified { id } => {
                let sorted_entries = sorted_entries_res.get(*id).unwrap();
                let mut image = images.get_mut(&sorted_entries.texture).unwrap();

                image.data = Some(bytemuck::cast_slice(sorted_entries.sorted.as_slice()).to_vec());
            }
            AssetEvent::Added { id: _ } => {}
            AssetEvent::Removed { id: _ } => {}
            AssetEvent::LoadedWithDependencies { id: _ } => {}
            AssetEvent::Unused { id: _ } => {}
        }
    }
}

#[allow(clippy::type_complexity)]
fn auto_insert_sorted_entries<R: PlanarSync>(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_clouds_res: Res<Assets<R::PlanarType>>,
    mut sorted_entries_res: ResMut<Assets<SortedEntries>>,
    gaussian_clouds: Query<
        (Entity, &R::PlanarTypeHandle, &CloudSettings),
        Without<SortedEntriesHandle>,
    >,
    gaussian_cameras: Query<Entity, (With<Camera>, With<GaussianCamera>)>,
    #[cfg(feature = "buffer_texture")] mut images: ResMut<Assets<Image>>,
) where
    R::PlanarType: CommonCloud,
{
    let camera_count = gaussian_cameras.iter().len();

    if camera_count == 0 {
        debug!("no gaussian cameras found");
        return;
    }

    for (entity, gaussian_cloud_handle, _settings) in gaussian_clouds.iter() {
        // // TODO: specialize vertex shader for sort mode (e.g. draw_indirect but no sort indirection)
        // if settings.sort_mode == SortMode::None {
        //     continue;
        // }

        if let Some(load_state) = asset_server.get_load_state(gaussian_cloud_handle.handle())
            && load_state.is_loading()
        {
            debug!("cloud asset is still loading");
            continue;
        }

        let cloud = gaussian_clouds_res.get(gaussian_cloud_handle.handle());
        if cloud.is_none() {
            debug!("cloud asset is not loaded");
            continue;
        }
        let cloud = cloud.unwrap();

        let sorted_entries = sorted_entries_res.add(SortedEntries::new(
            camera_count,
            cloud.len_sqrt_ceil().pow(2),
            #[cfg(feature = "buffer_texture")]
            &mut images,
        ));

        commands
            .entity(entity)
            .insert(SortedEntriesHandle(sorted_entries));
    }
}

fn update_sorted_entries_sizes<R: PlanarSync>(
    gaussian_clouds_res: Res<Assets<R::PlanarType>>,
    mut sorted_entries_res: ResMut<Assets<SortedEntries>>,
    sorted_entries: Query<(&R::PlanarTypeHandle, &SortedEntriesHandle)>,
    gaussian_cameras: Query<Entity, (With<Camera>, With<GaussianCamera>)>,
    #[cfg(feature = "buffer_texture")] mut images: ResMut<Assets<Image>>,
) where
    R::PlanarType: CommonCloud,
{
    let camera_count: usize = gaussian_cameras.iter().len();

    for (cloud_handle, sorted_handle) in sorted_entries.iter() {
        if camera_count == 0 {
            sorted_entries_res.remove(sorted_handle);
            continue;
        }

        let Some(cloud) = gaussian_clouds_res.get(cloud_handle.handle()) else {
            continue;
        };
        let required_entry_count = cloud.len_sqrt_ceil().pow(2);
        if let Some(sorted_entries) = sorted_entries_res.get(sorted_handle)
            && (sorted_entries.camera_count != camera_count
                || sorted_entries.entry_count < required_entry_count)
        {
            // The LoD bridge changes flat-source/atlas handles in PostUpdate,
            // after this Update system has run. Retain the per-camera high-water
            // mark so an exact-source bypass cannot shrink sort storage one
            // frame before the larger atlas is restored. This never raises peak
            // allocation: it only retains capacity already admitted for this
            // entity, and zero cameras still release the asset above.
            let retained_entry_count = sorted_entries.entry_count.max(required_entry_count);
            let new_entry = SortedEntries::new(
                camera_count,
                retained_entry_count,
                #[cfg(feature = "buffer_texture")]
                &mut images,
            );
            let _ = sorted_entries_res.insert(sorted_handle, new_entry);
        }
    }
}

/// Returns the exact binding size for one camera's sort entries, or `None`
/// while the uploaded entry asset still reflects an older/smaller cloud.
pub(crate) fn sort_entry_binding_size(
    entry_capacity: usize,
    required_entries: usize,
) -> Option<u64> {
    if required_entries == 0 || entry_capacity < required_entries {
        return None;
    }
    u64::try_from(required_entries)
        .ok()?
        .checked_mul(std::mem::size_of::<SortEntry>() as u64)
}

#[derive(Component, Clone, Debug, Default, PartialEq, Reflect)]
#[reflect(Component, Default)]
pub struct SortedEntriesHandle(pub Handle<SortedEntries>);

impl From<Handle<SortedEntries>> for SortedEntriesHandle {
    fn from(handle: Handle<SortedEntries>) -> Self {
        Self(handle)
    }
}

impl From<SortedEntriesHandle> for AssetId<SortedEntries> {
    fn from(handle: SortedEntriesHandle) -> Self {
        handle.0.id()
    }
}

impl From<&SortedEntriesHandle> for AssetId<SortedEntries> {
    fn from(handle: &SortedEntriesHandle) -> Self {
        handle.0.id()
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct SortEntry {
    pub key: u32,
    pub index: u32,
}

#[derive(Clone, Asset, Debug, Default, PartialEq, Reflect)]
pub struct SortedEntries {
    pub camera_count: usize,
    pub entry_count: usize,
    pub sorted: Vec<SortEntry>,

    #[cfg(feature = "buffer_texture")]
    pub texture: Handle<Image>,
}

impl SortedEntries {
    pub fn new(
        camera_count: usize,
        entry_count: usize,
        #[cfg(feature = "buffer_texture")] images: &mut Assets<Image>,
    ) -> Self {
        let sorted: Vec<SortEntry> = (0..camera_count)
            .flat_map(|_camera_idx| {
                (0..entry_count).map(|idx| SortEntry {
                    key: 1,
                    index: idx as u32,
                })
            })
            .collect();

        #[cfg(feature = "buffer_texture")]
        let mut sorted_entries = SortedEntries {
            camera_count,
            entry_count,
            sorted,
            texture: Handle::default(),
        };

        #[cfg(not(feature = "buffer_texture"))]
        let sorted_entries = SortedEntries {
            camera_count,
            entry_count,
            sorted,
        };

        #[cfg(feature = "buffer_texture")]
        {
            let side = (entry_count as f32).sqrt().ceil() as u32;
            let data = bytemuck::cast_slice(sorted_entries.sorted.as_slice()).to_vec();
            let mut image = Image::new(
                Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: camera_count as u32,
                },
                TextureDimension::D2,
                data,
                TextureFormat::Rg32Uint,
                RenderAssetUsages::default(),
            );
            image.texture_descriptor.usage =
                TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
            sorted_entries.texture = images.add(image);
        }

        sorted_entries
    }
}

impl RenderAsset for GpuSortedEntry {
    type SourceAsset = SortedEntries;
    type Param = SRes<RenderDevice>;

    fn prepare_asset(
        source: Self::SourceAsset,
        _: AssetId<Self::SourceAsset>,
        render_device: &mut SystemParamItem<Self::Param>,
        _: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let sorted_entry_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("sorted_entry_buffer"),
            contents: bytemuck::cast_slice(source.sorted.as_slice()),
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        });

        let count = source.sorted.len();

        Ok(GpuSortedEntry {
            sorted_entry_buffer,
            count,
            camera_count: source.camera_count,
            entry_count: source.entry_count,

            #[cfg(feature = "buffer_texture")]
            texture: source.texture,
        })
    }

    fn asset_usage(_: &Self::SourceAsset) -> RenderAssetUsages {
        RenderAssetUsages::default()
    }
}

// TODO: support instancing and multiple cameras
//       separate entry_buffer_a binding into unique a bind group to optimize buffer updates
#[derive(Debug, Clone)]
pub struct GpuSortedEntry {
    pub sorted_entry_buffer: Buffer,
    /// Total entries across every camera. Retained for compatibility with the
    /// original render asset surface.
    pub count: usize,
    /// Number of camera slices stored in [`Self::sorted_entry_buffer`].
    pub camera_count: usize,
    /// Capacity of one camera slice. Bind groups must use this value rather
    /// than the total count when a cloud handle changes size.
    pub entry_count: usize,

    #[cfg(feature = "buffer_texture")]
    pub texture: Handle<Image>,
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;
    use crate::gaussian::formats::planar_3d::{
        Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle,
    };

    #[cfg(feature = "lod")]
    #[derive(Resource)]
    struct PendingCloudHandle(Handle<PlanarGaussian3d>);

    #[cfg(feature = "lod")]
    fn apply_pending_cloud_handle(
        pending: Res<PendingCloudHandle>,
        mut clouds: Query<&mut PlanarGaussian3dHandle>,
    ) {
        for mut handle in &mut clouds {
            *handle = PlanarGaussian3dHandle(pending.0.clone());
        }
    }

    #[test]
    fn binding_size_rejects_an_older_smaller_entry_asset() {
        assert_eq!(sort_entry_binding_size(324, 2_048), None);
        assert_eq!(
            sort_entry_binding_size(2_116, 2_048),
            Some(2_048 * std::mem::size_of::<SortEntry>() as u64)
        );
    }

    #[test]
    fn sorted_entries_resize_after_a_cloud_handle_grows() {
        let mut world = World::new();
        world.init_resource::<Assets<PlanarGaussian3d>>();
        world.init_resource::<Assets<SortedEntries>>();

        let atlas = world
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(vec![Gaussian3d::default(); 2_048]));
        let sorted = world
            .resource_mut::<Assets<SortedEntries>>()
            .add(SortedEntries::new(1, 324));
        let cloud = world
            .spawn((
                PlanarGaussian3dHandle(atlas),
                SortedEntriesHandle(sorted.clone()),
            ))
            .id();
        world.spawn((Camera::default(), GaussianCamera::default()));

        world
            .run_system_once(update_sorted_entries_sizes::<Gaussian3d>)
            .expect("sorted-entry resize system runs");

        let handle = world
            .get::<SortedEntriesHandle>(cloud)
            .expect("cloud keeps its sorted-entry handle");
        let entries = world
            .resource::<Assets<SortedEntries>>()
            .get(handle)
            .expect("resized entry asset exists");
        assert_eq!(entries.camera_count, 1);
        assert_eq!(entries.entry_count, 2_116);
        assert_eq!(entries.sorted.len(), 2_116);
    }

    #[test]
    fn sorted_entries_retain_atlas_capacity_across_exact_source_bypass() {
        let mut world = World::new();
        world.init_resource::<Assets<PlanarGaussian3d>>();
        world.init_resource::<Assets<SortedEntries>>();

        let source = world
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(vec![Gaussian3d::default(); 256]));
        let atlas = world
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(vec![Gaussian3d::default(); 2_048]));
        let sorted = world
            .resource_mut::<Assets<SortedEntries>>()
            .add(SortedEntries::new(1, 2_116));
        let cloud = world
            .spawn((
                PlanarGaussian3dHandle(atlas.clone()),
                SortedEntriesHandle(sorted.clone()),
            ))
            .id();
        world.spawn((Camera::default(), GaussianCamera::default()));

        world
            .run_system_once(update_sorted_entries_sizes::<Gaussian3d>)
            .expect("atlas-sized sort storage is current");
        world
            .entity_mut(cloud)
            .insert(PlanarGaussian3dHandle(source));
        world
            .run_system_once(update_sorted_entries_sizes::<Gaussian3d>)
            .expect("exact-source bypass update runs");

        let bypass_entries = world
            .resource::<Assets<SortedEntries>>()
            .get(&sorted)
            .expect("sort storage remains allocated while bypassed");
        assert_eq!(bypass_entries.entry_count, 2_116);
        assert_eq!(bypass_entries.sorted.len(), 2_116);

        world
            .entity_mut(cloud)
            .insert(PlanarGaussian3dHandle(atlas));
        world
            .run_system_once(update_sorted_entries_sizes::<Gaussian3d>)
            .expect("atlas return update runs");

        let restored_entries = world
            .resource::<Assets<SortedEntries>>()
            .get(&sorted)
            .expect("retained storage remains addressable after atlas return");
        assert_eq!(restored_entries.entry_count, 2_116);
        assert!(sort_entry_binding_size(restored_entries.entry_count, 2_048).is_some());
    }

    #[cfg(feature = "lod")]
    #[test]
    fn post_bridge_sizing_covers_the_first_source_to_atlas_swap() {
        let mut world = World::new();
        world.init_resource::<Assets<PlanarGaussian3d>>();
        world.init_resource::<Assets<SortedEntries>>();

        let source = world
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(vec![Gaussian3d::default(); 256]));
        let atlas = world
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(vec![Gaussian3d::default(); 2_048]));
        world.insert_resource(PendingCloudHandle(atlas.clone()));
        let sorted = world
            .resource_mut::<Assets<SortedEntries>>()
            .add(SortedEntries::new(1, 256));
        let cloud = world
            .spawn((
                PlanarGaussian3dHandle(source),
                SortedEntriesHandle(sorted.clone()),
            ))
            .id();
        world.spawn((Camera::default(), GaussianCamera::default()));

        let mut post_update = Schedule::default();
        post_update.add_systems(apply_pending_cloud_handle.in_set(GaussianLodBridgeUpdate));
        post_update
            .add_systems(update_sorted_entries_sizes::<Gaussian3d>.after(GaussianLodBridgeUpdate));
        post_update.run(&mut world);

        assert_eq!(
            world
                .get::<PlanarGaussian3dHandle>(cloud)
                .unwrap()
                .handle()
                .id(),
            atlas.id()
        );
        let entries = world
            .resource::<Assets<SortedEntries>>()
            .get(&sorted)
            .expect("post-bridge sizing keeps the sort asset available");
        assert_eq!(entries.entry_count, 2_116);
        assert_eq!(entries.sorted.len(), 2_116);
        assert!(sort_entry_binding_size(entries.entry_count, 2_048).is_some());
    }
}
