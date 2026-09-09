#![allow(dead_code)] // ShaderType derives emit unused check helpers
use core::time::Duration;
use std::marker::PhantomData;

use bevy::{
    asset::RenderAssetUsages,
    ecs::system::{SystemParamItem, lifetimeless::SRes},
    platform::time::Instant,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
};
use bevy_interleave::prelude::*;
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use static_assertions::assert_cfg;

use crate::{CloudSettings, camera::GaussianCamera, gaussian::interface::CommonCloud};

#[cfg(feature = "lod")]
use crate::stream::{
    atlas_upload::LodTransientAtlasRegistry, bridge::GaussianLodBridgeUpdate,
    package::GaussianLodPackageUpdate,
};

#[cfg(feature = "lod")]
#[derive(SystemSet, Clone, Debug, Eq, Hash, PartialEq)]
struct SortStorageResize;

#[cfg(feature = "lod")]
#[derive(SystemSet, Clone, Debug, Eq, Hash, PartialEq)]
struct SortStorageCleanup;

#[derive(SystemSet, Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct CpuSort;

#[derive(SystemSet, Clone, Debug, Eq, Hash, PartialEq)]
struct CpuSortInputs;

#[cfg(feature = "sort_radix")]
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
        #[cfg(feature = "sort_radix")]
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
        #[cfg(feature = "sort_radix")]
        app.add_plugins(radix::RadixSortPlugin::<R>::default());

        #[cfg(feature = "sort_rayon")]
        app.add_plugins(rayon::RayonSortPlugin::<R>::default());

        #[cfg(feature = "sort_std")]
        app.add_plugins(std_sort::StdSortPlugin::<R>::default());

        app.add_systems(
            Update,
            (
                auto_insert_sorted_entries::<R>,
                update_sorted_entries_sizes::<R>,
            )
                .chain()
                .in_set(CpuSortInputs)
                .after(update_sort_trigger),
        );

        #[cfg(any(feature = "sort_std", feature = "sort_rayon"))]
        app.add_systems(
            Update,
            invalidate_cpu_sort_inputs::<R>
                .in_set(CpuSortInputs)
                .after(update_sort_trigger)
                .after(update_sorted_entries_sizes::<R>)
                .before(CpuSort),
        );

        #[cfg(feature = "lod")]
        app.add_systems(
            PostUpdate,
            (
                auto_insert_sorted_entries::<R>,
                update_sorted_entries_sizes::<R>,
            )
                .chain()
                .in_set(SortStorageResize),
        );

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

        app.add_systems(Update, update_sort_trigger.before(CpuSort));
        app.add_systems(Update, finish_cpu_sort.after(CpuSort));
        #[cfg(any(feature = "sort_std", feature = "sort_rayon"))]
        app.add_systems(
            Update,
            refresh_cpu_sort_camera_depth
                .after(CpuSortInputs)
                .before(CpuSort),
        );

        #[cfg(feature = "lod")]
        app.configure_sets(
            PostUpdate,
            (
                SortStorageResize
                    .after(GaussianLodBridgeUpdate)
                    .after(GaussianLodPackageUpdate),
                SortStorageCleanup.after(SortStorageResize),
            ),
        )
        .add_systems(
            PostUpdate,
            cleanup_orphaned_sorted_entries.in_set(SortStorageCleanup),
        );
    }
}

#[derive(Component, ExtractComponent, Debug, Default, Clone, PartialEq, Reflect)]
#[reflect(Component)]
pub struct SortTrigger {
    /// Dense Gaussian-camera slot, independent of Bevy's render order.
    pub camera_index: usize,
    pub needs_sort: bool,
    /// World-space plane evaluating forward camera depth (-view-space Z).
    pub last_camera_depth: Vec4,
    pub last_sort_time: Option<Instant>,
    #[reflect(ignore)]
    sorted_this_frame: bool,
    #[reflect(ignore)]
    pending_assets: bool,
}

#[allow(clippy::type_complexity)]
fn update_sort_trigger(
    mut commands: Commands,
    mut cameras: Query<
        (Entity, &GlobalTransform, Option<&mut SortTrigger>),
        (With<Camera>, With<GaussianCamera>),
    >,
    sort_config: Res<SortConfig>,
    mut camera_order: Local<Vec<Entity>>,
) {
    camera_order.clear();
    camera_order.extend(cameras.iter().map(|(entity, ..)| entity));
    camera_order.sort_unstable_by_key(|entity| entity.to_bits());
    for (camera_index, entity) in camera_order.iter().copied().enumerate() {
        let (_, camera_transform, trigger) = cameras.get_mut(entity).unwrap();
        let camera_depth = -camera_transform.to_matrix().inverse().transpose().z_axis;
        let Some(mut sort_trigger) = trigger else {
            commands.entity(entity).insert(SortTrigger {
                camera_index,
                needs_sort: true,
                last_camera_depth: camera_depth,
                last_sort_time: Some(Instant::now()),
                ..default()
            });
            continue;
        };
        if sort_trigger.camera_index != camera_index {
            sort_trigger.camera_index = camera_index;
            sort_trigger.needs_sort = true;
        }
        match sort_trigger.last_sort_time.as_ref() {
            None => {
                sort_trigger.needs_sort = true;
                sort_trigger.last_camera_depth = camera_depth;
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

        let camera_movement = sort_trigger.last_camera_depth != camera_depth;

        if camera_movement {
            sort_trigger.needs_sort = true;
            sort_trigger.last_sort_time = Some(Instant::now());
            sort_trigger.last_camera_depth = camera_depth;
        }
    }
}

/// Ascending unsigned keys for descending signed depth; matches helpers.wgsl.
fn depth_sort_key(depth: f32) -> u32 {
    if !depth.is_finite() {
        return u32::MAX;
    }
    let bits = if depth == 0.0 { 0 } else { depth.to_bits() };
    if depth < 0.0 {
        bits
    } else {
        !(bits ^ 0x8000_0000)
    }
}

/// Input invalidation may require a sort during the camera's throttle interval.
/// Every such sort uses the current camera, without changing when camera motion
/// alone is permitted to request another sort.
#[cfg(any(feature = "sort_std", feature = "sort_rayon"))]
#[allow(clippy::type_complexity)]
fn refresh_cpu_sort_camera_depth(
    mut cameras: Query<(&GlobalTransform, &mut SortTrigger), (With<Camera>, With<GaussianCamera>)>,
) {
    for (transform, mut trigger) in &mut cameras {
        if trigger.needs_sort {
            trigger.last_camera_depth = -transform.to_matrix().inverse().transpose().z_axis;
        }
    }
}

fn finish_cpu_sort(mut cameras: Query<&mut SortTrigger, With<GaussianCamera>>) {
    for mut trigger in &mut cameras {
        if !trigger.sorted_this_frame && !trigger.pending_assets {
            continue;
        }
        if trigger.sorted_this_frame && !trigger.pending_assets {
            trigger.needs_sort = false;
        }
        trigger.sorted_this_frame = false;
        trigger.pending_assets = false;
    }
}

/// Camera motion is only one source of stale CPU keys. Track the flat cloud's
/// ECS inputs and asset events once for both CPU backends; candidate draws own
/// separate sort storage and never enter this query.
#[cfg(any(feature = "sort_std", feature = "sort_rayon"))]
#[allow(clippy::type_complexity)]
fn invalidate_cpu_sort_inputs<R: PlanarSync>(
    mut events: MessageReader<AssetEvent<R::PlanarType>>,
    clouds: Query<
        (
            &R::PlanarTypeHandle,
            Ref<CloudSettings>,
            Ref<GlobalTransform>,
        ),
        With<SortedEntriesHandle>,
    >,
    mut cameras: Query<&mut SortTrigger, With<GaussianCamera>>,
) {
    let uses_cpu_sort = |settings: &CloudSettings| match settings.sort_mode {
        #[cfg(feature = "sort_std")]
        SortMode::Std => true,
        #[cfg(feature = "sort_rayon")]
        SortMode::Rayon => true,
        _ => false,
    };
    let mut invalidated = clouds.iter().any(|(_, settings, transform)| {
        uses_cpu_sort(&settings) && (settings.is_changed() || transform.is_changed())
    });
    // Drain every event even after finding a dirty input. Otherwise an unrelated
    // event left unread could spuriously invalidate a later held frame.
    for event in events.read() {
        if invalidated {
            continue;
        }
        let id = match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::Removed { id }
            | AssetEvent::LoadedWithDependencies { id } => id,
            AssetEvent::Unused { .. } => continue,
        };
        invalidated = clouds
            .iter()
            .any(|(handle, settings, _)| uses_cpu_sort(&settings) && handle.handle().id() == *id);
    }
    if invalidated {
        for mut trigger in &mut cameras {
            trigger.needs_sort = true;
        }
    }
}

// Bevy injects the independent resources and queries as system parameters.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn auto_insert_sorted_entries<R: PlanarSync>(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_clouds_res: Res<Assets<R::PlanarType>>,
    #[cfg(feature = "lod")] transient_atlases: Option<Res<LodTransientAtlasRegistry>>,
    #[cfg(lod_render_path)] candidates: Query<&crate::stream::render_commit::LodRenderCandidates>,
    mut sorted_entries_res: ResMut<Assets<SortedEntries>>,
    gaussian_clouds: Query<
        (Entity, &R::PlanarTypeHandle, &CloudSettings),
        Without<SortedEntriesHandle>,
    >,
    gaussian_cameras: Query<Entity, (With<Camera>, With<GaussianCamera>)>,
) where
    R::PlanarType: CommonCloud,
{
    let camera_count = gaussian_cameras.iter().len();

    if camera_count == 0 {
        debug!("no gaussian cameras found");
        return;
    }

    for (entity, gaussian_cloud_handle, _settings) in gaussian_clouds.iter() {
        #[cfg(lod_render_path)]
        if candidates
            .get(entity)
            .is_ok_and(|candidate| candidate.candidate_draw_required)
        {
            // Package and external active-set draws own private per-view
            // compaction/sort buffers. A dense fallback is not a valid cut.
            continue;
        }
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

        let Some(required_entry_count) = required_sort_entry_capacity::<R>(
            &gaussian_clouds_res,
            gaussian_cloud_handle,
            #[cfg(feature = "lod")]
            transient_atlases.as_deref(),
        ) else {
            debug!("cloud asset is not loaded");
            continue;
        };

        let sorted_entries =
            sorted_entries_res.add(SortedEntries::new(camera_count, required_entry_count));

        commands
            .entity(entity)
            .insert(SortedEntriesHandle(sorted_entries));
    }
}

#[allow(clippy::too_many_arguments)]
fn update_sorted_entries_sizes<R: PlanarSync>(
    mut commands: Commands,
    gaussian_clouds_res: Res<Assets<R::PlanarType>>,
    #[cfg(feature = "lod")] transient_atlases: Option<Res<LodTransientAtlasRegistry>>,
    #[cfg(lod_render_path)] candidates: Query<&crate::stream::render_commit::LodRenderCandidates>,
    mut sorted_entries_res: ResMut<Assets<SortedEntries>>,
    sorted_entries: Query<(Entity, Ref<R::PlanarTypeHandle>, &SortedEntriesHandle)>,
    gaussian_cameras: Query<Entity, (With<Camera>, With<GaussianCamera>)>,
    mut sort_triggers: Query<&mut SortTrigger>,
) where
    R::PlanarType: CommonCloud,
{
    let camera_count: usize = gaussian_cameras.iter().len();
    let mut invalidated = false;

    for (entity, cloud_handle, sorted_handle) in sorted_entries.iter() {
        invalidated |= cloud_handle.is_changed();
        #[cfg(lod_render_path)]
        if candidates
            .get(entity)
            .is_ok_and(|candidate| candidate.candidate_draw_required)
        {
            sorted_entries_res.remove(sorted_handle);
            commands.entity(entity).remove::<SortedEntriesHandle>();
            continue;
        }
        if camera_count == 0 {
            sorted_entries_res.remove(sorted_handle);
            commands.entity(entity).remove::<SortedEntriesHandle>();
            continue;
        }

        let Some(required_entry_count) = required_sort_entry_capacity::<R>(
            &gaussian_clouds_res,
            &cloud_handle,
            #[cfg(feature = "lod")]
            transient_atlases.as_deref(),
        ) else {
            continue;
        };
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
            let new_entry = SortedEntries::new(camera_count, retained_entry_count);
            let _ = sorted_entries_res.insert(sorted_handle, new_entry);
            invalidated = true;
        }
    }
    if invalidated {
        for mut trigger in &mut sort_triggers {
            trigger.needs_sort = true;
        }
    }
}

#[cfg(feature = "lod")]
type OrphanedSortedEntriesQuery<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static SortedEntriesHandle),
    (
        Without<crate::PlanarGaussian3dHandle>,
        Without<crate::PlanarGaussian4dHandle>,
    ),
>;

#[cfg(feature = "lod")]
/// Releases sort storage left behind when orchestration removes a cloud handle.
/// Both built-in planar handles are excluded so one representation's maintenance
/// can never tear down the other's live storage.
fn cleanup_orphaned_sorted_entries(
    mut commands: Commands,
    mut sorted_entries_res: ResMut<Assets<SortedEntries>>,
    orphaned: OrphanedSortedEntriesQuery<'_, '_>,
) {
    for (entity, sorted_handle) in &orphaned {
        sorted_entries_res.remove(sorted_handle);
        commands.entity(entity).remove::<SortedEntriesHandle>();
    }
}

fn required_sort_entry_capacity<R: PlanarSync>(
    gaussian_clouds: &Assets<R::PlanarType>,
    cloud_handle: &R::PlanarTypeHandle,
    #[cfg(feature = "lod")] transient_atlases: Option<&LodTransientAtlasRegistry>,
) -> Option<usize>
where
    R::PlanarType: CommonCloud,
{
    let dense_count = gaussian_clouds.get(cloud_handle.handle()).map(Planar::len);
    #[cfg(feature = "lod")]
    let count = dense_count.or_else(|| {
        transient_atlases
            .and_then(|atlases| atlases.physical_gaussians(cloud_handle.handle().id().untyped()))
            .and_then(|count| usize::try_from(count).ok())
    });
    #[cfg(not(feature = "lod"))]
    let count = dense_count;
    count
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
}

impl SortedEntries {
    /// Only the live records in a camera's retained-capacity slice are sorted.
    fn camera_entries_mut(&mut self, camera: usize, count: usize) -> Option<&mut [SortEntry]> {
        if camera >= self.camera_count || count > self.entry_count {
            return None;
        }
        let start = camera.checked_mul(self.entry_count)?;
        self.sorted.get_mut(start..start.checked_add(count)?)
    }

    pub fn new(camera_count: usize, entry_count: usize) -> Self {
        let sorted: Vec<SortEntry> = (0..camera_count)
            .flat_map(|_camera_idx| {
                (0..entry_count).map(|idx| SortEntry {
                    key: 1,
                    index: idx as u32,
                })
            })
            .collect();

        SortedEntries {
            camera_count,
            entry_count,
            sorted,
        }
    }
}

impl RenderAsset for GpuSortedEntry {
    type SourceAsset = SortedEntries;
    type Param = (SRes<RenderDevice>, SRes<RenderQueue>);

    fn prepare_asset(
        source: Self::SourceAsset,
        _: AssetId<Self::SourceAsset>,
        (render_device, render_queue): &mut SystemParamItem<Self::Param>,
        _: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let stride = aligned_sort_stride(
            source.entry_count,
            render_device.limits().min_storage_buffer_offset_alignment,
        );
        let Some((camera_stride, bytes)) = stride.and_then(|stride| {
            if source.sorted.len() != source.camera_count.checked_mul(source.entry_count)? {
                return None;
            }
            let bytes = stride.checked_mul(source.camera_count as u64)?;
            let last_offset = stride.checked_mul(source.camera_count.saturating_sub(1) as u64)?;
            (bytes <= render_device.limits().max_buffer_size && last_offset <= u64::from(u32::MAX))
                .then_some((stride, bytes))
        }) else {
            return Err(PrepareAssetError::RetryNextUpdate(source));
        };
        let sorted_entry_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("sorted_entry_buffer"),
            size: bytes.max(4),
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        if source.entry_count != 0 {
            for (camera, entries) in source.sorted.chunks_exact(source.entry_count).enumerate() {
                render_queue.write_buffer(
                    &sorted_entry_buffer,
                    camera as u64 * camera_stride,
                    bytemuck::cast_slice(entries),
                );
            }
        }

        Ok(GpuSortedEntry {
            sorted_entry_buffer,
            camera_count: source.camera_count,
            entry_count: source.entry_count,
            camera_stride,
        })
    }

    fn asset_usage(_: &Self::SourceAsset) -> RenderAssetUsages {
        RenderAssetUsages::default()
    }
}

fn aligned_sort_stride(entries: usize, alignment: u32) -> Option<u64> {
    let bytes = u64::try_from(entries)
        .ok()?
        .checked_mul(size_of::<SortEntry>() as u64)?;
    let alignment = u64::from(alignment.max(1));
    bytes
        .checked_add(alignment - 1)
        .map(|bytes| bytes / alignment * alignment)
}

#[derive(Debug, Clone)]
pub struct GpuSortedEntry {
    pub sorted_entry_buffer: Buffer,
    /// Number of camera slices stored in [`Self::sorted_entry_buffer`].
    pub camera_count: usize,
    /// Capacity of one camera slice. Bind groups must use this value rather
    /// than the total count when a cloud handle changes size.
    pub entry_count: usize,
    /// Device-aligned byte stride between camera slices, including retained capacity.
    pub camera_stride: u64,
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;

    #[test]
    fn multi_camera_slots_ignore_render_order_and_follow_removal() {
        let mut app = App::new();
        app.init_resource::<SortConfig>();
        let mut cameras: Vec<_> = [-7, 19, 19]
            .into_iter()
            .map(|order| {
                app.world_mut()
                    .spawn((
                        Camera { order, ..default() },
                        GaussianCamera::default(),
                        GlobalTransform::from_translation(Vec3::Z * 2.0),
                    ))
                    .id()
            })
            .collect();
        cameras.sort_unstable_by_key(|entity| entity.to_bits());
        app.world_mut()
            .run_system_once(update_sort_trigger)
            .unwrap();
        for (index, entity) in cameras.iter().enumerate() {
            let mut trigger = app.world_mut().get_mut::<SortTrigger>(*entity).unwrap();
            assert_eq!(trigger.camera_index, index);
            assert_eq!(trigger.last_camera_depth, Vec4::new(0.0, 0.0, -1.0, 2.0));
            trigger.needs_sort = false;
        }
        app.world_mut().despawn(cameras.remove(0));
        app.world_mut()
            .run_system_once(update_sort_trigger)
            .unwrap();
        for (index, entity) in cameras.into_iter().enumerate() {
            let trigger = app.world().get::<SortTrigger>(entity).unwrap();
            assert_eq!(trigger.camera_index, index);
            assert!(trigger.needs_sort);
        }
    }

    #[test]
    fn camera_depth_sort_handles_rotation_signed_depth_and_invalid_centers() {
        let depths = [f32::MAX, 3.0, 1.001, 1.0, 0.0, -1.0, -f32::MAX];
        let keys = depths.map(depth_sort_key);
        assert!(keys.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(depth_sort_key(-0.0), depth_sort_key(0.0));
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(depth_sort_key(invalid), u32::MAX);
        }
        // Full precision distinguishes depths that the explicit 16-bit option merges.
        assert_ne!(depth_sort_key(1.0001), depth_sort_key(1.0002));
        assert_eq!(depth_sort_key(1.0001) >> 16, depth_sort_key(1.0002) >> 16);

        let mut app = App::new();
        app.insert_resource(SortConfig { period_ms: 0 });
        let camera = app
            .world_mut()
            .spawn((
                Camera::default(),
                GaussianCamera::default(),
                GlobalTransform::IDENTITY,
            ))
            .id();
        app.world_mut()
            .run_system_once(update_sort_trigger)
            .unwrap();
        let initial = app
            .world()
            .get::<SortTrigger>(camera)
            .unwrap()
            .last_camera_depth;
        app.world_mut()
            .get_mut::<SortTrigger>(camera)
            .unwrap()
            .needs_sort = false;
        app.world_mut()
            .entity_mut(camera)
            .insert(GlobalTransform::from(Transform::from_rotation(
                Quat::from_rotation_y(0.5),
            )));
        app.world_mut()
            .run_system_once(update_sort_trigger)
            .unwrap();
        let trigger = app.world().get::<SortTrigger>(camera).unwrap();
        assert!(
            trigger.needs_sort,
            "rotation without translation must update ordering"
        );
        assert_ne!(trigger.last_camera_depth, initial);
    }

    #[test]
    fn multi_camera_retained_slices_and_device_alignment_are_independent() {
        let mut entries = SortedEntries::new(2, 65);
        entries.camera_entries_mut(1, 3).unwrap().reverse();
        assert_eq!(entries.sorted[0].index, 0);
        assert_eq!(entries.sorted[65].index, 2);
        assert_eq!(entries.sorted[68].index, 3);
        assert!(entries.camera_entries_mut(2, 3).is_none());
        assert!(entries.camera_entries_mut(0, 66).is_none());
        assert_eq!(aligned_sort_stride(65, 256), Some(768));
        assert_eq!(aligned_sort_stride(65, 32), Some(544));
        assert_eq!(aligned_sort_stride(0, 256), Some(0));
        assert_eq!(aligned_sort_stride(usize::MAX, 256), None);
    }

    #[cfg(any(feature = "sort_std", feature = "sort_rayon"))]
    #[test]
    fn held_camera_cpu_sort_tracks_transform_asset_and_mode_changes() {
        for mode in [
            #[cfg(feature = "sort_std")]
            SortMode::Std,
            #[cfg(feature = "sort_rayon")]
            SortMode::Rayon,
        ] {
            let mut app = App::new();
            app.add_plugins((MinimalPlugins, AssetPlugin::default()))
                .init_asset::<PlanarGaussian3d>()
                .init_asset::<SortedEntries>()
                .insert_resource(SortConfig { period_ms: 60_000 })
                .add_systems(
                    Update,
                    (
                        update_sort_trigger,
                        auto_insert_sorted_entries::<Gaussian3d>,
                        update_sorted_entries_sizes::<Gaussian3d>,
                        invalidate_cpu_sort_inputs::<Gaussian3d>,
                        refresh_cpu_sort_camera_depth,
                    )
                        .chain()
                        .before(CpuSort),
                )
                .add_systems(Update, finish_cpu_sort.after(CpuSort));
            #[cfg(feature = "sort_std")]
            app.add_plugins(std_sort::StdSortPlugin::<Gaussian3d>::default());
            #[cfg(feature = "sort_rayon")]
            app.add_plugins(rayon::RayonSortPlugin::<Gaussian3d>::default());
            let records = [-1.0, -3.0].map(|z| {
                let mut gaussian = Gaussian3d::default();
                gaussian.position_visibility.position = [0.0, 0.0, z];
                gaussian.position_visibility.visibility = 1.0;
                gaussian
            });
            let source = app
                .world_mut()
                .resource_mut::<Assets<PlanarGaussian3d>>()
                .add(PlanarGaussian3d::from(records.to_vec()));
            let cloud = app
                .world_mut()
                .spawn((
                    PlanarGaussian3dHandle(source.clone()),
                    CloudSettings {
                        sort_mode: mode.clone(),
                        ..default()
                    },
                    GlobalTransform::IDENTITY,
                ))
                .id();
            let camera = app
                .world_mut()
                .spawn((
                    Camera::default(),
                    GaussianCamera::default(),
                    GlobalTransform::IDENTITY,
                ))
                .id();
            app.update();
            app.update(); // Flush the source's initial Added event.
            let sorted = app
                .world()
                .get::<SortedEntriesHandle>(cloud)
                .unwrap()
                .0
                .clone();
            let order = |app: &App| {
                let assets = app.world().resource::<Assets<SortedEntries>>();
                let entries = &assets.get(&sorted).unwrap().sorted;
                [entries[0].index, entries[1].index]
            };
            assert_eq!(order(&app), [1, 0], "{mode:?}");
            app.world_mut()
                .entity_mut(cloud)
                .insert(GlobalTransform::from(Transform::from_rotation(
                    Quat::from_rotation_y(std::f32::consts::PI),
                )));
            app.update();
            assert_eq!(
                order(&app),
                [0, 1],
                "cloud rotation must invalidate {mode:?}"
            );
            {
                let mut assets = app.world_mut().resource_mut::<Assets<PlanarGaussian3d>>();
                let mut asset = assets.get_mut(&source).unwrap();
                asset.position_visibility[0].position[2] = -4.0;
                asset.position_visibility[1].position[2] = -1.0;
            }
            app.update();
            app.update(); // Asset mutations publish their typed event after Update.
            assert_eq!(
                order(&app),
                [1, 0],
                "same-size positions must invalidate {mode:?}"
            );
            app.world_mut()
                .resource_mut::<Assets<PlanarGaussian3d>>()
                .get_mut(&source)
                .unwrap()
                .position_visibility[1]
                .visibility = 0.0;
            app.update();
            app.update();
            assert_eq!(order(&app), [0, 1], "visibility must invalidate {mode:?}");

            app.world_mut()
                .get_mut::<CloudSettings>(cloud)
                .unwrap()
                .sort_mode = SortMode::None;
            app.update();
            app.world_mut()
                .resource_mut::<Assets<PlanarGaussian3d>>()
                .get_mut(&source)
                .unwrap()
                .position_visibility[1]
                .visibility = 1.0;
            app.update();
            app.update();
            assert_eq!(order(&app), [0, 1]);
            app.world_mut()
                .get_mut::<CloudSettings>(cloud)
                .unwrap()
                .sort_mode = mode.clone();
            app.update();
            assert_eq!(
                order(&app),
                [1, 0],
                "enabling CPU sort must refresh held ordering"
            );
            assert!(!app.world().get::<SortTrigger>(camera).unwrap().needs_sort);

            // A sentinel exposes unnecessary re-sorting on unchanged frames.
            app.world_mut()
                .resource_mut::<Assets<SortedEntries>>()
                .get_mut(&sorted)
                .unwrap()
                .sorted[0]
                .index = 99;
            app.update();
            app.update();
            assert_eq!(order(&app), [99, 0], "held inputs must not re-sort");

            app.world_mut().resource_mut::<SortConfig>().period_ms = 60_000;
            app.world_mut()
                .get_mut::<SortTrigger>(camera)
                .unwrap()
                .last_sort_time = Some(Instant::now());
            app.world_mut()
                .entity_mut(camera)
                .insert(GlobalTransform::from(Transform::from_rotation(
                    Quat::from_rotation_y(std::f32::consts::PI),
                )));
            app.update();
            assert_eq!(order(&app), [99, 0], "camera-only motion remains throttled");
            app.world_mut()
                .resource_mut::<Assets<PlanarGaussian3d>>()
                .get_mut(&source)
                .unwrap()
                .position_visibility[0]
                .position[2] = -5.0;
            app.update();
            app.update();
            assert_eq!(
                order(&app),
                [0, 1],
                "asset invalidation must use the rotated camera immediately"
            );

            let other_source = app
                .world_mut()
                .resource_mut::<Assets<PlanarGaussian3d>>()
                .add(PlanarGaussian3d::from(records.to_vec()));
            app.world_mut().spawn((
                PlanarGaussian3dHandle(other_source.clone()),
                CloudSettings {
                    sort_mode: mode,
                    ..default()
                },
                GlobalTransform::IDENTITY,
            ));
            app.update();
            app.update();
            let removed = app
                .world_mut()
                .resource_mut::<Assets<PlanarGaussian3d>>()
                .remove(other_source.id())
                .unwrap();
            app.update();
            app.update();
            assert!(!app.world().get::<SortTrigger>(camera).unwrap().needs_sort);
            app.world_mut()
                .resource_mut::<Assets<SortedEntries>>()
                .get_mut(&sorted)
                .unwrap()
                .sorted[0]
                .index = 99;
            app.update();
            app.update();
            assert_eq!(
                order(&app),
                [99, 1],
                "removed source must not repeatedly sort resident clouds"
            );
            app.world_mut()
                .resource_mut::<Assets<PlanarGaussian3d>>()
                .insert(other_source.id(), removed)
                .unwrap();
            app.update();
            app.update();
            assert_eq!(
                order(&app),
                [0, 1],
                "reloaded source event must wake CPU sorting"
            );
        }
    }

    use crate::gaussian::formats::planar_3d::{
        Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle,
    };
    use crate::gaussian::formats::planar_4d::{
        Gaussian4d, PlanarGaussian4d, PlanarGaussian4dHandle,
    };
    #[cfg(feature = "lod")]
    use crate::stream::atlas_upload::LodTransientAtlas;

    #[cfg(feature = "lod")]
    #[derive(Resource)]
    struct PendingCloudHandle(Handle<PlanarGaussian3d>);

    #[cfg(feature = "lod")]
    #[derive(Resource, Default)]
    struct FailPackageHandle(bool);

    #[cfg(feature = "lod")]
    fn apply_pending_cloud_handle(
        pending: Res<PendingCloudHandle>,
        mut clouds: Query<&mut PlanarGaussian3dHandle>,
    ) {
        for mut handle in &mut clouds {
            *handle = PlanarGaussian3dHandle(pending.0.clone());
        }
    }

    #[cfg(feature = "lod")]
    fn insert_pending_cloud_handle(
        mut commands: Commands,
        pending: Res<PendingCloudHandle>,
        clouds: Query<Entity, (With<CloudSettings>, Without<PlanarGaussian3dHandle>)>,
    ) {
        for cloud in &clouds {
            commands
                .entity(cloud)
                .insert(PlanarGaussian3dHandle(pending.0.clone()));
        }
    }

    #[cfg(feature = "lod")]
    fn remove_failed_package_handle(
        failure: Res<FailPackageHandle>,
        mut commands: Commands,
        clouds: Query<Entity, With<PlanarGaussian3dHandle>>,
    ) {
        if !failure.0 {
            return;
        }
        for cloud in &clouds {
            commands.entity(cloud).remove::<PlanarGaussian3dHandle>();
        }
    }

    #[cfg(lod_render_path)]
    fn required_candidate_sort_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(AssetPlugin::default())
            .init_asset::<PlanarGaussian3d>()
            .init_asset::<SortedEntries>()
            .init_resource::<LodTransientAtlasRegistry>()
            .add_systems(
                Update,
                (
                    auto_insert_sorted_entries::<Gaussian3d>,
                    update_sorted_entries_sizes::<Gaussian3d>,
                    cleanup_orphaned_sorted_entries,
                )
                    .chain(),
            );
        app
    }

    #[cfg(lod_render_path)]
    #[test]
    fn required_candidate_never_allocates_dense_sort_for_large_transient_atlas() {
        use crate::stream::render_commit::LodRenderCandidates;

        let mut app = required_candidate_sort_app();
        let atlas = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .reserve_handle();
        let owner = LodTransientAtlas::new(100_000_000).unwrap();
        app.world_mut()
            .resource_mut::<LodTransientAtlasRegistry>()
            .register(atlas.id(), atlas.id(), 1, &owner)
            .unwrap();
        let cloud = app
            .world_mut()
            .spawn((
                PlanarGaussian3dHandle(atlas),
                CloudSettings::default(),
                LodRenderCandidates {
                    candidate_draw_required: true,
                    ..default()
                },
            ))
            .id();
        for _ in 0..3 {
            app.world_mut()
                .spawn((Camera::default(), GaussianCamera::default()));
            app.update();
            assert!(app.world().get::<SortedEntriesHandle>(cloud).is_none());
            assert_eq!(app.world().resource::<Assets<SortedEntries>>().len(), 0);
            assert_eq!(owner.materialized_slot_count().unwrap(), 0);
        }
    }

    #[cfg(lod_render_path)]
    #[test]
    fn required_candidate_releases_dense_high_water_and_restores_flat_multiview_storage() {
        use crate::stream::render_commit::LodRenderCandidates;

        let mut app = required_candidate_sort_app();
        let flat = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(vec![Gaussian3d::default(); 4]));
        let atlas = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .reserve_handle();
        let owner = LodTransientAtlas::new(130).unwrap();
        app.world_mut()
            .resource_mut::<LodTransientAtlasRegistry>()
            .register(atlas.id(), flat.id(), 1, &owner)
            .unwrap();
        let cloud = app
            .world_mut()
            .spawn((PlanarGaussian3dHandle(atlas), CloudSettings::default()))
            .id();
        app.world_mut()
            .spawn((Camera::default(), GaussianCamera::default()));
        app.update();
        let old_sorted = app
            .world()
            .get::<SortedEntriesHandle>(cloud)
            .unwrap()
            .0
            .clone();
        assert_eq!(
            app.world()
                .resource::<Assets<SortedEntries>>()
                .get(&old_sorted)
                .unwrap()
                .entry_count,
            130
        );
        app.world_mut()
            .entity_mut(cloud)
            .insert(LodRenderCandidates {
                candidate_draw_required: true,
                ..default()
            });
        app.update();
        assert!(app.world().get::<SortedEntriesHandle>(cloud).is_none());
        assert!(
            app.world()
                .resource::<Assets<SortedEntries>>()
                .get(&old_sorted)
                .is_none()
        );
        assert_eq!(app.world().resource::<Assets<SortedEntries>>().len(), 0);

        app.world_mut()
            .entity_mut(cloud)
            .remove::<LodRenderCandidates>()
            .insert(PlanarGaussian3dHandle(flat));
        app.world_mut()
            .spawn((Camera::default(), GaussianCamera::default()));
        app.update();
        let restored = app.world().get::<SortedEntriesHandle>(cloud).unwrap();
        assert_ne!(restored.0.id(), old_sorted.id());
        let restored = app
            .world()
            .resource::<Assets<SortedEntries>>()
            .get(restored)
            .unwrap();
        assert_eq!(restored.camera_count, 2);
        assert_eq!(
            restored.entry_count, 4,
            "transient high-water must not follow a fresh flat allocation"
        );
        assert_eq!(restored.sorted.len(), 8);
        assert_eq!(app.world().resource::<Assets<SortedEntries>>().len(), 1);
    }

    #[cfg(feature = "lod")]
    #[test]
    fn reserved_transient_cloud_gets_sized_sort_storage_without_dense_asset() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(AssetPlugin::default())
            .init_asset::<PlanarGaussian3d>()
            .init_asset::<SortedEntries>()
            .init_resource::<LodTransientAtlasRegistry>()
            .add_systems(
                Update,
                (
                    auto_insert_sorted_entries::<Gaussian3d>,
                    update_sorted_entries_sizes::<Gaussian3d>,
                )
                    .chain(),
            )
            .add_systems(
                PostUpdate,
                insert_pending_cloud_handle.in_set(GaussianLodPackageUpdate),
            )
            .add_systems(
                PostUpdate,
                (
                    auto_insert_sorted_entries::<Gaussian3d>,
                    update_sorted_entries_sizes::<Gaussian3d>,
                )
                    .chain()
                    .after(GaussianLodPackageUpdate),
            );

        let atlas = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .reserve_handle();
        let transient = LodTransientAtlas::new(65).unwrap();
        app.world_mut()
            .resource_mut::<LodTransientAtlasRegistry>()
            .register(atlas.id(), atlas.id(), 1, &transient)
            .unwrap();
        app.insert_resource(PendingCloudHandle(atlas.clone()));
        assert!(
            app.world()
                .resource::<Assets<PlanarGaussian3d>>()
                .get(&atlas)
                .is_none(),
            "the sparse transient path must not create a dense main-world cloud"
        );

        let cloud = app.world_mut().spawn(CloudSettings::default()).id();
        app.world_mut()
            .spawn((Camera::default(), GaussianCamera::default()));

        app.update();

        let sorted_handle = app
            .world()
            .get::<SortedEntriesHandle>(cloud)
            .expect("a live transient cloud receives sort storage");
        let sorted = app
            .world()
            .resource::<Assets<SortedEntries>>()
            .get(sorted_handle)
            .expect("the transient cloud sort storage remains live");
        assert_eq!(sorted.camera_count, 1);
        assert_eq!(sorted.entry_count, 65);

        let old_atlas = app
            .world()
            .get::<PlanarGaussian3dHandle>(cloud)
            .unwrap()
            .0
            .id();
        assert!(
            app.world_mut()
                .resource_mut::<LodTransientAtlasRegistry>()
                .unregister(old_atlas)
        );
        drop(transient);
        let replacement = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .reserve_handle();
        let replacement_transient = LodTransientAtlas::new(130).unwrap();
        app.world_mut()
            .resource_mut::<LodTransientAtlasRegistry>()
            .register(
                replacement.id(),
                replacement.id(),
                1,
                &replacement_transient,
            )
            .unwrap();
        app.world_mut()
            .entity_mut(cloud)
            .insert(PlanarGaussian3dHandle(replacement));

        app.update();

        let sorted_handle = app.world().get::<SortedEntriesHandle>(cloud).unwrap();
        let sorted = app
            .world()
            .resource::<Assets<SortedEntries>>()
            .get(sorted_handle)
            .unwrap();
        assert_eq!(sorted.camera_count, 1);
        assert_eq!(sorted.entry_count, 130);
    }

    #[test]
    fn sort_storage_is_recreated_after_the_last_camera_returns() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(AssetPlugin::default())
            .init_asset::<PlanarGaussian3d>()
            .init_asset::<SortedEntries>()
            .add_systems(
                Update,
                (
                    auto_insert_sorted_entries::<Gaussian3d>,
                    update_sorted_entries_sizes::<Gaussian3d>,
                )
                    .chain(),
            );

        let cloud_asset = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(vec![Gaussian3d::default(); 65]));
        let cloud = app
            .world_mut()
            .spawn((
                PlanarGaussian3dHandle(cloud_asset),
                CloudSettings::default(),
            ))
            .id();
        let camera = app
            .world_mut()
            .spawn((Camera::default(), GaussianCamera::default()))
            .id();

        app.update();
        let first_sorted = app
            .world()
            .get::<SortedEntriesHandle>(cloud)
            .expect("a live camera creates sort storage")
            .0
            .clone();
        assert_eq!(
            app.world()
                .resource::<Assets<SortedEntries>>()
                .get(&first_sorted)
                .unwrap()
                .entry_count,
            65
        );

        assert!(app.world_mut().despawn(camera));
        app.update();
        assert!(app.world().get::<SortedEntriesHandle>(cloud).is_none());
        assert!(
            app.world()
                .resource::<Assets<SortedEntries>>()
                .get(&first_sorted)
                .is_none(),
            "the zero-camera lifecycle must release the old sort asset"
        );

        app.world_mut()
            .spawn((Camera::default(), GaussianCamera::default()));
        app.update();
        let recreated = app
            .world()
            .get::<SortedEntriesHandle>(cloud)
            .expect("camera recreation must recreate sort storage");
        let recreated = app
            .world()
            .resource::<Assets<SortedEntries>>()
            .get(recreated)
            .expect("the recreated sort handle must address a live asset");
        assert_eq!(recreated.camera_count, 1);
        assert_eq!(recreated.entry_count, 65);
    }

    #[cfg(feature = "lod")]
    #[test]
    fn recreated_transient_sort_storage_uses_the_current_registry_capacity() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(AssetPlugin::default())
            .init_asset::<PlanarGaussian3d>()
            .init_asset::<SortedEntries>()
            .init_resource::<LodTransientAtlasRegistry>()
            .add_systems(
                Update,
                (
                    auto_insert_sorted_entries::<Gaussian3d>,
                    update_sorted_entries_sizes::<Gaussian3d>,
                )
                    .chain(),
            );

        let first_atlas = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .reserve_handle();
        let first_owner = LodTransientAtlas::new(65).unwrap();
        app.world_mut()
            .resource_mut::<LodTransientAtlasRegistry>()
            .register(first_atlas.id(), first_atlas.id(), 1, &first_owner)
            .unwrap();
        let cloud = app
            .world_mut()
            .spawn((
                PlanarGaussian3dHandle(first_atlas.clone()),
                CloudSettings::default(),
            ))
            .id();
        let camera = app
            .world_mut()
            .spawn((Camera::default(), GaussianCamera::default()))
            .id();

        app.update();
        let first_sorted = app
            .world()
            .get::<SortedEntriesHandle>(cloud)
            .expect("the first transient atlas receives sort storage")
            .0
            .clone();
        assert_eq!(
            app.world()
                .resource::<Assets<SortedEntries>>()
                .get(&first_sorted)
                .unwrap()
                .entry_count,
            65
        );

        assert!(app.world_mut().despawn(camera));
        app.update();
        assert!(app.world().get::<SortedEntriesHandle>(cloud).is_none());

        assert!(
            app.world_mut()
                .resource_mut::<LodTransientAtlasRegistry>()
                .unregister(first_atlas.id())
        );
        drop(first_owner);
        let replacement = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .reserve_handle();
        let replacement_owner = LodTransientAtlas::new(130).unwrap();
        app.world_mut()
            .resource_mut::<LodTransientAtlasRegistry>()
            .register(replacement.id(), replacement.id(), 1, &replacement_owner)
            .unwrap();
        app.world_mut()
            .entity_mut(cloud)
            .insert(PlanarGaussian3dHandle(replacement));
        app.world_mut()
            .spawn((Camera::default(), GaussianCamera::default()));

        app.update();
        let recreated = app
            .world()
            .get::<SortedEntriesHandle>(cloud)
            .expect("the replacement transient atlas receives recreated sort storage");
        let recreated = app
            .world()
            .resource::<Assets<SortedEntries>>()
            .get(recreated)
            .expect("the replacement sort asset remains live");
        assert_eq!(recreated.camera_count, 1);
        assert_eq!(recreated.entry_count, 130);
    }

    #[test]
    fn reversed_representation_plugin_order_registers_both_sort_lifecycles() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(AssetPlugin::default())
            .add_plugins(bevy::render::sync_world::SyncWorldPlugin)
            .init_asset::<PlanarGaussian3d>()
            .init_asset::<PlanarGaussian4d>()
            .init_asset::<Shader>()
            .add_plugins((
                SortPlugin::<Gaussian4d>::default(),
                SortPlugin::<Gaussian3d>::default(),
            ));

        let small_3d = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(vec![Gaussian3d::default(); 4]));
        let large_3d = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(vec![Gaussian3d::default(); 65]));
        let small_4d = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian4d>>()
            .add(PlanarGaussian4d::from(vec![Gaussian4d::default(); 9]));
        let large_4d = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian4d>>()
            .add(PlanarGaussian4d::from(vec![Gaussian4d::default(); 130]));
        let cloud_3d = app
            .world_mut()
            .spawn((
                PlanarGaussian3dHandle(small_3d),
                CloudSettings::default(),
                GlobalTransform::IDENTITY,
            ))
            .id();
        let cloud_4d = app
            .world_mut()
            .spawn((
                PlanarGaussian4dHandle(small_4d),
                CloudSettings::default(),
                GlobalTransform::IDENTITY,
            ))
            .id();
        let camera = app
            .world_mut()
            .spawn((
                Camera::default(),
                GaussianCamera::default(),
                GlobalTransform::IDENTITY,
            ))
            .id();

        app.update();
        app.world_mut()
            .entity_mut(cloud_3d)
            .insert(PlanarGaussian3dHandle(large_3d));
        app.world_mut()
            .entity_mut(cloud_4d)
            .insert(PlanarGaussian4dHandle(large_4d));
        app.update();

        let sort_assets = app.world().resource::<Assets<SortedEntries>>();
        assert_eq!(
            sort_assets
                .get(app.world().get::<SortedEntriesHandle>(cloud_3d).unwrap())
                .unwrap()
                .entry_count,
            65,
            "3D must resize even when its sort plugin is registered second"
        );
        assert_eq!(
            sort_assets
                .get(app.world().get::<SortedEntriesHandle>(cloud_4d).unwrap())
                .unwrap()
                .entry_count,
            130,
            "4D must retain its independently registered resize lifecycle"
        );

        assert!(app.world_mut().despawn(camera));
        app.update();
        assert!(app.world().get::<SortedEntriesHandle>(cloud_3d).is_none());
        assert!(app.world().get::<SortedEntriesHandle>(cloud_4d).is_none());

        app.world_mut().spawn((
            Camera::default(),
            GaussianCamera::default(),
            GlobalTransform::IDENTITY,
        ));
        app.update();
        let sort_assets = app.world().resource::<Assets<SortedEntries>>();
        assert_eq!(
            sort_assets
                .get(app.world().get::<SortedEntriesHandle>(cloud_3d).unwrap())
                .unwrap()
                .entry_count,
            65
        );
        assert_eq!(
            sort_assets
                .get(app.world().get::<SortedEntriesHandle>(cloud_4d).unwrap())
                .unwrap()
                .entry_count,
            130
        );
    }

    #[cfg(feature = "lod")]
    #[test]
    fn failed_package_handle_cleanup_releases_orphaned_sort_storage() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(AssetPlugin::default())
            .add_plugins(bevy::render::sync_world::SyncWorldPlugin)
            .init_asset::<PlanarGaussian3d>()
            .init_asset::<Shader>()
            .init_resource::<FailPackageHandle>()
            .add_plugins(SortPlugin::<Gaussian3d>::default())
            .add_systems(
                PostUpdate,
                remove_failed_package_handle.in_set(GaussianLodPackageUpdate),
            );

        let cloud_asset = app
            .world_mut()
            .resource_mut::<Assets<PlanarGaussian3d>>()
            .add(PlanarGaussian3d::from(vec![Gaussian3d::default(); 65]));
        let cloud = app
            .world_mut()
            .spawn((
                PlanarGaussian3dHandle(cloud_asset),
                CloudSettings::default(),
                GlobalTransform::IDENTITY,
            ))
            .id();
        app.world_mut().spawn((
            Camera::default(),
            GaussianCamera::default(),
            GlobalTransform::IDENTITY,
        ));

        app.update();
        let sorted = app
            .world()
            .get::<SortedEntriesHandle>(cloud)
            .expect("the live package handle receives sort storage")
            .0
            .clone();
        assert!(
            app.world()
                .resource::<Assets<SortedEntries>>()
                .get(&sorted)
                .is_some()
        );

        app.world_mut().resource_mut::<FailPackageHandle>().0 = true;
        app.update();

        assert!(app.world().get::<PlanarGaussian3dHandle>(cloud).is_none());
        assert!(app.world().get::<SortedEntriesHandle>(cloud).is_none());
        assert!(
            app.world()
                .resource::<Assets<SortedEntries>>()
                .get(&sorted)
                .is_none(),
            "a failed package must not retain its atlas-sized CPU/GPU sort asset"
        );
    }

    #[test]
    fn binding_size_rejects_an_older_smaller_entry_asset() {
        assert_eq!(sort_entry_binding_size(256, 2_048), None);
        assert_eq!(
            sort_entry_binding_size(2_048, 2_048),
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
            .add(SortedEntries::new(1, 256));
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
        assert_eq!(entries.entry_count, 2_048);
        assert_eq!(entries.sorted.len(), 2_048);
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
            .add(SortedEntries::new(1, 2_048));
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
        assert_eq!(bypass_entries.entry_count, 2_048);
        assert_eq!(bypass_entries.sorted.len(), 2_048);

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
        assert_eq!(restored_entries.entry_count, 2_048);
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
        assert_eq!(entries.entry_count, 2_048);
        assert_eq!(entries.sorted.len(), 2_048);
        assert!(sort_entry_binding_size(entries.entry_count, 2_048).is_some());
    }
}
