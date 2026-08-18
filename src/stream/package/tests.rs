#![allow(clippy::field_reassign_with_default)]

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
use std::num::NonZeroU32;

use super::platform::validate_native_root;
use super::*;

#[test]
fn package_poll_preserves_backend_failure_category() {
    let poll = map_package_poll(PagePoll::Failed("timeout"), |detail| {
        GaussianLodPackageTransportError::Http(detail.to_owned())
    });
    let PagePoll::Failed(error) = poll else {
        panic!("failed transport poll must remain failed");
    };
    assert!(matches!(
        error,
        GaussianLodPackageTransportError::Http(ref detail) if detail == "timeout"
    ));
    let failure = LodOrchestrationFailure::from(&error);
    assert_eq!(
        failure.code(),
        LodOrchestrationFailureCode::TransportRequestFailed
    );
}
use crate::{
    GaussianLodBuildSettings, LodNodeId,
    gaussian::formats::planar_3d_lod::build_planar_3d_lod,
    io::lod::encode_page,
    stream::{
        cache::AtlasSlot,
        runtime::{LodRuntimeError, LodRuntimeViewId, LodStreamingRuntime},
        transport::MemoryPageTransport,
    },
    testing::LodTestScene,
};

fn sparse_selection_test_plan(
    slot_count: u32,
    gaussians_per_slot: u32,
) -> GaussianLodPackageAtlasPlan {
    GaussianLodPackageAtlasPlan {
        virtual_source_gaussians: u64::from(slot_count) * u64::from(gaussians_per_slot),
        gaussians_per_slot,
        slot_count,
        physical_gaussians: slot_count.checked_mul(gaussians_per_slot).unwrap(),
        physical_bytes: 0,
    }
}

fn sparse_selection_test_range(
    node: u64,
    page: u64,
    slot: u32,
    generation: u32,
    physical_start: u32,
    count: u32,
) -> LodPhysicalRange {
    LodPhysicalRange {
        node: LodNodeId(node),
        page: LodPageId(page),
        slot: AtlasSlot {
            index: slot,
            generation,
        },
        physical_start,
        count,
    }
}

#[test]
fn sparse_atlas_selection_matches_dense_reference_and_preserves_gaps() {
    let plan = sparse_selection_test_plan(3, 8);
    let ranges = [
        sparse_selection_test_range(1, 10, 1, 3, 9, 2),
        sparse_selection_test_range(2, 10, 1, 3, 13, 2),
        sparse_selection_test_range(3, 20, 2, 4, 16, 1),
    ];
    let selection = plan_package_atlas_selection(plan, &ranges).unwrap();

    let mut dense_reference = vec![false; plan.physical_gaussians as usize];
    for range in ranges {
        dense_reference[range.physical_start as usize..range.end().unwrap() as usize].fill(true);
    }
    let mut sparse_result = vec![false; plan.physical_gaussians as usize];
    for intervals in selection.intervals_by_slot.values() {
        for interval in intervals {
            sparse_result[interval.start as usize..interval.end as usize].fill(true);
        }
    }

    assert_eq!(sparse_result, dense_reference);
    assert!(!sparse_result[8]);
    assert!(!sparse_result[11]);
    assert!(!sparse_result[12]);
    assert!(!sparse_result[15]);
    assert_eq!(selection.scratch().slots, 2);
    assert_eq!(selection.scratch().intervals, 3);
}

#[test]
fn sparse_atlas_selection_scratch_is_independent_of_physical_capacity() {
    let plan = sparse_selection_test_plan(1_000_000, 4);
    let ranges = [
        sparse_selection_test_range(1, 10, 7, 1, 28, 2),
        sparse_selection_test_range(2, 10, 7, 1, 30, 2),
        sparse_selection_test_range(3, 20, 999_999, 9, 3_999_999, 1),
    ];
    let selection = plan_package_atlas_selection(plan, &ranges).unwrap();

    assert_eq!(plan.physical_gaussians, 4_000_000);
    assert_eq!(
        selection.scratch(),
        PackageAtlasSelectionScratch {
            slots: 2,
            intervals: 3,
            materializations: 2,
        }
    );
}

#[test]
fn sparse_atlas_selection_rejects_overlap_and_inconsistent_ranges() {
    let plan = sparse_selection_test_plan(2, 8);
    let overlapping = [
        sparse_selection_test_range(1, 10, 0, 1, 1, 3),
        sparse_selection_test_range(2, 10, 0, 1, 3, 2),
    ];
    assert!(matches!(
        plan_package_atlas_selection(plan, &overlapping),
        Err(GaussianLodPackageError::Runtime(
            LodRuntimeError::OverlappingPhysicalRanges {
                previous_end: 4,
                next_start: 3,
            }
        ))
    ));

    let conflicting_generation = [
        sparse_selection_test_range(1, 10, 0, 1, 0, 1),
        sparse_selection_test_range(2, 10, 0, 2, 1, 1),
    ];
    assert!(matches!(
        plan_package_atlas_selection(plan, &conflicting_generation),
        Err(GaussianLodPackageError::ConflictingAtlasSlot { index: 0, .. })
    ));

    let conflicting_page = [
        sparse_selection_test_range(1, 10, 0, 1, 0, 1),
        sparse_selection_test_range(2, 11, 0, 1, 1, 1),
    ];
    assert!(matches!(
        plan_package_atlas_selection(plan, &conflicting_page),
        Err(GaussianLodPackageError::ConflictingAtlasPage { index: 0, .. })
    ));

    let outside_declared_slot = [sparse_selection_test_range(1, 10, 1, 1, 0, 1)];
    assert!(matches!(
        plan_package_atlas_selection(plan, &outside_declared_slot),
        Err(GaussianLodPackageError::RenderCommit(
            LodRenderCommitError::FrontierReferencesUnsynchronizedPage { .. }
        ))
    ));
}
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
use crate::{
    gaussian::{
        formats::{
            planar_3d_chunked::{LodPageEncoding, LodPageKind, LodPageStorage},
            planar_3d_lod::lod_config_fingerprint,
        },
        lod_debug::LodDebugPreset,
    },
    io::lod::{
        LodCodecLimits, LodShardEntry, LodShardIndex, decode_manifest, decode_page,
        decode_page_with_descriptor, encode_lod_shard_index, encode_manifest,
        encode_page_with_encoding, lod_shard_prefix_len,
    },
    stream::{
        preprocess::{LodPagePreprocessError, LodPagePreprocessInput, LodPagePreprocessor},
        render_commit::LOD_RENDER_PREPARED,
        transport::PageRequest,
    },
};
#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
use std::sync::Arc;

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
struct NativeTestPackage {
    root: std::path::PathBuf,
    manifest: crate::GaussianLodManifest,
    source_count: usize,
    omitted_page: Option<LodPageId>,
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
impl Drop for NativeTestPackage {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
struct LocalPackageHttpServer {
    address: std::net::SocketAddr,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    requests: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ranges: RequestedByteRanges,
    worker: Option<std::thread::JoinHandle<()>>,
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
type RequestedByteRanges = std::sync::Arc<std::sync::Mutex<Vec<Option<(u64, u64)>>>>;

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
impl LocalPackageHttpServer {
    fn start(root: std::path::PathBuf) -> Self {
        use std::io::{Read as _, Write as _};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let requests = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let ranges = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let worker_stop = stop.clone();
        let worker_requests = requests.clone();
        let worker_ranges = ranges.clone();
        let worker = std::thread::spawn(move || {
            while !worker_stop.load(std::sync::atomic::Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(_) => break,
                };
                let mut request = [0_u8; 8192];
                let Ok(read) = stream.read(&mut request) else {
                    continue;
                };
                let line = String::from_utf8_lossy(&request[..read]);
                let Some(uri) = line
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                else {
                    continue;
                };
                let relative = uri.trim_start_matches('/');
                if relative.split('/').any(|part| part == "..") {
                    continue;
                }
                let byte_range = line.lines().find_map(|header| {
                    let (name, value) = header.split_once(':')?;
                    if !name.eq_ignore_ascii_case("range") {
                        return None;
                    }
                    let (start, end) = value.trim().strip_prefix("bytes=")?.split_once('-')?;
                    Some((start.parse::<u64>().ok()?, end.parse::<u64>().ok()?))
                });
                worker_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                worker_ranges.lock().unwrap().push(byte_range);
                match std::fs::read(root.join(relative)) {
                    Ok(bytes) => {
                        if let Some((start, end)) = byte_range {
                            let range = usize::try_from(start)
                                .ok()
                                .zip(usize::try_from(end).ok())
                                .filter(|(start, end)| start <= end && *end < bytes.len());
                            if let Some((start, end)) = range {
                                let payload = &bytes[start..=end];
                                let header = format!(
                                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\nETag: \"fixture-v1\"\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                    payload.len(),
                                    bytes.len()
                                );
                                let _ = stream.write_all(header.as_bytes());
                                let _ = stream.write_all(payload);
                            } else {
                                let _ = stream.write_all(
                                    b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nETag: \"fixture-v1\"\r\nConnection: close\r\n\r\n",
                                );
                            }
                        } else {
                            let header = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: \"fixture-v1\"\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                                bytes.len()
                            );
                            let _ = stream.write_all(header.as_bytes());
                            let _ = stream.write_all(&bytes);
                        }
                    }
                    Err(_) => {
                        let _ = stream.write_all(
                            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nETag: \"fixture-v1\"\r\nConnection: close\r\n\r\n",
                        );
                    }
                }
            }
        });
        Self {
            address,
            stop,
            requests,
            ranges,
            worker: Some(worker),
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}/", self.address)
    }
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
impl Drop for LocalPackageHttpServer {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
fn poll_package_transport(
    transport: &mut PackagePageTransport,
    request: PageRequest,
) -> crate::stream::transport::PagePayload {
    let ticket = transport.begin(request).unwrap();
    for _ in 0..10_000 {
        match transport.poll(&ticket) {
            PagePoll::Pending => std::thread::sleep(Duration::from_millis(1)),
            PagePoll::Ready(payload) => return payload,
            PagePoll::Failed(error) => panic!("package transport failed: {error}"),
        }
    }
    panic!("package transport timed out")
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn write_native_test_package(omit_leaf: bool) -> NativeTestPackage {
    write_native_test_package_with_degree(omit_leaf, None)
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn write_native_test_package_with_degree(
    omit_leaf: bool,
    representative_degree: Option<u8>,
) -> NativeTestPackage {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PACKAGE: AtomicU64 = AtomicU64::new(1);
    let unique = NEXT_PACKAGE.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "bevy-gaussian-lod-package-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir(&root).unwrap();
    std::fs::create_dir(root.join("pages")).unwrap();

    let source = LodTestScene::nested_octants(2).cloud();
    let mut built = build_planar_3d_lod(
        &source,
        GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 8,
            support_sigma: 3.0,
        },
    )
    .unwrap();
    let omitted_page = omit_leaf.then(|| {
        built
            .manifest
            .pages
            .iter()
            .find(|page| page.kind == LodPageKind::SourceLeaves)
            .expect("fixture must contain a source leaf")
            .id
    });
    if let Some(degree) = representative_degree {
        built.manifest.build.config_fingerprint =
            lod_config_fingerprint(built.manifest.build.settings, Some(degree));
    }
    let mut encoded_pages = Vec::new();
    for page in &built.pages {
        let descriptor = built
            .manifest
            .pages
            .iter_mut()
            .find(|descriptor| descriptor.id == page.id)
            .unwrap();
        let encoding = if descriptor.kind == LodPageKind::Representatives {
            representative_degree.map_or(LodPageEncoding::F32Planar, |degree| {
                LodPageEncoding::F16Sh { degree }
            })
        } else {
            LodPageEncoding::F32Planar
        };
        let encoded = encode_page_with_encoding(page, encoding).unwrap();
        let canonical = decode_page(&encoded, LodCodecLimits::default()).unwrap();
        descriptor.encoding = encoding;
        descriptor.content_hash = canonical.content_hash();
        if Some(page.id) != omitted_page {
            encoded_pages.push((page.id, encoded));
        }
    }
    encoded_pages.sort_unstable_by_key(|(page_id, _)| *page_id);
    let prefix_len = lod_shard_prefix_len(encoded_pages.len() as u32).unwrap();
    let mut cursor = prefix_len;
    let entries = encoded_pages
        .iter()
        .map(|(page_id, encoded)| {
            let descriptor = built
                .manifest
                .pages
                .iter()
                .find(|descriptor| descriptor.id == *page_id)
                .unwrap();
            let entry = LodShardEntry {
                page_id: *page_id,
                byte_offset: cursor,
                encoded_len: encoded.len() as u64,
                content_hash: descriptor.content_hash,
            };
            cursor += encoded.len() as u64;
            entry
        })
        .collect::<Vec<_>>();
    let shard_uri = "pages/shard-000000.bgslodpack";
    let mut shard = encode_lod_shard_index(&LodShardIndex {
        file_len: cursor,
        entries: entries.clone(),
    })
    .unwrap();
    for (_, encoded) in &encoded_pages {
        shard.extend_from_slice(encoded);
    }
    assert_eq!(shard.len() as u64, cursor);
    std::fs::write(root.join(shard_uri), shard).unwrap();

    for descriptor in &mut built.manifest.pages {
        if Some(descriptor.id) == omitted_page {
            let encoded_len = built
                .pages
                .iter()
                .find(|page| page.id == descriptor.id)
                .map(|page| {
                    encode_page_with_encoding(page, descriptor.encoding)
                        .unwrap()
                        .len() as u64
                })
                .unwrap();
            descriptor.storage = Some(LodPageStorage {
                uri: format!("pages/missing-page-{}.gspage", descriptor.id.0),
                byte_range: None,
                encoded_len,
            });
            continue;
        }
        let entry = entries
            .iter()
            .find(|entry| entry.page_id == descriptor.id)
            .unwrap();
        descriptor.storage = Some(LodPageStorage {
            uri: shard_uri.to_owned(),
            byte_range: Some((entry.byte_offset, entry.encoded_len)),
            encoded_len: entry.encoded_len,
        });
    }
    built.manifest.validate().unwrap();
    let encoded_manifest = encode_manifest(&built.manifest).unwrap();
    std::fs::write(root.join("scene.gsplatlod"), &encoded_manifest).unwrap();
    let manifest = decode_manifest(&encoded_manifest, LodCodecLimits::default()).unwrap();
    assert_eq!(manifest, built.manifest);
    NativeTestPackage {
        root,
        manifest,
        source_count: source.len(),
        omitted_page,
    }
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn package_test_settings(quality: f32) -> GaussianLodSettings {
    let mut settings = GaussianLodSettings::default();
    settings.quality = quality;
    settings.frustum_culling = false;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
    settings.budgets.max_resident_pages = 128;
    settings.budgets.max_pending_requests = 512;
    settings.budgets.max_requests_per_frame = 128;
    settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
    settings
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn package_test_world(
    package: &NativeTestPackage,
    settings: GaussianLodSettings,
    debug_metadata: bool,
    retry_limit: u32,
) -> (World, Entity, Entity, Handle<GaussianLodAsset>) {
    let mut world = World::new();
    world.init_resource::<Assets<GaussianLodAsset>>();
    world.init_resource::<Assets<PlanarGaussian3d>>();
    world.init_resource::<Messages<AssetEvent<GaussianLodAsset>>>();
    let mut config = GaussianLodPackageConfig {
        max_atlas_gaussians: 4096,
        max_atlas_bytes: 64 * 1024 * 1024,
        max_views_per_cloud: 4,
        ..default()
    };
    config.streaming.retry_limit = retry_limit;
    world.insert_resource(config);
    world.init_resource::<GaussianLodPackageManager>();
    world.init_resource::<LodAtlasUploadQueue>();
    let manifest_handle = world
        .resource_mut::<Assets<GaussianLodAsset>>()
        .add(GaussianLodAsset {
            manifest: package.manifest.clone(),
        });
    let mut cloud_settings = CloudSettings::default();
    cloud_settings.sort_mode = crate::sort::SortMode::Radix;
    if debug_metadata {
        cloud_settings
            .lod_debug
            .apply_preset(LodDebugPreset::Residency);
    }
    let cloud = world
        .spawn((
            GaussianLodHandle(manifest_handle.clone()),
            GaussianLodPackageSource::native_directory(package.root.to_string_lossy().into_owned()),
            settings,
            cloud_settings,
            ViewVisibility::VISIBLE,
            GlobalTransform::IDENTITY,
        ))
        .id();
    let camera = world
        .spawn((
            Camera {
                viewport: Some(bevy::camera::Viewport {
                    physical_size: UVec2::new(1280, 720),
                    ..default()
                }),
                ..default()
            },
            Projection::Perspective(default()),
            GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 5.0)),
            crate::GaussianCamera::default(),
        ))
        .id();
    (world, cloud, camera, manifest_handle)
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn run_package_frame(schedule: &mut Schedule, world: &mut World, cloud: Entity) -> usize {
    if let Some(candidates) = world.get::<LodRenderCandidates>(cloud) {
        for candidate in candidates.by_camera.values() {
            if !candidate.failed() && candidate.phase.load(Ordering::Acquire) != LOD_RENDER_ACTIVE {
                candidate
                    .phase
                    .store(LOD_RENDER_PREPARED, Ordering::Release);
            }
        }
    }
    // The real extraction schedule consumes this queue once per frame.
    // Replacing it here models that boundary without constructing a GPU.
    world.insert_resource(LodAtlasUploadQueue::default());
    schedule.run(world);
    std::thread::yield_now();
    world.resource::<LodAtlasUploadQueue>().queued_slot_count()
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
fn drive_package_to_active_count(
    schedule: &mut Schedule,
    world: &mut World,
    cloud: Entity,
    camera: Entity,
    expected: u32,
) -> usize {
    let mut maximum_queued = 0;
    for _ in 0..2048 {
        let queued = run_package_frame(schedule, world, cloud);
        maximum_queued = maximum_queued.max(queued);
        let active = world
            .get::<GaussianLodPackageStatus>(cloud)
            .is_some_and(|status| status.phase == GaussianLodPackagePhase::Active);
        let exact = world
            .get::<LodRenderCandidates>(cloud)
            .and_then(|candidates| candidates.get(camera))
            .is_some_and(|candidate| {
                candidate.frontier().candidate_count() == expected
                    && candidate.phase.load(Ordering::Acquire) == LOD_RENDER_ACTIVE
            });
        if active && exact {
            return maximum_queued;
        }
    }
    panic!(
        "native package did not reach {expected} active Gaussians; status={:?}, candidates={:?}",
        world.get::<GaussianLodPackageStatus>(cloud),
        world
            .get::<LodRenderCandidates>(cloud)
            .and_then(|candidates| candidates.get(camera))
            .map(|candidate| candidate.frontier().candidate_count())
    );
}

#[test]
fn native_roots_reject_url_schemes_while_http_sources_validate() {
    let error = validate_native_root("https://cdn.example/scene/").unwrap_err();
    assert_eq!(
        error,
        GaussianLodPackageError::UnsupportedUrlScheme("https".to_owned())
    );
    assert!(
        package_http_config(
            "https://cdn.example/scene/",
            &GaussianStreamingSettings::default(),
        )
        .is_ok()
    );
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
#[test]
fn two_http_packages_share_one_writer_and_reuse_it_offline() {
    let package = write_native_test_package(false);
    let server = LocalPackageHttpServer::start(package.root.clone());
    let request_count = server.requests.clone();
    let source = GaussianLodPackageSource::url(server.base_url());
    let descriptor = package.manifest.pages.first().unwrap();
    let request = PageRequest {
        page_id: descriptor.id,
        priority: crate::stream::transport::PageRequestPriority::fallback_critical(u32::MAX),
        expected_bytes: descriptor
            .storage
            .as_ref()
            .map(|storage| storage.encoded_len),
        fallback_page: None,
    };
    let mut config = GaussianLodPackageConfig::default();
    config.persistent_cache_root = Some(
        package
            .root
            .join("persistent-cache")
            .to_string_lossy()
            .into_owned(),
    );
    config.persistent_cache_namespace = Some("http-offline-fixture".to_owned());
    config.persistent_cache_max_entries = 32;
    let requested = GaussianStreamingSettings {
        persistent_cache: true,
        retry_limit: 0,
        retry_base_delay_seconds: 0.0,
        max_compressed_cache_bytes: 8 * 1024 * 1024,
        ..default()
    };
    let effective = package_streaming_settings(&requested).unwrap();
    assert!(effective.persistent_cache);

    let mut manager = GaussianLodPackageManager::default();
    let mut first = package_page_transport(
        &package.manifest,
        &source,
        &config,
        &effective,
        &mut manager.caches,
    )
    .unwrap();
    let expected = poll_package_transport(&mut first, request);
    assert_eq!(request_count.load(Ordering::Acquire), 1);
    let (range_start, range_len) = descriptor.storage.as_ref().unwrap().byte_range.unwrap();
    assert_eq!(
        server.ranges.lock().unwrap().as_slice(),
        [Some((range_start, range_start + range_len - 1))],
        "the HTTP package transport must consume the shard range from the manifest"
    );

    let mut second = package_page_transport(
        &package.manifest,
        &source,
        &config,
        &effective,
        &mut manager.caches,
    )
    .unwrap();
    assert!(Arc::ptr_eq(
        first.shared_native_cache_service().unwrap(),
        second.shared_native_cache_service().unwrap(),
    ));
    assert_eq!(manager.caches.len(), 1);

    let mut conflicting = config.clone();
    conflicting.persistent_cache_max_entries += 1;
    assert!(matches!(
        package_page_transport(
            &package.manifest,
            &source,
            &conflicting,
            &effective,
            &mut manager.caches,
        ),
        Err(GaussianLodPackageError::PersistentCacheConfigConflict { .. })
    ));

    // The second package remains cache-backed after the shared origin goes
    // offline and must not issue another request.
    drop(server);
    let actual = poll_package_transport(&mut second, request);
    assert_eq!(actual, expected);
    assert_eq!(request_count.load(Ordering::Acquire), 1);
    drop(first);
    drop(second);
    manager.prune_unused_caches();
    assert!(manager.caches.is_empty());
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
#[test]
fn corrupt_http_package_page_reaches_preprocessor_without_codec_retry() {
    let package = write_native_test_package(false);
    let descriptor = package.manifest.pages.first().unwrap().clone();
    let storage = descriptor.storage.as_ref().unwrap();
    let (range_start, range_len) = storage.byte_range.unwrap();
    let shard_path = package.root.join(&storage.uri);
    let mut shard = std::fs::read(&shard_path).unwrap();
    let last_byte = range_start
        .checked_add(range_len)
        .and_then(|end| end.checked_sub(1))
        .and_then(|index| usize::try_from(index).ok())
        .expect("fixture page range must fit memory");
    shard[last_byte] ^= 0x5a;
    std::fs::write(&shard_path, shard).unwrap();

    let server = LocalPackageHttpServer::start(package.root.clone());
    let request_count = server.requests.clone();
    let source = GaussianLodPackageSource::url(server.base_url());
    let streaming = GaussianStreamingSettings {
        persistent_cache: true,
        retry_limit: 3,
        retry_base_delay_seconds: 0.0,
        ..default()
    };
    let mut config = GaussianLodPackageConfig::default();
    config.persistent_cache_root = Some(
        package
            .root
            .join("corrupt-handoff-cache")
            .to_string_lossy()
            .into_owned(),
    );
    config.persistent_cache_namespace = Some("corrupt-handoff".to_owned());
    let mut manager = GaussianLodPackageManager::default();
    let mut transport = package_page_transport(
        &package.manifest,
        &source,
        &config,
        &streaming,
        &mut manager.caches,
    )
    .unwrap();
    let request = PageRequest {
        page_id: descriptor.id,
        priority: crate::stream::transport::PageRequestPriority::fallback_critical(u32::MAX),
        expected_bytes: Some(storage.encoded_len),
        fallback_page: None,
    };
    let payload = poll_package_transport(&mut transport, request);

    assert_eq!(
        request_count.load(Ordering::Acquire),
        1,
        "package HTTP must not synchronously decode and retry encoded page bytes"
    );
    let mut limits = LodCodecLimits::default();
    limits.max_page_bytes = limits.max_page_bytes.max(storage.encoded_len);
    limits.max_page_gaussians = descriptor.gaussian_count;
    let mut preprocessor = LodPagePreprocessor::new_cooperative_for_tests(1).unwrap();
    preprocessor
        .submit(LodPagePreprocessInput {
            request,
            payload,
            descriptor: descriptor.clone(),
            limits,
            max_encoded_page_bytes: streaming.effective_max_encoded_page_bytes(),
            support_sigma: package.manifest.build.settings.support_sigma,
        })
        .unwrap();
    let full_page_budget = NonZeroU32::new(u32::MAX).unwrap();
    preprocessor.advance(1, full_page_budget);
    preprocessor.advance(2, full_page_budget);
    let output = preprocessor.take_ready(descriptor.id).unwrap();
    assert!(matches!(
        output.result,
        Err(LodPagePreprocessError::Codec(_))
    ));
    assert_eq!(
        request_count.load(Ordering::Acquire),
        1,
        "preprocess rejection must not create a second HTTP retry layer"
    );

    transport.invalidate_cached_page(descriptor.id).unwrap();
    for _ in 0..10_000 {
        if transport.maintain_cache().unwrap() {
            break;
        }
        std::thread::yield_now();
    }
    assert!(transport.maintain_cache().unwrap());
    let _ = poll_package_transport(&mut transport, request);
    assert_eq!(
        request_count.load(Ordering::Acquire),
        2,
        "a preprocess-rejected cache entry must be evicted before retry"
    );
}

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "sort_radix",
    not(feature = "buffer_texture")
))]
#[test]
fn package_runtime_invalidates_rejected_cache_before_bounded_retry() {
    let package = write_native_test_package(false);
    let root_node = package
        .manifest
        .nodes
        .iter()
        .find(|node| node.id == package.manifest.roots[0])
        .unwrap();
    let root_page = root_node.representation.page;
    let descriptor = package
        .manifest
        .pages
        .iter()
        .find(|descriptor| descriptor.id == root_page)
        .unwrap()
        .clone();
    let storage = descriptor.storage.as_ref().unwrap();
    let (range_start, range_len) = storage.byte_range.unwrap();
    let shard_path = package.root.join(&storage.uri);
    let canonical_shard = std::fs::read(&shard_path).unwrap();
    let mut corrupt_shard = canonical_shard.clone();
    let last_byte = range_start
        .checked_add(range_len)
        .and_then(|end| end.checked_sub(1))
        .and_then(|index| usize::try_from(index).ok())
        .expect("fixture page range must fit memory");
    corrupt_shard[last_byte] ^= 0x5a;
    std::fs::write(&shard_path, &corrupt_shard).unwrap();

    let cache_root = package.root.join("preprocess-retry-order-cache");
    let streaming = GaussianStreamingSettings {
        persistent_cache: true,
        retry_limit: 0,
        retry_base_delay_seconds: 0.0,
        ..default()
    };
    let mut cache_config = GaussianLodPackageConfig::default();
    cache_config.persistent_cache_root = Some(cache_root.to_string_lossy().into_owned());
    cache_config.persistent_cache_namespace = Some("preprocess-retry-order".to_owned());
    cache_config.streaming = streaming.clone();
    let source =
        GaussianLodPackageSource::native_directory(package.root.to_string_lossy().into_owned());
    let mut seed_manager = GaussianLodPackageManager::default();
    let mut seed_transport = package_page_transport(
        &package.manifest,
        &source,
        &cache_config,
        &streaming,
        &mut seed_manager.caches,
    )
    .unwrap();
    let request = PageRequest {
        page_id: root_page,
        priority: crate::stream::transport::PageRequestPriority::fallback_critical(u32::MAX),
        expected_bytes: Some(storage.encoded_len),
        fallback_page: None,
    };
    let corrupt_payload = poll_package_transport(&mut seed_transport, request);
    assert!(
        decode_page_with_descriptor(
            &corrupt_payload.bytes,
            &descriptor,
            LodCodecLimits::default(),
        )
        .is_err(),
        "the seeded cache record must pass encoded-cache integrity but fail preprocessing"
    );
    std::fs::write(&shard_path, &canonical_shard).unwrap();

    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    {
        let mut config = world.resource_mut::<GaussianLodPackageConfig>();
        config.persistent_cache_root = cache_config.persistent_cache_root.clone();
        config.persistent_cache_namespace = cache_config.persistent_cache_namespace.clone();
        config.streaming = streaming;
    }
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let mut observed_rejection = false;
    for _ in 0..2048 {
        run_package_frame(&mut schedule, &mut world, cloud);
        let manager = world.resource::<GaussianLodPackageManager>();
        let Some(state) = manager.clouds.get(&cloud) else {
            continue;
        };
        assert_eq!(
            state.runtime_streaming.retry_limit, 0,
            "the regression must exercise the zero ordinary-retry budget"
        );
        let runtime = state.runtime.lock().unwrap();
        if state.preprocess_cache_repairs.contains(&root_page) {
            assert_eq!(
                runtime.page_attempts(root_page),
                None,
                "the cache-repair attempt must remain queued until the next frame"
            );
            assert!(!runtime.is_terminal_failure(root_page));
            observed_rejection = true;
            break;
        }
    }
    assert!(
        observed_rejection,
        "the full package runtime must observe the seeded preprocessing rejection"
    );

    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.manifest.quality.coarsest_gaussian_count as u32,
    );
    let manager = world.resource::<GaussianLodPackageManager>();
    let state = &manager.clouds[&cloud];
    let runtime = state.runtime.lock().unwrap();
    assert!(runtime.page_preprocess_error(root_page).is_none());
    assert!(!runtime.is_terminal_failure(root_page));
    assert!(!state.preprocess_cache_repairs.contains(&root_page));
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn automatic_native_package_bridge_streams_rebuilds_and_cleans_up() {
    let package = write_native_test_package(false);
    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, manifest_handle) =
        package_test_world(&package, settings, true, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let coarse_count = package.manifest.quality.coarsest_gaussian_count as u32;
    let coarse_uploads =
        drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse_count);
    assert!(coarse_uploads > 0);
    let (first_atlas, first_plan) = {
        let manager = world.resource::<GaussianLodPackageManager>();
        let state = &manager.clouds[&cloud];
        let debug = state.debug.as_ref().expect("debug annotations are enabled");
        assert!(
            debug
                .index
                .descriptor(package.manifest.pages[0].id)
                .is_some()
        );
        (state.atlas.clone(), state.plan)
    };
    assert!(first_plan.physical_gaussians <= 4096);
    assert!(first_plan.physical_bytes <= 64 * 1024 * 1024);
    assert_eq!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&first_atlas)
            .unwrap()
            .len(),
        first_plan.physical_gaussians as usize
    );
    assert!(world.get::<LodDebugMetadata>(cloud).is_some());
    assert!(
        world
            .resource::<LodAtlasUploadQueue>()
            .queued_slots()
            .all(|upload| {
                upload.atlas == first_atlas.id()
                    && upload.slot.index < first_plan.slot_count
                    && upload.gaussians_per_slot == first_plan.gaussians_per_slot
            })
    );

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    let exact_uploads = drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.source_count as u32,
    );
    assert!(exact_uploads > 1);
    let status = world.get::<GaussianLodPackageStatus>(cloud).unwrap();
    assert_eq!(status.active_gaussians, package.source_count as u64);
    assert_eq!(status.phase, GaussianLodPackagePhase::Active);

    world.clear_trackers();
    assert_eq!(run_package_frame(&mut schedule, &mut world, cloud), 0);

    // A runtime-structural budget change must retire the old atlas even
    // when the manifest/source/config handles are unchanged.
    let old_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    world
        .get_mut::<GaussianLodSettings>(cloud)
        .unwrap()
        .budgets
        .max_pending_requests -= 1;
    run_package_frame(&mut schedule, &mut world, cloud);
    let structural_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(structural_atlas.id(), old_atlas.id());
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&old_atlas)
            .is_none()
    );

    // A same-ID manifest reload is a new package generation and must not
    // inherit atlas residency or render handshakes from the old asset.
    let reloaded_manifest = package.manifest.clone();
    world
        .resource_mut::<Assets<GaussianLodAsset>>()
        .get_mut_untracked(&manifest_handle)
        .unwrap()
        .manifest = reloaded_manifest;
    world
        .resource_mut::<Messages<AssetEvent<GaussianLodAsset>>>()
        .write(AssetEvent::Modified {
            id: manifest_handle.id(),
        });
    run_package_frame(&mut schedule, &mut world, cloud);
    let reload_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(reload_atlas.id(), structural_atlas.id());
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&structural_atlas)
            .is_none()
    );

    // Removal and re-addition under the same AssetId are also generation
    // changes. The removed package must release its atlas and remain in a
    // loading state until the replacement asset is present.
    let replacement_manifest = package.manifest.clone();
    world
        .resource_mut::<Assets<GaussianLodAsset>>()
        .remove(manifest_handle.id());
    world
        .resource_mut::<Messages<AssetEvent<GaussianLodAsset>>>()
        .write(AssetEvent::Removed {
            id: manifest_handle.id(),
        });
    run_package_frame(&mut schedule, &mut world, cloud);
    assert!(
        world
            .resource::<GaussianLodPackageManager>()
            .clouds
            .is_empty()
    );
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&reload_atlas)
            .is_none()
    );
    assert!(world.get::<PlanarGaussian3dHandle>(cloud).is_none());
    assert_eq!(
        world.get::<GaussianLodPackageStatus>(cloud).unwrap().phase,
        GaussianLodPackagePhase::Loading
    );

    world
        .resource_mut::<Assets<GaussianLodAsset>>()
        .insert(
            manifest_handle.id(),
            GaussianLodAsset {
                manifest: replacement_manifest,
            },
        )
        .expect("removed manifest ID can be reinserted");
    world
        .resource_mut::<Messages<AssetEvent<GaussianLodAsset>>>()
        .write(AssetEvent::Added {
            id: manifest_handle.id(),
        });
    run_package_frame(&mut schedule, &mut world, cloud);
    let readded_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(readded_atlas.id(), reload_atlas.id());
    assert!(
        world
            .resource::<GaussianLodPackageManager>()
            .clouds
            .contains_key(&cloud)
    );

    // Debug metadata is genuinely lazy. Disabling the only metadata user
    // rebuilds without the bounded annotation atlas and removes the ECS
    // component instead of continuing page-level annotation work.
    world
        .get_mut::<CloudSettings>(cloud)
        .unwrap()
        .lod_debug
        .apply_preset(LodDebugPreset::Off);
    run_package_frame(&mut schedule, &mut world, cloud);
    let disabled_atlas = world
        .get::<PlanarGaussian3dHandle>(cloud)
        .unwrap()
        .handle()
        .clone();
    assert_ne!(disabled_atlas.id(), readded_atlas.id());
    assert!(
        world.resource::<GaussianLodPackageManager>().clouds[&cloud]
            .debug
            .is_none()
    );
    assert!(world.get::<LodDebugMetadata>(cloud).is_none());

    // Package rendering requires the GPU radix compaction path. An
    // incompatible cloud fails visibly and releases package-owned state.
    world.get_mut::<CloudSettings>(cloud).unwrap().sort_mode = crate::sort::SortMode::None;
    run_package_frame(&mut schedule, &mut world, cloud);
    let status = world.get::<GaussianLodPackageStatus>(cloud).unwrap();
    assert_eq!(status.phase, GaussianLodPackagePhase::Failed);
    assert_eq!(
        status.failure.as_ref().map(LodOrchestrationFailure::code),
        Some(LodOrchestrationFailureCode::UnsupportedConfiguration)
    );
    assert!(
        status
            .error_detail()
            .is_some_and(|error| error.contains("UnsupportedSortMode"))
    );
    assert!(
        world
            .resource::<GaussianLodPackageManager>()
            .clouds
            .is_empty()
    );
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&disabled_atlas)
            .is_none()
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn compressed_representative_pages_stream_through_the_canonical_atlas() {
    let representative_degree =
        crate::material::spherical_harmonics::SH_DEGREE.saturating_sub(1) as u8;
    let package = write_native_test_package_with_degree(false, Some(representative_degree));
    assert!(package.manifest.pages.iter().any(|descriptor| {
        descriptor.kind == LodPageKind::Representatives
            && descriptor.encoding
                == LodPageEncoding::F16Sh {
                    degree: representative_degree,
                }
    }));
    assert!(package.manifest.pages.iter().all(|descriptor| {
        descriptor.kind != LodPageKind::SourceLeaves
            || descriptor.encoding == LodPageEncoding::F32Planar
    }));

    let settings = package_test_settings(0.0);
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.manifest.quality.coarsest_gaussian_count as u32,
    );
    let candidates = world.get::<LodRenderCandidates>(cloud).unwrap();
    assert!(
        candidates
            .by_camera
            .values()
            .flat_map(LodRenderCandidate::render_ranges)
            .any(|range| {
                package.manifest.pages.iter().any(|descriptor| {
                    descriptor.id == range.page && descriptor.kind == LodPageKind::Representatives
                })
            })
    );

    world.get_mut::<GaussianLodSettings>(cloud).unwrap().quality = 1.0;
    drive_package_to_active_count(
        &mut schedule,
        &mut world,
        cloud,
        camera,
        package.source_count as u32,
    );
    let candidates = world.get::<LodRenderCandidates>(cloud).unwrap();
    assert!(
        candidates
            .by_camera
            .values()
            .flat_map(LodRenderCandidate::render_ranges)
            .all(|range| {
                package.manifest.pages.iter().any(|descriptor| {
                    descriptor.id == range.page && descriptor.kind == LodPageKind::SourceLeaves
                })
            })
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn native_package_missing_leaf_marks_ancestor_fallback_and_despawn_cleans_atlas() {
    let package = write_native_test_package(true);
    assert!(package.omitted_page.is_some());
    let settings = package_test_settings(1.0);
    let (mut world, cloud, _camera, _) = package_test_world(&package, settings, true, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);

    let atlas = (0..4096)
        .find_map(|_| {
            run_package_frame(&mut schedule, &mut world, cloud);
            let status = world.get::<GaussianLodPackageStatus>(cloud)?;
            let metadata = world.get::<LodDebugMetadata>(cloud)?;
            (status.terminal_failures > 0
                && metadata.records().iter().any(|record| {
                    record.residency_code() == LodDebugResidency::AncestorFallback as u32
                }))
            .then(|| {
                world
                    .get::<PlanarGaussian3dHandle>(cloud)
                    .unwrap()
                    .handle()
                    .clone()
            })
        })
        .unwrap_or_else(|| {
            panic!(
                "missing native leaf did not publish fallback provenance; status={:?}",
                world.get::<GaussianLodPackageStatus>(cloud)
            )
        });
    let status = world.get::<GaussianLodPackageStatus>(cloud).unwrap();
    assert_eq!(status.phase, GaussianLodPackagePhase::Degraded);
    assert!(status.active_gaussians > 0);
    assert!((status.active_gaussians as usize) < package.source_count);

    world.despawn(cloud);
    world.insert_resource(LodAtlasUploadQueue::default());
    schedule.run(&mut world);
    assert!(
        world
            .resource::<GaussianLodPackageManager>()
            .clouds
            .is_empty()
    );
    assert!(
        world
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&atlas)
            .is_none()
    );
    assert_eq!(
        world.resource::<LodAtlasUploadQueue>().queued_slot_count(),
        0
    );
}

#[cfg(all(feature = "sort_radix", not(feature = "buffer_texture")))]
#[test]
fn native_package_atlas_is_intrinsically_atomic_and_rejects_subpage_cap() {
    let package = write_native_test_package(false);
    let mut settings = package_test_settings(0.0);
    let stride = package
        .manifest
        .pages
        .iter()
        .map(|page| page.gaussian_count)
        .max()
        .unwrap();
    let bytes_per_slot = u64::from(stride) * gaussian_3d_gpu_bytes_per_record();
    settings.budgets.max_gpu_upload_bytes_per_commit = bytes_per_slot;
    let (mut world, cloud, camera, _) = package_test_world(&package, settings, false, 0);
    let mut schedule = Schedule::default();
    schedule.add_systems(update_lod_packages);
    let coarse = package.manifest.quality.coarsest_gaussian_count as u32;
    drive_package_to_active_count(&mut schedule, &mut world, cloud, camera, coarse);
    let plan = world.resource::<GaussianLodPackageManager>().clouds[&cloud].plan;
    assert_eq!(plan.slot_count, 1);
    assert_eq!(plan.physical_bytes, bytes_per_slot);
    assert!(world.resource::<LodAtlasUploadQueue>().queued_slot_count() <= 1);

    let mut too_small = package_test_settings(0.0);
    too_small.budgets.max_gpu_upload_bytes_per_commit = bytes_per_slot - 1;
    let config = world.resource::<GaussianLodPackageConfig>();
    assert_eq!(
        GaussianLodPackageAtlasPlan::from_manifest(&package.manifest, &too_small, config),
        Err(GaussianLodPackageError::AtlasCannotFitPage {
            gaussians_per_slot: stride,
            bytes_per_slot,
        })
    );
}

#[test]
fn package_streaming_settings_validate_and_rebuild_signatures_track_config() {
    let required = GaussianStreamingSettings::default();
    assert_eq!(
        package_streaming_settings(&required).unwrap(),
        required.clone()
    );
    let invalid = GaussianStreamingSettings {
        max_concurrent_requests: 0,
        ..required.clone()
    };
    assert!(matches!(
        package_streaming_settings(&invalid),
        Err(GaussianLodPackageError::InvalidStreaming(_))
    ));
    let render_path = validate_package_render_path(&crate::sort::SortMode::default());
    assert_eq!(
        render_path.is_ok(),
        crate::stream::lod_render_path_is_supported()
    );
    if !crate::stream::lod_render_path_is_supported() {
        assert_eq!(
            render_path,
            Err(GaussianLodPackageError::UnsupportedRenderPath(
                LodRenderPathSupportError::UnsupportedBuildConfiguration
            ))
        );
    }
    let cached = GaussianStreamingSettings {
        persistent_cache: true,
        ..required.clone()
    };
    let effective = package_streaming_settings(&cached).unwrap();
    assert!(effective.persistent_cache);

    let manifest = AssetId::<GaussianLodAsset>::default();
    let source = GaussianLodPackageSource::native_directory("scene");
    let config = GaussianLodPackageConfig::default();
    let lod_settings = GaussianLodSettings::default();
    let structural = PackageStructuralSignature::new(&lod_settings);
    let current = PackageBuildSignature {
        manifest,
        source: &source,
        config: &config,
        streaming: &required,
        structural,
        debug_metadata: false,
    };
    assert!(
        current
            == PackageBuildSignature {
                manifest,
                source: &source,
                config: &config,
                streaming: &required,
                structural,
                debug_metadata: false,
            }
    );

    let mut structural_change = config.clone();
    structural_change.max_atlas_gaussians /= 2;
    assert!(
        current
            != PackageBuildSignature {
                manifest,
                source: &source,
                config: &structural_change,
                streaming: &required,
                structural,
                debug_metadata: false,
            }
    );
    let effective_streaming_change = GaussianStreamingSettings {
        retry_limit: required.retry_limit + 1,
        ..required.clone()
    };
    assert!(
        current
            != PackageBuildSignature {
                manifest,
                source: &source,
                config: &config,
                streaming: &effective_streaming_change,
                structural,
                debug_metadata: false,
            }
    );
}

#[test]
fn http_package_has_one_authoritative_retry_budget() {
    let streaming = GaussianStreamingSettings {
        retry_limit: 3,
        ..default()
    };
    let http = package_runtime_streaming_settings(
        &GaussianLodPackageSource::url("https://cdn.example/scene/"),
        &streaming,
    );
    assert_eq!(http.retry_limit, 0);
    let native = package_runtime_streaming_settings(
        &GaussianLodPackageSource::native_directory("scene"),
        &streaming,
    );
    assert_eq!(native.retry_limit, 3);
    // HttpRangePageTransport's request-count regression separately proves
    // its configured retry budget emits only R + 1 bounded attempts.
}

#[test]
fn hundred_million_virtual_source_does_not_scale_physical_allocation() {
    let settings = GaussianLodSettings::default();
    let mut config = GaussianLodPackageConfig::default();
    config.max_atlas_gaussians = 16_384;
    config.max_atlas_bytes = u64::MAX;
    let plan =
        GaussianLodPackageAtlasPlan::from_limits(134_217_728, 4_096, &settings, &config).unwrap();
    assert_eq!(plan.virtual_source_gaussians, 134_217_728);
    assert_eq!(plan.physical_gaussians, 16_384);
    assert_eq!(plan.slot_count, 4);
}

#[test]
fn memory_backed_package_reaches_q0_and_exact_q1_frontiers() {
    let source = LodTestScene::nested_octants(3).cloud();
    let built = build_planar_3d_lod(
        &source,
        GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 8,
            support_sigma: 3.0,
        },
    )
    .unwrap();
    let mut transport = MemoryPageTransport::default();
    for page in &built.pages {
        transport.insert(page.id, encode_page(page).unwrap());
    }
    // Cooperative preprocessing performs a bounded checksum slice and a
    // bounded decode slice on separate application frames. Scale the eventual
    // bound with the physical package instead of assuming the older one-page
    // decode cadence.
    let max_updates = built.manifest.pages.len().saturating_mul(3) + 16;

    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
    settings.budgets.max_resident_pages = 256;
    settings.budgets.max_pending_requests = 512;
    settings.budgets.max_requests_per_frame = 256;
    settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
    let mut streaming = GaussianStreamingSettings::default();
    streaming.persistent_cache = false;
    streaming.retry_limit = 0;
    let mut runtime =
        LodStreamingRuntime::new(built.manifest, transport, &settings, &streaming).unwrap();
    let view = LodView::perspective(Vec3::new(0.0, 0.0, 5.0), 720.0, 1.0, 0.1);

    let drive = |runtime: &mut LodStreamingRuntime<MemoryPageTransport>,
                 settings: &GaussianLodSettings,
                 expected: Option<u32>| {
        let mut last_summary = String::new();
        for _ in 0..max_updates {
            let frame = runtime
                .update_view(LodRuntimeViewId(7), view, settings, &streaming)
                .unwrap();
            match frame.candidate_frontier(settings.max_active_gaussians_u32()) {
                Ok(candidate) => {
                    last_summary = format!(
                        "candidate={} requested={:?} cache={:?} preprocess={:?}",
                        candidate.candidate_count(),
                        frame.frontier().requested_nodes,
                        frame.cache_stats(),
                        frame.preprocess_stats(),
                    );
                    if expected.is_none_or(|count| candidate.candidate_count() == count) {
                        return candidate;
                    }
                }
                Err(error) => {
                    last_summary = format!(
                        "candidate_error={error:?} requested={:?} cache={:?} preprocess={:?}",
                        frame.frontier().requested_nodes,
                        frame.cache_stats(),
                        frame.preprocess_stats(),
                    );
                }
            }
        }
        let terminal = runtime
            .terminal_failures()
            .iter()
            .map(|&page| (page, runtime.page_preprocess_error(page).cloned()))
            .collect::<Vec<_>>();
        panic!(
            "package runtime did not reach the requested complete cut; last={last_summary}; terminal={terminal:?}"
        )
    };

    let coarse = drive(&mut runtime, &settings, None);
    assert!(coarse.candidate_count() > 0);
    assert!((coarse.candidate_count() as usize) < source.len());

    settings.quality = 1.0;
    let exact = drive(&mut runtime, &settings, Some(source.len() as u32));
    assert_eq!(exact.candidate_count() as usize, source.len());
    assert!(exact.candidate_count() > coarse.candidate_count());
}

#[test]
fn corrupt_leaf_retains_a_complete_ancestor_cut() {
    let source = LodTestScene::nested_octants(3).cloud();
    let built = build_planar_3d_lod(
        &source,
        GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 8,
            support_sigma: 3.0,
        },
    )
    .unwrap();
    let root_pages = built
        .manifest
        .roots
        .iter()
        .filter_map(|root| {
            built
                .manifest
                .nodes
                .iter()
                .find(|node| node.id == *root)
                .map(|node| node.representation.page)
        })
        .collect::<BTreeSet<_>>();
    let corrupt_page = built
        .manifest
        .nodes
        .iter()
        .find(|node| node.children.is_empty() && !root_pages.contains(&node.representation.page))
        .map(|node| node.representation.page)
        .expect("fixture must contain a non-root leaf page");
    let mut transport = MemoryPageTransport::default();
    for page in &built.pages {
        let mut bytes = encode_page(page).unwrap();
        if page.id == corrupt_page {
            let last = bytes.last_mut().expect("encoded page is non-empty");
            *last ^= 0x5a;
        }
        transport.insert(page.id, bytes);
    }
    let max_updates = built.manifest.pages.len().saturating_mul(3) + 16;

    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.frustum_culling = false;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
    settings.budgets.max_resident_pages = 256;
    settings.budgets.max_pending_requests = 512;
    settings.budgets.max_requests_per_frame = 256;
    settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
    let mut streaming = GaussianStreamingSettings::default();
    streaming.retry_limit = 0;
    let mut runtime =
        LodStreamingRuntime::new(built.manifest, transport, &settings, &streaming).unwrap();
    let view = LodView::perspective(Vec3::new(0.0, 0.0, 5.0), 720.0, 1.0, 0.1);

    let coarse = (0..max_updates)
        .find_map(|_| {
            runtime
                .update_view(LodRuntimeViewId(17), view, &settings, &streaming)
                .ok()?
                .candidate_frontier(settings.max_active_gaussians_u32())
                .ok()
        })
        .expect("root cut must become resident before refinement");
    settings.quality = 1.0;
    let degraded = (0..max_updates)
        .find_map(|_| {
            let frame = runtime
                .update_view(LodRuntimeViewId(17), view, &settings, &streaming)
                .ok()?;
            runtime.is_terminal_failure(corrupt_page).then(|| {
                frame
                    .candidate_frontier(settings.max_active_gaussians_u32())
                    .expect("ancestor fallback must remain a complete resident cut")
            })
        })
        .expect("corrupt leaf must exhaust its retry budget");
    assert!(runtime.is_terminal_failure(corrupt_page));
    assert!(degraded.candidate_count() >= coarse.candidate_count());
    assert!((degraded.candidate_count() as usize) < source.len());
    assert!(
        degraded
            .physical_ranges()
            .iter()
            .all(|range| range.page != corrupt_page)
    );
}

#[test]
fn pending_multi_camera_churn_rewrites_legacy_atlas_to_exact_root_cut() {
    let source = LodTestScene::nested_octants(3).cloud();
    let built = build_planar_3d_lod(
        &source,
        GaussianLodBuildSettings {
            branching_factor: 4,
            leaf_capacity: 8,
            support_sigma: 3.0,
        },
    )
    .unwrap();
    let manifest = built.manifest.clone();
    let max_updates = manifest.pages.len().saturating_mul(3) + 16;
    let mut transport = MemoryPageTransport::default();
    for page in &built.pages {
        transport.insert(page.id, encode_page(page).unwrap());
    }
    let mut settings = GaussianLodSettings::default();
    settings.quality = 0.0;
    settings.frustum_culling = false;
    settings.budgets.max_active_gaussians = 4096;
    settings.budgets.max_resident_gaussians = 8192;
    settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
    settings.budgets.max_resident_pages = 256;
    settings.budgets.max_pending_requests = 512;
    settings.budgets.max_requests_per_frame = 256;
    settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
    let mut streaming = GaussianStreamingSettings::default();
    streaming.persistent_cache = false;
    streaming.retry_limit = 0;
    let mut runtime =
        LodStreamingRuntime::new(manifest.clone(), transport, &settings, &streaming).unwrap();
    let stride = runtime.atlas_layout().gaussians_per_slot;
    let slot_count = settings.budgets.max_resident_pages;
    let physical_gaussians = slot_count * stride;
    let plan = GaussianLodPackageAtlasPlan {
        virtual_source_gaussians: source.len() as u64,
        gaussians_per_slot: stride,
        slot_count,
        physical_gaussians,
        physical_bytes: u64::from(physical_gaussians) * gaussian_3d_gpu_bytes_per_record(),
    };
    let mut mirror = LodPageAtlasMirror::new(runtime.atlas_layout(), slot_count).unwrap();
    let mut debug = PackageDebugAnnotations {
        atlas: LodDebugAnnotationAtlas::new(slot_count, stride).unwrap(),
        index: LodDebugManifestIndex::new(&manifest).unwrap(),
    };

    let mut drive = |view_id: LodRuntimeViewId,
                     view: LodView,
                     settings: &GaussianLodSettings,
                     expected: Option<u32>| {
        for _ in 0..max_updates {
            let frame = runtime
                .update_view(view_id, view, settings, &streaming)
                .unwrap();
            for &page in frame.completed_pages() {
                let slot = runtime.cache().get(page).unwrap().slot;
                mirror.stage_page(page, slot).unwrap();
            }
            if let Ok(candidate) = frame.candidate_frontier(settings.max_active_gaussians_u32())
                && expected.is_none_or(|count| candidate.candidate_count() == count)
            {
                return candidate;
            }
        }
        panic!("runtime did not reach requested candidate")
    };

    let root = drive(
        PACKAGE_ROOT_FALLBACK_VIEW,
        LodView::perspective(Vec3::ZERO, 1.0, 1.0, 0.1),
        &settings,
        None,
    );
    settings.quality = 1.0;
    let left = drive(
        LodRuntimeViewId(11),
        LodView::perspective(Vec3::new(-4.0, 0.0, 5.0), 720.0, 1.0, 0.1),
        &settings,
        Some(source.len() as u32),
    );
    let right = drive(
        LodRuntimeViewId(12),
        LodView::perspective(Vec3::new(4.0, 0.0, 5.0), 720.0, 1.0, 0.1),
        &settings,
        Some(source.len() as u32),
    );
    assert_eq!(left.candidate_count(), right.candidate_count());

    // Materialize a multi-camera exact cut, then enter pending staging.
    // The CPU fallback must contain exactly the camera-independent root
    // cut, and uploads must scale with old/new visible slots rather than
    // the configured atlas capacity.
    let mut atlas =
        PlanarGaussian3d::from(vec![Gaussian3d::default(); physical_gaussians as usize]);
    let exact_rewrite = rewrite_atlas_to_frontiers(
        &runtime,
        &mut mirror,
        Some(&mut debug),
        plan,
        &[left.clone(), right.clone()],
        &BTreeSet::new(),
        &BTreeMap::new(),
        &mut atlas,
    )
    .unwrap();
    assert!(!exact_rewrite.selected_slots.is_empty());
    assert_eq!(
        exact_rewrite.selection_scratch.slots,
        exact_rewrite.selected_slots.len()
    );
    assert!(
        exact_rewrite.selection_scratch.intervals
            <= left.physical_ranges().len() + right.physical_ranges().len()
    );
    assert!(exact_rewrite.dirty_slots.len() < plan.slot_count as usize);
    let exact_slots = exact_rewrite.selected_slots.clone();
    let exact_atlas_id = AssetId::<PlanarGaussian3d>::default();
    let mut exact_uploads = LodAtlasUploadQueue::default();
    enqueue_package_atlas_uploads(&mut exact_uploads, exact_atlas_id, plan, &exact_rewrite)
        .unwrap();
    assert_eq!(
        exact_uploads.queued_slot_count(),
        exact_rewrite.dirty_slots.len()
    );
    assert!(exact_uploads.queued_slots().all(|upload| {
        upload.slot.generation == exact_rewrite.selected_slots[&upload.slot.index].generation
    }));

    let root_rewrite = rewrite_atlas_to_frontiers(
        &runtime,
        &mut mirror,
        Some(&mut debug),
        plan,
        std::slice::from_ref(&root),
        &BTreeSet::new(),
        &exact_slots,
        &mut atlas,
    )
    .unwrap();
    let actual = atlas
        .iter()
        .enumerate()
        .filter_map(|(index, gaussian)| {
            (gaussian.scale_opacity.opacity != 0.0).then_some(index as u32)
        })
        .collect::<BTreeSet<_>>();
    let expected = root
        .physical_ranges()
        .iter()
        .flat_map(|range| range.physical_start..range.end().unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
    assert_eq!(actual.len() as u32, root.candidate_count());

    let atlas_id = AssetId::<PlanarGaussian3d>::default();
    let mut uploads = LodAtlasUploadQueue::default();
    enqueue_package_atlas_uploads(&mut uploads, atlas_id, plan, &root_rewrite).unwrap();
    let queued = uploads.queued_slots().collect::<Vec<_>>();
    assert_eq!(queued.len(), root_rewrite.dirty_slots.len());
    assert!(queued.len() < plan.slot_count as usize);
    for upload in queued {
        assert_eq!(upload.atlas, atlas_id);
        assert_eq!(upload.gaussians_per_slot, stride);
        assert!(upload.slot.index < plan.slot_count);
        assert_eq!(
            upload.slot.generation,
            root_rewrite
                .selected_slots
                .get(&upload.slot.index)
                .map_or(0, |slot| slot.generation)
        );
        assert!(
            upload.slot.index * stride + stride <= plan.physical_gaussians,
            "queued slot range must stay within the bounded atlas"
        );
    }

    // A slot occupant change must remain dirty even when the replacement uses
    // the same physical slot index.
    let mut previous_generation_slots = root_rewrite.selected_slots.clone();
    for slot in previous_generation_slots.values_mut() {
        slot.generation = if slot.generation == 1 { 2 } else { 1 };
    }
    let generation_rewrite = rewrite_atlas_to_frontiers(
        &runtime,
        &mut mirror,
        Some(&mut debug),
        plan,
        std::slice::from_ref(&root),
        &BTreeSet::new(),
        &previous_generation_slots,
        &mut atlas,
    )
    .unwrap();
    assert_eq!(
        generation_rewrite.selected_slots,
        root_rewrite.selected_slots
    );
    assert_eq!(
        generation_rewrite.dirty_slots.len(),
        generation_rewrite.selected_slots.len()
    );

    let mut churn_uploads = LodAtlasUploadQueue::default();
    let churn_rewrite = rewrite_atlas_to_frontiers(
        &runtime,
        &mut mirror,
        Some(&mut debug),
        plan,
        &[left.clone(), right],
        &BTreeSet::new(),
        &root_rewrite.selected_slots,
        &mut atlas,
    )
    .unwrap();
    enqueue_package_atlas_uploads(&mut churn_uploads, atlas_id, plan, &churn_rewrite).unwrap();
    assert_eq!(
        churn_uploads.queued_slot_count(),
        churn_rewrite.dirty_slots.len()
    );
    assert!(churn_uploads.queued_slot_count() < plan.slot_count as usize);

    let camera = Entity::from_bits(11);
    let mut current = LodRenderCandidates::default();
    current.insert(camera, left);
    current
        .get(camera)
        .unwrap()
        .phase
        .store(LOD_RENDER_ACTIVE, Ordering::Release);
    assert!(package_candidate_set_is_active(&current));
    current.get(camera).unwrap().phase.store(
        crate::stream::render_commit::LOD_RENDER_WAITING,
        Ordering::Release,
    );
    assert!(!package_candidate_set_is_active(&current));
}
