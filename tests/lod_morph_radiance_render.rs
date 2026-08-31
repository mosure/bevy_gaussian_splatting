#[cfg(not(all(
    feature = "headless",
    feature = "testing",
    feature = "lod_build",
    not(target_arch = "wasm32")
)))]
#[test]
fn lod_morph_radiance_render_requires_headless_testing_and_lod_build() {}

#[cfg(all(
    feature = "headless",
    feature = "testing",
    feature = "lod_build",
    not(target_arch = "wasm32")
))]
mod headless {
    use std::{
        collections::{BTreeMap, HashMap, HashSet},
        env, fs,
        mem::size_of,
        path::{Path, PathBuf},
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, Ordering},
        },
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use bevy::{
        app::{AppExit, ScheduleRunnerPlugin},
        camera::{PerspectiveProjection, Projection, RenderTarget, visibility::NoFrustumCulling},
        core_pipeline::tonemapping::Tonemapping,
        prelude::*,
        render::{
            Render, RenderApp, RenderSystems,
            extract_resource::{ExtractResource, ExtractResourcePlugin},
            pipelined_rendering::PipelinedRenderingPlugin,
            render_resource::TextureFormat,
            view::ExtractedView,
            view::screenshot::{Screenshot, ScreenshotCaptured},
        },
        window::ExitCondition,
        winit::WinitPlugin,
    };
    use bevy_gaussian_splatting::{
        CloudSettings, CpuExternalLodBatchPreprocessor, EXTERNAL_LOD_BUILDER_ABI_VERSION,
        ExternalLodBuildConfig, Gaussian3d, GaussianCamera, GaussianLodBridgeConfig,
        GaussianLodBuildSettings, GaussianLodHandle, GaussianLodPackageConfig,
        GaussianLodPackageSource, GaussianLodSettings, GaussianLodStatus, GaussianMode,
        GaussianSplattingPlugin, LodNodeId, PlanarGaussian3d, PlanarGaussian3dHandle,
        PlanarGaussian3dPage, PlanarGaussianSource, PlanarHandle, RadixSortDepthBits,
        SphericalHarmonicCoefficients, build_external_lod_package,
        gaussian::{f32::Rotation, settings::DrawMode},
        io::lod::{
            LOD_SHARD_HEADER_LEN, LodCodecLimits, decode_lod_shard_index, decode_manifest,
            decode_page_with_descriptor, lod_shard_prefix_len,
        },
        render::lod::{
            LodCompactionBuffers, LodLastRadixDrawableForTesting, LodViewBlendPublicationLabel,
            lod_view_blend_view_for_testing, lod_view_blend_weight_for_testing,
        },
        sort::SortMode,
        stream::{
            atlas_upload::LodAtlasUploadBudget,
            hierarchy::LodView,
            render_commit::{
                LodRenderCandidates, LodViewBlendEndpoint, LodViewBlendTestingSnapshot,
            },
            runtime::{LodTemporalTransitionMode, LodViewBlendIdentity},
        },
    };
    use sha2::{Digest, Sha256};

    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 128;
    const CAMERA_Z: f32 = 5.0;
    const MAX_FRAMES: u32 = 4_800;
    const CONTROL_SETTLE_FRAMES: u32 = 24;
    const PACKAGE_SETTLE_FRAMES: u32 = 8;
    const MAX_CAPTURE_REQUESTS_IN_FLIGHT: usize = 8;
    const PARENT_SIDE_QUALITY: f32 = 0.25;
    // `phase_at_compaction` is an intentionally raw testing readback. Mirror
    // the production PREPARED discriminant to prove the authored endpoint was
    // drawn before the aggregate activation CAS, rather than rediscovered at
    // a later camera-conditioned endpoint.
    const RENDER_PHASE_PREPARED_FOR_TESTING: u8 = 1;
    const SH_DC: f32 = 0.282_094_8;
    const EXPECTED_MANIFEST_LEN: usize = 2_613;
    const EXPECTED_SHARD_LEN: usize = 988;
    const EXPECTED_MANIFEST_SHA256: &str =
        "77b3a8f5f5cd4a993b1a8c0f2bc6c3d0957199f9a13b04d0b932c8b34305bb72";
    const EXPECTED_SHARD_SHA256: &str =
        "99249b83239337acae0513804bc779a329e936964c13c9d75aaa3147407c2064";

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn authenticated_abi16_k2_morph_radiance_visibility() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!(
                "skipping authenticated ABI16 K=2 GPU qualification; set RUN_GPU_RENDER_TESTS=1"
            );
            return;
        }

        let fixture = AuthenticatedK2Fixture::build();
        let asset_root = fixture.package_root.to_string_lossy().into_owned();
        let upload_budget = LodAtlasUploadBudget::try_new(size_of::<Gaussian3d>() as u64, 1)
            .expect("one K=2 page slot is a valid global staging budget");
        let probe = MorphRenderProbe::default();

        let mut app = App::new();
        app.insert_resource(ClearColor(Color::BLACK))
            .insert_resource(GaussianLodBridgeConfig {
                auto_build_flat_clouds: false,
                ..default()
            })
            .insert_resource(GaussianLodPackageConfig {
                max_atlas_gaussians: 16,
                max_atlas_bytes: 16 * 1024 * 1024,
                ..default()
            })
            .insert_resource(upload_budget)
            .insert_resource(probe)
            .insert_resource(QualificationState::new(fixture));
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: asset_root.clone(),
                    processed_file_path: asset_root,
                    meta_check: bevy::asset::AssetMetaCheck::Never,
                    unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
                    ..default()
                })
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>()
                .disable::<PipelinedRenderingPlugin>()
                .disable::<bevy::log::LogPlugin>(),
        );
        app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 120.0,
        )));
        app.add_plugins((
            GaussianSplattingPlugin,
            ExtractResourcePlugin::<MorphRenderProbe>::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, drive_qualification)
        .add_observer(on_capture);
        app.sub_app_mut(RenderApp).add_systems(
            Render,
            capture_morph_render_state
                .after(LodViewBlendPublicationLabel)
                .in_set(RenderSystems::Cleanup),
        );

        let exit = app.run();
        assert!(exit.is_success(), "K=2 GPU qualification failed: {exit:?}");
    }

    #[test]
    fn authenticated_abi16_k2_artifact_identity_is_pinned() {
        let fixture = AuthenticatedK2Fixture::build();
        assert_eq!(
            fixture.parent.position_visibility.visibility.to_bits(),
            0.51_f32.to_bits()
        );
    }

    struct TemporaryPackageRoot {
        base: PathBuf,
    }

    impl TemporaryPackageRoot {
        fn new() -> Self {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is after the Unix epoch")
                .as_nanos();
            let base = env::temp_dir().join(format!(
                "bgs-lod-morph-radiance-{}-{timestamp}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&base).expect("create isolated K=2 package root");
            Self { base }
        }
    }

    impl Drop for TemporaryPackageRoot {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.base) {
                eprintln!(
                    "failed to remove K=2 package fixture {}: {error}",
                    self.base.display()
                );
            }
        }
    }

    struct AuthenticatedK2Fixture {
        _temporary: TemporaryPackageRoot,
        package_root: PathBuf,
        parent: Gaussian3d,
        children: [Gaussian3d; 2],
        root_id: LodNodeId,
        manifest_sha256: String,
        shard_sha256: String,
    }

    impl AuthenticatedK2Fixture {
        fn build() -> Self {
            let temporary = TemporaryPackageRoot::new();
            let package_root = temporary.base.join("qualified");
            let replay_root = temporary.base.join("replayed");
            let source = k2_source();
            let cloud = PlanarGaussian3d::from(source.clone());
            let config = ExternalLodBuildConfig {
                settings: GaussianLodBuildSettings {
                    branching_factor: 2,
                    leaf_capacity: 1,
                    support_sigma: 3.0,
                },
                ..default()
            };

            for output in [&package_root, &replay_root] {
                let mut preprocessor = CpuExternalLodBatchPreprocessor;
                let report = build_external_lod_package(
                    &PlanarGaussianSource::new(&cloud),
                    output,
                    config,
                    &mut preprocessor,
                )
                .unwrap_or_else(|error| {
                    panic!("production ABI16 K=2 package build failed: {error}")
                });
                assert_eq!(
                    (
                        report.source_count,
                        report.node_count,
                        report.page_count,
                        report.stored_gaussian_count,
                        report.shard_count,
                    ),
                    (2, 3, 3, 3, 1),
                    "production K=2 artifact topology drifted"
                );
                assert_eq!(report.preprocessing_stage, "cpu-canonical-preprocess");
                assert_eq!(
                    report.hierarchy_stage,
                    "cpu-external-spatial-moment-merge-v4"
                );
            }

            let first_manifest = fs::read(package_root.join("scene.gsplatlod"))
                .expect("read qualified manifest bytes");
            let replay_manifest = fs::read(replay_root.join("scene.gsplatlod"))
                .expect("read replayed manifest bytes");
            assert_eq!(
                first_manifest, replay_manifest,
                "two exact production builds emitted different manifest bytes"
            );
            let first_shard = fs::read(package_root.join("pages/shard-000000.bgslodpack"))
                .expect("read qualified shard bytes");
            let replay_shard = fs::read(replay_root.join("pages/shard-000000.bgslodpack"))
                .expect("read replayed shard bytes");
            assert_eq!(
                first_shard, replay_shard,
                "two exact production builds emitted different shard bytes"
            );
            assert_eq!(
                package_files(&package_root),
                [
                    PathBuf::from("pages/shard-000000.bgslodpack"),
                    PathBuf::from("scene.gsplatlod"),
                ],
                "production K=2 package published an unexpected object set"
            );

            let manifest_sha256 = format!("{:x}", Sha256::digest(&first_manifest));
            let shard_sha256 = format!("{:x}", Sha256::digest(&first_shard));
            assert!(
                first_manifest.len() == EXPECTED_MANIFEST_LEN
                    && first_shard.len() == EXPECTED_SHARD_LEN
                    && manifest_sha256 == EXPECTED_MANIFEST_SHA256
                    && shard_sha256 == EXPECTED_SHARD_SHA256,
                "pin or restore the deterministic K=2 artifact: manifest_len={}, manifest_sha256={manifest_sha256}, shard_len={}, shard_sha256={shard_sha256}",
                first_manifest.len(),
                first_shard.len(),
            );
            let (manifest, pages) =
                authenticate_package(&package_root, &first_manifest, &first_shard);
            assert_eq!(
                manifest.build.builder_abi_version,
                EXTERNAL_LOD_BUILDER_ABI_VERSION
            );
            assert_eq!(manifest.header.source_gaussian_count, 2);
            assert_eq!(manifest.header.stored_gaussian_count, 3);
            assert_eq!(manifest.roots.len(), 1);
            let root_id = manifest.roots[0];
            let root_index = manifest
                .nodes
                .iter()
                .position(|node| node.id == root_id)
                .expect("root is present in node table");
            let root = &manifest.nodes[root_index];
            assert_eq!(root.representation.count, 1);
            assert_eq!(root.children.count, 2);
            assert_eq!(
                manifest.morph_child_run_lengths_at(root_index),
                Some(&[2_u16][..]),
                "K=2 root must map both children to its one parent proxy"
            );
            assert_eq!(manifest.morph_parent_record_at(root_index, 0), Some(0));
            assert_eq!(manifest.morph_parent_record_at(root_index, 1), Some(0));
            assert_eq!(manifest.morph_parent_record_at(root_index, 2), None);
            let child_end = root
                .children
                .end()
                .expect("root child range does not overflow");
            let child_nodes = &manifest.nodes[root.children.start as usize..child_end as usize];
            assert!(child_nodes.iter().all(|node| node.is_leaf()));
            assert_eq!(
                child_nodes
                    .iter()
                    .map(|node| node.representation.count)
                    .sum::<u32>(),
                2
            );

            let parent = record_for_node(root, &pages);
            let children = [
                record_for_node(&child_nodes[0], &pages),
                record_for_node(&child_nodes[1], &pages),
            ];
            assert_source_leaf_identity(&source, &children);
            assert_shared_geometry(parent, children);
            assert_eq!(
                children
                    .iter()
                    .map(|gaussian| gaussian.position_visibility.visibility.to_bits())
                    .collect::<Vec<_>>(),
                [0.49_f32.to_bits(), 0.51_f32.to_bits()],
                "canonical child order or authored visibility drifted"
            );
            assert_eq!(
                parent.position_visibility.visibility.to_bits(),
                0.51_f32.to_bits()
            );

            eprintln!(
                "authenticated ABI16 K=2 artifact: manifest_sha256={manifest_sha256}, shard_sha256={shard_sha256}, root={}",
                root_id.0
            );
            Self {
                _temporary: temporary,
                package_root,
                parent,
                children,
                root_id,
                manifest_sha256,
                shard_sha256,
            }
        }
    }

    fn k2_source() -> Vec<Gaussian3d> {
        fn endpoint(dc: [f32; 3], visibility: f32, opacity: f32) -> Gaussian3d {
            let mut spherical_harmonic = SphericalHarmonicCoefficients::default();
            for (index, value) in dc.into_iter().enumerate() {
                spherical_harmonic.set(index, value);
            }
            Gaussian3d {
                position_visibility: [0.0, 0.0, 0.0, visibility].into(),
                spherical_harmonic,
                rotation: Rotation {
                    rotation: [1.0, 0.0, 0.0, 0.0],
                },
                scale_opacity: [0.45, 0.45, 0.45, opacity].into(),
            }
        }
        vec![
            endpoint([1.4, -1.4, -1.4], 0.49, 0.2),
            endpoint([-1.4, -1.4, 1.4], 0.51, 0.7),
        ]
    }

    fn package_files(root: &Path) -> Vec<PathBuf> {
        fn visit(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) {
            for entry in fs::read_dir(directory).expect("read package directory") {
                let entry = entry.expect("read package entry");
                let path = entry.path();
                if path.is_dir() {
                    visit(root, &path, files);
                } else {
                    files.push(
                        path.strip_prefix(root)
                            .expect("package file remains under root")
                            .to_path_buf(),
                    );
                }
            }
        }
        let mut files = Vec::new();
        visit(root, root, &mut files);
        files.sort();
        files
    }

    fn authenticate_package(
        package_root: &Path,
        manifest_bytes: &[u8],
        shard_bytes: &[u8],
    ) -> (
        bevy_gaussian_splatting::GaussianLodManifest,
        BTreeMap<bevy_gaussian_splatting::LodPageId, PlanarGaussian3dPage>,
    ) {
        let limits = LodCodecLimits::default();
        let manifest = decode_manifest(manifest_bytes, limits)
            .expect("bounded manifest container and semantic authentication");
        manifest.validate().expect("decoded K=2 manifest validates");
        assert!(shard_bytes.len() >= LOD_SHARD_HEADER_LEN);
        let entry_count = u32::from_le_bytes(
            shard_bytes[12..16]
                .try_into()
                .expect("shard entry count bytes exist"),
        );
        let prefix_len = lod_shard_prefix_len(entry_count).expect("shard prefix length is valid");
        let prefix_len = usize::try_from(prefix_len).expect("shard prefix fits host usize");
        let index = decode_lod_shard_index(
            shard_bytes
                .get(..prefix_len)
                .expect("complete shard index prefix"),
            shard_bytes.len() as u64,
            manifest.header.page_count,
        )
        .expect("range-packed shard index authenticates");
        assert_eq!(index.entries.len(), manifest.pages.len());

        let mut pages = BTreeMap::new();
        for descriptor in &manifest.pages {
            let storage = descriptor
                .storage
                .as_ref()
                .expect("external package page has immutable storage");
            assert_eq!(storage.uri, "pages/shard-000000.bgslodpack");
            assert_eq!(
                package_root.join(&storage.uri),
                package_root.join("pages/shard-000000.bgslodpack")
            );
            let entry = index
                .entries
                .iter()
                .find(|entry| entry.page_id == descriptor.id)
                .expect("every descriptor has one shard entry");
            let (offset, encoded_len) = storage
                .byte_range
                .expect("range-packed page declares its byte range");
            assert_eq!(
                (entry.byte_offset, entry.encoded_len, entry.content_hash),
                (offset, encoded_len, descriptor.content_hash)
            );
            assert_eq!(storage.encoded_len, encoded_len);
            let start = usize::try_from(offset).expect("page offset fits host usize");
            let end = usize::try_from(offset + encoded_len).expect("page end fits host usize");
            let page = decode_page_with_descriptor(
                shard_bytes.get(start..end).expect("page range is in shard"),
                descriptor,
                limits,
            )
            .expect("page bytes authenticate against their descriptor");
            assert_eq!(page.content_hash(), descriptor.content_hash);
            assert!(pages.insert(descriptor.id, page).is_none());
        }
        (manifest, pages)
    }

    fn record_for_node(
        node: &bevy_gaussian_splatting::GaussianLodNode,
        pages: &BTreeMap<bevy_gaussian_splatting::LodPageId, PlanarGaussian3dPage>,
    ) -> Gaussian3d {
        assert_eq!(node.representation.count, 1);
        pages[&node.representation.page].gaussians[node.representation.offset as usize]
    }

    fn gaussian_bits(gaussian: Gaussian3d) -> Vec<u32> {
        gaussian
            .position_visibility
            .position
            .into_iter()
            .chain([gaussian.position_visibility.visibility])
            .chain(gaussian.spherical_harmonic.coefficients)
            .chain(gaussian.rotation.rotation)
            .chain(gaussian.scale_opacity.scale)
            .chain([gaussian.scale_opacity.opacity])
            .map(f32::to_bits)
            .collect()
    }

    fn assert_source_leaf_identity(source: &[Gaussian3d], children: &[Gaussian3d; 2]) {
        let mut expected = source
            .iter()
            .copied()
            .map(gaussian_bits)
            .collect::<Vec<_>>();
        let mut actual = children
            .iter()
            .copied()
            .map(gaussian_bits)
            .collect::<Vec<_>>();
        expected.sort();
        actual.sort();
        assert_eq!(
            actual, expected,
            "q=1 leaves must preserve exact source bits"
        );
    }

    fn assert_shared_geometry(parent: Gaussian3d, children: [Gaussian3d; 2]) {
        for (index, child) in children.into_iter().enumerate() {
            assert_eq!(
                parent.position_visibility.position.map(f32::to_bits),
                child.position_visibility.position.map(f32::to_bits),
                "parent/child {index} positions must be coincident for the pixel oracle"
            );
            assert_eq!(
                parent.rotation.rotation.map(f32::to_bits),
                child.rotation.rotation.map(f32::to_bits),
                "parent/child {index} rotations must match for the pixel oracle"
            );
            assert_eq!(
                parent.scale_opacity.scale.map(f32::to_bits),
                child.scale_opacity.scale.map(f32::to_bits),
                "parent/child {index} scales must match for the pixel oracle"
            );
        }
    }

    #[derive(Clone, Debug)]
    struct MorphFrame {
        candidate_prepared: bool,
        candidate_active: bool,
        candidate_transitioning: bool,
        temporal_mode: Option<LodTemporalTransitionMode>,
        draw_mode: DrawMode,
        render_commit_identity: usize,
        drawable: LodLastRadixDrawableForTesting,
        candidate_snapshot: Option<LodViewBlendTestingSnapshot>,
        current_view: Option<LodView>,
        current_oracle: Option<f32>,
    }

    #[derive(Default)]
    struct MorphProbeShared {
        latest: Option<MorphFrame>,
        armed_requests: HashSet<Entity>,
        latched_requests: HashMap<Entity, Option<MorphFrame>>,
    }

    #[derive(Resource, Clone, ExtractResource, Default)]
    struct MorphRenderProbe(Arc<Mutex<MorphProbeShared>>);

    impl MorphRenderProbe {
        fn latest(&self) -> Option<MorphFrame> {
            self.0.lock().ok()?.latest.clone()
        }

        fn arm(&self, request: Entity) {
            let mut shared = self.0.lock().expect("morph probe mutex is not poisoned");
            assert!(
                shared.armed_requests.insert(request),
                "screenshot request {request:?} was armed twice"
            );
            assert!(
                !shared.latched_requests.contains_key(&request),
                "screenshot request {request:?} reused an old render latch"
            );
        }

        fn take_latched(&self, request: Entity) -> Option<Option<MorphFrame>> {
            self.0.lock().ok()?.latched_requests.remove(&request)
        }
    }

    fn capture_morph_render_state(
        buffers: Res<LodCompactionBuffers<Gaussian3d>>,
        views: Query<&ExtractedView, With<GaussianCamera>>,
        clouds: Query<(
            Entity,
            &PlanarGaussian3dHandle,
            &GlobalTransform,
            &GaussianLodSettings,
            &CloudSettings,
            &LodRenderCandidates,
        )>,
        probe: Res<MorphRenderProbe>,
    ) {
        let mut latest = None;
        for view in &views {
            let camera = view.retained_view_entity.main_entity.id();
            for (cloud, handle, world_from_local, settings, cloud_settings, candidates) in &clouds {
                let Some(candidate) = candidates.get(camera) else {
                    continue;
                };
                let Some(compaction) =
                    buffers.get_ready(view.retained_view_entity, cloud, handle.handle().id())
                else {
                    continue;
                };
                let Some(drawable) = compaction.last_radix_drawable_for_testing(candidate) else {
                    continue;
                };
                let current_view = lod_view_blend_view_for_testing(view, world_from_local);
                let current_oracle = drawable
                    .view_blend
                    .as_ref()
                    .filter(|blend| blend.edges.len() == 1)
                    .and_then(|blend| {
                        let render_view = current_view?;
                        Some(lod_view_blend_weight_for_testing(
                            render_view,
                            settings.quality_target(),
                            &blend.edges[0],
                        ))
                    });
                assert!(
                    latest.is_none(),
                    "K=2 qualification expects one package consumer"
                );
                latest = Some(MorphFrame {
                    candidate_prepared: candidate.render_is_prepared(),
                    candidate_active: candidate.render_is_active_for_testing(),
                    candidate_transitioning: candidate.render_is_transitioning_for_testing(),
                    temporal_mode: candidate.temporal_transition_mode(),
                    draw_mode: cloud_settings.draw_mode,
                    render_commit_identity: candidate.render_commit_identity_for_testing(),
                    drawable,
                    candidate_snapshot: candidate.view_blend_testing_snapshot(),
                    current_view,
                    current_oracle,
                });
            }
        }
        let mut shared = probe.0.lock().expect("morph probe mutex is not poisoned");
        shared.latest = latest.clone();
        let armed = std::mem::take(&mut shared.armed_requests);
        for request in armed {
            assert!(
                shared
                    .latched_requests
                    .insert(request, latest.clone())
                    .is_none(),
                "screenshot request {request:?} was latched twice"
            );
        }
    }

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    enum CaptureKey {
        ParentAll,
        ParentControlAll,
        ChildrenControlAll,
        ChildrenAll,
        InteriorAll,
        ParentControlSelected,
        ParentSelected,
        ChildrenControlSelected,
        ChildrenSelected,
        InteriorSelected,
        ParentControlHighlight,
        ParentHighlight,
        ChildrenControlHighlight,
        ChildrenHighlight,
        InteriorHighlight,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Phase {
        PrepareFine(DrawMode),
        CaptureChildrenEndpoint(DrawMode),
        PrepareCoarse(DrawMode),
        CaptureParentEndpoint(DrawMode),
        CaptureInterior(DrawMode),
        ParentControl(DrawMode),
        ChildrenControl(DrawMode),
    }

    #[derive(Clone)]
    struct CapturedImage {
        rgba: Vec<u8>,
        morph_stamp: Option<MorphDrawableStamp>,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct MorphDrawableStamp {
        render_commit_identity: usize,
        compaction_generation: u64,
        radix_publication_generation: u64,
        morph_identity: LodViewBlendIdentity,
        weight_bits: u32,
        endpoint: LodViewBlendEndpoint,
        draw_mode: DrawMode,
    }

    #[derive(Clone, Copy, Debug)]
    enum CaptureExpectation {
        AuthoredEndpoint {
            endpoint: LodViewBlendEndpoint,
            draw_mode: DrawMode,
        },
        StableInterior {
            draw_mode: DrawMode,
            expected_weight_bits: Option<u32>,
        },
        CategoricalControl,
    }

    #[derive(Clone, Copy, Debug)]
    struct PendingCapture {
        key: CaptureKey,
        phase_generation: u64,
        expectation: CaptureExpectation,
    }

    #[derive(Resource)]
    struct QualificationState {
        fixture: AuthenticatedK2Fixture,
        phase: Phase,
        phase_frames: u32,
        phase_generation: u64,
        total_frames: u32,
        pending_captures: HashMap<Entity, PendingCapture>,
        target: Option<Handle<Image>>,
        package: Option<Entity>,
        parent_control: Option<Entity>,
        children_control: Option<Entity>,
        camera: Option<Entity>,
        captures: BTreeMap<CaptureKey, CapturedImage>,
        interior_quality: Option<f32>,
        interior_weight: Option<f32>,
    }

    impl QualificationState {
        fn new(fixture: AuthenticatedK2Fixture) -> Self {
            Self {
                fixture,
                phase: Phase::PrepareFine(DrawMode::All),
                phase_frames: 0,
                phase_generation: 0,
                total_frames: 0,
                pending_captures: HashMap::new(),
                target: None,
                package: None,
                parent_control: None,
                children_control: None,
                camera: None,
                captures: BTreeMap::new(),
                interior_quality: None,
                interior_weight: None,
            }
        }

        fn enter(&mut self, phase: Phase) {
            self.phase = phase;
            self.phase_frames = 0;
            self.phase_generation = self
                .phase_generation
                .checked_add(1)
                .expect("qualification phase generation does not overflow");
        }
    }

    fn cloud_settings(draw_mode: DrawMode) -> CloudSettings {
        CloudSettings {
            gaussian_mode: GaussianMode::Gaussian3d,
            sort_mode: SortMode::Radix,
            radix_sort_depth_bits: RadixSortDepthBits::Bits32,
            draw_mode,
            opacity_adaptive_radius: false,
            ..default()
        }
    }

    fn setup(
        mut commands: Commands,
        mut state: ResMut<QualificationState>,
        asset_server: Res<AssetServer>,
        mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let target = images.add(Image::new_target_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let manifest = asset_server.load("scene.gsplatlod");
        let package = commands
            .spawn((
                GaussianLodHandle(manifest),
                GaussianLodPackageSource::native_directory(
                    state.fixture.package_root.to_string_lossy().into_owned(),
                ),
                cloud_settings(DrawMode::All),
                GaussianLodSettings {
                    quality: 1.0,
                    hysteresis: 0.0,
                    frustum_culling: false,
                    ..default()
                },
                NoFrustumCulling,
                Transform::IDENTITY,
                Visibility::Visible,
                Name::new("authenticated_abi16_k2_package"),
            ))
            .id();
        let parent_control = commands
            .spawn((
                PlanarGaussian3dHandle(
                    gaussian_assets.add(PlanarGaussian3d::from(vec![state.fixture.parent])),
                ),
                cloud_settings(DrawMode::All),
                GaussianLodSettings::default(),
                NoFrustumCulling,
                Transform::IDENTITY,
                Visibility::Hidden,
                Name::new("k2_parent_categorical_control"),
            ))
            .id();
        let children_control = commands
            .spawn((
                PlanarGaussian3dHandle(
                    gaussian_assets.add(PlanarGaussian3d::from(state.fixture.children.to_vec())),
                ),
                cloud_settings(DrawMode::All),
                GaussianLodSettings::default(),
                NoFrustumCulling,
                Transform::IDENTITY,
                Visibility::Hidden,
                Name::new("k2_children_categorical_control"),
            ))
            .id();
        let camera = commands
            .spawn((
                Camera3d::default(),
                Camera::default(),
                Projection::Perspective(PerspectiveProjection {
                    fov: 60.0_f32.to_radians(),
                    near: 0.01,
                    far: 100.0,
                    ..default()
                }),
                RenderTarget::Image(target.clone().into()),
                Transform::from_translation(Vec3::new(0.0, 0.0, CAMERA_Z)),
                Tonemapping::None,
                GaussianCamera::default(),
                Name::new("k2_radiance_camera"),
            ))
            .id();
        state.target = Some(target);
        state.package = Some(package);
        state.parent_control = Some(parent_control);
        state.children_control = Some(children_control);
        state.camera = Some(camera);
    }

    #[allow(clippy::too_many_arguments)]
    fn drive_qualification(
        mut commands: Commands,
        mut state: ResMut<QualificationState>,
        probe: Res<MorphRenderProbe>,
        statuses: Query<&GaussianLodStatus>,
        mut lod_settings: Query<&mut GaussianLodSettings>,
        mut cloud_settings_query: Query<&mut CloudSettings>,
        mut visibilities: Query<&mut Visibility>,
    ) {
        state.total_frames += 1;
        state.phase_frames += 1;
        assert!(
            state.total_frames <= MAX_FRAMES,
            "K=2 GPU qualification timed out in {:?}; latest probe={:?}",
            state.phase,
            probe.latest()
        );
        let package = state.package.expect("package entity exists");
        let parent_control = state.parent_control.expect("parent control exists");
        let children_control = state.children_control.expect("children control exists");
        let phase = state.phase;
        let (source, quality, draw_mode) = match phase {
            Phase::PrepareFine(draw_mode) => (Source::Package, 1.0, draw_mode),
            Phase::CaptureChildrenEndpoint(draw_mode) => {
                (Source::Package, PARENT_SIDE_QUALITY, draw_mode)
            }
            Phase::PrepareCoarse(draw_mode) => (Source::Package, 0.0, draw_mode),
            Phase::CaptureParentEndpoint(draw_mode) | Phase::CaptureInterior(draw_mode) => (
                Source::Package,
                state
                    .interior_quality
                    .expect("runtime-derived interior quality exists"),
                draw_mode,
            ),
            Phase::ParentControl(draw_mode) => (Source::Parent, 0.0, draw_mode),
            Phase::ChildrenControl(draw_mode) => (Source::Children, 1.0, draw_mode),
        };

        *visibilities.get_mut(package).expect("package visibility") = if source == Source::Package {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        *visibilities
            .get_mut(parent_control)
            .expect("parent control visibility") = if source == Source::Parent {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        *visibilities
            .get_mut(children_control)
            .expect("children control visibility") = if source == Source::Children {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        lod_settings
            .get_mut(package)
            .expect("package LoD settings")
            .quality = quality;
        for entity in [package, parent_control, children_control] {
            cloud_settings_query
                .get_mut(entity)
                .expect("cloud settings")
                .draw_mode = draw_mode;
        }

        match phase {
            Phase::PrepareFine(draw_mode) => {
                if state.phase_frames >= PACKAGE_SETTLE_FRAMES
                    && categorical_package_is_ready(&probe, draw_mode, 2)
                    && package_target_is_satisfied(&statuses, package)
                {
                    state.enter(Phase::CaptureChildrenEndpoint(draw_mode));
                }
            }
            Phase::CaptureChildrenEndpoint(draw_mode) => request_capture(
                &mut commands,
                &mut state,
                &probe,
                children_key(draw_mode),
                CaptureExpectation::AuthoredEndpoint {
                    endpoint: LodViewBlendEndpoint::ChildrenExact,
                    draw_mode,
                },
            ),
            Phase::PrepareCoarse(draw_mode) => {
                if state.phase_frames >= PACKAGE_SETTLE_FRAMES
                    && categorical_package_is_ready(&probe, draw_mode, 1)
                    && package_target_is_satisfied(&statuses, package)
                {
                    state.enter(Phase::CaptureParentEndpoint(draw_mode));
                }
            }
            Phase::CaptureParentEndpoint(draw_mode) => request_capture(
                &mut commands,
                &mut state,
                &probe,
                parent_key(draw_mode),
                CaptureExpectation::AuthoredEndpoint {
                    endpoint: LodViewBlendEndpoint::ParentExact,
                    draw_mode,
                },
            ),
            Phase::CaptureInterior(draw_mode) => {
                let expected_weight_bits = state.interior_weight.map(f32::to_bits);
                if probe.latest().as_ref().is_some_and(|frame| {
                    qualified_stable_interior(
                        frame,
                        draw_mode,
                        state.fixture.root_id,
                        expected_weight_bits,
                    )
                    .is_some()
                }) {
                    request_capture(
                        &mut commands,
                        &mut state,
                        &probe,
                        interior_key(draw_mode),
                        CaptureExpectation::StableInterior {
                            draw_mode,
                            expected_weight_bits,
                        },
                    );
                }
            }
            Phase::ParentControl(draw_mode) => {
                if state.phase_frames >= CONTROL_SETTLE_FRAMES {
                    request_capture(
                        &mut commands,
                        &mut state,
                        &probe,
                        parent_control_key(draw_mode),
                        CaptureExpectation::CategoricalControl,
                    );
                }
            }
            Phase::ChildrenControl(draw_mode) => {
                if state.phase_frames >= CONTROL_SETTLE_FRAMES {
                    request_capture(
                        &mut commands,
                        &mut state,
                        &probe,
                        children_control_key(draw_mode),
                        CaptureExpectation::CategoricalControl,
                    );
                }
            }
        }
    }

    fn package_target_is_satisfied(statuses: &Query<&GaussianLodStatus>, package: Entity) -> bool {
        statuses
            .get(package)
            .ok()
            .and_then(|status| status.target_satisfied)
            == Some(true)
    }

    fn categorical_package_is_ready(
        probe: &MorphRenderProbe,
        draw_mode: DrawMode,
        expected_count: u32,
    ) -> bool {
        probe.latest().is_some_and(|frame| {
            frame.candidate_active
                && frame.temporal_mode.is_none()
                && frame.draw_mode == draw_mode
                && frame.drawable.candidate_token_matches
                && frame.drawable.candidate_content_matches
                && frame.drawable.rendered_candidate_count == expected_count
                && frame.drawable.view_blend.is_none()
        })
    }

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum Source {
        Package,
        Parent,
        Children,
    }

    struct QualifiedWeight {
        weight: f32,
        stamp: MorphDrawableStamp,
    }

    fn qualified_authored_endpoint(
        frame: &MorphFrame,
        endpoint: LodViewBlendEndpoint,
        draw_mode: DrawMode,
        root_id: LodNodeId,
    ) -> Option<QualifiedWeight> {
        if !frame.candidate_prepared
            || frame.candidate_active
            || frame.candidate_transitioning
            || frame.temporal_mode != Some(LodTemporalTransitionMode::Morphing)
            || frame.draw_mode != draw_mode
            || !frame.drawable.candidate_token_matches
            || !frame.drawable.candidate_content_matches
            || frame.drawable.phase_at_compaction != Some(RENDER_PHASE_PREPARED_FOR_TESTING)
        {
            return None;
        }
        let blend = frame.drawable.view_blend.as_ref()?;
        if blend.edges.len() != 1
            || blend.weights.len() != 1
            || blend.endpoints.len() != 1
            || blend.invalid_pressure != [false]
            || blend.recovery_lag != [false]
        {
            return None;
        }
        let edge = &blend.edges[0];
        let weight = blend.weights[0];
        let endpoint_bits = match endpoint {
            LodViewBlendEndpoint::ParentExact => 0.0_f32.to_bits(),
            LodViewBlendEndpoint::ChildrenExact => 1.0_f32.to_bits(),
            LodViewBlendEndpoint::Fractional => return None,
        };
        if blend.endpoints != [endpoint]
            || weight.displayed.to_bits() != endpoint_bits
            || edge.initial_weight_bits() != endpoint_bits
            || edge.activation_requires_slew()
            || edge.parent() != root_id
            || edge.children().len() != 2
        {
            return None;
        }
        let published = frame.candidate_snapshot.as_ref()?;
        if published.status.edge_count != 1
            || published.status.invalid_pressure_count != 0
            || published.status.missing_consumer_count != 0
            || published.endpoints != [endpoint]
            || published.weights.len() != 1
            || published.weights[0].displayed.to_bits() != endpoint_bits
        {
            return None;
        }
        Some(QualifiedWeight {
            weight: weight.displayed,
            stamp: MorphDrawableStamp {
                render_commit_identity: frame.render_commit_identity,
                compaction_generation: frame.drawable.compaction_generation,
                radix_publication_generation: frame.drawable.radix_publication_generation,
                morph_identity: blend.identity,
                weight_bits: weight.displayed.to_bits(),
                endpoint,
                draw_mode,
            },
        })
    }

    fn runtime_derived_interior_quality(frame: &MorphFrame) -> Option<(f32, f32)> {
        let blend = frame.drawable.view_blend.as_ref()?;
        let edge = blend.edges.first()?;
        let view = frame.current_view?;
        (1..100)
            .filter_map(|step| {
                let quality = step as f32 / 100.0;
                let target = GaussianLodSettings {
                    quality,
                    ..default()
                }
                .quality_target();
                let weight = lod_view_blend_weight_for_testing(view, target, edge);
                (weight.is_finite() && (0.15..=0.85).contains(&weight)).then_some((quality, weight))
            })
            .min_by(|left, right| {
                (left.1 - 0.5)
                    .abs()
                    .total_cmp(&(right.1 - 0.5).abs())
                    .then_with(|| left.0.total_cmp(&right.0))
            })
    }

    fn qualified_stable_interior(
        frame: &MorphFrame,
        draw_mode: DrawMode,
        root_id: LodNodeId,
        expected_weight_bits: Option<u32>,
    ) -> Option<QualifiedWeight> {
        if !frame.candidate_active
            || frame.temporal_mode != Some(LodTemporalTransitionMode::Morphing)
            || frame.draw_mode != draw_mode
            || !frame.drawable.candidate_token_matches
            || !frame.drawable.candidate_content_matches
        {
            return None;
        }
        let blend = frame.drawable.view_blend.as_ref()?;
        if blend.edges.len() != 1
            || blend.weights.len() != 1
            || blend.endpoints != [LodViewBlendEndpoint::Fractional]
            || blend.invalid_pressure != [false]
            || blend.recovery_lag != [false]
            || !blend.desired_evaluation_complete
            || blend.upload.lagging_edge_count != 0
        {
            return None;
        }
        let edge = &blend.edges[0];
        let weight = blend.weights[0];
        let oracle = frame.current_oracle?;
        if edge.parent() != root_id
            || edge.children().len() != 2
            || edge.activation_requires_slew()
            || !(0.15..=0.85).contains(&weight.displayed)
            || weight.displayed.to_bits() != weight.desired.to_bits()
            || weight.displayed.to_bits() != oracle.to_bits()
            || expected_weight_bits.is_some_and(|expected| weight.displayed.to_bits() != expected)
        {
            return None;
        }
        let published = frame.candidate_snapshot.as_ref()?;
        if published.status.edge_count != 1
            || published.status.lagging_count != 0
            || published.status.invalid_pressure_count != 0
            || published.status.missing_consumer_count != 0
            || published.endpoints != [LodViewBlendEndpoint::Fractional]
            || published.weights.len() != 1
            || published.weights[0].displayed.to_bits() != weight.displayed.to_bits()
            || published.weights[0].desired.to_bits() != weight.displayed.to_bits()
        {
            return None;
        }
        Some(QualifiedWeight {
            weight: weight.displayed,
            stamp: MorphDrawableStamp {
                render_commit_identity: frame.render_commit_identity,
                compaction_generation: frame.drawable.compaction_generation,
                radix_publication_generation: frame.drawable.radix_publication_generation,
                morph_identity: blend.identity,
                weight_bits: weight.displayed.to_bits(),
                endpoint: LodViewBlendEndpoint::Fractional,
                draw_mode,
            },
        })
    }

    fn request_capture(
        commands: &mut Commands,
        state: &mut QualificationState,
        probe: &MorphRenderProbe,
        key: CaptureKey,
        expectation: CaptureExpectation,
    ) {
        if state.pending_captures.len() >= MAX_CAPTURE_REQUESTS_IN_FLIGHT {
            return;
        }
        let request = commands
            .spawn(Screenshot::image(
                state.target.clone().expect("render target exists"),
            ))
            .id();
        probe.arm(request);
        assert!(
            state
                .pending_captures
                .insert(
                    request,
                    PendingCapture {
                        key,
                        phase_generation: state.phase_generation,
                        expectation,
                    },
                )
                .is_none(),
            "screenshot request entity {request:?} was reused"
        );
    }

    fn on_capture(
        trigger: On<ScreenshotCaptured>,
        mut state: ResMut<QualificationState>,
        probe: Res<MorphRenderProbe>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let pending = state
            .pending_captures
            .remove(&trigger.entity)
            .unwrap_or_else(|| panic!("unregistered screenshot request {:?}", trigger.entity));
        let latched_frame = probe.take_latched(trigger.entity).unwrap_or_else(|| {
            panic!(
                "screenshot request {:?} completed without its request-frame Render Cleanup latch",
                trigger.entity
            )
        });
        if pending.phase_generation != state.phase_generation {
            return;
        }
        let key = pending.key;
        let qualified = match pending.expectation {
            CaptureExpectation::AuthoredEndpoint {
                endpoint,
                draw_mode,
            } => latched_frame.as_ref().and_then(|frame| {
                qualified_authored_endpoint(frame, endpoint, draw_mode, state.fixture.root_id)
            }),
            CaptureExpectation::StableInterior {
                draw_mode,
                expected_weight_bits,
            } => latched_frame.as_ref().and_then(|frame| {
                qualified_stable_interior(
                    frame,
                    draw_mode,
                    state.fixture.root_id,
                    expected_weight_bits,
                )
            }),
            CaptureExpectation::CategoricalControl => None,
        };
        if !matches!(pending.expectation, CaptureExpectation::CategoricalControl)
            && qualified.is_none()
        {
            // Screenshot readback is asynchronous, while authored endpoints
            // may exist for one physical publication. Discard an image whose
            // own request-frame latch did not prove the requested state and
            // keep issuing bounded requests in the same phase.
            return;
        }
        let rgba = trigger
            .image
            .clone()
            .try_into_dynamic()
            .expect("qualification screenshot converts")
            .to_rgba8()
            .into_raw();
        assert_eq!(rgba.len(), (WIDTH * HEIGHT * 4) as usize);
        assert!(
            state
                .captures
                .insert(
                    key,
                    CapturedImage {
                        rgba,
                        morph_stamp: qualified.as_ref().map(|qualified| qualified.stamp),
                    },
                )
                .is_none(),
            "capture phase {key:?} ran twice"
        );

        if key == CaptureKey::InteriorAll {
            let qualified = qualified.expect("interior package capture has morph evidence");
            state.interior_weight = Some(qualified.weight);
        }
        if key == CaptureKey::ChildrenAll {
            let frame = latched_frame
                .as_ref()
                .expect("authored child endpoint has a request-frame render latch");
            let (quality, _) = runtime_derived_interior_quality(frame).unwrap_or_else(|| {
                panic!(
                    "authenticated K=2 edge has no balanced interior quality in the exact render view"
                )
            });
            state.interior_quality = Some(quality);
        }
        let next = match key {
            CaptureKey::ChildrenAll => Phase::PrepareCoarse(DrawMode::All),
            CaptureKey::ParentAll => Phase::CaptureInterior(DrawMode::All),
            CaptureKey::InteriorAll => Phase::ParentControl(DrawMode::All),
            CaptureKey::ParentControlAll => Phase::ChildrenControl(DrawMode::All),
            CaptureKey::ChildrenControlAll => Phase::PrepareFine(DrawMode::Selected),
            CaptureKey::ChildrenSelected => Phase::PrepareCoarse(DrawMode::Selected),
            CaptureKey::ParentSelected => Phase::CaptureInterior(DrawMode::Selected),
            CaptureKey::InteriorSelected => Phase::ParentControl(DrawMode::Selected),
            CaptureKey::ParentControlSelected => Phase::ChildrenControl(DrawMode::Selected),
            CaptureKey::ChildrenControlSelected => Phase::PrepareFine(DrawMode::HighlightSelected),
            CaptureKey::ChildrenHighlight => Phase::PrepareCoarse(DrawMode::HighlightSelected),
            CaptureKey::ParentHighlight => Phase::CaptureInterior(DrawMode::HighlightSelected),
            CaptureKey::InteriorHighlight => Phase::ParentControl(DrawMode::HighlightSelected),
            CaptureKey::ParentControlHighlight => {
                Phase::ChildrenControl(DrawMode::HighlightSelected)
            }
            CaptureKey::ChildrenControlHighlight => {
                assert_qualification_images(&state);
                exit.write(AppExit::Success);
                return;
            }
        };
        state.enter(next);
    }

    fn parent_key(draw_mode: DrawMode) -> CaptureKey {
        match draw_mode {
            DrawMode::All => CaptureKey::ParentAll,
            DrawMode::Selected => CaptureKey::ParentSelected,
            DrawMode::HighlightSelected => CaptureKey::ParentHighlight,
        }
    }

    fn children_key(draw_mode: DrawMode) -> CaptureKey {
        match draw_mode {
            DrawMode::All => CaptureKey::ChildrenAll,
            DrawMode::Selected => CaptureKey::ChildrenSelected,
            DrawMode::HighlightSelected => CaptureKey::ChildrenHighlight,
        }
    }

    fn interior_key(draw_mode: DrawMode) -> CaptureKey {
        match draw_mode {
            DrawMode::All => CaptureKey::InteriorAll,
            DrawMode::Selected => CaptureKey::InteriorSelected,
            DrawMode::HighlightSelected => CaptureKey::InteriorHighlight,
        }
    }

    fn parent_control_key(draw_mode: DrawMode) -> CaptureKey {
        match draw_mode {
            DrawMode::All => CaptureKey::ParentControlAll,
            DrawMode::Selected => CaptureKey::ParentControlSelected,
            DrawMode::HighlightSelected => CaptureKey::ParentControlHighlight,
        }
    }

    fn children_control_key(draw_mode: DrawMode) -> CaptureKey {
        match draw_mode {
            DrawMode::All => CaptureKey::ChildrenControlAll,
            DrawMode::Selected => CaptureKey::ChildrenControlSelected,
            DrawMode::HighlightSelected => CaptureKey::ChildrenControlHighlight,
        }
    }

    fn assert_qualification_images(state: &QualificationState) {
        let capture = |key| {
            state
                .captures
                .get(&key)
                .unwrap_or_else(|| panic!("missing {key:?} capture"))
        };
        let image = |key| &capture(key).rgba;
        let endpoint_proofs = [
            (
                CaptureKey::ParentAll,
                LodViewBlendEndpoint::ParentExact,
                DrawMode::All,
                0.0_f32.to_bits(),
            ),
            (
                CaptureKey::ChildrenAll,
                LodViewBlendEndpoint::ChildrenExact,
                DrawMode::All,
                1.0_f32.to_bits(),
            ),
            (
                CaptureKey::ParentSelected,
                LodViewBlendEndpoint::ParentExact,
                DrawMode::Selected,
                0.0_f32.to_bits(),
            ),
            (
                CaptureKey::ChildrenSelected,
                LodViewBlendEndpoint::ChildrenExact,
                DrawMode::Selected,
                1.0_f32.to_bits(),
            ),
            (
                CaptureKey::ParentHighlight,
                LodViewBlendEndpoint::ParentExact,
                DrawMode::HighlightSelected,
                0.0_f32.to_bits(),
            ),
            (
                CaptureKey::ChildrenHighlight,
                LodViewBlendEndpoint::ChildrenExact,
                DrawMode::HighlightSelected,
                1.0_f32.to_bits(),
            ),
        ];
        for (key, endpoint, draw_mode, weight_bits) in endpoint_proofs {
            let stamp = capture(key)
                .morph_stamp
                .unwrap_or_else(|| panic!("{key:?} omitted authored morph evidence"));
            assert_eq!(
                (stamp.endpoint, stamp.draw_mode, stamp.weight_bits),
                (endpoint, draw_mode, weight_bits),
                "{key:?} was not the exact authored endpoint requested"
            );
            assert_ne!(
                stamp.render_commit_identity, 0,
                "{key:?} omitted its candidate-token identity"
            );
            assert_ne!(
                stamp.radix_publication_generation, 0,
                "{key:?} was not paired with a radix publication"
            );
            assert_eq!(
                stamp.morph_identity.descriptor_count(),
                u32::try_from(state.fixture.children.len())
                    .expect("authenticated child count fits u32"),
                "{key:?} did not retain one physical descriptor per authenticated child range"
            );
            assert_eq!(stamp.morph_identity.mapping_record_count(), 2);
            eprintln!(
                "{key:?}: compaction_generation={}, radix_publication_generation={}",
                stamp.compaction_generation, stamp.radix_publication_generation
            );
        }
        for (key, draw_mode) in [
            (CaptureKey::InteriorAll, DrawMode::All),
            (CaptureKey::InteriorSelected, DrawMode::Selected),
            (CaptureKey::InteriorHighlight, DrawMode::HighlightSelected),
        ] {
            let stamp = capture(key)
                .morph_stamp
                .unwrap_or_else(|| panic!("{key:?} omitted stable interior evidence"));
            assert_eq!(stamp.endpoint, LodViewBlendEndpoint::Fractional);
            assert_eq!(stamp.draw_mode, draw_mode);
            assert_eq!(
                stamp.weight_bits,
                state
                    .interior_weight
                    .expect("interior weight exists")
                    .to_bits()
            );
        }
        for key in [
            CaptureKey::ParentControlAll,
            CaptureKey::ChildrenControlAll,
            CaptureKey::ParentControlSelected,
            CaptureKey::ChildrenControlSelected,
            CaptureKey::ParentControlHighlight,
            CaptureKey::ChildrenControlHighlight,
        ] {
            assert!(
                capture(key).morph_stamp.is_none(),
                "{key:?} categorical control carried package morph evidence"
            );
        }
        let endpoint_pairs = [
            (CaptureKey::ParentAll, CaptureKey::ParentControlAll),
            (CaptureKey::ChildrenAll, CaptureKey::ChildrenControlAll),
            (
                CaptureKey::ParentSelected,
                CaptureKey::ParentControlSelected,
            ),
            (
                CaptureKey::ChildrenSelected,
                CaptureKey::ChildrenControlSelected,
            ),
            (
                CaptureKey::ParentHighlight,
                CaptureKey::ParentControlHighlight,
            ),
            (
                CaptureKey::ChildrenHighlight,
                CaptureKey::ChildrenControlHighlight,
            ),
        ];
        for (actual, control) in endpoint_pairs {
            assert_endpoint_image(image(actual), image(control), actual);
        }

        let parent_all = image(CaptureKey::ParentControlAll);
        let children_all = image(CaptureKey::ChildrenControlAll);
        let interior_all = image(CaptureKey::InteriorAll);
        assert_nonblack(parent_all, "parent endpoint");
        assert_nonblack(children_all, "children endpoint");
        assert_nonblack(interior_all, "interior blend");
        assert_ne!(parent_all, children_all, "K=2 endpoint colors must differ");
        assert_ne!(interior_all, parent_all, "interior must differ from t=0");
        assert_ne!(interior_all, children_all, "interior must differ from t=1");
        assert_tau_weighted_interior(
            state.fixture.parent,
            state.fixture.children,
            state.interior_weight.expect("interior weight exists"),
            parent_all,
            children_all,
            interior_all,
        );

        assert!(
            rgb_distance(
                image(CaptureKey::ChildrenControlAll),
                image(CaptureKey::ChildrenControlSelected)
            ) > 100.0,
            "Selected must remove the authored visibility=.49 child"
        );
        assert!(
            rgb_distance(
                image(CaptureKey::InteriorAll),
                image(CaptureKey::InteriorSelected)
            ) > 50.0,
            "interior Selected must gate parent/child optical depth independently"
        );
        let child_highlight = image(CaptureKey::ChildrenHighlight);
        assert_green_population(child_highlight, "children highlight");
        assert_highlight_preserves_unselected_native_radiance(
            state.fixture.parent,
            state.fixture.children,
            None,
            parent_all,
            children_all,
            child_highlight,
            "children highlight",
        );
        let interior_highlight = image(CaptureKey::InteriorHighlight);
        assert_green_population(interior_highlight, "interior highlight");
        assert_highlight_preserves_unselected_native_radiance(
            state.fixture.parent,
            state.fixture.children,
            Some(state.interior_weight.expect("interior weight exists")),
            parent_all,
            children_all,
            interior_highlight,
            "interior highlight",
        );
        let parent_highlight = image(CaptureKey::ParentHighlight);
        assert_green_population(parent_highlight, "parent highlight");
        assert!(
            rgb_distance(child_highlight, image(CaptureKey::ChildrenSelected)) > 100.0,
            "HighlightSelected must not collapse to DrawSelected"
        );

        eprintln!(
            "ABI16 K=2 GPU radiance qualification passed: root={}, q={:.3}, t={:.6}, manifest_sha256={}, shard_sha256={}",
            state.fixture.root_id.0,
            state.interior_quality.expect("interior quality"),
            state.interior_weight.expect("interior weight"),
            state.fixture.manifest_sha256,
            state.fixture.shard_sha256,
        );
    }

    fn assert_endpoint_image(actual: &[u8], control: &[u8], label: CaptureKey) {
        const MAX_RGB_DELTA: u8 = 2;
        const BACKGROUND_MAX: u8 = 2;
        const DEFINITE_FOREGROUND_MIN: u8 = BACKGROUND_MAX + MAX_RGB_DELTA + 1;

        assert_eq!(actual.len(), control.len());
        let mut max_delta = 0_u8;
        let mut contradictory_support = 0_u32;
        for (actual, expected) in actual.chunks_exact(4).zip(control.chunks_exact(4)) {
            for channel in 0..3 {
                max_delta = max_delta.max(actual[channel].abs_diff(expected[channel]));
            }
            let actual_peak = actual[..3].iter().copied().max().unwrap();
            let expected_peak = expected[..3].iter().copied().max().unwrap();
            contradictory_support += u32::from(
                (actual_peak >= DEFINITE_FOREGROUND_MIN && expected_peak <= BACKGROUND_MAX)
                    || (expected_peak >= DEFINITE_FOREGROUND_MIN && actual_peak <= BACKGROUND_MAX),
            );
        }
        assert!(
            max_delta <= MAX_RGB_DELTA,
            "{label:?} differs from its categorical endpoint by {max_delta} RGBA8 codes"
        );
        assert_eq!(
            contradictory_support, 0,
            "{label:?} changed conservative endpoint support outside the RGBA8 quantization deadband"
        );
    }

    fn assert_nonblack(image: &[u8], label: &str) {
        let visible = image
            .chunks_exact(4)
            .filter(|pixel| pixel[..3].iter().copied().max().unwrap() > 4)
            .count();
        assert!(
            visible >= 64,
            "{label} was black or too small: {visible} pixels"
        );
    }

    fn linear_rgb(pixel: &[u8]) -> Vec3 {
        Vec3::new(
            srgb_to_linear(pixel[0] as f32 / 255.0),
            srgb_to_linear(pixel[1] as f32 / 255.0),
            srgb_to_linear(pixel[2] as f32 / 255.0),
        )
    }

    fn srgb_to_linear(value: f32) -> f32 {
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    fn endpoint_linear_color(gaussian: Gaussian3d) -> Vec3 {
        assert!(
            gaussian.spherical_harmonic.coefficients[3..]
                .iter()
                .all(|coefficient| coefficient.abs() <= 1e-6),
            "pixel oracle fixture must remain DC-only"
        );
        let display = Vec3::new(
            0.5 + SH_DC * gaussian.spherical_harmonic.coefficients[0],
            0.5 + SH_DC * gaussian.spherical_harmonic.coefficients[1],
            0.5 + SH_DC * gaussian.spherical_harmonic.coefficients[2],
        )
        .max(Vec3::ZERO);
        Vec3::new(
            srgb_to_linear(display.x),
            srgb_to_linear(display.y),
            srgb_to_linear(display.z),
        )
    }

    fn solve_parent_alpha(pixel: Vec3, color: Vec3) -> Option<f32> {
        let mut total = 0.0;
        let mut count = 0.0;
        for (value, radiance) in [(pixel.x, color.x), (pixel.y, color.y), (pixel.z, color.z)] {
            if radiance > 0.04 {
                total += value / radiance;
                count += 1.0;
            }
        }
        (count > 0.0).then(|| (total / count).clamp(0.0, 0.999))
    }

    fn solve_two_color_coefficients(pixel: Vec3, first: Vec3, second: Vec3) -> Option<(f32, f32)> {
        let aa = first.dot(first);
        let ab = first.dot(second);
        let bb = second.dot(second);
        let determinant = aa * bb - ab * ab;
        if determinant.abs() <= 1e-8 {
            return None;
        }
        let ap = first.dot(pixel);
        let bp = second.dot(pixel);
        let first_coefficient = (ap * bb - bp * ab) / determinant;
        let second_coefficient = (bp * aa - ap * ab) / determinant;
        Some((first_coefficient, second_coefficient))
    }

    fn solve_child_alphas(pixel: Vec3, colors: [Vec3; 2], order: [usize; 2]) -> Option<[f32; 2]> {
        let far = order[0];
        let near = order[1];
        let (near_alpha, far_premultiplied) =
            solve_two_color_coefficients(pixel, colors[near], colors[far])?;
        if !(-0.02..=1.02).contains(&near_alpha)
            || !(-0.02..=1.02).contains(&far_premultiplied)
            || near_alpha >= 0.999
        {
            return None;
        }
        let far_alpha = far_premultiplied / (1.0 - near_alpha);
        if !(-0.02..=1.02).contains(&far_alpha) {
            return None;
        }
        let mut result = [0.0; 2];
        result[near] = near_alpha.clamp(0.0, 0.999);
        result[far] = far_alpha.clamp(0.0, 0.999);
        Some(result)
    }

    fn optical_depth(alpha: f32) -> f32 {
        -(1.0 - alpha.clamp(0.0, 0.999_999)).ln()
    }

    fn interior_proxy(
        parent_alpha: f32,
        child_alpha: f32,
        parent_color: Vec3,
        child_color: Vec3,
        weight: f32,
    ) -> (Vec3, f32) {
        let parent_tau = (1.0 - weight) * 0.5 * optical_depth(parent_alpha);
        let child_tau = weight * optical_depth(child_alpha);
        let total_tau = parent_tau + child_tau;
        if total_tau <= 0.0 {
            return (Vec3::ZERO, 0.0);
        }
        let alpha = (1.0 - (-total_tau).exp()).min(0.999);
        let radiance = (parent_color * parent_tau + child_color * child_tau) / total_tau;
        (radiance * alpha, alpha)
    }

    fn composite_source_over(destination: &mut Vec3, source: Vec3, alpha: f32) {
        *destination = source + *destination * (1.0 - alpha);
    }

    fn assert_tau_weighted_interior(
        parent: Gaussian3d,
        children: [Gaussian3d; 2],
        weight: f32,
        parent_image: &[u8],
        children_image: &[u8],
        interior_image: &[u8],
    ) {
        let parent_color = endpoint_linear_color(parent);
        let child_colors = children.map(endpoint_linear_color);
        let mut best_error = f64::INFINITY;
        let mut compared = 0_u32;
        for order in [[0, 1], [1, 0]] {
            let mut error = 0.0_f64;
            let mut naive_error = 0.0_f64;
            let mut samples = 0_u32;
            for ((parent_pixel, child_pixel), actual_pixel) in parent_image
                .chunks_exact(4)
                .zip(children_image.chunks_exact(4))
                .zip(interior_image.chunks_exact(4))
            {
                let parent_rgb = linear_rgb(parent_pixel);
                let child_rgb = linear_rgb(child_pixel);
                let actual = linear_rgb(actual_pixel);
                let Some(parent_alpha) = solve_parent_alpha(parent_rgb, parent_color) else {
                    continue;
                };
                let Some(child_alphas) = solve_child_alphas(child_rgb, child_colors, order) else {
                    continue;
                };
                if parent_alpha < 0.02 && child_alphas.iter().copied().fold(0.0, f32::max) < 0.02 {
                    continue;
                }
                let mut expected = Vec3::ZERO;
                for child in order {
                    let (premultiplied, alpha) = interior_proxy(
                        parent_alpha,
                        child_alphas[child],
                        parent_color,
                        child_colors[child],
                        weight,
                    );
                    composite_source_over(&mut expected, premultiplied, alpha);
                }
                error += f64::from((actual - expected).abs().element_sum());
                let naive = parent_rgb.lerp(child_rgb, weight);
                naive_error += f64::from((actual - naive).abs().element_sum());
                samples += 1;
            }
            if samples >= 64 {
                let mean = error / f64::from(samples * 3);
                let naive_mean = naive_error / f64::from(samples * 3);
                if mean < best_error {
                    best_error = mean;
                    compared = samples;
                }
                assert!(
                    mean <= 0.002,
                    "tau-weighted K=2 interior mean error {mean:.6} exceeded tolerance for order {order:?}"
                );
                assert!(
                    mean <= 0.5 * naive_mean,
                    "tau-weighted oracle was not at least twice as accurate as naive endpoint interpolation: tau={mean:.6}, naive={naive_mean:.6}, order={order:?}"
                );
                let minimum_resolvable_separation = 2.0 * f64::from(srgb_to_linear(1.0 / 255.0));
                assert!(
                    naive_mean - mean >= minimum_resolvable_separation,
                    "tau-weighted and naive endpoint interpolation differed by less than two resolvable dark-region RGBA8 codes: tau={mean:.6}, naive={naive_mean:.6}, minimum={minimum_resolvable_separation:.6}, order={order:?}"
                );
                break;
            }
        }
        assert!(
            best_error.is_finite() && compared >= 64,
            "insufficient well-conditioned pixels for the tau-weighted oracle: {compared}"
        );
    }

    fn rgb_distance(left: &[u8], right: &[u8]) -> f64 {
        left.chunks_exact(4)
            .zip(right.chunks_exact(4))
            .map(|(left, right)| {
                (0..3)
                    .map(|channel| f64::from(left[channel].abs_diff(right[channel])))
                    .sum::<f64>()
            })
            .sum()
    }

    fn assert_highlight_preserves_unselected_native_radiance(
        parent: Gaussian3d,
        children: [Gaussian3d; 2],
        interior_weight: Option<f32>,
        parent_all: &[u8],
        children_all: &[u8],
        actual_highlight: &[u8],
        label: &str,
    ) {
        assert_eq!(parent_all.len(), children_all.len());
        assert_eq!(children_all.len(), actual_highlight.len());
        assert!(parent.position_visibility.visibility > 0.5);
        let unselected = children
            .iter()
            .position(|child| child.position_visibility.visibility <= 0.5)
            .expect("K=2 Highlight fixture has an unselected child");
        assert_eq!(
            children
                .iter()
                .filter(|child| child.position_visibility.visibility <= 0.5)
                .count(),
            1,
            "K=2 Highlight fixture must have exactly one unselected child"
        );
        assert_eq!(
            children
                .iter()
                .filter(|child| child.position_visibility.visibility > 0.5)
                .count(),
            1,
            "K=2 Highlight fixture must have exactly one selected child"
        );

        let parent_color = endpoint_linear_color(parent);
        let child_colors = children.map(endpoint_linear_color);
        let highlight_color = Vec3::new(0.3, 1.0, 0.1);
        let mut best: Option<(f64, f64, f64, u32, [usize; 2])> = None;
        for order in [[0, 1], [1, 0]] {
            let mut oracle_error = 0.0_f64;
            let mut dropped_error = 0.0_f64;
            let mut recolored_error = 0.0_f64;
            let mut samples = 0_u32;
            for ((parent_pixel, children_pixel), actual_pixel) in parent_all
                .chunks_exact(4)
                .zip(children_all.chunks_exact(4))
                .zip(actual_highlight.chunks_exact(4))
            {
                let parent_rgb = linear_rgb(parent_pixel);
                let children_rgb = linear_rgb(children_pixel);
                let actual = linear_rgb(actual_pixel);
                let Some(child_alphas) = solve_child_alphas(children_rgb, child_colors, order)
                else {
                    continue;
                };
                if child_alphas[unselected] < 0.02 {
                    continue;
                }
                let parent_alpha = if interior_weight.is_some() {
                    let Some(parent_alpha) = solve_parent_alpha(parent_rgb, parent_color) else {
                        continue;
                    };
                    parent_alpha
                } else {
                    0.0
                };

                let mut expected = Vec3::ZERO;
                let mut dropped = Vec3::ZERO;
                let mut recolored = Vec3::ZERO;
                for child in order {
                    let selected = child != unselected;
                    let expected_child_color = if selected {
                        highlight_color
                    } else {
                        child_colors[child]
                    };
                    if let Some(weight) = interior_weight {
                        let (premultiplied, alpha) = interior_proxy(
                            parent_alpha,
                            child_alphas[child],
                            highlight_color,
                            expected_child_color,
                            weight,
                        );
                        composite_source_over(&mut expected, premultiplied, alpha);

                        let dropped_child_alpha = if selected { child_alphas[child] } else { 0.0 };
                        let (premultiplied, alpha) = interior_proxy(
                            parent_alpha,
                            dropped_child_alpha,
                            highlight_color,
                            expected_child_color,
                            weight,
                        );
                        composite_source_over(&mut dropped, premultiplied, alpha);

                        let (premultiplied, alpha) = interior_proxy(
                            parent_alpha,
                            child_alphas[child],
                            highlight_color,
                            highlight_color,
                            weight,
                        );
                        composite_source_over(&mut recolored, premultiplied, alpha);
                    } else {
                        let alpha = child_alphas[child];
                        composite_source_over(&mut expected, expected_child_color * alpha, alpha);
                        if selected {
                            composite_source_over(&mut dropped, highlight_color * alpha, alpha);
                        }
                        composite_source_over(&mut recolored, highlight_color * alpha, alpha);
                    }
                }

                oracle_error += f64::from((actual - expected).abs().element_sum());
                dropped_error += f64::from((actual - dropped).abs().element_sum());
                recolored_error += f64::from((actual - recolored).abs().element_sum());
                samples += 1;
            }
            if samples < 64 {
                continue;
            }
            let denominator = f64::from(samples * 3);
            let means = (
                oracle_error / denominator,
                dropped_error / denominator,
                recolored_error / denominator,
                samples,
                order,
            );
            if best.as_ref().is_none_or(|current| means.0 < current.0) {
                best = Some(means);
            }
        }

        let Some((mean, dropped_mean, recolored_mean, samples, order)) = best else {
            panic!("{label} had fewer than 64 well-conditioned native-red pixels");
        };
        assert!(
            mean <= 0.002,
            "{label} Highlight oracle mean error {mean:.6} exceeded tolerance over {samples} pixels for order {order:?}"
        );
        let minimum_resolvable_separation = 2.0 * f64::from(srgb_to_linear(1.0 / 255.0));
        for (counterfactual, counterfactual_mean) in
            [("dropped", dropped_mean), ("recolored", recolored_mean)]
        {
            assert!(
                mean <= 0.5 * counterfactual_mean,
                "{label} native-red oracle was not at least twice as accurate as the {counterfactual} counterfactual: oracle={mean:.6}, counterfactual={counterfactual_mean:.6}, order={order:?}"
            );
            assert!(
                counterfactual_mean - mean >= minimum_resolvable_separation,
                "{label} native-red oracle and {counterfactual} counterfactual differed by less than two resolvable dark-region RGBA8 codes: oracle={mean:.6}, counterfactual={counterfactual_mean:.6}, minimum={minimum_resolvable_separation:.6}, order={order:?}"
            );
        }
    }

    fn assert_green_population(image: &[u8], label: &str) {
        let mut green_pixels = 0_u32;
        for pixel in image.chunks_exact(4) {
            green_pixels += u32::from(
                pixel[1] > 12 && u16::from(pixel[1]) * 5 > u16::from(pixel[0].max(pixel[2])) * 6,
            );
        }
        assert!(
            green_pixels >= 16,
            "{label} omitted selected green endpoint: {green_pixels}"
        );
    }
}
