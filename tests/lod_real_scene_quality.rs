#[cfg(not(all(feature = "headless", feature = "testing")))]
#[test]
fn lod_real_scene_quality_requires_headless_and_testing_features() {}

#[cfg(all(feature = "headless", feature = "testing"))]
mod headless {
    use std::{
        collections::{BTreeMap, BTreeSet, HashMap, HashSet},
        env,
        fmt::Write as _,
        fs,
        io::{self, BufReader, Read, Seek, SeekFrom},
        path::{Path, PathBuf},
        sync::{Arc, Mutex},
        thread,
        time::{Duration, Instant},
    };

    use bevy::{
        app::{AppExit, ScheduleRunnerPlugin},
        asset::{
            AssetMetaCheck, AssetPlugin, DependencyLoadState, LoadState,
            RecursiveDependencyLoadState, UnapprovedPathMode,
        },
        camera::{PerspectiveProjection, Projection, RenderTarget},
        core_pipeline::tonemapping::Tonemapping,
        prelude::*,
        render::{
            Render, RenderApp, RenderSystems,
            extract_resource::{ExtractResource, ExtractResourcePlugin},
            pipelined_rendering::PipelinedRenderingPlugin,
            render_resource::TextureFormat,
            renderer::{RenderDevice, RenderQueue},
            view::ExtractedView,
            view::screenshot::{Screenshot, ScreenshotCaptured},
        },
        window::ExitCondition,
        winit::WinitPlugin,
    };
    use bevy_gaussian_splatting::{
        CloudSettings, Gaussian3d, GaussianCamera, GaussianLodBridgeConfig,
        GaussianLodBuildSettings, GaussianLodLifecycle, GaussianLodManifest, GaussianLodSettings,
        GaussianLodSourceKind, GaussianLodStatus, GaussianMode, GaussianSplattingPlugin,
        LodDegradation, LodEffectiveStatus, LodNodeId, LodPageDescriptor, LodPageId, LodPageKind,
        LodQualityTarget, LodReducerKind, PlanarGaussian3d, PlanarGaussian3dHandle,
        PlanarGaussian3dLod, PlanarGaussian3dPage, PlanarHandle, SphericalHarmonicCoefficients,
        build_planar_3d_lod,
        gaussian::{covariance::compute_covariance_3d, settings::GaussianColorSpace},
        io::{
            IoPlugin,
            lod::{LodCodecLimits, decode_manifest, decode_page_with_descriptor},
            ply::stream_ply_3d,
            scene::GaussianScene,
        },
        material::spherical_harmonics::SH_DEGREE,
        render::{
            ShaderDefines, gaussian_mip_filter_covariance_2d,
            lod::{
                LodCompactionBuffers, LodIndirectArgs, LodLastRadixDrawableForTesting,
                LodViewBlendPublicationLabel, finalized_indirect_args,
                read_lod_indirect_args_for_testing,
            },
        },
        sort::SortMode,
        stream::{
            bridge::{GaussianLodBridgePhase, GaussianLodBridgeStatus},
            hierarchy::{
                AllResident, LodHierarchy, LodView, ManifestLodHierarchy, select_frontier,
                select_frontier_with_visibility,
            },
            render_commit::LodRenderCandidates,
        },
        testing::{
            BoundaryBandMetrics, ImageMetrics, LodProjection, LodTestCamera,
            SpatialResidualMetrics, compare_linear_rgba, compare_node_boundary_bands,
            gather_frontier_gaussians_with_nodes,
            render_production_flat_linear_gaussians_with_owners,
            render_production_lod_linear_gaussians,
        },
    };
    use sha2::{Digest, Sha256};

    const WIDTH: u32 = 192;
    const HEIGHT: u32 = 192;
    const FOV_Y: f32 = 60.0_f32.to_radians();
    const SYNTHETIC_SOURCE_COUNT: usize = 4_096;
    const REFERENCE_WARMUP_FRAMES: u32 = 45;
    const RESTORED_WARMUP_FRAMES: u32 = 18;
    // Require several active frames so image metrics sample a settled cut.
    const STABLE_ACTIVE_FRAMES: u32 = 30;
    const MAX_FRAMES: u32 = 900;
    const FOREGROUND_ALPHA: f32 = 0.02;
    const SPILL_DILATION_PX: usize = 2;

    const TRELLIS_ENV: &str = "BGS_TRELLIS_GLB";
    const TRELLIS_REPORT_ENV: &str = "BGS_LOD_REPORT_PATH";
    const TRELLIS_AUDIT_PROFILE_ENV: &str = "BGS_TRELLIS_AUDIT_PROFILE";
    const TRELLIS_BYTE_LEN: u64 = 112_899_460;
    const TRELLIS_SPLAT_COUNT: usize = 478_368;
    // Active cuts change by whole hierarchy nodes. Allow one default two-leaf
    // (2 * 64 record) domain of integer granularity beyond the 5% curve bound.
    const TRELLIS_CONTINUITY_DISCRETE_RECORD_SLACK: usize = 128;
    const TRELLIS_SHA256_FOR_PREFLIGHT: &str =
        "fbe9d96b6689a78228c121e5f1bc8c5ccc32cef1941294d25f1db66f4a901dc1";
    const TRELLIS_RASTER_SIZE: u32 = 192;
    const TRELLIS_DEPLOYMENT_VIEWPORT_HEIGHT_PX: f32 = 1_080.0;
    const TRELLIS_FOV_Y: f32 = 45.0_f32.to_radians();
    const TRELLIS_MORPHOLOGY_QUALITIES: [f32; 7] = [0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1.0];
    // At the 192px oracle resolution, an opacity-visible 3-sigma footprint
    // wider than 24px with an 8:1 axis ratio is an obvious needle-like
    // representative rather than sub-pixel covariance noise.
    const VISIBLE_ELONGATION_MIN_MAJOR_SIGMA_PX: f32 = 4.0;
    const VISIBLE_ELONGATION_MIN_ASPECT_RATIO: f32 = 8.0;

    const GARDEN_LOD_ENV: &str = "BGS_GARDEN_LOD";
    const GARDEN_PLY_ENV: &str = "BGS_GARDEN_PLY";
    const GARDEN_SOURCE_BYTE_LEN: u64 = 1_447_027_964;
    const GARDEN_SOURCE_GAUSSIANS: u64 = 5_834_784;
    const GARDEN_SOURCE_SHA256: &str =
        "16701d5e0630dfaca74f8794ed7ce2aa23fa922f87dc09a7e37484e8d3f82d5a";
    const GARDEN_MANIFEST_SHA256: &str =
        "67b9119222e1435fb88755698dcd916e608c9cd21c1417b687a7cce663729600";
    const GARDEN_SHARDS: [(&str, u64, &str); 3] = [
        (
            "pages/shard-000000.bgslodpack",
            536_660_028,
            "d8884945ff558d8a231d48511900f9cc97df407c9bd442d1a8ab35bc9a0766ea",
        ),
        (
            "pages/shard-000001.bgslodpack",
            536_660_028,
            "1232414ca7f0addbd4524516d06c205832468f685eb897c60e53412e24608504",
        ),
        (
            "pages/shard-000002.bgslodpack",
            527_570_716,
            "cdc3c896fba1f1aae469c09e913ba075c824fb6b8e0434b08206b48a03c9a8b2",
        ),
    ];
    const GARDEN_ABI16: u32 = 16;
    const GARDEN_MOMENT_MERGE_V4: u32 = 4;
    const GARDEN_NODE_PAGE_COUNT: u32 = 6_517;
    const GARDEN_STORED_GAUSSIANS: u64 = 6_668_314;
    const GARDEN_MIN: [f32; 3] = [-118.729_54, -130.432_02, -121.283_48];
    const GARDEN_MAX: [f32; 3] = [137.847_32, 109.880_554, 136.600_8];
    const GARDEN_CENTER: Vec3 = Vec3::new(9.558_891, -10.275_734, 7.658_661);
    const GARDEN_RADIUS: f32 = 217.994_34;
    const GARDEN_AUTO_FRAME_DISTANCE: f32 = 474.641_1;
    const GARDEN_BOUNDARY_WIDTH: u32 = 1_920;
    const GARDEN_BOUNDARY_HEIGHT: u32 = 1_080;
    const GARDEN_VIEWPORT_HEIGHT_PX: f32 = 1_080.0;
    const GARDEN_VIEWPORT_ASPECT: f32 = 16.0 / 9.0;
    const GARDEN_VIEWER_FOV: f32 = std::f32::consts::FRAC_PI_4;
    const GARDEN_VIEWER_NEAR: f32 = 0.1;
    const GARDEN_BOUNDARY_BAND_RADIUS: u32 = 1;
    const GARDEN_MIN_MATCHED_PIXELS: usize = 128;
    // A boundary-specific two-percentage-point signed shift is visible in
    // linear alpha/luminance even when the whole-frame score remains good.
    const GARDEN_MAX_MATCHED_SIGNED_BIAS_GAP: f64 = 0.02;
    const GARDEN_MAX_REFERENCE_MATCH_GAP: f64 = 0.05;
    // Regularization makes an exact/near-exact interior control meaningful:
    // the gate still limits boundary error to 0.5% when the control is zero.
    const GARDEN_ENRICHMENT_FLOOR: f64 = 0.005;
    const GARDEN_MAX_MATCHED_ENRICHMENT: f64 = 2.0;
    const SYNTHETIC_COARSE_QUALITY: f32 = 0.65;
    const SYNTHETIC_COARSE_THRESHOLDS: QualityThresholds = QualityThresholds {
        minimum_foreground_psnr: 30.0,
        minimum_ssim: 0.95,
        minimum_iou: 0.95,
        maximum_alpha_mae: 0.02,
        maximum_spill: 0.01,
    };
    const Q95_THRESHOLDS: QualityThresholds = QualityThresholds {
        minimum_foreground_psnr: 38.0,
        minimum_ssim: 0.99,
        minimum_iou: 0.99,
        maximum_alpha_mae: 0.005,
        maximum_spill: 0.005,
    };
    const Q99_THRESHOLDS: QualityThresholds = QualityThresholds {
        minimum_foreground_psnr: 40.0,
        minimum_ssim: 0.995,
        minimum_iou: 0.995,
        maximum_alpha_mae: 0.003,
        maximum_spill: 0.003,
    };

    /// Exercises the production GPU path with thin, curved diagonal ribbons. The
    /// source splats have strongly oriented covariance, so a transposed or
    /// over-inflated coarse covariance produces visible pixels perpendicular to
    /// the ribbons and fails the spill gate even when a whole-image PSNR is
    /// diluted by the transparent background.
    #[test]
    fn diagonal_ribbons_keep_high_quality_covariance_on_the_gpu() {
        if env::var("RUN_GPU_RENDER_TESTS").ok().as_deref() != Some("1") {
            eprintln!(
                "skipping real-pattern LoD GPU regression; set RUN_GPU_RENDER_TESTS=1 to enable"
            );
            return;
        }

        let mut app = App::new();
        let render_probe = GpuRenderProbe::default();
        app.insert_resource(ClearColor(Color::linear_rgba(0.0, 0.0, 0.0, 0.0)))
            .insert_resource(synthetic_bridge_config())
            .insert_resource(GpuQualityState::default())
            .insert_resource(render_probe);
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: "assets".to_owned(),
                    processed_file_path: "assets".to_owned(),
                    meta_check: AssetMetaCheck::Never,
                    unapproved_path_mode: UnapprovedPathMode::Allow,
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
            1.0 / 60.0,
        )));
        app.add_plugins((
            GaussianSplattingPlugin,
            ExtractResourcePlugin::<GpuRenderProbe>::default(),
        ))
        .add_systems(Startup, setup_gpu_fixture)
        .add_systems(Update, drive_gpu_capture)
        .add_observer(on_gpu_capture);
        app.sub_app_mut(RenderApp).add_systems(
            Render,
            capture_gpu_render_proof
                .after(LodViewBlendPublicationLabel)
                .in_set(RenderSystems::Cleanup),
        );
        app.run();
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum GpuPhase {
        ReferenceWarmup,
        ReferencePending,
        CoarseWaiting,
        CoarsePending,
        Quality95Waiting,
        Quality95Pending,
        Quality99Waiting,
        Quality99Pending,
        RestoredWaiting,
        RestoredPending,
    }

    #[derive(Resource)]
    struct GpuQualityState {
        phase: GpuPhase,
        phase_frames: u32,
        total_frames: u32,
        stable_frames: u32,
        stable_identity: Option<GpuMainDrawableIdentity>,
        target: Option<Handle<Image>>,
        cloud: Option<Entity>,
        camera: Option<Entity>,
        reference: Option<Vec<[f32; 4]>>,
        coarse: Option<GpuCapture>,
        quality95: Option<GpuCapture>,
        quality99: Option<GpuCapture>,
        pending_active_gaussians: u64,
        pending_request: Option<Entity>,
        pending_main_proof: Option<GpuMainDrawableIdentity>,
    }

    impl Default for GpuQualityState {
        fn default() -> Self {
            Self {
                phase: GpuPhase::ReferenceWarmup,
                phase_frames: 0,
                total_frames: 0,
                stable_frames: 0,
                stable_identity: None,
                target: None,
                cloud: None,
                camera: None,
                reference: None,
                coarse: None,
                quality95: None,
                quality99: None,
                pending_active_gaussians: 0,
                pending_request: None,
                pending_main_proof: None,
            }
        }
    }

    impl GpuQualityState {
        fn enter(&mut self, phase: GpuPhase) {
            self.phase = phase;
            self.phase_frames = 0;
            self.stable_frames = 0;
            self.stable_identity = None;
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct GpuMainDrawableIdentity {
        status_revision: u64,
        requested_target: LodQualityTarget,
        selected_gaussians: u64,
        submitted_candidates: u32,
        render_commit_identity: usize,
    }

    #[derive(Clone, Debug)]
    struct GpuRenderDrawableProof {
        requested_target: LodQualityTarget,
        rendered_quality: LodEffectiveStatus,
        frontier_candidate_count: u32,
        rendered_candidate_count: u32,
        compaction_candidate_count: u32,
        render_commit_identity: usize,
        candidate_active: bool,
        candidate_transitioning: bool,
        candidate_failed: bool,
        drawable: LodLastRadixDrawableForTesting,
        indirect: LodIndirectArgs,
        expected_indirect: LodIndirectArgs,
    }

    #[derive(Default)]
    struct GpuRenderProbeShared {
        armed_requests: HashSet<Entity>,
        latched_requests: HashMap<Entity, Option<GpuRenderDrawableProof>>,
    }

    #[derive(Resource, Clone, ExtractResource, Default)]
    struct GpuRenderProbe(Arc<Mutex<GpuRenderProbeShared>>);

    impl GpuRenderProbe {
        fn arm(&self, request: Entity) {
            let mut shared = self
                .0
                .lock()
                .expect("real-pattern render probe mutex is not poisoned");
            assert!(
                shared.armed_requests.insert(request),
                "screenshot request {request:?} was armed twice"
            );
            assert!(
                !shared.latched_requests.contains_key(&request),
                "screenshot request {request:?} reused an old render latch"
            );
        }

        fn take_latched(&self, request: Entity) -> Option<Option<GpuRenderDrawableProof>> {
            self.0.lock().ok()?.latched_requests.remove(&request)
        }
    }

    struct GpuCapture {
        active_gaussians: u64,
        image: Vec<[f32; 4]>,
    }

    fn synthetic_bridge_config() -> GaussianLodBridgeConfig {
        GaussianLodBridgeConfig {
            max_ephemeral_source_gaussians: 8_192,
            max_ephemeral_stored_gaussians: 16_384,
            max_atlas_gaussians: 16_384,
            max_atlas_bytes: 64 * 1024 * 1024,
            build_settings: GaussianLodBuildSettings {
                branching_factor: 8,
                leaf_capacity: 128,
                support_sigma: 3.0,
            },
            ..default()
        }
    }

    fn synthetic_lod_settings() -> GaussianLodSettings {
        let mut settings = GaussianLodSettings {
            quality: 1.0,
            hysteresis: 0.0,
            frustum_culling: false,
            ..default()
        };
        settings.budgets.max_active_gaussians = 8_192;
        settings.budgets.max_resident_gaussians = 16_384;
        settings.budgets.max_resident_bytes = 64 * 1024 * 1024;
        settings.budgets.max_resident_pages = 2_048;
        settings.budgets.max_pending_requests = 2_048;
        settings.budgets.max_requests_per_frame = 512;
        settings.budgets.max_upload_bytes_per_frame = 64 * 1024 * 1024;
        settings
    }

    fn setup_gpu_fixture(
        mut commands: Commands,
        mut state: ResMut<GpuQualityState>,
        mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
        mut images: ResMut<Assets<Image>>,
    ) {
        let cloud = diagonal_ribbon_cloud();
        assert_eq!(cloud.position_visibility.len(), SYNTHETIC_SOURCE_COUNT);
        let target = images.add(Image::new_target_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        let cloud = commands
            .spawn((
                PlanarGaussian3dHandle(gaussian_assets.add(cloud)),
                CloudSettings {
                    gaussian_mode: GaussianMode::Gaussian3d,
                    sort_mode: SortMode::Radix,
                    global_opacity: 1.0,
                    global_scale: 1.0,
                    opacity_adaptive_radius: false,
                    ..default()
                },
                synthetic_lod_settings(),
                Transform::IDENTITY,
                Visibility::Visible,
                Name::new("lod_covariance_diagonal_ribbons"),
            ))
            .id();
        let camera = commands
            .spawn((
                Camera3d::default(),
                Camera::default(),
                Projection::Perspective(PerspectiveProjection {
                    fov: FOV_Y,
                    near: 0.01,
                    far: 100.0,
                    ..default()
                }),
                RenderTarget::Image(target.clone().into()),
                Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)),
                Tonemapping::None,
                GaussianCamera::default(),
                Name::new("lod_covariance_ribbon_camera"),
            ))
            .id();
        state.target = Some(target);
        state.cloud = Some(cloud);
        state.camera = Some(camera);
    }

    fn capture_gpu_render_proof(
        render_device: Res<RenderDevice>,
        render_queue: Res<RenderQueue>,
        buffers: Res<LodCompactionBuffers<Gaussian3d>>,
        views: Query<&ExtractedView, With<GaussianCamera>>,
        clouds: Query<(
            Entity,
            &PlanarGaussian3dHandle,
            &GaussianLodSettings,
            &CloudSettings,
            &LodRenderCandidates,
        )>,
        probe: Res<GpuRenderProbe>,
    ) {
        if probe
            .0
            .lock()
            .expect("real-pattern render probe mutex is not poisoned")
            .armed_requests
            .is_empty()
        {
            return;
        }

        let mut latest = None;
        for view in &views {
            let camera = view.retained_view_entity.main_entity.id();
            for (cloud, handle, settings, cloud_settings, candidates) in &clouds {
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
                let indirect =
                    read_lod_indirect_args_for_testing(&render_device, &render_queue, compaction)
                        .unwrap_or_else(|error| {
                            panic!("real-pattern indirect-args readback failed: {error}")
                        });
                let defines =
                    ShaderDefines::for_radix_depth_bits(cloud_settings.radix_sort_depth_bits);
                let expected_indirect = finalized_indirect_args(
                    drawable.rendered_candidate_count,
                    compaction.output_capacity(),
                    defines.radix_base * defines.entries_per_invocation_a,
                    defines.workgroup_entries_c,
                );
                assert!(
                    latest.is_none(),
                    "real-pattern quality gate expected one retained view/cloud consumer"
                );
                latest = Some(GpuRenderDrawableProof {
                    requested_target: settings.quality_target(),
                    rendered_quality: candidate.rendered_quality_status(),
                    frontier_candidate_count: candidate.frontier().candidate_count(),
                    rendered_candidate_count: candidate.rendered_candidate_count(),
                    compaction_candidate_count: compaction.candidate_count(),
                    render_commit_identity: candidate.render_commit_identity_for_testing(),
                    candidate_active: candidate.render_is_active_for_testing(),
                    candidate_transitioning: candidate.render_is_transitioning_for_testing(),
                    candidate_failed: candidate.failed(),
                    drawable,
                    indirect,
                    expected_indirect,
                });
            }
        }

        let mut shared = probe
            .0
            .lock()
            .expect("real-pattern render probe mutex is not poisoned");
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

    fn current_gpu_main_identity(
        bridge: &GaussianLodBridgeStatus,
        status: &GaussianLodStatus,
        settings: &GaussianLodSettings,
        candidates: &LodRenderCandidates,
        camera: Entity,
    ) -> Option<GpuMainDrawableIdentity> {
        let candidate = candidates.get(camera)?;
        let requested_target = settings.quality_target();
        let rendered_quality = candidate.rendered_quality_status();
        let submitted_candidates = candidate.rendered_candidate_count();
        if bridge.phase != GaussianLodBridgePhase::Active
            || bridge.failure.is_some()
            || bridge.active_views != 1
            || bridge.active_gaussians != u64::from(submitted_candidates)
            || status.source != GaussianLodSourceKind::Ephemeral
            || status.lifecycle != GaussianLodLifecycle::Active
            || status.active_views != 1
            || status.requested_target != requested_target
            || status.target_satisfied != Some(true)
            || status.degradation != LodDegradation::None
            || status.view_blend_edges != 0
            || status.view_blend_lagging_edges != 0
            || status.view_blend_invalid_pressure_evaluations != 0
            || status.view_blend_missing_consumers != 0
            || status.submitted_candidates != submitted_candidates
            || status.selected_gaussians != rendered_quality.active_gaussians
            || rendered_quality.requested_target != requested_target
            || rendered_quality.degradation != LodDegradation::None
            || rendered_quality.achieved_max_target_ratio > 1.0
            || !candidate.render_is_active_for_testing()
            || candidate.failed()
        {
            return None;
        }
        Some(GpuMainDrawableIdentity {
            status_revision: status.revision,
            requested_target,
            selected_gaussians: status.selected_gaussians,
            submitted_candidates,
            render_commit_identity: candidate.render_commit_identity_for_testing(),
        })
    }

    fn gpu_render_proof_matches_main(
        render: &GpuRenderDrawableProof,
        main: &GpuMainDrawableIdentity,
    ) -> bool {
        render.requested_target == main.requested_target
            && render.rendered_quality.requested_target == main.requested_target
            && render.rendered_quality.degradation == LodDegradation::None
            && render.rendered_quality.achieved_max_target_ratio <= 1.0
            && render.rendered_quality.active_gaussians == main.selected_gaussians
            && u64::from(render.frontier_candidate_count) == main.selected_gaussians
            && render.rendered_candidate_count == main.submitted_candidates
            && render.compaction_candidate_count == main.submitted_candidates
            && render.render_commit_identity == main.render_commit_identity
            && render.candidate_active
            && !render.candidate_transitioning
            && !render.candidate_failed
            && render.drawable.candidate_token_matches
            && render.drawable.candidate_content_matches
            && render.drawable.rendered_candidate_count == main.submitted_candidates
            && render.drawable.morph_identity.is_none()
            && render.drawable.view_blend.is_none()
            && render.indirect == render.expected_indirect
            && render.indirect.instance_count != 0
    }

    fn drive_gpu_capture(
        mut commands: Commands,
        mut state: ResMut<GpuQualityState>,
        bridge_statuses: Query<&GaussianLodBridgeStatus>,
        statuses: Query<&GaussianLodStatus>,
        settings: Query<&GaussianLodSettings>,
        candidates: Query<&LodRenderCandidates>,
        probe: Res<GpuRenderProbe>,
    ) {
        state.total_frames += 1;
        state.phase_frames += 1;
        assert!(
            state.total_frames <= MAX_FRAMES,
            "real-pattern LoD GPU regression timed out in {:?}; status={:?}",
            state.phase,
            state
                .cloud
                .and_then(|cloud| bridge_statuses.get(cloud).ok())
        );

        let cloud = state.cloud.expect("ribbon cloud exists");
        match state.phase {
            GpuPhase::ReferenceWarmup if state.phase_frames >= REFERENCE_WARMUP_FRAMES => {
                assert!(
                    bridge_statuses.get(cloud).is_err(),
                    "quality one unexpectedly retained a LoD bridge"
                );
                state.enter(GpuPhase::ReferencePending);
                request_gpu_capture(&mut commands, &mut state, &probe, None);
            }
            GpuPhase::CoarseWaiting | GpuPhase::Quality95Waiting | GpuPhase::Quality99Waiting => {
                let (Ok(bridge), Ok(status), Ok(settings), Ok(candidates)) = (
                    bridge_statuses.get(cloud),
                    statuses.get(cloud),
                    settings.get(cloud),
                    candidates.get(cloud),
                ) else {
                    state.stable_frames = 0;
                    state.stable_identity = None;
                    return;
                };
                assert!(bridge.failure.is_none(), "LoD bridge failed: {bridge:?}");
                let Some(identity) = current_gpu_main_identity(
                    bridge,
                    status,
                    settings,
                    candidates,
                    state.camera.expect("ribbon camera exists"),
                ) else {
                    state.stable_frames = 0;
                    state.stable_identity = None;
                    return;
                };
                if state.stable_identity.as_ref() == Some(&identity) {
                    state.stable_frames += 1;
                } else {
                    state.pending_active_gaussians = u64::from(identity.submitted_candidates);
                    state.stable_identity = Some(identity.clone());
                    state.stable_frames = 1;
                }
                if state.stable_frames >= STABLE_ACTIVE_FRAMES {
                    let pending = match state.phase {
                        GpuPhase::CoarseWaiting => GpuPhase::CoarsePending,
                        GpuPhase::Quality95Waiting => GpuPhase::Quality95Pending,
                        GpuPhase::Quality99Waiting => GpuPhase::Quality99Pending,
                        _ => unreachable!(),
                    };
                    state.enter(pending);
                    request_gpu_capture(&mut commands, &mut state, &probe, Some(identity));
                }
            }
            GpuPhase::RestoredWaiting => {
                if bridge_statuses.get(cloud).is_err() {
                    state.stable_frames += 1;
                } else {
                    state.stable_frames = 0;
                }
                if state.stable_frames >= RESTORED_WARMUP_FRAMES {
                    state.enter(GpuPhase::RestoredPending);
                    request_gpu_capture(&mut commands, &mut state, &probe, None);
                }
            }
            GpuPhase::ReferenceWarmup
            | GpuPhase::ReferencePending
            | GpuPhase::CoarsePending
            | GpuPhase::Quality95Pending
            | GpuPhase::Quality99Pending
            | GpuPhase::RestoredPending => {}
        }
    }

    fn request_gpu_capture(
        commands: &mut Commands,
        state: &mut GpuQualityState,
        probe: &GpuRenderProbe,
        main_proof: Option<GpuMainDrawableIdentity>,
    ) {
        assert!(
            state.pending_request.is_none() && state.pending_main_proof.is_none(),
            "real-pattern quality gate already has a screenshot in flight"
        );
        let request = commands
            .spawn(Screenshot::image(
                state.target.clone().expect("render target exists"),
            ))
            .id();
        probe.arm(request);
        state.pending_request = Some(request);
        state.pending_main_proof = main_proof;
    }

    fn on_gpu_capture(
        trigger: On<ScreenshotCaptured>,
        mut state: ResMut<GpuQualityState>,
        probe: Res<GpuRenderProbe>,
        mut settings: Query<&mut GaussianLodSettings>,
        mut exit: MessageWriter<AppExit>,
    ) {
        let request = state
            .pending_request
            .take()
            .expect("screenshot completed without an in-flight request");
        assert_eq!(
            trigger.entity, request,
            "screenshot callback did not match the in-flight request"
        );
        let main_proof = state.pending_main_proof.take();
        let render_proof = probe.take_latched(request).unwrap_or_else(|| {
            panic!(
                "screenshot request {request:?} completed without its request-frame Render Cleanup latch"
            )
        });
        let proof_matches = match (&main_proof, &render_proof) {
            (Some(main), Some(render)) => gpu_render_proof_matches_main(render, main),
            (None, None) => true,
            _ => false,
        };
        if !proof_matches {
            let waiting = match state.phase {
                GpuPhase::ReferencePending => GpuPhase::ReferenceWarmup,
                GpuPhase::CoarsePending => GpuPhase::CoarseWaiting,
                GpuPhase::Quality95Pending => GpuPhase::Quality95Waiting,
                GpuPhase::Quality99Pending => GpuPhase::Quality99Waiting,
                GpuPhase::RestoredPending => GpuPhase::RestoredWaiting,
                phase => panic!("screenshot captured during unexpected phase {phase:?}"),
            };
            state.enter(waiting);
            return;
        }

        let image = linear_rgba(
            trigger
                .image
                .clone()
                .try_into_dynamic()
                .expect("screenshot converts")
                .to_rgba8()
                .as_raw(),
        );
        assert!(
            image.iter().any(|pixel| pixel[3] > FOREGROUND_ALPHA),
            "real-pattern LoD capture stayed transparent in {:?}",
            state.phase
        );
        let cloud = state.cloud.expect("ribbon cloud exists");
        match state.phase {
            GpuPhase::ReferencePending => {
                state.reference = Some(image);
                settings
                    .get_mut(cloud)
                    .expect("ribbon LoD settings exist")
                    .quality = SYNTHETIC_COARSE_QUALITY;
                state.enter(GpuPhase::CoarseWaiting);
            }
            GpuPhase::CoarsePending => {
                let active_gaussians = state.pending_active_gaussians;
                assert!(
                    active_gaussians < SYNTHETIC_SOURCE_COUNT as u64,
                    "q={SYNTHETIC_COARSE_QUALITY:.2} did not activate a material LoD cut: active={active_gaussians}, source={SYNTHETIC_SOURCE_COUNT}"
                );
                state.coarse = Some(GpuCapture {
                    active_gaussians,
                    image,
                });
                settings
                    .get_mut(cloud)
                    .expect("ribbon LoD settings exist")
                    .quality = 0.95;
                state.enter(GpuPhase::Quality95Waiting);
            }
            GpuPhase::Quality95Pending => {
                let active_gaussians = state.pending_active_gaussians;
                assert!(
                    active_gaussians <= SYNTHETIC_SOURCE_COUNT as u64,
                    "q=.95 exceeded the exact source count: active={active_gaussians}, source={SYNTHETIC_SOURCE_COUNT}"
                );
                state.quality95 = Some(GpuCapture {
                    active_gaussians,
                    image,
                });
                settings
                    .get_mut(cloud)
                    .expect("ribbon LoD settings exist")
                    .quality = 0.99;
                state.enter(GpuPhase::Quality99Waiting);
            }
            GpuPhase::Quality99Pending => {
                let active_gaussians = state.pending_active_gaussians;
                state.quality99 = Some(GpuCapture {
                    active_gaussians,
                    image,
                });
                assert_gpu_quality(&state);
                settings
                    .get_mut(cloud)
                    .expect("ribbon LoD settings exist")
                    .quality = 1.0;
                state.enter(GpuPhase::RestoredWaiting);
            }
            GpuPhase::RestoredPending => {
                let reference = state.reference.as_ref().expect("reference exists");
                let restored = compare_linear_rgba(reference, &image, FOREGROUND_ALPHA)
                    .expect("restored reference dimensions match");
                assert!(
                    restored.psnr_rgb.is_infinite() && restored.max_abs_error == 0.0,
                    "q=1 did not restore the exact source pixels: {restored:?}"
                );
                exit.write(AppExit::Success);
            }
            phase => panic!("screenshot captured during unexpected phase {phase:?}"),
        }
    }

    fn assert_gpu_quality(state: &GpuQualityState) {
        let reference = state.reference.as_ref().expect("reference exists");
        let coarse = state.coarse.as_ref().expect("coarse capture exists");
        let q95 = state.quality95.as_ref().expect("q=.95 capture exists");
        let q99 = state.quality99.as_ref().expect("q=.99 capture exists");
        assert!(coarse.active_gaussians < SYNTHETIC_SOURCE_COUNT as u64);
        assert!(q95.active_gaussians >= coarse.active_gaussians);
        assert!(q99.active_gaussians >= q95.active_gaussians);

        let coarse_observation = quality_observation(reference, &coarse.image, WIDTH, HEIGHT);
        let q95_observation = quality_observation(reference, &q95.image, WIDTH, HEIGHT);
        let q99_observation = quality_observation(reference, &q99.image, WIDTH, HEIGHT);
        eprintln!(
            "diagonal-ribbon LoD quality: q{SYNTHETIC_COARSE_QUALITY:.2} active={} {:?}; q95 active={} {:?}; q99 active={} {:?}",
            coarse.active_gaussians,
            coarse_observation,
            q95.active_gaussians,
            q95_observation,
            q99.active_gaussians,
            q99_observation
        );
        assert_thresholds(
            "synthetic coarse cut",
            coarse_observation,
            SYNTHETIC_COARSE_THRESHOLDS,
        );
        assert_alpha_morphology_bound("synthetic coarse cut", coarse_observation);
        assert_thresholds("synthetic q=.95", q95_observation, Q95_THRESHOLDS);
        assert_thresholds("synthetic q=.99", q99_observation, Q99_THRESHOLDS);
        assert_monotonic(
            "synthetic coarse -> q=.95",
            coarse_observation,
            q95_observation,
        );
        assert_monotonic("synthetic q=.95 -> q=.99", q95_observation, q99_observation);
    }

    fn diagonal_ribbon_cloud() -> PlanarGaussian3d {
        const RIBBONS: usize = 2;
        const LANES: usize = 4;
        const SAMPLES: usize = SYNTHETIC_SOURCE_COUNT / (RIBBONS * LANES);
        let mut gaussians = Vec::with_capacity(SYNTHETIC_SOURCE_COUNT);
        for ribbon in 0..RIBBONS {
            for lane in 0..LANES {
                for sample in 0..SAMPLES {
                    let t = sample as f32 / (SAMPLES - 1) as f32;
                    let x = -1.55 + 3.10 * t;
                    let phase = if ribbon == 0 { 0.0 } else { 1.7 };
                    let offset = if ribbon == 0 { -0.24 } else { 0.24 };
                    let y = 0.48 * x + offset + 0.13 * (t * 8.0 + phase).sin();
                    let tangent =
                        Vec2::new(3.10, 1.488 + 1.04 * (t * 8.0 + phase).cos()).normalize_or_zero();
                    let normal = Vec2::new(-tangent.y, tangent.x);
                    let lane_offset = (lane as f32 - 1.5) * 0.009;
                    let position = Vec3::new(
                        x + normal.x * lane_offset,
                        y + normal.y * lane_offset,
                        (ribbon as f32 - 0.5) * 0.006,
                    );
                    let angle = tangent.y.atan2(tangent.x);
                    let rotation = Quat::from_rotation_z(angle);
                    let color = if ribbon == 0 {
                        [0.95, 0.08, 0.04]
                    } else {
                        [0.04, 0.45, 0.95]
                    };
                    gaussians.push(test_gaussian(
                        position,
                        Vec3::new(0.020, 0.0055, 0.003),
                        rotation,
                        color,
                        0.72,
                    ));
                }
            }
        }
        gaussians.into()
    }

    fn test_gaussian(
        position: Vec3,
        scale: Vec3,
        rotation: Quat,
        color: [f32; 3],
        opacity: f32,
    ) -> Gaussian3d {
        let mut coefficients = SphericalHarmonicCoefficients::default();
        for (index, value) in color.into_iter().enumerate() {
            coefficients.set(index, value);
        }
        Gaussian3d {
            position_visibility: [position.x, position.y, position.z, 1.0].into(),
            // Canonical Gaussian rotations store the scalar component first.
            rotation: [rotation.w, rotation.x, rotation.y, rotation.z].into(),
            scale_opacity: [scale.x, scale.y, scale.z, opacity].into(),
            spherical_harmonic: coefficients,
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum TrellisAuditProfile {
        Pr,
        Full,
    }

    impl TrellisAuditProfile {
        fn from_environment() -> Self {
            match env::var(TRELLIS_AUDIT_PROFILE_ENV) {
                Ok(value) if value == "pr" => Self::Pr,
                Ok(value) if value == "full" => Self::Full,
                Ok(value) => panic!(
                    "unsupported {TRELLIS_AUDIT_PROFILE_ENV}={value:?}; expected `pr` or `full`"
                ),
                Err(env::VarError::NotPresent) => Self::Pr,
                Err(env::VarError::NotUnicode(_)) => {
                    panic!("{TRELLIS_AUDIT_PROFILE_ENV} must be valid Unicode")
                }
            }
        }

        const fn as_str(self) -> &'static str {
            match self {
                Self::Pr => "pr",
                Self::Full => "full",
            }
        }

        fn rendered_qualities(self) -> Vec<f32> {
            let steps = match self {
                Self::Pr => 20,
                Self::Full => 100,
            };
            let mut qualities = (0..=steps)
                .map(|step| step as f32 / steps as f32)
                .collect::<Vec<_>>();
            qualities.push(0.99);
            qualities.sort_by(f32::total_cmp);
            qualities.dedup_by(|left, right| (*left - *right).abs() <= 1e-6);
            qualities
        }

        fn orbit_specs(self) -> &'static [TrellisOrbitSpec] {
            match self {
                Self::Pr => &PR_TRELLIS_ORBIT_SPECS,
                Self::Full => &FULL_TRELLIS_ORBIT_SPECS,
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct TrellisOrbitSpec {
        label: &'static str,
        yaw_degrees: f32,
        pitch_degrees: f32,
    }

    const PR_TRELLIS_ORBIT_SPECS: [TrellisOrbitSpec; 2] = [
        TrellisOrbitSpec {
            label: "orbit-left-40",
            yaw_degrees: -40.0,
            pitch_degrees: 0.0,
        },
        TrellisOrbitSpec {
            label: "orbit-right-40",
            yaw_degrees: 40.0,
            pitch_degrees: 0.0,
        },
    ];

    const FULL_TRELLIS_ORBIT_SPECS: [TrellisOrbitSpec; 6] = [
        PR_TRELLIS_ORBIT_SPECS[0],
        PR_TRELLIS_ORBIT_SPECS[1],
        TrellisOrbitSpec {
            label: "orbit-left-75",
            yaw_degrees: -75.0,
            pitch_degrees: 0.0,
        },
        TrellisOrbitSpec {
            label: "orbit-right-75",
            yaw_degrees: 75.0,
            pitch_degrees: 0.0,
        },
        TrellisOrbitSpec {
            label: "orbit-high-25",
            yaw_degrees: 0.0,
            pitch_degrees: 25.0,
        },
        TrellisOrbitSpec {
            label: "orbit-low-25",
            yaw_degrees: 0.0,
            pitch_degrees: -25.0,
        },
    ];

    /// Authenticated CPU-only seam oracle for the canonical Garden source and
    /// its ABI-16 native package. The attributed image uses authored SH color;
    /// Page debug colors are deliberately absent because their palette luma
    /// would confound a density-bias measurement.
    ///
    /// Raw node-boundary/interior metrics remain useful diagnostics, but real
    /// scene edges correlate with hierarchy boundaries and isolated dominant
    /// contributors can alternate densely under translucent overlap. The
    /// acceptance checks therefore retain only coherent two-pixel ownership
    /// runs and compare the interface endpoints with their immediate same-node
    /// interiors. A second difference-in-differences check compares the
    /// residual jump across the interface with the immediately local residual
    /// slopes inside both nodes. Reference-only alpha/gradient matching decides
    /// whether a local pair is eligible, so candidate error cannot select its
    /// own controls.
    #[test]
    #[ignore = "requires canonical Garden ABI-16 package and PLY via BGS_GARDEN_LOD/BGS_GARDEN_PLY"]
    fn canonical_garden_abi16_node_boundary_oracle() {
        let manifest_path = required_fixture_path(GARDEN_LOD_ENV);
        let source_path = required_fixture_path(GARDEN_PLY_ENV);
        let encoded = fs::read(&manifest_path).unwrap_or_else(|error| {
            panic!(
                "failed to read Garden manifest {}: {error}",
                manifest_path.display()
            )
        });
        assert_eq!(
            format!("{:x}", Sha256::digest(&encoded)),
            GARDEN_MANIFEST_SHA256,
            "Garden host-Morton manifest SHA-256 drifted"
        );
        let manifest = decode_manifest(&encoded, LodCodecLimits::default())
            .expect("canonical Garden ABI-16 manifest decodes and validates");
        assert_authenticated_garden_abi16_manifest(&manifest);
        authenticate_garden_package_shards(&manifest_path);

        let views = garden_boundary_views(&manifest);
        assert!(
            views[1].active_gaussians < views[0].active_gaussians,
            "farther Garden q=.65 view must select fewer records: auto={}, far={}",
            views[0].active_gaussians,
            views[1].active_gaussians
        );
        assert!(
            views[2].active_gaussians < views[0].active_gaussians,
            "Garden q=.35 overview must be coarser than auto-frame q=.65: q35={}, q65={}",
            views[2].active_gaussians,
            views[0].active_gaussians
        );
        assert!(
            views
                .iter()
                .all(|view| view.active_gaussians < GARDEN_SOURCE_GAUSSIANS),
            "Garden q=.65 views must exercise an actual ABI-16 reduction"
        );
        let selected_pages = garden_selected_page_ids(&manifest, &views);

        // Authenticate the fixed PLY byte stream without retaining a second
        // 1.45 GB cloud. The package's hash-checked source-leaf pages are the
        // authoritative canonical order: its GPU preprocessor returned the
        // Morton keys used by the manifest fingerprint, while recomputing keys
        // with host division can differ at quantization boundaries.
        authenticate_garden_source(&source_path);
        let source = load_authenticated_garden_leaf_source(&manifest_path, &manifest);
        let mut references = Vec::with_capacity(views.len());
        for view in &views {
            let owners = garden_selected_source_owners(&manifest, &view.frontier, source.len());
            references.push(
                render_production_flat_linear_gaussians_with_owners(
                    &source,
                    &owners,
                    view.camera,
                    GARDEN_BOUNDARY_WIDTH,
                    GARDEN_BOUNDARY_HEIGHT,
                    GaussianColorSpace::SrgbRec709Display,
                )
                .unwrap_or_else(|error| {
                    panic!("{} flat Garden oracle render failed: {error}", view.label)
                }),
            );
        }
        drop(source);

        // Read only the pages needed by the two selected cuts. Packed shard
        // byte ranges are seek-read directly; whole shard payloads are never
        // materialized.
        let pages = decode_selected_garden_pages(&manifest_path, &manifest, &selected_pages);
        let lod = PlanarGaussian3dLod { manifest, pages };

        let mut accepted_class_metrics: [Vec<GardenMatchedBoundaryMetrics>; 4] =
            std::array::from_fn(|_| Vec::new());
        let mut acceptance_failures = Vec::new();
        for (view, reference) in views.iter().zip(&references) {
            let gaussians = gather_frontier_gaussians_with_nodes(&lod, &view.frontier)
                .unwrap_or_else(|error| {
                    panic!("{} ABI-16 frontier does not resolve: {error}", view.label)
                });
            assert_eq!(
                gaussians.len() as u64,
                view.active_gaussians,
                "{} gathered record count differs from selector status",
                view.label
            );
            let candidate = render_production_lod_linear_gaussians(
                &gaussians,
                view.camera,
                GARDEN_BOUNDARY_WIDTH,
                GARDEN_BOUNDARY_HEIGHT,
                GaussianColorSpace::SrgbRec709Display,
            )
            .unwrap_or_else(|error| panic!("{} ABI-16 render failed: {error}", view.label));
            let labels = reference
                .dominant_nodes
                .iter()
                .map(|node| node.map(|node| node.0))
                .collect::<Vec<_>>();
            let raw = compare_node_boundary_bands(
                &reference.rgba,
                &candidate,
                &labels,
                GARDEN_BOUNDARY_WIDTH,
                GARDEN_BOUNDARY_HEIGHT,
                GARDEN_BOUNDARY_BAND_RADIUS,
            )
            .map_err(|error| {
                eprintln!(
                    "Garden ABI16 node-boundary oracle {} raw diagnostic unavailable: {error}",
                    view.label
                );
            })
            .ok();
            let interfaces = garden_boundary_interfaces(
                &lod.manifest,
                &reference.rgba,
                &labels,
                GARDEN_BOUNDARY_WIDTH,
                GARDEN_BOUNDARY_HEIGHT,
            );
            let matched =
                garden_paired_boundary_metrics(&reference.rgba, &candidate, &interfaces.all);
            let overall = compare_linear_rgba(&reference.rgba, &candidate, 1.0 / 255.0)
                .expect("Garden oracle images are finite and equally sized");
            report_garden_boundary_metrics(view, overall, raw, matched);
            if view.acceptance {
                if let Some(matched) = matched {
                    acceptance_failures
                        .extend(garden_boundary_metric_failures(view.label, matched));
                } else {
                    acceptance_failures.push(format!(
                        "{} exposes no coherent, locally matched logical-node interface",
                        view.label
                    ));
                }
            }

            for (class_index, (class, class_interfaces)) in [
                ("same-depth", &interfaces.same_depth),
                ("mixed-depth", &interfaces.mixed_depth),
                ("same-parent", &interfaces.same_parent),
                ("cross-parent", &interfaces.cross_parent),
            ]
            .into_iter()
            .enumerate()
            {
                let metrics =
                    garden_paired_boundary_metrics(&reference.rgba, &candidate, class_interfaces);
                if view.acceptance {
                    accepted_class_metrics[class_index].extend(metrics);
                }
                report_garden_boundary_class(view.label, class, metrics);
                // A per-view subclass is an additional behavior sample only
                // when both estimators have enough observations. Sparse
                // subclasses retain their evidence for the mandatory
                // accepted-view class aggregate below; entering on endpoint
                // coverage alone would make 62 jump interfaces fail a
                // 64-interface gate before aggregation can supply power.
                if view.acceptance
                    && metrics.is_some_and(|metrics| {
                        metrics.boundary.pixels >= GARDEN_MIN_MATCHED_PIXELS
                            && metrics.jump_interfaces.saturating_mul(2)
                                >= GARDEN_MIN_MATCHED_PIXELS
                    })
                {
                    let metrics = metrics.unwrap();
                    acceptance_failures.extend(garden_boundary_metric_failures(
                        &format!("{} {class}", view.label),
                        metrics,
                    ));
                }
            }
        }
        for (class_index, class) in ["same-depth", "mixed-depth", "same-parent", "cross-parent"]
            .into_iter()
            .enumerate()
        {
            let aggregate =
                combine_garden_matched_boundary_metrics(&accepted_class_metrics[class_index]);
            report_garden_boundary_class("q=.65 accepted-view aggregate", class, aggregate);
            if let Some(aggregate) = aggregate {
                acceptance_failures.extend(garden_boundary_metric_failures(
                    &format!("q=.65 Garden accepted-view aggregate {class}"),
                    aggregate,
                ));
            } else {
                acceptance_failures.push(format!(
                    "q=.65 Garden acceptance views expose no coherent, locally matched {class} interface"
                ));
            }
        }
        assert!(
            acceptance_failures.is_empty(),
            "Garden ABI16 boundary acceptance failed:\n{}",
            acceptance_failures.join("\n")
        );
    }

    #[derive(Clone, Debug)]
    struct GardenBoundaryView {
        label: &'static str,
        camera: LodTestCamera,
        frontier: Vec<LodNodeId>,
        active_gaussians: u64,
        acceptance: bool,
    }

    fn required_fixture_path(variable: &str) -> PathBuf {
        PathBuf::from(
            env::var_os(variable)
                .unwrap_or_else(|| panic!("set {variable} to the canonical local fixture")),
        )
    }

    fn assert_authenticated_garden_abi16_manifest(manifest: &GaussianLodManifest) {
        assert_eq!(
            manifest.header.source_gaussian_count, GARDEN_SOURCE_GAUSSIANS,
            "Garden package source count drifted"
        );
        assert_eq!(manifest.header.node_count, GARDEN_NODE_PAGE_COUNT);
        assert_eq!(manifest.header.page_count, GARDEN_NODE_PAGE_COUNT);
        assert_eq!(manifest.nodes.len() as u32, GARDEN_NODE_PAGE_COUNT);
        assert_eq!(manifest.pages.len() as u32, GARDEN_NODE_PAGE_COUNT);
        assert_eq!(
            manifest.header.stored_gaussian_count, GARDEN_STORED_GAUSSIANS,
            "Garden package stored-record count drifted"
        );
        assert_eq!(
            manifest.build.builder_abi_version, GARDEN_ABI16,
            "Garden boundary oracle requires the ABI-16 spatial package"
        );
        assert_eq!(manifest.build.reducer, LodReducerKind::MomentMerge);
        assert_eq!(
            manifest.build.reducer_version, GARDEN_MOMENT_MERGE_V4,
            "Garden boundary oracle requires MomentMerge v4"
        );
        let morph_map = manifest
            .morph_map
            .as_ref()
            .expect("ABI-16 Garden package carries its monotone morph map");
        assert_eq!(morph_map.schema_version, 1);
        assert_eq!(
            morph_map.node_runs.len(),
            manifest.nodes.len(),
            "Garden morph map must cover every hierarchy node"
        );
        assert_ne!(
            manifest.build.source_fingerprint, 0,
            "Garden package omits its canonical-source fingerprint"
        );
        let bounds = manifest
            .scene_bounds
            .expect("canonical Garden manifest has scene bounds");
        assert_eq!(bounds.min, GARDEN_MIN, "Garden scene minimum drifted");
        assert_eq!(bounds.max, GARDEN_MAX, "Garden scene maximum drifted");
        let center = Vec3::from_array(bounds.center());
        assert!(
            center.distance(GARDEN_CENTER) <= 1.0e-4,
            "Garden center drifted: expected={GARDEN_CENTER:?}, actual={center:?}"
        );
        assert!(
            (bounds.radius() - GARDEN_RADIUS).abs() <= 1.0e-3,
            "Garden radius drifted: expected={GARDEN_RADIUS}, actual={}",
            bounds.radius()
        );
        assert!(
            (GARDEN_AUTO_FRAME_DISTANCE / bounds.radius() - 2.177_31).abs() <= 1.0e-5,
            "Garden viewer auto-frame ratio drifted"
        );
    }

    fn garden_boundary_views(manifest: &GaussianLodManifest) -> [GardenBoundaryView; 3] {
        let hierarchy =
            ManifestLodHierarchy::new(manifest).expect("Garden ABI-16 hierarchy compiles");
        let center = Vec3::from_array(
            manifest
                .scene_bounds
                .expect("canonical Garden manifest has scene bounds")
                .center(),
        );
        let radius = manifest.scene_bounds.unwrap().radius();
        let view_direction = Vec3::new(0.0, 1.5, 5.0).normalize();
        [
            ("viewer-auto-q65", GARDEN_AUTO_FRAME_DISTANCE, 0.65, true),
            ("far-4r-q65", 4.0 * radius, 0.65, true),
            (
                "viewer-auto-q35-diagnostic",
                GARDEN_AUTO_FRAME_DISTANCE,
                0.35,
                false,
            ),
        ]
        .map(|(label, distance, quality, acceptance)| {
            let mut settings = GaussianLodSettings {
                quality,
                hysteresis: 0.0,
                ..default()
            };
            settings.budgets.max_active_gaussians = 8_000_000;
            settings.budgets.max_traversal_nodes_per_view = manifest.header.node_count.max(1);
            let camera_position = center + view_direction * distance;
            let clip_from_world = Mat4::perspective_infinite_reverse_rh(
                GARDEN_VIEWER_FOV,
                GARDEN_VIEWPORT_ASPECT,
                GARDEN_VIEWER_NEAR,
            ) * Mat4::look_at_rh(camera_position, center, Vec3::Y);
            let view = LodView::perspective(
                camera_position,
                GARDEN_VIEWPORT_HEIGHT_PX,
                GARDEN_VIEWER_FOV,
                GARDEN_VIEWER_NEAR,
            )
            .with_clip_from_world(clip_from_world);
            let selected = select_frontier_with_visibility(
                &hierarchy,
                &AllResident,
                view,
                &settings,
                |_, metrics| view.node_is_visible(metrics, 0.0),
            )
            .unwrap_or_else(|error| panic!("{label} Garden selection failed: {error:?}"));
            assert!(
                selected.requested_nodes.is_empty(),
                "AllResident Garden selection requested pages"
            );
            GardenBoundaryView {
                label,
                camera: LodTestCamera {
                    position: camera_position,
                    target: center,
                    up: Vec3::Y,
                    projection: LodProjection::Perspective {
                        vertical_fov_radians: GARDEN_VIEWER_FOV,
                    },
                    near: GARDEN_VIEWER_NEAR,
                    far: distance + radius * 4.0,
                    viewport: [GARDEN_BOUNDARY_WIDTH, GARDEN_BOUNDARY_HEIGHT],
                },
                frontier: selected.nodes,
                active_gaussians: selected.status.active_gaussians,
                acceptance,
            }
        })
    }

    fn garden_selected_page_ids(
        manifest: &GaussianLodManifest,
        views: &[GardenBoundaryView],
    ) -> BTreeSet<LodPageId> {
        let nodes = manifest
            .nodes
            .iter()
            .map(|node| (node.id, node))
            .collect::<BTreeMap<_, _>>();
        views
            .iter()
            .flat_map(|view| view.frontier.iter())
            .map(|node| {
                nodes
                    .get(node)
                    .unwrap_or_else(|| panic!("selected Garden node {node:?} is absent"))
                    .representation
                    .page
            })
            .collect()
    }

    fn garden_selected_source_owners(
        manifest: &GaussianLodManifest,
        frontier: &[LodNodeId],
        source_len: usize,
    ) -> Vec<Option<LodNodeId>> {
        let nodes = manifest
            .nodes
            .iter()
            .map(|node| (node.id, node))
            .collect::<BTreeMap<_, _>>();
        let mut selected_ranges = Vec::with_capacity(frontier.len());
        for &node_id in frontier {
            let node = nodes
                .get(&node_id)
                .unwrap_or_else(|| panic!("selected Garden node {node_id:?} is absent"));
            let start = usize::try_from(node.source.start)
                .expect("Garden source range starts fit host usize");
            let end = usize::try_from(node.source.end().expect("Garden source range ends"))
                .expect("Garden source range ends fit host usize");
            assert!(
                start < end && end <= source_len,
                "selected Garden node {node_id:?} owns an empty or invalid source range"
            );
            selected_ranges.push((start, end, node_id));
        }
        selected_ranges.sort_unstable_by_key(|&(start, end, node_id)| (start, end, node_id));
        for pair in selected_ranges.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "selected Garden antichain source ranges overlap: {:?} and {:?}",
                pair[0].2,
                pair[1].2
            );
        }

        let mut owners = vec![None; source_len];
        let mut owned = 0_usize;
        for (start, end, node_id) in selected_ranges {
            for (source_index, owner) in owners[start..end].iter_mut().enumerate() {
                assert!(
                    owner.replace(node_id).is_none(),
                    "selected Garden frontier repeats canonical source position {}",
                    start + source_index
                );
            }
            owned += end - start;
        }
        assert!(
            owned > 0,
            "Garden frontier owns no canonical source records"
        );
        owners
    }

    struct GardenSha256Reader<R> {
        inner: R,
        digest: Sha256,
    }

    impl<R> GardenSha256Reader<R> {
        fn new(inner: R) -> Self {
            Self {
                inner,
                digest: Sha256::new(),
            }
        }

        fn finish(self) -> String {
            format!("{:x}", self.digest.finalize())
        }
    }

    impl<R: Read> Read for GardenSha256Reader<R> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let read = self.inner.read(buffer)?;
            self.digest.update(&buffer[..read]);
            Ok(read)
        }
    }

    fn authenticate_garden_package_shards(manifest_path: &Path) {
        let package_root = manifest_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        for (relative, expected_len, expected_sha256) in GARDEN_SHARDS {
            let path = package_root.join(relative);
            let file = fs::File::open(&path).unwrap_or_else(|error| {
                panic!(
                    "failed to open canonical Garden shard {}: {error}",
                    path.display()
                )
            });
            assert_eq!(
                file.metadata()
                    .expect("canonical Garden shard metadata is readable")
                    .len(),
                expected_len,
                "canonical Garden shard length drifted: {}",
                path.display()
            );
            let mut reader = GardenSha256Reader::new(BufReader::with_capacity(1024 * 1024, file));
            io::copy(&mut reader, &mut io::sink())
                .expect("canonical Garden shard authentication reads every byte");
            assert_eq!(
                reader.finish(),
                expected_sha256,
                "canonical Garden shard SHA-256 drifted: {}",
                path.display()
            );
        }
    }

    fn authenticate_garden_source(path: &Path) {
        let file = fs::File::open(path).unwrap_or_else(|error| {
            panic!(
                "failed to open canonical Garden PLY {}: {error}",
                path.display()
            )
        });
        assert_eq!(
            file.metadata()
                .expect("canonical Garden PLY metadata is readable")
                .len(),
            GARDEN_SOURCE_BYTE_LEN,
            "canonical Garden PLY byte length drifted"
        );
        let reader = GardenSha256Reader::new(file);
        let mut reader = BufReader::with_capacity(1024 * 1024, reader);
        let mut gaussian_count = 0_u64;
        stream_ply_3d(&mut reader, 32 * 1024, |batch| {
            gaussian_count = gaussian_count
                .checked_add(batch.len() as u64)
                .expect("canonical Garden PLY record count fits u64");
            Ok(())
        })
        .unwrap_or_else(|error| {
            panic!(
                "failed to parse canonical Garden PLY {}: {error}",
                path.display()
            )
        });
        io::copy(&mut reader, &mut io::sink())
            .expect("canonical Garden authentication consumes trailing bytes");
        let actual_sha256 = reader.into_inner().finish();
        assert_eq!(
            actual_sha256, GARDEN_SOURCE_SHA256,
            "canonical Garden PLY SHA-256 drifted"
        );
        assert_eq!(
            gaussian_count, GARDEN_SOURCE_GAUSSIANS,
            "canonical Garden PLY record count drifted"
        );
    }

    fn load_authenticated_garden_leaf_source(
        manifest_path: &Path,
        manifest: &GaussianLodManifest,
    ) -> Vec<Gaussian3d> {
        let descriptors = manifest
            .pages
            .iter()
            .map(|descriptor| (descriptor.id, descriptor))
            .collect::<BTreeMap<_, _>>();
        let mut leaves = manifest
            .nodes
            .iter()
            .filter(|node| node.is_leaf())
            .collect::<Vec<_>>();
        leaves.sort_unstable_by_key(|node| (node.source.start, node.id));
        assert!(!leaves.is_empty(), "Garden hierarchy has no source leaves");

        let mut leaf_starts = BTreeMap::new();
        let mut leaf_ends = BTreeMap::new();
        let mut cursor = 0_u64;
        let mut previous_morton_max = None;
        for leaf in &leaves {
            assert_eq!(
                leaf.source.start, cursor,
                "Garden source leaves are not a complete disjoint canonical partition at {:?}",
                leaf.id
            );
            cursor = leaf
                .source
                .end()
                .expect("Garden leaf source range ends without overflow");
            assert_eq!(
                u64::from(leaf.representation.count),
                leaf.source.count,
                "Garden leaf {:?} is not an exact-cardinality source page",
                leaf.id
            );
            assert!(
                leaf.morton.min <= leaf.morton.max,
                "Garden leaf {:?} has an inverted Morton range",
                leaf.id
            );
            if let Some(previous) = previous_morton_max {
                assert!(
                    previous <= leaf.morton.min,
                    "Garden leaf Morton ranges are not canonical and monotonic at {:?}",
                    leaf.id
                );
            }
            previous_morton_max = Some(leaf.morton.max);
            assert!(
                leaf_starts.insert(leaf.source.start, *leaf).is_none(),
                "Garden source leaves repeat a canonical start"
            );
            assert!(
                leaf_ends.insert(cursor, *leaf).is_none(),
                "Garden source leaves repeat a canonical end"
            );
        }
        assert_eq!(
            cursor, manifest.header.source_gaussian_count,
            "Garden source leaves do not cover the full canonical source"
        );

        // Every internal range must be exactly the union of consecutive leaf
        // ranges, including the Morton endpoints authenticated by the
        // manifest. This proves that assigning a selected range by canonical
        // position cannot invent or overlap an ownership domain.
        for node in &manifest.nodes {
            let first = leaf_starts.get(&node.source.start).unwrap_or_else(|| {
                panic!(
                    "Garden node {:?} source start is not a leaf boundary",
                    node.id
                )
            });
            let end = node
                .source
                .end()
                .expect("Garden node source range ends without overflow");
            let last = leaf_ends.get(&end).unwrap_or_else(|| {
                panic!(
                    "Garden node {:?} source end is not a leaf boundary",
                    node.id
                )
            });
            assert_eq!(
                node.morton.min, first.morton.min,
                "Garden node {:?} Morton minimum disagrees with its first source leaf",
                node.id
            );
            assert_eq!(
                node.morton.max, last.morton.max,
                "Garden node {:?} Morton maximum disagrees with its last source leaf",
                node.id
            );
        }

        let source_len = usize::try_from(manifest.header.source_gaussian_count)
            .expect("Garden source count fits host usize");
        let mut source = Vec::new();
        source
            .try_reserve_exact(source_len)
            .expect("host can reserve the authenticated Garden leaf source");
        let mut reader = GardenPackagePageReader::new(manifest_path);
        let mut seen_pages = BTreeSet::new();
        for leaf in leaves {
            assert!(
                seen_pages.insert(leaf.representation.page),
                "Garden source leaf page {:?} is referenced more than once",
                leaf.representation.page
            );
            let descriptor = descriptors
                .get(&leaf.representation.page)
                .unwrap_or_else(|| panic!("Garden leaf {:?} has no page descriptor", leaf.id));
            assert_eq!(
                descriptor.kind,
                LodPageKind::SourceLeaves,
                "Garden leaf {:?} does not reference an exact source-leaf page",
                leaf.id
            );
            let page = reader.decode(descriptor);
            let start = usize::try_from(leaf.representation.offset)
                .expect("Garden leaf page offset fits host usize");
            let end = usize::try_from(
                leaf.representation
                    .end()
                    .expect("Garden leaf page range ends without overflow"),
            )
            .expect("Garden leaf page end fits host usize");
            let records = page
                .gaussians
                .get(start..end)
                .unwrap_or_else(|| panic!("Garden leaf {:?} has an invalid page range", leaf.id));
            assert_eq!(
                records.len() as u64,
                leaf.source.count,
                "Garden leaf {:?} page range is not its exact source range",
                leaf.id
            );
            source.extend_from_slice(records);
        }
        let source_leaf_pages = manifest
            .pages
            .iter()
            .filter(|descriptor| descriptor.kind == LodPageKind::SourceLeaves)
            .map(|descriptor| descriptor.id)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            seen_pages, source_leaf_pages,
            "Garden source-leaf descriptors and leaf nodes are not one-to-one"
        );
        assert_eq!(
            source.len(),
            source_len,
            "decoded Garden source leaves do not reproduce the manifest source count"
        );
        source
    }

    struct GardenPackagePageReader {
        package_root: PathBuf,
        files: BTreeMap<PathBuf, fs::File>,
    }

    impl GardenPackagePageReader {
        fn new(manifest_path: &Path) -> Self {
            let package_root = manifest_path
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."))
                .canonicalize()
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to canonicalize Garden package root for {}: {error}",
                        manifest_path.display()
                    )
                });
            Self {
                package_root,
                files: BTreeMap::new(),
            }
        }

        fn decode(&mut self, descriptor: &LodPageDescriptor) -> PlanarGaussian3dPage {
            let storage = descriptor
                .storage
                .as_ref()
                .unwrap_or_else(|| panic!("Garden page {:?} has no native storage", descriptor.id));
            let relative = Path::new(&storage.uri);
            assert!(
                !relative.is_absolute()
                    && relative
                        .components()
                        .all(|component| matches!(component, std::path::Component::Normal(_))),
                "Garden page URI is not package-relative: {}",
                storage.uri
            );
            let page_path = self
                .package_root
                .join(relative)
                .canonicalize()
                .unwrap_or_else(|error| {
                    panic!("failed to resolve Garden page {}: {error}", storage.uri)
                });
            assert!(
                page_path.starts_with(&self.package_root),
                "Garden page URI escapes the package root: {}",
                storage.uri
            );
            let encoded_len = usize::try_from(storage.encoded_len)
                .expect("Garden encoded page length fits host usize");
            let file = self.files.entry(page_path.clone()).or_insert_with(|| {
                fs::File::open(&page_path).unwrap_or_else(|error| {
                    panic!(
                        "failed to open Garden page {}: {error}",
                        page_path.display()
                    )
                })
            });
            let (offset, length) = storage.byte_range.unwrap_or((0, storage.encoded_len));
            assert_eq!(length, storage.encoded_len);
            if storage.byte_range.is_none() {
                assert_eq!(
                    file.metadata()
                        .expect("Garden page metadata is readable")
                        .len(),
                    storage.encoded_len,
                    "standalone Garden page length drifted"
                );
            }
            file.seek(SeekFrom::Start(offset)).unwrap_or_else(|error| {
                panic!(
                    "failed to seek Garden page {}: {error}",
                    page_path.display()
                )
            });
            let mut encoded = vec![0_u8; encoded_len];
            file.read_exact(&mut encoded).unwrap_or_else(|error| {
                panic!(
                    "failed to read Garden page {}: {error}",
                    page_path.display()
                )
            });
            decode_page_with_descriptor(&encoded, descriptor, LodCodecLimits::default())
                .unwrap_or_else(|error| {
                    panic!(
                        "Garden page {:?} failed authentication: {error}",
                        descriptor.id
                    )
                })
        }
    }

    fn decode_selected_garden_pages(
        manifest_path: &Path,
        manifest: &GaussianLodManifest,
        selected: &BTreeSet<LodPageId>,
    ) -> Vec<bevy_gaussian_splatting::PlanarGaussian3dPage> {
        let mut reader = GardenPackagePageReader::new(manifest_path);
        let mut pages = Vec::with_capacity(selected.len());
        for descriptor in manifest
            .pages
            .iter()
            .filter(|descriptor| selected.contains(&descriptor.id))
        {
            pages.push(reader.decode(descriptor));
        }
        assert_eq!(
            pages.len(),
            selected.len(),
            "Garden manifest omits a selected page descriptor"
        );
        pages
    }

    #[derive(Clone, Copy, Debug)]
    struct GardenMatchedBoundaryMetrics {
        interfaces: usize,
        jump_interfaces: usize,
        boundary: SpatialResidualMetrics,
        control: SpatialResidualMetrics,
        boundary_jump: GardenResidualJumpMetrics,
        control_jump: GardenResidualJumpMetrics,
        rgb_rmse_enrichment: f64,
        alpha_abs_enrichment: f64,
        rgb_jump_enrichment: f64,
        alpha_jump_enrichment: f64,
        reference_alpha_gap: f64,
        reference_gradient_gap: f64,
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct GardenResidualJumpMetrics {
        edges: usize,
        rgb_rmse: f64,
        luminance_abs_mean: f64,
        alpha_abs_mean: f64,
        max_abs_delta: f32,
    }

    fn combine_garden_matched_boundary_metrics(
        metrics: &[GardenMatchedBoundaryMetrics],
    ) -> Option<GardenMatchedBoundaryMetrics> {
        if metrics.is_empty() {
            return None;
        }
        let boundary = combine_garden_spatial_residuals(
            &metrics
                .iter()
                .map(|metrics| metrics.boundary)
                .collect::<Vec<_>>(),
        );
        let control = combine_garden_spatial_residuals(
            &metrics
                .iter()
                .map(|metrics| metrics.control)
                .collect::<Vec<_>>(),
        );
        assert_eq!(boundary.pixels, control.pixels);
        let boundary_jump = combine_garden_residual_jumps(
            &metrics
                .iter()
                .map(|metrics| metrics.boundary_jump)
                .collect::<Vec<_>>(),
        );
        let control_jump = combine_garden_residual_jumps(
            &metrics
                .iter()
                .map(|metrics| metrics.control_jump)
                .collect::<Vec<_>>(),
        );
        let interfaces = metrics
            .iter()
            .map(|metrics| metrics.interfaces)
            .sum::<usize>();
        let jump_interfaces = metrics
            .iter()
            .map(|metrics| metrics.jump_interfaces)
            .sum::<usize>();
        let pixels = boundary.pixels as f64;
        Some(GardenMatchedBoundaryMetrics {
            interfaces,
            jump_interfaces,
            boundary,
            control,
            boundary_jump,
            control_jump,
            rgb_rmse_enrichment: garden_regularized_enrichment(boundary.rgb_rmse, control.rgb_rmse),
            alpha_abs_enrichment: garden_regularized_enrichment(
                boundary.alpha_abs_mean,
                control.alpha_abs_mean,
            ),
            rgb_jump_enrichment: garden_regularized_enrichment(
                boundary_jump.rgb_rmse,
                control_jump.rgb_rmse,
            ),
            alpha_jump_enrichment: garden_regularized_enrichment(
                boundary_jump.alpha_abs_mean,
                control_jump.alpha_abs_mean,
            ),
            reference_alpha_gap: metrics
                .iter()
                .map(|metrics| metrics.reference_alpha_gap * metrics.boundary.pixels as f64)
                .sum::<f64>()
                / pixels,
            reference_gradient_gap: metrics
                .iter()
                .map(|metrics| metrics.reference_gradient_gap * metrics.boundary.pixels as f64)
                .sum::<f64>()
                / pixels,
        })
    }

    fn combine_garden_residual_jumps(
        metrics: &[GardenResidualJumpMetrics],
    ) -> GardenResidualJumpMetrics {
        let edges = metrics.iter().map(|metrics| metrics.edges).sum::<usize>();
        if edges == 0 {
            return GardenResidualJumpMetrics::default();
        }
        let weighted = |value: fn(GardenResidualJumpMetrics) -> f64| {
            metrics
                .iter()
                .copied()
                .map(|metrics| value(metrics) * metrics.edges as f64)
                .sum::<f64>()
                / edges as f64
        };
        GardenResidualJumpMetrics {
            edges,
            rgb_rmse: weighted(|metrics| metrics.rgb_rmse * metrics.rgb_rmse).sqrt(),
            luminance_abs_mean: weighted(|metrics| metrics.luminance_abs_mean),
            alpha_abs_mean: weighted(|metrics| metrics.alpha_abs_mean),
            max_abs_delta: metrics
                .iter()
                .map(|metrics| metrics.max_abs_delta)
                .fold(0.0, f32::max),
        }
    }

    fn combine_garden_spatial_residuals(
        metrics: &[SpatialResidualMetrics],
    ) -> SpatialResidualMetrics {
        let pixels = metrics.iter().map(|metrics| metrics.pixels).sum::<usize>();
        assert!(pixels > 0);
        let weighted = |value: fn(SpatialResidualMetrics) -> f64| {
            metrics
                .iter()
                .copied()
                .map(|metrics| value(metrics) * metrics.pixels as f64)
                .sum::<f64>()
                / pixels as f64
        };
        SpatialResidualMetrics {
            pixels,
            rgb_rmse: weighted(|metrics| metrics.rgb_rmse * metrics.rgb_rmse).sqrt(),
            signed_luminance_mean: weighted(|metrics| metrics.signed_luminance_mean),
            signed_alpha_mean: weighted(|metrics| metrics.signed_alpha_mean),
            alpha_abs_mean: weighted(|metrics| metrics.alpha_abs_mean),
            max_abs_residual: metrics
                .iter()
                .map(|metrics| metrics.max_abs_residual)
                .fold(0.0, f32::max),
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct GardenBoundaryInterface {
        /// The two adjacent pixels on opposite sides of the logical interface.
        boundary: [usize; 2],
        /// Immediate same-node controls behind each interface endpoint.
        control: [usize; 2],
        reference_alpha_gap: f64,
        reference_gradient_gap: f64,
    }

    #[derive(Default)]
    struct GardenBoundaryInterfaces {
        all: Vec<GardenBoundaryInterface>,
        same_depth: Vec<GardenBoundaryInterface>,
        mixed_depth: Vec<GardenBoundaryInterface>,
        same_parent: Vec<GardenBoundaryInterface>,
        cross_parent: Vec<GardenBoundaryInterface>,
    }

    fn garden_boundary_interfaces(
        manifest: &GaussianLodManifest,
        reference: &[[f32; 4]],
        labels: &[Option<u64>],
        width: u32,
        height: u32,
    ) -> GardenBoundaryInterfaces {
        let topology = manifest
            .nodes
            .iter()
            .map(|node| (node.id.0, (node.depth, node.parent.map(|parent| parent.0))))
            .collect::<BTreeMap<_, _>>();
        garden_boundary_interfaces_with_topology(&topology, reference, labels, width, height)
    }

    fn garden_boundary_interfaces_with_topology(
        topology: &BTreeMap<u64, (u16, Option<u64>)>,
        reference: &[[f32; 4]],
        labels: &[Option<u64>],
        width: u32,
        height: u32,
    ) -> GardenBoundaryInterfaces {
        let width = width as usize;
        let height = height as usize;
        assert!(width > 0 && height > 0);
        assert_eq!(reference.len(), width * height);
        assert_eq!(labels.len(), reference.len());
        let mut interfaces = GardenBoundaryInterfaces::default();
        for y in 0..height {
            for x in 0..width {
                if x >= 1 && x + 2 < width {
                    let left = y * width + x;
                    let right = left + 1;
                    maybe_push_garden_boundary_interface(
                        labels,
                        reference,
                        topology,
                        [left, right],
                        [left - 1, right + 1],
                        &mut interfaces,
                    );
                }
                if y >= 1 && y + 2 < height {
                    let top = y * width + x;
                    let bottom = top + width;
                    maybe_push_garden_boundary_interface(
                        labels,
                        reference,
                        topology,
                        [top, bottom],
                        [top - width, bottom + width],
                        &mut interfaces,
                    );
                }
            }
        }
        interfaces
    }

    #[allow(clippy::too_many_arguments)]
    fn maybe_push_garden_boundary_interface(
        labels: &[Option<u64>],
        reference: &[[f32; 4]],
        topology: &BTreeMap<u64, (u16, Option<u64>)>,
        boundary: [usize; 2],
        control: [usize; 2],
        interfaces: &mut GardenBoundaryInterfaces,
    ) {
        let (Some(left_node), Some(right_node)) = (labels[boundary[0]], labels[boundary[1]]) else {
            return;
        };
        if left_node == right_node
            || labels[control[0]] != Some(left_node)
            || labels[control[1]] != Some(right_node)
        {
            return;
        }
        let reference_alpha_gaps = std::array::from_fn::<_, 2, _>(|side| {
            f64::from((reference[boundary[side]][3] - reference[control[side]][3]).abs())
        });
        let reference_gradient_gaps = std::array::from_fn::<_, 2, _>(|side| {
            (garden_reference_luminance(reference[boundary[side]])
                - garden_reference_luminance(reference[control[side]]))
            .abs()
        });
        let reference_alpha_gap = reference_alpha_gaps.iter().sum::<f64>() / 2.0;
        let reference_gradient_gap = reference_gradient_gaps.iter().sum::<f64>() / 2.0;
        // Eligibility depends only on the flat reference. Candidate residuals
        // can therefore neither choose nor reject their own controls.
        if reference_alpha_gaps
            .iter()
            .chain(&reference_gradient_gaps)
            .any(|gap| *gap > GARDEN_MAX_REFERENCE_MATCH_GAP)
        {
            return;
        }
        let &(left_depth, left_parent) = topology
            .get(&left_node)
            .unwrap_or_else(|| panic!("Garden dominant node {left_node} is absent"));
        let &(right_depth, right_parent) = topology
            .get(&right_node)
            .unwrap_or_else(|| panic!("Garden dominant node {right_node} is absent"));
        let interface = GardenBoundaryInterface {
            boundary,
            control,
            reference_alpha_gap,
            reference_gradient_gap,
        };
        interfaces.all.push(interface);
        if left_depth == right_depth {
            interfaces.same_depth.push(interface);
        } else {
            interfaces.mixed_depth.push(interface);
        }
        if left_parent.is_some() && left_parent == right_parent {
            interfaces.same_parent.push(interface);
        } else {
            interfaces.cross_parent.push(interface);
        }
    }

    fn garden_paired_boundary_metrics(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        interfaces: &[GardenBoundaryInterface],
    ) -> Option<GardenMatchedBoundaryMetrics> {
        if interfaces.is_empty() {
            return None;
        }
        assert_eq!(reference.len(), candidate.len());
        let boundary = interfaces
            .iter()
            .flat_map(|interface| interface.boundary)
            .collect::<Vec<_>>();
        let control = interfaces
            .iter()
            .flat_map(|interface| interface.control)
            .collect::<Vec<_>>();
        let boundary_metrics = garden_spatial_residual(reference, candidate, &boundary);
        let control_metrics = garden_spatial_residual(reference, candidate, &control);
        let boundary_jump = garden_boundary_residual_jumps(reference, candidate, interfaces);
        let control_jump = garden_control_residual_jumps(reference, candidate, interfaces);
        Some(GardenMatchedBoundaryMetrics {
            interfaces: interfaces.len(),
            jump_interfaces: boundary_jump.edges,
            boundary: boundary_metrics,
            control: control_metrics,
            boundary_jump,
            control_jump,
            rgb_rmse_enrichment: garden_regularized_enrichment(
                boundary_metrics.rgb_rmse,
                control_metrics.rgb_rmse,
            ),
            alpha_abs_enrichment: garden_regularized_enrichment(
                boundary_metrics.alpha_abs_mean,
                control_metrics.alpha_abs_mean,
            ),
            rgb_jump_enrichment: garden_regularized_enrichment(
                boundary_jump.rgb_rmse,
                control_jump.rgb_rmse,
            ),
            alpha_jump_enrichment: garden_regularized_enrichment(
                boundary_jump.alpha_abs_mean,
                control_jump.alpha_abs_mean,
            ),
            reference_alpha_gap: interfaces
                .iter()
                .map(|interface| interface.reference_alpha_gap)
                .sum::<f64>()
                / interfaces.len() as f64,
            reference_gradient_gap: interfaces
                .iter()
                .map(|interface| interface.reference_gradient_gap)
                .sum::<f64>()
                / interfaces.len() as f64,
        })
    }

    #[inline]
    fn garden_reference_luminance(pixel: [f32; 4]) -> f64 {
        0.2126 * pixel[0] as f64 + 0.7152 * pixel[1] as f64 + 0.0722 * pixel[2] as f64
    }

    fn garden_boundary_residual_jumps(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        interfaces: &[GardenBoundaryInterface],
    ) -> GardenResidualJumpMetrics {
        let pairs = interfaces
            .iter()
            .filter(|interface| garden_jump_eligible(reference, interface))
            .map(|interface| interface.boundary)
            .collect::<Vec<_>>();
        garden_residual_jumps(reference, candidate, &pairs)
    }

    fn garden_control_residual_jumps(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        interfaces: &[GardenBoundaryInterface],
    ) -> GardenResidualJumpMetrics {
        let pairs = interfaces
            .iter()
            .filter(|interface| garden_jump_eligible(reference, interface))
            .flat_map(|interface| {
                [
                    [interface.control[0], interface.boundary[0]],
                    [interface.boundary[1], interface.control[1]],
                ]
            })
            .collect::<Vec<_>>();
        garden_residual_jumps(reference, candidate, &pairs)
    }

    fn garden_jump_eligible(reference: &[[f32; 4]], interface: &GardenBoundaryInterface) -> bool {
        (0..4).all(|channel| {
            (reference[interface.boundary[0]][channel] - reference[interface.boundary[1]][channel])
                .abs()
                <= GARDEN_MAX_REFERENCE_MATCH_GAP as f32
        })
    }

    fn garden_residual_jumps(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        pairs: &[[usize; 2]],
    ) -> GardenResidualJumpMetrics {
        assert_eq!(reference.len(), candidate.len());
        if pairs.is_empty() {
            return GardenResidualJumpMetrics::default();
        }
        let mut rgb_squared_sum = 0.0_f64;
        let mut luminance_abs_sum = 0.0_f64;
        let mut alpha_abs_sum = 0.0_f64;
        let mut max_abs_delta = 0.0_f32;
        for &[left, right] in pairs {
            let left_residual = std::array::from_fn::<_, 4, _>(|channel| {
                f64::from(candidate[left][channel] - reference[left][channel])
            });
            let right_residual = std::array::from_fn::<_, 4, _>(|channel| {
                f64::from(candidate[right][channel] - reference[right][channel])
            });
            let mut rgb_delta = [0.0_f64; 3];
            for channel in 0..3 {
                rgb_delta[channel] = right_residual[channel] - left_residual[channel];
                rgb_squared_sum += rgb_delta[channel] * rgb_delta[channel];
                max_abs_delta = max_abs_delta.max(rgb_delta[channel].abs() as f32);
            }
            luminance_abs_sum +=
                (0.2126 * rgb_delta[0] + 0.7152 * rgb_delta[1] + 0.0722 * rgb_delta[2]).abs();
            let alpha_delta = right_residual[3] - left_residual[3];
            alpha_abs_sum += alpha_delta.abs();
            max_abs_delta = max_abs_delta.max(alpha_delta.abs() as f32);
        }
        GardenResidualJumpMetrics {
            edges: pairs.len(),
            rgb_rmse: (rgb_squared_sum / (pairs.len() * 3) as f64).sqrt(),
            luminance_abs_mean: luminance_abs_sum / pairs.len() as f64,
            alpha_abs_mean: alpha_abs_sum / pairs.len() as f64,
            max_abs_delta,
        }
    }

    fn garden_spatial_residual(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        indices: &[usize],
    ) -> SpatialResidualMetrics {
        assert_eq!(reference.len(), candidate.len());
        assert!(!indices.is_empty());
        let mut rgb_squared_sum = 0.0_f64;
        let mut signed_luminance_sum = 0.0_f64;
        let mut signed_alpha_sum = 0.0_f64;
        let mut alpha_abs_sum = 0.0_f64;
        let mut max_abs_residual = 0.0_f32;
        for &index in indices {
            let expected = reference[index];
            let actual = candidate[index];
            for channel in 0..3 {
                let residual = f64::from(actual[channel] - expected[channel]);
                rgb_squared_sum += residual * residual;
                max_abs_residual = max_abs_residual.max(residual.abs() as f32);
            }
            let expected_luminance = 0.2126 * expected[0] as f64
                + 0.7152 * expected[1] as f64
                + 0.0722 * expected[2] as f64;
            let actual_luminance =
                0.2126 * actual[0] as f64 + 0.7152 * actual[1] as f64 + 0.0722 * actual[2] as f64;
            signed_luminance_sum += actual_luminance - expected_luminance;
            let alpha = f64::from(actual[3] - expected[3]);
            signed_alpha_sum += alpha;
            alpha_abs_sum += alpha.abs();
            max_abs_residual = max_abs_residual.max(alpha.abs() as f32);
        }
        SpatialResidualMetrics {
            pixels: indices.len(),
            rgb_rmse: (rgb_squared_sum / (indices.len() * 3) as f64).sqrt(),
            signed_luminance_mean: signed_luminance_sum / indices.len() as f64,
            signed_alpha_mean: signed_alpha_sum / indices.len() as f64,
            alpha_abs_mean: alpha_abs_sum / indices.len() as f64,
            max_abs_residual,
        }
    }

    fn garden_regularized_enrichment(boundary: f64, control: f64) -> f64 {
        (boundary + GARDEN_ENRICHMENT_FLOOR) / (control + GARDEN_ENRICHMENT_FLOOR)
    }

    fn report_garden_boundary_metrics(
        view: &GardenBoundaryView,
        overall: ImageMetrics,
        raw: Option<BoundaryBandMetrics>,
        matched: Option<GardenMatchedBoundaryMetrics>,
    ) {
        eprintln!(
            "Garden ABI16 node-boundary oracle {} [{}]: active={} nodes={} overall_psnr={:.3} overall_alpha_mae={:.6}",
            view.label,
            if view.acceptance {
                "acceptance"
            } else {
                "diagnostic-only"
            },
            view.active_gaussians,
            view.frontier.len(),
            overall.foreground_psnr_rgb,
            overall.alpha_mae,
        );
        if let Some(raw) = raw {
            eprintln!(
                "Garden ABI16 node-boundary oracle {} raw diagnostic: boundary_px={} interior_px={} boundary_signed_luma={:+.6} interior_signed_luma={:+.6} boundary_signed_alpha={:+.6} interior_signed_alpha={:+.6} rgb_enrichment={:.3} alpha_enrichment={:.3}",
                view.label,
                raw.boundary.pixels,
                raw.interior.pixels,
                raw.boundary.signed_luminance_mean,
                raw.interior.signed_luminance_mean,
                raw.boundary.signed_alpha_mean,
                raw.interior.signed_alpha_mean,
                raw.rgb_rmse_enrichment,
                raw.alpha_abs_enrichment,
            );
        }
        report_garden_boundary_class(view.label, "all", matched);
    }

    fn report_garden_boundary_class(
        view: &str,
        class: &str,
        metrics: Option<GardenMatchedBoundaryMetrics>,
    ) {
        let Some(metrics) = metrics else {
            eprintln!(
                "Garden ABI16 node-boundary oracle {view} {class}: no measurable boundary band"
            );
            return;
        };
        eprintln!(
            "Garden ABI16 node-boundary oracle {view} {class}: paired_interfaces={} jump_eligible_interfaces={} boundary_px={} boundary_signed_luma={:+.6} control_signed_luma={:+.6} boundary_signed_alpha={:+.6} control_signed_alpha={:+.6} rgb_enrichment={:.3} alpha_enrichment={:.3} cross_rgb_jump={:.6} local_rgb_jump={:.6} rgb_jump_enrichment={:.3} cross_luma_jump={:.6} local_luma_jump={:.6} cross_alpha_jump={:.6} local_alpha_jump={:.6} alpha_jump_enrichment={:.3} reference_alpha_gap={:.6} reference_gradient_gap={:.6}",
            metrics.interfaces,
            metrics.jump_interfaces,
            metrics.boundary.pixels,
            metrics.boundary.signed_luminance_mean,
            metrics.control.signed_luminance_mean,
            metrics.boundary.signed_alpha_mean,
            metrics.control.signed_alpha_mean,
            metrics.rgb_rmse_enrichment,
            metrics.alpha_abs_enrichment,
            metrics.boundary_jump.rgb_rmse,
            metrics.control_jump.rgb_rmse,
            metrics.rgb_jump_enrichment,
            metrics.boundary_jump.luminance_abs_mean,
            metrics.control_jump.luminance_abs_mean,
            metrics.boundary_jump.alpha_abs_mean,
            metrics.control_jump.alpha_abs_mean,
            metrics.alpha_jump_enrichment,
            metrics.reference_alpha_gap,
            metrics.reference_gradient_gap,
        );
    }

    fn garden_boundary_metric_failures(
        label: &str,
        metrics: GardenMatchedBoundaryMetrics,
    ) -> Vec<String> {
        let mut failures = Vec::new();
        if metrics.boundary.pixels < GARDEN_MIN_MATCHED_PIXELS {
            failures.push(format!(
                "{label} has too few matched boundary pixels: {}",
                metrics.boundary.pixels
            ));
        }
        let jump_eligible_pixels = metrics.jump_interfaces.saturating_mul(2);
        if jump_eligible_pixels < GARDEN_MIN_MATCHED_PIXELS {
            failures.push(format!(
                "{label} has too few content-smooth jump-eligible boundary pixels: {jump_eligible_pixels}"
            ));
        }
        if metrics.boundary.pixels != metrics.control.pixels {
            failures.push(format!(
                "{label} boundary/control pixel counts differ: boundary={}, control={}",
                metrics.boundary.pixels, metrics.control.pixels
            ));
        }
        if !(metrics.reference_alpha_gap <= GARDEN_MAX_REFERENCE_MATCH_GAP) {
            failures.push(format!(
                "{label} reference-alpha control mismatch is too large: {}",
                metrics.reference_alpha_gap
            ));
        }
        if !(metrics.reference_gradient_gap <= GARDEN_MAX_REFERENCE_MATCH_GAP) {
            failures.push(format!(
                "{label} reference-gradient control mismatch is too large: {}",
                metrics.reference_gradient_gap
            ));
        }
        let luminance_bias_gap =
            metrics.boundary.signed_luminance_mean - metrics.control.signed_luminance_mean;
        let alpha_bias_gap = metrics.boundary.signed_alpha_mean - metrics.control.signed_alpha_mean;
        if !(luminance_bias_gap.abs() <= GARDEN_MAX_MATCHED_SIGNED_BIAS_GAP) {
            failures.push(format!(
                "{label} matched boundary luminance bias gap is {luminance_bias_gap:+.6}"
            ));
        }
        if !(alpha_bias_gap.abs() <= GARDEN_MAX_MATCHED_SIGNED_BIAS_GAP) {
            failures.push(format!(
                "{label} matched boundary alpha bias gap is {alpha_bias_gap:+.6}"
            ));
        }
        if !(metrics.rgb_rmse_enrichment <= GARDEN_MAX_MATCHED_ENRICHMENT) {
            failures.push(format!(
                "{label} matched boundary RGB enrichment is {}",
                metrics.rgb_rmse_enrichment
            ));
        }
        if !(metrics.alpha_abs_enrichment <= GARDEN_MAX_MATCHED_ENRICHMENT) {
            failures.push(format!(
                "{label} matched boundary alpha enrichment is {}",
                metrics.alpha_abs_enrichment
            ));
        }
        if !(metrics.rgb_jump_enrichment <= GARDEN_MAX_MATCHED_ENRICHMENT) {
            failures.push(format!(
                "{label} cross-interface residual RGB jump enrichment is {}",
                metrics.rgb_jump_enrichment
            ));
        }
        if !(metrics.alpha_jump_enrichment <= GARDEN_MAX_MATCHED_ENRICHMENT) {
            failures.push(format!(
                "{label} cross-interface residual alpha jump enrichment is {}",
                metrics.alpha_jump_enrichment
            ));
        }
        failures
    }

    #[test]
    fn garden_local_pairs_ignore_matched_scene_edges_and_content_correlated_error() {
        const WIDTH: u32 = 12;
        const HEIGHT: u32 = 128;
        let topology = BTreeMap::from([(1, (3, Some(10))), (2, (3, Some(10)))]);
        let mut reference = Vec::with_capacity((WIDTH * HEIGHT) as usize);
        let mut candidate = Vec::with_capacity(reference.capacity());
        let mut labels = Vec::with_capacity(reference.capacity());
        for y in 0..HEIGHT {
            let strong_content_edge = y % 2 == 0;
            let smooth_residual = (y as f32 / (HEIGHT - 1) as f32 - 0.5) * 0.004;
            for x in 0..WIDTH {
                let left = x < WIDTH / 2;
                let expected = if strong_content_edge {
                    if left {
                        [0.08, 0.04, 0.02, 0.20]
                    } else {
                        [0.72, 0.56, 0.40, 0.80]
                    }
                } else {
                    [0.30, 0.24, 0.18, 0.50]
                };
                reference.push(expected);
                candidate.push(expected.map(|value| value * 0.98 + smooth_residual));
                labels.push(Some(if left { 1 } else { 2 }));
            }
        }

        let interfaces =
            garden_boundary_interfaces_with_topology(&topology, &reference, &labels, WIDTH, HEIGHT);
        let metrics = garden_paired_boundary_metrics(&reference, &candidate, &interfaces.all)
            .expect("coherent synthetic interface is measurable");
        assert_eq!(metrics.interfaces, HEIGHT as usize);
        assert_eq!(metrics.boundary.pixels, 2 * HEIGHT as usize);
        assert_eq!(metrics.jump_interfaces, HEIGHT as usize / 2);
        assert!(
            garden_boundary_metric_failures("matched scene edge", metrics).is_empty(),
            "a global content-correlated error must not be diagnosed as a logical seam: {metrics:?}"
        );
    }

    #[test]
    fn garden_local_pairs_detect_a_signed_node_density_and_color_step() {
        const WIDTH: u32 = 12;
        const HEIGHT: u32 = 128;
        let topology = BTreeMap::from([(1, (3, Some(10))), (2, (3, Some(10)))]);
        let reference = vec![[0.40; 4]; (WIDTH * HEIGHT) as usize];
        let mut candidate = reference.clone();
        let mut labels = Vec::with_capacity(reference.len());
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let index = (y * WIDTH + x) as usize;
                let left = x < WIDTH / 2;
                candidate[index] = [if left { 0.36 } else { 0.44 }; 4];
                labels.push(Some(if left { 1 } else { 2 }));
            }
        }

        let interfaces =
            garden_boundary_interfaces_with_topology(&topology, &reference, &labels, WIDTH, HEIGHT);
        let metrics = garden_paired_boundary_metrics(&reference, &candidate, &interfaces.all)
            .expect("coherent synthetic interface is measurable");
        assert_eq!(metrics.jump_interfaces, HEIGHT as usize);
        let failures = garden_boundary_metric_failures("injected node step", metrics);
        assert!(
            failures.iter().any(|failure| failure.contains("RGB jump"))
                && failures
                    .iter()
                    .any(|failure| failure.contains("alpha jump")),
            "the seam oracle missed an injected signed color+density step: {metrics:?}; failures={failures:?}"
        );
    }

    #[test]
    fn garden_coherent_interfaces_exclude_one_pixel_dominant_label_noise() {
        const WIDTH: u32 = 12;
        const HEIGHT: u32 = 16;
        let topology = BTreeMap::from([(1, (3, Some(10))), (2, (3, Some(10)))]);
        let reference = vec![[0.40; 4]; (WIDTH * HEIGHT) as usize];
        let labels = (0..WIDTH * HEIGHT)
            .map(|index| Some(if index % WIDTH % 2 == 0 { 1 } else { 2 }))
            .collect::<Vec<_>>();
        let interfaces =
            garden_boundary_interfaces_with_topology(&topology, &reference, &labels, WIDTH, HEIGHT);
        assert!(
            interfaces.all.is_empty(),
            "one-pixel dominant-contributor alternation is not a coherent logical interface"
        );
    }

    /// Full-scene color and covariance audit for the documented Trellis
    /// artifact. It is ignored by default and accepts only a caller-supplied
    /// local path. The test contains no URL and performs no download; CI can
    /// provision the immutable artifact through a non-required cache or
    /// workflow-dispatch step.
    #[test]
    #[ignore = "requires the canonical local Trellis GLB via BGS_TRELLIS_GLB"]
    fn canonical_trellis_high_quality_color_and_covariance_audit() {
        let profile = TrellisAuditProfile::from_environment();
        let path = PathBuf::from(env::var_os(TRELLIS_ENV).unwrap_or_else(|| {
            panic!(
                "set {TRELLIS_ENV} to the canonical local Trellis GLB (sha256 {TRELLIS_SHA256_FOR_PREFLIGHT})"
            )
        }));
        verify_trellis_provenance(&path);
        let loaded = load_local_gaussian_scene(&path);
        assert_eq!(loaded.cloud.position_visibility.len(), TRELLIS_SPLAT_COUNT);

        let lod = build_planar_3d_lod(&loaded.cloud, GaussianLodBuildSettings::default())
            .expect("canonical Trellis hierarchy builds with the production default profile");
        assert_eq!(
            lod.manifest.build.settings,
            GaussianLodBuildSettings::default()
        );
        assert!(
            lod.manifest
                .nodes
                .iter()
                .filter(|node| node.is_leaf())
                .all(|node| node.source.count <= 64),
            "the production progressive profile must retain fine logical leaves"
        );
        assert!(
            lod.manifest
                .pages
                .iter()
                .all(|page| page.gaussian_count <= 1_024),
            "fine logical nodes must remain packed into bounded physical pages"
        );
        assert!(lod.manifest.header.page_count < lod.manifest.header.node_count);
        let hierarchy = ManifestLodHierarchy::new(&lod.manifest)
            .expect("canonical Trellis hierarchy manifest is valid");
        let mut certificates = lod
            .manifest
            .nodes
            .iter()
            .filter(|node| !node.is_leaf())
            .map(|node| node.high_fidelity_certificate)
            .collect::<Vec<_>>();
        certificates.sort_by(f32::total_cmp);
        let quantile = |fraction: f32| {
            certificates[((certificates.len() - 1) as f32 * fraction).round() as usize]
        };
        eprintln!(
            "Trellis internal certificate distribution: count={} min={:.6} p50={:.6} p90={:.6} p95={:.6} p99={:.6} max={:.6}; admitted={:?}",
            certificates.len(),
            quantile(0.0),
            quantile(0.5),
            quantile(0.9),
            quantile(0.95),
            quantile(0.99),
            quantile(1.0),
            [0.0001, 0.001, 0.01, 0.05, 0.1].map(|threshold| certificates
                .iter()
                .filter(|value| **value >= threshold)
                .count())
        );
        let rendered_qualities = profile.rendered_qualities();
        let selection_qualities = trellis_dense_selection_qualities();
        let mut distance_sweeps = Vec::with_capacity(3);
        let distance_cameras = trellis_distance_cameras(&loaded, &hierarchy);
        for audit_camera in distance_cameras {
            let exact = real_scene_quality_sample(
                &lod,
                &hierarchy,
                audit_camera.camera,
                1.0,
                loaded.color_space,
            );
            let rendered = rendered_qualities
                .iter()
                .copied()
                .map(|quality| {
                    if quality_eq(quality, 1.0) {
                        quality_sweep_point(quality, &exact, &exact, audit_camera.camera)
                    } else {
                        let sample = real_scene_quality_sample(
                            &lod,
                            &hierarchy,
                            audit_camera.camera,
                            quality,
                            loaded.color_space,
                        );
                        quality_sweep_point(quality, &sample, &exact, audit_camera.camera)
                    }
                })
                .collect();
            let deployment_selection = selection_qualities
                .iter()
                .copied()
                .map(|quality| SelectionSweepPoint {
                    quality,
                    active_gaussians: real_scene_active_count(
                        &lod,
                        &hierarchy,
                        audit_camera.camera,
                        quality,
                        TRELLIS_DEPLOYMENT_VIEWPORT_HEIGHT_PX,
                    ),
                })
                .collect();
            distance_sweeps.push(TrellisDistanceSweep {
                camera: audit_camera,
                rendered,
                deployment_selection,
            });
        }

        let authored_camera = trellis_authored_camera(&loaded);
        let authored_sweep =
            real_scene_morphology_sweep(&lod, &hierarchy, authored_camera, loaded.color_space);
        let orbit_sweeps = trellis_orbit_cameras(profile, distance_cameras[1].camera)
            .into_iter()
            .map(|audit_camera| TrellisMorphologySweep {
                label: audit_camera.label,
                rendered: real_scene_morphology_sweep(
                    &lod,
                    &hierarchy,
                    audit_camera.camera,
                    loaded.color_space,
                ),
            })
            .collect::<Vec<_>>();

        let report =
            trellis_quality_report(profile, &distance_sweeps, &authored_sweep, &orbit_sweeps);
        eprintln!("BGS_TRELLIS_LOD_REPORT_BEGIN\n{report}BGS_TRELLIS_LOD_REPORT_END");
        if let Some(path) = env::var_os(TRELLIS_REPORT_ENV) {
            let path = PathBuf::from(path);
            fs::write(&path, &report).unwrap_or_else(|error| {
                panic!(
                    "failed to write Trellis LoD report {}: {error}",
                    path.display()
                )
            });
        }

        assert_trellis_distance_graph(&distance_sweeps);
        assert_trellis_authored_safety(&authored_sweep);
        for sweep in &orbit_sweeps {
            assert_trellis_morphology_safety(sweep.label, &sweep.rendered);
        }
    }

    fn verify_trellis_provenance(path: &Path) {
        let bytes = fs::read(path).unwrap_or_else(|error| {
            panic!(
                "failed to read local Trellis GLB {}: {error}",
                path.display()
            )
        });
        assert_eq!(
            bytes.len() as u64,
            TRELLIS_BYTE_LEN,
            "unexpected Trellis byte length; preflight sha256 must be {TRELLIS_SHA256_FOR_PREFLIGHT}"
        );
        let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(
            actual_sha256, TRELLIS_SHA256_FOR_PREFLIGHT,
            "canonical Trellis SHA-256 drifted"
        );
        let gltf = gltf::Gltf::from_slice_without_validation(&bytes)
            .expect("local Trellis artifact parses as GLB");
        let position_counts = gltf
            .document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .filter_map(|primitive| primitive.get(&gltf::Semantic::Positions))
            .map(|accessor| accessor.count())
            .collect::<Vec<_>>();
        assert_eq!(
            position_counts,
            [TRELLIS_SPLAT_COUNT],
            "canonical Trellis must contain exactly one 478,368-splat primitive"
        );
    }

    struct LoadedRealScene {
        cloud: PlanarGaussian3d,
        cloud_transform: Transform,
        authored_camera: Transform,
        color_space: GaussianColorSpace,
    }

    fn load_local_gaussian_scene(path: &Path) -> LoadedRealScene {
        let canonical = fs::canonicalize(path)
            .unwrap_or_else(|error| panic!("failed to canonicalize {}: {error}", path.display()));
        let root = canonical.parent().expect("Trellis path has a parent");
        let file_name = canonical
            .file_name()
            .expect("Trellis path has a file name")
            .to_string_lossy()
            .into_owned();

        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(AssetPlugin {
            file_path: root.display().to_string(),
            processed_file_path: root.display().to_string(),
            meta_check: AssetMetaCheck::Never,
            unapproved_path_mode: UnapprovedPathMode::Allow,
            ..default()
        });
        app.init_asset::<PlanarGaussian3d>();
        app.add_plugins(IoPlugin);

        let handle: Handle<GaussianScene> = app.world().resource::<AssetServer>().load(file_name);
        let deadline = Instant::now() + Duration::from_secs(60);
        let scene = loop {
            app.update();
            if let Some((load, dependency, recursive)) = app
                .world()
                .resource::<AssetServer>()
                .get_load_states(&handle)
            {
                match (&load, &dependency, &recursive) {
                    (LoadState::Failed(error), _, _)
                    | (_, DependencyLoadState::Failed(error), _)
                    | (_, _, RecursiveDependencyLoadState::Failed(error)) => {
                        panic!("local Trellis asset load failed: {error}")
                    }
                    (LoadState::Loaded, _, RecursiveDependencyLoadState::Loaded) => {
                        if let Some(scene) = app
                            .world()
                            .resource::<Assets<GaussianScene>>()
                            .get(&handle)
                            .cloned()
                        {
                            break scene;
                        }
                    }
                    _ => {}
                }
            }
            assert!(
                Instant::now() < deadline,
                "timed out loading local Trellis asset"
            );
            thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(scene.bundles.len(), 1, "Trellis primitive count drifted");
        assert_eq!(scene.cameras.len(), 1, "Trellis authored camera drifted");
        let bundle = &scene.bundles[0];
        let cloud = app
            .world()
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&bundle.cloud)
            .cloned()
            .expect("Trellis cloud dependency is loaded");
        LoadedRealScene {
            cloud,
            cloud_transform: bundle.transform,
            authored_camera: scene.cameras[0].transform,
            color_space: bundle.settings.color_space,
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct TrellisDistanceCamera {
        label: &'static str,
        root_coverage: f32,
        distance_in_root_radii: f32,
        camera: LodTestCamera,
    }

    #[derive(Clone, Copy, Debug)]
    struct TrellisOrbitCamera {
        label: &'static str,
        camera: LodTestCamera,
    }

    fn trellis_authored_camera(scene: &LoadedRealScene) -> LodTestCamera {
        let world_from_cloud = scene.cloud_transform.to_matrix();
        let cloud_from_world = world_from_cloud.inverse();
        let authored_position =
            cloud_from_world.transform_point3(scene.authored_camera.translation);
        let authored_target = cloud_from_world.transform_point3(
            scene.authored_camera.translation + scene.authored_camera.rotation * Vec3::NEG_Z,
        );
        let authored_up = cloud_from_world
            .transform_vector3(scene.authored_camera.rotation * Vec3::Y)
            .normalize_or_zero();
        LodTestCamera {
            position: authored_position,
            target: authored_target,
            up: authored_up,
            projection: LodProjection::Perspective {
                vertical_fov_radians: TRELLIS_FOV_Y,
            },
            near: 0.01,
            far: 1_000.0,
            viewport: [TRELLIS_RASTER_SIZE, TRELLIS_RASTER_SIZE],
        }
    }

    fn trellis_distance_cameras(
        scene: &LoadedRealScene,
        hierarchy: &ManifestLodHierarchy<'_>,
    ) -> [TrellisDistanceCamera; 3] {
        let [root] = hierarchy.roots() else {
            panic!(
                "canonical Trellis quality graph requires one root, found {}",
                hierarchy.roots().len()
            )
        };
        let metrics = hierarchy
            .metrics(*root)
            .expect("canonical Trellis root metrics exist");
        let radius = metrics.radius;
        assert!(radius.is_finite() && radius > 0.0);
        let tangent = (0.5 * TRELLIS_FOV_Y).tan();
        let cloud_from_world = scene.cloud_transform.to_matrix().inverse();
        let forward = cloud_from_world
            .transform_vector3(scene.authored_camera.rotation * Vec3::NEG_Z)
            .normalize_or_zero();
        let authored_up = cloud_from_world
            .transform_vector3(scene.authored_camera.rotation * Vec3::Y)
            .normalize_or_zero();
        let right = forward.cross(authored_up).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        assert!(forward != Vec3::ZERO && right != Vec3::ZERO && up != Vec3::ZERO);
        [("near", 0.8_f32), ("mid", 0.4_f32), ("far", 0.2_f32)].map(|(label, root_coverage)| {
            // LodView uses distance to the support surface rather than the root
            // center. Solve C = R / (tan(fov / 2) * (D - R)) for D so the
            // named scales have an exact, scene-invariant projected coverage.
            let distance = radius + radius / (root_coverage * tangent);
            let camera = LodTestCamera {
                position: metrics.center - forward * distance,
                target: metrics.center,
                up,
                projection: LodProjection::Perspective {
                    vertical_fov_radians: TRELLIS_FOV_Y,
                },
                near: 0.001,
                far: distance + radius * 4.0,
                viewport: [TRELLIS_RASTER_SIZE, TRELLIS_RASTER_SIZE],
            };
            let actual_coverage = trellis_lod_view(camera, TRELLIS_DEPLOYMENT_VIEWPORT_HEIGHT_PX)
                .projected_coverage(metrics);
            assert!(
                (actual_coverage - root_coverage).abs() <= 1e-5,
                "{label} root coverage drifted: requested={root_coverage}, actual={actual_coverage}"
            );
            TrellisDistanceCamera {
                label,
                root_coverage,
                distance_in_root_radii: distance / radius,
                camera,
            }
        })
    }

    fn trellis_orbit_cameras(
        profile: TrellisAuditProfile,
        mid_camera: LodTestCamera,
    ) -> Vec<TrellisOrbitCamera> {
        let target = mid_camera.target;
        let forward = (target - mid_camera.position).normalize_or_zero();
        let right = forward.cross(mid_camera.up).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let distance = mid_camera.position.distance(target);
        assert!(
            forward != Vec3::ZERO
                && right != Vec3::ZERO
                && up != Vec3::ZERO
                && distance.is_finite()
                && distance > 0.0
        );

        profile
            .orbit_specs()
            .iter()
            .copied()
            .map(|spec| {
                let yaw = Quat::from_axis_angle(up, spec.yaw_degrees.to_radians());
                let yawed_right = yaw * right;
                let pitch = Quat::from_axis_angle(yawed_right, spec.pitch_degrees.to_radians());
                let outward = (pitch * (yaw * -forward)).normalize_or_zero();
                let up_hint = (pitch * up).normalize_or_zero();
                let orbit_forward = -outward;
                let orbit_right = orbit_forward.cross(up_hint).normalize_or_zero();
                let orbit_up = orbit_right.cross(orbit_forward).normalize_or_zero();
                assert!(
                    outward != Vec3::ZERO && orbit_right != Vec3::ZERO && orbit_up != Vec3::ZERO
                );
                TrellisOrbitCamera {
                    label: spec.label,
                    camera: LodTestCamera {
                        position: target + outward * distance,
                        target,
                        up: orbit_up,
                        ..mid_camera
                    },
                }
            })
            .collect()
    }

    struct RealSceneQualitySample {
        active_gaussians: usize,
        achieved_max_error_px: f32,
        image: Vec<[f32; 4]>,
        covariance: CovarianceDiagnostics,
    }

    fn real_scene_quality_sample(
        lod: &bevy_gaussian_splatting::PlanarGaussian3dLod,
        hierarchy: &ManifestLodHierarchy<'_>,
        camera: LodTestCamera,
        quality: f32,
        color_space: GaussianColorSpace,
    ) -> RealSceneQualitySample {
        let settings = trellis_lod_settings(lod, quality);
        let metric_view = trellis_lod_view(camera, camera.viewport[1] as f32);
        let frontier = select_frontier(hierarchy, &AllResident, metric_view, &settings)
            .expect("Trellis quality selection succeeds");
        let gaussians =
            bevy_gaussian_splatting::testing::gather_frontier_gaussians(lod, &frontier.nodes)
                .expect("Trellis frontier resolves");
        assert_eq!(
            gaussians.len() as u64,
            frontier.status.active_gaussians,
            "Trellis gathered frontier count differs from selector status"
        );
        let (image, covariance) = render_color_and_covariance_image(
            &gaussians,
            camera,
            camera.viewport[0],
            camera.viewport[1],
            color_space,
        );
        RealSceneQualitySample {
            active_gaussians: gaussians.len(),
            achieved_max_error_px: frontier.status.achieved_max_error_px,
            image,
            covariance,
        }
    }

    fn real_scene_morphology_sweep(
        lod: &bevy_gaussian_splatting::PlanarGaussian3dLod,
        hierarchy: &ManifestLodHierarchy<'_>,
        camera: LodTestCamera,
        color_space: GaussianColorSpace,
    ) -> Vec<QualitySweepPoint> {
        let exact = real_scene_quality_sample(lod, hierarchy, camera, 1.0, color_space);
        TRELLIS_MORPHOLOGY_QUALITIES
            .into_iter()
            .map(|quality| {
                if quality_eq(quality, 1.0) {
                    quality_sweep_point(quality, &exact, &exact, camera)
                } else {
                    let sample =
                        real_scene_quality_sample(lod, hierarchy, camera, quality, color_space);
                    quality_sweep_point(quality, &sample, &exact, camera)
                }
            })
            .collect()
    }

    fn real_scene_active_count(
        lod: &bevy_gaussian_splatting::PlanarGaussian3dLod,
        hierarchy: &ManifestLodHierarchy<'_>,
        camera: LodTestCamera,
        quality: f32,
        selection_viewport_height_px: f32,
    ) -> usize {
        let settings = trellis_lod_settings(lod, quality);
        let view = trellis_lod_view(camera, selection_viewport_height_px);
        let frontier = select_frontier(hierarchy, &AllResident, view, &settings)
            .expect("Trellis dense quality selection succeeds");
        usize::try_from(frontier.status.active_gaussians)
            .expect("Trellis active count fits host usize")
    }

    fn trellis_lod_settings(
        lod: &bevy_gaussian_splatting::PlanarGaussian3dLod,
        quality: f32,
    ) -> GaussianLodSettings {
        let mut settings = GaussianLodSettings {
            quality,
            hysteresis: 0.0,
            frustum_culling: false,
            ..default()
        };
        settings.budgets.max_active_gaussians = TRELLIS_SPLAT_COUNT as u64 * 2;
        settings.budgets.max_traversal_nodes_per_view = lod.manifest.header.node_count.max(1);
        settings
    }

    fn trellis_lod_view(camera: LodTestCamera, viewport_height_px: f32) -> LodView {
        LodView::perspective(
            camera.position,
            viewport_height_px,
            match camera.projection {
                LodProjection::Perspective {
                    vertical_fov_radians,
                } => vertical_fov_radians,
                LodProjection::Orthographic { .. } => unreachable!(),
            },
            camera.near,
        )
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct CovarianceDiagnostics {
        maximum_major_sigma_px: f32,
        maximum_aspect_ratio: f32,
        projected_splats: usize,
        extreme_projected_splats: usize,
        visible_elongated_splats: usize,
    }

    #[derive(Clone, Copy)]
    struct ProjectedCovariance {
        center: Vec2,
        depth: f32,
        covariance: Vec3,
        opacity: f32,
        linear_rgb: Vec3,
    }

    // Keep these values, basis order, and expressions equivalent to
    // `material/spherical_harmonics.wgsl`. Coefficients are stored as one RGB
    // triplet per basis function, not as three channel-major bands.
    const SH_BASIS_CONSTANTS: [f32; 16] = [
        0.282_094_8,
        -0.488_602_52,
        0.488_602_52,
        -0.488_602_52,
        1.092_548_5,
        -1.092_548_5,
        0.315_391_57,
        -1.092_548_5,
        0.546_274_24,
        -0.590_043_6,
        2.890_611_4,
        -0.457_045_8,
        0.373_176_34,
        -0.457_045_8,
        1.445_305_7,
        -0.590_043_6,
    ];

    fn spherical_harmonic_basis(ray_direction: Vec3) -> [f32; 16] {
        let squared = ray_direction * ray_direction;
        [
            SH_BASIS_CONSTANTS[0],
            SH_BASIS_CONSTANTS[1] * ray_direction.y,
            SH_BASIS_CONSTANTS[2] * ray_direction.z,
            SH_BASIS_CONSTANTS[3] * ray_direction.x,
            SH_BASIS_CONSTANTS[4] * ray_direction.x * ray_direction.y,
            SH_BASIS_CONSTANTS[5] * ray_direction.y * ray_direction.z,
            SH_BASIS_CONSTANTS[6] * (2.0 * squared.z - squared.x - squared.y),
            SH_BASIS_CONSTANTS[7] * ray_direction.x * ray_direction.z,
            SH_BASIS_CONSTANTS[8] * (squared.x - squared.y),
            SH_BASIS_CONSTANTS[9] * ray_direction.y * (3.0 * squared.x - squared.y),
            SH_BASIS_CONSTANTS[10] * ray_direction.x * ray_direction.y * ray_direction.z,
            SH_BASIS_CONSTANTS[11] * ray_direction.y * (4.0 * squared.z - squared.x - squared.y),
            SH_BASIS_CONSTANTS[12]
                * ray_direction.z
                * (2.0 * squared.z - 3.0 * squared.x - 3.0 * squared.y),
            SH_BASIS_CONSTANTS[13] * ray_direction.x * (4.0 * squared.z - squared.x - squared.y),
            SH_BASIS_CONSTANTS[14] * ray_direction.z * (squared.x - squared.y),
            SH_BASIS_CONSTANTS[15] * ray_direction.x * (squared.x - 3.0 * squared.y),
        ]
    }

    fn sh_coefficient_rgb(
        spherical_harmonic: &SphericalHarmonicCoefficients,
        basis: usize,
    ) -> Vec3 {
        let base = basis * 3;
        let coefficient = |offset| {
            spherical_harmonic
                .coefficients
                .get(base + offset)
                .copied()
                .unwrap_or(0.0)
        };
        Vec3::new(coefficient(0), coefficient(1), coefficient(2))
    }

    fn spherical_harmonics_lookup_degree(
        ray_direction: Vec3,
        spherical_harmonic: &SphericalHarmonicCoefficients,
        maximum_degree: usize,
    ) -> Vec3 {
        let squared = ray_direction * ray_direction;
        let mut color =
            Vec3::splat(0.5) + sh_coefficient_rgb(spherical_harmonic, 0) * SH_BASIS_CONSTANTS[0];

        if maximum_degree >= 1 {
            color += (sh_coefficient_rgb(spherical_harmonic, 1) * SH_BASIS_CONSTANTS[1])
                * ray_direction.y;
            color += (sh_coefficient_rgb(spherical_harmonic, 2) * SH_BASIS_CONSTANTS[2])
                * ray_direction.z;
            color += (sh_coefficient_rgb(spherical_harmonic, 3) * SH_BASIS_CONSTANTS[3])
                * ray_direction.x;
        }
        if maximum_degree >= 2 {
            color += (sh_coefficient_rgb(spherical_harmonic, 4) * SH_BASIS_CONSTANTS[4])
                * ray_direction.x
                * ray_direction.y;
            color += (sh_coefficient_rgb(spherical_harmonic, 5) * SH_BASIS_CONSTANTS[5])
                * ray_direction.y
                * ray_direction.z;
            color += (sh_coefficient_rgb(spherical_harmonic, 6) * SH_BASIS_CONSTANTS[6])
                * (2.0 * squared.z - squared.x - squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 7) * SH_BASIS_CONSTANTS[7])
                * ray_direction.x
                * ray_direction.z;
            color += (sh_coefficient_rgb(spherical_harmonic, 8) * SH_BASIS_CONSTANTS[8])
                * (squared.x - squared.y);
        }
        if maximum_degree >= 3 {
            color += (sh_coefficient_rgb(spherical_harmonic, 9) * SH_BASIS_CONSTANTS[9])
                * ray_direction.y
                * (3.0 * squared.x - squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 10) * SH_BASIS_CONSTANTS[10])
                * ray_direction.x
                * ray_direction.y
                * ray_direction.z;
            color += (sh_coefficient_rgb(spherical_harmonic, 11) * SH_BASIS_CONSTANTS[11])
                * ray_direction.y
                * (4.0 * squared.z - squared.x - squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 12) * SH_BASIS_CONSTANTS[12])
                * ray_direction.z
                * (2.0 * squared.z - 3.0 * squared.x - 3.0 * squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 13) * SH_BASIS_CONSTANTS[13])
                * ray_direction.x
                * (4.0 * squared.z - squared.x - squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 14) * SH_BASIS_CONSTANTS[14])
                * ray_direction.z
                * (squared.x - squared.y);
            color += (sh_coefficient_rgb(spherical_harmonic, 15) * SH_BASIS_CONSTANTS[15])
                * ray_direction.x
                * (squared.x - 3.0 * squared.y);
        }
        color
    }

    fn spherical_harmonics_linear_color(
        ray_direction: Vec3,
        spherical_harmonic: &SphericalHarmonicCoefficients,
        color_space: GaussianColorSpace,
    ) -> Vec3 {
        let display_color =
            spherical_harmonics_lookup_degree(ray_direction, spherical_harmonic, SH_DEGREE.min(3));
        match color_space {
            GaussianColorSpace::LinRec709Display => display_color,
            GaussianColorSpace::SrgbRec709Display => Vec3::new(
                srgb_display_channel_to_linear(display_color.x),
                srgb_display_channel_to_linear(display_color.y),
                srgb_display_channel_to_linear(display_color.z),
            ),
        }
    }

    fn srgb_display_channel_to_linear(value: f32) -> f32 {
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    fn composite_linear_source_over(pixel: &mut [f32; 4], linear_rgb: Vec3, alpha: f32) {
        let transmittance = 1.0 - alpha;
        pixel[0] = linear_rgb.x * alpha + pixel[0] * transmittance;
        pixel[1] = linear_rgb.y * alpha + pixel[1] * transmittance;
        pixel[2] = linear_rgb.z * alpha + pixel[2] * transmittance;
        pixel[3] = alpha + pixel[3] * transmittance;
    }

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        let maximum_error = (actual - expected).abs().max_element();
        assert!(
            maximum_error <= 1e-6,
            "vectors differ by {maximum_error}: actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_rgba_close(actual: [f32; 4], expected: [f32; 4]) {
        for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= 1e-6,
                "RGBA channel {channel} differs by {error}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn sh_basis_and_dc_match_the_renderer_shader() {
        let basis = spherical_harmonic_basis(Vec3::X);
        assert!((basis[0] - 0.282_094_8).abs() < 1e-7);
        assert!((basis[3] + 0.488_602_52).abs() < 1e-7);
        assert!((basis[8] - 0.546_274_24).abs() < 1e-7);
        assert!((basis[15] + 0.590_043_6).abs() < 1e-7);

        let mut spherical_harmonic = SphericalHarmonicCoefficients::default();
        assert_vec3_close(
            spherical_harmonics_lookup_degree(Vec3::Z, &spherical_harmonic, 0),
            Vec3::splat(0.5),
        );
        spherical_harmonic.set(0, 1.0);
        spherical_harmonic.set(1, -2.0);
        spherical_harmonic.set(2, 0.25);
        assert_vec3_close(
            spherical_harmonics_lookup_degree(Vec3::Z, &spherical_harmonic, 0),
            Vec3::splat(0.5) + Vec3::new(1.0, -2.0, 0.25) * 0.282_094_8,
        );

        let zero = SphericalHarmonicCoefficients::default();
        assert_vec3_close(
            spherical_harmonics_linear_color(Vec3::Z, &zero, GaussianColorSpace::LinRec709Display),
            Vec3::splat(0.5),
        );
        assert_vec3_close(
            spherical_harmonics_linear_color(Vec3::Z, &zero, GaussianColorSpace::SrgbRec709Display),
            Vec3::splat(srgb_display_channel_to_linear(0.5)),
        );
    }

    #[test]
    fn sh_directional_bands_match_the_renderer_shader() {
        let direction = Vec3::new(0.6, 0.8, 0.0);
        let mut degree_one = SphericalHarmonicCoefficients::default();
        degree_one.set(3, 1.0);
        degree_one.set(7, 1.0);
        degree_one.set(11, 1.0);
        assert_vec3_close(
            spherical_harmonics_lookup_degree(direction, &degree_one, 1),
            Vec3::new(0.5 - 0.488_602_52 * 0.8, 0.5, 0.5 - 0.488_602_52 * 0.6),
        );

        let mut upper_bands = SphericalHarmonicCoefficients::default();
        upper_bands.set(18, 1.0);
        upper_bands.set(37, 1.0);
        assert_vec3_close(
            spherical_harmonics_lookup_degree(Vec3::Z, &upper_bands, 3),
            Vec3::new(0.5 + 2.0 * 0.315_391_57, 0.5 + 2.0 * 0.373_176_34, 0.5),
        );
    }

    #[test]
    fn premultiplied_source_over_compositing_matches_the_renderer() {
        let mut pixel = [0.0; 4];
        composite_linear_source_over(&mut pixel, Vec3::new(0.8, 0.2, 0.1), 0.25);
        composite_linear_source_over(&mut pixel, Vec3::new(0.1, 0.4, 0.9), 0.5);
        assert_rgba_close(pixel, [0.15, 0.225, 0.4625, 0.625]);
    }

    fn render_color_and_covariance_image(
        gaussians: &[Gaussian3d],
        camera: LodTestCamera,
        width: u32,
        height: u32,
        color_space: GaussianColorSpace,
    ) -> (Vec<[f32; 4]>, CovarianceDiagnostics) {
        let forward = (camera.target - camera.position).normalize_or_zero();
        let right = forward.cross(camera.up).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        assert!(forward != Vec3::ZERO && right != Vec3::ZERO && up != Vec3::ZERO);
        let fov = match camera.projection {
            LodProjection::Perspective {
                vertical_fov_radians,
            } => vertical_fov_radians,
            LodProjection::Orthographic { .. } => {
                panic!("Trellis color/covariance audit requires perspective cameras")
            }
        };
        let focal_y = 0.5 * height as f32 / (0.5 * fov).tan();
        let focal_x = focal_y;
        let mut diagnostics = CovarianceDiagnostics::default();
        let mut projected = Vec::with_capacity(gaussians.len());
        for gaussian in gaussians {
            let position = Vec3::from_array(gaussian.position_visibility.position);
            let relative = position - camera.position;
            let depth = relative.dot(forward);
            if depth < camera.near || depth > camera.far || !depth.is_finite() {
                continue;
            }
            let view_x = relative.dot(right);
            let view_y = relative.dot(up);
            let center = Vec2::new(
                width as f32 * 0.5 + focal_x * view_x / depth,
                height as f32 * 0.5 - focal_y * view_y / depth,
            );
            let packed = compute_covariance_3d(
                Vec4::from_array(gaussian.rotation.rotation),
                Vec3::from_array(gaussian.scale_opacity.scale.map(f32::abs)),
            );
            let covariance = Mat3::from_cols(
                Vec3::new(packed[0], packed[1], packed[2]),
                Vec3::new(packed[1], packed[3], packed[4]),
                Vec3::new(packed[2], packed[4], packed[5]),
            );
            let reciprocal_depth = 1.0 / depth;
            let reciprocal_depth_squared = reciprocal_depth * reciprocal_depth;
            let jacobian_x = right * (focal_x * reciprocal_depth)
                - forward * (focal_x * view_x * reciprocal_depth_squared);
            let jacobian_y = -up * (focal_y * reciprocal_depth)
                + forward * (focal_y * view_y * reciprocal_depth_squared);
            let filtered_covariance = gaussian_mip_filter_covariance_2d([
                jacobian_x.dot(covariance * jacobian_x),
                jacobian_x.dot(covariance * jacobian_y),
                jacobian_y.dot(covariance * jacobian_y),
            ]);
            let [covariance_x, covariance_xy, covariance_y] = filtered_covariance.covariance;
            let midpoint = 0.5 * (covariance_x + covariance_y);
            let determinant = covariance_x * covariance_y - covariance_xy * covariance_xy;
            let discriminant = (midpoint * midpoint - determinant).max(0.0).sqrt();
            let major = (midpoint + discriminant).max(0.0).sqrt();
            let minor = (midpoint - discriminant).max(1e-8).sqrt();
            let aspect = major / minor;
            if determinant <= 1e-8
                || !center.is_finite()
                || !major.is_finite()
                || !aspect.is_finite()
            {
                continue;
            }
            diagnostics.projected_splats += 1;
            diagnostics.maximum_major_sigma_px = diagnostics.maximum_major_sigma_px.max(major);
            diagnostics.maximum_aspect_ratio = diagnostics.maximum_aspect_ratio.max(aspect);
            diagnostics.extreme_projected_splats += usize::from(major > 8.0 && aspect > 20.0);
            let opacity = (gaussian.scale_opacity.opacity
                * gaussian.position_visibility.visibility)
                * filtered_covariance.opacity_scale;
            let opacity = opacity.clamp(0.0, 0.999);
            let support_radius = 3.0 * major;
            let intersects_viewport = center.x + support_radius >= 0.0
                && center.x - support_radius < width as f32
                && center.y + support_radius >= 0.0
                && center.y - support_radius < height as f32;
            diagnostics.visible_elongated_splats += usize::from(
                opacity >= FOREGROUND_ALPHA
                    && intersects_viewport
                    && major > VISIBLE_ELONGATION_MIN_MAJOR_SIGMA_PX
                    && aspect > VISIBLE_ELONGATION_MIN_ASPECT_RATIO,
            );
            projected.push(ProjectedCovariance {
                center,
                depth,
                covariance: Vec3::new(covariance_x, covariance_xy, covariance_y),
                opacity,
                // The audit cameras and Gaussian positions are both expressed in
                // cloud-local coordinates, matching the renderer's world-to-local
                // direction before its spherical-harmonic lookup.
                linear_rgb: spherical_harmonics_linear_color(
                    relative.normalize_or_zero(),
                    &gaussian.spherical_harmonic,
                    color_space,
                ),
            });
        }
        projected.sort_by(|left, right| right.depth.total_cmp(&left.depth));

        let mut image = vec![[0.0_f32; 4]; (width * height) as usize];
        for gaussian in projected {
            let covariance_x = gaussian.covariance.x;
            let covariance_xy = gaussian.covariance.y;
            let covariance_y = gaussian.covariance.z;
            let determinant = covariance_x * covariance_y - covariance_xy * covariance_xy;
            if determinant <= 1e-8 {
                continue;
            }
            let midpoint = 0.5 * (covariance_x + covariance_y);
            let discriminant = (midpoint * midpoint - determinant).max(0.0).sqrt();
            let radius = (3.0 * (midpoint + discriminant).max(0.0).sqrt()).ceil() as i32;
            let min_x = (gaussian.center.x.floor() as i32 - radius).max(0);
            let max_x = (gaussian.center.x.ceil() as i32 + radius).min(width as i32 - 1);
            let min_y = (gaussian.center.y.floor() as i32 - radius).max(0);
            let max_y = (gaussian.center.y.ceil() as i32 + radius).min(height as i32 - 1);
            if min_x > max_x || min_y > max_y {
                continue;
            }
            let inverse_x = covariance_y / determinant;
            let inverse_xy = -covariance_xy / determinant;
            let inverse_y = covariance_x / determinant;
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let delta = Vec2::new(
                        x as f32 + 0.5 - gaussian.center.x,
                        y as f32 + 0.5 - gaussian.center.y,
                    );
                    let mahalanobis = inverse_x * delta.x * delta.x
                        + 2.0 * inverse_xy * delta.x * delta.y
                        + inverse_y * delta.y * delta.y;
                    if mahalanobis > 9.0 {
                        continue;
                    }
                    let source_alpha =
                        (gaussian.opacity * (-0.5 * mahalanobis).exp()).clamp(0.0, 0.999);
                    let pixel = &mut image[(y as u32 * width + x as u32) as usize];
                    composite_linear_source_over(pixel, gaussian.linear_rgb, source_alpha);
                }
            }
        }
        (image, diagnostics)
    }

    #[derive(Clone, Copy, Debug)]
    struct QualityThresholds {
        minimum_foreground_psnr: f64,
        minimum_ssim: f64,
        minimum_iou: f64,
        maximum_alpha_mae: f64,
        maximum_spill: f64,
    }

    #[derive(Clone, Copy, Debug)]
    struct QualityObservation {
        full: ImageMetrics,
        foreground_roi: ImageMetrics,
        spill_outside_dilated_reference: f64,
        foreground_spill_pixels_outside_dilated_reference: usize,
        maximum_foreground_distance_px: usize,
    }

    #[derive(Clone, Copy, Debug)]
    struct QualitySweepPoint {
        quality: f32,
        active_gaussians: usize,
        base_target_error_px: Option<f32>,
        error_authority: Option<f32>,
        effective_error_cap_px: Option<f32>,
        achieved_max_error_px: f32,
        covariance: CovarianceDiagnostics,
        observation: QualityObservation,
    }

    #[derive(Clone, Copy, Debug)]
    struct SelectionSweepPoint {
        quality: f32,
        active_gaussians: usize,
    }

    #[derive(Debug)]
    struct TrellisDistanceSweep {
        camera: TrellisDistanceCamera,
        rendered: Vec<QualitySweepPoint>,
        deployment_selection: Vec<SelectionSweepPoint>,
    }

    #[derive(Debug)]
    struct TrellisMorphologySweep {
        label: &'static str,
        rendered: Vec<QualitySweepPoint>,
    }

    fn quality_sweep_point(
        quality: f32,
        sample: &RealSceneQualitySample,
        exact: &RealSceneQualitySample,
        camera: LodTestCamera,
    ) -> QualitySweepPoint {
        let settings = GaussianLodSettings {
            quality,
            ..Default::default()
        };
        let target = settings.quality_target();
        QualitySweepPoint {
            quality,
            active_gaussians: sample.active_gaussians,
            base_target_error_px: target.max_screen_space_error_px(),
            error_authority: target.error_authority(),
            effective_error_cap_px: target.effective_max_screen_space_error_px(),
            achieved_max_error_px: sample.achieved_max_error_px,
            covariance: sample.covariance,
            observation: quality_observation(
                &exact.image,
                &sample.image,
                camera.viewport[0],
                camera.viewport[1],
            ),
        }
    }

    fn quality_observation(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        width: u32,
        height: u32,
    ) -> QualityObservation {
        let full = compare_linear_rgba(reference, candidate, FOREGROUND_ALPHA)
            .expect("quality images have matching dimensions");
        let (reference_roi, candidate_roi) =
            foreground_union_roi(reference, candidate, width, height, 4);
        let foreground_roi = compare_linear_rgba(&reference_roi, &candidate_roi, FOREGROUND_ALPHA)
            .expect("quality foreground ROIs match");
        let spill = alpha_spill_diagnostics(reference, candidate, width, height, SPILL_DILATION_PX);
        QualityObservation {
            full,
            foreground_roi,
            spill_outside_dilated_reference: spill.alpha_mass_ratio,
            foreground_spill_pixels_outside_dilated_reference: spill
                .foreground_pixels_outside_dilated_reference,
            maximum_foreground_distance_px: spill.maximum_foreground_distance_px,
        }
    }

    fn assert_thresholds(label: &str, observation: QualityObservation, limits: QualityThresholds) {
        assert!(
            observation.full.foreground_psnr_rgb >= limits.minimum_foreground_psnr,
            "{label} foreground PSNR below {} dB: {observation:?}",
            limits.minimum_foreground_psnr
        );
        assert!(
            observation.foreground_roi.luminance_ssim >= limits.minimum_ssim,
            "{label} foreground ROI SSIM below {}: {observation:?}",
            limits.minimum_ssim
        );
        assert!(
            observation.full.foreground_iou >= limits.minimum_iou,
            "{label} foreground IoU below {}: {observation:?}",
            limits.minimum_iou
        );
        assert!(
            observation.foreground_roi.alpha_mae <= limits.maximum_alpha_mae,
            "{label} foreground ROI alpha MAE above {}: {observation:?}",
            limits.maximum_alpha_mae
        );
        assert!(
            observation.spill_outside_dilated_reference <= limits.maximum_spill,
            "{label} covariance spill above {}: {observation:?}",
            limits.maximum_spill
        );
    }

    fn assert_alpha_morphology_bound(label: &str, observation: QualityObservation) {
        assert_eq!(
            observation.foreground_spill_pixels_outside_dilated_reference, 0,
            "{label} renders alpha >= {FOREGROUND_ALPHA} outside the q=1 foreground mask dilated by {SPILL_DILATION_PX}px: {observation:?}"
        );
        assert!(
            observation.maximum_foreground_distance_px <= SPILL_DILATION_PX,
            "{label} foreground extends {}px from the nearest q=1 foreground pixel (limit {SPILL_DILATION_PX}px): {observation:?}",
            observation.maximum_foreground_distance_px
        );
    }

    fn assert_monotonic(label: &str, lower: QualityObservation, higher: QualityObservation) {
        assert!(
            higher.full.foreground_psnr_rgb + 0.25 >= lower.full.foreground_psnr_rgb,
            "{label} foreground PSNR regressed: lower={lower:?}, higher={higher:?}"
        );
        assert!(
            higher.foreground_roi.luminance_ssim + 0.005 >= lower.foreground_roi.luminance_ssim,
            "{label} foreground SSIM regressed: lower={lower:?}, higher={higher:?}"
        );
        assert!(
            higher.full.foreground_iou + 0.005 >= lower.full.foreground_iou,
            "{label} foreground IoU regressed: lower={lower:?}, higher={higher:?}"
        );
        assert!(
            higher.foreground_roi.alpha_mae <= lower.foreground_roi.alpha_mae + 0.005,
            "{label} alpha MAE regressed: lower={lower:?}, higher={higher:?}"
        );
        assert!(
            higher.spill_outside_dilated_reference <= lower.spill_outside_dilated_reference + 0.002,
            "{label} covariance spill regressed: lower={lower:?}, higher={higher:?}"
        );
    }

    fn trellis_dense_selection_qualities() -> Vec<f32> {
        (0..=200).map(|step| step as f32 / 200.0).collect()
    }

    #[derive(Clone, Copy, Debug)]
    struct UtilityGoal {
        label: &'static str,
        minimum_savings_percent: usize,
        minimum_psnr_db: f64,
    }

    const UTILITY_GOALS: [UtilityGoal; 2] = [
        UtilityGoal {
            label: "10% savings at >=33 dB",
            minimum_savings_percent: 10,
            minimum_psnr_db: 33.0,
        },
        UtilityGoal {
            label: "5% savings at >=35 dB",
            minimum_savings_percent: 5,
            minimum_psnr_db: 35.0,
        },
    ];

    #[derive(Clone, Copy, Debug)]
    struct ContinuitySummary {
        distinct_cuts: usize,
        q20: f32,
        q50: f32,
        q80: f32,
        widest_interior_step: usize,
        widest_step_lower_quality: f32,
        widest_step_higher_quality: f32,
        mean_active: f64,
    }

    fn continuity_summary(sweep: &[SelectionSweepPoint]) -> ContinuitySummary {
        let distinct_cuts = sweep
            .iter()
            .map(|point| point.active_gaussians)
            .collect::<BTreeSet<_>>()
            .len();
        let interior = sweep
            .iter()
            .filter(|point| point.quality > 0.0 && point.quality < 1.0)
            .collect::<Vec<_>>();
        let mean_active = interior
            .iter()
            .map(|point| point.active_gaussians as f64)
            .sum::<f64>()
            / interior.len() as f64;
        let (widest_interior_step, widest_step_lower_quality, widest_step_higher_quality) = sweep
            .windows(2)
            .filter_map(|pair| {
                let [lower, higher] = pair else {
                    unreachable!()
                };
                (lower.quality > 0.0 && higher.quality < 1.0).then_some((
                    higher
                        .active_gaussians
                        .saturating_sub(lower.active_gaussians),
                    lower.quality,
                    higher.quality,
                ))
            })
            .max_by(|left, right| left.0.cmp(&right.0))
            .expect("dense deployment sweep has interior steps");
        ContinuitySummary {
            distinct_cuts,
            q20: deployment_crossing_quality(sweep, 20),
            q50: deployment_crossing_quality(sweep, 50),
            q80: deployment_crossing_quality(sweep, 80),
            widest_interior_step,
            widest_step_lower_quality,
            widest_step_higher_quality,
            mean_active,
        }
    }

    fn deployment_crossing_quality(sweep: &[SelectionSweepPoint], percent: usize) -> f32 {
        sweep
            .iter()
            .find(|point| point.active_gaussians * 100 >= TRELLIS_SPLAT_COUNT * percent)
            .unwrap_or_else(|| panic!("deployment sweep never crosses {percent}% active"))
            .quality
    }

    fn meets_utility_goal(point: &QualitySweepPoint, goal: UtilityGoal) -> bool {
        point.quality > 0.0
            && point.quality < 1.0
            && point.active_gaussians * 100
                <= TRELLIS_SPLAT_COUNT * (100 - goal.minimum_savings_percent)
            && point.observation.full.foreground_psnr_rgb >= goal.minimum_psnr_db
    }

    fn utility_anchor(
        sweep: &[QualitySweepPoint],
        goal: UtilityGoal,
    ) -> Option<&QualitySweepPoint> {
        sweep
            .iter()
            .filter(|point| meets_utility_goal(point, goal))
            .min_by(|left, right| {
                left.active_gaussians
                    .cmp(&right.active_gaussians)
                    .then_with(|| left.quality.total_cmp(&right.quality))
            })
    }

    fn common_utility_quality(sweeps: &[TrellisDistanceSweep], goal: UtilityGoal) -> Option<f32> {
        sweeps
            .first()?
            .rendered
            .iter()
            .filter_map(|candidate| {
                let quality = candidate.quality;
                let points = sweeps
                    .iter()
                    .map(|sweep| point_at_quality(&sweep.rendered, quality))
                    .collect::<Vec<_>>();
                points
                    .iter()
                    .all(|point| meets_utility_goal(point, goal))
                    .then_some((
                        quality,
                        points
                            .iter()
                            .map(|point| point.active_gaussians)
                            .sum::<usize>(),
                    ))
            })
            .min_by(|left, right| {
                left.1
                    .cmp(&right.1)
                    .then_with(|| left.0.total_cmp(&right.0))
            })
            .map(|(quality, _)| quality)
    }

    fn trellis_quality_report(
        profile: TrellisAuditProfile,
        sweeps: &[TrellisDistanceSweep],
        authored: &[QualitySweepPoint],
        orbits: &[TrellisMorphologySweep],
    ) -> String {
        assert_eq!(sweeps.len(), 3);
        let mut report = String::new();
        writeln!(report, "# Canonical Trellis LoD quality report").unwrap();
        writeln!(report).unwrap();
        writeln!(report, "- Audit profile: `{}`", profile.as_str()).unwrap();
        writeln!(
            report,
            "- Source: {TRELLIS_SPLAT_COUNT} Gaussians, SHA-256 `{TRELLIS_SHA256_FOR_PREFLIGHT}`"
        )
        .unwrap();
        writeln!(
            report,
            "- Metric cuts and deterministic image oracle: {TRELLIS_RASTER_SIZE}px selection height and {TRELLIS_RASTER_SIZE}x{TRELLIS_RASTER_SIZE} raster"
        )
        .unwrap();
        writeln!(
            report,
            "- Deployment count graph: {:.0}px selection height (reported separately; never paired with 192px PSNR)",
            TRELLIS_DEPLOYMENT_VIEWPORT_HEIGHT_PX
        )
        .unwrap();
        writeln!(
            report,
            "- PSNR: linear-RGB foreground-union PSNR against quality 1 at the same camera"
        )
        .unwrap();
        writeln!(
            report,
            "- Morphology: authored camera plus {} deterministic mid-coverage orbit views; visible elongation means major sigma >{VISIBLE_ELONGATION_MIN_MAJOR_SIGMA_PX:.1}px and aspect >{VISIBLE_ELONGATION_MIN_ASPECT_RATIO:.1}:1",
            orbits.len()
        )
        .unwrap();
        writeln!(report).unwrap();
        writeln!(report, "## Camera-distance contract").unwrap();
        writeln!(report).unwrap();
        writeln!(
            report,
            "| scale | root coverage | center distance / root radius |"
        )
        .unwrap();
        writeln!(report, "|---|---:|---:|").unwrap();
        for sweep in sweeps {
            writeln!(
                report,
                "| {} | {:.0}% | {:.3} R |",
                sweep.camera.label,
                sweep.camera.root_coverage * 100.0,
                sweep.camera.distance_in_root_radii
            )
            .unwrap();
        }
        writeln!(report).unwrap();
        writeln!(report, "## Quality graph").unwrap();
        writeln!(report).unwrap();
        writeln!(
            report,
            "| q | base target px | error authority | effective cap px | near 192px active (% source) | near PSNR dB | mid 192px active (% source) | mid PSNR dB | far 192px active (% source) | far PSNR dB |"
        )
        .unwrap();
        writeln!(
            report,
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        .unwrap();
        for (index, first) in sweeps[0].rendered.iter().enumerate() {
            let points = sweeps
                .iter()
                .map(|sweep| {
                    let point = &sweep.rendered[index];
                    assert!(quality_eq(point.quality, first.quality));
                    assert_eq!(point.base_target_error_px, first.base_target_error_px);
                    assert_eq!(point.error_authority, first.error_authority);
                    assert_eq!(point.effective_error_cap_px, first.effective_error_cap_px);
                    point
                })
                .collect::<Vec<_>>();
            writeln!(
                report,
                "| {:.4} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                first.quality,
                format_base_target_error(first),
                format_error_authority(first),
                format_effective_error_cap(first),
                format_active(points[0].active_gaussians),
                format_psnr(points[0].observation.full.foreground_psnr_rgb),
                format_active(points[1].active_gaussians),
                format_psnr(points[1].observation.full.foreground_psnr_rgb),
                format_active(points[2].active_gaussians),
                format_psnr(points[2].observation.full.foreground_psnr_rgb),
            )
            .unwrap();
        }
        writeln!(report).unwrap();
        writeln!(report, "## 1080p deployment continuity summary").unwrap();
        writeln!(report).unwrap();
        writeln!(
            report,
            "| scale | distinct cuts | q20 | q50 | q80 | widest interior .005 step | mean active (% source) |"
        )
        .unwrap();
        writeln!(report, "|---|---:|---:|---:|---:|---:|---:|").unwrap();
        for sweep in sweeps {
            let summary = continuity_summary(&sweep.deployment_selection);
            writeln!(
                report,
                "| {} | {} | {:.3} | {:.3} | {:.3} | {} ({:.2}%, q={:.3}->{:.3}) | {:.1} ({:.2}%) |",
                sweep.camera.label,
                summary.distinct_cuts,
                summary.q20,
                summary.q50,
                summary.q80,
                summary.widest_interior_step,
                summary.widest_interior_step as f64 * 100.0 / TRELLIS_SPLAT_COUNT as f64,
                summary.widest_step_lower_quality,
                summary.widest_step_higher_quality,
                summary.mean_active,
                summary.mean_active * 100.0 / TRELLIS_SPLAT_COUNT as f64,
            )
            .unwrap();
        }
        writeln!(report).unwrap();
        writeln!(report, "## Common utility anchors and safety diagnostics").unwrap();
        writeln!(report).unwrap();
        for goal in UTILITY_GOALS {
            match common_utility_quality(sweeps, goal) {
                Some(quality) => {
                    writeln!(report, "- {}: q={quality:.4}", goal.label).unwrap();
                }
                None => {
                    writeln!(report, "- {}: no common interior anchor", goal.label).unwrap();
                }
            }
        }
        writeln!(report).unwrap();
        writeln!(
            report,
            "| purpose | camera | q | 192px metric active (% source) | PSNR dB | achieved max px | ROI SSIM | IoU | alpha MAE | spill | alpha>=.02 outside q1+2px | max foreground distance px | max sigma px | max aspect | projected splats | extreme splats (%) | visible elongated splats |"
        )
        .unwrap();
        writeln!(
            report,
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        .unwrap();
        for goal in UTILITY_GOALS {
            if let Some(quality) = common_utility_quality(sweeps, goal) {
                for sweep in sweeps {
                    write_diagnostic_row(
                        &mut report,
                        goal.label,
                        sweep.camera.label,
                        point_at_quality(&sweep.rendered, quality),
                    );
                }
            }
        }
        for sweep in sweeps {
            for quality in [0.95, 0.99, 1.0] {
                write_diagnostic_row(
                    &mut report,
                    "safety",
                    sweep.camera.label,
                    point_at_quality(&sweep.rendered, quality),
                );
            }
        }
        for point in authored {
            write_diagnostic_row(&mut report, "morphology", "authored", point);
        }
        for sweep in orbits {
            for point in &sweep.rendered {
                write_diagnostic_row(&mut report, "morphology", sweep.label, point);
            }
        }
        report
    }

    fn write_diagnostic_row(
        report: &mut String,
        purpose: &str,
        label: &str,
        point: &QualitySweepPoint,
    ) {
        writeln!(
            report,
            "| {purpose} | {label} | {:.4} | {} | {} | {:.3} | {:.5} | {:.5} | {:.6} | {:.6} | {} | {} | {:.3} | {:.3} | {} | {} ({:.4}%) | {} |",
            point.quality,
            format_active(point.active_gaussians),
            format_psnr(point.observation.full.foreground_psnr_rgb),
            point.achieved_max_error_px,
            point.observation.foreground_roi.luminance_ssim,
            point.observation.full.foreground_iou,
            point.observation.foreground_roi.alpha_mae,
            point.observation.spill_outside_dilated_reference,
            point
                .observation
                .foreground_spill_pixels_outside_dilated_reference,
            point.observation.maximum_foreground_distance_px,
            point.covariance.maximum_major_sigma_px,
            point.covariance.maximum_aspect_ratio,
            point.covariance.projected_splats,
            point.covariance.extreme_projected_splats,
            extreme_projected_percentage(point.covariance),
            point.covariance.visible_elongated_splats,
        )
        .unwrap();
    }

    fn format_base_target_error(point: &QualitySweepPoint) -> String {
        match point.base_target_error_px {
            Some(value) => format!("{value:.3}"),
            None if point.quality <= 0.0 => "roots".to_owned(),
            None => "exact".to_owned(),
        }
    }

    fn format_error_authority(point: &QualitySweepPoint) -> String {
        match point.error_authority {
            Some(value) => format!("{value:.6}"),
            None if point.quality <= 0.0 => "roots".to_owned(),
            None => "exact".to_owned(),
        }
    }

    fn format_effective_error_cap(point: &QualitySweepPoint) -> String {
        match point.effective_error_cap_px {
            Some(value) => format!("{value:.3}"),
            None if point.quality <= 0.0 => "roots".to_owned(),
            None => "exact".to_owned(),
        }
    }

    fn extreme_projected_percentage(covariance: CovarianceDiagnostics) -> f64 {
        if covariance.projected_splats == 0 {
            0.0
        } else {
            covariance.extreme_projected_splats as f64 * 100.0 / covariance.projected_splats as f64
        }
    }

    fn format_active(active_gaussians: usize) -> String {
        format!(
            "{} ({:.2}%)",
            active_gaussians,
            active_gaussians as f64 * 100.0 / TRELLIS_SPLAT_COUNT as f64
        )
    }

    fn format_psnr(value: f64) -> String {
        if value.is_infinite() {
            "inf".to_owned()
        } else {
            format!("{value:.2}")
        }
    }

    fn point_at_quality(sweep: &[QualitySweepPoint], quality: f32) -> &QualitySweepPoint {
        sweep
            .iter()
            .find(|point| quality_eq(point.quality, quality))
            .unwrap_or_else(|| panic!("Trellis rendered sweep is missing q={quality:.4}"))
    }

    fn selection_point_at_quality(
        sweep: &[SelectionSweepPoint],
        quality: f32,
    ) -> SelectionSweepPoint {
        sweep
            .iter()
            .copied()
            .find(|point| quality_eq(point.quality, quality))
            .unwrap_or_else(|| panic!("Trellis deployment sweep is missing q={quality:.4}"))
    }

    fn quality_eq(left: f32, right: f32) -> bool {
        (left - right).abs() <= 1e-6
    }

    fn assert_trellis_distance_graph(sweeps: &[TrellisDistanceSweep]) {
        assert_eq!(sweeps.len(), 3);
        assert_eq!(
            sweeps
                .iter()
                .map(|sweep| sweep.camera.label)
                .collect::<Vec<_>>(),
            ["near", "mid", "far"]
        );
        for sweep in sweeps {
            assert_dense_selection_sweep(sweep);
            assert_running_best_quality(sweep.camera.label, &sweep.rendered);
            assert_trellis_scale_quality(sweep);
        }

        let root_active = sweeps[0].deployment_selection[0].active_gaussians;
        assert!(
            sweeps
                .iter()
                .all(|sweep| sweep.deployment_selection[0].active_gaussians == root_active),
            "quality zero must select the same roots at every distance: {sweeps:?}"
        );
        for index in 0..sweeps[0].deployment_selection.len() {
            let near = sweeps[0].deployment_selection[index];
            let mid = sweeps[1].deployment_selection[index];
            let far = sweeps[2].deployment_selection[index];
            assert!(quality_eq(near.quality, mid.quality) && quality_eq(mid.quality, far.quality));
            assert!(
                near.active_gaussians >= mid.active_gaussians
                    && mid.active_gaussians >= far.active_gaussians,
                "Trellis active count increased with distance at q={:.4}: near={near:?}, mid={mid:?}, far={far:?}",
                near.quality
            );
        }
        let metric_root_active = sweeps[0].rendered[0].active_gaussians;
        assert!(
            sweeps
                .iter()
                .all(|sweep| sweep.rendered[0].active_gaussians == metric_root_active),
            "192px quality zero must select the same roots at every distance: {sweeps:?}"
        );
        for index in 0..sweeps[0].rendered.len() {
            let near = sweeps[0].rendered[index];
            let mid = sweeps[1].rendered[index];
            let far = sweeps[2].rendered[index];
            assert!(quality_eq(near.quality, mid.quality) && quality_eq(mid.quality, far.quality));
            assert!(
                near.active_gaussians >= mid.active_gaussians
                    && mid.active_gaussians >= far.active_gaussians,
                "Trellis 192px metric active count increased with distance at q={:.4}: near={near:?}, mid={mid:?}, far={far:?}",
                near.quality
            );
            for (sweep, metric) in sweeps.iter().zip([near, mid, far]) {
                let deployment =
                    selection_point_at_quality(&sweep.deployment_selection, metric.quality);
                assert!(
                    deployment.active_gaussians >= metric.active_gaussians,
                    "Trellis {} 1080p deployment cut is coarser than its 192px metric cut at q={:.4}: deployment={deployment:?}, metric={metric:?}",
                    sweep.camera.label,
                    metric.quality
                );
            }
        }

        for goal in UTILITY_GOALS {
            let quality = common_utility_quality(sweeps, goal).unwrap_or_else(|| {
                panic!(
                    "no common interior Trellis anchor meets {} at every distance: {sweeps:?}",
                    goal.label
                )
            });
            for sweep in sweeps {
                let anchor = point_at_quality(&sweep.rendered, quality);
                let exact = point_at_quality(&sweep.rendered, 1.0);
                assert_covariance_morphology_bound(
                    &format!(
                        "Trellis {} common {} anchor q={quality:.4}",
                        sweep.camera.label, goal.label
                    ),
                    anchor,
                    exact,
                );
            }
        }

        let summaries = sweeps
            .iter()
            .map(|sweep| continuity_summary(&sweep.deployment_selection))
            .collect::<Vec<_>>();
        for (crossing, near, mid, far) in [
            ("q20", summaries[0].q20, summaries[1].q20, summaries[2].q20),
            ("q50", summaries[0].q50, summaries[1].q50, summaries[2].q50),
            ("q80", summaries[0].q80, summaries[1].q80, summaries[2].q80),
        ] {
            assert!(
                near <= mid + 1e-6 && mid <= far + 1e-6,
                "Trellis deployment {crossing} crossings are not ordered near <= mid <= far: near={near}, mid={mid}, far={far}"
            );
        }

        let mut interior_samples = 0_usize;
        let mut strict_distance_samples = 0_usize;
        let mut near_mid_gap = 0_usize;
        let mut mid_far_gap = 0_usize;
        for index in 0..sweeps[0].deployment_selection.len() {
            let near = sweeps[0].deployment_selection[index];
            let mid = sweeps[1].deployment_selection[index];
            let far = sweeps[2].deployment_selection[index];
            if near.quality <= 0.0 || near.quality >= 1.0 {
                continue;
            }
            interior_samples += 1;
            near_mid_gap += near.active_gaussians - mid.active_gaussians;
            mid_far_gap += mid.active_gaussians - far.active_gaussians;
            strict_distance_samples += usize::from(
                near.active_gaussians > mid.active_gaussians
                    && mid.active_gaussians > far.active_gaussians,
            );
        }
        assert!(interior_samples > 0);
        assert!(
            near_mid_gap * 50 >= interior_samples * TRELLIS_SPLAT_COUNT,
            "Trellis average near-to-mid deployment separation is below 2% of source: total_gap={near_mid_gap}, samples={interior_samples}"
        );
        assert!(
            mid_far_gap * 50 >= interior_samples * TRELLIS_SPLAT_COUNT,
            "Trellis average mid-to-far deployment separation is below 2% of source: total_gap={mid_far_gap}, samples={interior_samples}"
        );
        assert!(
            strict_distance_samples * 5 >= interior_samples,
            "Trellis has strict near > mid > far selection on fewer than 20% of interior samples: strict={strict_distance_samples}, samples={interior_samples}"
        );
    }

    fn assert_dense_selection_sweep(sweep: &TrellisDistanceSweep) {
        assert_eq!(sweep.deployment_selection.len(), 201);
        for pair in sweep.deployment_selection.windows(2) {
            let [lower, higher] = pair else {
                unreachable!()
            };
            assert!(higher.quality > lower.quality);
            assert!(higher.quality - lower.quality <= 0.005 + 1e-6);
            assert!(
                higher.active_gaussians >= lower.active_gaussians,
                "Trellis {} active count regressed with quality: lower={lower:?}, higher={higher:?}",
                sweep.camera.label
            );
            if lower.quality > 0.0 && higher.quality < 1.0 {
                assert!(
                    higher.active_gaussians - lower.active_gaussians
                        <= TRELLIS_SPLAT_COUNT / 20 + TRELLIS_CONTINUITY_DISCRETE_RECORD_SLACK,
                    "Trellis {} interior .005 quality step activates more than 5% of the source plus one two-leaf domain: lower={lower:?}, higher={higher:?}",
                    sweep.camera.label
                );
            }
        }
        let summary = continuity_summary(&sweep.deployment_selection);
        assert!(
            summary.q20 <= summary.q50 + 1e-6 && summary.q50 <= summary.q80 + 1e-6,
            "Trellis {} deployment crossings are not ordered q20 <= q50 <= q80: {summary:?}",
            sweep.camera.label
        );
        // At the dense .005 selector sampling rate this requires at least 15
        // independently checked slider intervals between the 20% and 80%
        // workload crossings. The pre-retune Trellis cliff occupied roughly
        // .01 quality, so this remains a strong anti-cliff contract without
        // pretending that active-record count must be linear in quality.
        assert!(
            summary.q80 - summary.q20 >= 0.075 - 1e-6,
            "Trellis {} deployment graph traverses q20 to q80 in less than .075 quality: {summary:?}",
            sweep.camera.label
        );
        assert_eq!(
            sweep
                .deployment_selection
                .last()
                .expect("dense deployment sweep has q=1")
                .active_gaussians,
            TRELLIS_SPLAT_COUNT
        );
    }

    fn assert_running_best_quality(label: &str, sweep: &[QualitySweepPoint]) {
        assert!(!sweep.is_empty());
        for pair in sweep.windows(2) {
            let [previous, point] = pair else {
                unreachable!()
            };
            assert!(point.quality > previous.quality);
            assert!(
                point.active_gaussians >= previous.active_gaussians,
                "Trellis {label} 192px metric active count regressed with quality: previous={previous:?}, point={point:?}"
            );
        }
        let mut best_psnr = f64::NEG_INFINITY;
        let mut best_ssim: Option<f64> = None;
        let mut best_iou: Option<f64> = None;
        let mut best_alpha_mae: Option<f64> = None;
        let mut best_spill: Option<f64> = None;
        for point in sweep {
            let observation = point.observation;
            // Root-level and very coarse cuts contain only a handful of
            // representatives, so exchanging one discrete page can move a
            // local foreground mask without indicating a broken quality
            // curve. Keep that noise bounded to 1 dB; once the useful half of
            // the slider begins, enforce the tighter 0.5 dB envelope.
            let psnr_tolerance = if point.quality < 0.5 - 1e-6 {
                1.0
            } else if point.quality < 0.7 - 1e-6 {
                // Mid-quality cuts still replace coarse discrete
                // representatives. Bound that quantization noise while the
                // image remains below the useful 30+ dB region.
                1.25
            } else {
                0.5
            };
            assert!(
                observation.full.foreground_psnr_rgb + psnr_tolerance >= best_psnr,
                "Trellis {label} q={:.4} PSNR regressed below the running-best envelope: best={best_psnr}, point={point:?}",
                point.quality
            );
            best_psnr = best_psnr.max(observation.full.foreground_psnr_rgb);
            if point.quality < 0.7 - 1e-6 {
                continue;
            }
            if let Some(best) = best_ssim {
                assert!(
                    observation.foreground_roi.luminance_ssim + 0.005 >= best,
                    "Trellis {label} q={:.4} SSIM regressed below the q>=.7 running-best envelope: best={best}, point={point:?}",
                    point.quality
                );
            }
            if let Some(best) = best_iou {
                assert!(
                    observation.full.foreground_iou + 0.01 >= best,
                    "Trellis {label} q={:.4} IoU regressed below the q>=.7 running-best envelope: best={best}, point={point:?}",
                    point.quality
                );
            }
            if let Some(best) = best_alpha_mae {
                assert!(
                    observation.foreground_roi.alpha_mae <= best + 0.005,
                    "Trellis {label} q={:.4} alpha MAE regressed above the q>=.7 running-best envelope: best={best}, point={point:?}",
                    point.quality
                );
            }
            if let Some(best) = best_spill {
                assert!(
                    observation.spill_outside_dilated_reference <= best + 0.002,
                    "Trellis {label} q={:.4} spill regressed above the q>=.7 running-best envelope: best={best}, point={point:?}",
                    point.quality
                );
            }
            best_ssim = Some(
                best_ssim.map_or(observation.foreground_roi.luminance_ssim, |best| {
                    best.max(observation.foreground_roi.luminance_ssim)
                }),
            );
            best_iou = Some(best_iou.map_or(observation.full.foreground_iou, |best| {
                best.max(observation.full.foreground_iou)
            }));
            best_alpha_mae = Some(
                best_alpha_mae.map_or(observation.foreground_roi.alpha_mae, |best| {
                    best.min(observation.foreground_roi.alpha_mae)
                }),
            );
            best_spill = Some(
                best_spill.map_or(observation.spill_outside_dilated_reference, |best| {
                    best.min(observation.spill_outside_dilated_reference)
                }),
            );
        }
    }

    fn assert_trellis_scale_quality(sweep: &TrellisDistanceSweep) {
        let label = sweep.camera.label;
        let exact = point_at_quality(&sweep.rendered, 1.0);
        assert_exact_trellis_point(label, exact);
        let q95 = point_at_quality(&sweep.rendered, 0.95);
        let q99 = point_at_quality(&sweep.rendered, 0.99);
        assert_thresholds(
            &format!("Trellis {label} q=.95"),
            q95.observation,
            Q95_THRESHOLDS,
        );
        assert_thresholds(
            &format!("Trellis {label} q=.99"),
            q99.observation,
            Q99_THRESHOLDS,
        );
        assert_covariance_major_sigma_bound(&format!("Trellis {label} q=.95"), q95, exact);
        assert_covariance_major_sigma_bound(&format!("Trellis {label} q=.99"), q99, exact);

        for goal in UTILITY_GOALS {
            assert!(
                utility_anchor(&sweep.rendered, goal).is_some(),
                "Trellis {label} has no interior cut meeting {} across the full slider: {:?}",
                goal.label,
                sweep.rendered
            );
        }
    }

    fn assert_trellis_authored_safety(sweep: &[QualitySweepPoint]) {
        assert_trellis_morphology_safety("authored", sweep);
        let q95 = point_at_quality(sweep, 0.95);
        let q99 = point_at_quality(sweep, 0.99);
        assert_thresholds("Trellis authored q=.95", q95.observation, Q95_THRESHOLDS);
        assert_thresholds("Trellis authored q=.99", q99.observation, Q99_THRESHOLDS);
        assert_monotonic(
            "Trellis authored q=.95 -> q=.99",
            q95.observation,
            q99.observation,
        );
    }

    fn assert_trellis_morphology_safety(label: &str, sweep: &[QualitySweepPoint]) {
        assert_eq!(
            sweep.len(),
            TRELLIS_MORPHOLOGY_QUALITIES.len(),
            "Trellis {label} morphology sweep length drifted"
        );
        for (point, expected_quality) in sweep.iter().zip(TRELLIS_MORPHOLOGY_QUALITIES) {
            assert!(
                quality_eq(point.quality, expected_quality),
                "Trellis {label} morphology sweep quality drifted: expected={expected_quality}, point={point:?}"
            );
        }
        for pair in sweep.windows(2) {
            let [lower, higher] = pair else {
                unreachable!()
            };
            assert!(
                higher.active_gaussians >= lower.active_gaussians,
                "Trellis {label} morphology active count regressed with quality: lower={lower:?}, higher={higher:?}"
            );
        }

        let exact = point_at_quality(sweep, 1.0);
        assert_exact_trellis_point(label, exact);
        for point in sweep.iter().filter(|point| point.quality < 1.0) {
            let point_label = format!("Trellis {label} q={:.2}", point.quality);
            assert_covariance_morphology_bound(&point_label, point, exact);
            assert!(
                point.covariance.visible_elongated_splats
                    <= exact.covariance.visible_elongated_splats,
                "{point_label} adds opacity-visible elongated splats beyond the same-view q=1 baseline (major sigma >{VISIBLE_ELONGATION_MIN_MAJOR_SIGMA_PX}px and aspect >{VISIBLE_ELONGATION_MIN_ASPECT_RATIO}:1): candidate={}, exact={}, point={point:?}",
                point.covariance.visible_elongated_splats,
                exact.covariance.visible_elongated_splats,
            );
            assert_alpha_morphology_bound(&point_label, point.observation);
        }
    }

    fn assert_covariance_major_sigma_bound(
        label: &str,
        candidate: &QualitySweepPoint,
        exact: &QualitySweepPoint,
    ) {
        let candidate_sigma = candidate.covariance.maximum_major_sigma_px;
        let exact_sigma = exact.covariance.maximum_major_sigma_px;
        assert!(candidate_sigma.is_finite() && exact_sigma.is_finite());
        let conservative_limit = exact_sigma * 1.25 + 1.0;
        assert!(
            candidate_sigma <= conservative_limit,
            "{label} maximum projected major sigma exceeds 1.25x exact plus one pixel: candidate={candidate_sigma}, exact={exact_sigma}, limit={conservative_limit}"
        );
    }

    fn assert_covariance_morphology_bound(
        label: &str,
        candidate: &QualitySweepPoint,
        exact: &QualitySweepPoint,
    ) {
        assert_covariance_major_sigma_bound(label, candidate, exact);
        assert!(
            candidate.covariance.projected_splats > 0 && exact.covariance.projected_splats > 0,
            "{label} has no valid projected covariance samples: candidate={candidate:?}, exact={exact:?}"
        );
        let candidate_aspect = candidate.covariance.maximum_aspect_ratio;
        let exact_aspect = exact.covariance.maximum_aspect_ratio;
        assert!(candidate_aspect.is_finite() && exact_aspect.is_finite());
        let conservative_limit = (exact_aspect * 2.0).max(8.0);
        assert!(
            candidate_aspect <= conservative_limit,
            "{label} maximum projected aspect exceeds max(8, 2x exact): candidate={candidate_aspect}, exact={exact_aspect}, limit={conservative_limit}"
        );
    }

    fn assert_exact_trellis_point(label: &str, point: &QualitySweepPoint) {
        assert_eq!(point.active_gaussians, TRELLIS_SPLAT_COUNT);
        assert!(
            point.observation.full.psnr_rgb.is_infinite()
                && point.observation.full.foreground_psnr_rgb.is_infinite()
                && point.observation.full.max_abs_error == 0.0
                && point.observation.foreground_roi.alpha_mae == 0.0
                && point.observation.spill_outside_dilated_reference == 0.0
                && point
                    .observation
                    .foreground_spill_pixels_outside_dilated_reference
                    == 0
                && point.observation.maximum_foreground_distance_px == 0,
            "Trellis {label} quality one did not restore the exact reference: {point:?}"
        );
    }

    fn foreground_union_roi(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        width: u32,
        height: u32,
        margin: usize,
    ) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
        assert_eq!(reference.len(), (width * height) as usize);
        assert_eq!(candidate.len(), reference.len());
        let width = width as usize;
        let height = height as usize;
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0;
        let mut max_y = 0;
        let mut found = false;
        for (index, (expected, actual)) in reference.iter().zip(candidate).enumerate() {
            if expected[3] <= FOREGROUND_ALPHA && actual[3] <= FOREGROUND_ALPHA {
                continue;
            }
            let x = index % width;
            let y = index / width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            found = true;
        }
        assert!(found, "quality comparison has no foreground");
        min_x = min_x.saturating_sub(margin);
        min_y = min_y.saturating_sub(margin);
        max_x = (max_x + margin).min(width - 1);
        max_y = (max_y + margin).min(height - 1);

        let mut reference_roi = Vec::with_capacity((max_x - min_x + 1) * (max_y - min_y + 1));
        let mut candidate_roi = Vec::with_capacity(reference_roi.capacity());
        for y in min_y..=max_y {
            let start = y * width + min_x;
            let end = y * width + max_x + 1;
            reference_roi.extend_from_slice(&reference[start..end]);
            candidate_roi.extend_from_slice(&candidate[start..end]);
        }
        (reference_roi, candidate_roi)
    }

    #[derive(Clone, Copy, Debug)]
    struct AlphaSpillDiagnostics {
        alpha_mass_ratio: f64,
        foreground_pixels_outside_dilated_reference: usize,
        maximum_foreground_distance_px: usize,
    }

    fn alpha_spill_diagnostics(
        reference: &[[f32; 4]],
        candidate: &[[f32; 4]],
        width: u32,
        height: u32,
        dilation: usize,
    ) -> AlphaSpillDiagnostics {
        let width = width as usize;
        let height = height as usize;
        assert_eq!(reference.len(), width * height);
        assert_eq!(candidate.len(), reference.len());
        let unreachable = width.max(height) + 1;
        let mut foreground_distance = vec![unreachable; reference.len()];
        for (index, pixel) in reference.iter().enumerate() {
            if pixel[3] >= FOREGROUND_ALPHA {
                foreground_distance[index] = 0;
            }
        }
        assert!(
            foreground_distance.contains(&0),
            "alpha spill reference has no foreground"
        );

        // Exact two-pass chessboard-distance transform. Its <= dilation mask
        // is identical to the square dilation used by the spill contract, and
        // it also reports how far an opacity-visible candidate reaches beyond
        // the exact q=1 geometry.
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                let mut best = foreground_distance[index];
                if x > 0 {
                    best = best.min(foreground_distance[index - 1].saturating_add(1));
                }
                if y > 0 {
                    for neighbor_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                        best = best.min(
                            foreground_distance[(y - 1) * width + neighbor_x].saturating_add(1),
                        );
                    }
                }
                foreground_distance[index] = best;
            }
        }
        for y in (0..height).rev() {
            for x in (0..width).rev() {
                let index = y * width + x;
                let mut best = foreground_distance[index];
                if x + 1 < width {
                    best = best.min(foreground_distance[index + 1].saturating_add(1));
                }
                if y + 1 < height {
                    for neighbor_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                        best = best.min(
                            foreground_distance[(y + 1) * width + neighbor_x].saturating_add(1),
                        );
                    }
                }
                foreground_distance[index] = best;
            }
        }

        let reference_alpha = reference
            .iter()
            .map(|pixel| f64::from(pixel[3]))
            .sum::<f64>();
        assert!(reference_alpha > 0.0);
        let alpha_mass_ratio = candidate
            .iter()
            .zip(&foreground_distance)
            .filter(|(_, distance)| **distance > dilation)
            .map(|(pixel, _)| f64::from(pixel[3]))
            .sum::<f64>()
            / reference_alpha;
        let foreground_pixels_outside_dilated_reference = candidate
            .iter()
            .zip(&foreground_distance)
            .filter(|(pixel, distance)| pixel[3] >= FOREGROUND_ALPHA && **distance > dilation)
            .count();
        let maximum_foreground_distance_px = candidate
            .iter()
            .zip(foreground_distance)
            .filter_map(|(pixel, distance)| (pixel[3] >= FOREGROUND_ALPHA).then_some(distance))
            .max()
            .unwrap_or(0);
        AlphaSpillDiagnostics {
            alpha_mass_ratio,
            foreground_pixels_outside_dilated_reference,
            maximum_foreground_distance_px,
        }
    }

    #[test]
    fn alpha_spill_diagnostics_tracks_direct_foreground_distance() {
        let mut reference = vec![[0.0; 4]; 49];
        let mut candidate = reference.clone();
        reference[3 * 7 + 3][3] = 1.0;
        candidate[3 * 7 + 3][3] = 1.0;
        candidate[3 * 7 + 5][3] = FOREGROUND_ALPHA;
        candidate[3 * 7 + 6][3] = FOREGROUND_ALPHA;

        let diagnostics = alpha_spill_diagnostics(&reference, &candidate, 7, 7, 2);
        assert_eq!(diagnostics.foreground_pixels_outside_dilated_reference, 1);
        assert_eq!(diagnostics.maximum_foreground_distance_px, 3);
        assert!((diagnostics.alpha_mass_ratio - f64::from(FOREGROUND_ALPHA)).abs() <= 1e-9);
    }

    fn linear_rgba(bytes: &[u8]) -> Vec<[f32; 4]> {
        bytes
            .chunks_exact(4)
            .map(|pixel| {
                [
                    srgb_to_linear(pixel[0]),
                    srgb_to_linear(pixel[1]),
                    srgb_to_linear(pixel[2]),
                    f32::from(pixel[3]) / 255.0,
                ]
            })
            .collect()
    }

    fn srgb_to_linear(value: u8) -> f32 {
        srgb_display_channel_to_linear(f32::from(value) / 255.0)
    }
}
