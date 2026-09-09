//! Opt-in procedural source-size scaling through the production package renderer.
//!
//! `prepare` hashes every real page with bounded scratch memory and never opens a
//! GPU. `run` regenerates requested pages over a two-worker loopback server and
//! reuses production package startup, residency, atlas, compaction, radix and draw
//! attestation. Synthetic representatives establish no image-quality claim.

mod data;
mod recovery;
mod server;

use super::*;
use crate::{
    GaussianLodManifest,
    stream::{
        bridge::GaussianLodBridgeConfig,
        memory::{LodMemoryLedger, LodMemoryLimits},
        package::{GaussianLodPackageStatus, GaussianLodPackageTestingSnapshot},
        render_commit::LodRenderCandidates,
    },
    testing::VirtualCityScene,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::Ordering,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VirtualCityConfig {
    pub source_gaussians: u64,
    pub records_per_page: u32,
    pub grid_width: u32,
    pub seed: u64,
    pub prepared: PathBuf,
    pub output: PathBuf,
    pub viewport: [u32; 2],
    pub quality: f32,
    pub max_active_gaussians: u64,
    pub max_resident_pages: u32,
    pub max_cpu_bytes: u64,
    pub max_gpu_bytes: u64,
    pub phase_frames: u32,
    pub capture_every: u32,
    pub timeout_seconds: u64,
    pub capture_images: bool,
}

impl Default for VirtualCityConfig {
    fn default() -> Self {
        Self {
            source_gaussians: 10_000_000,
            records_per_page: 4096,
            grid_width: 128,
            seed: 0x47a5_51a7_d15c_1a5e,
            prepared: "virtual-city-prepared".into(),
            output: "virtual-city-capture".into(),
            viewport: [960, 540],
            quality: 0.9,
            max_active_gaussians: 65_536,
            max_resident_pages: 64,
            max_cpu_bytes: 1024 * 1024 * 1024,
            max_gpu_bytes: 512 * 1024 * 1024,
            phase_frames: 180,
            capture_every: 8,
            timeout_seconds: 300,
            capture_images: false,
        }
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;

    #[test]
    fn virtual_payload_identity_ignores_capture_settings_but_tracks_source_inputs() {
        let config = VirtualCityConfig::default();
        let original = generator_identity(&config).unwrap();
        let mut capture = config.clone();
        capture.output = "another-capture".into();
        capture.prepared = "another-preparation".into();
        capture.quality = 1.0;
        capture.max_resident_pages = 256;
        capture.max_active_gaussians = 1_048_576;
        capture.viewport = [640, 360];
        capture.phase_frames = 1200;
        capture.capture_every = 16;
        capture.capture_images = true;
        assert_eq!(generator_identity(&capture).unwrap(), original);
        assert_ne!(
            hash_bytes(&serde_json::to_vec(&capture).unwrap()),
            hash_bytes(&serde_json::to_vec(&config).unwrap())
        );
        for source in [
            VirtualCityConfig {
                source_gaussians: config.source_gaussians - 1,
                ..config.clone()
            },
            VirtualCityConfig {
                records_per_page: config.records_per_page / 2,
                ..config.clone()
            },
            VirtualCityConfig {
                grid_width: config.grid_width / 2,
                ..config.clone()
            },
            VirtualCityConfig {
                seed: config.seed + 1,
                ..config.clone()
            },
        ] {
            assert_ne!(generator_identity(&source).unwrap(), original);
        }
    }
}

impl VirtualCityConfig {
    fn validate(&self) -> CaptureResult<()> {
        if !(1..=100_000_000).contains(&self.source_gaussians)
            || !(1..=4096).contains(&self.records_per_page)
            || !(1..=256).contains(&self.grid_width)
            || self
                .source_gaussians
                .div_ceil(u64::from(self.records_per_page.max(1)))
                > 32_768
            || self
                .viewport
                .iter()
                .any(|&value| value == 0 || value > 2048)
            || !self.quality.is_finite()
            || !(0.0..=1.0).contains(&self.quality)
            || !(4..=256).contains(&self.max_resident_pages)
            || self.max_active_gaussians == 0
            || self.max_active_gaussians > 1_048_576
            || !(1..=4 * 1024 * 1024 * 1024).contains(&self.max_cpu_bytes)
            || !(1..=4 * 1024 * 1024 * 1024).contains(&self.max_gpu_bytes)
            || !(30..=1200).contains(&self.phase_frames)
            || !(1..=120).contains(&self.capture_every)
            || !(10..=1800).contains(&self.timeout_seconds)
        {
            return Err("virtual-city configuration exceeds the bounded fixture contract".into());
        }
        Ok(())
    }

    fn settings(&self) -> GaussianLodSettings {
        let mut settings = GaussianLodSettings {
            quality: self.quality,
            presentation_mode: LodPresentationMode::Discrete,
            ..Default::default()
        };
        settings.budgets.max_active_gaussians = self.max_active_gaussians;
        settings.budgets.max_resident_pages = self.max_resident_pages;
        settings.budgets.max_resident_gaussians =
            u64::from(self.max_resident_pages) * u64::from(self.records_per_page);
        settings.budgets.max_resident_bytes = settings.budgets.max_resident_gaussians
            * std::mem::size_of::<crate::Gaussian3d>() as u64;
        settings.budgets.max_upload_bytes_per_frame =
            u64::from(self.records_per_page) * std::mem::size_of::<crate::Gaussian3d>() as u64 * 4;
        settings
    }
}

fn read_config(path: &Path) -> CaptureResult<VirtualCityConfig> {
    let mut config: VirtualCityConfig = serde_json::from_slice(&fs::read(path)?)?;
    config.validate()?;
    let base = path.parent().unwrap_or(Path::new("."));
    config.prepared = base.join(&config.prepared);
    config.output = base.join(&config.output);
    Ok(config)
}

fn generator_identity(config: &VirtualCityConfig) -> CaptureResult<String> {
    // Payload identity excludes camera, capture, and lifecycle orchestration.
    // Changes to either actual generation module still invalidate preparation.
    let mut bytes = b"bevy_gaussian_splatting:virtual_city_payload_v2:abi16\0".to_vec();
    bytes.extend(serde_json::to_vec(&(
        config.source_gaussians,
        config.records_per_page,
        config.grid_width,
        config.seed,
        crate::material::spherical_harmonics::SH_DEGREE,
    ))?);
    bytes.extend_from_slice(include_bytes!("virtual_city/data.rs"));
    bytes.extend_from_slice(include_bytes!("../lod_scenes.rs"));
    Ok(hash_bytes(&bytes))
}

/// Expensive CPU-only prehash; run separately from renderer timing.
pub fn prepare(config_path: &Path) -> CaptureResult<()> {
    let config = read_config(config_path)?;
    if let Some(parent) = config.prepared.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&config.prepared)?;
    let started = Instant::now();
    let rss_before = rss_bytes();
    let manifest = data::build(&config)?;
    let bytes = encode_manifest(&manifest)?;
    fs::write(config.prepared.join("scene.gsplatlod"), &bytes)?;
    let preparation = serde_json::json!({
        "schema_version": 1, "generator_identity": generator_identity(&config)?,
        "manifest_sha256": hash_bytes(&bytes), "source_gaussians": config.source_gaussians,
        "real_leaf_pages": data::city(&config).page_count,
        "nodes": manifest.nodes.len(), "pages": manifest.pages.len(),
        "manifest_bytes": bytes.len(), "cpu_prehash_seconds": started.elapsed().as_secs_f64(),
        "rss_before_bytes": rss_before, "rss_after_bytes": rss_bytes(),
        "rss_samples_are_not_peak": true,
        "payload_archive_bytes": 0, "prehash_parallel_pages": 1,
        "executable_sha256": hash_file(&std::env::current_exe()?)?,
        "quality_qualified": false,
    });
    fs::write(
        config.prepared.join("preparation.json"),
        serde_json::to_vec_pretty(&preparation)?,
    )?;
    eprintln!(
        "prepared {} real source records in {:.2}s; payload archive: 0 bytes",
        config.source_gaussians,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Cold,
    Stationary,
    Move,
    RapidReturn,
    Eviction,
    Unload,
    Reload,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Stationary => "stationary",
            Self::Move => "move",
            Self::RapidReturn => "rapid_return",
            Self::Eviction => "eviction",
            Self::Unload => "unload",
            Self::Reload => "reload",
        }
    }
    fn next(self) -> Option<Self> {
        match self {
            Self::Cold => Some(Self::Stationary),
            Self::Stationary => Some(Self::Move),
            Self::Move => Some(Self::RapidReturn),
            Self::RapidReturn => Some(Self::Eviction),
            Self::Eviction => Some(Self::Unload),
            Self::Unload => Some(Self::Reload),
            Self::Reload => None,
        }
    }
    fn position(self, progress: f32) -> Vec3 {
        let home = Vec3::new(8.0, 10.0, 38.0);
        match self {
            Self::Move => home + Vec3::X * (768.0 * progress),
            Self::Eviction => home + Vec3::X * (768.0 * (1.0 - progress)),
            _ => home,
        }
    }
}

#[derive(Resource)]
struct CityRun {
    config: VirtualCityConfig,
    manifest: Arc<GaussianLodManifest>,
    manifest_handle: Handle<GaussianLodAsset>,
    server: server::Server,
    cloud: Option<Entity>,
    camera: Option<Entity>,
    target: Option<Handle<Image>>,
    phase: Phase,
    phase_frame: u64,
    frame: u64,
    phase_started: Instant,
    started: Instant,
    frames: BufWriter<File>,
    captures: BufWriter<File>,
    evidence: BufWriter<File>,
    receiver: Mutex<mpsc::Receiver<CompletedCapture>>,
    pending: Vec<CompletedCapture>,
    previous_frame: Option<(Instant, Arc<Mutex<Option<f64>>>)>,
    observed: BTreeSet<String>,
    detail_recovery: recovery::DetailRecovery,
    // At most max_resident_pages entries; records prove real slot replacement.
    slots: BTreeMap<u32, (u64, u64)>,
    slot_replacements_before_unload: u64,
    observed_leaf_pages: BTreeSet<u64>,
    unload_zero_observed: bool,
    peak_rss: u64,
    failure: Option<String>,
}

/// Opens an actual GPU and executes the fixed lifecycle against a prepared,
/// authenticated procedural manifest. This function is never called by tests.
pub fn run(config_path: &Path) -> CaptureResult<()> {
    let config = read_config(config_path)?;
    let bytes = fs::read(config.prepared.join("scene.gsplatlod"))?;
    let preparation: serde_json::Value =
        serde_json::from_slice(&fs::read(config.prepared.join("preparation.json"))?)?;
    if preparation["generator_identity"].as_str() != Some(generator_identity(&config)?.as_str())
        || preparation["manifest_sha256"].as_str() != Some(hash_bytes(&bytes).as_str())
    {
        return Err(
            "prepared identity does not match this generator/configuration/manifest".into(),
        );
    }
    let manifest = Arc::new(decode_manifest(&bytes, LodCodecLimits::default())?);
    if manifest.header.source_gaussian_count != config.source_gaussians {
        return Err("source count differs from the prepared manifest".into());
    }
    if let Some(parent) = config.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&config.output)?;
    fs::write(
        config.output.join("settings.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    fs::write(
        config.output.join("preparation.json"),
        serde_json::to_vec_pretty(&preparation)?,
    )?;
    let server = server::Server::start(config.clone(), manifest.clone())?;
    let settings_sha256 = hash_bytes(&serde_json::to_vec(&config)?);
    let renderer_sha256 = hash_file(&std::env::current_exe()?)?;
    let identity = LodCaptureIdentity {
        manifest_sha256: hash_bytes(&bytes), source_sha256: generator_identity(&config)?,
        builder_revision: "procedural-city-synthetic-lifecycle-abi16-v1".into(),
        renderer_revision: std::process::Command::new("git").args(["rev-parse", "HEAD"]).output().ok()
            .filter(|out| out.status.success()).map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned()).unwrap_or_else(|| "see-executable-identity".into()),
        renderer_sha256: renderer_sha256.clone(), features: capture_features(), backend: "vulkan".into(), adapter: None, driver: None,
        camera_path_sha256: hash_bytes(b"virtual_city_v1:960?viewport-in-settings:home8,10,38:travel768x:fixed-target-offset0,-6,-24"),
        settings_sha256, instrumentation: "procedural_loopback;actual_draw_attestation;async_indirect;memory_partial;source_sha256_is_generator_identity;no_quality_or_deployment_qualification".into(),
    };
    let (sender, receiver) = mpsc::sync_channel(3);
    let request = CaptureRequest {
        current: None,
        run_id: hash_bytes(
            format!(
                "{renderer_sha256}:{}:{}",
                identity.settings_sha256, identity.manifest_sha256
            )
            .as_bytes(),
        ),
        identity,
        viewport: config.viewport,
        slots: 3,
        pipeline: LodCapturePipeline::Hierarchy,
        point_gpu: None,
        ordered_gpu: None,
        capture_images: config.capture_images,
        cut_capacity: 0,
        sender,
        stats: Default::default(),
    };
    let session = CityRun {
        frames: BufWriter::new(File::create(config.output.join("lifecycle.jsonl"))?),
        captures: BufWriter::new(File::create(config.output.join("capture.jsonl"))?),
        evidence: BufWriter::new(File::create(
            config.output.join("submission_evidence.jsonl"),
        )?),
        config: config.clone(),
        manifest,
        manifest_handle: default(),
        server,
        cloud: None,
        camera: None,
        target: None,
        phase: Phase::Cold,
        phase_frame: 0,
        frame: 0,
        phase_started: Instant::now(),
        started: Instant::now(),
        receiver: Mutex::new(receiver),
        pending: Vec::new(),
        previous_frame: None,
        observed: BTreeSet::new(),
        detail_recovery: default(),
        slots: BTreeMap::new(),
        slot_replacements_before_unload: 0,
        observed_leaf_pages: BTreeSet::new(),
        unload_zero_observed: false,
        peak_rss: 0,
        failure: None,
    };
    let mut app = App::new();
    app.insert_resource(session)
        .insert_resource(request)
        .insert_resource(LodMemoryLedger::new(LodMemoryLimits {
            max_cpu_bytes: config.max_cpu_bytes,
            max_gpu_bytes: config.max_gpu_bytes,
        }))
        .insert_resource(ClearColor(Color::linear_rgba(0.0, 0.0, 0.0, 0.0)))
        .insert_resource(GaussianLodBridgeConfig {
            auto_build_flat_clouds: false,
            ..Default::default()
        })
        .insert_resource(GaussianLodPackageConfig {
            max_atlas_gaussians: config.max_resident_pages * config.records_per_page,
            max_atlas_bytes: config.max_gpu_bytes / 2,
            max_views_per_cloud: 1,
            streaming: crate::GaussianStreamingSettings {
                max_concurrent_requests: 2,
                persistent_cache: false,
                ..Default::default()
            },
            ..Default::default()
        })
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..Default::default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                        force_fallback_adapter: false,
                        ..Default::default()
                    })),
                    ..Default::default()
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..Default::default()
                })
                .disable::<WinitPlugin>()
                .disable::<PipelinedRenderingPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO))
        .add_plugins(GaussianSplattingPlugin)
        .add_plugins(ExtractResourcePlugin::<CaptureRequest>::default())
        .add_systems(Startup, setup)
        .add_systems(First, advance)
        .add_systems(Last, collect);
    gpu::install(app.sub_app_mut(RenderApp));
    if app.run().is_success() {
        Ok(())
    } else {
        Err("virtual-city lifecycle failed; inspect status.json".into())
    }
}

fn spawn_cloud(commands: &mut Commands, session: &CityRun) -> Entity {
    commands
        .spawn((
            GaussianLodHandle(session.manifest_handle.clone()),
            GaussianLodPackageSource::url(&session.server.base_url),
            session.config.settings(),
            CloudSettings {
                sort_mode: SortMode::Radix,
                opacity_adaptive_radius: false,
                ..Default::default()
            },
            Transform::IDENTITY,
            Visibility::Visible,
        ))
        .id()
}

fn setup(
    mut commands: Commands,
    mut session: ResMut<CityRun>,
    mut request: ResMut<CaptureRequest>,
    adapter: Res<bevy::render::renderer::RenderAdapterInfo>,
    mut assets: ResMut<Assets<GaussianLodAsset>>,
    mut images: ResMut<Assets<Image>>,
) {
    request.identity.backend = format!("{:?}", adapter.backend).to_lowercase();
    request.identity.adapter = Some(adapter.name.clone());
    request.identity.driver = Some(format!("{} {}", adapter.driver, adapter.driver_info));
    if adapter.device_type == wgpu::DeviceType::Cpu {
        session.failure = Some("software adapter is not qualification".into());
    }
    session.manifest_handle =
        assets.add(GaussianLodAsset::new((*session.manifest).clone()).unwrap());
    session.cloud = Some(spawn_cloud(&mut commands, &session));
    let [width, height] = session.config.viewport;
    let mut image = Image::new_target_texture(width, height, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = images.add(image);
    let position = Phase::Cold.position(0.0);
    let camera = commands
        .spawn((
            Camera3d::default(),
            Camera::default(),
            Projection::Perspective(PerspectiveProjection {
                fov: 55_f32.to_radians(),
                near: 0.05,
                far: 160.0,
                ..Default::default()
            }),
            RenderTarget::Image(target.clone().into()),
            Transform::from_translation(position)
                .looking_at(position + Vec3::new(0.0, -6.0, -24.0), Vec3::Y),
            GaussianCamera::default(),
            Tonemapping::None,
            Msaa::Off,
        ))
        .id();
    session.camera = Some(camera);
    session.target = Some(target);
    session.started = Instant::now();
    session.phase_started = session.started;
}

fn advance(
    mut session: ResMut<CityRun>,
    mut request: ResMut<CaptureRequest>,
    mut cameras: Query<&mut Transform, With<GaussianCamera>>,
) {
    let now = Instant::now();
    if let Some((started, duration)) = session.previous_frame.take() {
        *duration.lock().unwrap() = Some(now.duration_since(started).as_secs_f64() * 1000.0);
    }
    request.current = None;
    let Some(camera) = session.camera else {
        return;
    };
    let progress = (session.phase_frame as f32 / session.config.phase_frames as f32).min(1.0);
    let position = session.phase.position(progress);
    if let Ok(mut camera) = cameras.get_mut(camera) {
        *camera = Transform::from_translation(position)
            .looking_at(position + Vec3::new(0.0, -6.0, -24.0), Vec3::Y);
    }
    session
        .server
        .phase
        .store(session.phase as u8, Ordering::Release);
    let duration = Arc::new(Mutex::new(None));
    session.previous_frame = Some((now, duration.clone()));
    if session
        .frame
        .is_multiple_of(u64::from(session.config.capture_every))
    {
        request.current = Some(FrameRequest {
            frame: session.frame,
            path_frame: session.phase_frame,
            scenario: session.phase.name().into(),
            camera,
            target: session.target.clone().unwrap(),
            started: now,
            frame_wall_ms: duration,
        });
        request.stats.lock().unwrap().requested += 1;
    }
    session.frame += 1;
    session.phase_frame += 1;
}

fn json_line(writer: &mut BufWriter<File>, value: &serde_json::Value) -> CaptureResult<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn write_completed(session: &mut CityRun, mut capture: CompletedCapture) -> CaptureResult<()> {
    session.detail_recovery.observe(
        &capture.record.scenario,
        recovery::DetailSample::from_capture(&capture),
        session.config.capture_every,
    );
    if capture.evidence["draw_command_attested"] == true
        && capture.record.counts.drawn.is_some_and(|count| count > 0)
    {
        session.observed.insert(capture.record.scenario.clone());
    }
    if session.config.capture_images {
        let name = format!("frame-{:08}.png", capture.record.stamp.frame);
        Image::new(
            Extent3d {
                width: session.config.viewport[0],
                height: session.config.viewport[1],
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            std::mem::take(&mut capture.rgba),
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        )
        .try_into_dynamic()?
        .save(session.config.output.join(&name))?;
        capture.record.image = Some(LodCaptureImage {
            stamp: capture.record.stamp.clone(),
            path: name.clone(),
            sha256: hash_file(&session.config.output.join(name))?,
            viewport: session.config.viewport,
        });
    }
    if let Some(timings) = &mut capture.record.timings {
        timings.frame_wall_ms = *capture.frame_wall_ms.lock().unwrap();
    }
    capture.record.write_jsonl(&mut session.captures)?;
    json_line(&mut session.evidence, &capture.evidence)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect(
    mut commands: Commands,
    mut session: ResMut<CityRun>,
    request: Res<CaptureRequest>,
    ledger: Res<LodMemoryLedger>,
    clouds: Query<(
        &GaussianLodPackageStatus,
        Option<&GaussianLodPackageTestingSnapshot>,
        Option<&LodRenderCandidates>,
    )>,
    mut exit: MessageWriter<AppExit>,
) {
    let received: Vec<_> = session.receiver.lock().unwrap().try_iter().collect();
    session.pending.extend(received);
    let pending = std::mem::take(&mut session.pending);
    for capture in pending {
        if capture.frame_wall_ms.lock().unwrap().is_none() {
            session.pending.push(capture);
            continue;
        }
        if let Err(error) = write_completed(&mut session, capture) {
            session.failure = Some(error.to_string());
        } else {
            // Completion belongs to the consumer that persisted both the
            // capture and its submission evidence, just as in native capture.
            // Without this acknowledgement unload can never drain submitted
            // readbacks, even after every package allocation has retired.
            request.stats.lock().unwrap().completed += 1;
        }
    }
    let memory = ledger.snapshot();
    let rss = rss_bytes();
    session.peak_rss = session.peak_rss.max(rss.unwrap_or(0));
    let mut status = serde_json::Value::Null;
    let mut ranges = Vec::new();
    if let Some(cloud) = session.cloud
        && let Ok((package, work, candidates)) = clouds.get(cloud)
    {
        status = serde_json::json!({ "phase": format!("{:?}", package.phase), "resident_pages": package.resident_pages,
            "active_gaussians": package.active_gaussians, "terminal_failures": package.terminal_failures,
            "failure": package.failure.as_ref().map(|failure| format!("{failure:?}")),
            "transport_in_flight": work.map(|work| work.runtime_transport_in_flight_requests),
            "queue": work.map(|work| work.runtime_request_queue_len),
        });
        if package.terminal_failures > 0 || package.failure.is_some() {
            session.failure = Some(format!("package failure: {package:?}"));
        }
        if let Some(candidate) =
            candidates.and_then(|candidates| candidates.get(session.camera.unwrap()))
        {
            for range in candidate.render_ranges() {
                let page = range.page.0;
                let stamp = (page, u64::from(range.slot.generation));
                if session.phase < Phase::Unload
                    && session
                        .slots
                        .insert(range.slot.index, stamp)
                        .is_some_and(|old| old != stamp && old.0 != page)
                {
                    session.slot_replacements_before_unload += 1;
                }
                if session.manifest.nodes[page as usize - 1].is_leaf() {
                    session.observed_leaf_pages.insert(page);
                }
                ranges.push(serde_json::json!({ "page": page, "node": range.node.0, "slot": range.slot.index, "slot_generation": range.slot.generation, "count": range.count }));
            }
        }
    }
    if session
        .frame
        .is_multiple_of(u64::from(session.config.capture_every))
        || (session.phase == Phase::Unload && memory.total_bytes == 0)
    {
        let event = serde_json::json!({ "frame": session.frame - 1, "phase": session.phase, "phase_frame": session.phase_frame - 1,
            "elapsed_seconds": session.started.elapsed().as_secs_f64(), "rss_bytes": rss, "memory": memory,
            "package": status, "render_ranges": ranges,
            "slot_replacements_before_unload": session.slot_replacements_before_unload,
            "observed_leaf_pages": session.observed_leaf_pages.len(),
            "server_requests": session.server.stats.lock().unwrap().requests,
        });
        if let Err(error) = json_line(&mut session.frames, &event) {
            session.failure = Some(error.to_string());
        }
    }
    let stats = request.stats.lock().unwrap();
    if !stats.mapping_errors.is_empty() {
        session.failure = Some(format!("mapping failures: {:?}", stats.mapping_errors));
    }
    let mapped_drained = session.pending.is_empty() && stats.completed >= stats.submitted;
    drop(stats);
    let target = if session.phase == Phase::RapidReturn {
        2
    } else {
        session.config.phase_frames
    };
    let phase_ready = session.phase_frame >= u64::from(target)
        && if session.phase == Phase::Unload {
            memory.total_bytes == 0 && mapped_drained
        } else if session.phase == Phase::Stationary {
            session.detail_recovery.stationary_ready()
        } else if session.phase == Phase::RapidReturn {
            session.detail_recovery.recovered()
        } else {
            session.observed.contains(session.phase.name())
        };
    let mut finished = false;
    if phase_ready {
        if session.phase == Phase::Unload {
            session.unload_zero_observed = true;
        }
        if let Some(next) = session.phase.next() {
            if next == Phase::Unload {
                if let Some(cloud) = session.cloud.take() {
                    commands.entity(cloud).despawn();
                }
                session.slots.clear();
            }
            if next == Phase::Reload {
                session.cloud = Some(spawn_cloud(&mut commands, &session));
            }
            session.phase = next;
            session.phase_frame = 0;
            session.phase_started = Instant::now();
        } else {
            finished = mapped_drained;
        }
    }
    if session.started.elapsed() > Duration::from_secs(session.config.timeout_seconds) {
        session.failure = Some(format!("bounded timeout in {}", session.phase.name()));
    }
    if finished || session.failure.is_some() {
        let counts_ok = session.observed.len() == 6
            && session.detail_recovery.recovered()
            && session.slot_replacements_before_unload > 0
            && session.observed_leaf_pages.len() > 1
            && session.unload_zero_observed;
        let server = session.server.stats.lock().unwrap();
        let stats = request.stats.lock().unwrap();
        let verified = finished
            && session.failure.is_none()
            && counts_ok
            && stats.mapping_errors.is_empty()
            && server.peak_active_handlers <= 2;
        let result = serde_json::json!({ "schema_version": 1, "execution_verified": verified, "release_qualified": false,
            "source_gaussians": session.config.source_gaussians, "real_leaf_pages": data::city(&session.config).page_count,
            "observed_draw_phases": session.observed, "observed_leaf_pages": session.observed_leaf_pages,
            "slot_replacements_before_unload": session.slot_replacements_before_unload,
            "unload_zero_ledger_observed": session.unload_zero_observed, "sampled_peak_rss_bytes": session.peak_rss,
            "elapsed_seconds": session.started.elapsed().as_secs_f64(), "failure": session.failure,
            "capture_stats": &*stats, "server": &*server,
            "scope": "procedural loopback lifecycle only; generator source identity; synthetic proxy quality unqualified; runtime timing includes on-demand page generation; CPU prehash excluded; server/manifest fixture memory visible in RSS but outside scene ledger",
            "matched_visible_work_requires_comparing_actual_capture_counts": true,
            "detail_recovery_required_consecutive_samples": recovery::REQUIRED_SAMPLES,
            "detail_recovery": session.detail_recovery,
            "detail_recovery_scope": "same-camera selected/candidate/compacted/drawn counts match ten consecutive stationary reference captures; first nonzero proxy draw is reported separately; no pixel or visible-ID quality qualification",
        });
        drop(stats);
        drop(server);
        let result = fs::write(
            session.config.output.join("status.json"),
            serde_json::to_vec_pretty(&result).unwrap(),
        );
        let flush = session
            .frames
            .flush()
            .and_then(|_| session.captures.flush())
            .and_then(|_| session.evidence.flush());
        exit.write(if verified && result.is_ok() && flush.is_ok() {
            AppExit::Success
        } else {
            AppExit::error()
        });
    }
}
