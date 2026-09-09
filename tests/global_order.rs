//! One bounded GPU image contract: interleaved clouds equal one merged per-cloud stream.
#![cfg(all(
    feature = "headless",
    feature = "testing",
    lod_render_path,
    not(target_arch = "wasm32")
))]

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    asset::AssetId,
    camera::{RenderTarget, ScalingMode, SubCameraView, visibility::NoFrustumCulling},
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{CachedPipelineState, PipelineCache, TextureFormat},
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    shader::ShaderCacheError,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_gaussian_splatting::{
    CloudSettings, Gaussian3d, GaussianCamera, GaussianLodBridgeConfig, GaussianLodBuildSettings,
    GaussianLodHandle, GaussianLodSettings, GaussianSplattingPlugin, LodPresentationMode,
    PlanarGaussian3d, PlanarGaussian3dHandle, RadixSortDepthBits, SphericalHarmonicCoefficients,
    gaussian::{
        f32::Rotation,
        formats::{planar_3d_chunked::LodPageStorage, planar_3d_lod::build_planar_3d_lod},
        settings::GaussianColorSpace,
    },
    io::lod::{GaussianLodAsset, encode_page},
    render::{
        ordered::{
            GaussianGlobalOrderDiagnostics, GaussianGlobalOrderFrame, GaussianGlobalOrderSettings,
            GlobalOrderDispatchTestLimit,
        },
        spatial_morph::GaussianLodSpatialTransitionSettings,
        traversal::{
            GpuLodDrawAcknowledgements, GpuLodDrawRenderer, GpuLodHierarchy,
            GpuLodTraversalSettings,
        },
    },
    sort::SortMode,
    stream::{
        package::{
            GaussianGpuLodPackage, GaussianGpuLodPackageStatus, GaussianLodPackageConfig,
            GaussianLodPackagePhase, GaussianLodPackageSource, GaussianLodPackageStatus,
        },
        render_commit::LodRenderCandidates,
    },
    testing::upgrade_manifest_to_synthetic_abi16_lifecycle_fixture,
};
#[path = "support/global_order_spatial_motion.rs"]
mod spatial_motion;

use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Resource)]
struct TempPackage(PathBuf);
impl Drop for TempPackage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn interleaved_clouds_match_merged_cloud_order_and_admission_never_duplicates_draws() {
    if std::env::var("RUN_GPU_RENDER_TESTS").as_deref() != Ok("1") {
        return;
    }
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GaussianLodBridgeConfig {
            auto_build_flat_clouds: false,
            ..default()
        })
        .insert_resource(GaussianLodPackageConfig {
            max_atlas_gaussians: 64,
            max_atlas_bytes: 4 * 1024 * 1024,
            ..default()
        })
        .insert_resource(State::default())
        .init_resource::<spatial_motion::Probe>()
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    meta_check: bevy::asset::AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>()
                .disable::<PipelinedRenderingPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 120.0,
        )))
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, capture)
        .add_observer(captured);
    let probe = app.world().resource::<spatial_motion::Probe>().clone();
    app.sub_app_mut(RenderApp)
        .insert_resource(probe)
        .insert_resource(GlobalOrderDispatchTestLimit(2))
        .add_systems(
            Render,
            spatial_motion::readback
                .after(RenderSystems::Render)
                .before(RenderSystems::Cleanup),
        )
        .add_systems(Render, fail_shader_errors.in_set(RenderSystems::Cleanup));
    assert_eq!(app.run(), AppExit::Success);
}

fn fail_shader_errors(cache: Res<PipelineCache>) {
    for pipeline in cache.pipelines() {
        if let CachedPipelineState::Err(error) = &pipeline.state {
            match error {
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable => {}
                _ => panic!("global-order shader failed: {error}"),
            }
        }
    }
}

#[derive(Component)]
struct Source(u8);

#[derive(Resource)]
struct State {
    started: Instant,
    phase: u8,
    frames: u32,
    requested: bool,
    camera: Option<Entity>,
    target: Option<Handle<Image>>,
    baseline: Vec<u8>,
    lod_cloud: Option<Entity>,
    active_moving_submissions: u32,
    last_submission: u64,
    prepared_source: Option<AssetId<PlanarGaussian3d>>,
    source_verified: bool,
    coarse_count: u32,
    gpu_clouds: Vec<Entity>,
    coarse_generations: Vec<u64>,
    spatial_stage: u8,
    spatial_receipt: Option<GaussianGlobalOrderFrame>,
    spatial_cut: Vec<(u64, u64, u32)>,
    spatial_artifacts: Option<PathBuf>,
    perspective: Option<spatial_motion::Motion>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            phase: 0,
            frames: 0,
            requested: false,
            camera: None,
            target: None,
            baseline: Vec::new(),
            lod_cloud: None,
            active_moving_submissions: 0,
            last_submission: 0,
            prepared_source: None,
            source_verified: false,
            coarse_count: 0,
            gpu_clouds: Vec::new(),
            coarse_generations: Vec::new(),
            spatial_stage: 0,
            spatial_receipt: None,
            spatial_cut: Vec::new(),
            spatial_artifacts: None,
            perspective: None,
        }
    }
}

fn record(z: f32, color: [f32; 3], opacity: f32) -> Gaussian3d {
    let mut sh = SphericalHarmonicCoefficients::default();
    for (channel, value) in color.into_iter().enumerate() {
        sh.set(channel, (value - 0.5) / 0.282_094_8);
    }
    Gaussian3d {
        position_visibility: [0.0, 0.0, z, 1.0].into(),
        rotation: Rotation {
            rotation: [0.9238795, 0.0, 0.0, 0.3826834],
        },
        scale_opacity: [0.32, 0.16, 0.12, opacity].into(),
        spherical_harmonic: sh,
    }
}

fn ordering_records() -> [Gaussian3d; 2] {
    [
        (Vec3::new(-0.5, 0.0, 0.01), [0.9, 0.05, 0.05]),
        (Vec3::new(0.5, 0.0, -0.01), [0.05, 0.9, 0.05]),
    ]
    .map(|(position, color)| {
        let mut gaussian = record(position.z, color, 0.75);
        gaussian.position_visibility.position = position.to_array();
        gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
        gaussian.scale_opacity.scale = [0.8, 0.8, 0.1];
        gaussian
    })
}

fn settings() -> GaussianGlobalOrderSettings {
    GaussianGlobalOrderSettings {
        max_projected_gaussians: 1024,
        max_gpu_bytes: 8 * 1024 * 1024,
    }
}

fn setup(
    mut commands: Commands,
    mut state: ResMut<State>,
    mut assets: ResMut<Assets<PlanarGaussian3d>>,
    mut images: ResMut<Assets<Image>>,
) {
    let target = images.add(Image::new_target_texture(
        64,
        64,
        TextureFormat::Rgba8UnormSrgb,
        None,
    ));
    state.target = Some(target.clone());
    let empty = record(0.0, [0.0; 3], 0.0);
    // Three logical groups dispatch as 2x2: row one contains a visible record,
    // and the final group is padding. Exactly four records must render.
    let mut first = vec![empty; 513];
    let mut second = vec![empty; 152];
    first[0] = record(0.45, [0.8, 0.1, 0.05], 0.65);
    first[512] = record(-0.15, [0.05, 0.1, 0.8], 0.65);
    second[0] = record(0.15, [0.05, 0.8, 0.1], 0.65);
    second[130] = record(-0.45, [0.8, 0.7, 0.05], 0.65);
    let merged: Vec<_> = first.iter().chain(&second).copied().collect();
    for (index, records) in [first, second, merged].into_iter().enumerate() {
        commands.spawn((
            Source(index as u8),
            PlanarGaussian3dHandle(assets.add(PlanarGaussian3d::from(records))),
            CloudSettings {
                sort_mode: SortMode::Radix,
                radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                color_space: GaussianColorSpace::LinRec709Display,
                opacity_adaptive_radius: false,
                ..default()
            },
            Transform::default(),
            if index == 2 {
                Visibility::Hidden
            } else {
                Visibility::Visible
            },
            NoFrustumCulling,
        ));
    }
    state.camera = Some(
        commands
            .spawn((
                Camera3d::default(),
                Camera::default(),
                Projection::Orthographic(OrthographicProjection {
                    near: 0.1,
                    far: 10.0,
                    scaling_mode: ScalingMode::FixedVertical {
                        viewport_height: 2.0,
                    },
                    ..OrthographicProjection::default_3d()
                }),
                RenderTarget::Image(target.into()),
                Transform::from_xyz(0.0, 0.0, 3.0),
                Tonemapping::None,
                Msaa::Off,
                GaussianCamera::default(),
                settings(),
            ))
            .id(),
    );
}

#[allow(clippy::too_many_arguments)]
fn capture(
    mut commands: Commands,
    mut state: ResMut<State>,
    diagnostics: Res<GaussianGlobalOrderDiagnostics>,
    probe: Res<spatial_motion::Probe>,
    acknowledgements: Res<GpuLodDrawAcknowledgements>,
    gpu_clouds: Query<(
        &GaussianGpuLodPackageStatus,
        &GpuLodHierarchy,
        &PlanarGaussian3dHandle,
    )>,
    mut cameras: Query<&mut Transform, With<GaussianCamera>>,
    lod_clouds: Query<(
        &GaussianLodPackageStatus,
        &LodRenderCandidates,
        &PlanarGaussian3dHandle,
    )>,
) {
    assert!(
        state.started.elapsed() < Duration::from_secs(45),
        "global-order fixture timed out in phase {}: {:?}; package={:?}, prepared_source={:?}, probe={probe:?}",
        state.phase,
        state.camera.and_then(|camera| diagnostics.get(camera)),
        state
            .lod_cloud
            .and_then(|entity| lod_clouds.get(entity).ok())
            .map(|(status, _, _)| status),
        state.prepared_source,
    );
    state.frames += 1;
    let frame = state.camera.and_then(|camera| diagnostics.get(camera));
    let candidate_active = state
        .lod_cloud
        .and_then(|entity| lod_clouds.get(entity).ok())
        .is_some_and(|(status, candidates, source)| {
            assert!(
                status.failure.is_none(),
                "global-order CPU package failed: {status:?}"
            );
            let candidate = candidates.get(state.camera.unwrap());
            // PREPARED can advance to ACTIVE between main-world observations.
            // The public predicate includes both and preserves that guarantee.
            if candidate.is_some_and(|candidate| candidate.render_is_prepared()) {
                state.prepared_source.get_or_insert(source.0.id());
            }
            if candidate.is_some_and(|candidate| candidate.render_is_active_for_testing())
                && state
                    .prepared_source
                    .is_some_and(|prepared| prepared == source.0.id())
            {
                state.source_verified = true;
            }
            status.phase == GaussianLodPackagePhase::Active
                && status.resident_pages > 0
                && status.active_gaussians > 0
                && status.active_gaussians < 4
                && candidates
                    .get(state.camera.unwrap())
                    .is_some_and(|candidate| {
                        candidate.render_is_active_for_testing()
                            && candidate.frontier().presentation_mode()
                                == LodPresentationMode::Discrete
                    })
        });
    if state.phase == 6 {
        cameras
            .get_mut(state.camera.unwrap())
            .unwrap()
            .translation
            .x = 0.15 * (state.frames as f32 * 0.07).sin();
        if !candidate_active {
            state.active_moving_submissions = 0;
        } else if let Some(frame) = &frame
            && frame.ready
            && !frame.overflow
            && frame.source_clouds == 1
            && (1..4).contains(&frame.projected_gaussians)
            && frame.submission > state.last_submission
        {
            state.last_submission = frame.submission;
            state.active_moving_submissions += 1;
        }
    }
    if state.requested || state.frames < 40 {
        return;
    }
    let gpu_acknowledged = frame.as_ref().is_some_and(|frame| {
        !state.gpu_clouds.is_empty()
            && state.gpu_clouds.iter().enumerate().all(|(index, entity)| {
                let Ok((status, hierarchy, source)) = gpu_clouds.get(*entity) else {
                    return false;
                };
                let Some(ack) = acknowledgements.get(state.camera.unwrap(), *entity) else {
                    return false;
                };
                status.acknowledged_views == 1
                    && ack.renderer == GpuLodDrawRenderer::OrderedQuads
                    && ack.submission == frame.submission
                    && ack.residency_generation == hierarchy.0.generation
                    && ack.source == source.0.id()
                    && (state.phase != 8
                        || ack.residency_generation > state.coarse_generations[index])
            })
    });
    if state.phase == 16 {
        probe.request(
            state.perspective.as_ref().unwrap().step,
            state.camera.unwrap(),
            state.gpu_clouds[0],
        );
    }
    let ready = match state.phase {
        0 | 3 => frame.as_ref().is_some_and(|frame| {
            frame.ready
                && frame.source_clouds == 2
                && frame.projected_gaussians == 4
                && frame.projection_dispatch_rows == 2
                && frame.gather_dispatch == [2, 2, 1]
                && !frame.overflow
        }),
        1 => state.frames >= 90,
        2 => frame.as_ref().is_some_and(|frame| {
            !frame.ready
                && frame
                    .error
                    .as_deref()
                    .is_some_and(|error| error.contains("projected-record budget"))
        }),
        4 => frame.as_ref().is_some_and(|frame| {
            frame.ready && frame.source_clouds == 0 && frame.projected_gaussians == 0
        }),
        5 | 6 => {
            candidate_active
                && state.source_verified
                && (state.phase == 5 || state.active_moving_submissions >= 12)
                && frame.as_ref().is_some_and(|frame| {
                    frame.ready
                        && frame.source_clouds == 1
                        && !frame.overflow
                        && (1..4).contains(&frame.projected_gaussians)
                })
        }
        7 | 8 => {
            gpu_acknowledged
                && frame.as_ref().is_some_and(|frame| {
                    frame.ready
                        && !frame.overflow
                        && !frame.traversal_failed
                        && frame.source_clouds == 3
                        && frame.projected_gaussians
                            == if state.phase == 7 {
                                state.coarse_count * 2 + 1
                            } else {
                                9
                            }
                })
        }
        9 => frame
            .as_ref()
            .is_some_and(|frame| !frame.ready && frame.error.is_some()),
        10 | 11 => {
            gpu_acknowledged
                && state.gpu_clouds.iter().all(|entity| {
                    gpu_clouds.get(*entity).is_ok_and(|(status, _, _)| {
                        status.queued_requests == 0
                            && status.in_flight_requests == 0
                            && status.capacity_blocked_requests == 0
                            && status.pending_publication_pages == 0
                            && status.selected_gaussians == 3
                    })
                })
                && frame.as_ref().is_some_and(|frame| {
                    frame.ready
                        && !frame.overflow
                        && !frame.traversal_failed
                        && !frame.spatial_unavailable
                        && frame.spatial_transition_edges > 0
                        && frame.spatial_transition_records > 0
                        && frame.spatial_transition_flags == 0
                })
        }
        12..=15 => {
            gpu_acknowledged
                && frame.as_ref().is_some_and(|frame| {
                    frame.ready
                        && !frame.overflow
                        && !frame.traversal_failed
                        && frame.spatial_transition_edges == 0
                        && frame.projected_gaussians
                            == if state.phase <= 13 {
                                state.coarse_count * 2 + 1
                            } else {
                                9
                            }
                })
        }
        16 => {
            gpu_acknowledged
                && frame.as_ref().is_some_and(|frame| {
                    frame.ready
                        && !frame.overflow
                        && !frame.traversal_failed
                        && !frame.spatial_unavailable
                        && frame.spatial_transition_flags
                            & if state.perspective.as_ref().unwrap().step >= 8 {
                                !4
                            } else {
                                u32::MAX
                            }
                            == 0
                        && probe.sample().is_some_and(|sample| {
                            // Ordered submissions are a global renderer counter;
                            // traversal submissions are per cloud. Authenticate
                            // this held pose against its immutable generation,
                            // never compare numbers from the two sequences.
                            sample.step == state.perspective.as_ref().unwrap().step
                                && gpu_clouds.get(state.gpu_clouds[0]).is_ok_and(
                                    |(_, hierarchy, _)| {
                                        sample.residency_generation == hierarchy.0.generation
                                    },
                                )
                        })
                })
        }
        17..=19 => frame.as_ref().is_some_and(|frame| {
            frame.ready
                && !frame.overflow
                && !frame.traversal_failed
                && frame.source_clouds == 1
                && frame.projected_gaussians > 0
        }),
        20..=23 | 25 => frame.as_ref().is_some_and(|frame| {
            frame.ready
                && !frame.overflow
                && !frame.traversal_failed
                && frame.source_clouds == 2
                && frame.projected_gaussians == 2
                && frame.spatial_transition_edges == 0
        }),
        24 | 26 => state.frames >= 90,
        _ => false,
    };
    if ready {
        if let Some(frame) = frame {
            state.last_submission = frame.submission;
        }
        state.requested = true;
        commands.spawn(Screenshot::image(state.target.clone().unwrap()));
    }
}

fn write_spatial_motion_evidence(
    state: &State,
    template: &Image,
    label: &str,
    pixels: &[u8],
    receipt: &GaussianGlobalOrderFrame,
    cut: &[(u64, u64, u32)],
) {
    let directory = state.spatial_artifacts.as_ref().unwrap();
    let mut image = template.clone().try_into_dynamic().unwrap().to_rgba8();
    image.as_mut().copy_from_slice(pixels);
    image.save(directory.join(format!("{label}.png"))).unwrap();
    let evidence = serde_json::json!({
        "scope": "fractional-cut projection continuity; orthographic pan/zoom preserves score ratios",
        "receipt_scope": "latest completed draw during a 40-frame held pose, preceding screenshot readback",
        "submission": receipt.submission,
        "projected_gaussians": receipt.projected_gaussians,
        "spatial_transition_edges": receipt.spatial_transition_edges,
        "spatial_transition_records": receipt.spatial_transition_records,
        "spatial_transition_flags": receipt.spatial_transition_flags,
        "package_generation_selected_records_snapshot_pages": cut,
        "pan_world": if label == "baseline" {0.0} else if label == "tiny_step" {0.0001} else {0.005},
        "ortho_scale": if label == "baseline" {1.0} else if label == "tiny_step" {1.00001} else {1.01},
    });
    fs::write(
        directory.join(format!("{label}.json")),
        serde_json::to_vec_pretty(&evidence).unwrap(),
    )
    .unwrap();
}

#[allow(clippy::too_many_arguments)]
fn captured(
    event: On<ScreenshotCaptured>,
    mut commands: Commands,
    mut state: ResMut<State>,
    mut clouds: Query<(Entity, &Source, &mut Visibility)>,
    mut exit: MessageWriter<AppExit>,
    mut manifests: ResMut<Assets<GaussianLodAsset>>,
    mut assets: ResMut<Assets<PlanarGaussian3d>>,
    package_sources: Query<(&GaussianLodHandle, &GaussianLodPackageSource)>,
    mut quality: Query<&mut GaussianLodSettings>,
    acknowledgements: Res<GpuLodDrawAcknowledgements>,
    mut cameras: Query<(&mut Transform, &mut Projection), With<GaussianCamera>>,
    mut images: ResMut<Assets<Image>>,
    all_clouds: Query<Entity, With<PlanarGaussian3dHandle>>,
    diagnostics: Res<GaussianGlobalOrderDiagnostics>,
    gpu_status: Query<&GaussianGpuLodPackageStatus>,
    probe: Res<spatial_motion::Probe>,
) {
    let image = event
        .image
        .clone()
        .try_into_dynamic()
        .unwrap()
        .to_rgba8()
        .into_raw();
    let camera = state.camera.unwrap();
    match state.phase {
        0 => {
            assert!(
                image.chunks_exact(4).any(|pixel| pixel[0] > 50),
                "global ordered output was blank"
            );
            state.baseline = image;
            commands
                .entity(camera)
                .remove::<GaussianGlobalOrderSettings>();
            for (_, source, mut visibility) in &mut clouds {
                *visibility = if source.0 == 2 {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
        1 => {
            let maximum = image
                .iter()
                .zip(&state.baseline)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(
                maximum <= 2,
                "split global order differs from merged per-cloud order by {maximum} quantization levels"
            );
            commands.entity(camera).insert(GaussianGlobalOrderSettings {
                max_projected_gaussians: 2,
                ..settings()
            });
            for (_, source, mut visibility) in &mut clouds {
                *visibility = if source.0 < 2 {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
        2 => {
            assert!(
                image.chunks_exact(4).all(|pixel| pixel[..3] == [0, 0, 0]),
                "rejected shared admission fell back to per-cloud draws"
            );
            commands.entity(camera).insert(settings());
        }
        3 => {
            assert_eq!(
                image, state.baseline,
                "restored global order changed the completed image"
            );
            for (_, _, mut visibility) in &mut clouds {
                *visibility = Visibility::Hidden;
            }
        }
        4 => {
            assert!(
                image.chunks_exact(4).all(|pixel| pixel[..3] == [0, 0, 0]),
                "empty view reused a stale projected record"
            );
            // Start a native CPU package with no already-visible flat asset.
            // Its first candidate must complete the prepared/active handshake
            // against the same atlas installed at package instantiation.
            for (entity, _, _) in &mut clouds {
                commands.entity(entity).despawn();
            }
            let mut records = Vec::new();
            for x in [-0.1, 0.1] {
                for y in [-0.1, 0.1] {
                    let mut gaussian = record(0.0, [0.65, 0.3, 0.15], 0.7);
                    // Separate the two sibling cohorts in depth while retaining the
                    // orthographic XY footprint used by the earlier checks.
                    gaussian.position_visibility = [x, y, 10.0 * x, 1.0].into();
                    gaussian.scale_opacity = [0.08, 0.08, 0.08, 0.7].into();
                    records.push(gaussian);
                }
            }
            let source = PlanarGaussian3d::from(records);
            let mut package = build_planar_3d_lod(
                &source,
                GaussianLodBuildSettings {
                    branching_factor: 2,
                    leaf_capacity: 1,
                    ..default()
                },
            )
            .unwrap();
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let directory = TempPackage(std::env::temp_dir().join(format!(
                "bgs_global_order_package_{}_{nonce}",
                std::process::id()
            )));
            fs::create_dir_all(&directory.0).unwrap();
            for (descriptor, page) in package.manifest.pages.iter_mut().zip(&package.pages) {
                let encoded = encode_page(page).unwrap();
                let filename = format!("{}.gspage", page.id.0);
                fs::write(directory.0.join(&filename), &encoded).unwrap();
                descriptor.storage = Some(LodPageStorage {
                    uri: filename,
                    byte_range: None,
                    encoded_len: encoded.len() as u64,
                });
            }
            // Raise conservative geometric errors to prescribe finite scores:
            // root4, adjacent internal siblings1.5/1. The cap admits only the
            // higher sibling, so its score/cutoff=1.5 gives a fractional edge.
            let root = package.manifest.roots[0];
            let mut sibling = 0;
            for node in &mut package.manifest.nodes {
                if node.is_leaf() {
                    continue;
                }
                let geometric = if node.id == root {
                    4.0
                } else {
                    let value = if node.bounds.center()[2] < 0.0 {
                        1.5
                    } else {
                        1.0
                    };
                    sibling += 1;
                    value
                };
                assert!(geometric >= node.error.geometric);
                node.error.geometric = geometric;
                node.error.combined = node.error.combined.max(geometric);
            }
            assert_eq!(sibling, 2, "fixture needs two adjacent binary subtrees");
            package.manifest.quality.max_error = package.manifest.nodes.iter().fold(
                Default::default(),
                |error: bevy_gaussian_splatting::gaussian::formats::planar_3d_lod::LodError,
                 node| error.max(node.error),
            );
            package.validate().unwrap();
            package.manifest =
                upgrade_manifest_to_synthetic_abi16_lifecycle_fixture(package.manifest).unwrap();
            state.perspective = Some(spatial_motion::Motion::new(&package.manifest));
            state.coarse_count = package.manifest.quality.coarsest_gaussian_count as u32;
            let manifest = manifests.add(GaussianLodAsset::new(package.manifest).unwrap());
            let package_source = GaussianLodPackageSource::native_directory(
                directory.0.to_string_lossy().into_owned(),
            );
            commands.insert_resource(directory);
            let mut lod = GaussianLodSettings {
                quality: 0.0,
                presentation_mode: LodPresentationMode::Discrete,
                frustum_culling: false,
                ..default()
            };
            lod.budgets.max_active_gaussians = 4;
            lod.budgets.max_resident_gaussians = 64;
            lod.budgets.max_resident_pages = 16;
            lod.budgets.max_requests_per_frame = 1;
            state.lod_cloud = Some(
                commands
                    .spawn((
                        Source(3),
                        GaussianLodHandle(manifest),
                        package_source,
                        CloudSettings {
                            sort_mode: SortMode::Radix,
                            radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                            color_space: GaussianColorSpace::LinRec709Display,
                            ..default()
                        },
                        lod,
                        Transform::default(),
                        Visibility::Visible,
                        NoFrustumCulling,
                    ))
                    .id(),
            );
        }
        5 => {
            assert!(
                image.chunks_exact(4).any(|pixel| pixel[..3] != [0, 0, 0]),
                "cold CPU discrete candidate activated without a visible global-order image"
            );
            state.baseline = image;
        }
        6 => {
            assert!(state.active_moving_submissions >= 12);
            assert!(
                image.chunks_exact(4).any(|pixel| pixel[..3] != [0, 0, 0]),
                "moving CPU discrete candidate activated without a visible global-order image"
            );
            assert_ne!(
                image, state.baseline,
                "moving camera reused the stationary candidate image"
            );
            // Exercise the same native package reader through GPU selection,
            // with two hierarchy clouds and one fixed source sharing one image.
            // The hierarchy request exceeds the projection cap: admission must
            // reserve the flat record before sharing the eight remaining slots.
            let old = state.lod_cloud.take().unwrap();
            let (manifest, source) = package_sources.get(old).unwrap();
            for x in [-0.15, 0.15] {
                state.gpu_clouds.push(
                    commands
                        .spawn((
                            manifest.clone(),
                            source.clone(),
                            GaussianGpuLodPackage,
                            GaussianLodSettings {
                                quality: 0.0,
                                presentation_mode: LodPresentationMode::Discrete,
                                frustum_culling: false,
                                ..default()
                            },
                            CloudSettings {
                                sort_mode: SortMode::Radix,
                                radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                                color_space: GaussianColorSpace::LinRec709Display,
                                opacity_adaptive_radius: false,
                                ..default()
                            },
                            Transform::from_xyz(x, 0.0, 0.0),
                            Visibility::Visible,
                            NoFrustumCulling,
                        ))
                        .id(),
                );
            }
            commands.entity(old).despawn();
            commands.spawn((
                PlanarGaussian3dHandle(assets.add(PlanarGaussian3d::from(vec![record(
                    0.1,
                    [0.15, 0.65, 0.3],
                    0.6,
                )]))),
                CloudSettings {
                    sort_mode: SortMode::Radix,
                    radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                    color_space: GaussianColorSpace::LinRec709Display,
                    opacity_adaptive_radius: false,
                    ..default()
                },
                Transform::default(),
                Visibility::Visible,
                NoFrustumCulling,
            ));
            commands.entity(camera).insert(GaussianGlobalOrderSettings {
                max_projected_gaussians: 9,
                ..settings()
            });
            commands.entity(camera).insert(GpuLodTraversalSettings {
                max_selected_gaussians: 16,
                max_frontier_nodes: 16,
                max_visited_nodes: 64,
                max_page_requests: 16,
                max_gpu_bytes: 4 * 1024 * 1024,
            });
        }
        7 => {
            assert!(
                image.chunks_exact(4).any(|pixel| pixel[..3] != [0, 0, 0]),
                "coarse GPU packages acknowledged a blank ordered image"
            );
            state.coarse_generations = state
                .gpu_clouds
                .iter()
                .map(|entity| {
                    let ack = acknowledgements.get(camera, *entity).unwrap();
                    quality.get_mut(*entity).unwrap().quality = 1.0;
                    ack.residency_generation
                })
                .collect();
        }
        8 => {
            assert!(
                image.chunks_exact(4).any(|pixel| pixel[..3] != [0, 0, 0]),
                "exact GPU packages acknowledged a blank ordered image"
            );
            commands.entity(camera).insert(GaussianGlobalOrderSettings {
                max_projected_gaussians: 1,
                ..settings()
            });
        }
        9 => {
            assert!(
                image.chunks_exact(4).all(|pixel| pixel[..3] == [0, 0, 0]),
                "rejected aggregate GPU root budget exposed a partial or fallback draw"
            );
            commands.entity(camera).insert((
                GaussianGlobalOrderSettings {
                    max_projected_gaussians: 7,
                    ..settings()
                },
                GaussianLodSpatialTransitionSettings::default(),
            ));
            for entity in &state.gpu_clouds {
                let mut lod = quality.get_mut(*entity).unwrap();
                lod.quality = 0.95;
                lod.presentation_mode = LodPresentationMode::ContinuousMorph;
            }
            cameras.get_mut(camera).unwrap().0.translation.x = 0.0;
        }
        10 => {
            state.baseline = image;
            state.spatial_cut = state
                .gpu_clouds
                .iter()
                .map(|entity| {
                    let status = gpu_status.get(*entity).unwrap();
                    (
                        status.residency_generation,
                        status.selected_gaussians,
                        status.snapshot_pages,
                    )
                })
                .collect();
            let receipt = diagnostics.get(camera).unwrap();
            state.spatial_receipt = Some(receipt.clone());
            let directory = std::env::temp_dir().join(format!(
                "bgs_spatial_motion_{}_{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(&directory).unwrap();
            state.spatial_artifacts = Some(directory);
            write_spatial_motion_evidence(
                &state,
                &event.image,
                "baseline",
                &state.baseline,
                &receipt,
                &state.spatial_cut,
            );
            let (mut transform, mut projection) = cameras.get_mut(camera).unwrap();
            // At 32px/world this moves centers 0.0032px. The 0.001% zoom adds
            // <0.00032px across this viewport. Quantization can mask the step;
            // a multi-level jump would expose a representation discontinuity.
            transform.translation.x = 0.0001;
            let Projection::Orthographic(projection) = &mut *projection else {
                unreachable!()
            };
            projection.scale = 1.00001;
        }
        11 => {
            let maximum = image
                .iter()
                .zip(&state.baseline)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            let cut = state
                .gpu_clouds
                .iter()
                .map(|entity| {
                    let status = gpu_status.get(*entity).unwrap();
                    (
                        status.residency_generation,
                        status.selected_gaussians,
                        status.snapshot_pages,
                    )
                })
                .collect::<Vec<_>>();
            let receipt = diagnostics.get(camera).unwrap();
            let baseline = state.spatial_receipt.as_ref().unwrap();
            let label = if state.spatial_stage == 0 {
                "tiny_step"
            } else {
                "larger_move_diagnostic"
            };
            write_spatial_motion_evidence(&state, &event.image, label, &image, &receipt, &cut);
            assert_eq!(
                cut, state.spatial_cut,
                "spatial motion changed settled residency/cut counts; artifacts={:?}",
                state.spatial_artifacts
            );
            assert_eq!(
                (
                    receipt.projected_gaussians,
                    receipt.spatial_transition_edges,
                    receipt.spatial_transition_records
                ),
                (
                    baseline.projected_gaussians,
                    baseline.spatial_transition_edges,
                    baseline.spatial_transition_records
                ),
                "spatial motion changed the fractional cut; artifacts={:?}",
                state.spatial_artifacts
            );
            if state.spatial_stage == 0 {
                assert!(
                    maximum <= 4,
                    "infinitesimal fractional projection step changed by {maximum} levels; artifacts={:?}",
                    state.spatial_artifacts
                );
                state.spatial_stage = 1;
                let (mut transform, mut projection) = cameras.get_mut(camera).unwrap();
                transform.translation.x = 0.005;
                let Projection::Orthographic(projection) = &mut *projection else {
                    unreachable!()
                };
                projection.scale = 1.01;
                state.frames = 0;
                state.requested = false;
                return;
            }
            // Larger pan/zoom proves input changes this fractional image, but
            // its max-pixel delta is diagnostic, not a temporal-quality gate.
            assert!(
                maximum > 0,
                "larger motion reused the held fractional image"
            );
            eprintln!(
                "spatial larger-move diagnostic max={maximum}; artifacts={:?}",
                state.spatial_artifacts
            );
            let (mut transform, mut projection) = cameras.get_mut(camera).unwrap();
            transform.translation.x = 0.0;
            let Projection::Orthographic(projection) = &mut *projection else {
                unreachable!()
            };
            projection.scale = 1.0;
            for entity in &state.gpu_clouds {
                quality.get_mut(*entity).unwrap().quality = 0.0;
            }
        }
        12 | 14 => {
            state.baseline = image;
            commands
                .entity(camera)
                .remove::<GaussianLodSpatialTransitionSettings>();
        }
        13 => {
            assert_eq!(
                image, state.baseline,
                "spatial coarse endpoint differs from actual discrete parents"
            );
            for entity in &state.gpu_clouds {
                quality.get_mut(*entity).unwrap().quality = 1.0;
            }
            commands.entity(camera).insert((
                GaussianLodSpatialTransitionSettings::default(),
                GaussianGlobalOrderSettings {
                    max_projected_gaussians: 9,
                    ..settings()
                },
            ));
        }
        15 => {
            assert_eq!(
                image, state.baseline,
                "spatial Original endpoint differs from actual discrete leaves"
            );
            // Keep one fully resident package for eight perspective and four
            // near-plane observations; no I/O drives these images.
            let cloud = state.gpu_clouds[0];
            for entity in &all_clouds {
                if entity != cloud {
                    commands.entity(entity).despawn();
                }
            }
            state.gpu_clouds.truncate(1);
            commands.entity(cloud).insert(Transform::default());
            quality.get_mut(cloud).unwrap().quality = 0.95;
            commands.entity(camera).insert((
                GaussianGlobalOrderSettings {
                    max_projected_gaussians: 3,
                    ..settings()
                },
                GaussianLodSpatialTransitionSettings::default(),
                Projection::Perspective(PerspectiveProjection {
                    fov: spatial_motion::FOV,
                    near: 0.1,
                    far: 50.0,
                    ..default()
                }),
                Transform::from_xyz(0.0, 0.0, state.perspective.as_ref().unwrap().depths[0]),
            ));
        }
        16 => {
            let sample = probe.sample().unwrap();
            let receipt = diagnostics.get(camera).unwrap();
            let directory = state.spatial_artifacts.as_ref().unwrap().clone();
            let motion = state.perspective.as_mut().unwrap();
            assert_eq!(sample.step, motion.step);
            assert_eq!(sample.traversal_flags & !2, 0, "{sample:?}");
            assert_eq!(
                sample.tail[5] & if motion.step >= 8 { !4 } else { u32::MAX },
                0,
                "{sample:?}"
            );
            assert_eq!(
                (sample.tail[0], sample.tail[7]),
                (sample.tail[3], sample.tail[6])
            );
            assert_eq!(receipt.spatial_transition_edges, sample.tail[3]);
            assert_eq!(receipt.spatial_transition_records, sample.tail[6]);
            assert!(
                sample
                    .weights
                    .iter()
                    .all(|t| t.is_finite() && *t > 0.0 && *t < 1.0)
            );
            let weight = sample.weights.first().copied().unwrap_or(0.0);
            assert!(sample.weights.iter().all(|t| *t == weight));
            let maximum =
                |a: &[u8], b: &[u8]| a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
            let mut saved = event.image.clone().try_into_dynamic().unwrap().to_rgba8();
            saved.as_mut().copy_from_slice(&image);
            saved
                .save(directory.join(format!("perspective-{}.png", motion.step)))
                .unwrap();
            fs::write(directory.join(format!("perspective-{}.json", motion.step)),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "scope": "existing descriptor tail and traversal header copied from the same producer submission; screenshot at unchanged held camera",
                    "camera_z": motion.depths[motion.step], "capacity": motion.capacity(),
                    "projection": if motion.step >= 8 { "orthographic_near_clearance" } else { "perspective" },
                    "traversal_submission": sample.submission, "draw_receipt_submission": receipt.submission,
                    "residency_generation": sample.residency_generation,
                    "selected": sample.selected, "traversal_flags": sample.traversal_flags,
                    "tail": sample.tail, "weights": sample.weights,
                    "stable_identity_physical_entries": sample.entries,
                })).unwrap()).unwrap();
            match motion.step {
                0 => {
                    assert!((weight - 0.5).abs() < 0.0001, "{sample:?}");
                    assert_eq!(sample.tail[3], 1);
                    motion.baseline = image.clone();
                }
                1 => {
                    assert!(
                        (weight - motion.previous_weight).abs() > 0.000001,
                        "camera failed to change GPU weight: {sample:?}"
                    );
                    assert!(
                        (weight - motion.previous_weight).abs() < 0.001,
                        "{sample:?}"
                    );
                    // 0.0001 world-depth change moves these >1px Gaussians by
                    // <0.002px and changes t by <0.001. Four RGBA8 levels cover
                    // quantization and the finite-support edge; no large jump.
                    assert!(
                        maximum(&image, &motion.previous) <= 4,
                        "fractional perspective discontinuity; {sample:?}; {directory:?}"
                    );
                }
                2 => {
                    assert!((weight - 0.25).abs() < 0.0001, "{sample:?}");
                    assert_ne!(
                        image, motion.baseline,
                        "changed GPU weight reused the old image"
                    );
                }
                3 => assert!(
                    weight < 0.0001,
                    "camera crossover did not approach actual parents: {sample:?}"
                ),
                4 => {
                    assert_eq!(sample.selected, 2);
                    assert_eq!(sample.tail[3], 0);
                    // The approaching endpoint splits one parent's optical
                    // depth across two coincident fragments. Their two alpha
                    // blends round through this RGBA8 sRGB attachment, whereas
                    // the exact parent is blended once. Allow two stored channel
                    // levels for that difference; this is not float-image parity.
                    assert!(
                        maximum(&image, &motion.previous) <= 2,
                        "t=0 crossover differs from actual parent cut; {sample:?}; {directory:?}"
                    );
                }
                5 => {
                    assert_eq!(sample.selected, 3);
                    assert_eq!(
                        sample.tail[3], 0,
                        "camera has passed the exact child endpoint: {sample:?}"
                    );
                }
                6 => {
                    assert!(weight > 0.999 && weight < 1.0, "{sample:?}");
                    assert_eq!(sample.tail[3], 1);
                    // The shared C1 tail reaches zero inside the existing quad;
                    // even dim boundary pixels must obey the same four-level
                    // endpoint gate as the rest of this tiny camera step.
                    let bounded = maximum(&image, &motion.previous) <= 4;
                    assert!(
                        bounded,
                        "camera-driven t=1 handoff jumps; {sample:?}; {directory:?}"
                    );
                }
                7 => {
                    assert!((weight - 0.5).abs() < 0.0001, "{sample:?}");
                    assert_eq!(
                        image, motion.baseline,
                        "returned perspective pose retained history"
                    );
                }
                8 => {
                    assert_eq!(sample.selected, 3);
                    assert_eq!(sample.tail[3], 1);
                    assert!((weight - 0.5).abs() < 0.0001, "{sample:?}");
                    motion.near_cut = Some((sample.residency_generation, sample.entries.clone()));
                    motion.baseline = image.clone();
                }
                9 => {
                    // The old hard envelope test still used t=.5 here, then
                    // jumped to exact children in the next observation. The
                    // current-camera near remap must already approach t=1.
                    assert!(weight > 0.999 && weight < 1.0, "{sample:?}");
                    assert_eq!(sample.tail[3], 1);
                }
                10 => {
                    assert_eq!(sample.tail[3], 0);
                    assert_eq!(sample.tail[5], 4, "actual child near endpoint");
                    assert!(
                        maximum(&image, &motion.previous) <= 2,
                        "stable-cut near-plane handoff jumps; {sample:?}; {directory:?}"
                    );
                }
                11 => {
                    assert!((weight - 0.5).abs() < 0.0001, "{sample:?}");
                    assert_eq!(
                        image, motion.baseline,
                        "returned near-clearance pose retained history"
                    );
                }
                _ => unreachable!(),
            }
            if motion.step >= 9 {
                let (generation, entries) = motion.near_cut.as_ref().unwrap();
                assert_eq!(sample.selected, 3);
                assert_eq!(sample.residency_generation, *generation);
                assert_eq!(
                    &sample.entries, entries,
                    "near motion changed the physical source cut"
                );
            }
            motion.previous = image;
            motion.previous_weight = weight;
            probe.clear();
            if motion.step + 1 < motion.depths.len() {
                motion.step += 1;
                let depth = motion.depths[motion.step];
                let capacity = motion.capacity();
                cameras.get_mut(camera).unwrap().0.translation.z = depth;
                commands.entity(camera).insert(GaussianGlobalOrderSettings {
                    max_projected_gaussians: capacity,
                    ..settings()
                });
                if motion.step == 8 {
                    commands.entity(camera).insert(Projection::Orthographic(
                        OrthographicProjection {
                            near: 0.1,
                            far: 50.0,
                            scaling_mode: ScalingMode::FixedVertical {
                                viewport_height: 2.0,
                            },
                            ..OrthographicProjection::default_3d()
                        },
                    ));
                }
                state.frames = 0;
                state.requested = false;
                return;
            }
            for entity in &all_clouds {
                commands.entity(entity).despawn();
            }
            commands
                .entity(camera)
                .remove::<(
                    GpuLodTraversalSettings,
                    GaussianLodSpatialTransitionSettings,
                )>()
                .insert((
                    settings(),
                    Transform::from_xyz(0.0, 0.0, 3.0),
                    Projection::Perspective(PerspectiveProjection {
                        fov: std::f32::consts::FRAC_PI_3,
                        near: 0.1,
                        far: 10.0,
                        ..default()
                    }),
                ));
            let records = [-0.4, 0.0, 0.4]
                .into_iter()
                .enumerate()
                .map(|(index, x)| {
                    let mut gaussian = record(index as f32 * 0.1, [0.75, 0.25, 0.1], 0.65);
                    gaussian.position_visibility.position[0] = x;
                    gaussian
                })
                .collect::<Vec<_>>();
            commands.spawn((
                PlanarGaussian3dHandle(assets.add(PlanarGaussian3d::from(records))),
                CloudSettings {
                    sort_mode: SortMode::Radix,
                    radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                    color_space: GaussianColorSpace::LinRec709Display,
                    opacity_adaptive_radius: false,
                    ..default()
                },
                Transform::default(),
                Visibility::Visible,
                NoFrustumCulling,
            ));
        }
        17 => {
            assert_eq!(image.len(), 64 * 64 * 4);
            for x in [31, 32] {
                assert!(
                    (0..64).any(|y| image[(y * 64 + x) * 4] > 50),
                    "fixture must draw across the tile boundary"
                );
            }
            state.baseline = image;
            let target = images.add(Image::new_target_texture(
                32,
                64,
                TextureFormat::Rgba8UnormSrgb,
                None,
            ));
            state.target = Some(target.clone());
            commands.entity(camera).insert((
                RenderTarget::Image(target.into()),
                Camera {
                    sub_camera_view: Some(SubCameraView {
                        full_size: UVec2::splat(64),
                        offset: Vec2::ZERO,
                        size: UVec2::new(32, 64),
                    }),
                    ..default()
                },
            ));
        }
        18 | 19 => {
            assert_eq!(image.len(), 32 * 64 * 4);
            let origin = if state.phase == 18 { 0 } else { 32 };
            let mut maximum = 0;
            for y in 0..64 {
                for x in 0..32 {
                    for channel in 0..4 {
                        maximum = maximum.max(
                            image[(y * 32 + x) * 4 + channel]
                                .abs_diff(state.baseline[(y * 64 + origin + x) * 4 + channel]),
                        );
                    }
                }
            }
            assert!(
                maximum <= 2,
                "perspective tile x={origin} differs from full image by {maximum} levels"
            );
            if state.phase == 18 {
                commands.entity(camera).insert(Camera {
                    sub_camera_view: Some(SubCameraView {
                        full_size: UVec2::splat(64),
                        offset: Vec2::new(32.0, 0.0),
                        size: UVec2::new(32, 64),
                    }),
                    ..default()
                });
            } else {
                for entity in &all_clouds {
                    commands.entity(entity).despawn();
                }
                let target = images.add(Image::new_target_texture(
                    64,
                    64,
                    TextureFormat::Rgba8UnormSrgb,
                    None,
                ));
                state.target = Some(target.clone());
                commands.entity(camera).insert((
                    Camera::default(),
                    RenderTarget::Image(target.into()),
                    Transform::from_xyz(-0.125, 0.0, 3.0),
                    Projection::Orthographic(OrthographicProjection {
                        near: 0.1,
                        far: 10.0,
                        scaling_mode: ScalingMode::FixedVertical {
                            viewport_height: 4.0,
                        },
                        ..OrthographicProjection::default_3d()
                    }),
                ));
                // A lateral pan leaves forward depth unchanged, but reverses
                // radial distance for these overlapping unequal-depth means.
                let near = Vec3::new(-0.5, 0.0, 0.01);
                let far = Vec3::new(0.5, 0.0, -0.01);
                for camera_x in [-0.125, 0.125] {
                    let eye = Vec3::new(camera_x, 0.0, 3.0);
                    assert_eq!(
                        near.distance_squared(eye) < far.distance_squared(eye),
                        camera_x < 0.0,
                        "fixture must expose the radial-depth ordering reversal"
                    );
                }
                for (index, gaussian) in ordering_records().into_iter().enumerate() {
                    commands.spawn((
                        Source(4 + index as u8),
                        PlanarGaussian3dHandle(assets.add(PlanarGaussian3d::from(vec![gaussian]))),
                        CloudSettings {
                            sort_mode: SortMode::Radix,
                            radix_sort_depth_bits: RadixSortDepthBits::Bits32,
                            color_space: GaussianColorSpace::LinRec709Display,
                            opacity_adaptive_radius: false,
                            ..default()
                        },
                        Transform::default(),
                        Visibility::Visible,
                        NoFrustumCulling,
                    ));
                }
            }
        }
        20 => {
            assert_eq!(image.len(), 64 * 64 * 4);
            let center = &image[(32 * 64 + 32) * 4..][..3];
            assert!(
                center[0] > 50 && center[1] > 50,
                "both differently colored supports must overlap: {center:?}"
            );
            state.baseline = image;
            cameras.get_mut(camera).unwrap().0.translation.x = 0.125;
        }
        21 => {
            assert_eq!(image.len(), state.baseline.len());
            let mut maximum = 0;
            for y in 0..64 {
                for x in 0..60 {
                    for channel in 0..3 {
                        maximum = maximum.max(
                            image[(y * 64 + x) * 4 + channel]
                                .abs_diff(state.baseline[(y * 64 + x + 4) * 4 + channel]),
                        );
                    }
                }
            }
            // Moving +0.25 world units in a four-unit orthographic viewport
            // translates every Gaussian exactly four pixels left. Their depth
            // order and optical density remain identical; only RGBA8 rendering
            // roundoff (as in the tiled parity check above) gets two levels.
            assert!(
                maximum <= 2,
                "lateral pan changed constant-forward-depth compositing by {maximum} levels"
            );
            cameras.get_mut(camera).unwrap().0.translation.x = -0.125;
        }
        22 => {
            assert_eq!(
                image, state.baseline,
                "returned lateral pose changed ordering"
            );
            commands.entity(camera).insert(
                Transform::from_xyz(0.0, 0.0, 3.0).with_rotation(Quat::from_rotation_y(0.125)),
            );
        }
        23 | 25 => {
            assert_eq!(image.len(), 64 * 64 * 4);
            state.baseline = image;
            let (transform, _) = cameras.get_mut(camera).unwrap();
            assert_eq!(transform.translation, Vec3::new(0.0, 0.0, 3.0));
            let mut records = ordering_records();
            let forward = *transform.forward();
            let depth = |record: &Gaussian3d| {
                (Vec3::from_array(record.position_visibility.position) - transform.translation)
                    .dot(forward)
            };
            assert_eq!(
                depth(&records[0]) > depth(&records[1]),
                state.phase == 23,
                "pure camera rotation must reverse actual forward-depth order"
            );
            if state.phase == 25 {
                records.reverse();
            }
            assert!(depth(&records[0]) > depth(&records[1]));
            for (_, _, mut visibility) in &mut clouds {
                *visibility = Visibility::Hidden;
            }
            commands
                .entity(camera)
                .remove::<GaussianGlobalOrderSettings>();
            // Identity entries in an unsorted stream are an independent,
            // explicitly far-to-near reference for this exact rotated view.
            commands.spawn((
                Source(6),
                PlanarGaussian3dHandle(assets.add(PlanarGaussian3d::from(records.to_vec()))),
                CloudSettings {
                    sort_mode: SortMode::None,
                    color_space: GaussianColorSpace::LinRec709Display,
                    opacity_adaptive_radius: false,
                    ..default()
                },
                Transform::default(),
                Visibility::Visible,
                NoFrustumCulling,
            ));
        }
        24 | 26 => {
            let maximum = image
                .iter()
                .zip(&state.baseline)
                .map(|(actual, reference)| actual.abs_diff(*reference))
                .max()
                .unwrap();
            assert!(
                maximum <= 2,
                "rotated global order differs from explicit far-to-near reference by {maximum} levels"
            );
            if state.phase == 26 {
                exit.write(AppExit::Success);
            } else {
                for (entity, source, mut visibility) in &mut clouds {
                    if source.0 == 6 {
                        commands.entity(entity).despawn();
                    } else {
                        *visibility = Visibility::Visible;
                    }
                }
                commands.entity(camera).insert(settings());
                cameras.get_mut(camera).unwrap().0.rotation = Quat::from_rotation_y(-0.125);
            }
        }
        _ => unreachable!(),
    }
    state.phase += 1;
    state.frames = 0;
    state.requested = false;
}
