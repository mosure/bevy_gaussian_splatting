//! Opt-in view GPU timing and conservative hierarchy-cap control.
//!
//! The measured interval starts before this view's LoD compaction/traversal and
//! ends after upscaling. It excludes CPU work, shadows and other camera views.
//! Hierarchy record counts are not assumed to predict stochastic point cost.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};

use bevy::{
    camera::visibility::{VisibilitySystems, VisibleEntities},
    core_pipeline::{Core3d, upscaling::upscaling},
    prelude::*,
    render::{
        GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{ExtractedView, RenderVisibleEntities, RetainedViewEntity},
    },
};
use bevy_args::{Deserialize, Serialize};

use super::{GaussianPointSplattingSettings, gpu::PointViews, point_splatting_for_cloud};
use crate::{
    CloudSettings, GaussianCamera,
    gaussian::cloud::CloudVisibilityClass,
    render::{
        lod::LodCompactionLabel,
        traversal::{
            GpuLodHierarchy, GpuLodTraversalOutputs, GpuLodTraversalRender, GpuLodTraversalSettings,
        },
    },
    stream::memory::{LodMemoryCategory, LodMemoryLease, LodMemoryLedger},
};

const MAX_PRESSURE_SOURCES: usize = 1_024;
const HEADER_BYTES: u64 = 48; // Two timestamps and the same-frame GPS header.
const TRAVERSAL_BYTES: u64 = 64;
const RECOVERY_OBSERVATIONS: u8 = 32;
const TRIAL_OBSERVATIONS: u8 = 8;

/// Outer camera policy; leave absent to preserve an authored traversal cap.
///
/// Time-driven coarsening starts only at the configured GPS sampling floor.
/// Configure the inner GPS `target_gpu_ms` to reach that floor. Hard point or
/// traversal work overflow may reduce the record cap at any sampling count.
#[derive(Component, Clone, Debug, PartialEq, ExtractComponent, Reflect, Serialize, Deserialize)]
#[reflect(Component)]
#[serde(default)]
pub struct GaussianPointSplattingViewBudget {
    pub target_view_gpu_ms: f32,
    pub min_selected_gaussians: u32,
    pub max_selected_gaussians: u32,
}

impl Default for GaussianPointSplattingViewBudget {
    fn default() -> Self {
        Self {
            target_view_gpu_ms: 16.0,
            min_selected_gaussians: 16_384,
            max_selected_gaussians: 1_048_576,
        }
    }
}

impl GaussianPointSplattingViewBudget {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.target_view_gpu_ms.is_finite()
            || self.target_view_gpu_ms <= 0.0
            || self.min_selected_gaussians == 0
            || self.min_selected_gaussians > self.max_selected_gaussians
            || self.max_selected_gaussians > 0x0fff_ffff
        {
            return Err(
                "view GPU budget requires a positive time target and ordered nonzero record limits",
            );
        }
        Ok(())
    }
}

fn admission_bounds(
    policy: &GaussianPointSplattingViewBudget,
    projected_limit: u32,
    root_records: u32,
) -> Result<(u32, u32), &'static str> {
    let minimum = root_records.max(policy.min_selected_gaussians);
    let maximum = policy.max_selected_gaussians.min(projected_limit);
    if minimum > maximum {
        return Err(
            "visible GPU hierarchy roots or minimum policy exceed the view projected-record budget",
        );
    }
    Ok((minimum, maximum))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GaussianPointSplattingViewBudgetAction {
    #[default]
    Unchanged,
    WaitingForSamples,
    Coarsened,
    Recovered,
    RolledBack,
    AtMinimum,
}

/// Latest completed measurement and current policy result, never CPU frame time.
#[derive(Clone, Debug, Default)]
pub struct GaussianPointSplattingViewBudgetFrame {
    pub submission: u64,
    pub view_gpu_ms: Option<f32>,
    pub samples_per_pixel: u32,
    pub requested_points: u32,
    pub selected_gaussian_limit: u32,
    pub traversal_flags: u32,
    pub requested_pages: u32,
    pub target_unmet: bool,
    pub action: GaussianPointSplattingViewBudgetAction,
    pub error: Option<String>,
}

#[derive(Resource, Clone, Default)]
pub struct GaussianPointSplattingViewBudgetDiagnostics(
    Arc<Mutex<HashMap<Entity, GaussianPointSplattingViewBudgetFrame>>>,
);

impl GaussianPointSplattingViewBudgetDiagnostics {
    pub fn get(&self, camera: Entity) -> Option<GaussianPointSplattingViewBudgetFrame> {
        self.0.lock().unwrap().get(&camera).cloned()
    }

    fn error(&self, camera: Entity, error: impl Into<String>) {
        let mut frames = self.0.lock().unwrap();
        let frame = frames.entry(camera).or_default();
        frame.error = Some(error.into());
        frame.target_unmet = true;
    }
}

#[derive(Clone)]
struct Observation {
    policy: GaussianPointSplattingViewBudget,
    submission: u64,
    record_cap: u32,
    samples: u32,
    minimum_samples: u32,
    gpu_ms: f32,
    points: u32,
    point_flags: u32,
    traversal_flags: u32,
    requested_pages: u32,
}

impl Observation {
    fn hard_overflow(&self) -> bool {
        self.point_flags & 1 != 0 || self.traversal_flags & (8 | 16) != 0
    }

    fn valid_image(&self) -> bool {
        self.point_flags == 0 && self.traversal_flags & 1 == 0
    }

    fn points_per_layer(&self) -> f64 {
        f64::from(self.points) / f64::from(self.samples.max(1))
    }
}

struct Trial {
    previous_cap: u32,
    baseline_ms: f32,
    baseline_points: f64,
    total_ms: f64,
    total_points: f64,
    observations: u8,
}

struct Controller {
    policy: GaussianPointSplattingViewBudget,
    cap: u32,
    last_submission: u64,
    headroom: u8,
    cooldown: u8,
    trial: Option<Trial>,
    minimum_cap: u32,
    maximum_cap: u32,
    minimum_samples: u32,
}

impl Controller {
    fn new(policy: &GaussianPointSplattingViewBudget, cap: u32) -> Self {
        Self {
            policy: policy.clone(),
            cap,
            last_submission: 0,
            headroom: 0,
            cooldown: 0,
            trial: None,
            minimum_cap: policy.min_selected_gaussians,
            maximum_cap: policy.max_selected_gaussians,
            minimum_samples: 1,
        }
    }

    fn set_admission_bounds(&mut self, minimum: u32, maximum: u32) {
        if self.minimum_cap != minimum || self.maximum_cap != maximum {
            self.minimum_cap = minimum;
            self.maximum_cap = maximum;
            self.cap = self.cap.clamp(minimum, maximum);
            self.trial = None;
            self.headroom = 0;
        }
    }

    fn observe(&mut self, sample: &Observation) -> Option<GaussianPointSplattingViewBudgetAction> {
        use GaussianPointSplattingViewBudgetAction::*;
        if sample.submission <= self.last_submission
            || sample.record_cap != self.cap
            || sample.policy != self.policy
            || sample.minimum_samples != self.minimum_samples
            || !sample.gpu_ms.is_finite()
            || sample.gpu_ms < 0.0
        {
            return None;
        }
        self.last_submission = sample.submission;
        let over = f64::from(sample.gpu_ms) > f64::from(self.policy.target_view_gpu_ms) * 1.10;
        let under = f64::from(sample.gpu_ms) < f64::from(self.policy.target_view_gpu_ms) * 0.80;
        if self.cooldown > 0 {
            self.cooldown -= 1;
        }
        if let Some(trial) = &mut self.trial {
            if sample.traversal_flags & 1 != 0 || sample.point_flags & 1 != 0 {
                // The baseline was a complete image. A trial that loses root
                // coverage or exceeds point work must restore it immediately;
                // fewer records can mean larger, more expensive proxies.
                self.cap = trial.previous_cap;
                self.trial = None;
                self.headroom = 0;
                self.cooldown = 64;
                return Some(RolledBack);
            }
            if sample.valid_image() {
                trial.observations += 1;
                trial.total_ms += f64::from(sample.gpu_ms);
                trial.total_points += sample.points_per_layer();
                if trial.observations < TRIAL_OBSERVATIONS {
                    return Some(Unchanged);
                }
                let count = f64::from(trial.observations);
                // Complete coarser representatives can cover more pixels and
                // request more points. Undo a demonstrably worse replacement.
                if trial.total_ms / count > f64::from(trial.baseline_ms) * 1.25
                    || trial.total_points / count > trial.baseline_points * 1.25
                {
                    self.cap = trial.previous_cap;
                    self.trial = None;
                    self.headroom = 0;
                    self.cooldown = 64;
                    return Some(RolledBack);
                }
                self.trial = None;
            } else if !sample.hard_overflow() {
                return Some(Unchanged);
            } else {
                self.trial = None;
            }
        }
        if sample.hard_overflow() || over {
            self.headroom = 0;
            if !sample.hard_overflow() && sample.samples > self.minimum_samples {
                return Some(WaitingForSamples);
            }
            if self.cooldown > 0 && !sample.hard_overflow() {
                return Some(Unchanged);
            }
            let next = self
                .cap
                .saturating_sub((self.cap / 8).max(1))
                .max(self.minimum_cap);
            if next == self.cap {
                return Some(AtMinimum);
            }
            if sample.valid_image() {
                self.trial = Some(Trial {
                    previous_cap: self.cap,
                    baseline_ms: sample.gpu_ms,
                    baseline_points: sample.points_per_layer(),
                    total_ms: 0.0,
                    total_points: 0.0,
                    observations: 0,
                });
            }
            self.cap = next;
            return Some(Coarsened);
        }
        if under
            && sample.valid_image()
            && sample.requested_pages == 0
            && sample.traversal_flags == 0
            && self.cooldown == 0
        {
            self.headroom += 1;
            if self.headroom == RECOVERY_OBSERVATIONS {
                self.headroom = 0;
                let next = self
                    .cap
                    .saturating_add((self.cap / 16).max(1))
                    .min(self.maximum_cap);
                if next != self.cap {
                    self.cap = next;
                    return Some(Recovered);
                }
            }
        } else {
            self.headroom = 0;
        }
        Some(Unchanged)
    }
}

#[derive(Resource, Default)]
struct Controllers(HashMap<Entity, Controller>, u64);

#[derive(Resource, Clone, Default)]
struct Measurements {
    latest: Arc<Mutex<HashMap<Entity, Observation>>>,
    next_submission: Arc<AtomicU64>,
    recovery_epoch: Arc<AtomicU64>,
}

fn reset_measurements(
    measurements: Res<Measurements>,
    diagnostics: Res<GaussianPointSplattingViewBudgetDiagnostics>,
) {
    measurements.latest.lock().unwrap().clear();
    diagnostics.0.lock().unwrap().clear();
    measurements.recovery_epoch.fetch_add(1, Ordering::Release);
}

type BudgetCameraQuery = (
    Entity,
    &'static GaussianPointSplattingViewBudget,
    &'static mut GpuLodTraversalSettings,
    &'static GaussianPointSplattingSettings,
    Option<&'static VisibleEntities>,
);

fn update_policy(
    mut controllers: ResMut<Controllers>,
    measurements: Res<Measurements>,
    diagnostics: Res<GaussianPointSplattingViewBudgetDiagnostics>,
    mut cameras: Query<BudgetCameraQuery>,
    hierarchies: Query<(Entity, &GpuLodHierarchy, &CloudSettings)>,
) {
    let epoch = measurements.recovery_epoch.load(Ordering::Acquire);
    if controllers.1 != epoch {
        controllers.0.clear();
        controllers.1 = epoch;
    }
    controllers.0.retain(|entity, _| cameras.contains(*entity));
    let mut measurements = measurements.latest.lock().unwrap();
    measurements.retain(|entity, _| cameras.contains(*entity));
    diagnostics
        .0
        .lock()
        .unwrap()
        .retain(|entity, _| cameras.contains(*entity));
    for (camera, policy, mut traversal, points, visible) in &mut cameras {
        if let Err(error) = policy.validate() {
            diagnostics.error(camera, error);
            continue;
        }
        if let Err(error) = traversal.validate() {
            diagnostics.error(camera, error);
            continue;
        }
        let minimum = hierarchies
            .iter()
            .filter(|(entity, _, cloud)| {
                point_splatting_for_cloud(Some(points), cloud)
                    && visible.is_none_or(|visible| {
                        visible
                            .iter(std::any::TypeId::of::<CloudVisibilityClass>())
                            .any(|visible| visible == entity)
                    })
            })
            .try_fold(0u32, |sum, (_, hierarchy, _)| {
                sum.checked_add(hierarchy.0.tree.root_gaussian_count())
            });
        let Some(minimum) = minimum else {
            diagnostics.error(camera, "visible GPU hierarchy root count exceeds u32");
            continue;
        };
        let (minimum, maximum) =
            match admission_bounds(policy, points.max_projected_gaussians, minimum) {
                Ok(bounds) => bounds,
                Err(error) => {
                    diagnostics.error(camera, error);
                    continue;
                }
            };
        // Apply the irreducible root floor without waiting for a GPU image or
        // timing sample: an inadmissible cut cannot generate either receipt.
        let bounded = traversal.max_selected_gaussians.clamp(minimum, maximum);
        if traversal.max_selected_gaussians != bounded {
            traversal.max_selected_gaussians = bounded;
        }
        let controller = controllers
            .0
            .entry(camera)
            .or_insert_with(|| Controller::new(policy, bounded));
        if controller.policy != *policy
            || controller.cap != bounded
            || controller.minimum_samples != points.min_samples_per_pixel
        {
            *controller = Controller::new(policy, bounded);
            controller.minimum_samples = points.min_samples_per_pixel;
        }
        controller.set_admission_bounds(minimum, maximum);
        diagnostics
            .0
            .lock()
            .unwrap()
            .entry(camera)
            .or_default()
            .selected_gaussian_limit = bounded;
        let Some(sample) = measurements.get(&camera) else {
            continue;
        };
        let Some(action) = controller.observe(sample) else {
            continue;
        };
        if traversal.max_selected_gaussians != controller.cap {
            traversal.max_selected_gaussians = controller.cap;
        }
        let frame = GaussianPointSplattingViewBudgetFrame {
            submission: sample.submission,
            view_gpu_ms: Some(sample.gpu_ms),
            samples_per_pixel: sample.samples,
            requested_points: sample.points,
            selected_gaussian_limit: controller.cap,
            traversal_flags: sample.traversal_flags,
            requested_pages: sample.requested_pages,
            target_unmet: sample.gpu_ms > policy.target_view_gpu_ms
                || !sample.valid_image()
                || sample.hard_overflow(),
            action,
            error: None,
        };
        diagnostics.0.lock().unwrap().insert(camera, frame);
    }
}

struct TimerSlot {
    queries: wgpu::QuerySet,
    resolve: Buffer,
    staging: Buffer,
    phase: Arc<AtomicU8>, // free, submitted, mapping, mapped, failed
    sample: Option<Observation>,
    sources: usize,
}

struct TimerView {
    slots: [TimerSlot; 3],
    source_capacity: usize,
    active: Option<usize>,
    admitted: bool,
    _lease: LodMemoryLease,
}

#[derive(Resource, Default)]
struct Timers(HashMap<RetainedViewEntity, TimerView>);

fn make_buffer(device: &RenderDevice, size: u64, usage: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some("GPS view budget timing"),
        size,
        usage,
        mapped_at_creation: false,
    })
}

#[allow(clippy::type_complexity)]
fn prepare_timers(
    mut timers: ResMut<Timers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    ledger: Res<LodMemoryLedger>,
    diagnostics: Res<GaussianPointSplattingViewBudgetDiagnostics>,
    views: Query<
        (
            &ExtractedView,
            &GaussianPointSplattingViewBudget,
            &RenderVisibleEntities,
        ),
        (
            With<GaussianCamera>,
            With<GaussianPointSplattingSettings>,
            With<GpuLodTraversalSettings>,
        ),
    >,
    hierarchies: Query<(), With<GpuLodHierarchy>>,
) {
    timers.0.retain(|key, state| {
        if views
            .iter()
            .any(|(view, ..)| view.retained_view_entity == *key)
        {
            state.active = None;
            state.admitted = false;
            return true;
        }
        let lease = state._lease.clone();
        queue.on_submitted_work_done(move || drop(lease));
        false
    });
    for (view, policy, visible) in &views {
        let key = view.retained_view_entity;
        if let Err(error) = policy.validate() {
            diagnostics.error(key.main_entity.id(), error);
            continue;
        }
        if !device.features().contains(WgpuFeatures::TIMESTAMP_QUERY) {
            diagnostics.error(
                key.main_entity.id(),
                "view GPU budget needs TIMESTAMP_QUERY; CPU time is not substituted",
            );
            continue;
        }
        let sources = visible.get::<CloudVisibilityClass>().map_or(0, |visible| {
            visible
                .entities_cpu_culling
                .iter()
                .filter(|(entity, _)| hierarchies.contains(*entity))
                .count()
        });
        if sources == 0 {
            diagnostics.error(
                key.main_entity.id(),
                "view GPU budget is waiting for a GPU hierarchy source",
            );
            continue;
        }
        if sources > MAX_PRESSURE_SOURCES {
            diagnostics.error(
                key.main_entity.id(),
                "view GPU budget exceeds its 1024-source bounded pressure readback",
            );
            continue;
        }
        if let Some(state) = timers.0.get_mut(&key)
            && state.source_capacity >= sources
        {
            state.admitted = true;
            continue;
        }
        let staging_bytes = HEADER_BYTES + sources as u64 * TRAVERSAL_BYTES;
        let lease =
            match ledger.try_reserve(LodMemoryCategory::CompactionGpu, 3 * (staging_bytes + 16)) {
                Ok(lease) => lease,
                Err(error) => {
                    diagnostics.error(key.main_entity.id(), error.to_string());
                    continue;
                }
            };
        let state = TimerView {
            source_capacity: sources,
            active: None,
            admitted: true,
            _lease: lease,
            slots: std::array::from_fn(|_| TimerSlot {
                queries: device
                    .wgpu_device()
                    .create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("GPS view budget timestamps"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 2,
                    }),
                resolve: make_buffer(
                    &device,
                    16,
                    BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                ),
                staging: make_buffer(
                    &device,
                    staging_bytes,
                    BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                ),
                phase: Arc::new(AtomicU8::new(0)),
                sample: None,
                sources: 0,
            }),
        };
        state._lease.mark_gpu_materialized();
        if let Some(old) = timers.0.insert(key, state) {
            queue.on_submitted_work_done(move || drop(old));
        }
    }
}

fn start_timer(
    mut context: RenderContext,
    mut timers: ResMut<Timers>,
    measurements: Res<Measurements>,
    view: ViewQuery<(
        &ExtractedView,
        &GaussianPointSplattingViewBudget,
        &GpuLodTraversalSettings,
        &GaussianPointSplattingSettings,
    )>,
) {
    let (view, policy, traversal, points) = view.into_inner();
    let Some(timer) = timers.0.get_mut(&view.retained_view_entity) else {
        return;
    };
    if !timer.admitted {
        return;
    }
    let Some(index) = timer
        .slots
        .iter()
        .position(|slot| slot.phase.load(Ordering::Acquire) == 0)
    else {
        return;
    };
    let Ok(submission) =
        measurements
            .next_submission
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
    else {
        return;
    };
    let slot = &mut timer.slots[index];
    slot.sample = Some(Observation {
        policy: policy.clone(),
        submission: submission + 1,
        record_cap: traversal.max_selected_gaussians,
        samples: 0,
        minimum_samples: points.min_samples_per_pixel,
        gpu_ms: 0.0,
        points: 0,
        point_flags: 0,
        traversal_flags: 0,
        requested_pages: 0,
    });
    drop(
        context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("GPS view budget begin"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &slot.queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                }),
            }),
    );
    timer.active = Some(index);
}

fn finish_timer(
    mut context: RenderContext,
    mut timers: ResMut<Timers>,
    points: Res<PointViews>,
    traversals: Res<GpuLodTraversalOutputs>,
    view: ViewQuery<&ExtractedView>,
) {
    let key = view.into_inner().retained_view_entity;
    let Some(timer) = timers.0.get_mut(&key) else {
        return;
    };
    let Some(index) = timer.active.take() else {
        return;
    };
    let Some(point_frame) = points.capture(key) else {
        timer.slots[index].sample = None;
        return;
    };
    let sources: Vec<_> = points
        .traversed_inputs(key)
        .filter_map(|entity| traversals.get(key, entity))
        .collect();
    if sources.is_empty() || sources.len() > timer.source_capacity {
        timer.slots[index].sample = None;
        return;
    }
    let slot = &mut timer.slots[index];
    slot.sources = sources.len();
    slot.sample.as_mut().unwrap().samples = point_frame.samples_per_pixel;
    drop(
        context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("GPS view budget end"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &slot.queries,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: Some(1),
                }),
            }),
    );
    let encoder = context.command_encoder();
    encoder.resolve_query_set(&slot.queries, 0..2, &slot.resolve, 0);
    encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.staging, 0, 16);
    encoder.copy_buffer_to_buffer(&point_frame.feedback, 0, &slot.staging, 16, 32);
    for (index, source) in sources.iter().enumerate() {
        encoder.copy_buffer_to_buffer(
            &source.feedback,
            0,
            &slot.staging,
            HEADER_BYTES + index as u64 * TRAVERSAL_BYTES,
            TRAVERSAL_BYTES,
        );
    }
    slot.phase.store(1, Ordering::Release);
}

fn collect_timers(
    mut timers: ResMut<Timers>,
    measurements: Res<Measurements>,
    diagnostics: Res<GaussianPointSplattingViewBudgetDiagnostics>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    if timers.0.is_empty() {
        return;
    }
    let _ = device.poll(PollType::Poll);
    for (view, timer) in &mut timers.0 {
        for slot in &mut timer.slots {
            match slot.phase.load(Ordering::Acquire) {
                1 => {
                    slot.phase.store(2, Ordering::Release);
                    let phase = slot.phase.clone();
                    slot.staging
                        .slice(..)
                        .map_async(MapMode::Read, move |result| {
                            phase.store(if result.is_ok() { 3 } else { 4 }, Ordering::Release);
                        });
                }
                3 => {
                    let mut sample = slot.sample.take().unwrap();
                    let bytes = slot.staging.slice(..).get_mapped_range();
                    let timestamps: &[u64] = bytemuck::cast_slice(&bytes[..16]);
                    let elapsed = timestamps[1].checked_sub(timestamps[0]);
                    let point: &[u32] = bytemuck::cast_slice(&bytes[16..48]);
                    sample.points = point[3];
                    sample.point_flags = point[4];
                    for index in 0..slot.sources {
                        let start = HEADER_BYTES as usize + index * TRAVERSAL_BYTES as usize;
                        let header: &[u32] =
                            bytemuck::cast_slice(&bytes[start..start + TRAVERSAL_BYTES as usize]);
                        sample.traversal_flags |= header[12];
                        sample.requested_pages = sample.requested_pages.saturating_add(header[10]);
                    }
                    drop(bytes);
                    slot.staging.unmap();
                    if let Some(elapsed) = elapsed {
                        sample.gpu_ms = (elapsed as f64 * f64::from(queue.get_timestamp_period())
                            / 1_000_000.0) as f32;
                        if sample.gpu_ms.is_finite() {
                            let mut latest = measurements.latest.lock().unwrap();
                            if latest
                                .get(&view.main_entity.id())
                                .is_none_or(|old| old.submission < sample.submission)
                            {
                                latest.insert(view.main_entity.id(), sample);
                            }
                        }
                    } else {
                        diagnostics.error(
                            view.main_entity.id(),
                            "view GPU timestamp interval was invalid",
                        );
                    }
                    slot.phase.store(0, Ordering::Release);
                }
                4 => {
                    slot.sample = None;
                    diagnostics.error(view.main_entity.id(), "view GPU timing readback failed");
                    slot.phase.store(0, Ordering::Release);
                }
                _ => {}
            }
        }
    }
}

pub(super) fn install(app: &mut App) {
    app.register_type::<GaussianPointSplattingViewBudget>()
        .add_plugins(ExtractComponentPlugin::<GaussianPointSplattingViewBudget>::default())
        .init_resource::<GaussianPointSplattingViewBudgetDiagnostics>()
        .init_resource::<Measurements>()
        .init_resource::<Controllers>()
        .add_systems(
            PostUpdate,
            update_policy.after(VisibilitySystems::CheckVisibility),
        );
    let measurements = app.world().resource::<Measurements>().clone();
    let diagnostics = app
        .world()
        .resource::<GaussianPointSplattingViewBudgetDiagnostics>()
        .clone();
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render
            .insert_resource(measurements)
            .insert_resource(diagnostics)
            .init_gpu_resource::<Timers>()
            .add_systems(RenderStartup, reset_measurements)
            .add_systems(
                Render,
                prepare_timers.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                start_timer
                    .before(LodCompactionLabel)
                    .before(GpuLodTraversalRender),
            )
            .add_systems(Core3d, finish_timer.after(upscaling))
            .add_systems(
                Render,
                collect_timers
                    .in_set(RenderSystems::Cleanup)
                    .after(RenderSystems::Render),
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_admission_bounds_apply_before_feedback_and_stop_coarsening() {
        let policy = GaussianPointSplattingViewBudget {
            target_view_gpu_ms: 10.0,
            min_selected_gaussians: 1,
            max_selected_gaussians: 800,
        };
        let (minimum, maximum) = admission_bounds(&policy, 700, 600).unwrap();
        let mut controller = Controller::new(&policy, 100);
        controller.set_admission_bounds(minimum, maximum);
        assert_eq!(
            controller.cap, 600,
            "root admission cannot wait for GPU feedback"
        );
        let sample = Observation {
            policy: policy.clone(),
            submission: 1,
            record_cap: 600,
            samples: 1,
            minimum_samples: 1,
            gpu_ms: 20.0,
            points: 1_000,
            point_flags: 1,
            traversal_flags: 0,
            requested_pages: 0,
        };
        assert_eq!(
            controller.observe(&sample),
            Some(GaussianPointSplattingViewBudgetAction::AtMinimum)
        );
        controller.set_admission_bounds(650, 700);
        assert_eq!(controller.cap, 650);
        assert_eq!(
            controller.observe(&Observation {
                submission: 2,
                ..sample
            }),
            None
        );
        assert!(admission_bounds(&policy, 599, 600).is_err());
    }

    #[test]
    fn view_budget_rejects_stale_data_waits_for_sampling_and_rolls_back_worse_coarsening() {
        use GaussianPointSplattingViewBudgetAction::*;
        let policy = GaussianPointSplattingViewBudget {
            target_view_gpu_ms: 10.0,
            min_selected_gaussians: 100,
            max_selected_gaussians: 800,
        };
        let mut controller = Controller::new(&policy, 800);
        let mut sample = Observation {
            policy,
            submission: 1,
            record_cap: 800,
            samples: 4,
            minimum_samples: 1,
            gpu_ms: 20.0,
            points: 4_000,
            point_flags: 0,
            traversal_flags: 0,
            requested_pages: 0,
        };
        let mut elevated_floor = Controller::new(&sample.policy, 800);
        elevated_floor.minimum_samples = 4;
        assert_eq!(elevated_floor.observe(&sample), None);
        let mut floor_sample = sample.clone();
        floor_sample.minimum_samples = 4;
        assert_eq!(elevated_floor.observe(&floor_sample), Some(Coarsened));
        assert_eq!(controller.observe(&sample), Some(WaitingForSamples));
        sample.samples = 1;
        assert_eq!(controller.observe(&sample), None);
        sample.submission = 2;
        sample.points = 1_000;
        assert_eq!(controller.observe(&sample), Some(Coarsened));
        assert_eq!(controller.cap, 700);
        sample.submission = 3;
        assert_eq!(
            controller.observe(&sample),
            None,
            "old-cap feedback changed the new policy"
        );
        sample.record_cap = 700;
        sample.gpu_ms = 30.0;
        sample.points = 2_000;
        for submission in 4..11 {
            sample.submission = submission;
            assert_eq!(controller.observe(&sample), Some(Unchanged));
        }
        sample.submission = 11;
        assert_eq!(controller.observe(&sample), Some(RolledBack));
        assert_eq!(controller.cap, 800);
        sample.record_cap = 800;
        sample.submission = 12;
        assert_eq!(controller.observe(&sample), Some(Unchanged));
        sample.submission = 13;
        sample.samples = 8;
        sample.point_flags = 1;
        assert_eq!(
            controller.observe(&sample),
            Some(Coarsened),
            "hard overflow must remain bounded above the sample floor"
        );

        controller = Controller::new(&sample.policy, 800);
        sample.record_cap = 800;
        sample.samples = 1;
        sample.point_flags = 0;
        sample.submission = 14;
        assert_eq!(controller.observe(&sample), Some(Coarsened));
        assert_eq!(controller.cap, 700);
        sample.record_cap = 700;
        sample.submission = 15;
        sample.point_flags = 1;
        // An aborted image's short timing is not evidence of cheaper rendering.
        sample.gpu_ms = 0.1;
        assert_eq!(controller.observe(&sample), Some(RolledBack));
        assert_eq!(controller.cap, 800);
        assert!(controller.trial.is_none());
        assert_eq!(controller.cooldown, 64);
        sample.submission = 16;
        assert_eq!(
            controller.observe(&sample),
            None,
            "retired trial feedback must remain stale"
        );
    }
}
