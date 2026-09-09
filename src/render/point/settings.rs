//! Camera policy and bounded allocation for Gaussian point splatting.
//!
//! Automatic control changes complete image sampling layers. It never changes
//! individual Gaussian sampling probabilities or the hierarchy quality target.

use std::fmt;

use bevy::{prelude::*, render::extract_component::ExtractComponent};
use bevy_args::{Deserialize, Serialize};

/// Maximum number of complete sampling layers supported by the point renderer.
pub const GAUSSIAN_POINT_SPLATTING_MAX_SAMPLES_PER_PIXEL: u32 = 8;
/// Portable bound for the projected-Gaussian prefix scan.
pub const GAUSSIAN_POINT_SPLATTING_MAX_PROJECTED_GAUSSIANS: u32 = 16_777_216;
/// Bound preserving `u32` point-prefix and padded-dispatch arithmetic.
pub const GAUSSIAN_POINT_SPLATTING_MAX_POINTS_PER_FRAME: u32 = 1 << 30;
/// Stable default seed; callers can change it without changing sample budgets.
pub const GAUSSIAN_POINT_SPLATTING_DEFAULT_SEED: u32 = 0x4750_5331;

/// Current camera admission, independent of asynchronously completed counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GaussianPointSplattingAvailability {
    /// No complete image can currently be displayed; eligible per-cloud work is
    /// still suppressed because the camera explicitly selected this backend.
    #[default]
    Unavailable,
    /// Reuse the last complete image while new work cannot be admitted.
    Retained,
    /// A bounded workspace and the required pipelines are available.
    Ready,
}

/// Opt-in settings attached to each camera using Gaussian point splatting.
///
/// Call [`Self::validate`] before admitting GPU work. The byte limit applies to
/// the renderer's accounted allocation; actual buffer strides and fixed storage
/// are supplied to [`Self::sample_layer_allocation`] by the render path.
#[derive(Clone, Component, Debug, ExtractComponent, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Component)]
#[serde(default)]
pub struct GaussianPointSplattingSettings {
    /// Initial complete sampling layers; also the fixed count without feedback.
    pub samples_per_pixel: u32,
    /// Minimum complete sampling layers admitted by automatic control. This is
    /// an application noise policy, not an image-quality certificate. Allocation
    /// failure at this floor leaves the current image unavailable or retained.
    pub min_samples_per_pixel: u32,
    /// Maximum projected Gaussian records admitted to the prefix scan.
    pub max_projected_gaussians: u32,
    /// Maximum requested point attempts admitted in a frame.
    pub max_points_per_frame: u32,
    /// Maximum accounted GPU allocation for this camera's point renderer.
    pub max_gpu_bytes: u64,
    /// Deterministic sampling seed, independent of automatic layer control.
    pub seed: u32,
    /// Vary the RNG with the frame identity. Disable for repeatable image tests.
    pub temporal_sampling: bool,
    /// Optional target for measured backend GPU time. Automatic admission may
    /// reduce complete layers to fit memory; capacity grows only when needed.
    /// `None` keeps the requested sampling count fixed.
    pub target_gpu_ms: Option<f32>,
}

impl Default for GaussianPointSplattingSettings {
    fn default() -> Self {
        Self {
            samples_per_pixel: 4,
            min_samples_per_pixel: 1,
            max_projected_gaussians: 1_048_576,
            max_points_per_frame: 16_777_216,
            max_gpu_bytes: 256 * 1024 * 1024,
            seed: GAUSSIAN_POINT_SPLATTING_DEFAULT_SEED,
            temporal_sampling: true,
            target_gpu_ms: None,
        }
    }
}

impl GaussianPointSplattingSettings {
    pub fn validate(&self) -> Result<(), GaussianPointSplattingSettingsError> {
        validate_samples_per_pixel(self.samples_per_pixel)?;
        validate_samples_per_pixel(self.min_samples_per_pixel)?;
        if self.samples_per_pixel < self.min_samples_per_pixel {
            return Err(GaussianPointSplattingSettingsError::SamplesBelowMinimum {
                requested: self.samples_per_pixel,
                minimum: self.min_samples_per_pixel,
            });
        }
        for (field, value) in [
            (
                "max_projected_gaussians",
                u64::from(self.max_projected_gaussians),
            ),
            ("max_points_per_frame", u64::from(self.max_points_per_frame)),
            ("max_gpu_bytes", self.max_gpu_bytes),
        ] {
            if value == 0 {
                return Err(GaussianPointSplattingSettingsError::Zero(field));
            }
        }
        if self.max_projected_gaussians > GAUSSIAN_POINT_SPLATTING_MAX_PROJECTED_GAUSSIANS {
            return Err(
                GaussianPointSplattingSettingsError::ProjectedGaussiansExceedLimit {
                    requested: self.max_projected_gaussians,
                },
            );
        }
        if self.max_points_per_frame > GAUSSIAN_POINT_SPLATTING_MAX_POINTS_PER_FRAME {
            return Err(
                GaussianPointSplattingSettingsError::PointsPerFrameExceedLimit {
                    requested: self.max_points_per_frame,
                },
            );
        }
        if self
            .target_gpu_ms
            .is_some_and(|target| !target.is_finite() || target <= 0.0)
        {
            return Err(GaussianPointSplattingSettingsError::InvalidTargetGpuMilliseconds);
        }
        Ok(())
    }

    /// Checks the complete pixel-layer allocation without allocating memory.
    ///
    /// `bytes_per_sample` must include all per-sample storage, and
    /// `fixed_gpu_bytes` all other allocations charged to this camera. This
    /// helper checks arithmetic, `u32` shader indexing, and the configured byte
    /// budget. Device-specific texture and buffer limits remain render admission
    /// checks. Pixel slots are distinct from the emitted-point work budget.
    pub fn sample_layer_allocation(
        &self,
        viewport: UVec2,
        samples_per_pixel: u32,
        bytes_per_sample: u64,
        fixed_gpu_bytes: u64,
    ) -> Result<GaussianPointSplattingAllocation, GaussianPointSplattingSettingsError> {
        self.validate()?;
        validate_samples_per_pixel(samples_per_pixel)?;
        if samples_per_pixel < self.min_samples_per_pixel {
            return Err(GaussianPointSplattingSettingsError::SamplesBelowMinimum {
                requested: samples_per_pixel,
                minimum: self.min_samples_per_pixel,
            });
        }
        if viewport.x == 0 || viewport.y == 0 {
            return Err(GaussianPointSplattingSettingsError::Zero("viewport extent"));
        }
        if bytes_per_sample == 0 {
            return Err(GaussianPointSplattingSettingsError::Zero(
                "bytes_per_sample",
            ));
        }
        let pixel_count = u64::from(viewport.x)
            .checked_mul(u64::from(viewport.y))
            .ok_or(GaussianPointSplattingSettingsError::ArithmeticOverflow(
                "pixel count",
            ))?;
        let sample_slots = pixel_count
            .checked_mul(u64::from(samples_per_pixel))
            .ok_or(GaussianPointSplattingSettingsError::ArithmeticOverflow(
                "sample slots",
            ))?;
        let total_gpu_bytes = sample_slots
            .checked_mul(bytes_per_sample)
            .and_then(|bytes| bytes.checked_add(fixed_gpu_bytes))
            .ok_or(GaussianPointSplattingSettingsError::ArithmeticOverflow(
                "GPU bytes",
            ))?;
        let sample_slots = u32::try_from(sample_slots)
            .map_err(|_| GaussianPointSplattingSettingsError::SampleSlotsExceedIndexLimit)?;
        if total_gpu_bytes > self.max_gpu_bytes {
            return Err(GaussianPointSplattingSettingsError::GpuByteBudgetExceeded {
                required: total_gpu_bytes,
                limit: self.max_gpu_bytes,
            });
        }
        Ok(GaussianPointSplattingAllocation {
            // Each pixel owns at least one slot, whose index was checked above.
            pixel_count: pixel_count as u32,
            sample_slots,
            total_gpu_bytes,
        })
    }
}

/// A checked allocation size, before device-specific resource admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GaussianPointSplattingAllocation {
    pub pixel_count: u32,
    pub sample_slots: u32,
    pub total_gpu_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GaussianPointSplattingSettingsError {
    Zero(&'static str),
    SamplesPerPixelExceedLimit { requested: u32 },
    SamplesBelowMinimum { requested: u32, minimum: u32 },
    ProjectedGaussiansExceedLimit { requested: u32 },
    PointsPerFrameExceedLimit { requested: u32 },
    InvalidTargetGpuMilliseconds,
    ArithmeticOverflow(&'static str),
    SampleSlotsExceedIndexLimit,
    GpuByteBudgetExceeded { required: u64, limit: u64 },
}

impl fmt::Display for GaussianPointSplattingSettingsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero(field) => write!(formatter, "{field} must be nonzero"),
            Self::SamplesPerPixelExceedLimit { requested } => write!(
                formatter,
                "samples_per_pixel {requested} exceeds {GAUSSIAN_POINT_SPLATTING_MAX_SAMPLES_PER_PIXEL}"
            ),
            Self::SamplesBelowMinimum { requested, minimum } => write!(
                formatter,
                "sampling request {requested} is below the configured minimum {minimum}"
            ),
            Self::ProjectedGaussiansExceedLimit { requested } => write!(
                formatter,
                "max_projected_gaussians {requested} exceeds the portable scan limit {GAUSSIAN_POINT_SPLATTING_MAX_PROJECTED_GAUSSIANS}"
            ),
            Self::PointsPerFrameExceedLimit { requested } => write!(
                formatter,
                "max_points_per_frame {requested} exceeds the safe prefix/dispatch limit {GAUSSIAN_POINT_SPLATTING_MAX_POINTS_PER_FRAME}"
            ),
            Self::InvalidTargetGpuMilliseconds => {
                write!(formatter, "target_gpu_ms must be finite and positive")
            }
            Self::ArithmeticOverflow(field) => write!(formatter, "{field} arithmetic overflow"),
            Self::SampleSlotsExceedIndexLimit => {
                write!(formatter, "sample slots exceed u32 shader indexing")
            }
            Self::GpuByteBudgetExceeded { required, limit } => write!(
                formatter,
                "point splatting requires {required} GPU bytes, exceeding the {limit} byte budget"
            ),
        }
    }
}

impl std::error::Error for GaussianPointSplattingSettingsError {}

fn validate_samples_per_pixel(samples: u32) -> Result<(), GaussianPointSplattingSettingsError> {
    if samples == 0 {
        Err(GaussianPointSplattingSettingsError::Zero(
            "samples_per_pixel",
        ))
    } else if samples > GAUSSIAN_POINT_SPLATTING_MAX_SAMPLES_PER_PIXEL {
        Err(GaussianPointSplattingSettingsError::SamplesPerPixelExceedLimit { requested: samples })
    } else {
        Ok(())
    }
}

/// Result of one timestamp or point-budget observation; ignored input changes no state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GaussianPointSplattingSampleUpdate {
    IgnoredStaleSubmission,
    IgnoredInvalidGpuTime,
    AutomaticDisabled,
    Unchanged,
    Decreased { from: u32, to: u32 },
    Increased { from: u32, to: u32 },
}

/// Pure controller for complete global sampling layers, without render state.
///
/// One fresh observation above 110% of the target removes at most one layer.
/// Eight consecutive fresh observations below 80% add at most one layer. The
/// intervening band resets the under-budget streak. Decisions stay within
/// `min_samples_per_pixel..=8`; callers must admit the resulting allocation.
#[derive(Clone, Debug)]
pub struct GaussianPointSplattingSampleController {
    samples_per_pixel: u32,
    min_samples_per_pixel: u32,
    target_gpu_ms: Option<f32>,
    last_submission_id: Option<u64>,
    under_budget_streak: u8,
}

impl GaussianPointSplattingSampleController {
    pub fn new(
        settings: &GaussianPointSplattingSettings,
    ) -> Result<Self, GaussianPointSplattingSettingsError> {
        settings.validate()?;
        Ok(Self {
            samples_per_pixel: settings.samples_per_pixel,
            min_samples_per_pixel: settings.min_samples_per_pixel,
            target_gpu_ms: settings.target_gpu_ms,
            last_submission_id: None,
            under_budget_streak: 0,
        })
    }

    pub fn samples_per_pixel(&self) -> u32 {
        self.samples_per_pixel
    }

    pub fn last_submission_id(&self) -> Option<u64> {
        self.last_submission_id
    }

    /// Apply policy edits without forgetting completed feedback when only
    /// resource ceilings or the random seed change.
    pub(crate) fn reconfigure(
        &mut self,
        previous: &GaussianPointSplattingSettings,
        next: &GaussianPointSplattingSettings,
    ) {
        if previous.samples_per_pixel != next.samples_per_pixel
            || previous.min_samples_per_pixel != next.min_samples_per_pixel
            || previous.target_gpu_ms.is_some() != next.target_gpu_ms.is_some()
        {
            self.samples_per_pixel = next.samples_per_pixel;
        }
        if self.target_gpu_ms != next.target_gpu_ms
            || self.min_samples_per_pixel != next.min_samples_per_pixel
        {
            self.under_budget_streak = 0;
        }
        self.target_gpu_ms = next.target_gpu_ms;
        self.min_samples_per_pixel = next.min_samples_per_pixel;
    }

    /// Clamp an automatic request to complete layers that were actually
    /// admitted. This is resource feedback, not an invented GPU duration.
    pub(crate) fn admit_layers(&mut self, admitted: u32) {
        debug_assert!(
            (self.min_samples_per_pixel..=GAUSSIAN_POINT_SPLATTING_MAX_SAMPLES_PER_PIXEL)
                .contains(&admitted)
        );
        if admitted < self.min_samples_per_pixel {
            return;
        }
        if self.target_gpu_ms.is_some() && self.samples_per_pixel > admitted {
            self.samples_per_pixel = admitted;
            self.under_budget_streak = 0;
        }
    }

    /// Apply asynchronous renderer feedback only to the sampling count that
    /// produced it. Queued timings from a replaced count must not cascade
    /// downshifts or contribute headroom toward another growth decision.
    pub(crate) fn observe_rendered_frame(
        &mut self,
        submission_id: u64,
        rendered_samples: u32,
        gpu_ms: Option<f32>,
        point_overflow: bool,
    ) -> Option<GaussianPointSplattingSampleUpdate> {
        if rendered_samples != self.samples_per_pixel {
            return None;
        }
        if point_overflow {
            Some(self.observe_point_overflow(submission_id))
        } else {
            gpu_ms.map(|ms| self.observe_gpu_time(submission_id, ms))
        }
    }

    /// Reduces one complete layer after actual point-budget overflow, without
    /// fabricating a GPU duration. Fixed sampling stays fixed; at the floor an
    /// overflow remains an explicit failure requiring a different work budget.
    pub fn observe_point_overflow(
        &mut self,
        submission_id: u64,
    ) -> GaussianPointSplattingSampleUpdate {
        if self
            .last_submission_id
            .is_some_and(|last| submission_id <= last)
        {
            return GaussianPointSplattingSampleUpdate::IgnoredStaleSubmission;
        }
        self.last_submission_id = Some(submission_id);
        if self.target_gpu_ms.is_none() {
            return GaussianPointSplattingSampleUpdate::AutomaticDisabled;
        }
        self.under_budget_streak = 0;
        let from = self.samples_per_pixel;
        self.samples_per_pixel = self
            .samples_per_pixel
            .saturating_sub(1)
            .max(self.min_samples_per_pixel);
        if self.samples_per_pixel == from {
            GaussianPointSplattingSampleUpdate::Unchanged
        } else {
            GaussianPointSplattingSampleUpdate::Decreased {
                from,
                to: self.samples_per_pixel,
            }
        }
    }

    /// Accepts only fresh, finite, nonnegative timestamp durations. A zero
    /// duration is valid (for example after timer quantization); unavailable
    /// measurements must not be supplied as artificial zeros.
    pub fn observe_gpu_time(
        &mut self,
        submission_id: u64,
        observed_gpu_ms: f32,
    ) -> GaussianPointSplattingSampleUpdate {
        if self
            .last_submission_id
            .is_some_and(|last| submission_id <= last)
        {
            return GaussianPointSplattingSampleUpdate::IgnoredStaleSubmission;
        }
        if !observed_gpu_ms.is_finite() || observed_gpu_ms < 0.0 {
            return GaussianPointSplattingSampleUpdate::IgnoredInvalidGpuTime;
        }
        self.last_submission_id = Some(submission_id);
        let Some(target_gpu_ms) = self.target_gpu_ms else {
            return GaussianPointSplattingSampleUpdate::AutomaticDisabled;
        };
        // f64 avoids overflowing threshold arithmetic for valid f32 targets.
        let ratio = f64::from(observed_gpu_ms) / f64::from(target_gpu_ms);
        let previous = self.samples_per_pixel;
        if ratio > 1.10 {
            self.under_budget_streak = 0;
            self.samples_per_pixel = self
                .samples_per_pixel
                .saturating_sub(1)
                .max(self.min_samples_per_pixel);
            if self.samples_per_pixel != previous {
                return GaussianPointSplattingSampleUpdate::Decreased {
                    from: previous,
                    to: self.samples_per_pixel,
                };
            }
        } else if ratio < 0.80 {
            self.under_budget_streak += 1;
            if self.under_budget_streak == 8 {
                self.under_budget_streak = 0;
                self.samples_per_pixel = (self.samples_per_pixel + 1)
                    .min(GAUSSIAN_POINT_SPLATTING_MAX_SAMPLES_PER_PIXEL);
                if self.samples_per_pixel != previous {
                    return GaussianPointSplattingSampleUpdate::Increased {
                        from: previous,
                        to: self.samples_per_pixel,
                    };
                }
            }
        } else {
            self.under_budget_streak = 0;
        }
        GaussianPointSplattingSampleUpdate::Unchanged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn automatic() -> GaussianPointSplattingSampleController {
        GaussianPointSplattingSampleController::new(&GaussianPointSplattingSettings {
            target_gpu_ms: Some(10.0),
            ..default()
        })
        .unwrap()
    }

    #[test]
    fn defaults_are_valid_and_fixed_without_gpu_feedback() {
        let settings = GaussianPointSplattingSettings::default();
        assert!(settings.validate().is_ok());
        assert_eq!(settings.samples_per_pixel, 4);
        assert_eq!(settings.min_samples_per_pixel, 1);
        assert_eq!(settings.max_projected_gaussians, 1_048_576);
        assert_eq!(settings.max_points_per_frame, 16_777_216);
        assert_eq!(settings.max_gpu_bytes, 256 * 1024 * 1024);
        assert!(settings.temporal_sampling);
        let mut controller = GaussianPointSplattingSampleController::new(&settings).unwrap();
        for submission in 0..32 {
            assert_eq!(
                controller.observe_gpu_time(submission, 1_000.0),
                GaussianPointSplattingSampleUpdate::AutomaticDisabled
            );
        }
        assert_eq!(controller.samples_per_pixel(), 4);
        assert_eq!(
            controller.observe_point_overflow(32),
            GaussianPointSplattingSampleUpdate::AutomaticDisabled
        );
        assert_eq!(controller.samples_per_pixel(), 4);
    }

    #[test]
    fn sampling_floor_survives_timing_overflow_and_memory_admission() {
        let settings = GaussianPointSplattingSettings {
            min_samples_per_pixel: 3,
            target_gpu_ms: Some(10.0),
            ..default()
        };
        let mut controller = GaussianPointSplattingSampleController::new(&settings).unwrap();
        controller.observe_gpu_time(1, 20.0);
        controller.observe_point_overflow(2);
        controller.observe_gpu_time(3, 20.0);
        assert_eq!(controller.samples_per_pixel(), 3);
        assert!(matches!(
            settings.sample_layer_allocation(UVec2::ONE, 2, 4, 0),
            Err(GaussianPointSplattingSettingsError::SamplesBelowMinimum { .. })
        ));
        let mut constrained = settings.clone();
        constrained.max_gpu_bytes = 8;
        assert!(matches!(
            constrained.sample_layer_allocation(UVec2::ONE, 3, 4, 0),
            Err(GaussianPointSplattingSettingsError::GpuByteBudgetExceeded { .. })
        ));
        let mut raised = settings.clone();
        raised.min_samples_per_pixel = 4;
        controller.reconfigure(&settings, &raised);
        controller.observe_point_overflow(4);
        assert_eq!(controller.samples_per_pixel(), 4);
        raised.min_samples_per_pixel = 5;
        assert!(raised.validate().is_err());
    }

    #[test]
    fn invalid_settings_fail_before_controller_or_allocation() {
        let defaults = GaussianPointSplattingSettings::default();
        let invalid = [
            GaussianPointSplattingSettings {
                samples_per_pixel: 0,
                ..defaults.clone()
            },
            GaussianPointSplattingSettings {
                samples_per_pixel: 9,
                ..defaults.clone()
            },
            GaussianPointSplattingSettings {
                max_projected_gaussians: 0,
                ..defaults.clone()
            },
            GaussianPointSplattingSettings {
                max_projected_gaussians: 16_777_217,
                ..defaults.clone()
            },
            GaussianPointSplattingSettings {
                max_points_per_frame: 0,
                ..defaults.clone()
            },
            GaussianPointSplattingSettings {
                max_points_per_frame: GAUSSIAN_POINT_SPLATTING_MAX_POINTS_PER_FRAME + 1,
                ..defaults.clone()
            },
            GaussianPointSplattingSettings {
                max_gpu_bytes: 0,
                ..defaults.clone()
            },
        ];
        for settings in invalid {
            assert!(settings.validate().is_err());
            assert!(GaussianPointSplattingSampleController::new(&settings).is_err());
            assert!(
                settings
                    .sample_layer_allocation(UVec2::ONE, 1, 4, 0)
                    .is_err()
            );
        }
        let at_limit = GaussianPointSplattingSettings {
            max_points_per_frame: GAUSSIAN_POINT_SPLATTING_MAX_POINTS_PER_FRAME,
            ..defaults.clone()
        };
        assert!(at_limit.validate().is_ok());
        assert_eq!(
            GaussianPointSplattingSettings {
                max_points_per_frame: GAUSSIAN_POINT_SPLATTING_MAX_POINTS_PER_FRAME + 1,
                ..defaults.clone()
            }
            .validate(),
            Err(
                GaussianPointSplattingSettingsError::PointsPerFrameExceedLimit {
                    requested: GAUSSIAN_POINT_SPLATTING_MAX_POINTS_PER_FRAME + 1,
                }
            )
        );
        for target in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                GaussianPointSplattingSettings {
                    target_gpu_ms: Some(target),
                    ..defaults.clone()
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn allocation_checks_exact_budget_overflow_and_shader_indexing() {
        let mut settings = GaussianPointSplattingSettings {
            max_gpu_bytes: 16 * 8 * 4 * 12 + 256,
            ..default()
        };
        let allocation = settings
            .sample_layer_allocation(UVec2::new(16, 8), 4, 12, 256)
            .unwrap();
        assert_eq!(allocation.pixel_count, 128);
        assert_eq!(allocation.sample_slots, 512);
        assert_eq!(allocation.total_gpu_bytes, settings.max_gpu_bytes);
        settings.max_gpu_bytes -= 1;
        assert!(matches!(
            settings.sample_layer_allocation(UVec2::new(16, 8), 4, 12, 256),
            Err(GaussianPointSplattingSettingsError::GpuByteBudgetExceeded { .. })
        ));
        settings.max_gpu_bytes = u64::MAX;
        for (viewport, spp, stride, fixed) in [
            (UVec2::ZERO, 1, 4, 0),
            (UVec2::ONE, 0, 4, 0),
            (UVec2::ONE, 9, 4, 0),
            (UVec2::ONE, 1, 0, 0),
            (UVec2::splat(u32::MAX), 8, 4, 0),
            (UVec2::new(u32::MAX, 2), 1, 1, 0),
            (UVec2::new(2, 1), 1, u64::MAX, 0),
            (UVec2::ONE, 1, 1, u64::MAX),
        ] {
            assert!(
                settings
                    .sample_layer_allocation(viewport, spp, stride, fixed)
                    .is_err()
            );
        }
    }

    #[test]
    fn stale_and_invalid_feedback_cannot_change_layers_or_streak() {
        let mut controller = automatic();
        for submission in 1..8 {
            assert_eq!(
                controller.observe_gpu_time(submission, 7.0),
                GaussianPointSplattingSampleUpdate::Unchanged
            );
        }
        assert_eq!(
            controller.observe_gpu_time(7, 100.0),
            GaussianPointSplattingSampleUpdate::IgnoredStaleSubmission
        );
        for milliseconds in [f32::NAN, f32::INFINITY, -1.0] {
            assert_eq!(
                controller.observe_gpu_time(8, milliseconds),
                GaussianPointSplattingSampleUpdate::IgnoredInvalidGpuTime
            );
        }
        assert_eq!(controller.last_submission_id(), Some(7));
        assert_eq!(
            controller.observe_gpu_time(8, 7.0),
            GaussianPointSplattingSampleUpdate::Increased { from: 4, to: 5 }
        );
        assert_eq!(
            controller.observe_gpu_time(6, 100.0),
            GaussianPointSplattingSampleUpdate::IgnoredStaleSubmission
        );
        assert_eq!(controller.samples_per_pixel(), 5);
        assert_eq!(
            controller.observe_point_overflow(9),
            GaussianPointSplattingSampleUpdate::Decreased { from: 5, to: 4 }
        );
        assert_eq!(
            controller.observe_point_overflow(9),
            GaussianPointSplattingSampleUpdate::IgnoredStaleSubmission
        );
        assert_eq!(
            controller.observe_gpu_time(8, 0.0),
            GaussianPointSplattingSampleUpdate::IgnoredStaleSubmission
        );
    }

    #[test]
    fn hysteresis_reduces_immediately_and_requires_consecutive_headroom_to_grow() {
        let mut controller = automatic();
        assert_eq!(
            controller.observe_gpu_time(1, 20.0),
            GaussianPointSplattingSampleUpdate::Decreased { from: 4, to: 3 }
        );
        for submission in 2..9 {
            controller.observe_gpu_time(submission, 7.0);
        }
        controller.observe_gpu_time(9, 9.0);
        assert_eq!(controller.samples_per_pixel(), 3);
        for submission in 10..17 {
            controller.observe_gpu_time(submission, 7.0);
            assert_eq!(controller.samples_per_pixel(), 3);
        }
        assert_eq!(
            controller.observe_gpu_time(17, 7.0),
            GaussianPointSplattingSampleUpdate::Increased { from: 3, to: 4 }
        );
    }

    #[test]
    fn admission_and_budget_edits_preserve_fresh_feedback_without_fake_timing() {
        let mut previous = GaussianPointSplattingSettings {
            target_gpu_ms: Some(10.0),
            ..default()
        };
        let mut controller = GaussianPointSplattingSampleController::new(&previous).unwrap();
        controller.observe_gpu_time(12, 20.0);
        controller.admit_layers(2);
        assert_eq!(controller.samples_per_pixel(), 2);
        assert_eq!(controller.last_submission_id(), Some(12));
        let mut next = previous.clone();
        next.max_points_per_frame = 1;
        next.max_gpu_bytes /= 2;
        controller.reconfigure(&previous, &next);
        assert_eq!(controller.samples_per_pixel(), 2);
        assert_eq!(controller.last_submission_id(), Some(12));
        assert_eq!(
            controller.observe_point_overflow(12),
            GaussianPointSplattingSampleUpdate::IgnoredStaleSubmission
        );
        assert_eq!(
            controller.observe_point_overflow(13),
            GaussianPointSplattingSampleUpdate::Decreased { from: 2, to: 1 }
        );
        previous = next.clone();
        next.target_gpu_ms = None;
        controller.reconfigure(&previous, &next);
        controller.admit_layers(1);
        assert_eq!(
            controller.samples_per_pixel(),
            next.samples_per_pixel,
            "fixed sampling must not silently reduce quality"
        );
    }

    #[test]
    fn delayed_rendered_samples_do_not_cascade_sampling_changes() {
        use GaussianPointSplattingSampleUpdate::{Decreased, Increased, Unchanged};
        let mut controller =
            GaussianPointSplattingSampleController::new(&GaussianPointSplattingSettings {
                samples_per_pixel: 8,
                target_gpu_ms: Some(10.0),
                ..default()
            })
            .unwrap();
        assert_eq!(
            controller.observe_rendered_frame(1, 8, Some(20.0), false),
            Some(Decreased { from: 8, to: 7 })
        );
        // Both delayed timing and delayed overflow still describe eight layers.
        assert_eq!(
            controller.observe_rendered_frame(2, 8, Some(20.0), false),
            None
        );
        assert_eq!(controller.observe_rendered_frame(3, 8, None, true), None);
        assert_eq!(controller.samples_per_pixel(), 7);
        assert_eq!(controller.last_submission_id(), Some(1));
        assert_eq!(
            controller.observe_rendered_frame(4, 7, None, true),
            Some(Decreased { from: 7, to: 6 })
        );
        assert_eq!(
            controller.observe_rendered_frame(5, 7, Some(1.0), false),
            None
        );
        for submission in 6..13 {
            assert_eq!(
                controller.observe_rendered_frame(submission, 6, Some(1.0), false),
                Some(Unchanged)
            );
        }
        assert_eq!(controller.samples_per_pixel(), 6);
        assert_eq!(
            controller.observe_rendered_frame(13, 6, Some(1.0), false),
            Some(Increased { from: 6, to: 7 })
        );
    }

    #[test]
    fn automatic_changes_are_bounded_to_one_whole_layer_and_never_wrap() {
        let mut controller = automatic();
        for submission in 0..32 {
            let before = controller.samples_per_pixel();
            controller.observe_point_overflow(submission);
            assert!(before - controller.samples_per_pixel() <= 1);
        }
        assert_eq!(controller.samples_per_pixel(), 1);
        for submission in 32..128 {
            let before = controller.samples_per_pixel();
            controller.observe_gpu_time(submission, 0.0);
            assert!(controller.samples_per_pixel() - before <= 1);
        }
        assert_eq!(controller.samples_per_pixel(), 8);
        controller.observe_gpu_time(u64::MAX, 20.0);
        let before = controller.samples_per_pixel();
        assert_eq!(
            controller.observe_gpu_time(0, 0.0),
            GaussianPointSplattingSampleUpdate::IgnoredStaleSubmission
        );
        assert_eq!(controller.samples_per_pixel(), before);
    }
}
