//! Runtime controls for hierarchical Gaussian level-of-detail rendering.
//!
//! This module deliberately contains policy, not hierarchy or renderer state. A
//! cloud can therefore serialize these settings, an editor can reflect them, and
//! CPU/GPU selectors can share the same endpoint and error-curve contract.

use std::{fmt, time::Duration};

use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy_args::{Deserialize, Serialize};

const HIGH_QUALITY_FIDELITY_GUARD_START: f32 = 0.90;
const HIGH_QUALITY_FIDELITY_GUARD_FULL: f32 = 0.99;
const HIGH_QUALITY_CERTIFICATE_GUARD_START: f32 = 0.90;
const HIGH_QUALITY_CERTIFICATE_GUARD_FULL: f32 = 0.95;
const PROJECTED_ERROR_AUTHORITY_FULL: f32 = 0.99;
const MIN_QUANTIZED_HIGH_FIDELITY_CERTIFICATE: f32 = 1.0 / u16::MAX as f32;

/// Hard public bound matching the process/realm HTTP admission capacity.
pub const MAX_STREAMING_CONCURRENT_REQUESTS: u32 = 256;

/// Default decoded-record work budget for caller-thread page preprocessing.
///
/// Native worker-pool decoding is not frame-coupled. The cooperative Wasm
/// backend uses this limit to spread checksum, decode, validation, and support
/// bound work across application frames.
pub const DEFAULT_COOPERATIVE_PREPROCESS_GAUSSIANS_PER_FRAME: u32 = 4_096;

/// What a quality value means after it has been validated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Reflect, Serialize, Deserialize)]
pub enum LodQualityEndpoint {
    /// Render only the coarsest complete representation (normally the roots).
    Coarsest,
    /// Select a continuous cut using structural detail and projected error.
    Continuous,
    /// Select original leaf data. Culling remains enabled, but coarsening does not.
    Original,
}

/// Whether hierarchy selection follows the live render camera or a captured
/// per-view camera snapshot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Reflect, Serialize, Deserialize)]
pub enum LodSelectionMode {
    /// Re-evaluate selection and streaming demand against the current camera.
    #[default]
    Dynamic,
    /// Capture each view on entry and keep selection/streaming demand fixed to
    /// that snapshot until the mode returns to [`Self::Dynamic`]. Residency and
    /// physical page availability continue to converge normally.
    Frozen,
}

/// Resolved runtime quality contract.
///
/// [`GaussianLodSettings::quality`] remains the serialized presentation slider
/// for source compatibility. Selection and status reporting should use this
/// target so an interior slider value has an explicit screen-space meaning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect, Serialize, Deserialize)]
pub enum LodQualityTarget {
    /// Render only the coarsest complete representation (normally the roots).
    Coarsest,
    /// Refine against structural detail and projected error. Cubic
    /// projected-error authority rises continuously with the quality slider
    /// and becomes a hard cap at `.99`. A structural guard removes
    /// projected-coverage relaxation from `.90` to `.99`; positive
    /// builder-authored fidelity-certificate pressure is gated off through
    /// `.90` and reaches full authority at `.95`.
    Balanced {
        detail_fraction: f32,
        max_error_px: f32,
    },
    /// Select original leaf data without permitting representative error.
    #[default]
    Original,
}

impl LodQualityTarget {
    pub fn endpoint(self) -> LodQualityEndpoint {
        match self {
            Self::Coarsest => LodQualityEndpoint::Coarsest,
            Self::Balanced { .. } => LodQualityEndpoint::Continuous,
            Self::Original => LodQualityEndpoint::Original,
        }
    }

    /// Returns the finite nominal projected-error part of a balanced target.
    /// Use [`Self::effective_max_screen_space_error_px`] for the independently
    /// enforced cap. `Coarsest` has no upper error bound; `Original` is exact
    /// rather than a zero-pixel approximation target.
    pub fn max_screen_space_error_px(self) -> Option<f32> {
        match self {
            Self::Balanced { max_error_px, .. } => Some(max_error_px),
            Self::Coarsest | Self::Original => None,
        }
    }

    /// Returns the structural-detail part of a balanced target.
    pub fn detail_fraction(self) -> Option<f32> {
        match self {
            Self::Balanced {
                detail_fraction, ..
            } => Some(detail_fraction),
            Self::Coarsest | Self::Original => None,
        }
    }

    /// Fraction of projected-error pressure that is independently
    /// authoritative for this target.
    ///
    /// A balanced request follows a normalized cubic curve and reaches full
    /// authority at quality `.99`. Endpoints and a manually constructed
    /// balanced target with zero authority return `None` because they do not
    /// expose a finite approximate-error contract.
    pub fn error_authority(self) -> Option<f32> {
        match self {
            Self::Balanced {
                detail_fraction, ..
            } => {
                let authority = projected_error_authority(detail_fraction);
                (authority > 0.0 && authority.is_finite()).then_some(authority)
            }
            Self::Coarsest | Self::Original => None,
        }
    }

    /// Effective maximum projected error after applying
    /// [`Self::error_authority`].
    ///
    /// The nominal error curve remains one half of balanced selection. This
    /// value reports the independent cap implied by `authority * error_pressure
    /// <= 1`. Endpoints, zero authority, and manually constructed invalid
    /// limits return `None`.
    pub fn effective_max_screen_space_error_px(self) -> Option<f32> {
        match self {
            Self::Balanced { max_error_px, .. }
                if max_error_px.is_finite() && max_error_px >= 0.0 =>
            {
                self.error_authority().and_then(|authority| {
                    let effective = max_error_px / authority;
                    effective.is_finite().then_some(effective)
                })
            }
            Self::Coarsest | Self::Original | Self::Balanced { .. } => None,
        }
    }

    /// Structural detail requested at a node's projected coverage.
    ///
    /// Up to quality `.90`, this is exactly `quality * coverage`. Above `.90`,
    /// a fixed smoothstep guard blends out coverage-based relaxation so `.99`
    /// requests essentially full structural fidelity. Projected-error
    /// authority is an independent continuous mapping.
    pub fn structural_detail_demand(self, projected_coverage: f32) -> f32 {
        match self {
            Self::Coarsest => 0.0,
            Self::Original => 1.0,
            Self::Balanced {
                detail_fraction, ..
            } => {
                let detail_fraction = finite_clamp01(detail_fraction);
                let coverage = finite_clamp01(projected_coverage);
                let covered_demand = detail_fraction * coverage;
                let fidelity_guard = high_quality_fidelity_guard(detail_fraction);
                if fidelity_guard <= 0.0 {
                    covered_demand
                } else if fidelity_guard >= 1.0 {
                    detail_fraction
                } else {
                    covered_demand + (detail_fraction - covered_demand) * fidelity_guard
                }
            }
        }
    }

    fn error_limit_px(self) -> f32 {
        match self {
            Self::Coarsest => f32::INFINITY,
            Self::Balanced { max_error_px, .. } => max_error_px,
            Self::Original => 0.0,
        }
    }

    /// Returns the effective pressure on one selected node. Values at or below
    /// one satisfy the target. A balanced target ordinarily accepts a node when
    /// either its structural interval or projected error is satisfied, while a
    /// continuous authority term increasingly caps how far the structural path
    /// may exceed the advertised pixel target. Positive builder-authored
    /// fidelity certificates independently contribute coverage-aware pressure;
    /// legacy zero/tiny certificates fail closed at quality `.95` and above.
    pub fn node_pressure(
        self,
        node_quality_threshold: f32,
        error_px: f32,
        projected_coverage: f32,
        high_fidelity_certificate: f32,
        is_original_representation: bool,
    ) -> f32 {
        self.node_pressure_with_error_limit(
            node_quality_threshold,
            error_px,
            self.error_limit_px(),
            projected_coverage,
            high_fidelity_certificate,
            is_original_representation,
        )
    }

    pub(crate) fn node_pressure_with_error_limit(
        self,
        node_quality_threshold: f32,
        error_px: f32,
        error_limit_px: f32,
        projected_coverage: f32,
        high_fidelity_certificate: f32,
        is_original_representation: bool,
    ) -> f32 {
        match self {
            Self::Coarsest => 0.0,
            Self::Original => {
                if is_original_representation {
                    0.0
                } else {
                    f32::MAX
                }
            }
            Self::Balanced {
                detail_fraction, ..
            } => {
                let structural_pressure = detail_ratio(
                    self.structural_detail_demand(projected_coverage),
                    node_quality_threshold,
                );
                let error_pressure = error_ratio(error_px, error_limit_px);
                let balanced_pressure = structural_pressure.min(error_pressure);
                let error_authority = projected_error_authority(detail_fraction);
                balanced_pressure.max(error_authority * error_pressure).max(
                    high_fidelity_certificate_pressure(
                        detail_fraction,
                        projected_coverage,
                        high_fidelity_certificate,
                        is_original_representation,
                    ),
                )
            }
        }
    }
}

/// Continuous authority applied independently to projected-error pressure.
///
/// The normalized cubic mapping deliberately uses only presentation
/// quality and the fixed `.99` fidelity anchor. Projection supplies the
/// scene-scale-independent distance response. Keeping this separate from the
/// structural coverage guard lets that compatibility contract remain unchanged.
pub(crate) fn projected_error_authority(detail_fraction: f32) -> f32 {
    let normalized =
        (finite_clamp01(detail_fraction) / PROJECTED_ERROR_AUTHORITY_FULL).clamp(0.0, 1.0);
    normalized * normalized * normalized
}

/// Shared scalar guard for near-original structural coverage policy.
/// Keeping this independent of node/view data gives renderer parity tests one
/// canonical CPU oracle without adding another user-facing setting.
pub(crate) fn high_quality_fidelity_guard(detail_fraction: f32) -> f32 {
    smooth_quality_guard(
        detail_fraction,
        HIGH_QUALITY_FIDELITY_GUARD_START,
        HIGH_QUALITY_FIDELITY_GUARD_FULL,
    )
}

/// High-quality authority for builder-authored fidelity certificates.
/// Ordinary qualities through `.90` deliberately carry no certificate
/// pressure; authority then rises smoothly to one at `.95`.
pub(crate) fn high_quality_certificate_guard(detail_fraction: f32) -> f32 {
    smooth_quality_guard(
        detail_fraction,
        HIGH_QUALITY_CERTIFICATE_GUARD_START,
        HIGH_QUALITY_CERTIFICATE_GUARD_FULL,
    )
}

/// Coverage-aware certificate demand for positive, quantized-compatible
/// certificates. The legacy quadratic/cubic demand shape is retained inside
/// the high-quality gate, but its pressure is exactly zero through `.90` and
/// reaches its full value at `.95`.
pub(crate) fn high_quality_certificate_demand(
    detail_fraction: f32,
    projected_coverage: f32,
) -> f32 {
    let detail = finite_clamp01(detail_fraction);
    let normalized = (detail / HIGH_QUALITY_CERTIFICATE_GUARD_FULL).clamp(0.0, 1.0);
    let base_demand = detail * normalized;
    let coverage_authority = normalized * normalized * normalized;
    let coverage = finite_clamp01(projected_coverage);
    let effective_coverage = coverage + (1.0 - coverage) * coverage_authority;
    high_quality_certificate_guard(detail) * base_demand * effective_coverage
}

fn smooth_quality_guard(detail_fraction: f32, start: f32, full: f32) -> f32 {
    let detail_fraction = finite_clamp01(detail_fraction);
    if detail_fraction <= start {
        0.0
    } else if detail_fraction >= full {
        1.0
    } else {
        let t = (detail_fraction - start) / (full - start);
        t * t * (3.0 - 2.0 * t)
    }
}

/// Pressure contributed by a builder-authored high-fidelity certificate.
///
/// Positive, finite, quantized-compatible certificates carry a coverage-aware
/// demand only in the high-quality `.90` to `.95` guard band. A zero, tiny, or
/// invalid value denotes a legacy or uncertified hierarchy: it remains
/// compatible below quality `.95`, then fails closed for non-original
/// representatives at `.95` and above.
pub(crate) fn high_fidelity_certificate_pressure(
    detail_fraction: f32,
    projected_coverage: f32,
    high_fidelity_certificate: f32,
    is_original_representation: bool,
) -> f32 {
    let detail = finite_clamp01(detail_fraction);
    let certificate_is_usable = high_fidelity_certificate.is_finite()
        && high_fidelity_certificate > MIN_QUANTIZED_HIGH_FIDELITY_CERTIFICATE
        && high_fidelity_certificate <= 1.0;
    if !certificate_is_usable {
        return if detail >= HIGH_QUALITY_CERTIFICATE_GUARD_FULL && !is_original_representation {
            f32::MAX
        } else {
            0.0
        };
    }
    detail_ratio(
        high_quality_certificate_demand(detail, projected_coverage),
        high_fidelity_certificate,
    )
}

fn detail_ratio(detail_fraction: f32, node_quality_threshold: f32) -> f32 {
    if node_quality_threshold <= 0.0 {
        if detail_fraction <= 0.0 {
            0.0
        } else {
            f32::MAX
        }
    } else {
        (detail_fraction / node_quality_threshold).min(f32::MAX)
    }
}

fn error_ratio(error_px: f32, limit_px: f32) -> f32 {
    if limit_px.is_infinite() {
        0.0
    } else if limit_px <= 0.0 {
        if error_px <= 0.0 { 0.0 } else { f32::MAX }
    } else {
        (error_px / limit_px).min(f32::MAX)
    }
}

/// Maps continuous quality to its nominal screen-space error target. The
/// independent error-authority curve progressively turns this target into a
/// hard cap.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ProjectedErrorCurve {
    /// Error limit approached just above quality zero, in physical pixels.
    coarse_error_px: f32,
    /// Error limit approached just below quality one, in physical pixels.
    fine_error_px: f32,
    /// Shapes quality before exponential interpolation. Values above one reserve
    /// more of the slider for coarse representations.
    quality_exponent: f32,
}

impl Default for ProjectedErrorCurve {
    fn default() -> Self {
        Self {
            // A log-uniform presentation mapping: q=.25 -> ~5.66 px,
            // q=.50 -> 2 px, q=.75 -> ~0.71 px. Structural demand is q weighted
            // by projected coverage through q=.90, then progressively guarded
            // from coverage relaxation through q=.99. Independent pixel-error
            // authority reaches one at q=.99.
            coarse_error_px: 16.0,
            fine_error_px: 0.25,
            quality_exponent: 1.0,
        }
    }
}

impl ProjectedErrorCurve {
    /// Returns the error limit for an interior quality value.
    ///
    /// Endpoints are handled by [`GaussianLodSettings::quality_endpoint`]; this
    /// function still behaves safely for unvalidated input by clamping it.
    fn error_limit_px(&self, quality: f32) -> f32 {
        let quality = finite_clamp01(quality);
        let exponent = finite_positive_or(self.quality_exponent, 1.0);
        let coarse = finite_positive_or(self.coarse_error_px, 16.0);
        let fine = finite_positive_or(self.fine_error_px, 0.25).min(coarse);
        let t = quality.powf(exponent);
        coarse * (fine / coarse).powf(t)
    }
}

/// Network, cache, and upload controls kept separate from visual selection policy.
#[derive(Component, Clone, Debug, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Component)]
#[serde(default)]
pub struct GaussianStreamingSettings {
    pub max_concurrent_requests: u32,
    /// Per-attempt HTTP timeout. Native and browser transports enforce it.
    pub request_timeout_seconds: f32,
    /// Number of attempts after the initial request. This control is operational.
    pub retry_limit: u32,
    /// Exponential HTTP retry/backoff base delay.
    pub retry_base_delay_seconds: f32,
    /// Enable the content-addressed native/browser encoded-page cache.
    pub persistent_cache: bool,
    /// Hard limit for one encoded page before transport or codec work.
    pub max_encoded_page_bytes: u64,
    /// Hard aggregate byte budget for the persistent encoded-page cache.
    pub max_compressed_cache_bytes: u64,
}

impl Default for GaussianStreamingSettings {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 8,
            request_timeout_seconds: 30.0,
            retry_limit: 3,
            retry_base_delay_seconds: 0.25,
            persistent_cache: false,
            max_encoded_page_bytes: 64 * 1024 * 1024,
            max_compressed_cache_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

impl GaussianStreamingSettings {
    pub fn validate(&self) -> Result<(), LodSettingsError> {
        if self.max_concurrent_requests == 0 {
            return Err(LodSettingsError::ZeroBudget(
                "streaming.max_concurrent_requests",
            ));
        }
        if self.max_concurrent_requests > MAX_STREAMING_CONCURRENT_REQUESTS {
            return Err(LodSettingsError::OutOfRange {
                field: "streaming.max_concurrent_requests",
                min: "1",
                max: "256",
            });
        }
        if self.max_compressed_cache_bytes == 0 {
            return Err(LodSettingsError::ZeroBudget(
                "streaming.max_compressed_cache_bytes",
            ));
        }
        if self.max_encoded_page_bytes == 0 {
            return Err(LodSettingsError::ZeroBudget(
                "streaming.max_encoded_page_bytes",
            ));
        }
        finite_positive(
            "streaming.request_timeout_seconds",
            self.request_timeout_seconds,
        )?;
        finite_non_negative(
            "streaming.retry_base_delay_seconds",
            self.retry_base_delay_seconds,
        )?;
        let timeout = Duration::try_from_secs_f32(self.request_timeout_seconds).map_err(|_| {
            LodSettingsError::DurationOutOfRange("streaming.request_timeout_seconds")
        })?;
        let retry_base =
            Duration::try_from_secs_f32(self.retry_base_delay_seconds).map_err(|_| {
                LodSettingsError::DurationOutOfRange("streaming.retry_base_delay_seconds")
            })?;
        if Instant::now().checked_add(timeout).is_none() {
            return Err(LodSettingsError::DurationOutOfRange(
                "streaming.request_timeout_seconds",
            ));
        }
        let retry_multiplier = 1_u32
            .checked_shl(self.retry_limit.min(31))
            .unwrap_or(u32::MAX);
        let retry_max = retry_base.saturating_mul(retry_multiplier);
        if Instant::now().checked_add(retry_max).is_none() {
            return Err(LodSettingsError::DurationOutOfRange(
                "streaming.retry_base_delay_seconds",
            ));
        }
        Ok(())
    }

    /// Returns the live per-page encoded transport/codec bound.
    pub fn effective_max_encoded_page_bytes(&self) -> u64 {
        self.max_encoded_page_bytes
    }
}

/// Limits work and memory independently. Every limit is explicit; zero is
/// rejected so it never ambiguously means either "disabled" or "unlimited".
#[derive(Clone, Copy, Debug, PartialEq, Reflect, Serialize, Deserialize)]
#[serde(default)]
pub struct LodBudgets {
    pub max_active_gaussians: u64,
    pub max_resident_gaussians: u64,
    pub max_resident_bytes: u64,
    pub max_resident_pages: u32,
    pub max_pending_requests: u32,
    pub max_requests_per_frame: u32,
    /// Maximum canonical/derived GPU atlas bytes materialized and enqueued in
    /// one bridge staging step. Larger complete cuts keep the previous
    /// drawable source/current cut and progress across multiple bounded steps
    /// before one atomic render handoff.
    pub max_gpu_upload_bytes_per_commit: u64,
    /// Maximum decoded page bytes admitted by the streaming runtime per frame.
    pub max_upload_bytes_per_frame: u64,
    /// Maximum Gaussian records decoded and validated by a caller-thread
    /// cooperative preprocessor in one application frame. Native worker-pool
    /// preprocessing is asynchronous and does not consume this frame budget.
    pub max_cooperative_preprocess_gaussians_per_frame: u32,
    pub max_traversal_nodes_per_view: u32,
}

impl Default for LodBudgets {
    fn default() -> Self {
        Self {
            max_active_gaussians: 2_000_000,
            max_resident_gaussians: 8_000_000,
            max_resident_bytes: 2 * 1024 * 1024 * 1024,
            max_resident_pages: 4096,
            max_pending_requests: 8192,
            max_requests_per_frame: 64,
            max_gpu_upload_bytes_per_commit: 512 * 1024 * 1024,
            max_upload_bytes_per_frame: 64 * 1024 * 1024,
            max_cooperative_preprocess_gaussians_per_frame:
                DEFAULT_COOPERATIVE_PREPROCESS_GAUSSIANS_PER_FRAME,
            max_traversal_nodes_per_view: 1_000_000,
        }
    }
}

/// Fully serializable, reflectable LoD policy for one cloud.
#[derive(Component, Clone, Debug, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Component)]
#[serde(default)]
pub struct GaussianLodSettings {
    /// Continuous quality in `[0, 1]`; zero is roots-only and one is original leaves.
    pub quality: f32,
    /// Camera policy for hierarchy selection. Freezing affects the selection
    /// view only; page loading and render publication continue.
    pub selection_mode: LodSelectionMode,
    #[reflect(ignore)]
    pub budgets: LodBudgets,
    /// Relative screen-space error that must be crossed before changing cuts.
    /// This prevents rapid cut oscillation under small camera movements.
    #[reflect(ignore)]
    pub hysteresis: f32,
    /// Allow conservative frustum filtering independently of coarsening quality.
    #[reflect(ignore)]
    pub frustum_culling: bool,
    /// Expands node bounds for conservative frustum testing, in world units.
    #[reflect(ignore)]
    pub frustum_margin: f32,
}

impl Default for GaussianLodSettings {
    fn default() -> Self {
        Self {
            quality: 1.0,
            selection_mode: LodSelectionMode::default(),
            budgets: LodBudgets::default(),
            hysteresis: 0.1,
            frustum_culling: true,
            frustum_margin: 0.0,
        }
    }
}

/// Registers the promoted LoD controls and status values used by inspectors.
#[derive(Default)]
pub struct GaussianLodSettingsPlugin;

impl Plugin for GaussianLodSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<GaussianLodSettings>()
            .register_type::<LodSelectionMode>()
            .register_type::<LodQualityTarget>()
            .register_type::<LodDegradation>();
        #[cfg(feature = "lod")]
        app.register_type::<super::lodge_settings::GaussianLodgeSettings>()
            .register_type::<super::lodge_settings::GaussianLodRepresentationKind>();
    }
}

impl GaussianLodSettings {
    /// Classifies the endpoint contract. Invalid values are made safe here, but
    /// callers loading external configuration should still call [`Self::validate`].
    pub fn quality_endpoint(&self) -> LodQualityEndpoint {
        self.quality_target().endpoint()
    }

    /// Resolves the compatibility quality slider into the runtime contract.
    ///
    /// This is the authoritative selection target. In particular, quality one
    /// is reported as [`LodQualityTarget::Original`], not as an approximation
    /// with an inferred quality score.
    pub fn quality_target(&self) -> LodQualityTarget {
        let quality = finite_clamp01(self.quality);
        if quality >= 1.0 {
            LodQualityTarget::Original
        } else if quality <= 0.0 {
            LodQualityTarget::Coarsest
        } else {
            LodQualityTarget::Balanced {
                detail_fraction: quality,
                max_error_px: ProjectedErrorCurve::default().error_limit_px(quality),
            }
        }
    }

    pub fn quality_clamped(&self) -> f32 {
        finite_clamp01(self.quality)
    }

    /// Device-safe active-list capacity. GPU index and indirect argument formats
    /// are `u32` even when manifest/source counts are `u64`.
    pub fn max_active_gaussians_u32(&self) -> u32 {
        self.budgets
            .max_active_gaussians
            .clamp(1, u64::from(u32::MAX)) as u32
    }

    /// Nominal projected-error part of the resolved target. Balanced selection
    /// also considers [`LodQualityTarget::detail_fraction`] and
    /// [`LodQualityTarget::error_authority`].
    pub fn screen_space_error_limit_px(&self) -> f32 {
        self.quality_target().error_limit_px()
    }

    /// Strict validation for serialized/cloud-provided configuration.
    pub fn validate(&self) -> Result<(), LodSettingsError> {
        finite_range("quality", self.quality, 0.0, 1.0)?;
        finite_range("hysteresis", self.hysteresis, 0.0, 1.0)?;
        finite_non_negative("frustum_margin", self.frustum_margin)?;
        for (field, value) in [
            (
                "budgets.max_active_gaussians",
                self.budgets.max_active_gaussians,
            ),
            (
                "budgets.max_resident_gaussians",
                self.budgets.max_resident_gaussians,
            ),
            (
                "budgets.max_resident_bytes",
                self.budgets.max_resident_bytes,
            ),
            (
                "budgets.max_gpu_upload_bytes_per_commit",
                self.budgets.max_gpu_upload_bytes_per_commit,
            ),
            (
                "budgets.max_upload_bytes_per_frame",
                self.budgets.max_upload_bytes_per_frame,
            ),
        ] {
            if value == 0 {
                return Err(LodSettingsError::ZeroBudget(field));
            }
        }
        for (field, value) in [
            (
                "budgets.max_resident_pages",
                self.budgets.max_resident_pages,
            ),
            (
                "budgets.max_pending_requests",
                self.budgets.max_pending_requests,
            ),
            (
                "budgets.max_requests_per_frame",
                self.budgets.max_requests_per_frame,
            ),
            (
                "budgets.max_cooperative_preprocess_gaussians_per_frame",
                self.budgets.max_cooperative_preprocess_gaussians_per_frame,
            ),
            (
                "budgets.max_traversal_nodes_per_view",
                self.budgets.max_traversal_nodes_per_view,
            ),
        ] {
            if value == 0 {
                return Err(LodSettingsError::ZeroBudget(field));
            }
        }
        Ok(())
    }
}

/// Why a runtime selection could not realize the requested quality.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Reflect, Serialize, Deserialize)]
pub enum LodDegradation {
    #[default]
    None,
    ActiveBudget,
    Residency,
    TraversalBudget,
    Multiple,
}

impl LodDegradation {
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::None, value) | (value, Self::None) => value,
            (left, right) if left == right => left,
            _ => Self::Multiple,
        }
    }
}

/// Observable per-view result; consumers must not infer full quality solely from
/// the requested slider value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect, Serialize, Deserialize)]
pub struct LodEffectiveStatus {
    /// Explicit structural-detail and projected-error target resolved from the
    /// configured presentation curve.
    pub requested_target: LodQualityTarget,
    /// Maximum selection error of the emitted frontier, in pixels.
    pub achieved_max_error_px: f32,
    /// Maximum effective target pressure over the emitted frontier. For a
    /// balanced target, let `S` be structural demand divided by the node
    /// threshold, `E` be projected error divided by the pixel limit, `a` be
    /// continuous error authority, and `C` be high-fidelity certificate
    /// pressure. Pressure is `max(min(S, E), a * E, C)`. Values at or below one
    /// satisfy the quality contract. A bounded overshoot can be intentional
    /// inside the hysteresis band; larger values expose quality, budget, or
    /// residency degradation.
    pub achieved_max_target_ratio: f32,
    pub degradation: LodDegradation,
    pub active_gaussians: u64,
    pub visited_nodes: u32,
    pub requested_pages: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LodSettingsError {
    NonFinite(&'static str),
    OutOfRange {
        field: &'static str,
        min: &'static str,
        max: &'static str,
    },
    NonPositive(&'static str),
    InvalidOrder {
        lower: &'static str,
        upper: &'static str,
    },
    ZeroBudget(&'static str),
    DurationOutOfRange(&'static str),
}

impl fmt::Display for LodSettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(field) => write!(f, "{field} must be finite"),
            Self::OutOfRange { field, min, max } => {
                write!(f, "{field} must be in [{min}, {max}]")
            }
            Self::NonPositive(field) => write!(f, "{field} must be greater than zero"),
            Self::InvalidOrder { lower, upper } => write!(f, "{lower} must be less than {upper}"),
            Self::ZeroBudget(field) => write!(f, "{field} must be non-zero"),
            Self::DurationOutOfRange(field) => {
                write!(f, "{field} does not fit a std::time::Duration")
            }
        }
    }
}

impl std::error::Error for LodSettingsError {}

fn finite_clamp01(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn finite_positive_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn finite_positive(field: &'static str, value: f32) -> Result<(), LodSettingsError> {
    if !value.is_finite() {
        Err(LodSettingsError::NonFinite(field))
    } else if value <= 0.0 {
        Err(LodSettingsError::NonPositive(field))
    } else {
        Ok(())
    }
}

fn finite_non_negative(field: &'static str, value: f32) -> Result<(), LodSettingsError> {
    if !value.is_finite() {
        Err(LodSettingsError::NonFinite(field))
    } else if value < 0.0 {
        Err(LodSettingsError::OutOfRange {
            field,
            min: "0",
            max: "+inf",
        })
    } else {
        Ok(())
    }
}

fn finite_range(
    field: &'static str,
    value: f32,
    min: f32,
    max: f32,
) -> Result<(), LodSettingsError> {
    if !value.is_finite() {
        Err(LodSettingsError::NonFinite(field))
    } else if !(min..=max).contains(&value) {
        Err(LodSettingsError::OutOfRange {
            field,
            min: "0",
            max: "1",
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use bevy::reflect::ReflectRef;

    use super::*;

    #[test]
    fn quality_endpoints_are_exact() {
        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.0;
        assert_eq!(settings.quality_endpoint(), LodQualityEndpoint::Coarsest);
        assert_eq!(settings.quality_target(), LodQualityTarget::Coarsest);
        assert!(settings.screen_space_error_limit_px().is_infinite());
        assert_eq!(settings.quality_target().error_authority(), None);
        assert_eq!(
            settings
                .quality_target()
                .effective_max_screen_space_error_px(),
            None
        );

        settings.quality = 0.5;
        assert_eq!(settings.quality_endpoint(), LodQualityEndpoint::Continuous);
        assert!(matches!(
            settings.quality_target(),
            LodQualityTarget::Balanced {
                detail_fraction: 0.5,
                max_error_px,
            } if max_error_px == ProjectedErrorCurve::default().error_limit_px(0.5)
        ));
        assert!(settings.screen_space_error_limit_px().is_finite());

        settings.quality = 1.0;
        assert_eq!(settings.quality_endpoint(), LodQualityEndpoint::Original);
        assert_eq!(settings.quality_target(), LodQualityTarget::Original);
        assert_eq!(settings.screen_space_error_limit_px(), 0.0);
        assert_eq!(settings.quality_target().error_authority(), None);
        assert_eq!(
            settings
                .quality_target()
                .effective_max_screen_space_error_px(),
            None
        );
    }

    #[test]
    fn projected_error_authority_and_effective_cap_are_numeric_and_monotonic() {
        let expected = [
            (0.25, (0.25_f32 / 0.99).powi(3)),
            (0.50, (0.50_f32 / 0.99).powi(3)),
            (0.75, (0.75_f32 / 0.99).powi(3)),
            (0.95, (0.95_f32 / 0.99).powi(3)),
            (0.99, 1.0),
        ];
        for (quality, expected_authority) in expected {
            let target = LodQualityTarget::Balanced {
                detail_fraction: quality,
                max_error_px: 2.0,
            };
            let authority = target.error_authority().unwrap();
            assert!((authority - expected_authority).abs() < 1e-6);
            assert!((projected_error_authority(quality) - expected_authority).abs() < 1e-6);
            assert!(
                (target.effective_max_screen_space_error_px().unwrap() - 2.0 / expected_authority)
                    .abs()
                    < 1e-5
            );
        }

        let authorities = (1..=99)
            .map(|step| projected_error_authority(step as f32 / 100.0))
            .collect::<Vec<_>>();
        assert!(authorities.windows(2).all(|pair| pair[1] >= pair[0]));

        let zero = LodQualityTarget::Balanced {
            detail_fraction: 0.0,
            max_error_px: 2.0,
        };
        assert_eq!(zero.error_authority(), None);
        assert_eq!(zero.effective_max_screen_space_error_px(), None);
        let invalid_limit = LodQualityTarget::Balanced {
            detail_fraction: 0.5,
            max_error_px: f32::NAN,
        };
        assert_eq!(invalid_limit.effective_max_screen_space_error_px(), None);
    }

    #[test]
    fn balanced_quality_mapping_is_exact_and_monotonic() {
        let mut settings = GaussianLodSettings::default();
        let expected = [
            (0.25, 5.656_854_f32),
            (0.50, 2.0),
            (0.75, std::f32::consts::FRAC_1_SQRT_2),
        ];
        for (quality, expected_error) in expected {
            settings.quality = quality;
            let LodQualityTarget::Balanced {
                detail_fraction,
                max_error_px,
            } = settings.quality_target()
            else {
                panic!("interior quality must resolve to a balanced target");
            };
            assert_eq!(detail_fraction, quality);
            assert!((max_error_px - expected_error).abs() < 1e-5);
        }

        let targets = (1..10)
            .map(|step| {
                settings.quality = step as f32 / 10.0;
                settings.quality_target()
            })
            .collect::<Vec<_>>();
        assert!(targets.windows(2).all(|pair| {
            pair[1].detail_fraction() > pair[0].detail_fraction()
                && pair[1].max_screen_space_error_px() < pair[0].max_screen_space_error_px()
        }));
    }

    #[test]
    fn lower_quality_pressure_continuously_bounds_the_structural_shortcut() {
        let target = LodQualityTarget::Balanced {
            detail_fraction: 0.5,
            max_error_px: 2.0,
        };
        let authority = (0.5_f32 / 0.99).powi(3);
        let high_error_pressure = target.node_pressure(0.25, 20.0, 0.25, 1.0, false);
        assert!((high_error_pressure - authority * 10.0).abs() < 1e-6);
        let extreme_error_pressure = target.node_pressure(0.25, 100.0, 0.25, 1.0, false);
        assert!((extreme_error_pressure - authority * 50.0).abs() < 1e-6);
        assert_eq!(target.node_pressure(0.25, 1.0, 1.0, 1.0, false), 0.5);
        assert_eq!(
            LodQualityTarget::Original.node_pressure(1.0, 0.0, 1.0, 1.0, true),
            0.0
        );
        assert_eq!(
            LodQualityTarget::Original.node_pressure(1.0, 0.0, 1.0, 1.0, false),
            f32::MAX
        );
    }

    #[test]
    fn high_quality_guarded_pressure_caps_structurally_accepted_visual_error() {
        let quality = 0.95;
        let structural_guard = high_quality_fidelity_guard(quality);
        assert!((structural_guard - 0.582_990_4).abs() < 1e-6);
        let authority = (quality / 0.99).powi(3);
        assert!((projected_error_authority(quality) - authority).abs() < 1e-6);

        let target = LodQualityTarget::Balanced {
            detail_fraction: quality,
            max_error_px: 1.0,
        };
        let structural_only = target.structural_detail_demand(0.0) / 0.6;
        assert!(structural_only < 1.0);

        let huge_error_pressure = target.node_pressure(0.6, 128.0, 0.0, 1.0, false);
        assert!((huge_error_pressure - authority * 128.0).abs() < 1e-4);
        assert!(huge_error_pressure > 1.0);

        let guarded_error_limit = 1.0 / authority;
        let boundary = target.node_pressure(0.6, guarded_error_limit, 0.0, 1.0, false);
        assert!((boundary - 1.0).abs() < 1e-6);
        assert!(target.node_pressure(0.6, guarded_error_limit * 1.01, 0.0, 1.0, false) > 1.0);
    }

    #[test]
    fn quality_point_ninety_nine_makes_the_pixel_limit_authoritative() {
        let target = LodQualityTarget::Balanced {
            detail_fraction: 0.99,
            max_error_px: 0.25,
        };
        assert_eq!(high_quality_fidelity_guard(0.99), 1.0);
        for (error_px, expected_pressure) in [(0.125, 0.99), (0.25, 1.0), (8.0, 32.0)] {
            assert_eq!(
                target.node_pressure(1.0, error_px, 0.0, 1.0, false),
                expected_pressure
            );
        }
        assert_eq!(target.node_pressure(1.0, 0.0, 0.0, 0.99, false), 1.0);
        assert!(target.node_pressure(1.0, 0.0, 0.0, 0.989, false) > 1.0);
    }

    #[test]
    fn high_quality_certificate_guards_uncertified_representatives() {
        let quality = 0.95;
        let certificate_guard = high_quality_certificate_guard(quality);
        let certificate_demand = high_quality_certificate_demand(quality, 0.0);
        assert_eq!(certificate_guard, 1.0);
        assert_eq!(certificate_demand, quality);
        let target = LodQualityTarget::Balanced {
            detail_fraction: quality,
            max_error_px: 1.0,
        };

        // Structural and pixel-error paths both accept this node. Its
        // certificate alone must still force refinement at quality .95.
        assert!(target.node_pressure(1.0, 0.0, 0.0, 0.0, false) > 1.0);
        assert_eq!(
            target.node_pressure(1.0, 0.0, 0.0, certificate_demand, false),
            1.0
        );
        assert!(target.node_pressure(1.0, 0.0, 0.0, certificate_demand * 0.5, false) > 1.0);
        assert!(target.node_pressure(1.0, 0.0, 0.0, 1.0, false) <= 1.0);
    }

    #[test]
    fn certificate_authority_is_high_quality_gated_monotonic_and_coverage_aware() {
        assert_eq!(high_quality_certificate_guard(0.475), 0.0);
        assert_eq!(high_quality_certificate_guard(0.90), 0.0);
        assert!((high_quality_certificate_guard(0.925) - 0.5).abs() < 1e-6);
        assert_eq!(high_quality_certificate_guard(0.95), 1.0);
        assert_eq!(high_quality_certificate_guard(0.99), 1.0);

        let quality = 0.925_f32;
        let normalized = quality / 0.95;
        let coverage_authority = normalized.powi(3);
        let guard = high_quality_certificate_guard(quality);
        let base = quality * normalized;
        for coverage in [0.0_f32, 0.2, 0.4, 0.8, 1.0] {
            let expected = guard * base * (coverage + (1.0 - coverage) * coverage_authority);
            assert!((high_quality_certificate_demand(quality, coverage) - expected).abs() < 1e-6);
        }
        let by_coverage = [0.0, 0.2, 0.4, 0.8, 1.0]
            .map(|coverage| high_quality_certificate_demand(quality, coverage));
        assert!(by_coverage.windows(2).all(|pair| pair[1] > pair[0]));
        for coverage in [0.0, 0.25, 0.75, 1.0] {
            assert_eq!(high_quality_certificate_demand(0.90, coverage), 0.0);
            assert_eq!(high_quality_certificate_demand(0.95, coverage), 0.95);
            assert_eq!(high_quality_certificate_demand(0.99, coverage), 0.99);
            let by_quality = (0..=99)
                .map(|step| high_quality_certificate_demand(step as f32 / 100.0, coverage))
                .collect::<Vec<_>>();
            assert!(by_quality.windows(2).all(|pair| pair[1] >= pair[0]));
        }
        assert!(high_quality_fidelity_guard(0.95) < 1.0);
    }

    #[test]
    fn certificate_pressure_preserves_legacy_values_then_fails_closed_at_point_ninety_five() {
        assert_eq!(
            high_fidelity_certificate_pressure(0.5, 0.25, 0.0, false),
            0.0
        );
        assert_eq!(
            high_fidelity_certificate_pressure(0.5, 0.25, f32::NAN, false),
            0.0
        );
        assert_eq!(
            high_fidelity_certificate_pressure(
                0.5,
                0.25,
                MIN_QUANTIZED_HIGH_FIDELITY_CERTIFICATE,
                false,
            ),
            0.0
        );
        assert_eq!(
            high_fidelity_certificate_pressure(0.95, 0.0, 0.0, false),
            f32::MAX
        );
        assert_eq!(
            high_fidelity_certificate_pressure(0.95, 1.0, f32::NAN, false),
            f32::MAX
        );
        assert_eq!(
            high_fidelity_certificate_pressure(0.95, 0.0, 0.0, true),
            0.0
        );
        assert_eq!(
            high_fidelity_certificate_pressure(0.95, 0.0, 0.95, false),
            1.0
        );

        let certificate = 0.5;
        assert_eq!(high_quality_certificate_demand(0.5, 0.25), 0.0);
        assert_eq!(
            high_fidelity_certificate_pressure(0.5, 0.25, certificate, false),
            0.0
        );
    }

    #[test]
    fn revised_authorities_preserve_monotonic_pressure_and_categorical_endpoints() {
        for coverage in [0.0_f32, 0.1, 0.5, 1.0] {
            let mut previous = 0.0_f32;
            for step in 0..=99 {
                let quality = step as f32 / 100.0;
                let target = LodQualityTarget::Balanced {
                    detail_fraction: quality,
                    max_error_px: 1.0,
                };
                let pressure =
                    target.node_pressure_with_error_limit(0.6, 8.0, 1.0, coverage, 0.5, false);
                assert!(
                    pressure >= previous,
                    "pressure regressed at q={quality} coverage={coverage}: {pressure} < {previous}"
                );
                previous = pressure;
            }
        }

        assert_eq!(projected_error_authority(0.0), 0.0);
        assert_eq!(projected_error_authority(0.99), 1.0);
        assert_eq!(high_quality_certificate_guard(0.90), 0.0);
        assert_eq!(high_quality_certificate_guard(0.95), 1.0);
        assert_eq!(
            LodQualityTarget::Coarsest.endpoint(),
            LodQualityEndpoint::Coarsest
        );
        assert_eq!(
            LodQualityTarget::Original.endpoint(),
            LodQualityEndpoint::Original
        );
    }

    #[test]
    fn structural_guard_stays_smooth_while_error_authority_spans_the_slider() {
        for quality in [0.0, 0.25, 0.5, 0.9] {
            let target = LodQualityTarget::Balanced {
                detail_fraction: quality,
                max_error_px: 1.0,
            };
            for coverage in [0.0, 0.25, 0.75, 1.0] {
                assert_eq!(
                    target.structural_detail_demand(coverage).to_bits(),
                    (quality * coverage).to_bits(),
                    "q={quality} coverage={coverage}"
                );
                for (threshold, error, limit) in
                    [(0.25, 4.0, 2.0), (0.75, 128.0, 1.0), (1.0, 0.25, 2.0)]
                {
                    let error_pressure = error_ratio(error, limit);
                    let expected = detail_ratio(quality * coverage, threshold)
                        .min(error_pressure)
                        .max(projected_error_authority(quality) * error_pressure);
                    let pressure = target.node_pressure_with_error_limit(
                        threshold, error, limit, coverage, 0.0, false,
                    );
                    assert!(
                        (pressure - expected).abs() <= f32::EPSILON * expected.abs().max(1.0),
                        "q={quality} coverage={coverage} pressure={pressure} expected={expected}"
                    );
                }
            }
        }

        for coverage in [0.0, 0.25, 0.75, 1.0] {
            let demands = [0.9, 0.91, 0.95, 0.98, 0.99].map(|quality| {
                let target = LodQualityTarget::Balanced {
                    detail_fraction: quality,
                    max_error_px: 1.0,
                };
                let demand = target.structural_detail_demand(coverage);
                assert!((quality * coverage..=quality).contains(&demand));
                demand
            });
            assert!(demands.windows(2).all(|pair| pair[1] > pair[0]));
        }

        let near_exact = LodQualityTarget::Balanced {
            detail_fraction: 0.99,
            max_error_px: 1.0,
        };
        for coverage in [0.0, 0.01, 0.25, 0.75, 1.0] {
            assert_eq!(near_exact.structural_detail_demand(coverage), 0.99);
        }
    }

    #[test]
    fn defaults_are_a_neutral_screen_space_contract() {
        let settings = GaussianLodSettings::default();
        let curve = ProjectedErrorCurve::default();
        assert_eq!(settings.quality, 1.0);
        assert_eq!(curve.error_limit_px(0.5), 2.0);
        assert_eq!(settings.quality_target(), LodQualityTarget::Original);
    }

    #[test]
    fn rejects_nan_infinity_and_invalid_ranges() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            let mut settings = GaussianLodSettings::default();
            settings.quality = invalid;
            assert!(settings.validate().is_err(), "accepted quality {invalid}");
        }

        let mut settings = GaussianLodSettings::default();
        settings.budgets.max_active_gaussians = 0;
        assert!(matches!(
            settings.validate(),
            Err(LodSettingsError::ZeroBudget(_))
        ));

        let mut settings = GaussianLodSettings::default();
        settings.budgets.max_gpu_upload_bytes_per_commit = 0;
        assert_eq!(
            settings.validate(),
            Err(LodSettingsError::ZeroBudget(
                "budgets.max_gpu_upload_bytes_per_commit"
            ))
        );

        let mut settings = GaussianLodSettings::default();
        settings
            .budgets
            .max_cooperative_preprocess_gaussians_per_frame = 0;
        assert_eq!(
            settings.validate(),
            Err(LodSettingsError::ZeroBudget(
                "budgets.max_cooperative_preprocess_gaussians_per_frame"
            ))
        );
    }

    #[test]
    fn invalid_runtime_values_fail_safe_to_original_quality() {
        let mut settings = GaussianLodSettings::default();
        settings.quality = f32::NAN;
        assert_eq!(settings.quality_endpoint(), LodQualityEndpoint::Original);
        assert_eq!(settings.screen_space_error_limit_px(), 0.0);
    }

    #[test]
    fn partial_cloud_config_uses_stable_defaults_and_round_trips() {
        let settings: GaussianLodSettings =
            serde_json::from_str(r#"{"quality":0.25,"budgets":{"max_active_gaussians":12345}}"#)
                .unwrap();
        assert_eq!(settings.quality, 0.25);
        assert_eq!(settings.budgets.max_active_gaussians, 12_345);
        assert_eq!(
            settings.budgets.max_resident_pages,
            LodBudgets::default().max_resident_pages
        );
        assert_eq!(
            settings
                .budgets
                .max_cooperative_preprocess_gaussians_per_frame,
            DEFAULT_COOPERATIVE_PREPROCESS_GAUSSIANS_PER_FRAME
        );
        settings.validate().unwrap();

        let encoded = serde_json::to_string(&settings).unwrap();
        let decoded: GaussianLodSettings = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, settings);
    }

    #[test]
    fn selection_mode_is_serde_compatible_and_defaults_dynamic() {
        let dynamic: GaussianLodSettings = serde_json::from_str(r#"{"quality":0.5}"#).unwrap();
        assert_eq!(dynamic.selection_mode, LodSelectionMode::Dynamic);

        let frozen: GaussianLodSettings =
            serde_json::from_str(r#"{"quality":0.5,"selection_mode":"Frozen"}"#).unwrap();
        assert_eq!(frozen.selection_mode, LodSelectionMode::Frozen);
        let encoded = serde_json::to_string(&frozen).unwrap();
        let decoded: GaussianLodSettings = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, frozen);
    }

    #[test]
    fn default_reflection_exposes_only_primary_lod_controls() {
        let settings = GaussianLodSettings::default();
        let ReflectRef::Struct(reflected) = settings.reflect_ref() else {
            panic!("GaussianLodSettings must reflect as a struct");
        };
        let fields = (0..reflected.field_len())
            .filter_map(|index| reflected.name_at(index))
            .collect::<Vec<_>>();
        assert_eq!(fields, ["quality", "selection_mode"]);
    }

    #[test]
    fn streaming_configuration_is_independently_validated() {
        let mut settings = GaussianStreamingSettings::default();
        settings.validate().unwrap();
        settings.max_concurrent_requests = 0;
        assert!(matches!(
            settings.validate(),
            Err(LodSettingsError::ZeroBudget(_))
        ));
    }

    #[test]
    fn defaults_only_enable_implemented_lod_capabilities() {
        let lod = GaussianLodSettings::default();
        lod.validate().unwrap();

        let streaming = GaussianStreamingSettings::default();
        assert_eq!(streaming.request_timeout_seconds, 30.0);
        assert_eq!(streaming.retry_base_delay_seconds, 0.25);
        assert!(!streaming.persistent_cache);
        streaming.validate().unwrap();
    }

    #[test]
    fn partial_serde_configs_preserve_neutral_capability_defaults() {
        let lod: GaussianLodSettings = serde_json::from_str(r#"{"quality":0.5}"#).unwrap();
        lod.validate().unwrap();

        let streaming: GaussianStreamingSettings =
            serde_json::from_str(r#"{"retry_limit":7}"#).unwrap();
        assert_eq!(streaming.retry_limit, 7);
        assert_eq!(streaming.request_timeout_seconds, 30.0);
        assert_eq!(streaming.retry_base_delay_seconds, 0.25);
        assert!(!streaming.persistent_cache);
        streaming.validate().unwrap();

        let encoded = serde_json::to_string(&streaming).unwrap();
        let decoded: GaussianStreamingSettings = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, streaming);

        let cached_streaming: GaussianStreamingSettings =
            serde_json::from_str(r#"{"persistent_cache":true}"#).unwrap();
        cached_streaming.validate().unwrap();
    }

    #[test]
    fn operational_streaming_controls_validate() {
        let mut settings = GaussianStreamingSettings {
            request_timeout_seconds: 0.0,
            ..Default::default()
        };
        let error = settings.validate().unwrap_err();
        assert_eq!(
            error,
            LodSettingsError::NonPositive("streaming.request_timeout_seconds")
        );

        settings.request_timeout_seconds = 1.0;
        settings.retry_base_delay_seconds = 0.25;
        settings.validate().unwrap();

        settings.max_concurrent_requests = MAX_STREAMING_CONCURRENT_REQUESTS + 1;
        assert_eq!(
            settings.validate(),
            Err(LodSettingsError::OutOfRange {
                field: "streaming.max_concurrent_requests",
                min: "1",
                max: "256",
            })
        );
        settings.max_concurrent_requests = MAX_STREAMING_CONCURRENT_REQUESTS;

        settings.request_timeout_seconds = f32::MAX;
        assert_eq!(
            settings.validate(),
            Err(LodSettingsError::DurationOutOfRange(
                "streaming.request_timeout_seconds"
            ))
        );
        settings.request_timeout_seconds = 1.0e19_f32;
        assert_eq!(
            settings.validate(),
            Err(LodSettingsError::DurationOutOfRange(
                "streaming.request_timeout_seconds"
            ))
        );
        settings.request_timeout_seconds = 1.0;
        settings.retry_base_delay_seconds = f32::MAX;
        assert_eq!(
            settings.validate(),
            Err(LodSettingsError::DurationOutOfRange(
                "streaming.retry_base_delay_seconds"
            ))
        );
        settings.retry_base_delay_seconds = 1.0e19_f32;
        assert_eq!(
            settings.validate(),
            Err(LodSettingsError::DurationOutOfRange(
                "streaming.retry_base_delay_seconds"
            ))
        );

        settings.retry_base_delay_seconds = 0.0;
        settings.persistent_cache = true;
        settings.validate().unwrap();
    }

    #[test]
    fn retry_limit_remains_operational_and_compressed_budget_is_reserved() {
        for retry_limit in [0, 1, 7, u32::MAX] {
            let settings = GaussianStreamingSettings {
                retry_limit,
                max_encoded_page_bytes: 8 * 1024 * 1024,
                max_compressed_cache_bytes: 1024,
                ..Default::default()
            };
            settings.validate().unwrap();
            assert_eq!(settings.effective_max_encoded_page_bytes(), 8 * 1024 * 1024);
        }
    }
}
