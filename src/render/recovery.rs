//! Recovery policy and observability for Gaussian GPU resources.
//!
//! Bevy recreates its render device after a [`RenderErrorPolicy::Recover`]
//! response and reruns [`RenderStartup`]. Gaussian render resources register
//! through `init_gpu_resource`, so a successful recovery drops every bind
//! group, pipeline specialization, compaction buffer, radix buffer, and atlas
//! generation proof that belonged to the old device.

use std::sync::{Arc, Mutex};

use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup,
        error_handler::{ErrorType, RenderError, RenderErrorHandler, RenderErrorPolicy},
        renderer::{RenderAdapterInfo, RenderDevice},
        settings::{
            Backends, Dx12Compiler, Gles3MinorVersion, InstanceFlags, MemoryHints, PowerPreference,
            RenderCreation, WgpuFeatures, WgpuLimits, WgpuSettings, WgpuSettingsPriority,
        },
    },
};

/// Adapter selection applied to a replacement render device.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Reflect)]
pub enum GaussianRecoveryAdapterPolicy {
    /// Try the adapter that was active before the loss. If repeated recovery
    /// attempts fail, permit any compatible high-performance adapter.
    #[default]
    SameThenAny,
    /// Immediately permit any adapter compatible with the configured limits.
    AnyCompatible,
    /// Require a fallback/software adapter. This is mainly useful for fault
    /// isolation and CI rather than interactive rendering.
    ForceFallback,
}

/// Application-wide device-loss policy installed by the Gaussian plugin.
///
/// The settings are deliberately a resource rather than part of a cloud: a
/// wgpu device and Bevy render sub-app are shared by every cloud. Insert a
/// customized value before adding [`crate::GaussianSplattingPlugin`] to retain
/// backend, feature, or limit constraints used by the application.
#[derive(Resource, Clone)]
pub struct GaussianRenderRecoverySettings {
    /// Install the Gaussian handler as a composable wrapper around the handler
    /// already registered by the application. Set this to `false` when the
    /// application owns recovery and only wants Gaussian recovery status.
    pub install_error_handler: bool,
    /// Attempt recovery for device-loss errors. Validation, OOM, and internal
    /// API errors are delegated to the application's previous handler.
    pub enabled: bool,
    /// Maximum consecutive replacement-device requests before failing closed.
    pub max_attempts: u32,
    /// Successful render frames required before a recovered device clears the
    /// consecutive-attempt budget. This prevents a device that repeatedly
    /// reaches `RenderStartup` and immediately fails from retrying forever.
    pub healthy_frames_before_reset: u32,
    pub adapter_policy: GaussianRecoveryAdapterPolicy,
    /// Base settings used when switching adapters. The first same-adapter
    /// attempt is derived from the actual active device features and limits.
    pub wgpu: WgpuSettings,
}

impl Default for GaussianRenderRecoverySettings {
    fn default() -> Self {
        Self {
            install_error_handler: true,
            enabled: true,
            max_attempts: 3,
            healthy_frames_before_reset: 60,
            adapter_policy: GaussianRecoveryAdapterPolicy::SameThenAny,
            wgpu: deterministic_wgpu_settings(),
        }
    }
}

/// Constructs a portable recovery baseline without consulting process
/// environment variables or probing the host filesystem. Applications can
/// still replace this complete profile before installing the plugin.
fn deterministic_wgpu_settings() -> WgpuSettings {
    let backends = if cfg!(all(target_arch = "wasm32", feature = "webgl2")) {
        Backends::GL
    } else if cfg!(target_arch = "wasm32") {
        Backends::BROWSER_WEBGPU
    } else {
        Backends::all()
    };
    let limits = if cfg!(all(target_arch = "wasm32", feature = "webgl2")) {
        WgpuLimits::downlevel_webgl2_defaults()
    } else {
        WgpuLimits::default()
    };
    WgpuSettings {
        device_label: None,
        backends: Some(backends),
        power_preference: PowerPreference::HighPerformance,
        priority: WgpuSettingsPriority::Functionality,
        features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        disabled_features: None,
        limits,
        constrained_limits: None,
        dx12_shader_compiler: Dx12Compiler::Fxc,
        gles3_minor_version: Gles3MinorVersion::Automatic,
        instance_flags: InstanceFlags::default(),
        memory_hints: MemoryHints::default(),
        instance_memory_budget_thresholds: Default::default(),
        force_fallback_adapter: false,
        adapter_name: None,
    }
}

impl GaussianRenderRecoverySettings {
    pub fn validate(&self) -> Result<(), GaussianRenderRecoveryError> {
        if self.enabled && self.max_attempts == 0 {
            Err(GaussianRenderRecoveryError::ZeroAttempts)
        } else if self.enabled && self.healthy_frames_before_reset == 0 {
            Err(GaussianRenderRecoveryError::ZeroHealthyFrames)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GaussianRenderRecoveryPhase {
    #[default]
    Initializing,
    Ready,
    Recovering,
    Exhausted,
    Stopped,
}

/// Immutable observation returned from [`GaussianRenderRecoveryStatus`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GaussianRenderRecoverySnapshot {
    pub phase: GaussianRenderRecoveryPhase,
    pub device_generation: u64,
    pub consecutive_attempts: u32,
    pub healthy_frames: u32,
    pub total_device_losses: u64,
    pub adapter_name: Option<String>,
    pub backend: Option<String>,
    pub last_error: Option<String>,
}

/// Thread-safe recovery status shared by the main and render worlds.
#[derive(Resource, Clone, Default)]
pub struct GaussianRenderRecoveryStatus(Arc<Mutex<GaussianRenderRecoverySnapshot>>);

impl GaussianRenderRecoveryStatus {
    pub fn snapshot(&self) -> GaussianRenderRecoverySnapshot {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn begin_recovery(&self, description: &str) -> u32 {
        let mut status = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        status.phase = GaussianRenderRecoveryPhase::Recovering;
        status.consecutive_attempts = status.consecutive_attempts.saturating_add(1);
        status.total_device_losses = status.total_device_losses.saturating_add(1);
        status.last_error = Some(description.to_owned());
        status.consecutive_attempts
    }

    fn mark_ready(&self, adapter: Option<&RenderAdapterInfo>) {
        let mut status = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        status.phase = GaussianRenderRecoveryPhase::Ready;
        status.device_generation = status.device_generation.saturating_add(1);
        status.healthy_frames = 0;
        status.last_error = None;
        if let Some(adapter) = adapter {
            status.adapter_name = Some(adapter.name.clone());
            status.backend = Some(format!("{:?}", adapter.backend));
        }
    }

    fn observe_healthy_frame(&self, required_frames: u32) {
        let mut status = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if status.phase != GaussianRenderRecoveryPhase::Ready {
            return;
        }
        status.healthy_frames = status.healthy_frames.saturating_add(1);
        if status.healthy_frames >= required_frames {
            status.consecutive_attempts = 0;
        }
    }

    fn stop(&self, phase: GaussianRenderRecoveryPhase, description: &str) {
        let mut status = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        status.phase = phase;
        status.last_error = Some(description.to_owned());
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GaussianRenderRecoveryError {
    ZeroAttempts,
    ZeroHealthyFrames,
}

impl std::fmt::Display for GaussianRenderRecoveryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroAttempts => formatter.write_str(
                "Gaussian render recovery max_attempts must be non-zero when recovery is enabled",
            ),
            Self::ZeroHealthyFrames => formatter.write_str(
                "Gaussian render recovery healthy_frames_before_reset must be non-zero when recovery is enabled",
            ),
        }
    }
}

impl std::error::Error for GaussianRenderRecoveryError {}

/// Installs automatic device-loss recovery and status observation.
#[derive(Default)]
pub struct GaussianRenderRecoveryPlugin;

type RenderErrorHandlerFn =
    for<'a> fn(&'a RenderError, &'a mut World, &'a mut World) -> RenderErrorPolicy;

#[derive(Resource, Clone, Copy)]
struct GaussianPreviousRenderErrorHandler(RenderErrorHandlerFn);

#[derive(Resource, Clone)]
struct GaussianRenderRecoveryDeviceProfile(Arc<Mutex<Option<WgpuSettings>>>);

impl GaussianRenderRecoveryDeviceProfile {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }

    fn capture(&self, device: &RenderDevice, adapter: Option<&RenderAdapterInfo>) {
        let Some(adapter) = adapter else {
            return;
        };
        let mut settings = deterministic_wgpu_settings();
        settings.backends = Some(adapter.backend.into());
        settings.features = device.features();
        settings.disabled_features = None;
        settings.limits = device.limits();
        settings.constrained_limits = None;
        settings.force_fallback_adapter = false;
        settings.adapter_name = Some(adapter.name.clone());
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(settings);
    }

    fn snapshot(&self) -> Option<WgpuSettings> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl Plugin for GaussianRenderRecoveryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GaussianRenderRecoverySettings>();
        let settings = app
            .world()
            .resource::<GaussianRenderRecoverySettings>()
            .clone();
        let status = app
            .world()
            .get_resource::<GaussianRenderRecoveryStatus>()
            .cloned()
            .unwrap_or_default();
        let device_profile = GaussianRenderRecoveryDeviceProfile::new();
        let previous_handler = app
            .world()
            .get_resource::<RenderErrorHandler>()
            .map_or_else(|| RenderErrorHandler::default().0, |handler| handler.0);
        app.insert_resource(status.clone());
        app.insert_resource(GaussianPreviousRenderErrorHandler(previous_handler));
        app.insert_resource(device_profile.clone());
        if settings.install_error_handler {
            app.insert_resource(RenderErrorHandler(gaussian_render_error_handler));
        }
        app.register_type::<GaussianRecoveryAdapterPolicy>();

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(status)
                .insert_resource(settings)
                .insert_resource(device_profile)
                .add_systems(
                    RenderStartup,
                    mark_gaussian_render_device_ready.ambiguous_with_all(),
                )
                .add_systems(Render, mark_gaussian_render_device_healthy);
        }
    }
}

fn mark_gaussian_render_device_ready(
    status: Res<GaussianRenderRecoveryStatus>,
    adapter: Option<Res<RenderAdapterInfo>>,
    device: Option<Res<RenderDevice>>,
    profile: Res<GaussianRenderRecoveryDeviceProfile>,
) {
    if let Some(device) = device {
        profile.capture(&device, adapter.as_deref());
    }
    status.mark_ready(adapter.as_deref());
}

fn mark_gaussian_render_device_healthy(
    status: Res<GaussianRenderRecoveryStatus>,
    settings: Res<GaussianRenderRecoverySettings>,
) {
    status.observe_healthy_frame(settings.healthy_frames_before_reset);
}

fn delegate_render_error(
    error: &RenderError,
    main_world: &mut World,
    render_world: &mut World,
) -> RenderErrorPolicy {
    let handler = main_world
        .get_resource::<GaussianPreviousRenderErrorHandler>()
        .map_or_else(|| RenderErrorHandler::default().0, |handler| handler.0);
    handler(error, main_world, render_world)
}

fn gaussian_render_error_handler(
    error: &RenderError,
    main_world: &mut World,
    render_world: &mut World,
) -> RenderErrorPolicy {
    let settings = main_world
        .get_resource::<GaussianRenderRecoverySettings>()
        .cloned()
        .unwrap_or_default();
    let status = main_world
        .get_resource::<GaussianRenderRecoveryStatus>()
        .cloned()
        .unwrap_or_default();

    if error.ty != ErrorType::DeviceLost || !settings.enabled {
        let policy = delegate_render_error(error, main_world, render_world);
        if matches!(policy, RenderErrorPolicy::StopRendering) {
            status.stop(GaussianRenderRecoveryPhase::Stopped, &error.description);
        }
        return policy;
    }

    if let Err(validation_error) = settings.validate() {
        status.stop(
            GaussianRenderRecoveryPhase::Exhausted,
            &validation_error.to_string(),
        );
        return delegate_render_error(error, main_world, render_world);
    }

    let attempt = status.begin_recovery(&error.description);
    if attempt > settings.max_attempts {
        status.stop(GaussianRenderRecoveryPhase::Exhausted, &error.description);
        return delegate_render_error(error, main_world, render_world);
    }

    let previous_adapter = main_world
        .get_resource::<RenderAdapterInfo>()
        .map(|adapter| adapter.name.clone());
    let mut wgpu =
        if attempt == 1 && settings.adapter_policy == GaussianRecoveryAdapterPolicy::SameThenAny {
            main_world
                .get_resource::<GaussianRenderRecoveryDeviceProfile>()
                .and_then(GaussianRenderRecoveryDeviceProfile::snapshot)
                .unwrap_or_else(|| settings.wgpu.clone())
        } else {
            settings.wgpu.clone()
        };
    match settings.adapter_policy {
        GaussianRecoveryAdapterPolicy::SameThenAny => {
            wgpu.force_fallback_adapter = false;
            wgpu.adapter_name = if attempt == 1 { previous_adapter } else { None };
        }
        GaussianRecoveryAdapterPolicy::AnyCompatible => {
            wgpu.force_fallback_adapter = false;
            wgpu.adapter_name = None;
        }
        GaussianRecoveryAdapterPolicy::ForceFallback => {
            wgpu.force_fallback_adapter = true;
            wgpu.adapter_name = None;
        }
    }

    RenderErrorPolicy::Recover(RenderCreation::Automatic(Box::new(wgpu)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_reject_zero_recovery_attempts() {
        let settings = GaussianRenderRecoverySettings {
            max_attempts: 0,
            ..default()
        };
        assert_eq!(
            settings.validate(),
            Err(GaussianRenderRecoveryError::ZeroAttempts)
        );

        let settings = GaussianRenderRecoverySettings {
            healthy_frames_before_reset: 0,
            ..default()
        };
        assert_eq!(
            settings.validate(),
            Err(GaussianRenderRecoveryError::ZeroHealthyFrames)
        );
    }

    #[test]
    fn recovery_status_tracks_loss_and_new_device_generation() {
        let status = GaussianRenderRecoveryStatus::default();
        status.mark_ready(None);
        assert_eq!(status.snapshot().device_generation, 1);
        assert_eq!(status.begin_recovery("injected loss"), 1);
        assert_eq!(
            status.snapshot().phase,
            GaussianRenderRecoveryPhase::Recovering
        );
        status.mark_ready(None);
        let snapshot = status.snapshot();
        assert_eq!(snapshot.phase, GaussianRenderRecoveryPhase::Ready);
        assert_eq!(snapshot.device_generation, 2);
        assert_eq!(snapshot.consecutive_attempts, 1);
        assert_eq!(snapshot.healthy_frames, 0);
        status.observe_healthy_frame(2);
        assert_eq!(status.snapshot().consecutive_attempts, 1);
        status.observe_healthy_frame(2);
        let snapshot = status.snapshot();
        assert_eq!(snapshot.consecutive_attempts, 0);
        assert_eq!(snapshot.healthy_frames, 2);
        assert_eq!(snapshot.total_device_losses, 1);
    }

    #[derive(Resource, Default)]
    struct DelegatedErrorCount(u32);

    fn counting_render_error_handler(
        _error: &RenderError,
        main_world: &mut World,
        _render_world: &mut World,
    ) -> RenderErrorPolicy {
        main_world.resource_mut::<DelegatedErrorCount>().0 += 1;
        RenderErrorPolicy::Ignore
    }

    #[test]
    fn device_loss_requests_recovery_and_other_errors_fail_closed() {
        let mut main_world = World::new();
        main_world.init_resource::<Messages<AppExit>>();
        main_world.insert_resource(GaussianRenderRecoverySettings {
            adapter_policy: GaussianRecoveryAdapterPolicy::AnyCompatible,
            ..default()
        });
        main_world.init_resource::<GaussianRenderRecoveryStatus>();
        let mut render_world = World::new();

        let loss = RenderError {
            ty: ErrorType::DeviceLost,
            description: "injected".to_owned(),
            source: None,
        };
        assert!(matches!(
            gaussian_render_error_handler(&loss, &mut main_world, &mut render_world),
            RenderErrorPolicy::Recover(_)
        ));

        let validation = RenderError {
            ty: ErrorType::Validation,
            description: "bad binding".to_owned(),
            source: None,
        };
        assert!(matches!(
            gaussian_render_error_handler(&validation, &mut main_world, &mut render_world),
            RenderErrorPolicy::StopRendering
        ));
        assert_eq!(
            main_world
                .resource::<GaussianRenderRecoveryStatus>()
                .snapshot()
                .phase,
            GaussianRenderRecoveryPhase::Stopped
        );
    }

    #[test]
    fn non_device_errors_delegate_to_the_application_handler() {
        let mut main_world = World::new();
        main_world.init_resource::<GaussianRenderRecoverySettings>();
        main_world.init_resource::<GaussianRenderRecoveryStatus>();
        main_world
            .resource::<GaussianRenderRecoveryStatus>()
            .mark_ready(None);
        main_world.init_resource::<DelegatedErrorCount>();
        main_world.insert_resource(GaussianPreviousRenderErrorHandler(
            counting_render_error_handler,
        ));
        let mut render_world = World::new();
        let validation = RenderError {
            ty: ErrorType::Validation,
            description: "bad binding".to_owned(),
            source: None,
        };

        assert!(matches!(
            gaussian_render_error_handler(&validation, &mut main_world, &mut render_world),
            RenderErrorPolicy::Ignore
        ));
        assert_eq!(main_world.resource::<DelegatedErrorCount>().0, 1);
        assert_eq!(
            main_world
                .resource::<GaussianRenderRecoveryStatus>()
                .snapshot()
                .phase,
            GaussianRenderRecoveryPhase::Ready
        );
    }

    #[test]
    fn early_recovery_startup_does_not_reset_the_attempt_budget() {
        let status = GaussianRenderRecoveryStatus::default();
        status.mark_ready(None);
        assert_eq!(status.begin_recovery("first loss"), 1);
        status.mark_ready(None);
        assert_eq!(status.begin_recovery("second early loss"), 2);
        status.mark_ready(None);
        status.observe_healthy_frame(3);
        status.observe_healthy_frame(3);
        assert_eq!(status.snapshot().consecutive_attempts, 2);
        status.observe_healthy_frame(3);
        assert_eq!(status.snapshot().consecutive_attempts, 0);
    }

    #[test]
    fn recovery_handler_installation_can_be_disabled_without_replacing_the_application_handler() {
        let mut app = App::new();
        app.insert_resource(RenderErrorHandler(counting_render_error_handler));
        app.insert_resource(GaussianRenderRecoverySettings {
            install_error_handler: false,
            ..default()
        });
        app.add_plugins(GaussianRenderRecoveryPlugin);

        let installed = app.world().resource::<RenderErrorHandler>().0;
        assert!(std::ptr::fn_addr_eq(
            installed,
            counting_render_error_handler as RenderErrorHandlerFn
        ));
    }

    #[test]
    fn recovery_honors_fallback_policy_and_exhausts_attempt_budget() {
        let mut main_world = World::new();
        main_world.init_resource::<Messages<AppExit>>();
        main_world.insert_resource(GaussianRenderRecoverySettings {
            max_attempts: 1,
            adapter_policy: GaussianRecoveryAdapterPolicy::ForceFallback,
            ..default()
        });
        main_world.init_resource::<GaussianRenderRecoveryStatus>();
        let mut render_world = World::new();
        let loss = RenderError {
            ty: ErrorType::DeviceLost,
            description: "injected".to_owned(),
            source: None,
        };

        let first = gaussian_render_error_handler(&loss, &mut main_world, &mut render_world);
        let RenderErrorPolicy::Recover(RenderCreation::Automatic(settings)) = first else {
            panic!("the first device loss should request automatic recovery");
        };
        assert!(settings.force_fallback_adapter);
        assert!(settings.adapter_name.is_none());

        assert!(matches!(
            gaussian_render_error_handler(&loss, &mut main_world, &mut render_world),
            RenderErrorPolicy::StopRendering
        ));
        let snapshot = main_world
            .resource::<GaussianRenderRecoveryStatus>()
            .snapshot();
        assert_eq!(snapshot.phase, GaussianRenderRecoveryPhase::Exhausted);
        assert_eq!(snapshot.consecutive_attempts, 2);
        assert_eq!(snapshot.total_device_losses, 2);
    }
}
