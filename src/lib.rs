use bevy::prelude::*;
pub use bevy_interleave::prelude::*;

pub use camera::GaussianCamera;

pub use gaussian::{
    formats::{
        planar_3d::{
            Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle, random_gaussians_3d,
            random_gaussians_3d_seeded,
        },
        planar_3d_chunked::{
            LodBounds, LodBoundsError, LodIndexRange, LodNodeId, LodPageDescriptor,
            LodPageEncoding, LodPageId, LodPageKind, LodPageRange, LodPageStorage,
            LodPageValidationError, LodSourceRange, PlanarGaussian3dPage,
        },
        planar_3d_lod::{
            CpuGaussianLodBuilder, GaussianLodBuildMetadata, GaussianLodBuildSettings,
            GaussianLodManifest, GaussianLodManifestHeader, GaussianLodNode,
            GaussianLodQualityMetadata, LOD_MORTON_AXIS_MAX, LOD_MORTON_BITS_PER_AXIS,
            LodBuildError, LodBuildSettingsError, LodError, LodMortonRange, LodQualityInterval,
            LodReducerKind, LodValidationError, MomentMergeReducer, MomentMergeResult,
            PROGRESSIVE_MOMENT_MERGE_BUILDER_ABI_VERSION, PlanarGaussian3dLod, build_planar_3d_lod,
            canonical_lod_morton_code, gaussian_support_bounds,
        },
        planar_4d::{
            Gaussian4d, PlanarGaussian4d, PlanarGaussian4dHandle, random_gaussians_4d,
            random_gaussians_4d_seeded,
        },
    },
    lod_debug::{LodDebugPreset, LodDebugSettings},
    lod_settings::{
        GaussianLodSettings, GaussianStreamingSettings, LodBudgets, LodDegradation,
        LodEffectiveStatus, LodQualityEndpoint, LodQualityTarget, LodSelectionMode,
        LodSettingsError,
    },
    settings::{CloudSettings, GaussianMode, RadixSortDepthBits, RasterizeMode},
};

#[cfg(feature = "lod")]
pub use stream::{LodRenderPathSupportError, lod_render_path_is_supported};

#[cfg(feature = "lod")]
pub use render::recovery::{
    GaussianRecoveryAdapterPolicy, GaussianRenderRecoveryError, GaussianRenderRecoveryPhase,
    GaussianRenderRecoveryPlugin, GaussianRenderRecoverySettings, GaussianRenderRecoverySnapshot,
    GaussianRenderRecoveryStatus,
};

#[cfg(feature = "lod")]
pub use io::lod::{GaussianLodAsset, GaussianLodHandle, GaussianLodManifestLoaderSettings};

#[cfg(feature = "lod")]
pub use stream::bridge::{GaussianLodBridgeConfig, GaussianLodBridgePlugin};

#[cfg(feature = "lod")]
pub use stream::status::{
    GaussianLodDebugAvailability, GaussianLodLifecycle, GaussianLodSourceKind, GaussianLodStatus,
    GaussianLodStatusPlugin,
};

#[cfg(feature = "lod")]
pub use stream::render_commit::{
    LodOrchestrationFailure, LodOrchestrationFailureCategory, LodOrchestrationFailureCode,
};

#[cfg(feature = "lod")]
pub use stream::package::{
    GaussianLodPackageConfig, GaussianLodPackagePlugin, GaussianLodPackageSource,
};

#[cfg(feature = "lod_build")]
pub use gaussian::lod_build_gpu::hierarchy::{
    GpuLodHierarchyBuilder, GpuLodHierarchyError, GpuLodHierarchyLimits,
};

#[cfg(all(feature = "lod_build", not(target_arch = "wasm32")))]
pub use io::lod_build_external::{
    CpuExternalLodBatchPreprocessor, EXTERNAL_LOD_BUILDER_ABI_VERSION,
    ExternalLodBatchPreprocessor, ExternalLodBuildConfig, ExternalLodBuildError,
    ExternalLodBuildLimits, ExternalLodBuildPlan, ExternalLodBuildReport,
    ExternalLodPreprocessorOutputOrder, GpuHierarchyExternalLodBatchPreprocessor,
    PlyGaussianSource, ReplayableGaussianSource, build_external_lod_package,
};

pub use io::scene::{
    GaussianKernel, GaussianPrimitiveMetadata, GaussianPrimitiveSpec, GaussianProjection,
    GaussianScene, GaussianSceneHandle, GaussianSortingMethod, SceneCamera, SceneExportCamera,
    SceneExportCloud, write_khr_gaussian_scene_glb, write_khr_gaussian_scene_gltf,
};

pub use material::spherical_harmonics::SphericalHarmonicCoefficients;

use io::IoPlugin;

pub mod camera;
pub mod gaussian;
pub mod io;
pub mod material;
pub mod math;
pub mod morph;
pub mod query;
pub mod render;
pub mod sort;
pub mod stream;
pub mod utils;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

#[cfg(feature = "noise")]
pub mod noise;

pub struct GaussianSplattingPlugin;

impl Plugin for GaussianSplattingPlugin {
    fn build(&self, app: &mut App) {
        // TODO: allow hot reloading of Cloud handle through inspector UI
        app.register_type::<SphericalHarmonicCoefficients>();

        app.add_plugins(IoPlugin);

        app.add_plugins((
            camera::GaussianCameraPlugin,
            gaussian::settings::SettingsPlugin,
            gaussian::cloud::CloudPlugin::<Gaussian3d>::default(),
            gaussian::cloud::CloudPlugin::<Gaussian4d>::default(),
        ));

        #[cfg(feature = "lod")]
        app.add_plugins((
            gaussian::lod_settings::GaussianLodSettingsPlugin,
            render::recovery::GaussianRenderRecoveryPlugin,
        ));

        // TODO: add half types
        app.add_plugins((
            PlanarStoragePlugin::<Gaussian3d>::default(),
            PlanarStoragePlugin::<Gaussian4d>::default(),
        ));

        app.add_plugins((
            render::RenderPipelinePlugin::<Gaussian3d>::default(),
            render::RenderPipelinePlugin::<Gaussian4d>::default(),
        ));

        #[cfg(feature = "lod")]
        app.add_plugins((
            stream::bridge::GaussianLodBridgePlugin,
            stream::package::GaussianLodPackagePlugin,
            stream::status::GaussianLodStatusPlugin,
        ));

        app.add_plugins((material::MaterialPlugin, query::QueryPlugin));

        #[cfg(feature = "noise")]
        app.add_plugins(noise::NoisePlugin);
    }
}
