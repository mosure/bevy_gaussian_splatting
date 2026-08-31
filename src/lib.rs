#[cfg(all(
    feature = "lod_build",
    not(any(
        feature = "sh0",
        feature = "sh1",
        feature = "sh2",
        feature = "sh3",
        feature = "sh4"
    ))
))]
compile_error!(
    "lod_build requires exactly one spherical-harmonic profile; enable lod_build_sh0 or lod_build_sh3"
);

#[cfg(all(
    feature = "lod_build",
    any(
        all(
            feature = "sh0",
            any(feature = "sh1", feature = "sh2", feature = "sh3", feature = "sh4")
        ),
        all(
            feature = "sh1",
            any(feature = "sh2", feature = "sh3", feature = "sh4")
        ),
        all(feature = "sh2", any(feature = "sh3", feature = "sh4")),
        all(feature = "sh3", feature = "sh4")
    )
))]
compile_error!(
    "lod_build package output has one SH ABI; disable default features and enable exactly one lod_build_sh* profile"
);

#[cfg(all(
    feature = "web",
    any(feature = "sh1", feature = "sh2", feature = "sh3", feature = "sh4")
))]
compile_error!(
    "the web profile has an SH0 package/render ABI; build it with --no-default-features --features web"
);

use bevy::prelude::*;
pub use bevy_interleave::prelude::*;

pub use camera::GaussianCamera;

pub use gaussian::{
    formats::{
        lodge::{
            GaussianLodgeManifest, GaussianLodgeManifestHeader,
            LODGE_FEATURE_AUTHENTICATED_DEPENDENCIES, LODGE_FEATURE_CAMERA_CLUSTERS,
            LODGE_FEATURE_DELTA_ULEB128_MEMBERSHIPS, LODGE_FEATURE_DEPTH_FILTER_METADATA,
            LODGE_FEATURE_STABLE_GAUSSIAN_IDS, LODGE_MANIFEST_MAGIC, LODGE_MANIFEST_VERSION,
            LODGE_MEMBERSHIP_SCHEMA_VERSION, LODGE_REQUIRED_FEATURES, LodgeAuthenticatedObject,
            LodgeCameraCluster, LodgeClusterId, LodgeGaussianId, LodgeLevelDescriptor,
            LodgeLevelFilter, LodgeLevelId, LodgeMembershipEncoding, LodgeMembershipEntry,
            LodgeMembershipIndexDescriptor, LodgePageAuthentication, LodgePageLocator,
            LodgeRecordRun, LodgeValidationError,
        },
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
pub use gaussian::lodge_settings::{
    GaussianLodRepresentationKind, GaussianLodStrategy, GaussianLodgeSettings,
    GaussianLodgeSettingsError,
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

#[cfg(any(feature = "lod", feature = "lod_build"))]
pub use io::lodge::{
    LODGE_CONTAINER_MAGIC, LODGE_CONTAINER_VERSION, LODGE_HEADER_LEN, LodgeCodecError,
    LodgeCodecLimits, LodgeManifestEncoding, decode_lodge_manifest, decode_lodge_membership_entry,
    decode_lodge_membership_ids, encode_lodge_manifest, encode_lodge_manifest_with_encoding,
    encode_lodge_membership_ids, sha256_bytes, verify_lodge_authenticated_object,
    verify_lodge_page_bytes,
};

#[cfg(feature = "lod")]
pub use io::lodge::{
    GaussianLodgeAsset, GaussianLodgeHandle, GaussianLodgeManifestLoader,
    GaussianLodgeManifestLoaderSettings, LodgeAssetLoaderError,
};

#[cfg(feature = "lod")]
pub use stream::bridge::{GaussianLodBridgeConfig, GaussianLodBridgePlugin};

#[cfg(feature = "lod")]
pub use stream::lodge::{
    LodgeClassifiedGaussian, LodgeClassifiedPageRun, LodgeMembership, LodgeMembershipClass,
    LodgePairCandidate, LodgePairCommitResult, LodgePairCounts, LodgePairIdentity, LodgePairLimits,
    LodgePairOpacityWeights, LodgePairPublicationPhase, LodgePairPublicationState,
    LodgePairSelection, LodgePairStageResult, LodgePairStatus, LodgePlanError,
    LodgeRecordLocationResolver, build_lodge_pair_candidate, classify_lodge_membership_union,
    coalesce_lodge_classified_runs, lodge_multi_view_page_demand, projected_center_line_weight,
    select_lodge_pair,
};

#[cfg(feature = "lod")]
pub use stream::lodge_status::{GaussianLodgeLifecycle, GaussianLodgeStatus};

#[cfg(lod_render_path)]
pub use stream::lodge_resident::{
    AuthenticatedLodgeBaseManifest, AuthenticatedLodgeMembership,
    AuthenticatedLodgeMembershipObject, AuthenticatedLodgePage, GaussianLodgeResidentCatalog,
    GaussianLodgeResidentError, GaussianLodgeResidentPlugin,
};

#[cfg(feature = "lod")]
pub use stream::status::{
    GaussianLodDebugAvailability, GaussianLodLifecycle, GaussianLodSourceKind, GaussianLodStatus,
    GaussianLodStatusPlugin,
};

#[cfg(feature = "lod")]
pub use stream::render_commit::{
    LodOrchestrationFailure, LodOrchestrationFailureCategory, LodOrchestrationFailureCode,
    LodOrchestrationSource, LodOrchestrationTransition, LodOrchestrationTransitionKind,
};

#[cfg(feature = "lod")]
pub use stream::package::{
    GaussianLodPackageConfig, GaussianLodPackagePlugin, GaussianLodPackageSource,
};
#[cfg(feature = "lod")]
pub use stream::package_source::GaussianLodPackageSourceError;

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
    PlanarGaussianSource, PlyGaussianSource, ReplayableGaussianSource, build_external_lod_package,
};

#[cfg(all(feature = "lod_build", not(target_arch = "wasm32")))]
pub use io::lodge_build_external::{
    CanonicalLodgeMembershipArtifact, LODGE_MEMBERSHIP_DIRECTORY_ENTRY_LEN,
    LODGE_MEMBERSHIP_OBJECT_HEADER_LEN, LODGE_MEMBERSHIP_OBJECT_MAGIC,
    LODGE_MEMBERSHIP_OBJECT_VERSION, LodgeClusterMembershipInput, LodgeMembershipArtifactConfig,
    LodgeMembershipBuildError, LodgeMembershipSliceSource, ReplayableLodgeMembershipSource,
    build_canonical_lodge_membership_artifact, validate_canonical_lodge_membership_artifact,
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

        // External active-set presentation shares the resident Gaussian GPU
        // storage and radix renderer, so it is installed only when that full
        // capability is compiled. The format/codec and CPU planning APIs stay
        // available to portable `lod` builds.
        #[cfg(lod_render_path)]
        app.add_plugins(stream::lodge_resident::GaussianLodgeResidentPlugin);

        app.add_plugins((material::MaterialPlugin, query::QueryPlugin));

        #[cfg(feature = "noise")]
        app.add_plugins(noise::NoisePlugin);
    }
}
