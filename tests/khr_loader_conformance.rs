use std::{
    collections::HashMap,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use bevy::{
    asset::{
        AssetMetaCheck, AssetPlugin, DependencyLoadState, LoadState, RecursiveDependencyLoadState,
        UnapprovedPathMode,
    },
    prelude::*,
};
use bevy_gaussian_splatting::{
    GaussianKernel, GaussianProjection, GaussianSortingMethod, PlanarGaussian3d, SceneExportCloud,
    gaussian::settings::GaussianColorSpace,
    io::{
        IoPlugin,
        scene::{GaussianScene, encode_khr_gaussian_scene_gltf_bytes},
    },
};

const FIXTURE_ROOT: &str = "tests/fixtures/khr_gaussian_splatting";

#[derive(Clone, Copy)]
struct ExpectedCase {
    scale_raw: [f32; 3],
    opacity: f32,
    sh_degree: usize,
    color_space: GaussianColorSpace,
}

fn approx_eq(actual: f32, expected: f32, epsilon: f32) {
    assert!(
        (actual - expected).abs() <= epsilon,
        "expected {expected}, got {actual}"
    );
}

fn max_supported_test_sh_degree() -> usize {
    if cfg!(feature = "sh4") {
        4
    } else if cfg!(feature = "sh3") {
        3
    } else if cfg!(feature = "sh2") {
        2
    } else if cfg!(feature = "sh1") {
        1
    } else {
        0
    }
}

fn try_load_fixture_scene(path: &str) -> Result<(GaussianScene, HashMap<String, PlanarGaussian3d>), String> {
    let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_ROOT);
    if !fixture_root.exists() {
        return Err(format!(
            "fixture root does not exist: {}",
            fixture_root.display()
        ));
    }

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(AssetPlugin {
        file_path: fixture_root.display().to_string(),
        processed_file_path: fixture_root.display().to_string(),
        meta_check: AssetMetaCheck::Never,
        unapproved_path_mode: UnapprovedPathMode::Allow,
        ..default()
    });
    app.init_asset::<PlanarGaussian3d>();
    app.add_plugins(IoPlugin);

    let scene_handle: Handle<GaussianScene> = {
        let asset_server = app.world().resource::<AssetServer>();
        asset_server.load(path.to_owned())
    };

    let mut loaded_scene = None;
    let mut last_states = (
        LoadState::NotLoaded,
        DependencyLoadState::NotLoaded,
        RecursiveDependencyLoadState::NotLoaded,
    );
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        app.update();

        if let Some((load_state, dep_state, rec_dep_state)) = app
            .world()
            .resource::<AssetServer>()
            .get_load_states(&scene_handle)
        {
            last_states = (load_state.clone(), dep_state.clone(), rec_dep_state.clone());

            match (&load_state, &dep_state, &rec_dep_state) {
                (LoadState::Failed(err), _, _)
                | (_, DependencyLoadState::Failed(err), _)
                | (_, _, RecursiveDependencyLoadState::Failed(err)) => {
                    return Err(format!("fixture '{path}' failed to load: {err}"));
                }
                (LoadState::Loaded, _, RecursiveDependencyLoadState::Loaded) => {
                    loaded_scene = app
                        .world()
                        .resource::<Assets<GaussianScene>>()
                        .get(&scene_handle)
                        .cloned();
                    if loaded_scene.is_some() {
                        break;
                    }
                }
                _ => {}
            }
        }

        thread::sleep(Duration::from_millis(1));
    }

    let scene = loaded_scene.ok_or_else(|| {
        format!(
            "fixture scene '{path}' failed to load (load_state={:?}, dependency_state={:?}, recursive_dependency_state={:?})",
            last_states.0, last_states.1, last_states.2
        )
    });
    let scene = scene?;
    let mut clouds_by_case = HashMap::new();

    for bundle in &scene.bundles {
        let case_name = bundle
            .name
            .split("_mesh")
            .next()
            .expect("bundle name should include mesh suffix")
            .to_owned();

        let cloud = app
            .world()
            .resource::<Assets<PlanarGaussian3d>>()
            .get(&bundle.cloud)
            .cloned()
            .ok_or_else(|| format!("cloud asset for case '{case_name}' missing"))?;
        clouds_by_case.insert(case_name, cloud);
    }

    Ok((scene, clouds_by_case))
}

fn load_fixture_scene(path: &str) -> (GaussianScene, HashMap<String, PlanarGaussian3d>) {
    try_load_fixture_scene(path).unwrap_or_else(|err| panic!("{err}"))
}

fn expected_cases() -> HashMap<&'static str, ExpectedCase> {
    let mut cases = HashMap::new();

    // Default attributes used by most matrix entries.
    let default_scale = [0.0, 0.5, -0.5];
    let default_opacity = 0.25;

    for name in [
        "rotation_f32",
        "rotation_i8_norm",
        "rotation_i16_norm",
        "opacity_u8_norm",
        "opacity_u16_norm",
        "opacity_f32",
        "sh_degree0",
        "sh_degree1",
        "sh_degree2",
        "sh_degree3",
    ] {
        cases.insert(
            name,
            ExpectedCase {
                scale_raw: default_scale,
                opacity: default_opacity,
                sh_degree: 0,
                color_space: GaussianColorSpace::LinRec709Display,
            },
        );
    }

    // Per-case overrides.
    cases.insert(
        "scale_f32",
        ExpectedCase {
            scale_raw: [0.2, -0.1, 0.7],
            opacity: default_opacity,
            sh_degree: 0,
            color_space: GaussianColorSpace::SrgbRec709Display,
        },
    );
    cases.insert(
        "scale_i8",
        ExpectedCase {
            scale_raw: [1.0, -2.0, 3.0],
            opacity: default_opacity,
            sh_degree: 0,
            color_space: GaussianColorSpace::LinRec709Display,
        },
    );
    cases.insert(
        "scale_i8_norm",
        ExpectedCase {
            scale_raw: [1.0, 0.0, -1.0],
            opacity: default_opacity,
            sh_degree: 0,
            color_space: GaussianColorSpace::LinRec709Display,
        },
    );
    cases.insert(
        "scale_i16",
        ExpectedCase {
            scale_raw: [2.0, -3.0, 4.0],
            opacity: default_opacity,
            sh_degree: 0,
            color_space: GaussianColorSpace::LinRec709Display,
        },
    );
    cases.insert(
        "scale_i16_norm",
        ExpectedCase {
            scale_raw: [1.0, 0.0, -1.0],
            opacity: default_opacity,
            sh_degree: 0,
            color_space: GaussianColorSpace::LinRec709Display,
        },
    );
    cases.insert(
        "opacity_f32",
        ExpectedCase {
            scale_raw: default_scale,
            opacity: 0.75,
            sh_degree: 0,
            color_space: GaussianColorSpace::LinRec709Display,
        },
    );
    cases.insert(
        "opacity_u8_norm",
        ExpectedCase {
            scale_raw: default_scale,
            opacity: 64.0 / 255.0,
            sh_degree: 0,
            color_space: GaussianColorSpace::LinRec709Display,
        },
    );
    cases.insert(
        "opacity_u16_norm",
        ExpectedCase {
            scale_raw: default_scale,
            opacity: 16384.0 / 65535.0,
            sh_degree: 0,
            color_space: GaussianColorSpace::LinRec709Display,
        },
    );
    cases.insert(
        "sh_degree1",
        ExpectedCase {
            scale_raw: default_scale,
            opacity: default_opacity,
            sh_degree: 1,
            color_space: GaussianColorSpace::LinRec709Display,
        },
    );
    cases.insert(
        "sh_degree2",
        ExpectedCase {
            scale_raw: default_scale,
            opacity: default_opacity,
            sh_degree: 2,
            color_space: GaussianColorSpace::LinRec709Display,
        },
    );
    cases.insert(
        "sh_degree3",
        ExpectedCase {
            scale_raw: default_scale,
            opacity: default_opacity,
            sh_degree: 3,
            color_space: GaussianColorSpace::LinRec709Display,
        },
    );

    cases
}

fn assert_case_cloud(case_name: &str, cloud: &PlanarGaussian3d, expected: ExpectedCase) {
    assert_eq!(cloud.position_visibility.len(), 1, "case {case_name}");
    assert_eq!(cloud.rotation.len(), 1, "case {case_name}");
    assert_eq!(cloud.scale_opacity.len(), 1, "case {case_name}");
    assert_eq!(cloud.spherical_harmonic.len(), 1, "case {case_name}");

    let position = cloud.position_visibility[0].position;
    approx_eq(position[0], 1.0, 1e-6);
    approx_eq(position[1], 2.0, 1e-6);
    approx_eq(position[2], 3.0, 1e-6);

    // All matrix fixtures encode identity quaternion in different component encodings.
    let rotation = cloud.rotation[0].rotation;
    approx_eq(rotation[0], 1.0, 1e-5);
    approx_eq(rotation[1], 0.0, 1e-5);
    approx_eq(rotation[2], 0.0, 1e-5);
    approx_eq(rotation[3], 0.0, 1e-5);

    let scale = cloud.scale_opacity[0].scale;
    approx_eq(scale[0], expected.scale_raw[0].exp(), 1e-5);
    approx_eq(scale[1], expected.scale_raw[1].exp(), 1e-5);
    approx_eq(scale[2], expected.scale_raw[2].exp(), 1e-5);
    approx_eq(cloud.scale_opacity[0].opacity, expected.opacity, 1e-5);

    let coeffs = &cloud.spherical_harmonic[0].coefficients;
    let expected_coeff_count = (expected.sh_degree + 1) * (expected.sh_degree + 1);
    for coefficient in 0..expected_coeff_count {
        let base = coefficient * 3;
        approx_eq(coeffs[base], coefficient as f32 + 0.1, 1e-6);
        approx_eq(coeffs[base + 1], coefficient as f32 + 0.2, 1e-6);
        approx_eq(coeffs[base + 2], coefficient as f32 + 0.3, 1e-6);
    }
}

fn assert_scene_cases(scene: &GaussianScene, clouds: &HashMap<String, PlanarGaussian3d>) {
    let expected = expected_cases();
    assert_eq!(scene.bundles.len(), expected.len());
    assert_eq!(clouds.len(), expected.len());
    assert_eq!(scene.cameras.len(), 1);
    let scene_camera = &scene.cameras[0];
    assert_eq!(scene_camera.name, "fixture_camera");
    let translation = scene_camera.transform.translation;
    approx_eq(translation.x, 4.0, 1e-6);
    approx_eq(translation.y, 5.0, 1e-6);
    approx_eq(translation.z, 6.0, 1e-6);

    for bundle in &scene.bundles {
        let case_name = bundle
            .name
            .split("_mesh")
            .next()
            .expect("bundle name should include mesh suffix");
        let expected_case = *expected
            .get(case_name)
            .unwrap_or_else(|| panic!("unexpected case '{case_name}'"));
        assert_eq!(bundle.settings.color_space, expected_case.color_space);

        let cloud = clouds
            .get(case_name)
            .unwrap_or_else(|| panic!("missing cloud for case '{case_name}'"));
        assert_case_cloud(case_name, cloud, expected_case);
    }
}

#[test]
fn khr_loader_conformance_matrix_gltf_and_glb() {
    let supported_sh_degree = max_supported_test_sh_degree();
    for fixture in ["khr_conformance_matrix.gltf", "khr_conformance_matrix.glb"] {
        if supported_sh_degree >= 3 {
            let (scene, clouds) = load_fixture_scene(fixture);
            assert_scene_cases(&scene, &clouds);
            continue;
        }

        let err = try_load_fixture_scene(fixture).unwrap_err();
        assert!(
            err.contains("supports up to degree"),
            "expected unsupported SH degree error for fixture '{fixture}', got: {err}"
        );
    }
}

#[test]
fn khr_loader_extensibility_and_color0_fallback() {
    let (scene, clouds) = load_fixture_scene("khr_extensible_fallback.gltf");
    assert_eq!(scene.bundles.len(), 1);
    assert_eq!(scene.cameras.len(), 0);

    let bundle = &scene.bundles[0];
    assert_eq!(
        bundle.settings.color_space,
        GaussianColorSpace::SrgbRec709Display
    );
    assert_eq!(bundle.metadata.kernel, GaussianKernel::Ellipse);
    assert_eq!(bundle.metadata.projection, GaussianProjection::Perspective);
    assert_eq!(
        bundle.metadata.sorting_method,
        GaussianSortingMethod::CameraDistance
    );
    assert_eq!(bundle.metadata.spec.kernel, "customShape");
    assert_eq!(bundle.metadata.spec.color_space, "custom_space_display");
    assert_eq!(bundle.metadata.spec.projection, "perspective");
    assert_eq!(bundle.metadata.spec.sorting_method, "cameraDistance");
    assert!(
        bundle.metadata.spec.extension_object.is_some(),
        "raw extension payload should be preserved"
    );
    let extension_object = bundle.metadata.spec.extension_object.as_ref().unwrap();
    assert!(
        extension_object["extensions"]["EXT_gaussian_splatting_kernel_customShape"].is_object()
    );

    let cloud = clouds
        .get("extensible_unknown")
        .expect("missing cloud loaded from extensible fixture");
    let coeffs = &cloud.spherical_harmonic[0].coefficients;
    approx_eq(coeffs[0], 1.0, 1e-4);
    approx_eq(coeffs[1], 2.0, 1e-4);
    approx_eq(coeffs[2], 3.0, 1e-4);
    approx_eq(cloud.scale_opacity[0].opacity, 0.5, 1e-6);

    let exported = encode_khr_gaussian_scene_gltf_bytes(
        &[SceneExportCloud {
            cloud: cloud.clone(),
            name: "extensible_unknown".to_owned(),
            settings: bundle.settings.clone(),
            transform: bundle.transform,
            metadata: bundle.metadata.clone(),
        }],
        None,
    )
    .expect("failed to export extensible fixture");

    let root: serde_json::Value = serde_json::from_slice(&exported).unwrap();
    let exported_extension =
        &root["meshes"][0]["primitives"][0]["extensions"]["KHR_gaussian_splatting"];
    assert_eq!(exported_extension["kernel"].as_str(), Some("customShape"));
    assert_eq!(
        exported_extension["colorSpace"].as_str(),
        Some("custom_space_display")
    );
    assert_eq!(
        exported_extension["projection"].as_str(),
        Some("perspective")
    );
    assert_eq!(
        exported_extension["sortingMethod"].as_str(),
        Some("cameraDistance")
    );
    assert!(
        exported_extension["extensions"]["EXT_gaussian_splatting_kernel_customShape"].is_object()
    );
}
