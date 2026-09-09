//! Explicit Discrete mode may publish bounded adjacent, complete resident
//! cohorts. Unbounded-refinement hierarchies and continuous morphs use the
//! standard package publication path.

use super::*;

pub(super) fn enabled(
    state: &PackageInstantiation,
    cameras: &[PackageCameraView],
    settings: &GaussianLodSettings,
) -> bool {
    policy_enabled(state, settings)
        && !cameras.is_empty()
        && state.current.as_ref().is_some_and(|current| {
            current.len() == cameras.len()
                && cameras.iter().all(|camera| {
                    current
                        .get(camera.entity)
                        .is_some_and(LodRenderCandidate::render_is_active)
                })
        })
}

pub(super) fn policy_enabled(state: &PackageInstantiation, settings: &GaussianLodSettings) -> bool {
    settings.presentation_mode == LodPresentationMode::Discrete
        && state.runtime.lock().is_ok_and(|runtime| {
            runtime
                .hierarchy()
                .manifest()
                .build
                .has_bounded_refinement_amplification()
        })
}

pub(super) fn current_nodes(
    current: &LodRenderCandidates,
) -> Vec<(LodRuntimeViewId, Vec<LodNodeId>)> {
    current
        .by_camera
        .values()
        .map(|candidate| {
            (
                candidate.frontier().view(),
                candidate
                    .target_render_ranges()
                    .iter()
                    .map(|range| range.node)
                    .collect(),
            )
        })
        .collect()
}

pub(super) fn prepare(
    state: &mut PackageInstantiation,
    request: &PackageCutRequestSignature,
    cameras: &[PackageCameraView],
    settings: &GaussianLodSettings,
    world_from_local: Mat4,
) -> Result<(), GaussianLodPackageError> {
    if state
        .cold_direct_target
        .as_ref()
        .is_some_and(|target| target.request.same_critical_request(request))
    {
        return Ok(());
    }
    if state.cold_direct_target.is_some() {
        clear_package_direct_target(state)?;
    }
    let nodes = current_nodes(
        state
            .current
            .as_ref()
            .expect("enabled mode retains a current cut"),
    );
    let views = cameras
        .iter()
        .map(|camera| {
            (
                LodRuntimeViewId(camera.entity.to_bits()),
                camera.view.with_world_from_local(world_from_local),
            )
        })
        .collect::<Vec<_>>();
    let plan = state
        .runtime
        .get_mut()
        .map_err(|_| GaussianLodPackageError::RuntimePoisoned)?
        .package_discrete_target_plan(&views, &nodes, &state.current_page_leases, settings)
        .map_err(GaussianLodPackageError::Runtime)?;
    state.bootstrap_handoff = None;
    state.cold_direct_target = Some(PackageColdDirectTarget {
        request: request.clone(),
        plan,
    });
    Ok(())
}
