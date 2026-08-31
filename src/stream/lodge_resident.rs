//! Resident-host integration for authenticated LODGE active sets.
//!
//! This is the first deliberately simple package path: the complete stable-ID
//! catalog and every authenticated cluster membership are resident on the CPU
//! before the entity can become drawable. Camera updates therefore never
//! publish a partially resident active set. Demand streaming can replace the
//! resident page proof later without changing the pair planner or the external
//! render-candidate ABI.

use std::{
    any::TypeId,
    collections::{BTreeMap, BTreeSet, HashSet},
    error::Error,
    fmt,
    mem::size_of,
    sync::{Arc, atomic::Ordering},
};

use bevy::{
    camera::{
        CameraUpdateSystems,
        visibility::{VisibilitySystems, VisibleEntities},
    },
    prelude::*,
    transform::TransformSystems,
};
use bevy_interleave::prelude::{Planar, PlanarHandle};
use sha2::{Digest, Sha256};

use crate::{
    CloudSettings, GaussianCamera,
    gaussian::{
        cloud::CloudVisibilityClass,
        formats::{
            lodge::{
                GaussianLodgeManifest, LodgeAuthenticatedObject, LodgeCameraCluster,
                LodgeClusterId, LodgeGaussianId, LodgeLevelId, LodgeMembershipEntry,
                LodgeMembershipIndexDescriptor, LodgePageAuthentication,
            },
            planar_3d::{Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle},
            planar_3d_chunked::{LodNodeId, LodPageDescriptor, LodPageId, PlanarGaussian3dPage},
            planar_3d_lod::GaussianLodManifest,
        },
        lod_debug::LodDebugMetadata,
        lod_settings::{GaussianLodSettings, LodSelectionMode},
        lodge_settings::{GaussianLodRepresentationKind, GaussianLodgeSettings},
        settings::GaussianMode,
    },
    io::{
        lod::{
            GaussianLodHandle, LodCodecLimits as LodPageCodecLimits,
            decode_manifest as decode_lod_manifest, decode_page_with_descriptor,
        },
        lodge::{
            GaussianLodgeAsset, GaussianLodgeHandle, LodgeCodecLimits,
            decode_lodge_membership_entry, sha256_bytes, verify_lodge_authenticated_object,
            verify_lodge_page_bytes,
        },
    },
    render::{lod::representable_source_count, recovery::GaussianRenderRecoveryStatus},
};

use super::{
    bridge::{GaussianLodBridgeStatus, GaussianLodBridgeUpdate},
    cache::AtlasSlot,
    lodge::{
        LodgeMembership, LodgePairCandidate, LodgePairIdentity, LodgePairLimits,
        LodgePairPublicationState, LodgePairSelection, LodgePlanError, LodgeRecordLocationResolver,
        build_lodge_pair_candidate, lodge_multi_view_page_demand, projected_center_line_weight,
        select_lodge_pair_from_validated_clusters,
    },
    lodge_status::{GaussianLodgeLifecycle, GaussianLodgeStatus},
    package::GaussianLodPackageUpdate,
    render_commit::{
        GaussianLodRenderCommitPlugin, LOD_RENDER_WAITING, LodExternalActiveSetPresentation,
        LodOrchestrationFailure, LodOrchestrationFailureCode, LodOrchestrationSource,
        LodOrchestrationTransition, LodOrchestrationTransitionKind, LodRenderCandidate,
        LodRenderCandidates, LodRenderEnvironmentEpoch,
    },
    runtime::{LodCandidateFrontier, LodPhysicalRange, LodRuntimeViewId},
};

const MAX_RESIDENT_LODGE_VIEWS: usize = 16;

/// Companion hierarchy manifest authenticated by the exact dependency object
/// declared in a validated `.gslodge` sidecar.
#[derive(Clone, Debug)]
pub struct AuthenticatedLodgeBaseManifest {
    manifest: GaussianLodManifest,
    dependency: LodgeAuthenticatedObject,
}

impl AuthenticatedLodgeBaseManifest {
    pub fn decode(
        lodge_manifest: &GaussianLodgeManifest,
        encoded: &[u8],
        limits: LodPageCodecLimits,
        max_authenticated_bytes: u64,
    ) -> Result<Self, GaussianLodgeResidentError> {
        verify_lodge_authenticated_object(
            encoded,
            &lodge_manifest.base_manifest,
            max_authenticated_bytes,
        )
        .map_err(|error| GaussianLodgeResidentError::PageCodec(error.to_string()))?;
        let manifest = decode_lod_manifest(encoded, limits)
            .map_err(|error| GaussianLodgeResidentError::PageCodec(error.to_string()))?;
        lodge_manifest
            .validate_against_base(&manifest)
            .map_err(|error| GaussianLodgeResidentError::Manifest(error.to_string()))?;
        Ok(Self {
            manifest,
            dependency: lodge_manifest.base_manifest.clone(),
        })
    }

    pub fn manifest(&self) -> &GaussianLodManifest {
        &self.manifest
    }
}

/// One page which crossed both LODGE SHA-256 authentication and the ordinary
/// bounded page decoder. Private fields prevent arbitrary decoded pages from
/// being mislabeled authenticated before resident materialization.
#[derive(Clone, Debug)]
pub struct AuthenticatedLodgePage {
    page: PlanarGaussian3dPage,
    descriptor: LodPageDescriptor,
    authentication: LodgePageAuthentication,
}

impl AuthenticatedLodgePage {
    pub fn decode(
        manifest: &GaussianLodgeManifest,
        descriptor: &LodPageDescriptor,
        encoded: &[u8],
        limits: LodPageCodecLimits,
        max_authenticated_bytes: u64,
    ) -> Result<Self, GaussianLodgeResidentError> {
        let authentication = manifest.authentication_for_page(descriptor.id).ok_or(
            GaussianLodgeResidentError::UnauthenticatedPage(descriptor.id),
        )?;
        verify_lodge_page_bytes(encoded, descriptor, authentication, max_authenticated_bytes)
            .map_err(|error| GaussianLodgeResidentError::PageCodec(error.to_string()))?;
        let page = decode_page_with_descriptor(encoded, descriptor, limits)
            .map_err(|error| GaussianLodgeResidentError::PageCodec(error.to_string()))?;
        Ok(Self {
            page,
            descriptor: descriptor.clone(),
            authentication: *authentication,
        })
    }

    pub fn page(&self) -> &PlanarGaussian3dPage {
        &self.page
    }
}

/// One cluster membership decoded from the exact per-entry authenticated byte
/// range declared by the sidecar manifest.
#[derive(Clone, Debug)]
pub struct AuthenticatedLodgeMembershipObject {
    encoded: Arc<[u8]>,
    descriptor: Arc<LodgeMembershipIndexDescriptor>,
}

impl AuthenticatedLodgeMembershipObject {
    /// Verifies the complete membership object and its independently pinned
    /// directory/index range before any cluster stream can be decoded.
    pub fn decode(
        manifest: &GaussianLodgeManifest,
        encoded: Vec<u8>,
        max_authenticated_bytes: u64,
    ) -> Result<Self, GaussianLodgeResidentError> {
        let descriptor = &manifest.membership_index;
        verify_lodge_authenticated_object(&encoded, &descriptor.object, max_authenticated_bytes)
            .map_err(|error| GaussianLodgeResidentError::MembershipCodec(error.to_string()))?;
        let (start, len) = descriptor.index_byte_range;
        let start = usize::try_from(start)
            .map_err(|_| GaussianLodgeResidentError::InvalidMembershipIndexRange)?;
        let end = descriptor
            .index_byte_range
            .0
            .checked_add(len)
            .and_then(|end| usize::try_from(end).ok())
            .ok_or(GaussianLodgeResidentError::InvalidMembershipIndexRange)?;
        let index = encoded
            .get(start..end)
            .ok_or(GaussianLodgeResidentError::InvalidMembershipIndexRange)?;
        if sha256_bytes(index) != descriptor.index_sha256 {
            return Err(GaussianLodgeResidentError::MembershipIndexHashMismatch);
        }
        Ok(Self {
            encoded: encoded.into(),
            descriptor: Arc::new(descriptor.clone()),
        })
    }

    pub fn decode_membership(
        &self,
        manifest: &GaussianLodgeManifest,
        cluster: LodgeClusterId,
        limits: LodgeCodecLimits,
    ) -> Result<AuthenticatedLodgeMembership, GaussianLodgeResidentError> {
        if self.descriptor.as_ref() != &manifest.membership_index {
            return Err(GaussianLodgeResidentError::MembershipObjectMismatch);
        }
        let entry = manifest
            .membership_for_cluster(cluster)
            .ok_or(GaussianLodgeResidentError::MissingMembership(cluster))?;
        let (start, len) = entry.byte_range;
        let start = usize::try_from(start)
            .map_err(|_| GaussianLodgeResidentError::InvalidMembership(cluster))?;
        let end = entry
            .byte_range
            .0
            .checked_add(len)
            .and_then(|end| usize::try_from(end).ok())
            .ok_or(GaussianLodgeResidentError::InvalidMembership(cluster))?;
        let encoded_range = self
            .encoded
            .get(start..end)
            .ok_or(GaussianLodgeResidentError::InvalidMembership(cluster))?;
        AuthenticatedLodgeMembership::decode_range(
            manifest,
            cluster,
            encoded_range,
            limits,
            Arc::clone(&self.descriptor),
            entry.clone(),
        )
    }
}

/// One cluster membership decoded from an authenticated membership object.
#[derive(Clone, Debug)]
pub struct AuthenticatedLodgeMembership {
    membership: LodgeMembership,
    descriptor: Arc<LodgeMembershipIndexDescriptor>,
    entry: LodgeMembershipEntry,
}

impl AuthenticatedLodgeMembership {
    fn decode_range(
        manifest: &GaussianLodgeManifest,
        cluster: LodgeClusterId,
        encoded_range: &[u8],
        limits: LodgeCodecLimits,
        descriptor: Arc<LodgeMembershipIndexDescriptor>,
        entry: LodgeMembershipEntry,
    ) -> Result<Self, GaussianLodgeResidentError> {
        if entry.cluster != cluster {
            return Err(GaussianLodgeResidentError::MissingMembership(cluster));
        }
        let ids = decode_lodge_membership_entry(
            encoded_range,
            &entry,
            manifest.header.stable_gaussian_count,
            limits,
        )
        .map_err(|error| GaussianLodgeResidentError::MembershipCodec(error.to_string()))?;
        let membership = LodgeMembership::new(cluster, ids)
            .map_err(|error| GaussianLodgeResidentError::Planning(error.to_string()))?;
        Ok(Self {
            membership,
            descriptor,
            entry,
        })
    }

    pub fn membership(&self) -> &LodgeMembership {
        &self.membership
    }
}

/// Attach-ready canonical catalog and authenticated cluster memberships.
///
/// There is intentionally no safe constructor from an arbitrary planar asset
/// handle: a same-length cloud cannot prove the required invariant that stable
/// ID `N` lives at physical catalog index `N - 1`.
#[derive(Component, Clone, Debug)]
pub struct GaussianLodgeResidentCatalog {
    catalog: Handle<PlanarGaussian3d>,
    manifest: Arc<GaussianLodgeManifest>,
    memberships: Arc<BTreeMap<LodgeClusterId, LodgeMembership>>,
    page_slots: Arc<BTreeMap<LodPageId, AtlasSlot>>,
    resident_pages: u32,
    catalog_sha256: [u8; 32],
}

impl GaussianLodgeResidentCatalog {
    /// Authenticates the dependency closure, materializes records in manifest
    /// run order, inserts the canonical planar asset, and returns an attachable
    /// resident component.
    ///
    /// The authenticated producer is authoritative for the per-cluster mixture
    /// of discrete levels. The LODGE manifest does not encode a reconstructable
    /// cluster-radius level-selection policy, so the optional conservative
    /// sphere audit is reported separately by
    /// [`Self::audit_conservative_sphere_levels`].
    pub fn from_authenticated_pages(
        manifest: Arc<GaussianLodgeManifest>,
        base_manifest: &AuthenticatedLodgeBaseManifest,
        settings: &GaussianLodgeSettings,
        pages: impl IntoIterator<Item = AuthenticatedLodgePage>,
        memberships: impl IntoIterator<Item = AuthenticatedLodgeMembership>,
        assets: &mut Assets<PlanarGaussian3d>,
    ) -> Result<Self, GaussianLodgeResidentError> {
        Self::validate_manifest_budget(&manifest, settings)?;
        if base_manifest.dependency != manifest.base_manifest {
            return Err(GaussianLodgeResidentError::UnauthenticatedBaseManifest);
        }
        manifest
            .validate_against_base(&base_manifest.manifest)
            .map_err(|error| GaussianLodgeResidentError::Manifest(error.to_string()))?;

        let mut descriptors = BTreeMap::new();
        for descriptor in base_manifest
            .manifest
            .pages
            .iter()
            .chain(&manifest.extra_pages)
        {
            if descriptors.insert(descriptor.id, descriptor).is_some() {
                return Err(GaussianLodgeResidentError::DuplicatePage(descriptor.id));
            }
        }

        let mut decoded_pages = BTreeMap::new();
        for authenticated in pages {
            let page = authenticated.page;
            let descriptor = descriptors
                .get(&page.id)
                .copied()
                .ok_or(GaussianLodgeResidentError::UnexpectedPage(page.id))?;
            let expected_authentication = manifest
                .authentication_for_page(page.id)
                .ok_or(GaussianLodgeResidentError::UnauthenticatedPage(page.id))?;
            if authenticated.descriptor != *descriptor
                || authenticated.authentication != *expected_authentication
            {
                return Err(GaussianLodgeResidentError::UnauthenticatedPage(page.id));
            }
            page.validate(descriptor)
                .map_err(|error| GaussianLodgeResidentError::InvalidPage {
                    page: page.id,
                    detail: error.to_string(),
                })?;
            let page_id = page.id;
            if decoded_pages.insert(page_id, page).is_some() {
                return Err(GaussianLodgeResidentError::DuplicatePage(page_id));
            }
        }

        let authenticated_pages = manifest
            .page_authentication
            .iter()
            .map(|authentication| authentication.page)
            .collect::<BTreeSet<_>>();
        if !decoded_pages
            .keys()
            .copied()
            .eq(authenticated_pages.iter().copied())
        {
            return Err(GaussianLodgeResidentError::IncompletePageClosure);
        }

        let mut decoded_memberships = BTreeMap::new();
        let mut membership_descriptor: Option<Arc<LodgeMembershipIndexDescriptor>> = None;
        for authenticated in memberships {
            let cluster = authenticated.membership.cluster();
            let entry = manifest
                .membership_for_cluster(cluster)
                .ok_or(GaussianLodgeResidentError::UnexpectedMembership(cluster))?;
            if let Some(descriptor) = &membership_descriptor {
                if !Arc::ptr_eq(descriptor, &authenticated.descriptor) {
                    return Err(GaussianLodgeResidentError::MembershipObjectMismatch);
                }
            } else {
                if authenticated.descriptor.as_ref() != &manifest.membership_index {
                    return Err(GaussianLodgeResidentError::MembershipObjectMismatch);
                }
                membership_descriptor = Some(Arc::clone(&authenticated.descriptor));
            }
            if authenticated.entry != *entry
                || authenticated.membership.len() as u64 != entry.member_count
                || authenticated.membership.ids().first().copied() != Some(entry.first_id)
                || authenticated.membership.ids().last().copied() != Some(entry.last_id)
            {
                return Err(GaussianLodgeResidentError::InvalidMembership(cluster));
            }
            if decoded_memberships
                .insert(cluster, authenticated.membership)
                .is_some()
            {
                return Err(GaussianLodgeResidentError::DuplicateMembership(cluster));
            }
        }
        let expected_clusters = manifest
            .clusters
            .iter()
            .map(|cluster| cluster.id)
            .collect::<BTreeSet<_>>();
        if !decoded_memberships
            .keys()
            .copied()
            .eq(expected_clusters.iter().copied())
        {
            return Err(GaussianLodgeResidentError::IncompleteMembershipClosure);
        }

        let stable_count = manifest.header.stable_gaussian_count;
        if usize::try_from(stable_count)
            .ok()
            .and_then(representable_source_count)
            .is_none()
        {
            return Err(GaussianLodgeResidentError::CatalogCountNotRepresentable(
                stable_count,
            ));
        }
        let capacity = usize::try_from(stable_count)
            .map_err(|_| GaussianLodgeResidentError::CatalogCountNotRepresentable(stable_count))?;
        let mut records = Vec::with_capacity(capacity);
        for run in &manifest.record_runs {
            let page = decoded_pages
                .get(&run.page)
                .ok_or(GaussianLodgeResidentError::MissingPage(run.page))?;
            let start = usize::try_from(run.page_offset)
                .map_err(|_| GaussianLodgeResidentError::PageRangeOverflow(run.page))?;
            let end = run
                .page_offset
                .checked_add(run.count)
                .and_then(|end| usize::try_from(end).ok())
                .ok_or(GaussianLodgeResidentError::PageRangeOverflow(run.page))?;
            let slice = page
                .gaussians
                .get(start..end)
                .ok_or(GaussianLodgeResidentError::PageRangeOverflow(run.page))?;
            records.extend_from_slice(slice);
        }
        if records.len() != capacity {
            return Err(GaussianLodgeResidentError::CatalogCountMismatch {
                expected: stable_count,
                actual: records.len() as u64,
            });
        }

        let catalog_sha256 = gaussian_catalog_sha256(records.iter().copied());
        let catalog = assets.add(PlanarGaussian3d::from(records));
        let page_slots = authenticated_pages
            .iter()
            .copied()
            .enumerate()
            .map(|(index, page)| {
                let index = u32::try_from(index)
                    .map_err(|_| GaussianLodgeResidentError::PageCountNotRepresentable)?;
                Ok((
                    page,
                    AtlasSlot {
                        index,
                        generation: 1,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, GaussianLodgeResidentError>>()?;
        let resident_pages = u32::try_from(page_slots.len())
            .map_err(|_| GaussianLodgeResidentError::PageCountNotRepresentable)?;
        Ok(Self {
            catalog,
            manifest,
            memberships: Arc::new(decoded_memberships),
            page_slots: Arc::new(page_slots),
            resident_pages,
            catalog_sha256,
        })
    }

    /// Allocation-free preflight for the fully resident implementation.
    /// Applications should call this immediately after loading the sidecar,
    /// before fetching or decoding its page closure.
    pub fn validate_manifest_budget(
        manifest: &GaussianLodgeManifest,
        settings: &GaussianLodgeSettings,
    ) -> Result<(), GaussianLodgeResidentError> {
        manifest
            .validate()
            .map_err(|error| GaussianLodgeResidentError::Manifest(error.to_string()))?;
        settings
            .validate()
            .map_err(|error| GaussianLodgeResidentError::Settings(error.to_string()))?;
        let count = manifest.header.stable_gaussian_count;
        let representable = usize::try_from(count)
            .ok()
            .and_then(representable_source_count)
            .is_some();
        if !representable {
            return Err(GaussianLodgeResidentError::CatalogCountNotRepresentable(
                count,
            ));
        }
        if count > settings.budgets.max_resident_gaussians {
            return Err(GaussianLodgeResidentError::ResidentGaussianBudgetExceeded);
        }
        if lodge_resident_materialization_bytes(manifest)
            .is_none_or(|bytes| bytes > settings.budgets.max_resident_bytes)
        {
            return Err(GaussianLodgeResidentError::ResidentByteBudgetExceeded);
        }
        if manifest.page_authentication.len() as u64
            > u64::from(settings.budgets.max_resident_pages)
        {
            return Err(GaussianLodgeResidentError::ResidentPageBudgetExceeded);
        }
        Ok(())
    }

    pub fn catalog_handle(&self) -> &Handle<PlanarGaussian3d> {
        &self.catalog
    }

    pub fn manifest(&self) -> &GaussianLodgeManifest {
        &self.manifest
    }

    pub fn memberships(&self) -> impl Iterator<Item = &LodgeMembership> {
        self.memberships.values()
    }

    /// Optional stricter diagnostic, not a generic LODGE import requirement.
    /// It requires every selected record's entire camera-cluster sphere
    /// distance interval to remain inside the record's authored level band.
    pub fn audit_conservative_sphere_levels(
        &self,
        catalog: &PlanarGaussian3d,
    ) -> Result<(), GaussianLodgeResidentError> {
        if catalog.len() as u64 != self.manifest.header.stable_gaussian_count {
            return Err(GaussianLodgeResidentError::CatalogCountMismatch {
                expected: self.manifest.header.stable_gaussian_count,
                actual: catalog.len() as u64,
            });
        }
        for cluster in &self.manifest.clusters {
            let membership = self
                .memberships
                .get(&cluster.id)
                .ok_or(GaussianLodgeResidentError::MissingMembership(cluster.id))?;
            for &id in membership.ids() {
                let gaussian = catalog.get((id.0 - 1) as usize);
                let position = gaussian.position_visibility.position;
                let center_distance = squared_distance(position, cluster.center).sqrt() as f32;
                let lower = (center_distance - cluster.radius).max(0.0);
                let upper = center_distance + cluster.radius;
                let level = level_for_stable_id(&self.manifest, id)
                    .ok_or(GaussianLodgeResidentError::InvalidMembership(cluster.id))?;
                let descriptor = &self.manifest.levels[usize::from(level.0)];
                let next = self
                    .manifest
                    .levels
                    .get(usize::from(level.0) + 1)
                    .map(|level| level.distance_min)
                    .unwrap_or(f32::INFINITY);
                if lower < descriptor.distance_min || upper >= next {
                    return Err(GaussianLodgeResidentError::SphereLevelMismatch {
                        cluster: cluster.id,
                        gaussian: id,
                        level,
                        lower,
                        upper,
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Component, Default)]
struct GaussianLodgeResidentState {
    manifest: Option<Arc<GaussianLodgeManifest>>,
    catalog_id: Option<AssetId<PlanarGaussian3d>>,
    frozen_views: BTreeMap<Entity, [f32; 3]>,
    publications: BTreeMap<Entity, LodgePairPublicationState>,
    current: Option<LodRenderCandidates>,
    current_plans: BTreeMap<Entity, Arc<LodgePairCandidate>>,
    pending: Option<LodRenderCandidates>,
    pending_plans: BTreeMap<Entity, Arc<LodgePairCandidate>>,
    /// Render failures which are not rejected by the static policy preflight
    /// remain visible until their exact request changes or a recovered device
    /// generation becomes available. Rebuilding the same token every frame
    /// would hide the failure and spin pipeline/descriptor work indefinitely.
    failed_render_request: Option<LodgeFailedRenderRequest>,
}

/// Marks the hierarchy settings component as the private compatibility
/// adapter installed by the resident active-set path. Strategy-switch cleanup
/// can then restore hierarchy defaults without overwriting settings which the
/// application explicitly supplied during the switch.
#[derive(Component, Clone)]
struct GaussianLodgeRenderAdapter(GaussianLodSettings);

#[derive(Clone)]
struct LodgeFailedRenderRequest {
    plans: BTreeMap<Entity, Arc<LodgePairCandidate>>,
    frozen: bool,
    cloud_settings: CloudSettings,
    device_generation: u64,
    render_environment_epoch: u64,
    detail: String,
}

fn validate_lodge_external_render_settings(settings: &CloudSettings) -> Result<(), &'static str> {
    if settings.gaussian_mode != GaussianMode::Gaussian3d {
        return Err("LODGE external active sets require Gaussian3d rendering");
    }
    if settings.sort_mode != crate::sort::SortMode::Radix {
        return Err("LODGE external active sets require radix sorting");
    }
    if !crate::render::gaussian_rasterization_is_supported(
        settings.gaussian_mode,
        settings.rasterize_mode,
    ) {
        return Err("LODGE external active sets do not support this rasterization mode");
    }
    if settings.lod_debug.requires_metadata() {
        return Err("LODGE external active sets require the LoD debug preset Off");
    }
    Ok(())
}

fn reset_lodge_candidate_phases(candidates: Option<&LodRenderCandidates>) {
    let Some(candidates) = candidates else {
        return;
    };
    for candidate in candidates.by_camera.values() {
        candidate.phase.store(LOD_RENDER_WAITING, Ordering::Release);
    }
}

/// Hidden/no-camera clouds cannot retain an ACTIVE shared phase: the render
/// consumer which proved that phase no longer exists. Keep the complete
/// current pair as a reusable logical plan, but require a fresh radix
/// activation when a camera observes it again. A pending replacement has no
/// drawable ownership and is canceled outright.
fn suspend_lodge_render_publication(state: &mut GaussianLodgeResidentState) {
    reset_lodge_candidate_phases(state.current.as_ref());
    reset_lodge_candidate_phases(state.pending.as_ref());
    for publication in state.publications.values_mut() {
        publication.discard_pending();
    }
    state.pending = None;
    state.pending_plans.clear();
    // RenderWorld has discarded the old consumer. Reappearance is an explicit
    // fresh-retry event even when its selected pair is unchanged.
    state.failed_render_request = None;
}

/// Static-invalid render policy is fail-closed and must not resurrect a token
/// which was ACTIVE under an earlier policy. Preserve only camera-selection
/// state so Frozen remains a selection-view contract across presentation-only
/// recovery.
fn clear_lodge_render_publication(state: &mut GaussianLodgeResidentState) {
    suspend_lodge_render_publication(state);
    state.current = None;
    state.current_plans.clear();
    state.publications.clear();
}

/// Asset mutation invalidates the stable-ID proof permanently for that asset
/// identity. Re-materialization creates a fresh handle/identity.
#[derive(Resource, Default)]
struct InvalidatedLodgeCatalogs(HashSet<AssetId<PlanarGaussian3d>>);

#[derive(Clone, Copy)]
struct LodgeCameraObservation {
    entity: Entity,
    world_position: Vec3,
}

type LodgeCameraQueryItem = (
    Entity,
    &'static Camera,
    &'static GlobalTransform,
    Option<&'static VisibleEntities>,
);

/// Additive host plugin for fully resident LODGE artifacts.
///
/// The public [`GaussianLodgeHandle`] is also the exclusion marker used by the
/// hierarchy bridge/status queries, so the private compatibility settings
/// installed below can never trigger ephemeral MomentMerge construction.
#[derive(Default)]
pub struct GaussianLodgeResidentPlugin;

impl Plugin for GaussianLodgeResidentPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<GaussianLodRenderCommitPlugin>() {
            app.add_plugins(GaussianLodRenderCommitPlugin);
        }
        app.init_resource::<InvalidatedLodgeCatalogs>()
            .register_type::<GaussianLodgeSettings>()
            .register_type::<GaussianLodRepresentationKind>()
            .register_type::<GaussianLodgeStatus>()
            .register_type::<GaussianLodgeLifecycle>()
            .register_required_components::<GaussianLodgeHandle, GaussianLodgeSettings>()
            .register_required_components::<GaussianLodgeHandle, CloudSettings>()
            .register_required_components::<GaussianLodgeHandle, Transform>()
            .register_required_components::<GaussianLodgeHandle, Visibility>()
            .add_systems(
                PostUpdate,
                update_resident_lodge_clouds
                    .after(GaussianLodBridgeUpdate)
                    .after(GaussianLodPackageUpdate)
                    .after(CameraUpdateSystems)
                    .after(VisibilitySystems::CheckVisibility)
                    .after(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                cleanup_stale_resident_lodge_clouds
                    .before(GaussianLodBridgeUpdate)
                    .before(GaussianLodPackageUpdate),
            )
            .add_systems(
                PostUpdate,
                publish_lodge_status_transitions.after(update_resident_lodge_clouds),
            );
    }
}

fn publish_lodge_status_transitions(
    statuses: Query<(Entity, &GaussianLodgeStatus), Changed<GaussianLodgeStatus>>,
    mut removed: RemovedComponents<GaussianLodgeStatus>,
    mut previous: Local<
        BTreeMap<Entity, (GaussianLodgeLifecycle, Option<LodOrchestrationFailureCode>)>,
    >,
    mut recovery_pending: Local<BTreeSet<Entity>>,
    mut transitions: MessageWriter<LodOrchestrationTransition>,
) {
    for entity in removed.read() {
        previous.remove(&entity);
        recovery_pending.remove(&entity);
    }
    for (entity, status) in &statuses {
        let next = (status.lifecycle, status.failure_code);
        let old = previous.insert(entity, next);
        if old == Some(next) {
            continue;
        }
        if status.failure_code.is_some() {
            recovery_pending.insert(entity);
        }
        let kind = match status.lifecycle {
            GaussianLodgeLifecycle::Failed => Some(LodOrchestrationTransitionKind::Failed),
            GaussianLodgeLifecycle::Degraded => Some(LodOrchestrationTransitionKind::Degraded),
            GaussianLodgeLifecycle::Active if recovery_pending.contains(&entity) => {
                Some(LodOrchestrationTransitionKind::Recovered)
            }
            GaussianLodgeLifecycle::LoadingManifest
            | GaussianLodgeLifecycle::LoadingMembership
            | GaussianLodgeLifecycle::LoadingPages
            | GaussianLodgeLifecycle::WaitingForRender
            | GaussianLodgeLifecycle::Active => None,
        };
        if let Some(kind) = kind {
            transitions.write(LodOrchestrationTransition {
                entity,
                source: LodOrchestrationSource::ExternalActiveSets,
                kind,
                failure: status.failure_code.map(|code| {
                    status.failure.as_ref().map_or_else(
                        || LodOrchestrationFailure::new(code),
                        |detail| LodOrchestrationFailure::with_detail(code, detail.clone()),
                    )
                }),
            });
        }
        if status.lifecycle == GaussianLodgeLifecycle::Active && status.failure_code.is_none() {
            recovery_pending.remove(&entity);
        }
    }
}

#[allow(clippy::type_complexity)]
fn cleanup_stale_resident_lodge_clouds(
    mut commands: Commands,
    stale: Query<
        (
            Entity,
            &GaussianLodgeResidentState,
            Option<&PlanarGaussian3dHandle>,
            Option<&GaussianLodHandle>,
            Option<Ref<GaussianLodSettings>>,
            Option<Ref<GaussianLodgeRenderAdapter>>,
        ),
        Or<(
            Without<GaussianLodgeHandle>,
            Without<GaussianLodgeResidentCatalog>,
            Without<GaussianLodgeSettings>,
        )>,
    >,
) {
    for (entity, state, planar, hierarchy, settings, private_adapter) in &stale {
        let mut entity_commands = commands.entity(entity);
        if planar.is_some_and(|handle| Some(handle.handle().id()) == state.catalog_id) {
            entity_commands.remove::<PlanarGaussian3dHandle>();
        }
        entity_commands
            .remove::<GaussianLodgeResidentState>()
            .remove::<LodRenderCandidates>()
            .remove::<GaussianLodgeStatus>()
            .remove::<GaussianLodgeRenderAdapter>();
        if hierarchy.is_some() {
            let private_settings_owned = settings.as_ref().is_some_and(|settings| {
                private_adapter.as_ref().is_some_and(|adapter| {
                    adapter.0 == **settings && adapter.last_changed() == settings.last_changed()
                })
            });
            if settings.is_none() || private_settings_owned {
                entity_commands.insert(GaussianLodSettings::default());
            }
        } else {
            entity_commands.remove::<GaussianLodSettings>();
        }
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn update_resident_lodge_clouds(
    mut commands: Commands,
    lodge_assets: Res<Assets<GaussianLodgeAsset>>,
    catalogs: Res<Assets<PlanarGaussian3d>>,
    recovery_status: Option<Res<GaussianRenderRecoveryStatus>>,
    render_environment: Res<LodRenderEnvironmentEpoch>,
    mut catalog_events: MessageReader<AssetEvent<PlanarGaussian3d>>,
    mut invalidated_catalogs: ResMut<InvalidatedLodgeCatalogs>,
    cameras: Query<LodgeCameraQueryItem, With<GaussianCamera>>,
    mut unmaterialized_clouds: Query<
        (
            Entity,
            Option<&GaussianLodHandle>,
            Option<&mut GaussianLodgeResidentState>,
            Option<&mut GaussianLodgeStatus>,
        ),
        (
            With<GaussianLodgeHandle>,
            Without<GaussianLodgeResidentCatalog>,
        ),
    >,
    mut clouds: Query<(
        Entity,
        &GaussianLodgeHandle,
        &GaussianLodgeSettings,
        &CloudSettings,
        &GaussianLodgeResidentCatalog,
        &GlobalTransform,
        Option<&GaussianLodHandle>,
        Option<&PlanarGaussian3dHandle>,
        Option<&GaussianLodSettings>,
        Option<&GaussianLodgeRenderAdapter>,
        Option<&mut GaussianLodgeResidentState>,
        Option<&mut GaussianLodgeStatus>,
    )>,
) {
    let device_generation = recovery_status
        .as_deref()
        .map(GaussianRenderRecoveryStatus::snapshot)
        .map_or(0, |snapshot| snapshot.device_generation);
    let render_environment_epoch = render_environment.current();
    for (entity, legacy_handle, state, status) in &mut unmaterialized_clouds {
        if legacy_handle.is_some() {
            if let Some(mut state) = state {
                clear_lodge_render_publication(&mut state);
            }
            publish_lodge_failure(
                &mut commands,
                entity,
                status.map(|status| status.into_inner()),
                LodOrchestrationFailureCode::InvalidConfiguration,
                "GaussianLodgeHandle conflicts with GaussianLodHandle",
            );
            continue;
        }
        if let Some(mut state) = state {
            suspend_lodge_render_publication(&mut state);
        }
        publish_lodge_waiting(
            &mut commands,
            entity,
            status.map(|status| status.into_inner()),
            GaussianLodgeLifecycle::LoadingPages,
        );
    }
    let modified_catalogs = catalog_events
        .read()
        .filter_map(|event| match event {
            AssetEvent::Modified { id } | AssetEvent::Removed { id } => Some(*id),
            AssetEvent::Added { .. }
            | AssetEvent::LoadedWithDependencies { .. }
            | AssetEvent::Unused { .. } => None,
        })
        .collect::<HashSet<_>>();
    let mut live_catalogs = HashSet::new();
    for (
        entity,
        lodge_handle,
        settings,
        cloud_settings,
        resident,
        transform,
        legacy_handle,
        planar_handle,
        adapter,
        private_adapter,
        mut state,
        status,
    ) in &mut clouds
    {
        live_catalogs.insert(resident.catalog.id());
        let mut status = status.map(|status| status.into_inner());
        if legacy_handle.is_some() {
            if let Some(state) = state.as_deref_mut() {
                clear_lodge_render_publication(state);
            }
            publish_lodge_failure(
                &mut commands,
                entity,
                status.as_deref_mut(),
                LodOrchestrationFailureCode::InvalidConfiguration,
                "GaussianLodgeHandle conflicts with GaussianLodHandle",
            );
            continue;
        }
        let Some(asset) = lodge_assets.get(&lodge_handle.0) else {
            if let Some(state) = state.as_deref_mut() {
                suspend_lodge_render_publication(state);
            }
            publish_lodge_waiting(
                &mut commands,
                entity,
                status.as_deref_mut(),
                GaussianLodgeLifecycle::LoadingManifest,
            );
            continue;
        };
        if modified_catalogs.contains(&resident.catalog.id()) {
            invalidated_catalogs.0.insert(resident.catalog.id());
        }
        let new_catalog_binding = state
            .as_ref()
            .is_none_or(|state| state.catalog_id != Some(resident.catalog.id()));
        if new_catalog_binding
            && catalogs.get(&resident.catalog).is_some_and(|catalog| {
                gaussian_catalog_sha256(catalog.iter()) != resident.catalog_sha256
            })
        {
            invalidated_catalogs.0.insert(resident.catalog.id());
        }
        if invalidated_catalogs.0.contains(&resident.catalog.id()) {
            if let Some(state) = state.as_deref_mut() {
                clear_lodge_render_publication(state);
            }
            publish_lodge_failure(
                &mut commands,
                entity,
                status.as_deref_mut(),
                LodOrchestrationFailureCode::DecodeValidationFailed,
                "authenticated LODGE catalog asset was modified or removed; rematerialize it",
            );
            continue;
        }
        if !Arc::ptr_eq(&asset.shared_manifest(), &resident.manifest) {
            if let Some(state) = state.as_deref_mut() {
                clear_lodge_render_publication(state);
            }
            publish_lodge_failure(
                &mut commands,
                entity,
                status.as_deref_mut(),
                LodOrchestrationFailureCode::DecodeValidationFailed,
                "resident catalog was not materialized from this LODGE asset identity",
            );
            continue;
        }
        let Some(catalog) = catalogs.get(&resident.catalog) else {
            if let Some(state) = state.as_deref_mut() {
                suspend_lodge_render_publication(state);
            }
            publish_lodge_waiting(
                &mut commands,
                entity,
                status.as_deref_mut(),
                GaussianLodgeLifecycle::LoadingPages,
            );
            continue;
        };
        if let Err(error) = validate_resident_budget(settings, resident, catalog) {
            if let Some(state) = state.as_deref_mut() {
                clear_lodge_render_publication(state);
            }
            publish_lodge_failure(
                &mut commands,
                entity,
                status.as_deref_mut(),
                lodge_resident_error_code(&error),
                &error.to_string(),
            );
            continue;
        }
        if let Err(error) = validate_lodge_external_render_settings(cloud_settings) {
            if let Some(state) = state.as_deref_mut() {
                clear_lodge_render_publication(state);
            }
            publish_lodge_failure(
                &mut commands,
                entity,
                status.as_deref_mut(),
                LodOrchestrationFailureCode::UnsupportedConfiguration,
                error,
            );
            continue;
        }

        if planar_handle.is_none_or(|handle| handle.handle().id() != resident.catalog.id()) {
            commands
                .entity(entity)
                .insert(PlanarGaussian3dHandle(resident.catalog.clone()));
        }
        let internal_settings = lodge_render_adapter(settings);
        if adapter != Some(&internal_settings)
            || private_adapter.is_none_or(|adapter| adapter.0 != internal_settings)
        {
            commands.entity(entity).insert((
                internal_settings.clone(),
                GaussianLodgeRenderAdapter(internal_settings),
            ));
        }
        if state.is_none() {
            commands
                .entity(entity)
                .insert(GaussianLodgeResidentState::default())
                .remove::<LodRenderCandidates>()
                .remove::<GaussianLodBridgeStatus>()
                .remove::<LodDebugMetadata>();
            if status.is_none() {
                commands
                    .entity(entity)
                    .insert(GaussianLodgeStatus::default());
            }
            continue;
        }
        if status.is_none() {
            commands
                .entity(entity)
                .insert(GaussianLodgeStatus::default());
            continue;
        }
        let state = state.unwrap().into_inner();
        let status = status.unwrap();
        if state.catalog_id != Some(resident.catalog.id())
            || state
                .manifest
                .as_ref()
                .is_none_or(|manifest| !Arc::ptr_eq(manifest, &resident.manifest))
        {
            *state = GaussianLodgeResidentState {
                manifest: Some(resident.manifest.clone()),
                catalog_id: Some(resident.catalog.id()),
                ..Default::default()
            };
        }

        if state
            .pending
            .as_ref()
            .is_some_and(|pending| pending.by_camera.values().any(LodRenderCandidate::failed))
        {
            for publication in state.publications.values_mut() {
                publication.discard_pending();
            }
            state.pending = None;
            state.failed_render_request = Some(LodgeFailedRenderRequest {
                plans: std::mem::take(&mut state.pending_plans),
                frozen: settings.selection_mode == LodSelectionMode::Frozen,
                cloud_settings: cloud_settings.clone(),
                device_generation,
                render_environment_epoch,
                detail: "LODGE render candidate failed; retained pair preserved".into(),
            });
        }
        if state.pending.as_ref().is_some_and(|pending| {
            !pending.by_camera.is_empty()
                && pending
                    .by_camera
                    .values()
                    .all(LodRenderCandidate::render_is_active)
        }) {
            for publication in state.publications.values_mut() {
                let _ = publication
                    .commit_pending_if_drawable(|page| resident.page_slots.contains_key(&page));
            }
            let mut current = state.pending.take().unwrap();
            current.retained_current = true;
            current.candidates_are_current = true;
            current.retained_current_is_stale = false;
            state.current = Some(current);
            state.current_plans = std::mem::take(&mut state.pending_plans);
        }
        if state
            .current
            .as_ref()
            .is_some_and(|current| current.by_camera.values().any(LodRenderCandidate::failed))
        {
            // A failed token cannot recover in place. Latch the exact request
            // instead of rebuilding an equivalent token every update; a
            // changed request or recovered device generation retries below.
            state.current = None;
            state.failed_render_request = Some(LodgeFailedRenderRequest {
                plans: std::mem::take(&mut state.current_plans),
                frozen: settings.selection_mode == LodSelectionMode::Frozen,
                cloud_settings: cloud_settings.clone(),
                device_generation,
                render_environment_epoch,
                detail: "LODGE current drawable was invalidated".into(),
            });
            state.publications.clear();
        }

        let observations = match lodge_camera_observations(&cameras, entity) {
            Ok(observations) => observations,
            Err(error) => {
                publish_lodge_request_failure(
                    &mut commands,
                    entity,
                    Some(&mut *status),
                    lodge_resident_error_code(&error),
                    &error.to_string(),
                    state.current.as_ref(),
                );
                continue;
            }
        };
        if observations.is_empty() {
            suspend_lodge_render_publication(state);
            publish_lodge_waiting(
                &mut commands,
                entity,
                Some(&mut *status),
                GaussianLodgeLifecycle::WaitingForRender,
            );
            commands
                .entity(entity)
                .insert(empty_required_candidates(state.current.is_some()));
            continue;
        }

        state
            .frozen_views
            .retain(|camera, _| observations.iter().any(|view| view.entity == *camera));
        let inverse = transform.affine().inverse();
        let limits = LodgePairLimits {
            max_union_gaussians: settings.budgets.max_active_gaussians,
            max_classified_runs: settings.budgets.max_traversal_nodes_per_view,
            max_required_pages: settings.budgets.max_resident_pages,
        };
        let locations = LodgeRecordLocationResolver::from_validated(&resident.manifest.record_runs);
        let mut desired = BTreeMap::new();
        let mut pair_templates = BTreeMap::<LodgePairIdentity, Arc<LodgePairCandidate>>::new();
        for observation in &observations {
            let live_local = inverse
                .transform_point3(observation.world_position)
                .to_array();
            let local = match settings.selection_mode {
                LodSelectionMode::Dynamic => {
                    state.frozen_views.remove(&observation.entity);
                    live_local
                }
                LodSelectionMode::Frozen => *state
                    .frozen_views
                    .entry(observation.entity)
                    .or_insert(live_local),
            };
            let retained = state
                .current_plans
                .get(&observation.entity)
                .or_else(|| state.pending_plans.get(&observation.entity));
            let selection = match select_lodge_pair_with_hysteresis(
                local,
                &resident.manifest.clusters,
                retained.map(|candidate| candidate.identity()),
                settings.pair_hysteresis,
            ) {
                Ok(selection) => selection,
                Err(error) => {
                    publish_lodge_request_failure(
                        &mut commands,
                        entity,
                        Some(&mut *status),
                        lodge_resident_error_code(&error),
                        &error.to_string(),
                        state.current.as_ref(),
                    );
                    desired.clear();
                    break;
                }
            };
            let reusable = retained
                .filter(|candidate| candidate.identity() == selection.identity)
                .or_else(|| pair_templates.get(&selection.identity))
                .or_else(|| {
                    state
                        .current_plans
                        .values()
                        .chain(state.pending_plans.values())
                        .find(|candidate| candidate.identity() == selection.identity)
                });
            let candidate = if let Some(reusable) = reusable {
                reusable.retarget(selection)
            } else {
                let Some(first) = resident.memberships.get(&selection.identity.first) else {
                    desired.clear();
                    break;
                };
                let Some(second) = resident.memberships.get(&selection.identity.second) else {
                    desired.clear();
                    break;
                };
                build_lodge_pair_candidate(selection, first, second, &locations, limits)
            };
            match candidate {
                Ok(candidate) => {
                    let candidate = Arc::new(candidate);
                    pair_templates
                        .entry(selection.identity)
                        .or_insert_with(|| Arc::clone(&candidate));
                    desired.insert(observation.entity, candidate);
                }
                Err(error) => {
                    publish_lodge_request_failure(
                        &mut commands,
                        entity,
                        Some(&mut *status),
                        lodge_plan_error_code(&error),
                        &error.to_string(),
                        state.current.as_ref(),
                    );
                    desired.clear();
                    break;
                }
            }
        }
        if desired.len() != observations.len() {
            if let Some(current) = state.current.as_ref() {
                let mut retained = current.clone();
                retained.retained_current = true;
                retained.candidates_are_current = true;
                retained.retained_current_is_stale = true;
                commands.entity(entity).insert(retained);
            } else {
                commands
                    .entity(entity)
                    .insert(empty_required_candidates(false));
            }
            continue;
        }
        if let Err(error) = lodge_multi_view_page_demand(
            desired.values().map(Arc::as_ref),
            settings.budgets.max_resident_pages,
        ) {
            publish_lodge_request_failure(
                &mut commands,
                entity,
                Some(&mut *status),
                lodge_plan_error_code(&error),
                &error.to_string(),
                state.current.as_ref(),
            );
            continue;
        }

        if let Some(failed) = state.failed_render_request.as_ref() {
            if lodge_failed_render_request_matches(
                failed,
                &desired,
                settings.selection_mode == LodSelectionMode::Frozen,
                cloud_settings,
                device_generation,
                render_environment_epoch,
            ) {
                publish_lodge_request_failure(
                    &mut commands,
                    entity,
                    Some(&mut *status),
                    LodOrchestrationFailureCode::RenderCommitFailed,
                    &failed.detail,
                    state.current.as_ref(),
                );
                continue;
            }
            state.failed_render_request = None;
        }

        let same_current = same_plan_unions(&desired, &state.current_plans);
        let same_pending = same_plan_unions(&desired, &state.pending_plans);
        let frozen = settings.selection_mode == LodSelectionMode::Frozen;
        let current_policy_matches = state.current.as_ref().is_some_and(|candidates| {
            candidates
                .by_camera
                .values()
                .all(|candidate| candidate.frontier().selection_view_frozen() == frozen)
        });
        let pending_policy_matches = state.pending.as_ref().is_some_and(|candidates| {
            candidates
                .by_camera
                .values()
                .all(|candidate| candidate.frontier().selection_view_frozen() == frozen)
        });
        if same_current && current_policy_matches && state.pending.is_none() {
            state.current_plans = desired;
            let current = state.current.as_ref().unwrap().clone();
            let render_active = current
                .by_camera
                .values()
                .all(LodRenderCandidate::render_is_active);
            commands.entity(entity).insert(current);
            update_lodge_status(
                status,
                &state.current_plans,
                if render_active {
                    GaussianLodgeLifecycle::Active
                } else {
                    GaussianLodgeLifecycle::WaitingForRender
                },
                render_active,
                false,
            );
            continue;
        }
        if same_pending && pending_policy_matches {
            state.pending_plans = desired;
            let pending = state.pending.as_ref().unwrap().clone();
            commands.entity(entity).insert(pending);
            update_lodge_status(
                status,
                &state.pending_plans,
                GaussianLodgeLifecycle::WaitingForRender,
                false,
                state.current.is_some(),
            );
            continue;
        }

        let candidate_set = match build_external_candidate_set(
            &desired,
            resident,
            frozen,
            state.current.as_ref(),
        ) {
            Ok(candidate_set) => candidate_set,
            Err(error) => {
                publish_lodge_request_failure(
                    &mut commands,
                    entity,
                    Some(&mut *status),
                    lodge_resident_error_code(&error),
                    &error.to_string(),
                    state.current.as_ref(),
                );
                continue;
            }
        };
        if same_current && !current_policy_matches {
            state.publications.clear();
        }
        for (&camera, candidate) in &desired {
            state
                .publications
                .entry(camera)
                .or_default()
                .stage(candidate.clone());
        }
        state
            .publications
            .retain(|camera, _| desired.contains_key(camera));
        state.pending = Some(candidate_set.clone());
        state.pending_plans = desired;
        commands.entity(entity).insert(candidate_set);
        update_lodge_status(
            status,
            &state.pending_plans,
            GaussianLodgeLifecycle::WaitingForRender,
            false,
            state.current.is_some(),
        );
    }
    invalidated_catalogs
        .0
        .retain(|catalog| live_catalogs.contains(catalog));
}

fn lodge_camera_observations(
    cameras: &Query<LodgeCameraQueryItem, With<GaussianCamera>>,
    cloud: Entity,
) -> Result<Vec<LodgeCameraObservation>, GaussianLodgeResidentError> {
    let mut observations = Vec::new();
    for (entity, camera, transform, visible_entities) in cameras {
        if !camera.is_active
            || visible_entities.is_some_and(|visible| {
                !visible
                    .iter(TypeId::of::<CloudVisibilityClass>())
                    .any(|visible_cloud| *visible_cloud == cloud)
            })
        {
            continue;
        }
        if observations.len() == MAX_RESIDENT_LODGE_VIEWS {
            return Err(GaussianLodgeResidentError::ViewLimitExceeded {
                limit: MAX_RESIDENT_LODGE_VIEWS as u32,
            });
        }
        observations.push(LodgeCameraObservation {
            entity,
            world_position: transform.translation(),
        });
    }
    observations.sort_by_key(|observation| observation.entity);
    Ok(observations)
}

fn build_external_candidate_set(
    plans: &BTreeMap<Entity, Arc<LodgePairCandidate>>,
    resident: &GaussianLodgeResidentCatalog,
    frozen: bool,
    current: Option<&LodRenderCandidates>,
) -> Result<LodRenderCandidates, GaussianLodgeResidentError> {
    let mut candidates = LodRenderCandidates::package_required();
    candidates.retained_current = current.is_some();
    candidates.candidates_are_current = false;
    candidates.retained_current_is_stale = current.is_some();
    for (&camera, plan) in plans {
        let mut ranges = Vec::with_capacity(plan.runs().len());
        let mut classes = Vec::with_capacity(plan.runs().len());
        for run in plan.runs() {
            let slot = *resident
                .page_slots
                .get(&run.page)
                .ok_or(GaussianLodgeResidentError::MissingPage(run.page))?;
            let physical_start = run
                .first_id
                .0
                .checked_sub(1)
                .and_then(|start| u32::try_from(start).ok())
                .ok_or(GaussianLodgeResidentError::CatalogCountNotRepresentable(
                    resident.manifest.header.stable_gaussian_count,
                ))?;
            ranges.push(LodPhysicalRange {
                node: LodNodeId(run.first_id.0),
                page: run.page,
                slot,
                physical_start,
                count: run.count,
            });
            classes.push(run.class);
        }
        let frontier = LodCandidateFrontier::complete_external_active_set(
            LodRuntimeViewId(camera.to_bits()),
            ranges,
            frozen,
        )
        .ok_or(GaussianLodgeResidentError::InvalidExternalFrontier)?;
        let first_center = cluster_center(&resident.manifest, plan.identity().first)?;
        let second_center = cluster_center(&resident.manifest, plan.identity().second)?;
        let presentation = LodExternalActiveSetPresentation::new(
            plan.identity(),
            first_center,
            second_center,
            classes,
        )
        .and_then(|presentation| {
            if frozen {
                presentation.with_frozen_second_weight(plan.second_weight())
            } else {
                Some(presentation)
            }
        })
        .ok_or(GaussianLodgeResidentError::InvalidExternalPresentation)?;
        let mut candidate = LodRenderCandidate::new_external_active_set(frontier, presentation)
            .ok_or(GaussianLodgeResidentError::InvalidExternalPresentation)?;
        if let Some(previous) = current
            .and_then(|current| current.by_camera.get(&camera))
            .filter(|previous| previous.render_is_active() && previous.same_payload(&candidate))
        {
            candidate.inherit_active_payload_state(previous);
        }
        candidates.by_camera.insert(camera, candidate);
    }
    Ok(candidates)
}

fn select_lodge_pair_with_hysteresis(
    view: [f32; 3],
    clusters: &[LodgeCameraCluster],
    retained: Option<LodgePairIdentity>,
    hysteresis: f32,
) -> Result<LodgePairSelection, GaussianLodgeResidentError> {
    let raw = select_lodge_pair_from_validated_clusters(view, clusters)
        .map_err(|error| GaussianLodgeResidentError::Planning(error.to_string()))?;
    if hysteresis.to_bits() == 0.0_f32.to_bits() {
        return Ok(raw);
    }
    let Some(retained) = retained else {
        return Ok(raw);
    };
    if raw.identity == retained || (raw.nearest != retained.first && raw.nearest != retained.second)
    {
        return Ok(raw);
    }
    let retained_other = if raw.nearest == retained.first {
        retained.second
    } else {
        retained.first
    };
    let raw_other = if raw.nearest == raw.identity.first {
        raw.identity.second
    } else {
        raw.identity.first
    };
    if retained_other == raw_other {
        return Ok(raw);
    }
    let retained_center = cluster_center_slice(clusters, retained_other)?;
    let raw_center = cluster_center_slice(clusters, raw_other)?;
    let retained_distance = squared_distance(view, retained_center).sqrt();
    let raw_distance = squared_distance(view, raw_center).sqrt();
    let advantage = if retained_distance == 0.0 {
        0.0
    } else {
        ((retained_distance - raw_distance) / retained_distance).max(0.0)
    };
    if advantage > f64::from(hysteresis) {
        return Ok(raw);
    }
    let identity = if raw.nearest <= retained_other {
        LodgePairIdentity {
            first: raw.nearest,
            second: retained_other,
        }
    } else {
        LodgePairIdentity {
            first: retained_other,
            second: raw.nearest,
        }
    };
    let first_center = cluster_center_slice(clusters, identity.first)?;
    let second_center = cluster_center_slice(clusters, identity.second)?;
    Ok(LodgePairSelection {
        identity,
        nearest: raw.nearest,
        second_weight: projected_center_line_weight(view, first_center, second_center)
            .map_err(|error| GaussianLodgeResidentError::Planning(error.to_string()))?,
    })
}

fn same_plan_unions(
    left: &BTreeMap<Entity, Arc<LodgePairCandidate>>,
    right: &BTreeMap<Entity, Arc<LodgePairCandidate>>,
) -> bool {
    left.len() == right.len()
        && left.iter().all(|(camera, left)| {
            right
                .get(camera)
                .is_some_and(|right| left.same_union(right))
        })
}

/// Cloud-owned portion of the renderer specialization/request identity. A
/// successful device recovery is tracked separately by generation; all cloud
/// shader/sort choices which can make the same external candidate admissible
/// are compared here.
fn lodge_render_policy_matches(left: &CloudSettings, right: &CloudSettings) -> bool {
    left.aabb == right.aabb
        && left.opacity_adaptive_radius == right.opacity_adaptive_radius
        && left.visualize_bounding_box == right.visualize_bounding_box
        && left.sort_mode == right.sort_mode
        && left.radix_sort_depth_bits == right.radix_sort_depth_bits
        && left.draw_mode == right.draw_mode
        && left.gaussian_mode == right.gaussian_mode
        && left.rasterize_mode == right.rasterize_mode
        && left.lod_debug == right.lod_debug
}

fn lodge_failed_render_request_matches(
    failed: &LodgeFailedRenderRequest,
    plans: &BTreeMap<Entity, Arc<LodgePairCandidate>>,
    frozen: bool,
    cloud_settings: &CloudSettings,
    device_generation: u64,
    render_environment_epoch: u64,
) -> bool {
    same_plan_unions(plans, &failed.plans)
        && failed.frozen == frozen
        && lodge_render_policy_matches(&failed.cloud_settings, cloud_settings)
        && failed.device_generation == device_generation
        && failed.render_environment_epoch == render_environment_epoch
}

fn lodge_render_adapter(settings: &GaussianLodgeSettings) -> GaussianLodSettings {
    GaussianLodSettings {
        quality: 0.0,
        selection_mode: settings.selection_mode,
        budgets: settings.budgets,
        hysteresis: 0.0,
        frustum_culling: settings.frustum_culling,
        frustum_margin: settings.frustum_margin,
    }
}

fn validate_resident_budget(
    settings: &GaussianLodgeSettings,
    resident: &GaussianLodgeResidentCatalog,
    catalog: &PlanarGaussian3d,
) -> Result<(), GaussianLodgeResidentError> {
    settings
        .validate()
        .map_err(|error| GaussianLodgeResidentError::Settings(error.to_string()))?;
    let count = catalog.len() as u64;
    if count != resident.manifest.header.stable_gaussian_count {
        return Err(GaussianLodgeResidentError::CatalogCountMismatch {
            expected: resident.manifest.header.stable_gaussian_count,
            actual: count,
        });
    }
    if count > settings.budgets.max_resident_gaussians {
        return Err(GaussianLodgeResidentError::ResidentGaussianBudgetExceeded);
    }
    if lodge_resident_materialization_bytes(&resident.manifest)
        .is_none_or(|bytes| bytes > settings.budgets.max_resident_bytes)
    {
        return Err(GaussianLodgeResidentError::ResidentByteBudgetExceeded);
    }
    if resident.resident_pages > settings.budgets.max_resident_pages {
        return Err(GaussianLodgeResidentError::ResidentPageBudgetExceeded);
    }
    Ok(())
}

/// Conservative peak for the resident constructor: decoded page records and
/// the canonical planar copy coexist while records are materialized, and the
/// authenticated membership object plus decoded stable IDs may coexist while
/// cluster memberships are produced.
fn lodge_resident_materialization_bytes(manifest: &GaussianLodgeManifest) -> Option<u64> {
    let catalog = manifest
        .header
        .stable_gaussian_count
        .checked_mul(size_of::<Gaussian3d>() as u64)?;
    let decoded_memberships = manifest
        .header
        .total_membership_ids
        .checked_mul(size_of::<LodgeGaussianId>() as u64)?;
    catalog
        .checked_mul(2)?
        .checked_add(manifest.membership_index.object.encoded_len)?
        .checked_add(decoded_memberships)
}

fn update_lodge_status(
    status: &mut GaussianLodgeStatus,
    plans: &BTreeMap<Entity, Arc<LodgePairCandidate>>,
    lifecycle: GaussianLodgeLifecycle,
    target_satisfied: bool,
    retained_stale_pair: bool,
) {
    let Some((_, primary)) = plans.first_key_value() else {
        return;
    };
    let previous_revision = status.revision;
    let previous_state = (
        status.representation,
        status.lifecycle,
        status.target_satisfied,
        status.retained_stale_pair,
        status.failure_code,
        status.failure.clone(),
        status.required_pages,
        status.resident_required_pages,
        status.visible_views,
        status.distinct_pairs,
    );
    status.observe_pair(
        primary.identity(),
        primary.selection().nearest.0,
        primary.second_weight(),
        primary.counts(),
    );
    status.representation = GaussianLodRepresentationKind::LodgeActiveSets;
    status.lifecycle = lifecycle;
    status.target_satisfied = target_satisfied;
    status.retained_stale_pair = retained_stale_pair;
    status.failure_code = None;
    status.failure = None;
    let required_pages = plans
        .values()
        .flat_map(|candidate| candidate.required_pages())
        .copied()
        .collect::<BTreeSet<_>>();
    status.required_pages = u32::try_from(required_pages.len()).unwrap_or(u32::MAX);
    // This path admits only a fully materialized canonical catalog, so every
    // pair-required page is resident before candidate construction.
    status.resident_required_pages = status.required_pages;
    status.visible_views = u32::try_from(plans.len()).unwrap_or(u32::MAX);
    status.distinct_pairs = u32::try_from(
        plans
            .values()
            .map(|candidate| candidate.identity())
            .collect::<BTreeSet<_>>()
            .len(),
    )
    .unwrap_or(u32::MAX);
    if status.revision == previous_revision
        && previous_state
            != (
                status.representation,
                status.lifecycle,
                status.target_satisfied,
                status.retained_stale_pair,
                status.failure_code,
                status.failure.clone(),
                status.required_pages,
                status.resident_required_pages,
                status.visible_views,
                status.distinct_pairs,
            )
    {
        status.revision = status.revision.saturating_add(1);
    }
}

fn empty_required_candidates(retained: bool) -> LodRenderCandidates {
    let mut candidates = LodRenderCandidates::package_required();
    candidates.retained_current = retained;
    candidates.candidates_are_current = false;
    candidates.retained_current_is_stale = retained;
    candidates
}

fn publish_lodge_waiting(
    commands: &mut Commands,
    entity: Entity,
    status: Option<&mut GaussianLodgeStatus>,
    lifecycle: GaussianLodgeLifecycle,
) {
    commands
        .entity(entity)
        .insert(empty_required_candidates(false));
    if let Some(status) = status {
        let presentation_changed = status.clear_presentation();
        let changed = status.representation != GaussianLodRepresentationKind::LodgeActiveSets
            || status.lifecycle != lifecycle
            || status.target_satisfied
            || status.retained_stale_pair
            || status.failure_code.is_some()
            || status.failure.is_some()
            || presentation_changed;
        status.representation = GaussianLodRepresentationKind::LodgeActiveSets;
        status.lifecycle = lifecycle;
        status.target_satisfied = false;
        status.retained_stale_pair = false;
        status.failure_code = None;
        status.failure = None;
        if changed {
            status.revision = status.revision.saturating_add(1);
        }
    } else {
        commands.entity(entity).insert(GaussianLodgeStatus {
            lifecycle,
            ..Default::default()
        });
    }
}

fn publish_lodge_failure(
    commands: &mut Commands,
    entity: Entity,
    status: Option<&mut GaussianLodgeStatus>,
    code: LodOrchestrationFailureCode,
    detail: &str,
) {
    commands
        .entity(entity)
        .insert(empty_required_candidates(false));
    if let Some(status) = status {
        let failure = detail.to_owned();
        let presentation_changed = status.clear_presentation();
        let changed = status.representation != GaussianLodRepresentationKind::LodgeActiveSets
            || status.lifecycle != GaussianLodgeLifecycle::Failed
            || status.target_satisfied
            || status.retained_stale_pair
            || status.failure_code != Some(code)
            || status.failure.as_ref() != Some(&failure)
            || presentation_changed;
        status.representation = GaussianLodRepresentationKind::LodgeActiveSets;
        status.lifecycle = GaussianLodgeLifecycle::Failed;
        status.target_satisfied = false;
        status.retained_stale_pair = false;
        status.failure_code = Some(code);
        status.failure = Some(failure);
        if changed {
            status.revision = status.revision.saturating_add(1);
        }
    } else {
        commands.entity(entity).insert(GaussianLodgeStatus {
            lifecycle: GaussianLodgeLifecycle::Failed,
            failure_code: Some(code),
            failure: Some(detail.to_owned()),
            ..Default::default()
        });
    }
}

fn publish_lodge_request_failure(
    commands: &mut Commands,
    entity: Entity,
    status: Option<&mut GaussianLodgeStatus>,
    code: LodOrchestrationFailureCode,
    detail: &str,
    current: Option<&LodRenderCandidates>,
) {
    let Some(current) = current else {
        publish_lodge_failure(commands, entity, status, code, detail);
        return;
    };
    let mut retained = current.clone();
    retained.retained_current = true;
    retained.candidates_are_current = true;
    retained.retained_current_is_stale = true;
    commands.entity(entity).insert(retained);
    if let Some(status) = status {
        let failure = detail.to_owned();
        let changed = status.representation != GaussianLodRepresentationKind::LodgeActiveSets
            || status.lifecycle != GaussianLodgeLifecycle::Degraded
            || status.target_satisfied
            || !status.retained_stale_pair
            || status.failure_code != Some(code)
            || status.failure.as_ref() != Some(&failure);
        status.representation = GaussianLodRepresentationKind::LodgeActiveSets;
        status.lifecycle = GaussianLodgeLifecycle::Degraded;
        status.target_satisfied = false;
        status.retained_stale_pair = true;
        status.failure_code = Some(code);
        status.failure = Some(failure);
        if changed {
            status.revision = status.revision.saturating_add(1);
        }
    }
}

fn lodge_plan_error_code(error: &LodgePlanError) -> LodOrchestrationFailureCode {
    match error {
        LodgePlanError::NonFiniteView | LodgePlanError::ZeroLimit => {
            LodOrchestrationFailureCode::InvalidConfiguration
        }
        LodgePlanError::CountOverflow
        | LodgePlanError::UnionLimitExceeded { .. }
        | LodgePlanError::RangeLimitExceeded { .. }
        | LodgePlanError::PageLimitExceeded { .. } => LodOrchestrationFailureCode::CapacityExceeded,
        LodgePlanError::InsufficientClusters
        | LodgePlanError::InvalidCluster
        | LodgePlanError::EmptyMembership
        | LodgePlanError::PairMembershipMismatch
        | LodgePlanError::CoincidentClusterCenters
        | LodgePlanError::MembershipNotStrictlySorted
        | LodgePlanError::InvalidRecordRun
        | LodgePlanError::MissingRecordLocation(_) => {
            LodOrchestrationFailureCode::DecodeValidationFailed
        }
    }
}

fn lodge_resident_error_code(error: &GaussianLodgeResidentError) -> LodOrchestrationFailureCode {
    match error {
        GaussianLodgeResidentError::Settings(_) => {
            LodOrchestrationFailureCode::InvalidConfiguration
        }
        GaussianLodgeResidentError::CatalogCountNotRepresentable(_)
        | GaussianLodgeResidentError::PageCountNotRepresentable
        | GaussianLodgeResidentError::ResidentGaussianBudgetExceeded
        | GaussianLodgeResidentError::ResidentByteBudgetExceeded
        | GaussianLodgeResidentError::ResidentPageBudgetExceeded
        | GaussianLodgeResidentError::ViewLimitExceeded { .. } => {
            LodOrchestrationFailureCode::CapacityExceeded
        }
        GaussianLodgeResidentError::Planning(_) => LodOrchestrationFailureCode::RuntimeFailed,
        GaussianLodgeResidentError::InvalidExternalFrontier
        | GaussianLodgeResidentError::InvalidExternalPresentation => {
            LodOrchestrationFailureCode::InternalInvariant
        }
        GaussianLodgeResidentError::Manifest(_)
        | GaussianLodgeResidentError::UnauthenticatedBaseManifest
        | GaussianLodgeResidentError::PageCodec(_)
        | GaussianLodgeResidentError::MembershipCodec(_)
        | GaussianLodgeResidentError::InvalidMembershipIndexRange
        | GaussianLodgeResidentError::MembershipIndexHashMismatch
        | GaussianLodgeResidentError::MembershipObjectMismatch
        | GaussianLodgeResidentError::DuplicatePage(_)
        | GaussianLodgeResidentError::UnexpectedPage(_)
        | GaussianLodgeResidentError::MissingPage(_)
        | GaussianLodgeResidentError::UnauthenticatedPage(_)
        | GaussianLodgeResidentError::InvalidPage { .. }
        | GaussianLodgeResidentError::IncompletePageClosure
        | GaussianLodgeResidentError::PageRangeOverflow(_)
        | GaussianLodgeResidentError::DuplicateMembership(_)
        | GaussianLodgeResidentError::UnexpectedMembership(_)
        | GaussianLodgeResidentError::MissingMembership(_)
        | GaussianLodgeResidentError::InvalidMembership(_)
        | GaussianLodgeResidentError::IncompleteMembershipClosure
        | GaussianLodgeResidentError::MissingCluster(_)
        | GaussianLodgeResidentError::InsufficientClusters
        | GaussianLodgeResidentError::CoincidentClusterCenters { .. }
        | GaussianLodgeResidentError::CatalogCountMismatch { .. }
        | GaussianLodgeResidentError::SphereLevelMismatch { .. } => {
            LodOrchestrationFailureCode::DecodeValidationFailed
        }
    }
}

fn cluster_center(
    manifest: &GaussianLodgeManifest,
    id: LodgeClusterId,
) -> Result<[f32; 3], GaussianLodgeResidentError> {
    cluster_center_slice(&manifest.clusters, id)
}

fn cluster_center_slice(
    clusters: &[LodgeCameraCluster],
    id: LodgeClusterId,
) -> Result<[f32; 3], GaussianLodgeResidentError> {
    clusters
        .binary_search_by_key(&id, |cluster| cluster.id)
        .ok()
        .map(|index| clusters[index].center)
        .ok_or(GaussianLodgeResidentError::MissingCluster(id))
}

fn level_for_stable_id(
    manifest: &GaussianLodgeManifest,
    id: LodgeGaussianId,
) -> Option<LodgeLevelId> {
    let run_index = manifest
        .record_runs
        .partition_point(|run| run.first_id.0 <= id.0)
        .checked_sub(1)?;
    let run = manifest.record_runs.get(run_index)?;
    (id.0 < run.stable_end()?).then_some(())?;
    manifest.levels.iter().find_map(|level| {
        let end = level.records.end()? as usize;
        (run_index >= level.records.start as usize && run_index < end).then_some(level.id)
    })
}

fn squared_distance(left: [f32; 3], right: [f32; 3]) -> f64 {
    (0..3)
        .map(|axis| {
            let delta = f64::from(left[axis]) - f64::from(right[axis]);
            delta * delta
        })
        .sum()
}

fn gaussian_catalog_sha256(records: impl IntoIterator<Item = Gaussian3d>) -> [u8; 32] {
    let mut digest = Sha256::new();
    for gaussian in records {
        digest.update(bytemuck::bytes_of(&gaussian));
    }
    digest.finalize().into()
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum GaussianLodgeResidentError {
    Manifest(String),
    UnauthenticatedBaseManifest,
    PageCodec(String),
    MembershipCodec(String),
    InvalidMembershipIndexRange,
    MembershipIndexHashMismatch,
    MembershipObjectMismatch,
    Planning(String),
    Settings(String),
    DuplicatePage(LodPageId),
    UnexpectedPage(LodPageId),
    MissingPage(LodPageId),
    UnauthenticatedPage(LodPageId),
    InvalidPage {
        page: LodPageId,
        detail: String,
    },
    IncompletePageClosure,
    PageRangeOverflow(LodPageId),
    DuplicateMembership(LodgeClusterId),
    UnexpectedMembership(LodgeClusterId),
    MissingMembership(LodgeClusterId),
    InvalidMembership(LodgeClusterId),
    IncompleteMembershipClosure,
    MissingCluster(LodgeClusterId),
    InsufficientClusters,
    CoincidentClusterCenters {
        first: LodgeClusterId,
        second: LodgeClusterId,
    },
    CatalogCountNotRepresentable(u64),
    CatalogCountMismatch {
        expected: u64,
        actual: u64,
    },
    PageCountNotRepresentable,
    InvalidExternalFrontier,
    InvalidExternalPresentation,
    ResidentGaussianBudgetExceeded,
    ResidentByteBudgetExceeded,
    ResidentPageBudgetExceeded,
    ViewLimitExceeded {
        limit: u32,
    },
    SphereLevelMismatch {
        cluster: LodgeClusterId,
        gaussian: LodgeGaussianId,
        level: LodgeLevelId,
        lower: f32,
        upper: f32,
    },
}

impl fmt::Display for GaussianLodgeResidentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(formatter, "invalid LODGE manifest: {error}"),
            Self::UnauthenticatedBaseManifest => write!(
                formatter,
                "the authenticated base manifest proof belongs to a different LODGE sidecar"
            ),
            Self::PageCodec(error) => write!(formatter, "LODGE page codec: {error}"),
            Self::MembershipCodec(error) => write!(formatter, "LODGE membership codec: {error}"),
            Self::InvalidMembershipIndexRange => {
                write!(formatter, "LODGE membership index range is invalid")
            }
            Self::MembershipIndexHashMismatch => {
                write!(formatter, "LODGE membership index SHA-256 mismatch")
            }
            Self::MembershipObjectMismatch => write!(
                formatter,
                "the authenticated membership object belongs to a different LODGE sidecar"
            ),
            Self::Planning(error) => write!(formatter, "LODGE planning: {error}"),
            Self::Settings(error) => write!(formatter, "invalid LODGE settings: {error}"),
            Self::DuplicatePage(page) => write!(formatter, "duplicate LODGE page {}", page.0),
            Self::UnexpectedPage(page) => write!(formatter, "unexpected LODGE page {}", page.0),
            Self::MissingPage(page) => write!(formatter, "missing LODGE page {}", page.0),
            Self::UnauthenticatedPage(page) => {
                write!(
                    formatter,
                    "LODGE page {} lacks matching authentication",
                    page.0
                )
            }
            Self::InvalidPage { page, detail } => {
                write!(formatter, "invalid LODGE page {}: {detail}", page.0)
            }
            Self::IncompletePageClosure => write!(formatter, "LODGE page closure is incomplete"),
            Self::PageRangeOverflow(page) => {
                write!(formatter, "LODGE run exceeds page {}", page.0)
            }
            Self::DuplicateMembership(cluster) => {
                write!(formatter, "duplicate LODGE membership {}", cluster.0)
            }
            Self::UnexpectedMembership(cluster) => {
                write!(formatter, "unexpected LODGE membership {}", cluster.0)
            }
            Self::MissingMembership(cluster) => {
                write!(formatter, "missing LODGE membership {}", cluster.0)
            }
            Self::InvalidMembership(cluster) => {
                write!(formatter, "invalid LODGE membership {}", cluster.0)
            }
            Self::IncompleteMembershipClosure => {
                write!(formatter, "LODGE membership closure is incomplete")
            }
            Self::MissingCluster(cluster) => {
                write!(formatter, "missing LODGE cluster {}", cluster.0)
            }
            Self::InsufficientClusters => write!(
                formatter,
                "LODGE resident pair selection requires at least two clusters"
            ),
            Self::CoincidentClusterCenters { first, second } => write!(
                formatter,
                "LODGE cluster centers {} and {} coincide",
                first.0, second.0
            ),
            Self::CatalogCountNotRepresentable(count) => write!(
                formatter,
                "LODGE stable catalog count {count} exceeds the 28-bit Entry source-index ABI"
            ),
            Self::CatalogCountMismatch { expected, actual } => write!(
                formatter,
                "LODGE stable catalog expected {expected} records, found {actual}"
            ),
            Self::PageCountNotRepresentable => {
                write!(formatter, "LODGE resident page count exceeds u32")
            }
            Self::InvalidExternalFrontier => {
                write!(formatter, "LODGE external frontier is invalid")
            }
            Self::InvalidExternalPresentation => {
                write!(formatter, "LODGE external presentation is invalid")
            }
            Self::ResidentGaussianBudgetExceeded => write!(
                formatter,
                "LODGE resident catalog exceeds max_resident_gaussians"
            ),
            Self::ResidentByteBudgetExceeded => write!(
                formatter,
                "LODGE resident catalog exceeds max_resident_bytes"
            ),
            Self::ResidentPageBudgetExceeded => write!(
                formatter,
                "LODGE resident catalog exceeds max_resident_pages"
            ),
            Self::ViewLimitExceeded { limit } => {
                write!(formatter, "LODGE visible camera count exceeds {limit}")
            }
            Self::SphereLevelMismatch {
                cluster,
                gaussian,
                level,
                lower,
                upper,
            } => write!(
                formatter,
                "LODGE cluster {} Gaussian {} level {} does not cover conservative distance interval {lower}..={upper}",
                cluster.0, gaussian.0, level.0
            ),
        }
    }
}

impl Error for GaussianLodgeResidentError {}

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    use bevy::ecs::message::MessageCursor;

    use super::*;
    use crate::{
        gaussian::formats::{
            lodge::{
                LodgeMembershipEntry, LodgePageAuthentication, LodgeRecordRun,
                tests::fixture as lodge_manifest_fixture,
            },
            planar_3d_chunked::{
                LodBounds, LodIndexRange, LodPageDescriptor, LodPageEncoding, LodPageKind,
                LodPageStorage,
            },
            planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        },
        gaussian::lod_debug::LodDebugPreset,
        io::{
            lod::{encode_manifest as encode_lod_manifest, encode_page},
            lodge::{decode_lodge_manifest, encode_lodge_manifest, encode_lodge_membership_ids},
        },
        stream::{
            bridge::GaussianLodBridgePhase, lodge::select_lodge_pair,
            render_commit::LOD_RENDER_ACTIVE,
        },
    };

    struct AuthenticatedResidentFixture {
        asset: GaussianLodgeAsset,
        manifest: Arc<GaussianLodgeManifest>,
        base: AuthenticatedLodgeBaseManifest,
        pages: Vec<AuthenticatedLodgePage>,
        membership_object: AuthenticatedLodgeMembershipObject,
        memberships: Vec<AuthenticatedLodgeMembership>,
        expected_catalog: Vec<Gaussian3d>,
    }

    fn gaussian(x: f32, visibility: f32) -> Gaussian3d {
        Gaussian3d {
            position_visibility: [x, 0.0, 0.0, visibility].into(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.1, 0.1, 0.1, 0.75].into(),
            ..Default::default()
        }
    }

    fn authenticated_resident_fixture() -> AuthenticatedResidentFixture {
        let source = PlanarGaussian3d::from(vec![gaussian(-1.0, 0.25), gaussian(1.0, 0.75)]);
        let mut base = build_planar_3d_lod(
            &source,
            GaussianLodBuildSettings {
                branching_factor: 2,
                leaf_capacity: 8,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(base.pages.len(), base.manifest.pages.len());

        let mut encoded_base_pages = BTreeMap::new();
        for page in &base.pages {
            let encoded = encode_page(page).unwrap();
            let descriptor = base
                .manifest
                .pages
                .iter_mut()
                .find(|descriptor| descriptor.id == page.id)
                .unwrap();
            descriptor.storage = Some(LodPageStorage {
                uri: format!("pages/base-{}.gspage", page.id.0),
                byte_range: None,
                encoded_len: encoded.len() as u64,
            });
            encoded_base_pages.insert(page.id, encoded);
        }
        base.manifest.validate().unwrap();
        let encoded_base_manifest = encode_lod_manifest(&base.manifest).unwrap();

        let page_by_id = base
            .pages
            .iter()
            .map(|page| (page.id, page))
            .collect::<BTreeMap<_, _>>();
        let mut leaves = base
            .manifest
            .nodes
            .iter()
            .filter(|node| node.is_leaf())
            .collect::<Vec<_>>();
        leaves.sort_unstable_by_key(|node| node.source.start);
        let mut first_id = 1_u64;
        let mut record_runs = Vec::with_capacity(leaves.len() + 1);
        let mut expected_catalog = Vec::new();
        for leaf in leaves {
            record_runs.push(LodgeRecordRun {
                first_id: LodgeGaussianId(first_id),
                count: leaf.representation.count,
                page: leaf.representation.page,
                page_offset: leaf.representation.offset,
            });
            let page = page_by_id[&leaf.representation.page];
            let start = leaf.representation.offset as usize;
            let end = start + leaf.representation.count as usize;
            expected_catalog.extend_from_slice(&page.gaussians[start..end]);
            first_id += u64::from(leaf.representation.count);
        }
        assert_eq!(first_id - 1, base.manifest.header.source_gaussian_count);

        let extra_page_id = LodPageId(
            base.manifest
                .pages
                .iter()
                .map(|descriptor| descriptor.id.0)
                .max()
                .unwrap()
                + 1,
        );
        let mut extra_records = expected_catalog.clone();
        extra_records.reverse();
        let extra_page = PlanarGaussian3dPage::new(extra_page_id, extra_records);
        let encoded_extra_page = encode_page(&extra_page).unwrap();
        let extra_descriptor = LodPageDescriptor {
            id: extra_page_id,
            kind: LodPageKind::Representatives,
            encoding: LodPageEncoding::F32Planar,
            gaussian_count: extra_page.gaussians.len() as u32,
            decoded_len: (extra_page.gaussians.len() * size_of::<Gaussian3d>()) as u64,
            content_hash: extra_page.content_hash(),
            bounds: LodBounds {
                min: [-2.0; 3],
                max: [2.0; 3],
            },
            storage: Some(LodPageStorage {
                uri: "pages/level-1.gspage".into(),
                byte_range: None,
                encoded_len: encoded_extra_page.len() as u64,
            }),
        };
        extra_page.validate(&extra_descriptor).unwrap();

        let first_extra_id = first_id;
        record_runs.push(LodgeRecordRun {
            first_id: LodgeGaussianId(first_extra_id),
            count: extra_page.gaussians.len() as u32,
            page: extra_page.id,
            page_offset: 0,
        });
        let first_ids = [LodgeGaussianId(1), LodgeGaussianId(first_extra_id)];
        let second_ids = [LodgeGaussianId(2), LodgeGaussianId(first_extra_id + 1)];
        let first_membership = encode_lodge_membership_ids(&first_ids).unwrap();
        let second_membership = encode_lodge_membership_ids(&second_ids).unwrap();
        let index_bytes = b"LODGE-index-v1";
        let first_start = index_bytes.len() as u64;
        let second_start = first_start + first_membership.len() as u64;
        let mut encoded_memberships = index_bytes.to_vec();
        encoded_memberships.extend_from_slice(&first_membership);
        encoded_memberships.extend_from_slice(&second_membership);

        let mut sidecar = lodge_manifest_fixture();
        sidecar.base_manifest = LodgeAuthenticatedObject {
            uri: "scene.gsplatlod".into(),
            encoded_len: encoded_base_manifest.len() as u64,
            sha256: sha256_bytes(&encoded_base_manifest),
        };
        sidecar.extra_pages = vec![extra_descriptor];
        sidecar.page_authentication = encoded_base_pages
            .iter()
            .map(|(&page, encoded)| LodgePageAuthentication {
                page,
                encoded_sha256: sha256_bytes(encoded),
            })
            .chain(std::iter::once(LodgePageAuthentication {
                page: extra_page.id,
                encoded_sha256: sha256_bytes(&encoded_extra_page),
            }))
            .collect();
        sidecar.header.base_page_count = base.manifest.header.page_count;
        sidecar.header.record_run_count = record_runs.len() as u32;
        sidecar.levels[0].records = LodIndexRange {
            start: 0,
            count: record_runs.len() as u32 - 1,
        };
        sidecar.levels[1].records = LodIndexRange {
            start: record_runs.len() as u32 - 1,
            count: 1,
        };
        sidecar.record_runs = record_runs;
        sidecar.header.stable_gaussian_count =
            base.manifest.header.source_gaussian_count + extra_page.gaussians.len() as u64;
        sidecar.header.total_membership_ids = (first_ids.len() + second_ids.len()) as u64;
        sidecar.membership_index.object = LodgeAuthenticatedObject {
            uri: "lodge/members.bgslmem".into(),
            encoded_len: encoded_memberships.len() as u64,
            sha256: sha256_bytes(&encoded_memberships),
        };
        sidecar.membership_index.index_byte_range = (0, index_bytes.len() as u64);
        sidecar.membership_index.index_sha256 = sha256_bytes(index_bytes);
        sidecar.membership_index.entries = vec![
            LodgeMembershipEntry {
                cluster: LodgeClusterId(1),
                byte_range: (first_start, first_membership.len() as u64),
                member_count: first_ids.len() as u64,
                first_id: first_ids[0],
                last_id: first_ids[1],
                encoded_sha256: sha256_bytes(&first_membership),
            },
            LodgeMembershipEntry {
                cluster: LodgeClusterId(2),
                byte_range: (second_start, second_membership.len() as u64),
                member_count: second_ids.len() as u64,
                first_id: second_ids[0],
                last_id: second_ids[1],
                encoded_sha256: sha256_bytes(&second_membership),
            },
        ];
        sidecar.validate_against_base(&base.manifest).unwrap();

        let encoded_sidecar = encode_lodge_manifest(&sidecar).unwrap();
        let decoded_sidecar =
            decode_lodge_manifest(&encoded_sidecar, LodgeCodecLimits::default()).unwrap();
        assert_eq!(decoded_sidecar, sidecar);
        let asset = GaussianLodgeAsset::new(decoded_sidecar).unwrap();
        let manifest = asset.shared_manifest();
        let authenticated_base = AuthenticatedLodgeBaseManifest::decode(
            &manifest,
            &encoded_base_manifest,
            LodPageCodecLimits::default(),
            encoded_base_manifest.len() as u64,
        )
        .unwrap();
        let authenticated_pages = base
            .manifest
            .pages
            .iter()
            .map(|descriptor| (descriptor, encoded_base_pages[&descriptor.id].as_slice()))
            .chain(std::iter::once((
                &manifest.extra_pages[0],
                encoded_extra_page.as_slice(),
            )))
            .map(|(descriptor, encoded)| {
                AuthenticatedLodgePage::decode(
                    &manifest,
                    descriptor,
                    encoded,
                    LodPageCodecLimits::default(),
                    encoded.len() as u64,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let membership_object = AuthenticatedLodgeMembershipObject::decode(
            &manifest,
            encoded_memberships,
            manifest.membership_index.object.encoded_len,
        )
        .unwrap();
        let memberships = manifest
            .clusters
            .iter()
            .map(|cluster| {
                membership_object
                    .decode_membership(&manifest, cluster.id, LodgeCodecLimits::default())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        expected_catalog.extend_from_slice(&extra_page.gaussians);

        AuthenticatedResidentFixture {
            asset,
            manifest,
            base: authenticated_base,
            pages: authenticated_pages,
            membership_object,
            memberships,
            expected_catalog,
        }
    }

    fn cluster(id: u32, x: f32) -> LodgeCameraCluster {
        LodgeCameraCluster {
            id: LodgeClusterId(id),
            center: [x, 0.0, 0.0],
            radius: 0.0,
            neighbors: LodIndexRange { start: 0, count: 0 },
            membership_entry: id - 1,
        }
    }

    #[test]
    fn authenticated_dependency_closure_materializes_stable_id_catalog() {
        let fixture = authenticated_resident_fixture();
        let mut pages = fixture.pages.clone();
        let mut memberships = fixture.memberships.clone();
        pages.reverse();
        memberships.reverse();
        let mut assets = Assets::<PlanarGaussian3d>::default();
        let resident = GaussianLodgeResidentCatalog::from_authenticated_pages(
            Arc::clone(&fixture.manifest),
            &fixture.base,
            &GaussianLodgeSettings::default(),
            pages,
            memberships,
            &mut assets,
        )
        .unwrap();

        let catalog = assets.get(resident.catalog_handle()).unwrap();
        assert_eq!(catalog.iter().collect::<Vec<_>>(), fixture.expected_catalog);
        assert_eq!(resident.manifest(), fixture.manifest.as_ref());
        assert_eq!(
            resident
                .memberships()
                .map(|membership| (membership.cluster(), membership.ids().to_vec()))
                .collect::<Vec<_>>(),
            vec![
                (
                    LodgeClusterId(1),
                    vec![LodgeGaussianId(1), LodgeGaussianId(3)]
                ),
                (
                    LodgeClusterId(2),
                    vec![LodgeGaussianId(2), LodgeGaussianId(4)]
                ),
            ]
        );
    }

    #[test]
    fn resident_budget_is_rejected_before_catalog_allocation() {
        let fixture = authenticated_resident_fixture();
        let mut settings = GaussianLodgeSettings::default();
        settings.budgets.max_active_gaussians = 3;
        settings.budgets.max_resident_gaussians = 3;
        let mut assets = Assets::<PlanarGaussian3d>::default();
        let before = assets.len();
        let result = GaussianLodgeResidentCatalog::from_authenticated_pages(
            Arc::clone(&fixture.manifest),
            &fixture.base,
            &settings,
            fixture.pages.clone(),
            fixture.memberships.clone(),
            &mut assets,
        );
        assert!(matches!(
            result,
            Err(GaussianLodgeResidentError::ResidentGaussianBudgetExceeded)
        ));
        assert_eq!(assets.len(), before);
    }

    #[test]
    fn authenticated_proofs_cannot_be_reused_with_another_sidecar() {
        let fixture = authenticated_resident_fixture();
        let mut mismatched = fixture.manifest.as_ref().clone();
        mismatched.base_manifest.uri = "alternate.gsplatlod".into();
        mismatched.validate().unwrap();
        let mut assets = Assets::<PlanarGaussian3d>::default();
        let result = GaussianLodgeResidentCatalog::from_authenticated_pages(
            Arc::new(mismatched),
            &fixture.base,
            &GaussianLodgeSettings::default(),
            fixture.pages.clone(),
            fixture.memberships.clone(),
            &mut assets,
        );
        assert!(matches!(
            result,
            Err(GaussianLodgeResidentError::UnauthenticatedBaseManifest)
        ));
        assert!(assets.is_empty());

        let mut mismatched = fixture.manifest.as_ref().clone();
        mismatched.membership_index.object.uri = "lodge/alternate.bgslmem".into();
        mismatched.validate().unwrap();
        assert!(matches!(
            fixture.membership_object.decode_membership(
                &mismatched,
                LodgeClusterId(1),
                LodgeCodecLimits::default(),
            ),
            Err(GaussianLodgeResidentError::MembershipObjectMismatch)
        ));

        let mut assets = Assets::<PlanarGaussian3d>::default();
        assert!(matches!(
            GaussianLodgeResidentCatalog::from_authenticated_pages(
                Arc::new(mismatched),
                &fixture.base,
                &GaussianLodgeSettings::default(),
                fixture.pages.clone(),
                fixture.memberships.clone(),
                &mut assets,
            ),
            Err(GaussianLodgeResidentError::MembershipObjectMismatch)
        ));
        assert!(assets.is_empty());

        let mut mismatched = fixture.manifest.as_ref().clone();
        let mismatched_page = mismatched.extra_pages[0].id;
        mismatched.extra_pages[0].storage.as_mut().unwrap().uri =
            "pages/alternate-level-1.gspage".into();
        mismatched.validate().unwrap();
        assert!(matches!(
            GaussianLodgeResidentCatalog::from_authenticated_pages(
                Arc::new(mismatched),
                &fixture.base,
                &GaussianLodgeSettings::default(),
                fixture.pages.clone(),
                fixture.memberships.clone(),
                &mut assets,
            ),
            Err(GaussianLodgeResidentError::UnauthenticatedPage(page))
                if page == mismatched_page
        ));
        assert!(assets.is_empty());
    }

    #[test]
    fn static_invalid_debug_policy_is_revision_stable_and_recovers_from_off() {
        let fixture = authenticated_resident_fixture();
        let mut catalogs = Assets::<PlanarGaussian3d>::default();
        let resident = GaussianLodgeResidentCatalog::from_authenticated_pages(
            Arc::clone(&fixture.manifest),
            &fixture.base,
            &GaussianLodgeSettings::default(),
            fixture.pages,
            fixture.memberships,
            &mut catalogs,
        )
        .unwrap();
        let mut lodge_assets = Assets::<GaussianLodgeAsset>::default();
        let lodge = lodge_assets.add(fixture.asset);
        let mut cloud_settings = CloudSettings::default();
        cloud_settings.lod_debug.apply_preset(LodDebugPreset::Page);

        let mut app = App::new();
        app.insert_resource(lodge_assets)
            .insert_resource(catalogs)
            .init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>()
            .add_plugins(GaussianLodgeResidentPlugin);
        let entity = app
            .world_mut()
            .spawn((GaussianLodgeHandle(lodge), resident, cloud_settings))
            .id();

        app.update();
        let first = app
            .world()
            .entity(entity)
            .get::<GaussianLodgeStatus>()
            .unwrap()
            .clone();
        assert_eq!(first.lifecycle, GaussianLodgeLifecycle::Failed);
        assert!(!first.target_satisfied);
        assert_eq!(
            first.failure.as_deref(),
            Some("LODGE external active sets require the LoD debug preset Off")
        );
        assert!(
            app.world()
                .entity(entity)
                .get::<LodRenderCandidates>()
                .unwrap()
                .by_camera
                .is_empty()
        );
        let mut transitions = MessageCursor::<LodOrchestrationTransition>::default();
        let emitted = transitions
            .read(
                app.world()
                    .resource::<Messages<LodOrchestrationTransition>>(),
            )
            .collect::<Vec<_>>();
        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].entity, entity);
        assert_eq!(
            emitted[0].source,
            LodOrchestrationSource::ExternalActiveSets
        );
        assert_eq!(emitted[0].kind, LodOrchestrationTransitionKind::Failed);
        assert_eq!(
            emitted[0]
                .failure
                .as_ref()
                .map(LodOrchestrationFailure::code),
            Some(LodOrchestrationFailureCode::UnsupportedConfiguration)
        );

        app.update();
        let unchanged = app
            .world()
            .entity(entity)
            .get::<GaussianLodgeStatus>()
            .unwrap();
        assert_eq!(unchanged.revision, first.revision);
        assert_eq!(unchanged.failure, first.failure);
        assert_eq!(
            transitions
                .read(
                    app.world()
                        .resource::<Messages<LodOrchestrationTransition>>(),
                )
                .count(),
            0
        );

        app.world_mut()
            .entity_mut(entity)
            .get_mut::<CloudSettings>()
            .unwrap()
            .lod_debug
            .apply_preset(LodDebugPreset::Off);
        app.update();
        app.update();
        let recovered = app
            .world()
            .entity(entity)
            .get::<GaussianLodgeStatus>()
            .unwrap();
        assert_eq!(
            recovered.lifecycle,
            GaussianLodgeLifecycle::WaitingForRender
        );
        assert!(!recovered.target_satisfied);
        assert_eq!(recovered.failure, None);
        assert!(recovered.revision > first.revision);
    }

    #[test]
    fn hidden_cloud_resets_shared_current_phase_and_cancels_pending() {
        let fixture = authenticated_resident_fixture();
        let mut catalogs = Assets::<PlanarGaussian3d>::default();
        let resident = GaussianLodgeResidentCatalog::from_authenticated_pages(
            Arc::clone(&fixture.manifest),
            &fixture.base,
            &GaussianLodgeSettings::default(),
            fixture.pages,
            fixture.memberships,
            &mut catalogs,
        )
        .unwrap();
        let selection = select_lodge_pair([0.0, 0.0, 0.0], &fixture.manifest.clusters).unwrap();
        let locations = LodgeRecordLocationResolver::from_validated(&fixture.manifest.record_runs);
        let plan = Arc::new(
            build_lodge_pair_candidate(
                selection,
                resident.memberships.get(&selection.identity.first).unwrap(),
                resident
                    .memberships
                    .get(&selection.identity.second)
                    .unwrap(),
                &locations,
                LodgePairLimits {
                    max_union_gaussians: 16,
                    max_classified_runs: 16,
                    max_required_pages: 16,
                },
            )
            .unwrap(),
        );
        let camera = Entity::from_bits(1);
        let plans = BTreeMap::from([(camera, Arc::clone(&plan))]);
        let current = build_external_candidate_set(&plans, &resident, false, None).unwrap();
        for candidate in current.by_camera.values() {
            candidate.phase.store(LOD_RENDER_ACTIVE, Ordering::Release);
        }
        let pending =
            build_external_candidate_set(&plans, &resident, false, Some(&current)).unwrap();
        assert!(
            current
                .by_camera
                .values()
                .all(LodRenderCandidate::render_is_active)
        );
        assert!(
            pending
                .by_camera
                .values()
                .all(LodRenderCandidate::render_is_active)
        );

        let mut publication = LodgePairPublicationState::default();
        publication.stage(Arc::clone(&plan));
        let mut state = GaussianLodgeResidentState {
            publications: BTreeMap::from([(camera, publication)]),
            current: Some(current),
            current_plans: plans.clone(),
            pending: Some(pending),
            pending_plans: plans.clone(),
            ..Default::default()
        };
        suspend_lodge_render_publication(&mut state);

        assert!(state.pending.is_none());
        assert!(state.pending_plans.is_empty());
        assert!(state.publications[&camera].pending().is_none());
        assert!(same_plan_unions(&plans, &state.current_plans));
        let current = state.current.as_ref().unwrap();
        assert!(
            current
                .by_camera
                .values()
                .all(|candidate| !candidate.render_is_active() && !candidate.render_is_prepared())
        );
    }

    #[test]
    fn failed_render_request_is_latched_until_request_device_or_render_environment_changes() {
        let fixture = authenticated_resident_fixture();
        let mut catalogs = Assets::<PlanarGaussian3d>::default();
        let resident = GaussianLodgeResidentCatalog::from_authenticated_pages(
            Arc::clone(&fixture.manifest),
            &fixture.base,
            &GaussianLodgeSettings::default(),
            fixture.pages,
            fixture.memberships,
            &mut catalogs,
        )
        .unwrap();
        let selection = select_lodge_pair([0.0, 0.0, 0.0], &fixture.manifest.clusters).unwrap();
        let locations = LodgeRecordLocationResolver::from_validated(&fixture.manifest.record_runs);
        let plan = Arc::new(
            build_lodge_pair_candidate(
                selection,
                resident.memberships.get(&selection.identity.first).unwrap(),
                resident
                    .memberships
                    .get(&selection.identity.second)
                    .unwrap(),
                &locations,
                LodgePairLimits {
                    max_union_gaussians: 16,
                    max_classified_runs: 16,
                    max_required_pages: 16,
                },
            )
            .unwrap(),
        );
        let camera = Entity::from_bits(1);
        let plans = BTreeMap::from([(camera, Arc::clone(&plan))]);
        let settings = CloudSettings::default();
        let failed = LodgeFailedRenderRequest {
            plans: plans.clone(),
            frozen: false,
            cloud_settings: settings.clone(),
            device_generation: 4,
            render_environment_epoch: 7,
            detail: "pipeline failed".into(),
        };

        assert!(lodge_failed_render_request_matches(
            &failed, &plans, false, &settings, 4, 7
        ));
        assert!(!lodge_failed_render_request_matches(
            &failed, &plans, false, &settings, 5, 7
        ));
        assert!(!lodge_failed_render_request_matches(
            &failed, &plans, false, &settings, 4, 8
        ));
        assert!(!lodge_failed_render_request_matches(
            &failed, &plans, true, &settings, 4, 7
        ));
        let mut changed_policy = settings.clone();
        changed_policy.aabb = !changed_policy.aabb;
        assert!(!lodge_failed_render_request_matches(
            &failed,
            &plans,
            false,
            &changed_policy,
            4,
            7,
        ));

        let retargeted = plan
            .retarget(select_lodge_pair([1.0, 0.0, 0.0], &fixture.manifest.clusters).unwrap())
            .unwrap();
        let same_union = BTreeMap::from([(camera, Arc::new(retargeted))]);
        assert!(lodge_failed_render_request_matches(
            &failed,
            &same_union,
            false,
            &settings,
            4,
            7,
        ));
    }

    #[test]
    fn plugin_registers_public_types_required_components_and_cleans_stale_state() {
        let fixture = authenticated_resident_fixture();
        let mut catalogs = Assets::<PlanarGaussian3d>::default();
        let resident = GaussianLodgeResidentCatalog::from_authenticated_pages(
            Arc::clone(&fixture.manifest),
            &fixture.base,
            &GaussianLodgeSettings::default(),
            fixture.pages,
            fixture.memberships,
            &mut catalogs,
        )
        .unwrap();
        let catalog_id = resident.catalog_handle().id();
        let catalog_handle = resident.catalog_handle().clone();

        let mut lodge_assets = Assets::<GaussianLodgeAsset>::default();
        let valid_lodge = lodge_assets.add(fixture.asset);
        let mut app = App::new();
        app.insert_resource(lodge_assets)
            .insert_resource(catalogs)
            .init_resource::<Messages<AssetEvent<PlanarGaussian3d>>>()
            .add_plugins(GaussianLodgeResidentPlugin);
        let entity = app
            .world_mut()
            .spawn((
                GaussianLodgeHandle::default(),
                resident.clone(),
                GaussianLodgeStatus {
                    representation: GaussianLodRepresentationKind::FinestHierarchy,
                    lifecycle: GaussianLodgeLifecycle::Active,
                    target_satisfied: true,
                    retained_stale_pair: true,
                    failure: Some("stale".into()),
                    revision: 7,
                    ..Default::default()
                },
            ))
            .id();
        let unmaterialized = app.world_mut().spawn(GaussianLodgeHandle::default()).id();
        let unmaterialized_conflict = app
            .world_mut()
            .spawn((GaussianLodgeHandle::default(), GaussianLodHandle::default()))
            .id();

        let entity_ref = app.world().entity(entity);
        assert!(entity_ref.contains::<GaussianLodgeSettings>());
        assert!(entity_ref.contains::<CloudSettings>());
        assert!(entity_ref.contains::<Transform>());
        assert!(entity_ref.contains::<GlobalTransform>());
        assert!(entity_ref.contains::<Visibility>());
        let registry = app.world().resource::<AppTypeRegistry>().read();
        assert!(
            registry
                .get(TypeId::of::<GaussianLodgeSettings>())
                .is_some()
        );
        assert!(
            registry
                .get(TypeId::of::<GaussianLodRepresentationKind>())
                .is_some()
        );
        drop(registry);

        app.update();
        let status = app
            .world()
            .entity(entity)
            .get::<GaussianLodgeStatus>()
            .unwrap();
        assert_eq!(status.lifecycle, GaussianLodgeLifecycle::LoadingManifest);
        assert_eq!(
            status.representation,
            GaussianLodRepresentationKind::LodgeActiveSets
        );
        assert!(!status.target_satisfied);
        assert!(!status.retained_stale_pair);
        assert_eq!(status.failure, None);
        assert_eq!(status.revision, 8);
        assert!(app.world().entity(entity).contains::<LodRenderCandidates>());
        let unmaterialized_status = app
            .world()
            .entity(unmaterialized)
            .get::<GaussianLodgeStatus>()
            .unwrap();
        assert_eq!(
            unmaterialized_status.lifecycle,
            GaussianLodgeLifecycle::LoadingPages
        );
        assert!(!unmaterialized_status.target_satisfied);
        assert!(
            app.world()
                .entity(unmaterialized)
                .contains::<LodRenderCandidates>()
        );
        let conflict_status = app
            .world()
            .entity(unmaterialized_conflict)
            .get::<GaussianLodgeStatus>()
            .unwrap();
        assert_eq!(conflict_status.lifecycle, GaussianLodgeLifecycle::Failed);
        assert_eq!(
            conflict_status.failure_code,
            Some(LodOrchestrationFailureCode::InvalidConfiguration)
        );

        let mut state = GaussianLodgeResidentState::default();
        state.catalog_id = Some(catalog_id);
        app.world_mut().entity_mut(entity).insert((
            state,
            PlanarGaussian3dHandle(catalog_handle),
            GaussianLodSettings::default(),
        ));
        app.world_mut()
            .entity_mut(entity)
            .remove::<GaussianLodgeHandle>();
        app.update();
        let entity_ref = app.world().entity(entity);
        assert!(!entity_ref.contains::<GaussianLodgeResidentState>());
        assert!(!entity_ref.contains::<PlanarGaussian3dHandle>());
        assert!(!entity_ref.contains::<LodRenderCandidates>());
        assert!(!entity_ref.contains::<GaussianLodgeStatus>());
        assert!(!entity_ref.contains::<GaussianLodSettings>());

        let private_adapter = lodge_render_adapter(&GaussianLodgeSettings::default());
        let switching = app
            .world_mut()
            .spawn((
                GaussianLodgeHandle(valid_lodge.clone()),
                resident.clone(),
                LodRenderCandidates::package_required(),
                GaussianLodBridgeStatus {
                    phase: GaussianLodBridgePhase::Active,
                    active_views: 1,
                    resident_pages: 1,
                    active_gaussians: 1,
                    failure: None,
                },
            ))
            .id();
        // The resident system itself installs settings and the ownership
        // marker together after cleanup. Switching before another Lodge frame
        // must still recognize that exact tick pair as private.
        app.update();
        assert!(
            !app.world()
                .entity(switching)
                .contains::<LodRenderCandidates>()
        );
        assert!(
            !app.world()
                .entity(switching)
                .contains::<GaussianLodBridgeStatus>()
        );
        app.world_mut()
            .entity_mut(switching)
            .remove::<GaussianLodgeHandle>()
            .insert(GaussianLodHandle::default());
        app.update();
        let switching = app.world().entity(switching);
        assert!(!switching.contains::<GaussianLodgeResidentState>());
        assert!(!switching.contains::<GaussianLodgeRenderAdapter>());
        assert_eq!(
            switching.get::<GaussianLodSettings>(),
            Some(&GaussianLodSettings::default())
        );

        let mut explicit_state = GaussianLodgeResidentState::default();
        explicit_state.catalog_id = Some(catalog_id);
        let explicit_same_value = app
            .world_mut()
            .spawn((GaussianLodgeHandle(valid_lodge), resident, explicit_state))
            .id();
        app.update();
        app.world_mut()
            .entity_mut(explicit_same_value)
            .remove::<GaussianLodgeHandle>()
            .insert((GaussianLodHandle::default(), private_adapter.clone()));
        app.update();
        let explicit_same_value = app.world().entity(explicit_same_value);
        assert!(!explicit_same_value.contains::<GaussianLodgeRenderAdapter>());
        assert_eq!(
            explicit_same_value.get::<GaussianLodSettings>(),
            Some(&private_adapter)
        );
    }

    #[test]
    fn retained_secondary_hysteresis_changes_only_pair_identity() {
        let clusters = [cluster(1, 0.0), cluster(2, 2.0), cluster(3, 2.1)];
        let retained = LodgePairIdentity {
            first: LodgeClusterId(1),
            second: LodgeClusterId(3),
        };
        let held =
            select_lodge_pair_with_hysteresis([0.1, 0.0, 0.0], &clusters, Some(retained), 0.1)
                .unwrap();
        assert_eq!(held.identity, retained);
        let exact =
            select_lodge_pair_with_hysteresis([0.1, 0.0, 0.0], &clusters, Some(retained), 0.0)
                .unwrap();
        assert_eq!(
            exact,
            select_lodge_pair([0.1, 0.0, 0.0], &clusters).unwrap()
        );
        let replaced =
            select_lodge_pair_with_hysteresis([0.1, 0.0, 0.0], &clusters, Some(retained), 0.01)
                .unwrap();
        assert_eq!(
            replaced.identity,
            LodgePairIdentity {
                first: LodgeClusterId(1),
                second: LodgeClusterId(2),
            }
        );
    }

    #[test]
    fn internal_adapter_uses_only_private_coarsest_compatibility() {
        let settings = GaussianLodgeSettings {
            selection_mode: LodSelectionMode::Frozen,
            frustum_culling: false,
            frustum_margin: 2.0,
            ..Default::default()
        };
        let adapter = lodge_render_adapter(&settings);
        assert_eq!(adapter.quality, 0.0);
        assert_eq!(adapter.selection_mode, LodSelectionMode::Frozen);
        assert!(!adapter.frustum_culling);
        assert_eq!(adapter.frustum_margin, 2.0);
        assert_eq!(adapter.budgets, settings.budgets);
    }
}
