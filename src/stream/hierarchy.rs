//! Deterministic CPU reference traversal for a Gaussian LoD hierarchy.
//!
//! The renderer is expected to execute the same policy on the GPU. Keeping this
//! allocation-light oracle independent of a concrete manifest gives tests and
//! offline tools a precise source of truth for endpoint, budget, and residency
//! behavior.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
    sync::Arc,
};

#[cfg(any(feature = "lod", test))]
use std::collections::BTreeSet;

use bevy::math::{Mat4, Vec3, Vec4};

use crate::{
    gaussian::{
        formats::{
            planar_3d_chunked::{LodIndexRange, LodNodeId, LodPageId, LodPageRange},
            planar_3d_lod::GaussianLodManifest,
        },
        lod_settings::{
            GaussianLodSettings, LodDegradation, LodEffectiveStatus, LodQualityEndpoint,
            LodQualityTarget, LodSettingsError,
        },
    },
    stream::cache::LodPageCache,
};

/// Renderer-independent metrics stored for every hierarchy node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodNodeMetrics {
    pub center: Vec3,
    pub radius: f32,
    /// Conservative manifest-local spatial deviation between this node's
    /// representatives and original descendants.
    pub geometric_error: f32,
    /// Dimensionless SH/color representative error.
    pub appearance_error: f32,
    /// Normalized opacity representative error.
    pub opacity_error: f32,
    /// Inclusive structural-detail interval authored by the hierarchy builder.
    /// Interior selection uses its midpoint as the nearest-level threshold.
    pub quality_min: f32,
    pub quality_max: f32,
    /// Builder-authored confidence that this representation is safe for the
    /// near-original fidelity regime. Exact source leaves use one; legacy or
    /// otherwise uncertified internal representatives use zero.
    pub high_fidelity_certificate: f32,
    /// Number of representatives drawn when this node is in the frontier.
    pub representative_count: u32,
}

impl LodNodeMetrics {
    /// Finite-safe midpoint of the validated structural-detail interval.
    pub fn quality_threshold(self) -> f32 {
        self.quality_min + (self.quality_max - self.quality_min) * 0.5
    }

    pub fn validate(self) -> bool {
        self.center.is_finite()
            && self.radius.is_finite()
            && self.radius >= 0.0
            && self.geometric_error.is_finite()
            && self.geometric_error >= 0.0
            && self.appearance_error.is_finite()
            && self.appearance_error >= 0.0
            && self.opacity_error.is_finite()
            && self.opacity_error >= 0.0
            && self.quality_min.is_finite()
            && self.quality_max.is_finite()
            && self.quality_min >= 0.0
            && self.quality_max <= 1.0
            && self.quality_min <= self.quality_max
            && self.high_fidelity_certificate.is_finite()
            && (0.0..=1.0).contains(&self.high_fidelity_certificate)
            && self.representative_count > 0
    }
}

/// Minimal interface needed by CPU and test traversal.
///
/// Returning slices makes traversal allocation-free for common flat-manifest
/// layouts while keeping node identifier types manifest-specific.
pub trait LodHierarchy {
    type NodeId: Copy + Debug + Eq + Hash + Ord;

    fn roots(&self) -> &[Self::NodeId];
    fn parent(&self, node: Self::NodeId) -> Option<Self::NodeId>;
    fn children(&self, node: Self::NodeId) -> &[Self::NodeId];
    fn metrics(&self, node: Self::NodeId) -> Option<LodNodeMetrics>;
}

/// Residency is separate from hierarchy topology because it changes every frame.
pub trait LodResidency<NodeId> {
    fn is_resident(&self, node: NodeId) -> bool;
}

/// Validated adapter from the portable manifest into the allocation-free traversal interface.
/// Child identifiers are materialized once when a manifest is opened, not once per camera/frame.
#[derive(Clone, Debug)]
pub struct ManifestLodHierarchy<'a> {
    manifest: &'a GaussianLodManifest,
    node_indices: BTreeMap<LodNodeId, usize>,
    node_ids: Vec<LodNodeId>,
}

/// Owned, shareable form used by long-lived streaming/runtime state.
/// Topology is compiled once when the manifest is opened, so camera traversal
/// never rebuilds child vectors.
#[derive(Clone, Debug)]
pub struct CompiledManifestLodHierarchy {
    manifest: Arc<GaussianLodManifest>,
    node_indices: CompiledManifestNodeIndices,
    #[cfg(any(feature = "lod", test))]
    page_indices: CompiledManifestPageIndices,
    /// Manifest-order node identifiers. A node's already-validated
    /// `children` range indexes this single flat allocation directly.
    node_ids: Vec<LodNodeId>,
}

#[derive(Clone, Debug)]
enum CompiledManifestNodeIndices {
    /// Promoted manifests assign `LodNodeId(index + 1)` in manifest order.
    /// Keep that common path arithmetic-only during per-camera traversal.
    DenseOneBased,
    /// Portable manifests only promise unique, valid identifiers, so retain a
    /// lookup table for external producers that use arbitrary node IDs.
    Sparse(HashMap<LodNodeId, usize>),
}

#[cfg(any(feature = "lod", test))]
#[derive(Clone, Debug)]
enum CompiledManifestPageIndices {
    /// Canonical packages assign `LodPageId(index + 1)` in manifest order.
    /// Avoid allocating and populating a second descriptor map for that common
    /// web/streaming path.
    DenseOneBased,
    /// The portable format permits arbitrary nonzero page identifiers.
    Sparse(HashMap<LodPageId, usize>),
}

impl CompiledManifestLodHierarchy {
    pub fn new(manifest: GaussianLodManifest) -> Result<Self, ManifestHierarchyError> {
        manifest
            .validate()
            .map_err(|error| ManifestHierarchyError::InvalidManifest(error.to_string()))?;
        Ok(Self::from_validated_shared_manifest(Arc::new(manifest)))
    }

    /// Compiles traversal indexes for an immutable manifest whose complete
    /// semantic contract has already been validated by an in-crate owner.
    ///
    /// Keeping this constructor crate-private preserves validation on the
    /// public owned API while allowing package startup to share its asset Arc.
    pub(crate) fn from_validated_shared_manifest(manifest: Arc<GaussianLodManifest>) -> Self {
        let dense_one_based = manifest.nodes.iter().enumerate().all(|(index, node)| {
            u64::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1))
                .is_some_and(|expected| node.id == LodNodeId(expected))
        });
        let node_indices = if dense_one_based {
            CompiledManifestNodeIndices::DenseOneBased
        } else {
            CompiledManifestNodeIndices::Sparse(
                manifest
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(index, node)| (node.id, index))
                    .collect(),
            )
        };
        #[cfg(any(feature = "lod", test))]
        let dense_pages_one_based = manifest.pages.iter().enumerate().all(|(index, page)| {
            u64::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1))
                .is_some_and(|expected| page.id == LodPageId(expected))
        });
        #[cfg(any(feature = "lod", test))]
        let page_indices = if dense_pages_one_based {
            CompiledManifestPageIndices::DenseOneBased
        } else {
            CompiledManifestPageIndices::Sparse(
                manifest
                    .pages
                    .iter()
                    .enumerate()
                    .map(|(index, page)| (page.id, index))
                    .collect(),
            )
        };
        let node_ids = manifest.nodes.iter().map(|node| node.id).collect();
        Self {
            manifest,
            node_indices,
            #[cfg(any(feature = "lod", test))]
            page_indices,
            node_ids,
        }
    }

    pub fn manifest(&self) -> &GaussianLodManifest {
        &self.manifest
    }

    #[inline]
    pub(crate) fn node_index(&self, node: LodNodeId) -> Option<usize> {
        match &self.node_indices {
            CompiledManifestNodeIndices::DenseOneBased => {
                let index = usize::try_from(node.0.checked_sub(1)?).ok()?;
                (index < self.manifest.nodes.len()).then_some(index)
            }
            CompiledManifestNodeIndices::Sparse(node_indices) => node_indices.get(&node).copied(),
        }
    }

    pub fn node(
        &self,
        node: LodNodeId,
    ) -> Option<&crate::gaussian::formats::planar_3d_lod::GaussianLodNode> {
        self.node_index(node)
            .and_then(|index| self.manifest.nodes.get(index))
    }

    pub fn representation(&self, node: LodNodeId) -> Option<LodPageRange> {
        self.node(node).map(|node| node.representation)
    }

    pub fn page(&self, node: LodNodeId) -> Option<LodPageId> {
        self.representation(node).map(|range| range.page)
    }

    #[inline]
    #[cfg(any(feature = "lod", test))]
    fn page_index(&self, page: LodPageId) -> Option<usize> {
        match &self.page_indices {
            CompiledManifestPageIndices::DenseOneBased => {
                let index = usize::try_from(page.0.checked_sub(1)?).ok()?;
                (index < self.manifest.pages.len()).then_some(index)
            }
            CompiledManifestPageIndices::Sparse(page_indices) => page_indices.get(&page).copied(),
        }
    }

    /// Looks up one validated descriptor without cloning the manifest's page
    /// table into a second map at package-open time.
    #[cfg(any(feature = "lod", test))]
    pub(crate) fn page_descriptor(
        &self,
        page: LodPageId,
    ) -> Option<&crate::gaussian::formats::planar_3d_chunked::LodPageDescriptor> {
        self.page_index(page)
            .and_then(|index| self.manifest.pages.get(index))
    }
}

impl LodHierarchy for CompiledManifestLodHierarchy {
    type NodeId = LodNodeId;

    fn roots(&self) -> &[Self::NodeId] {
        &self.manifest.roots
    }

    fn parent(&self, node: Self::NodeId) -> Option<Self::NodeId> {
        self.node(node).and_then(|node| node.parent)
    }

    fn children(&self, node: Self::NodeId) -> &[Self::NodeId] {
        self.node(node)
            .and_then(|node| manifest_child_ids(&self.node_ids, node.children))
            .unwrap_or(&[])
    }

    fn metrics(&self, node: Self::NodeId) -> Option<LodNodeMetrics> {
        let node = self.node(node)?;
        Some(LodNodeMetrics {
            center: Vec3::from_array(node.bounds.center()),
            radius: node.bounds.radius(),
            geometric_error: node.error.geometric,
            appearance_error: node.error.appearance,
            opacity_error: node.error.opacity,
            quality_min: node.quality.min,
            quality_max: node.quality.max,
            high_fidelity_certificate: node.high_fidelity_certificate,
            representative_count: node.representation.count,
        })
    }
}

impl<'a> ManifestLodHierarchy<'a> {
    pub fn new(manifest: &'a GaussianLodManifest) -> Result<Self, ManifestHierarchyError> {
        manifest
            .validate()
            .map_err(|error| ManifestHierarchyError::InvalidManifest(error.to_string()))?;
        let node_indices = manifest
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id, index))
            .collect::<BTreeMap<_, _>>();
        let node_ids = compile_manifest_node_ids(manifest)?;
        Ok(Self {
            manifest,
            node_indices,
            node_ids,
        })
    }

    pub fn manifest(&self) -> &'a GaussianLodManifest {
        self.manifest
    }

    pub fn node(
        &self,
        node: LodNodeId,
    ) -> Option<&'a crate::gaussian::formats::planar_3d_lod::GaussianLodNode> {
        self.node_indices
            .get(&node)
            .and_then(|index| self.manifest.nodes.get(*index))
    }

    pub fn representation(&self, node: LodNodeId) -> Option<LodPageRange> {
        self.node(node).map(|node| node.representation)
    }

    pub fn page(&self, node: LodNodeId) -> Option<LodPageId> {
        self.representation(node).map(|range| range.page)
    }
}

impl LodHierarchy for ManifestLodHierarchy<'_> {
    type NodeId = LodNodeId;

    fn roots(&self) -> &[Self::NodeId] {
        &self.manifest.roots
    }

    fn parent(&self, node: Self::NodeId) -> Option<Self::NodeId> {
        self.node(node).and_then(|node| node.parent)
    }

    fn children(&self, node: Self::NodeId) -> &[Self::NodeId] {
        self.node(node)
            .and_then(|node| manifest_child_ids(&self.node_ids, node.children))
            .unwrap_or(&[])
    }

    fn metrics(&self, node: Self::NodeId) -> Option<LodNodeMetrics> {
        let node = self.node(node)?;
        Some(LodNodeMetrics {
            center: Vec3::from_array(node.bounds.center()),
            radius: node.bounds.radius(),
            geometric_error: node.error.geometric,
            appearance_error: node.error.appearance,
            opacity_error: node.error.opacity,
            quality_min: node.quality.min,
            quality_max: node.quality.max,
            high_fidelity_certificate: node.high_fidelity_certificate,
            representative_count: node.representation.count,
        })
    }
}

fn compile_manifest_node_ids(
    manifest: &GaussianLodManifest,
) -> Result<Vec<LodNodeId>, ManifestHierarchyError> {
    let node_ids = manifest
        .nodes
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    for node in &manifest.nodes {
        let start = node.children.start as usize;
        let end = node
            .children
            .end()
            .ok_or(ManifestHierarchyError::ChildRangeOverflow)? as usize;
        node_ids
            .get(start..end)
            .ok_or(ManifestHierarchyError::ChildRangeOutOfBounds(node.id))?;
    }
    Ok(node_ids)
}

fn manifest_child_ids(node_ids: &[LodNodeId], range: LodIndexRange) -> Option<&[LodNodeId]> {
    let start = range.start as usize;
    let end = range.end()? as usize;
    node_ids.get(start..end)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestHierarchyError {
    InvalidManifest(String),
    ChildRangeOverflow,
    ChildRangeOutOfBounds(LodNodeId),
}

/// Dynamic residency projection for a manifest hierarchy backed by the bounded page cache.
#[derive(Clone, Copy, Debug)]
pub struct ManifestPageResidency<'a> {
    pub hierarchy: &'a ManifestLodHierarchy<'a>,
    pub cache: &'a LodPageCache,
}

impl LodResidency<LodNodeId> for ManifestPageResidency<'_> {
    fn is_resident(&self, node: LodNodeId) -> bool {
        self.hierarchy
            .page(node)
            .is_some_and(|page| self.cache.contains(page))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AllResident;

impl<NodeId> LodResidency<NodeId> for AllResident {
    fn is_resident(&self, _node: NodeId) -> bool {
        true
    }
}

impl<NodeId, F> LodResidency<NodeId> for F
where
    F: Fn(NodeId) -> bool,
{
    fn is_resident(&self, node: NodeId) -> bool {
        self(node)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LodViewProjection {
    Perspective { vertical_fov_radians: f32 },
    Orthographic { vertical_world_size: f32 },
}

/// Six normalized world-space half-spaces for conservative node-sphere tests.
/// A point is inside a plane when `normal.dot(point) + d > 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodFrustum {
    half_spaces: [Vec4; 6],
}

impl LodFrustum {
    /// Extracts Bevy/WGPU clip half-spaces, including reverse-Z near and far
    /// planes, from a world-to-clip matrix.
    pub fn from_clip_from_world(clip_from_world: Mat4) -> Self {
        let row3 = clip_from_world.row(3);
        let raw = [
            row3 + clip_from_world.row(0),
            row3 - clip_from_world.row(0),
            row3 + clip_from_world.row(1),
            row3 - clip_from_world.row(1),
            row3 - clip_from_world.row(2),
            clip_from_world.row(2),
        ];
        Self {
            half_spaces: raw.map(normalize_half_space),
        }
    }

    pub fn validate(self) -> bool {
        let valid_planes = self.half_spaces.iter().all(|plane| {
            let normal_length_squared = plane.truncate().length_squared();
            plane.is_finite()
                && normal_length_squared.is_finite()
                && (normal_length_squared > 0.0 || *plane == Vec4::ZERO)
        });
        valid_planes
            && self
                .half_spaces
                .iter()
                .filter(|plane| plane.truncate().length_squared() > 0.0)
                .count()
                >= 5
    }

    pub fn intersects_sphere(self, center: Vec3, radius: f32) -> bool {
        let center = center.extend(1.0);
        self.half_spaces
            .iter()
            .all(|plane| plane.dot(center) + radius >= 0.0)
    }
}

fn normalize_half_space(plane: Vec4) -> Vec4 {
    let length = plane.truncate().length();
    if length.is_finite() && length > 0.0 {
        plane / length
    } else if plane.is_finite() && plane.w >= 0.0 {
        // Infinite reverse-Z projections encode their absent far plane as a
        // zero normal with a non-negative constant. Keep it as an inert plane:
        // `0 + radius >= 0` accepts every conservative support sphere.
        Vec4::ZERO
    } else {
        Vec4::splat(f32::NAN)
    }
}

/// Camera quantities required for conservative perspective or orthographic error projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodView {
    /// Camera origin in world space.
    pub camera_position: Vec3,
    pub viewport_height_px: f32,
    pub projection: LodViewProjection,
    pub near_plane: f32,
    /// Affine transform from manifest/node coordinates into world space.
    pub world_from_local: Mat4,
    /// Optional orientation-aware frustum. Without it, selection remains
    /// projection/distance-aware but cannot reject offscreen hierarchy nodes.
    pub frustum: Option<LodFrustum>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LodProjectedNode {
    error_px: f32,
    support_radius_px: f32,
    coverage: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum LodViewEvaluatorProjection {
    Perspective { focal_length_px: f32 },
    Orthographic { scale_px_per_world: f32 },
}

/// Validated, per-selection projection state. Expensive view-only quantities
/// are computed once and reused for every hierarchy node in the traversal.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LodViewEvaluator {
    view: LodView,
    world_scale_upper_bound: f32,
    projection: LodViewEvaluatorProjection,
}

impl LodView {
    pub fn perspective(
        camera_position: Vec3,
        viewport_height_px: f32,
        vertical_fov_radians: f32,
        near_plane: f32,
    ) -> Self {
        Self {
            camera_position,
            viewport_height_px,
            projection: LodViewProjection::Perspective {
                vertical_fov_radians,
            },
            near_plane,
            world_from_local: Mat4::IDENTITY,
            frustum: None,
        }
    }

    pub fn orthographic(
        camera_position: Vec3,
        viewport_height_px: f32,
        vertical_world_size: f32,
        near_plane: f32,
    ) -> Self {
        Self {
            camera_position,
            viewport_height_px,
            projection: LodViewProjection::Orthographic {
                vertical_world_size,
            },
            near_plane,
            world_from_local: Mat4::IDENTITY,
            frustum: None,
        }
    }

    /// Adds orientation-aware conservative hierarchy visibility.
    pub fn with_clip_from_world(mut self, clip_from_world: Mat4) -> Self {
        self.frustum = Some(LodFrustum::from_clip_from_world(clip_from_world));
        self
    }

    pub fn with_frustum(mut self, frustum: LodFrustum) -> Self {
        self.frustum = Some(frustum);
        self
    }

    /// Applies the same cloud-local to world transform used by rendering.
    /// Node bounds and geometric errors remain manifest-local; selection
    /// transforms their centers and conservatively expands radii/errors by an
    /// upper bound on the affine transform's largest singular value.
    pub fn with_world_from_local(mut self, world_from_local: Mat4) -> Self {
        self.world_from_local = world_from_local;
        self
    }

    pub fn validate(self) -> Result<(), LodSelectionError<()>> {
        if !self.camera_position.is_finite() {
            return Err(LodSelectionError::InvalidView("camera_position"));
        }
        if !self.world_from_local.is_finite()
            || self.world_from_local.x_axis.w != 0.0
            || self.world_from_local.y_axis.w != 0.0
            || self.world_from_local.z_axis.w != 0.0
            || self.world_from_local.w_axis.w != 1.0
        {
            return Err(LodSelectionError::InvalidView("world_from_local"));
        }
        if !self.viewport_height_px.is_finite() || self.viewport_height_px <= 0.0 {
            return Err(LodSelectionError::InvalidView("viewport_height_px"));
        }
        match self.projection {
            LodViewProjection::Perspective {
                vertical_fov_radians,
            } if !vertical_fov_radians.is_finite()
                || vertical_fov_radians <= 0.0
                || vertical_fov_radians >= std::f32::consts::PI =>
            {
                return Err(LodSelectionError::InvalidView("vertical_fov_radians"));
            }
            LodViewProjection::Orthographic {
                vertical_world_size,
            } if !vertical_world_size.is_finite() || vertical_world_size <= 0.0 => {
                return Err(LodSelectionError::InvalidView("vertical_world_size"));
            }
            _ => {}
        }
        if !self.near_plane.is_finite() || self.near_plane <= 0.0 {
            return Err(LodSelectionError::InvalidView("near_plane"));
        }
        if self.frustum.is_some_and(|frustum| !frustum.validate()) {
            return Err(LodSelectionError::InvalidView("frustum"));
        }
        Ok(())
    }

    /// Tests a node's conservative bounding sphere against the optional
    /// frustum. `margin` expands support in world units.
    pub fn node_is_visible(self, metrics: LodNodeMetrics, margin: f32) -> bool {
        let (center, radius) = self.world_support_sphere(metrics);
        self.frustum.is_none_or(|frustum| {
            frustum.intersects_sphere(center, (radius + margin.max(0.0)).max(0.0))
        })
    }

    pub fn distance_to_center(self, metrics: LodNodeMetrics) -> f32 {
        self.camera_position
            .distance(self.world_from_local.transform_point3(metrics.center))
    }

    pub fn distance_to_surface(self, metrics: LodNodeMetrics) -> f32 {
        let (_, radius) = self.world_support_sphere(metrics);
        (self.distance_to_center(metrics) - radius).max(self.near_plane)
    }

    pub fn projected_error_px(self, metrics: LodNodeMetrics) -> f32 {
        self.projected_node(metrics).error_px
    }

    /// Conservative projected radius of the node support sphere, in pixels.
    pub fn projected_support_radius_px(self, metrics: LodNodeMetrics) -> f32 {
        self.projected_node(metrics).support_radius_px
    }

    /// Fraction of viewport height covered by the node support diameter.
    /// Structural detail demand is weighted by this value so receding nodes
    /// smoothly require fewer hierarchy levels without a world-unit distance
    /// curve. A support diameter at least as tall as the viewport returns one.
    pub fn projected_coverage(self, metrics: LodNodeMetrics) -> f32 {
        self.projected_node(metrics).coverage
    }

    /// Evaluates the exact stateless selector pressure for one hierarchy node.
    ///
    /// Camera-continuous presentation uses this same calculation in the render
    /// world so a parent/child blend boundary cannot drift away from the CPU
    /// selector's quality boundary. Hysteresis and residency are deliberately
    /// absent: they control topology preparation, not the view-conditioned
    /// presentation weight.
    #[cfg_attr(not(lod_render_path), allow(dead_code))]
    pub(crate) fn selection_pressure(
        self,
        metrics: LodNodeMetrics,
        target: LodQualityTarget,
        is_original_representation: bool,
    ) -> f32 {
        let projected = self.projected_node(metrics);
        target.node_pressure(
            metrics.quality_threshold(),
            projected.error_px,
            projected.coverage,
            metrics.high_fidelity_certificate,
            is_original_representation,
        )
    }

    fn projected_node(self, metrics: LodNodeMetrics) -> LodProjectedNode {
        LodViewEvaluator::from_view(self).projected_node(metrics)
    }

    /// Returns the projected-error part of the balanced selection target.
    /// Perspective distance is already represented by [`Self::projected_error_px`];
    /// applying an additional world-distance curve would break scale invariance.
    /// Orthographic translation likewise leaves this threshold unchanged.
    pub fn selection_error_limit_px(
        self,
        _metrics: LodNodeMetrics,
        settings: &GaussianLodSettings,
    ) -> f32 {
        settings.screen_space_error_limit_px()
    }

    fn world_support_sphere(self, metrics: LodNodeMetrics) -> (Vec3, f32) {
        (
            self.world_from_local.transform_point3(metrics.center),
            metrics.radius.max(0.0) * self.world_scale_upper_bound(),
        )
    }

    /// Gershgorin bound for the spectral norm of the affine linear part. This
    /// is exact for axis-aligned scale and conservative under rotation/shear.
    fn world_scale_upper_bound(self) -> f32 {
        let x = self.world_from_local.x_axis.truncate();
        let y = self.world_from_local.y_axis.truncate();
        let z = self.world_from_local.z_axis.truncate();
        let xx = x.dot(x);
        let xy = x.dot(y).abs();
        let xz = x.dot(z).abs();
        let yy = y.dot(y);
        let yz = y.dot(z).abs();
        let zz = z.dot(z);
        (xx + xy + xz).max(yy + xy + yz).max(zz + xz + yz).sqrt()
    }
}

impl LodViewEvaluator {
    fn from_validated(view: LodView) -> Result<Self, LodSelectionError<()>> {
        view.validate()?;
        Ok(Self::from_view(view))
    }

    fn from_view(view: LodView) -> Self {
        let projection = match view.projection {
            LodViewProjection::Perspective {
                vertical_fov_radians,
            } => LodViewEvaluatorProjection::Perspective {
                focal_length_px: 0.5 * view.viewport_height_px / (0.5 * vertical_fov_radians).tan(),
            },
            LodViewProjection::Orthographic {
                vertical_world_size,
            } => LodViewEvaluatorProjection::Orthographic {
                scale_px_per_world: view.viewport_height_px / vertical_world_size,
            },
        };
        Self {
            view,
            world_scale_upper_bound: view.world_scale_upper_bound(),
            projection,
        }
    }

    #[cfg(test)]
    fn node_is_visible(self, metrics: LodNodeMetrics, margin: f32) -> bool {
        let (center, radius) = self.world_support_sphere(metrics);
        self.view.frustum.is_none_or(|frustum| {
            frustum.intersects_sphere(center, (radius + margin.max(0.0)).max(0.0))
        })
    }

    #[cfg(test)]
    fn distance_to_center(self, metrics: LodNodeMetrics) -> f32 {
        self.view
            .camera_position
            .distance(self.view.world_from_local.transform_point3(metrics.center))
    }

    #[cfg(test)]
    fn distance_to_surface(self, metrics: LodNodeMetrics) -> f32 {
        let (center, radius) = self.world_support_sphere(metrics);
        (self.view.camera_position.distance(center) - radius).max(self.view.near_plane)
    }

    fn projected_node(self, metrics: LodNodeMetrics) -> LodProjectedNode {
        let (center, radius) = self.world_support_sphere(metrics);
        let projection_scale_px_per_world = match self.projection {
            LodViewEvaluatorProjection::Perspective { focal_length_px } => {
                let distance_to_surface =
                    (self.view.camera_position.distance(center) - radius).max(self.view.near_plane);
                focal_length_px / distance_to_surface
            }
            LodViewEvaluatorProjection::Orthographic { scale_px_per_world } => scale_px_per_world,
        };
        let support_radius_px = radius * projection_scale_px_per_world;
        LodProjectedNode {
            error_px: metrics.geometric_error
                * self.world_scale_upper_bound
                * projection_scale_px_per_world,
            support_radius_px,
            coverage: (2.0 * support_radius_px / self.view.viewport_height_px).clamp(0.0, 1.0),
        }
    }

    fn selection_error_limit_px(
        self,
        _metrics: LodNodeMetrics,
        settings: &GaussianLodSettings,
    ) -> f32 {
        settings.screen_space_error_limit_px()
    }

    fn world_support_sphere(self, metrics: LodNodeMetrics) -> (Vec3, f32) {
        (
            self.view.world_from_local.transform_point3(metrics.center),
            metrics.radius.max(0.0) * self.world_scale_upper_bound,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LodFrontier<NodeId> {
    /// A stable node-id-ordered complete global cut. Frustum visibility may
    /// limit refinement, but never removes source coverage from this cut.
    pub nodes: Vec<NodeId>,
    /// Missing desired nodes, deduplicated and ordered for deterministic requests.
    pub requested_nodes: Vec<NodeId>,
    pub status: LodEffectiveStatus,
}

/// Direction of one density-correct temporal hierarchy substitution.
///
/// A substitution always replaces a parent with all of its immediate children,
/// or all of those children with their parent. The old and new cohorts are
/// never present in the same cut, so this seam cannot introduce double density.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LodTemporalDirection {
    Coarsen,
    Refine,
}

/// Stable identity used to debounce one hierarchy boundary independently from
/// unrelated branches whose canonical target may still be changing.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LodTemporalSubstitutionKey<NodeId> {
    pub parent: NodeId,
    pub direction: LodTemporalDirection,
}

/// One parent-to-children (or children-to-parent) topology transaction.
///
/// `previous_gaussians + next_gaussians` is the conservative transition-energy
/// charge. ABI 16 packages consume this explicit transaction together with
/// their authored monotone parent-record runs; legacy packages or adapters
/// without the morph capability retain the bounded categorical endpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LodTemporalSubstitution<NodeId> {
    pub key: LodTemporalSubstitutionKey<NodeId>,
    pub previous_nodes: Vec<NodeId>,
    pub next_nodes: Vec<NodeId>,
    pub previous_gaussians: u64,
    pub next_gaussians: u64,
}

impl<NodeId> LodTemporalSubstitution<NodeId> {
    pub fn changed_gaussians(&self) -> u64 {
        self.previous_gaussians.saturating_add(self.next_gaussians)
    }
}

/// One bounded complete-cut advance toward the canonical stateless target.
#[cfg(any(feature = "lod", test))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LodTemporalFrontierStep<NodeId> {
    pub nodes: Vec<NodeId>,
    pub substitutions: Vec<LodTemporalSubstitution<NodeId>>,
    /// Nonresident nodes needed by the next topology transaction. The current
    /// complete cut remains unchanged while these pages stream.
    pub requested_nodes: Vec<NodeId>,
    pub changed_gaussians: u64,
    /// One topology cohort is indivisible. This is the amount by which that
    /// single cohort exceeded the ordinary per-frame energy budget.
    pub atomic_budget_overshoot: u64,
    pub reached_target: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LodSelectionError<NodeId> {
    InvalidSettings(LodSettingsError),
    InvalidView(&'static str),
    MissingNode(NodeId),
    InvalidNode(NodeId),
    HierarchyCycle(NodeId),
    CountOverflow,
}

struct PreviousCut<NodeId> {
    accepted: HashSet<NodeId>,
    refined: HashSet<NodeId>,
    /// Previously refined nodes held temporarily during a fine-to-coarse
    /// handoff. This is deliberately separate from error hysteresis: a hold
    /// only delays coarsening and can never make the rendered cut less
    /// detailed than the requested selector result.
    held_refined: HashSet<NodeId>,
    traversal_nodes_visited: u32,
}

impl<NodeId> Default for PreviousCut<NodeId> {
    fn default() -> Self {
        Self {
            accepted: HashSet::new(),
            refined: HashSet::new(),
            held_refined: HashSet::new(),
            traversal_nodes_visited: 0,
        }
    }
}

impl<NodeId: Copy + Hash + Ord> PreviousCut<NodeId> {
    fn compile<H: LodHierarchy<NodeId = NodeId>>(
        hierarchy: &H,
        previous_frontier: &[NodeId],
        settings: &GaussianLodSettings,
    ) -> Result<Self, LodSelectionError<NodeId>> {
        let mut cut = Self::default();
        let traversal_limit = settings.budgets.max_traversal_nodes_per_view;
        // History is optional. Reserve enough work to inspect every root so a
        // tight budget can still produce a valid current cut instead of being
        // consumed entirely by hysteresis bookkeeping.
        let reserved_root_work = u32::try_from(hierarchy.roots().len())
            .unwrap_or(u32::MAX)
            .min(traversal_limit);
        let mut remaining_work = traversal_limit - reserved_root_work;
        for &node in previous_frontier {
            if remaining_work == 0 {
                break;
            }
            checked_metrics(hierarchy, node)?;
            remaining_work -= 1;
            cut.traversal_nodes_visited += 1;
            cut.accepted.insert(node);
            let mut cursor = node;
            let mut chain = HashSet::new();
            while remaining_work > 0 {
                let Some(parent) = hierarchy.parent(cursor) else {
                    break;
                };
                if !chain.insert(parent) {
                    return Err(LodSelectionError::HierarchyCycle(parent));
                }
                // A prior frontier node already validated this ancestor and
                // every node above it. Stop at the shared suffix instead of
                // walking the same root path once per selected leaf.
                if cut.refined.contains(&parent) {
                    break;
                }
                checked_metrics(hierarchy, parent)?;
                remaining_work -= 1;
                cut.traversal_nodes_visited += 1;
                cut.refined.insert(parent);
                cursor = parent;
            }
            // Hysteresis history is optional. If a caller lowers the current
            // traversal budget below the old frontier depth, retain only the
            // bounded suffix instead of misclassifying a valid hierarchy as a
            // cycle. Cycles encountered inside that suffix are still rejected.
        }
        Ok(cut)
    }

    fn error_limit(&self, node: NodeId, base: f32, hysteresis: f32) -> f32 {
        if self.refined.contains(&node) {
            base * (1.0 - hysteresis)
        } else if self.accepted.contains(&node) {
            base * (1.0 + hysteresis)
        } else {
            base
        }
    }
}

/// Selects without frustum filtering. Useful as the normative quality oracle.
pub fn select_frontier<H, R>(
    hierarchy: &H,
    residency: &R,
    view: LodView,
    settings: &GaussianLodSettings,
) -> Result<LodFrontier<H::NodeId>, LodSelectionError<H::NodeId>>
where
    H: LodHierarchy,
    R: LodResidency<H::NodeId>,
{
    select_frontier_internal(
        hierarchy,
        residency,
        view,
        settings,
        &PreviousCut::default(),
        |_, _| true,
    )
}

/// Stateful selector that applies split/merge hysteresis relative to a prior
/// complete cut. Endpoint zero/one behavior remains exact and ignores hysteresis.
pub fn select_frontier_with_previous<H, R>(
    hierarchy: &H,
    residency: &R,
    view: LodView,
    settings: &GaussianLodSettings,
    previous_frontier: &[H::NodeId],
) -> Result<LodFrontier<H::NodeId>, LodSelectionError<H::NodeId>>
where
    H: LodHierarchy,
    R: LodResidency<H::NodeId>,
{
    let previous = if settings.quality_endpoint() == LodQualityEndpoint::Continuous
        && settings.hysteresis > 0.0
    {
        PreviousCut::compile(hierarchy, previous_frontier, settings)?
    } else {
        PreviousCut::default()
    };
    select_frontier_internal(hierarchy, residency, view, settings, &previous, |_, _| true)
}

/// Selects a camera-aware global hierarchy cut with a conservative
/// caller-supplied visibility predicate. Visibility gates refinement only;
/// missing descendants never replace a resident ancestor and off-screen
/// branches retain complete coarse coverage for arbitrary camera motion.
pub fn select_frontier_with_visibility<H, R, V>(
    hierarchy: &H,
    residency: &R,
    view: LodView,
    settings: &GaussianLodSettings,
    visible: V,
) -> Result<LodFrontier<H::NodeId>, LodSelectionError<H::NodeId>>
where
    H: LodHierarchy,
    R: LodResidency<H::NodeId>,
    V: FnMut(H::NodeId, LodNodeMetrics) -> bool,
{
    select_frontier_internal(
        hierarchy,
        residency,
        view,
        settings,
        &PreviousCut::default(),
        visible,
    )
}

/// Stateful camera-aware global selector with caller-supplied conservative
/// visibility. Visibility gates refinement only, never coverage.
pub fn select_frontier_with_previous_and_visibility<H, R, V>(
    hierarchy: &H,
    residency: &R,
    view: LodView,
    settings: &GaussianLodSettings,
    previous_frontier: &[H::NodeId],
    visible: V,
) -> Result<LodFrontier<H::NodeId>, LodSelectionError<H::NodeId>>
where
    H: LodHierarchy,
    R: LodResidency<H::NodeId>,
    V: FnMut(H::NodeId, LodNodeMetrics) -> bool,
{
    let previous = if settings.quality_endpoint() == LodQualityEndpoint::Continuous
        && settings.hysteresis > 0.0
    {
        PreviousCut::compile(hierarchy, previous_frontier, settings)?
    } else {
        PreviousCut::default()
    };
    select_frontier_internal(hierarchy, residency, view, settings, &previous, visible)
}

/// Stateful camera-aware selector with a bounded set of fine-to-coarse holds.
///
/// Runtime orchestration uses this narrow hook to stagger whole-subtree merges
/// over a few frames. Every result is still one complete hierarchy cut; no
/// old/new union, opacity cross-fade, or additional GPU candidate storage is
/// introduced. Refinement is never held.
#[cfg(test)]
pub(crate) fn select_frontier_with_previous_holds_and_visibility<H, R, V>(
    hierarchy: &H,
    residency: &R,
    view: LodView,
    settings: &GaussianLodSettings,
    previous_frontier: &[H::NodeId],
    held_refined: &BTreeSet<H::NodeId>,
    visible: V,
) -> Result<LodFrontier<H::NodeId>, LodSelectionError<H::NodeId>>
where
    H: LodHierarchy,
    R: LodResidency<H::NodeId>,
    V: FnMut(H::NodeId, LodNodeMetrics) -> bool,
{
    let mut previous = if settings.quality_endpoint() == LodQualityEndpoint::Continuous
        && settings.hysteresis > 0.0
    {
        PreviousCut::compile(hierarchy, previous_frontier, settings)?
    } else {
        PreviousCut::default()
    };
    if settings.quality_endpoint() == LodQualityEndpoint::Continuous {
        previous.held_refined.extend(held_refined.iter().copied());
    }
    select_frontier_internal(hierarchy, residency, view, settings, &previous, visible)
}

/// Finds the next one-level topology transactions between two complete cuts.
///
/// Coarsening candidates are bottom-up: all immediate children must be in the
/// current cut and a target ancestor must cover the same branch. Refinement
/// candidates are top-down: the current parent must contain target descendants.
/// Consequently every returned transaction can be applied independently while
/// preserving a complete source-space antichain.
#[cfg(any(feature = "lod", test))]
pub(crate) fn temporal_substitution_candidates<H>(
    hierarchy: &H,
    current_frontier: &[H::NodeId],
    target_frontier: &[H::NodeId],
) -> Result<Vec<LodTemporalSubstitution<H::NodeId>>, LodSelectionError<H::NodeId>>
where
    H: LodHierarchy,
{
    if current_frontier == target_frontier {
        return Ok(Vec::new());
    }

    let current = current_frontier.iter().copied().collect::<HashSet<_>>();
    let target = target_frontier.iter().copied().collect::<HashSet<_>>();
    let mut coarsening_parents = BTreeSet::new();
    for &node in current_frontier {
        checked_metrics(hierarchy, node)?;
        let Some(parent) = hierarchy.parent(node) else {
            continue;
        };
        if hierarchy.children(parent).is_empty()
            || !hierarchy
                .children(parent)
                .iter()
                .all(|child| current.contains(child))
        {
            continue;
        }
        let mut cursor = Some(parent);
        let mut covered_by_target_ancestor = false;
        let mut chain = HashSet::new();
        while let Some(candidate) = cursor {
            if !chain.insert(candidate) {
                return Err(LodSelectionError::HierarchyCycle(candidate));
            }
            if target.contains(&candidate) {
                covered_by_target_ancestor = true;
                break;
            }
            cursor = hierarchy.parent(candidate);
        }
        if covered_by_target_ancestor {
            coarsening_parents.insert(parent);
        }
    }

    let mut refinement_parents = BTreeSet::new();
    for &target_node in target_frontier {
        checked_metrics(hierarchy, target_node)?;
        let mut cursor = target_node;
        let mut chain = HashSet::new();
        while let Some(parent) = hierarchy.parent(cursor) {
            if !chain.insert(parent) {
                return Err(LodSelectionError::HierarchyCycle(parent));
            }
            if current.contains(&parent) {
                refinement_parents.insert(parent);
                break;
            }
            cursor = parent;
        }
    }

    let mut substitutions = Vec::with_capacity(
        coarsening_parents
            .len()
            .saturating_add(refinement_parents.len()),
    );
    for parent in coarsening_parents {
        // Preserve the hierarchy's validated child order: ABI16 morph runs
        // address the concatenated child representations in this order.
        let children = hierarchy.children(parent).to_vec();
        let previous_gaussians = children.iter().try_fold(0_u64, |count, child| {
            count
                .checked_add(u64::from(
                    checked_metrics(hierarchy, *child)?.representative_count,
                ))
                .ok_or(LodSelectionError::CountOverflow)
        })?;
        let next_gaussians = u64::from(checked_metrics(hierarchy, parent)?.representative_count);
        substitutions.push(LodTemporalSubstitution {
            key: LodTemporalSubstitutionKey {
                parent,
                direction: LodTemporalDirection::Coarsen,
            },
            previous_nodes: children,
            next_nodes: vec![parent],
            previous_gaussians,
            next_gaussians,
        });
    }
    for parent in refinement_parents {
        let children = hierarchy.children(parent).to_vec();
        let previous_gaussians =
            u64::from(checked_metrics(hierarchy, parent)?.representative_count);
        let next_gaussians = children.iter().try_fold(0_u64, |count, child| {
            count
                .checked_add(u64::from(
                    checked_metrics(hierarchy, *child)?.representative_count,
                ))
                .ok_or(LodSelectionError::CountOverflow)
        })?;
        substitutions.push(LodTemporalSubstitution {
            key: LodTemporalSubstitutionKey {
                parent,
                direction: LodTemporalDirection::Refine,
            },
            previous_nodes: vec![parent],
            next_nodes: children,
            previous_gaussians,
            next_gaussians,
        });
    }
    Ok(substitutions)
}

/// Applies a deterministic, energy-bounded subset of eligible transactions.
/// Coarsening is ordered first so mixed camera changes release active capacity
/// before refinement consumes it. At most one indivisible cohort may exceed
/// the ordinary energy budget; the overshoot remains explicit to callers.
#[cfg(any(feature = "lod", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LodTemporalStepBudget {
    pub max_active_gaussians: u64,
    pub max_changed_gaussians: u64,
    pub max_substitutions: usize,
}

#[cfg(any(feature = "lod", test))]
pub(crate) fn apply_temporal_substitution_step<NodeId, R>(
    current_frontier: &[NodeId],
    target_frontier: &[NodeId],
    current_active_gaussians: u64,
    substitutions: &[LodTemporalSubstitution<NodeId>],
    eligible: &BTreeSet<LodTemporalSubstitutionKey<NodeId>>,
    mut is_resident: R,
    budget: LodTemporalStepBudget,
) -> Result<LodTemporalFrontierStep<NodeId>, LodSelectionError<NodeId>>
where
    NodeId: Copy + Debug + Eq + Hash + Ord,
    R: FnMut(NodeId) -> bool,
{
    let mut nodes = current_frontier.iter().copied().collect::<BTreeSet<_>>();
    let mut active_gaussians = current_active_gaussians;
    let mut applied = Vec::new();
    let mut requested = BTreeSet::new();
    let mut changed_gaussians = 0_u64;
    let mut atomic_budget_overshoot = 0_u64;

    for substitution in substitutions {
        if !substitution
            .previous_nodes
            .iter()
            .all(|node| nodes.contains(node))
        {
            continue;
        }
        let missing = substitution
            .next_nodes
            .iter()
            .copied()
            .filter(|node| !is_resident(*node))
            .collect::<Vec<_>>();
        requested.extend(missing.iter().copied());
        if !missing.is_empty() || !eligible.contains(&substitution.key) {
            continue;
        }
        if applied.len() >= budget.max_substitutions.max(1) {
            continue;
        }
        let Some(next_active) = active_gaussians
            .checked_sub(substitution.previous_gaussians)
            .and_then(|count| count.checked_add(substitution.next_gaussians))
        else {
            return Err(LodSelectionError::CountOverflow);
        };
        if next_active > budget.max_active_gaussians {
            continue;
        }
        let work = substitution.changed_gaussians();
        let Some(next_work) = changed_gaussians.checked_add(work) else {
            return Err(LodSelectionError::CountOverflow);
        };
        if !applied.is_empty() && next_work > budget.max_changed_gaussians.max(1) {
            continue;
        }
        if applied.is_empty() && next_work > budget.max_changed_gaussians.max(1) {
            atomic_budget_overshoot = next_work - budget.max_changed_gaussians.max(1);
        }
        for node in &substitution.previous_nodes {
            nodes.remove(node);
        }
        nodes.extend(substitution.next_nodes.iter().copied());
        active_gaussians = next_active;
        changed_gaussians = next_work;
        applied.push(substitution.clone());
    }

    let nodes = nodes.into_iter().collect::<Vec<_>>();
    Ok(LodTemporalFrontierStep {
        reached_target: nodes.as_slice() == target_frontier,
        nodes,
        substitutions: applied,
        requested_nodes: requested.into_iter().collect(),
        changed_gaussians,
        atomic_budget_overshoot,
    })
}

/// Re-evaluates quality/status for a complete temporal cut without rerunning
/// hierarchy selection. The canonical target retains authority for budget and
/// traversal degradation, while the emitted cut reports its own exact count
/// and projected error.
#[cfg(any(feature = "lod", test))]
pub(crate) fn temporal_frontier_with_visibility<H, V>(
    hierarchy: &H,
    target: &LodFrontier<H::NodeId>,
    step: &LodTemporalFrontierStep<H::NodeId>,
    view: LodView,
    settings: &GaussianLodSettings,
    mut visible: V,
) -> Result<LodFrontier<H::NodeId>, LodSelectionError<H::NodeId>>
where
    H: LodHierarchy,
    V: FnMut(H::NodeId, LodNodeMetrics) -> bool,
{
    settings
        .validate()
        .map_err(LodSelectionError::InvalidSettings)?;
    let view = LodViewEvaluator::from_validated(view).map_err(|error| match error {
        LodSelectionError::InvalidView(field) => LodSelectionError::InvalidView(field),
        _ => unreachable!(),
    })?;
    let requested_target = settings.quality_target();
    let mut active_gaussians = 0_u64;
    let mut achieved_max_error_px = 0.0_f32;
    let mut achieved_max_target_ratio = 0.0_f32;
    for &node in &step.nodes {
        let metrics = checked_metrics(hierarchy, node)?;
        active_gaussians = active_gaussians
            .checked_add(u64::from(metrics.representative_count))
            .ok_or(LodSelectionError::CountOverflow)?;
        if visible(node, metrics) {
            let projected = view.projected_node(metrics);
            achieved_max_error_px = achieved_max_error_px.max(projected.error_px);
            achieved_max_target_ratio =
                achieved_max_target_ratio.max(requested_target.node_pressure(
                    metrics.quality_threshold(),
                    projected.error_px,
                    projected.coverage,
                    metrics.high_fidelity_certificate,
                    hierarchy.children(node).is_empty(),
                ));
        }
    }

    let mut requested_nodes = target
        .requested_nodes
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    requested_nodes.extend(step.requested_nodes.iter().copied());
    let degradation = if requested_nodes.is_empty() {
        target.status.degradation
    } else {
        target.status.degradation.merge(LodDegradation::Residency)
    };
    Ok(LodFrontier {
        nodes: step.nodes.clone(),
        requested_nodes: requested_nodes.iter().copied().collect(),
        status: LodEffectiveStatus {
            requested_target,
            achieved_max_error_px,
            achieved_max_target_ratio,
            degradation,
            active_gaussians,
            visited_nodes: target.status.visited_nodes,
            requested_pages: requested_nodes.len().try_into().unwrap_or(u32::MAX),
        },
    })
}

fn select_frontier_internal<H, R, V>(
    hierarchy: &H,
    residency: &R,
    view: LodView,
    settings: &GaussianLodSettings,
    previous: &PreviousCut<H::NodeId>,
    mut visible: V,
) -> Result<LodFrontier<H::NodeId>, LodSelectionError<H::NodeId>>
where
    H: LodHierarchy,
    R: LodResidency<H::NodeId>,
    V: FnMut(H::NodeId, LodNodeMetrics) -> bool,
{
    settings
        .validate()
        .map_err(LodSelectionError::InvalidSettings)?;
    let view = LodViewEvaluator::from_validated(view).map_err(|error| match error {
        LodSelectionError::InvalidView(field) => LodSelectionError::InvalidView(field),
        _ => unreachable!(),
    })?;

    let endpoint = settings.quality_endpoint();
    let traversal_limit = settings.budgets.max_traversal_nodes_per_view;
    let mut state = SelectionState::<H::NodeId>::new(previous.traversal_nodes_visited);

    for &root in hierarchy.roots() {
        state.visit(traversal_limit)?;
        let metrics = checked_metrics(hierarchy, root)?;
        let root_visible = visible(root, metrics);
        if residency.is_resident(root) {
            state.insert_frontier(root, metrics, root_visible)?;
            if root_visible {
                state.maybe_queue_candidate(hierarchy, view, settings, previous, root, metrics);
            }
        } else {
            // Roots are the permanent global coverage guard. An off-screen
            // missing root is still required so the published cut remains
            // valid after an arbitrary camera teleport.
            state.requested.insert(root);
            state.degradation = state.degradation.merge(LodDegradation::Residency);
        }
    }

    if endpoint != LodQualityEndpoint::Coarsest {
        while let Some(candidate) = state.candidates.pop() {
            if !state.frontier.contains_key(&candidate.node)
                || state.expanded.contains(&candidate.node)
            {
                continue;
            }
            state.expanded.insert(candidate.node);

            let mut children = Vec::new();
            let mut missing_children = Vec::new();
            let mut child_count = 0_u64;
            let mut enumeration_complete = true;
            for &child in hierarchy.children(candidate.node) {
                if state.visited_nodes >= traversal_limit {
                    state.degradation = state.degradation.merge(LodDegradation::TraversalBudget);
                    enumeration_complete = false;
                    break;
                }
                let metrics = checked_metrics(hierarchy, child)?;
                state.visit(traversal_limit)?;
                let child_visible = visible(child, metrics);
                // Selection budgets constrain the logical split, independent
                // of whether its child pages happen to be resident yet. A
                // parent is replaced atomically by all children, including
                // off-screen siblings, so the resulting frontier remains a
                // complete source-space antichain after camera motion.
                child_count = child_count
                    .checked_add(u64::from(metrics.representative_count))
                    .ok_or(LodSelectionError::CountOverflow)?;
                if !residency.is_resident(child) {
                    missing_children.push(child);
                }
                children.push((child, metrics, child_visible));
            }

            if !enumeration_complete {
                // An incomplete child enumeration cannot safely replace its ancestor.
                continue;
            }

            let parent_count = state.frontier[&candidate.node] as u64;
            let next_count = state
                .active_gaussians
                .checked_sub(parent_count)
                .and_then(|count| count.checked_add(child_count))
                .ok_or(LodSelectionError::CountOverflow)?;
            if next_count > settings.budgets.max_active_gaussians {
                state.degradation = state.degradation.merge(LodDegradation::ActiveBudget);
                continue;
            }

            if !missing_children.is_empty() {
                state.requested.extend(missing_children.iter().copied());
                state.degradation = state.degradation.merge(LodDegradation::Residency);
                continue;
            }

            state.frontier.remove(&candidate.node);
            state.visible_frontier.remove(&candidate.node);
            state.active_gaussians = next_count;
            for (child, metrics, child_visible) in children {
                state.frontier.insert(child, metrics.representative_count);
                if child_visible {
                    state.visible_frontier.insert(child);
                    state
                        .maybe_queue_candidate(hierarchy, view, settings, previous, child, metrics);
                }
            }
        }
    }

    if state.active_gaussians > settings.budgets.max_active_gaussians {
        state.degradation = state.degradation.merge(LodDegradation::ActiveBudget);
    }

    let mut achieved_max_error_px = 0.0_f32;
    let mut achieved_max_target_ratio = 0.0_f32;
    let requested_target = settings.quality_target();
    for &node in &state.visible_frontier {
        let metrics = checked_metrics(hierarchy, node)?;
        let projected = view.projected_node(metrics);
        let selection_error_px = projected.error_px;
        achieved_max_error_px = achieved_max_error_px.max(selection_error_px);
        achieved_max_target_ratio = achieved_max_target_ratio.max(requested_target.node_pressure(
            metrics.quality_threshold(),
            selection_error_px,
            projected.coverage,
            metrics.high_fidelity_certificate,
            hierarchy.children(node).is_empty(),
        ));
    }

    let status = LodEffectiveStatus {
        requested_target,
        achieved_max_error_px,
        achieved_max_target_ratio,
        degradation: state.degradation,
        active_gaussians: state.active_gaussians,
        visited_nodes: state.visited_nodes,
        requested_pages: state.requested.len().try_into().unwrap_or(u32::MAX),
    };

    let mut nodes = state.frontier.into_keys().collect::<Vec<_>>();
    let mut requested_nodes = state.requested.into_iter().collect::<Vec<_>>();
    nodes.sort_unstable();
    requested_nodes.sort_unstable();
    Ok(LodFrontier {
        nodes,
        requested_nodes,
        status,
    })
}

fn checked_metrics<H: LodHierarchy>(
    hierarchy: &H,
    node: H::NodeId,
) -> Result<LodNodeMetrics, LodSelectionError<H::NodeId>> {
    let metrics = hierarchy
        .metrics(node)
        .ok_or(LodSelectionError::MissingNode(node))?;
    if !metrics.validate() {
        return Err(LodSelectionError::InvalidNode(node));
    }
    Ok(metrics)
}

#[derive(Clone, Copy, Debug)]
struct Candidate<NodeId> {
    node: NodeId,
    priority: f32,
}

impl<NodeId: Ord> PartialEq for Candidate<NodeId> {
    fn eq(&self, other: &Self) -> bool {
        self.priority.total_cmp(&other.priority) == Ordering::Equal && self.node == other.node
    }
}

impl<NodeId: Ord> Eq for Candidate<NodeId> {}

impl<NodeId: Ord> PartialOrd for Candidate<NodeId> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<NodeId: Ord> Ord for Candidate<NodeId> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            // Lower stable identifiers win exact ties.
            .then_with(|| other.node.cmp(&self.node))
    }
}

struct SelectionState<NodeId: Copy + Hash + Ord> {
    frontier: HashMap<NodeId, u32>,
    /// Nodes in the global frontier whose support intersects the current
    /// view. Quality status is view-local even though coverage is global.
    visible_frontier: HashSet<NodeId>,
    candidates: BinaryHeap<Candidate<NodeId>>,
    requested: HashSet<NodeId>,
    expanded: HashSet<NodeId>,
    active_gaussians: u64,
    visited_nodes: u32,
    degradation: LodDegradation,
}

impl<NodeId: Copy + Hash + Ord> SelectionState<NodeId> {
    fn new(visited_nodes: u32) -> Self {
        Self {
            frontier: HashMap::new(),
            visible_frontier: HashSet::new(),
            candidates: BinaryHeap::new(),
            requested: HashSet::new(),
            expanded: HashSet::new(),
            active_gaussians: 0,
            visited_nodes,
            degradation: LodDegradation::None,
        }
    }

    fn visit<ErrorNode>(&mut self, limit: u32) -> Result<(), LodSelectionError<ErrorNode>> {
        if self.visited_nodes >= limit {
            return Err(LodSelectionError::CountOverflow);
        }
        self.visited_nodes += 1;
        Ok(())
    }

    fn insert_frontier<ErrorNode>(
        &mut self,
        node: NodeId,
        metrics: LodNodeMetrics,
        visible: bool,
    ) -> Result<(), LodSelectionError<ErrorNode>> {
        if self
            .frontier
            .insert(node, metrics.representative_count)
            .is_none()
        {
            self.active_gaussians = self
                .active_gaussians
                .checked_add(u64::from(metrics.representative_count))
                .ok_or(LodSelectionError::CountOverflow)?;
        }
        if visible {
            self.visible_frontier.insert(node);
        } else {
            self.visible_frontier.remove(&node);
        }
        Ok(())
    }

    fn maybe_queue_candidate<H: LodHierarchy<NodeId = NodeId>>(
        &mut self,
        hierarchy: &H,
        view: LodViewEvaluator,
        settings: &GaussianLodSettings,
        previous: &PreviousCut<NodeId>,
        node: NodeId,
        metrics: LodNodeMetrics,
    ) {
        if hierarchy.children(node).is_empty() {
            return;
        }
        let projected = view.projected_node(metrics);
        let projected_error = projected.error_px;
        let target = settings.quality_target();
        let endpoint = target.endpoint();
        let base_error_limit = view.selection_error_limit_px(metrics, settings);
        let error_limit = if endpoint == LodQualityEndpoint::Continuous {
            previous.error_limit(node, base_error_limit, settings.hysteresis)
        } else {
            base_error_limit
        };
        let pressure = target.node_pressure_with_error_limit(
            metrics.quality_threshold(),
            projected_error,
            error_limit,
            projected.coverage,
            metrics.high_fidelity_certificate,
            false,
        );
        let should_refine = match endpoint {
            LodQualityEndpoint::Coarsest => false,
            LodQualityEndpoint::Original => true,
            LodQualityEndpoint::Continuous => {
                pressure > 1.0 || previous.held_refined.contains(&node)
            }
        };
        if should_refine {
            self.candidates.push(Candidate {
                node,
                priority: pressure,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use std::collections::{BTreeMap, BTreeSet};

    use std::hint::black_box;
    use std::time::Instant;

    use super::*;
    use crate::{
        gaussian::formats::{
            planar_3d::PlanarGaussian3d,
            planar_3d_lod::{GaussianLodBuildSettings, build_planar_3d_lod},
        },
        testing::LodTestScene,
    };

    struct TestHierarchy {
        roots: Vec<u32>,
        nodes: BTreeMap<u32, (LodNodeMetrics, Vec<u32>)>,
    }

    struct DenseBenchmarkHierarchy {
        roots: Vec<u32>,
        nodes: Vec<(Option<u32>, LodNodeMetrics, Vec<u32>)>,
    }

    impl DenseBenchmarkHierarchy {
        fn binary(levels: u32) -> Self {
            let node_count = (1usize << levels) - 1;
            let first_leaf = (1usize << (levels - 1)) - 1;
            let nodes = (0..node_count)
                .map(|index| {
                    let parent = (index > 0).then(|| ((index - 1) / 2) as u32);
                    let first_child = index * 2 + 1;
                    let children = if first_child < node_count {
                        vec![first_child as u32, (first_child + 1) as u32]
                    } else {
                        Vec::new()
                    };
                    let leaf = index >= first_leaf;
                    (
                        parent,
                        LodNodeMetrics {
                            center: Vec3::new(0.0, 0.0, 10.0),
                            radius: 0.5,
                            geometric_error: if leaf { 0.0 } else { 10.0 },
                            appearance_error: 0.0,
                            opacity_error: 0.0,
                            quality_min: if leaf { 1.0 } else { 0.0 },
                            quality_max: if leaf { 1.0 } else { 0.1 },
                            high_fidelity_certificate: 1.0,
                            representative_count: if leaf { 128 } else { 1 },
                        },
                        children,
                    )
                })
                .collect();
            Self {
                roots: vec![0],
                nodes,
            }
        }

        fn leaves(&self) -> Vec<u32> {
            self.nodes
                .iter()
                .enumerate()
                .filter_map(|(index, (_, _, children))| children.is_empty().then_some(index as u32))
                .collect()
        }
    }

    impl LodHierarchy for DenseBenchmarkHierarchy {
        type NodeId = u32;

        fn roots(&self) -> &[Self::NodeId] {
            &self.roots
        }

        fn parent(&self, node: Self::NodeId) -> Option<Self::NodeId> {
            self.nodes.get(node as usize).and_then(|node| node.0)
        }

        fn children(&self, node: Self::NodeId) -> &[Self::NodeId] {
            self.nodes
                .get(node as usize)
                .map(|node| node.2.as_slice())
                .unwrap_or_default()
        }

        fn metrics(&self, node: Self::NodeId) -> Option<LodNodeMetrics> {
            self.nodes.get(node as usize).map(|node| node.1)
        }
    }

    impl LodHierarchy for TestHierarchy {
        type NodeId = u32;

        fn roots(&self) -> &[Self::NodeId] {
            &self.roots
        }

        fn parent(&self, node: Self::NodeId) -> Option<Self::NodeId> {
            self.nodes
                .iter()
                .find_map(|(parent, (_, children))| children.contains(&node).then_some(*parent))
        }

        fn children(&self, node: Self::NodeId) -> &[Self::NodeId] {
            &self.nodes[&node].1
        }

        fn metrics(&self, node: Self::NodeId) -> Option<LodNodeMetrics> {
            self.nodes.get(&node).map(|entry| entry.0)
        }
    }

    fn hierarchy() -> TestHierarchy {
        let metric = |z, error, quality_min, quality_max, count| LodNodeMetrics {
            center: Vec3::new(0.0, 0.0, z),
            radius: 0.5,
            geometric_error: error,
            appearance_error: 0.0,
            opacity_error: 0.0,
            quality_min,
            quality_max,
            high_fidelity_certificate: 1.0,
            representative_count: count,
        };
        TestHierarchy {
            roots: vec![0],
            nodes: BTreeMap::from([
                (0, (metric(10.0, 4.0, 0.0, 0.02, 1), vec![1, 2])),
                (1, (metric(9.0, 1.0, 0.02, 0.75, 2), vec![3, 4])),
                (2, (metric(11.0, 1.0, 0.02, 0.75, 2), vec![5, 6])),
                (3, (metric(9.0, 0.0, 0.75, 1.0, 4), vec![])),
                (4, (metric(9.5, 0.0, 0.75, 1.0, 4), vec![])),
                (5, (metric(10.5, 0.0, 0.75, 1.0, 4), vec![])),
                (6, (metric(11.0, 0.0, 0.75, 1.0, 4), vec![])),
            ]),
        }
    }

    fn view() -> LodView {
        LodView::perspective(Vec3::ZERO, 1080.0, std::f32::consts::FRAC_PI_2, 0.1)
    }

    fn hierarchy_source_range(node: u32) -> std::ops::Range<u32> {
        match node {
            0 => 0..16,
            1 => 0..8,
            2 => 8..16,
            3 => 0..4,
            4 => 4..8,
            5 => 8..12,
            6 => 12..16,
            _ => panic!("unknown test node {node}"),
        }
    }

    fn assert_complete_source_antichain(hierarchy: &TestHierarchy, nodes: &[u32]) {
        let selected = nodes.iter().copied().collect::<HashSet<_>>();
        for &node in nodes {
            let mut cursor = node;
            while let Some(parent) = hierarchy.parent(cursor) {
                assert!(
                    !selected.contains(&parent),
                    "cut contains ancestor {parent} and descendant {node}"
                );
                cursor = parent;
            }
        }

        let mut ranges = nodes
            .iter()
            .copied()
            .map(hierarchy_source_range)
            .collect::<Vec<_>>();
        ranges.sort_unstable_by_key(|range| range.start);
        let mut covered_until = 0;
        for range in ranges {
            assert_eq!(
                range.start, covered_until,
                "cut has a source-space gap or overlap"
            );
            covered_until = range.end;
        }
        assert_eq!(covered_until, 16, "cut does not cover the source tail");
    }

    #[test]
    fn view_frustum_is_orientation_aware_and_margin_is_conservative() {
        let frustum_view = view().with_clip_from_world(Mat4::IDENTITY);
        let metric = |center| LodNodeMetrics {
            center,
            radius: 0.1,
            geometric_error: 0.0,
            appearance_error: 0.0,
            opacity_error: 0.0,
            quality_min: 0.0,
            quality_max: 1.0,
            high_fidelity_certificate: 1.0,
            representative_count: 1,
        };
        assert!(frustum_view.node_is_visible(metric(Vec3::new(0.0, 0.0, 0.5)), 0.0));
        assert!(!frustum_view.node_is_visible(metric(Vec3::new(2.0, 0.0, 0.5)), 0.0));
        assert!(frustum_view.node_is_visible(metric(Vec3::new(2.0, 0.0, 0.5)), 1.0));

        let invalid = view().with_clip_from_world(Mat4::ZERO);
        assert!(matches!(
            invalid.validate(),
            Err(LodSelectionError::InvalidView("frustum"))
        ));
    }

    #[test]
    fn infinite_reverse_z_frustum_treats_the_absent_far_plane_as_inert() {
        let clip_from_view =
            Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1);
        let frustum = LodFrustum::from_clip_from_world(clip_from_view);
        assert!(frustum.validate());
        assert!(frustum.intersects_sphere(Vec3::new(0.0, 0.0, -1.0), 0.01));
        assert!(!frustum.intersects_sphere(Vec3::new(100.0, 0.0, -1.0), 0.01));
    }

    #[test]
    fn transformed_clouds_use_world_space_distance_error_and_visibility() {
        let world_from_local = Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            bevy::math::Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 0.0),
        );
        let clip_from_world = Mat4::from_translation(Vec3::new(-10.0, 0.0, 0.0));
        let transformed = LodView::perspective(
            Vec3::new(10.0, 0.0, 8.0),
            100.0,
            std::f32::consts::FRAC_PI_2,
            0.1,
        )
        .with_world_from_local(world_from_local)
        .with_clip_from_world(clip_from_world);
        let metrics = LodNodeMetrics {
            center: Vec3::ZERO,
            radius: 1.0,
            geometric_error: 2.0,
            appearance_error: 0.0,
            opacity_error: 0.0,
            quality_min: 0.0,
            quality_max: 1.0,
            high_fidelity_certificate: 1.0,
            representative_count: 1,
        };

        assert!(transformed.validate().is_ok());
        assert!((transformed.distance_to_center(metrics) - 8.0).abs() < 1e-5);
        assert!((transformed.distance_to_surface(metrics) - 6.0).abs() < 1e-5);
        assert!((transformed.projected_error_px(metrics) - 100.0 / 3.0).abs() < 1e-4);
        assert!((transformed.projected_support_radius_px(metrics) - 50.0 / 3.0).abs() < 1e-4);
        assert!((transformed.projected_coverage(metrics) - 1.0 / 3.0).abs() < 1e-5);
        assert!(transformed.node_is_visible(metrics, 0.0));

        let missing_transform = LodView::perspective(
            Vec3::new(10.0, 0.0, 8.0),
            100.0,
            std::f32::consts::FRAC_PI_2,
            0.1,
        )
        .with_clip_from_world(clip_from_world);
        assert!(!missing_transform.node_is_visible(metrics, 0.0));
    }

    #[test]
    fn precomputed_view_evaluator_matches_analytic_view_contracts() {
        fn analytic_world_scale(world_from_local: Mat4) -> f32 {
            let x = world_from_local.x_axis.truncate();
            let y = world_from_local.y_axis.truncate();
            let z = world_from_local.z_axis.truncate();
            let xx = x.dot(x);
            let xy = x.dot(y).abs();
            let xz = x.dot(z).abs();
            let yy = y.dot(y);
            let yz = y.dot(z).abs();
            let zz = z.dot(z);
            (xx + xy + xz).max(yy + xy + yz).max(zz + xz + yz).sqrt()
        }

        fn analytic_projection(view: LodView, metrics: LodNodeMetrics) -> LodProjectedNode {
            let transform_scale = analytic_world_scale(view.world_from_local);
            let center = view.world_from_local.transform_point3(metrics.center);
            let radius = metrics.radius.max(0.0) * transform_scale;
            let projection_scale_px_per_world = match view.projection {
                LodViewProjection::Perspective {
                    vertical_fov_radians,
                } => {
                    let focal_length_px =
                        0.5 * view.viewport_height_px / (0.5 * vertical_fov_radians).tan();
                    let distance_to_surface =
                        (view.camera_position.distance(center) - radius).max(view.near_plane);
                    focal_length_px / distance_to_surface
                }
                LodViewProjection::Orthographic {
                    vertical_world_size,
                } => view.viewport_height_px / vertical_world_size,
            };
            let support_radius_px = radius * projection_scale_px_per_world;
            LodProjectedNode {
                error_px: metrics.geometric_error * transform_scale * projection_scale_px_per_world,
                support_radius_px,
                coverage: (2.0 * support_radius_px / view.viewport_height_px).clamp(0.0, 1.0),
            }
        }

        fn assert_close(actual: f32, expected: f32) {
            let tolerance = expected.abs().max(1.0) * 1e-5;
            assert!(
                (actual - expected).abs() <= tolerance,
                "actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }

        fn assert_projected_close(actual: LodProjectedNode, expected: LodProjectedNode) {
            assert_close(actual.error_px, expected.error_px);
            assert_close(actual.support_radius_px, expected.support_radius_px);
            assert_close(actual.coverage, expected.coverage);
        }

        let metrics = LodNodeMetrics {
            center: Vec3::new(0.25, -0.5, 1.0),
            radius: 0.7,
            geometric_error: 0.35,
            appearance_error: 0.0,
            opacity_error: 0.0,
            quality_min: 0.0,
            quality_max: 1.0,
            high_fidelity_certificate: 1.0,
            representative_count: 1,
        };
        let transforms = [
            Mat4::IDENTITY,
            Mat4::from_scale_rotation_translation(
                Vec3::new(2.0, 0.75, 1.5),
                bevy::math::Quat::from_rotation_y(0.37),
                Vec3::new(-2.0, 0.5, 4.0),
            ),
            Mat4::from_cols(
                Vec4::new(1.25, 0.15, 0.0, 0.0),
                Vec4::new(0.35, 0.9, 0.1, 0.0),
                Vec4::new(0.0, 0.2, 1.5, 0.0),
                Vec4::new(-2.0, 0.5, 4.0, 1.0),
            ),
        ];

        for world_from_local in transforms {
            let views = [
                LodView::perspective(Vec3::new(3.0, -1.0, 12.0), 1440.0, 1.1, 0.25)
                    .with_world_from_local(world_from_local),
                LodView::orthographic(Vec3::new(-4.0, 2.0, 9.0), 900.0, 12.0, 0.1)
                    .with_world_from_local(world_from_local),
            ];
            for view in views {
                let evaluator = LodViewEvaluator::from_validated(view).unwrap();
                let expected = analytic_projection(view, metrics);
                assert_projected_close(evaluator.projected_node(metrics), expected);
                assert_projected_close(view.projected_node(metrics), expected);

                let transform_scale = analytic_world_scale(world_from_local);
                let center = world_from_local.transform_point3(metrics.center);
                let radius = metrics.radius * transform_scale;
                assert_close(
                    evaluator.distance_to_center(metrics),
                    view.camera_position.distance(center),
                );
                assert_close(
                    evaluator.distance_to_surface(metrics),
                    (view.camera_position.distance(center) - radius).max(view.near_plane),
                );
            }
        }

        let frustum_view = LodView::perspective(Vec3::new(0.0, 0.0, 4.0), 1080.0, 1.0, 0.1)
            .with_world_from_local(Mat4::from_scale_rotation_translation(
                Vec3::new(2.0, 1.0, 0.5),
                bevy::math::Quat::IDENTITY,
                Vec3::new(0.25, 0.0, 0.0),
            ))
            .with_clip_from_world(Mat4::IDENTITY);
        let evaluator = LodViewEvaluator::from_validated(frustum_view).unwrap();
        for (center, margin) in [
            (Vec3::new(0.0, 0.0, 1.0), 0.0_f32),
            (Vec3::new(1.0, 0.0, 1.0), 0.0_f32),
            (Vec3::new(1.0, 0.0, 1.0), 1.1_f32),
        ] {
            let node = LodNodeMetrics { center, ..metrics };
            let world_scale = analytic_world_scale(frustum_view.world_from_local);
            let world_center = frustum_view.world_from_local.transform_point3(center);
            let world_radius = node.radius * world_scale;
            let expected = frustum_view
                .frustum
                .unwrap()
                .intersects_sphere(world_center, (world_radius + margin.max(0.0)).max(0.0));
            assert_eq!(evaluator.node_is_visible(node, margin), expected);
            assert_eq!(frustum_view.node_is_visible(node, margin), expected);
        }
    }

    #[test]
    fn endpoint_zero_is_roots_and_one_is_leaves() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.budgets.max_active_gaussians = 100;

        settings.quality = 0.0;
        let coarse = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(coarse.nodes, vec![0]);
        assert_eq!(coarse.status.active_gaussians, 1);

        settings.quality = 0.99;
        let near_exact = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(near_exact.nodes, vec![3, 4, 5, 6]);
        assert_eq!(near_exact.status.active_gaussians, 16);
        assert!(matches!(
            near_exact.status.requested_target,
            crate::gaussian::lod_settings::LodQualityTarget::Balanced {
                detail_fraction: 0.99,
                ..
            }
        ));

        settings.quality = 1.0;
        let exact = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(exact.nodes, vec![3, 4, 5, 6]);
        assert_eq!(exact.status.active_gaussians, 16);
        assert_eq!(exact.status.degradation, LodDegradation::None);
    }

    #[test]
    fn temporal_cohorts_advance_both_directions_without_double_density() {
        let hierarchy = hierarchy();
        let leaves = vec![3, 4, 5, 6];
        let root = vec![0];

        let mut current = leaves.clone();
        let mut active = 16;
        let mut coarsening_cuts = vec![current.clone()];
        for _ in 0..4 {
            let candidates = temporal_substitution_candidates(&hierarchy, &current, &root).unwrap();
            let eligible = candidates.iter().map(|candidate| candidate.key).collect();
            let step = apply_temporal_substitution_step(
                &current,
                &root,
                active,
                &candidates,
                &eligible,
                |_| true,
                LodTemporalStepBudget {
                    max_active_gaussians: 100,
                    max_changed_gaussians: 10,
                    max_substitutions: 16,
                },
            )
            .unwrap();
            assert!(step.changed_gaussians <= 10);
            assert_eq!(step.atomic_budget_overshoot, 0);
            assert_complete_source_antichain(&hierarchy, &step.nodes);
            active = step
                .nodes
                .iter()
                .map(|node| u64::from(hierarchy.metrics(*node).unwrap().representative_count))
                .sum();
            current = step.nodes;
            coarsening_cuts.push(current.clone());
            if current == root {
                break;
            }
        }
        assert_eq!(current, root);
        assert_eq!(coarsening_cuts, [leaves, vec![1, 5, 6], vec![1, 2], root]);

        let mut current = vec![0];
        let mut active = 1;
        let target = vec![3, 4, 5, 6];
        let mut refinement_cuts = vec![current.clone()];
        for _ in 0..4 {
            let candidates =
                temporal_substitution_candidates(&hierarchy, &current, &target).unwrap();
            let eligible = candidates.iter().map(|candidate| candidate.key).collect();
            let step = apply_temporal_substitution_step(
                &current,
                &target,
                active,
                &candidates,
                &eligible,
                |_| true,
                LodTemporalStepBudget {
                    max_active_gaussians: 100,
                    max_changed_gaussians: 10,
                    max_substitutions: 16,
                },
            )
            .unwrap();
            assert!(step.changed_gaussians <= 10);
            assert_complete_source_antichain(&hierarchy, &step.nodes);
            active = step
                .nodes
                .iter()
                .map(|node| u64::from(hierarchy.metrics(*node).unwrap().representative_count))
                .sum();
            current = step.nodes;
            refinement_cuts.push(current.clone());
            if current == target {
                break;
            }
        }
        assert_eq!(current, target);
        assert_eq!(
            refinement_cuts,
            [vec![0], vec![1, 2], vec![2, 3, 4], target]
        );
    }

    #[test]
    fn temporal_cohort_budget_allows_only_one_explicit_atomic_overshoot() {
        let hierarchy = hierarchy();
        let current = vec![3, 4, 5, 6];
        let target = vec![0];
        let candidates = temporal_substitution_candidates(&hierarchy, &current, &target).unwrap();
        let eligible = candidates.iter().map(|candidate| candidate.key).collect();
        let step = apply_temporal_substitution_step(
            &current,
            &target,
            16,
            &candidates,
            &eligible,
            |_| true,
            LodTemporalStepBudget {
                max_active_gaussians: 100,
                max_changed_gaussians: 4,
                max_substitutions: 16,
            },
        )
        .unwrap();
        assert_eq!(step.substitutions.len(), 1);
        assert_eq!(step.changed_gaussians, 10);
        assert_eq!(step.atomic_budget_overshoot, 6);
        assert_complete_source_antichain(&hierarchy, &step.nodes);
    }

    #[test]
    fn temporal_coarsening_requests_missing_intermediate_parent_without_changing_cut() {
        let hierarchy = hierarchy();
        let current = vec![3, 4, 5, 6];
        let target = vec![0];
        let candidates = temporal_substitution_candidates(&hierarchy, &current, &target).unwrap();
        let eligible = candidates.iter().map(|candidate| candidate.key).collect();
        let step = apply_temporal_substitution_step(
            &current,
            &target,
            16,
            &candidates,
            &eligible,
            |node| node != 1,
            LodTemporalStepBudget {
                max_active_gaussians: 100,
                max_changed_gaussians: 100,
                max_substitutions: 16,
            },
        )
        .unwrap();
        assert_eq!(step.nodes, vec![2, 3, 4]);
        assert_eq!(step.requested_nodes, vec![1]);
        assert_eq!(step.substitutions.len(), 1);
        assert_complete_source_antichain(&hierarchy, &step.nodes);
    }

    #[test]
    fn temporal_settled_cut_is_canonical_from_coarse_and_fine_histories() {
        fn settle(hierarchy: &TestHierarchy, mut current: Vec<u32>, target: &[u32]) -> Vec<u32> {
            for _ in 0..8 {
                if current == target {
                    return current;
                }
                let active = current
                    .iter()
                    .map(|node| u64::from(hierarchy.metrics(*node).unwrap().representative_count))
                    .sum();
                let candidates =
                    temporal_substitution_candidates(hierarchy, &current, target).unwrap();
                let eligible = candidates.iter().map(|candidate| candidate.key).collect();
                current = apply_temporal_substitution_step(
                    &current,
                    target,
                    active,
                    &candidates,
                    &eligible,
                    |_| true,
                    LodTemporalStepBudget {
                        max_active_gaussians: 100,
                        max_changed_gaussians: 10,
                        max_substitutions: 16,
                    },
                )
                .unwrap()
                .nodes;
            }
            current
        }

        let hierarchy = hierarchy();
        let canonical = vec![1, 2];
        assert_eq!(settle(&hierarchy, vec![0], &canonical), canonical);
        assert_eq!(settle(&hierarchy, vec![3, 4, 5, 6], &canonical), canonical);
    }

    #[test]
    fn temporal_coarsening_holds_preserve_one_complete_finer_cut() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings {
            quality: 0.10,
            hysteresis: 0.0,
            ..Default::default()
        };
        settings.budgets.max_active_gaussians = 100;
        let previous = [3, 4, 5, 6];

        let desired = select_frontier_with_previous_and_visibility(
            &hierarchy,
            &AllResident,
            view(),
            &settings,
            &previous,
            |_, _| true,
        )
        .unwrap();
        assert_eq!(desired.nodes, vec![0]);

        let held = select_frontier_with_previous_holds_and_visibility(
            &hierarchy,
            &AllResident,
            view(),
            &settings,
            &previous,
            &BTreeSet::from([0, 1, 2]),
            |_, _| true,
        )
        .unwrap();
        assert_eq!(held.nodes, previous);
        assert_eq!(held.status.active_gaussians, 16);
        assert!(held.requested_nodes.is_empty());

        // Endpoint zero remains categorical; temporal smoothing never changes
        // the explicit coarsest contract.
        settings.quality = 0.0;
        let endpoint = select_frontier_with_previous_holds_and_visibility(
            &hierarchy,
            &AllResident,
            view(),
            &settings,
            &previous,
            &BTreeSet::from([0, 1, 2]),
            |_, _| true,
        )
        .unwrap();
        assert_eq!(endpoint.nodes, vec![0]);
    }

    #[test]
    fn near_exact_balanced_quality_may_accept_low_error_internal_nodes() {
        let mut hierarchy = hierarchy();
        hierarchy.nodes.get_mut(&0).unwrap().0.geometric_error = 0.001;
        let mut settings = GaussianLodSettings::default();
        settings.budgets.max_active_gaussians = 100;
        settings.hysteresis = 0.0;

        settings.quality = 0.99;
        let balanced = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(balanced.nodes, vec![0]);
        assert!(balanced.status.achieved_max_error_px > 0.0);
        assert!(balanced.status.achieved_max_error_px <= settings.screen_space_error_limit_px());
        assert!(matches!(
            balanced.status.requested_target,
            crate::gaussian::lod_settings::LodQualityTarget::Balanced {
                detail_fraction: 0.99,
                ..
            }
        ));

        settings.quality = 1.0;
        let original = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(original.nodes, vec![3, 4, 5, 6]);
        assert_eq!(
            original.status.requested_target,
            crate::gaussian::lod_settings::LodQualityTarget::Original
        );
    }

    #[test]
    fn continuous_error_authority_bounds_huge_structural_shortcuts() {
        let metric = |error, quality_min, quality_max, count| LodNodeMetrics {
            center: Vec3::new(0.0, 0.0, 10.0),
            radius: 0.5,
            geometric_error: error,
            appearance_error: 0.0,
            opacity_error: 0.0,
            quality_min,
            quality_max,
            high_fidelity_certificate: 1.0,
            representative_count: count,
        };
        let hierarchy = TestHierarchy {
            roots: vec![0],
            nodes: BTreeMap::from([
                (0, (metric(4.0, 0.55, 0.65, 1), vec![1, 2])),
                (1, (metric(0.0, 0.65, 1.0, 4), vec![])),
                (2, (metric(0.0, 0.65, 1.0, 4), vec![])),
            ]),
        };
        let mut settings = GaussianLodSettings {
            quality: 0.25,
            hysteresis: 0.0,
            ..Default::default()
        };
        settings.budgets.max_active_gaussians = 100;

        let coarse = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(coarse.nodes, vec![0]);
        assert!(coarse.status.achieved_max_error_px > 100.0);
        assert!(coarse.status.achieved_max_target_ratio <= 1.0);

        settings.quality = 0.50;
        let authoritative = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(authoritative.nodes, vec![1, 2]);
        assert_eq!(authoritative.status.active_gaussians, 8);
        assert_eq!(authoritative.status.achieved_max_error_px, 0.0);
        assert!(authoritative.status.achieved_max_target_ratio <= 1.0);
    }

    #[test]
    fn near_top_quality_refines_an_uncertified_low_error_internal_node() {
        let metric = |certificate, count| LodNodeMetrics {
            center: Vec3::new(0.0, 0.0, 10.0),
            radius: 0.5,
            geometric_error: 0.0,
            appearance_error: 0.0,
            opacity_error: 0.0,
            quality_min: 0.0,
            quality_max: 1.0,
            high_fidelity_certificate: certificate,
            representative_count: count,
        };
        let mut hierarchy = TestHierarchy {
            roots: vec![0],
            nodes: BTreeMap::from([
                (0, (metric(0.0, 1), vec![1, 2])),
                (1, (metric(1.0, 2), vec![])),
                (2, (metric(1.0, 2), vec![])),
            ]),
        };
        let mut settings = GaussianLodSettings {
            quality: 0.90,
            hysteresis: 0.0,
            ..Default::default()
        };
        settings.budgets.max_active_gaussians = 100;

        let ordinary = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(ordinary.nodes, vec![0]);
        assert_eq!(ordinary.status.achieved_max_target_ratio, 0.0);

        settings.quality = 0.95;
        let guarded = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(guarded.nodes, vec![1, 2]);
        assert!(guarded.status.achieved_max_target_ratio <= 1.0);

        hierarchy
            .nodes
            .get_mut(&0)
            .unwrap()
            .0
            .high_fidelity_certificate =
            crate::gaussian::lod_settings::high_quality_certificate_demand(0.95, 0.0);
        let certified = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(certified.nodes, vec![0]);
        assert_eq!(certified.status.achieved_max_target_ratio, 1.0);
    }

    #[test]
    fn active_budget_keeps_a_complete_coarser_cut() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = 4;

        let selected = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(selected.nodes, vec![1, 2]);
        assert_eq!(selected.status.active_gaussians, 4);
        assert_eq!(selected.status.degradation, LodDegradation::ActiveBudget);
        assert_eq!(
            selected.status.requested_target,
            crate::gaussian::lod_settings::LodQualityTarget::Original
        );
        assert!(selected.status.achieved_max_error_px > 0.0);
        assert_eq!(selected.status.achieved_max_target_ratio, f32::MAX);
    }

    #[test]
    fn over_budget_nonresident_split_does_not_grow_demand_or_residency() {
        let hierarchy = hierarchy();
        let mut resident = BTreeSet::from([0]);
        let mut demand_driven_residency_revision = 0_u64;
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        // Replacing the one-record root with its two two-record children
        // would require four active records, so this split cannot fit.
        settings.budgets.max_active_gaussians = 3;

        for _ in 0..8 {
            let selected = select_frontier(
                &hierarchy,
                &|node| resident.contains(&node),
                view(),
                &settings,
            )
            .unwrap();
            assert_eq!(selected.nodes, vec![0]);
            assert!(selected.requested_nodes.is_empty());
            assert_eq!(selected.status.active_gaussians, 1);
            assert_eq!(selected.status.degradation, LodDegradation::ActiveBudget);

            for requested in selected.requested_nodes {
                if resident.insert(requested) {
                    demand_driven_residency_revision += 1;
                }
            }
        }

        assert_eq!(resident, BTreeSet::from([0]));
        assert_eq!(demand_driven_residency_revision, 0);
    }

    #[test]
    fn fitting_nonresident_split_preserves_child_demand_at_budget_boundary() {
        let hierarchy = hierarchy();
        let mut resident = BTreeSet::from([0]);
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        // The root-to-children split requires exactly four active records and
        // must therefore retain the existing inclusive budget semantics.
        settings.budgets.max_active_gaussians = 4;

        let waiting = select_frontier(
            &hierarchy,
            &|node| resident.contains(&node),
            view(),
            &settings,
        )
        .unwrap();
        assert_eq!(waiting.nodes, vec![0]);
        assert_eq!(waiting.requested_nodes, vec![1, 2]);
        assert_eq!(waiting.status.active_gaussians, 1);
        assert_eq!(waiting.status.degradation, LodDegradation::Residency);

        resident.extend(waiting.requested_nodes);
        let resident_children = select_frontier(
            &hierarchy,
            &|node| resident.contains(&node),
            view(),
            &settings,
        )
        .unwrap();
        assert_eq!(resident_children.nodes, vec![1, 2]);
        assert!(resident_children.requested_nodes.is_empty());
        assert_eq!(resident_children.status.active_gaussians, 4);
        assert_eq!(
            resident_children.status.degradation,
            LodDegradation::ActiveBudget
        );
    }

    #[test]
    fn missing_child_requests_once_and_keeps_resident_ancestor() {
        let hierarchy = hierarchy();
        let resident = BTreeSet::from([0, 1]);
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = 100;

        let selected = select_frontier(
            &hierarchy,
            &|node| resident.contains(&node),
            view(),
            &settings,
        )
        .unwrap();
        assert_eq!(selected.nodes, vec![0]);
        assert_eq!(selected.requested_nodes, vec![2]);
        assert_eq!(selected.status.degradation, LodDegradation::Residency);
    }

    #[test]
    fn frontier_and_requests_publish_in_node_order() {
        let mut hierarchy = hierarchy();
        hierarchy.nodes.get_mut(&0).unwrap().1.reverse();
        let resident = HashSet::from([0]);
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = 100;

        let selected = select_frontier(
            &hierarchy,
            &|node| resident.contains(&node),
            view(),
            &settings,
        )
        .unwrap();
        assert_eq!(selected.nodes, vec![0]);
        assert_eq!(selected.requested_nodes, vec![1, 2]);
    }

    #[test]
    fn visibility_refines_only_intersecting_branches_but_keeps_invisible_siblings() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = 100;

        let left = select_frontier_with_visibility(
            &hierarchy,
            &AllResident,
            view(),
            &settings,
            |node, _| matches!(node, 0 | 1 | 3 | 4),
        )
        .unwrap();
        assert_eq!(left.nodes, vec![2, 3, 4]);
        assert_eq!(left.status.active_gaussians, 10);
        assert_complete_source_antichain(&hierarchy, &left.nodes);

        let right = select_frontier_with_visibility(
            &hierarchy,
            &AllResident,
            view(),
            &settings,
            |node, _| matches!(node, 0 | 2 | 5 | 6),
        )
        .unwrap();
        assert_eq!(right.nodes, vec![1, 5, 6]);
        assert_eq!(right.status.active_gaussians, 10);
        assert_complete_source_antichain(&hierarchy, &right.nodes);

        // The cut selected before an arbitrary camera teleport is still a
        // complete source partition for the newly visible half. Camera motion
        // only changes which covered branch should be refined next.
        assert_complete_source_antichain(&hierarchy, &left.nodes);
    }

    #[test]
    fn offscreen_roots_remain_in_the_global_guard_cut_and_demand() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = 100;

        let resident =
            select_frontier_with_visibility(&hierarchy, &AllResident, view(), &settings, |_, _| {
                false
            })
            .unwrap();
        assert_eq!(resident.nodes, vec![0]);
        assert!(resident.requested_nodes.is_empty());
        assert_eq!(resident.status.achieved_max_error_px, 0.0);
        assert_eq!(resident.status.achieved_max_target_ratio, 0.0);
        assert_complete_source_antichain(&hierarchy, &resident.nodes);

        let missing =
            select_frontier_with_visibility(&hierarchy, &|_| false, view(), &settings, |_, _| {
                false
            })
            .unwrap();
        assert!(missing.nodes.is_empty());
        assert_eq!(missing.requested_nodes, vec![0]);
        assert_eq!(missing.status.degradation, LodDegradation::Residency);
    }

    #[test]
    fn missing_invisible_sibling_keeps_parent_until_atomic_split_is_resident() {
        let hierarchy = hierarchy();
        let resident = BTreeSet::from([0, 1, 3, 4]);
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = 100;

        let selected = select_frontier_with_visibility(
            &hierarchy,
            &|node| resident.contains(&node),
            view(),
            &settings,
            |node, _| matches!(node, 0 | 1 | 3 | 4),
        )
        .unwrap();
        assert_eq!(selected.nodes, vec![0]);
        assert_eq!(selected.requested_nodes, vec![2]);
        assert_eq!(selected.status.degradation, LodDegradation::Residency);
        assert_complete_source_antichain(&hierarchy, &selected.nodes);
    }

    #[test]
    fn incomplete_atomic_split_neither_substitutes_nor_requests_children() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = 100;
        // The root plus only its first child fit in the traversal budget. That
        // is insufficient evidence for a complete sibling substitution.
        settings.budgets.max_traversal_nodes_per_view = 2;

        let selected = select_frontier_with_visibility(
            &hierarchy,
            &|node| node == 0,
            view(),
            &settings,
            |node, _| matches!(node, 0 | 1),
        )
        .unwrap();
        assert_eq!(selected.nodes, vec![0]);
        assert!(selected.requested_nodes.is_empty());
        assert_eq!(selected.status.degradation, LodDegradation::TraversalBudget);
        assert_complete_source_antichain(&hierarchy, &selected.nodes);
    }

    #[test]
    fn global_coverage_refinement_stays_within_active_budget() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        // One visible half may refine: 2 off-screen representatives plus 8
        // visible representatives exactly fill the global cut budget.
        settings.budgets.max_active_gaussians = 10;

        let selected = select_frontier_with_visibility(
            &hierarchy,
            &AllResident,
            view(),
            &settings,
            |node, _| matches!(node, 0 | 1 | 3 | 4),
        )
        .unwrap();
        assert_eq!(selected.nodes, vec![2, 3, 4]);
        assert_eq!(selected.status.active_gaussians, 10);
        assert!(selected.status.active_gaussians <= settings.budgets.max_active_gaussians);
        assert_complete_source_antichain(&hierarchy, &selected.nodes);
    }

    #[test]
    fn default_quality_counts_are_monotonic_across_quality_and_distance() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.budgets.max_active_gaussians = 100;
        settings.hysteresis = 0.0;
        let views = [
            LodView::perspective(
                Vec3::new(0.0, 0.0, 8.0),
                1080.0,
                std::f32::consts::FRAC_PI_2,
                0.1,
            ),
            view(),
            LodView::perspective(
                Vec3::new(0.0, 0.0, -100.0),
                1080.0,
                std::f32::consts::FRAC_PI_2,
                0.1,
            ),
        ];
        let mut rows = Vec::new();
        for quality in [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0] {
            settings.quality = quality;
            rows.push(
                views
                    .into_iter()
                    .map(|view| {
                        select_frontier(&hierarchy, &AllResident, view, &settings)
                            .unwrap()
                            .status
                            .active_gaussians
                    })
                    .collect::<Vec<_>>(),
            );
        }
        assert!(
            (0..views.len()).all(|view_index| rows
                .windows(2)
                .all(|pair| pair[1][view_index] >= pair[0][view_index])),
            "{rows:?}"
        );
        assert!(
            rows.iter()
                .all(|row| row.windows(2).all(|pair| pair[1] <= pair[0])),
            "{rows:?}"
        );
        assert_eq!(rows[0], [1, 1, 1]);
        assert_eq!(rows[1], [10, 4, 1]);
        assert_eq!(rows[2], [16, 16, 4]);
        assert!(rows[3..].iter().all(|row| row == &[16, 16, 16]));
    }

    #[test]
    fn lowering_traversal_budget_truncates_hysteresis_history_without_false_cycle() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.75;
        settings.budgets.max_active_gaussians = 100;
        settings.budgets.max_traversal_nodes_per_view = 100;
        let previous = select_frontier(&hierarchy, &AllResident, view(), &settings)
            .unwrap()
            .nodes;
        assert!(
            previous
                .iter()
                .any(|node| hierarchy.parent(*node).is_some())
        );

        settings.budgets.max_traversal_nodes_per_view = 1;
        assert!(
            select_frontier_with_previous(&hierarchy, &AllResident, view(), &settings, &previous)
                .is_ok()
        );
    }

    #[test]
    fn previous_cut_history_shares_one_traversal_work_budget() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.5;
        settings.budgets.max_traversal_nodes_per_view = 3;
        let cut = PreviousCut::compile(&hierarchy, &[3, 4, 5, 6], &settings).unwrap();
        assert_eq!(cut.accepted, HashSet::from([3]));
        assert_eq!(cut.refined, HashSet::from([1]));
        assert_eq!(cut.traversal_nodes_visited, 2);

        let selected = select_frontier_with_previous(
            &hierarchy,
            &AllResident,
            view(),
            &settings,
            &[3, 4, 5, 6],
        )
        .unwrap();
        assert_eq!(selected.nodes, vec![0]);
        assert_eq!(selected.status.visited_nodes, 3);
        assert_eq!(selected.status.degradation, LodDegradation::TraversalBudget);

        for quality in [0.0, 1.0] {
            settings.quality = quality;
            assert!(
                select_frontier_with_previous(
                    &hierarchy,
                    &AllResident,
                    view(),
                    &settings,
                    &[u32::MAX],
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn invalid_metrics_and_view_are_rejected() {
        let mut invalid_metrics = hierarchy();
        assert_eq!(
            invalid_metrics.metrics(0).unwrap().quality_threshold(),
            0.01
        );
        invalid_metrics.nodes.get_mut(&0).unwrap().0.geometric_error = f32::NAN;
        assert!(matches!(
            select_frontier(
                &invalid_metrics,
                &AllResident,
                view(),
                &GaussianLodSettings::default()
            ),
            Err(LodSelectionError::InvalidNode(0))
        ));

        let mut invalid_interval = hierarchy();
        invalid_interval.nodes.get_mut(&0).unwrap().0.quality_min = 0.5;
        invalid_interval.nodes.get_mut(&0).unwrap().0.quality_max = 0.25;
        assert!(matches!(
            select_frontier(
                &invalid_interval,
                &AllResident,
                view(),
                &GaussianLodSettings::default()
            ),
            Err(LodSelectionError::InvalidNode(0))
        ));

        let mut invalid_certificate = hierarchy();
        invalid_certificate
            .nodes
            .get_mut(&0)
            .unwrap()
            .0
            .high_fidelity_certificate = f32::NAN;
        assert!(matches!(
            select_frontier(
                &invalid_certificate,
                &AllResident,
                view(),
                &GaussianLodSettings::default()
            ),
            Err(LodSelectionError::InvalidNode(0))
        ));

        let invalid_view = LodView::perspective(Vec3::ZERO, f32::INFINITY, 1.0, 0.1);
        assert!(matches!(
            invalid_view.validate(),
            Err(LodSelectionError::InvalidView("viewport_height_px"))
        ));
    }

    #[test]
    fn orthographic_error_is_distance_invariant() {
        let orthographic = LodView::orthographic(Vec3::ZERO, 1_000.0, 10.0, 0.1);
        orthographic.validate().unwrap();
        let near = LodNodeMetrics {
            center: Vec3::new(0.0, 0.0, 2.0),
            radius: 0.5,
            geometric_error: 0.25,
            appearance_error: 0.0,
            opacity_error: 0.0,
            quality_min: 0.0,
            quality_max: 1.0,
            high_fidelity_certificate: 1.0,
            representative_count: 1,
        };
        let far = LodNodeMetrics {
            center: Vec3::new(0.0, 0.0, 200.0),
            ..near
        };
        assert_eq!(
            orthographic.projected_error_px(near),
            orthographic.projected_error_px(far)
        );
        assert_eq!(orthographic.projected_error_px(near), 25.0);
        assert_eq!(orthographic.projected_support_radius_px(near), 50.0);
        assert_eq!(orthographic.projected_support_radius_px(far), 50.0);
        assert_eq!(orthographic.projected_coverage(near), 0.1);
        assert_eq!(orthographic.projected_coverage(far), 0.1);

        let invalid = LodView::orthographic(Vec3::ZERO, 1_000.0, 0.0, 0.1);
        assert!(matches!(
            invalid.validate(),
            Err(LodSelectionError::InvalidView("vertical_world_size"))
        ));
    }

    #[test]
    fn opposing_camera_distances_select_independent_frontiers() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.25;
        settings.budgets.max_active_gaussians = 100;

        let near = LodView::perspective(
            Vec3::new(0.0, 0.0, 8.0),
            1080.0,
            std::f32::consts::FRAC_PI_2,
            0.1,
        );
        let far = LodView::perspective(
            Vec3::new(0.0, 0.0, -100.0),
            1080.0,
            std::f32::consts::FRAC_PI_2,
            0.1,
        );
        let near_selection = select_frontier(&hierarchy, &AllResident, near, &settings).unwrap();
        let far_selection = select_frontier(&hierarchy, &AllResident, far, &settings).unwrap();
        assert!(
            near_selection.status.active_gaussians > far_selection.status.active_gaussians,
            "near={near_selection:?}, far={far_selection:?}"
        );
    }

    #[test]
    fn projection_coherence_uses_only_projected_error() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.5;
        settings.hysteresis = 0.0;
        settings.budgets.max_active_gaussians = 100;

        let near = LodView::orthographic(Vec3::ZERO, 100.0, 10.0, 0.1);
        let far = LodView::orthographic(Vec3::new(0.0, 0.0, -1_000.0), 100.0, 10.0, 0.1);
        let perspective = LodView::perspective(
            Vec3::new(0.0, 0.0, 4.5),
            100.0,
            std::f32::consts::FRAC_PI_2,
            0.1,
        );
        let near_selection = select_frontier(&hierarchy, &AllResident, near, &settings).unwrap();
        let far_selection = select_frontier(&hierarchy, &AllResident, far, &settings).unwrap();
        let perspective_selection =
            select_frontier(&hierarchy, &AllResident, perspective, &settings).unwrap();

        assert_eq!(near_selection.nodes, far_selection.nodes);
        assert_eq!(near_selection.nodes, perspective_selection.nodes);
        assert_eq!(near_selection.status.active_gaussians, 4);
        assert_eq!(near_selection.status.achieved_max_error_px, 10.0);
        assert!(near_selection.status.achieved_max_target_ratio <= 1.0);
        assert_eq!(
            near_selection.status.achieved_max_target_ratio,
            far_selection.status.achieved_max_target_ratio
        );
        let root = hierarchy.metrics(0).unwrap();
        assert!(
            (near.projected_error_px(root) - perspective.projected_error_px(root)).abs() < 1e-5
        );
        assert!(
            (near.projected_support_radius_px(root)
                - perspective.projected_support_radius_px(root))
            .abs()
                < 1e-5
        );
        assert!(
            (near.projected_coverage(root) - perspective.projected_coverage(root)).abs() < 1e-6
        );
    }

    #[test]
    fn default_perspective_selection_is_uniform_scale_invariant() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        // Keep the assertion below a leaf-exact target so the comparison
        // exercises nonzero projected error and target-ratio math.
        settings.quality = 0.25;
        settings.hysteresis = 0.0;
        settings.budgets.max_active_gaussians = 100;

        let base = view();
        let scaled = view().with_world_from_local(Mat4::from_scale(Vec3::splat(10.0)));
        let base_selection = select_frontier(&hierarchy, &AllResident, base, &settings).unwrap();
        let scaled_selection =
            select_frontier(&hierarchy, &AllResident, scaled, &settings).unwrap();

        assert_eq!(base_selection.nodes, vec![1, 2]);
        assert_eq!(base_selection.nodes, scaled_selection.nodes);
        assert_eq!(
            base_selection.status.active_gaussians,
            scaled_selection.status.active_gaussians
        );
        assert!(
            (base_selection.status.achieved_max_error_px
                - scaled_selection.status.achieved_max_error_px)
                .abs()
                < 1e-4
        );
        assert!(
            (base_selection.status.achieved_max_target_ratio
                - scaled_selection.status.achieved_max_target_ratio)
                .abs()
                < 1e-5
        );
        let root = hierarchy.metrics(0).unwrap();
        assert!((base.projected_error_px(root) - scaled.projected_error_px(root)).abs() < 1e-4);
        assert!(
            (base.projected_support_radius_px(root) - scaled.projected_support_radius_px(root))
                .abs()
                < 1e-4
        );
        assert!((base.projected_coverage(root) - scaled.projected_coverage(root)).abs() < 1e-6);
    }

    #[test]
    fn previous_cut_hysteresis_prevents_split_merge_chatter() {
        let mut hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 0.5;
        settings.hysteresis = 0.1;
        settings.budgets.max_active_gaussians = 100;

        // Keep structural pressure below one so this fixture isolates the
        // fixed projected-error contract. Place the root just inside the
        // effective cap: a previously refined cut crosses the 10% split
        // threshold, while a stateless or previously accepted cut stays coarse.
        let root = &mut hierarchy.nodes.get_mut(&0).unwrap().0;
        root.quality_min = 0.25;
        root.quality_max = 0.75;
        let reference_view = view();
        let current_error_px = reference_view.projected_error_px(*root);
        let target_error_px = settings
            .quality_target()
            .effective_max_screen_space_error_px()
            .unwrap()
            * 0.99;
        let hysteresis_view = LodView::perspective(
            Vec3::ZERO,
            reference_view.viewport_height_px * target_error_px / current_error_px,
            std::f32::consts::FRAC_PI_2,
            0.1,
        );

        let stateless =
            select_frontier(&hierarchy, &AllResident, hysteresis_view, &settings).unwrap();
        assert_eq!(stateless.nodes, vec![0]);

        // The same projected error remains refined when approached from a
        // finer cut, but remains coarse when approached from the root cut.
        let from_fine = select_frontier_with_previous(
            &hierarchy,
            &AllResident,
            hysteresis_view,
            &settings,
            &[3, 4, 5, 6],
        )
        .unwrap();
        let from_coarse = select_frontier_with_previous(
            &hierarchy,
            &AllResident,
            hysteresis_view,
            &settings,
            &[0],
        )
        .unwrap();
        assert_eq!(from_fine.nodes, vec![1, 2]);
        assert_eq!(from_coarse.nodes, vec![0]);
    }

    fn force_sparse_compiled_index(
        mut hierarchy: CompiledManifestLodHierarchy,
    ) -> CompiledManifestLodHierarchy {
        hierarchy.node_indices = CompiledManifestNodeIndices::Sparse(
            hierarchy
                .manifest
                .nodes
                .iter()
                .enumerate()
                .map(|(index, node)| (node.id, index))
                .collect(),
        );
        hierarchy.page_indices = CompiledManifestPageIndices::Sparse(
            hierarchy
                .manifest
                .pages
                .iter()
                .enumerate()
                .map(|(index, page)| (page.id, index))
                .collect(),
        );
        hierarchy
    }

    fn remap_manifest_to_even_node_ids(mut manifest: GaussianLodManifest) -> GaussianLodManifest {
        let remap = |node: LodNodeId| {
            LodNodeId(
                node.0
                    .checked_mul(2)
                    .expect("test manifest node IDs fit in u64"),
            )
        };
        for root in &mut manifest.roots {
            *root = remap(*root);
        }
        for node in &mut manifest.nodes {
            node.id = remap(node.id);
            node.parent = node.parent.map(remap);
        }
        manifest
    }

    #[test]
    fn portable_manifest_runs_through_the_normative_selector() {
        let source = PlanarGaussian3d::from(
            LodTestScene::nested_octants(2)
                .gaussians
                .into_iter()
                .map(|entry| entry.gaussian)
                .collect::<Vec<_>>(),
        );
        let built = build_planar_3d_lod(
            &source,
            GaussianLodBuildSettings {
                leaf_capacity: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let hierarchy = ManifestLodHierarchy::new(&built.manifest).unwrap();
        let compiled_dense = CompiledManifestLodHierarchy::new(built.manifest.clone()).unwrap();
        assert!(matches!(
            &compiled_dense.node_indices,
            CompiledManifestNodeIndices::DenseOneBased
        ));
        assert!(matches!(
            &compiled_dense.page_indices,
            CompiledManifestPageIndices::DenseOneBased
        ));
        assert!(compiled_dense.node(LodNodeId::INVALID).is_none());
        assert!(compiled_dense.node(LodNodeId(u64::MAX)).is_none());
        assert!(compiled_dense.page_descriptor(LodPageId::INVALID).is_none());
        assert!(
            compiled_dense
                .page_descriptor(LodPageId(u64::MAX))
                .is_none()
        );

        let compiled_forced_sparse = force_sparse_compiled_index(compiled_dense.clone());
        for descriptor in &built.manifest.pages {
            assert_eq!(
                compiled_dense.page_descriptor(descriptor.id),
                compiled_forced_sparse.page_descriptor(descriptor.id)
            );
        }
        let compiled_sparse = CompiledManifestLodHierarchy::new(remap_manifest_to_even_node_ids(
            built.manifest.clone(),
        ))
        .unwrap();
        assert!(matches!(
            &compiled_sparse.node_indices,
            CompiledManifestNodeIndices::Sparse(_)
        ));
        assert!(compiled_sparse.node(LodNodeId(1)).is_none());
        assert!(compiled_sparse.node(LodNodeId(2)).is_some());
        let mut settings = GaussianLodSettings::default();
        settings.budgets.max_active_gaussians = 1_000;

        // The borrowed hierarchy keeps its ordered node index and is the
        // reference for both compiled lookup strategies. Direct and hash-based
        // lookup must publish identical deterministic cuts at every quality.
        for quality in [0.0, 0.35, 0.65, 1.0] {
            settings.quality = quality;
            let ordered = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
            let dense = select_frontier(&compiled_dense, &AllResident, view(), &settings).unwrap();
            let forced_sparse =
                select_frontier(&compiled_forced_sparse, &AllResident, view(), &settings).unwrap();
            let sparse =
                select_frontier(&compiled_sparse, &AllResident, view(), &settings).unwrap();
            assert_eq!(dense, ordered, "quality={quality}");
            assert_eq!(forced_sparse, dense, "forced sparse quality={quality}");
            assert_eq!(
                sparse.status, dense.status,
                "sparse status quality={quality}"
            );
            assert_eq!(
                sparse.nodes,
                dense
                    .nodes
                    .iter()
                    .map(|node| LodNodeId(node.0 * 2))
                    .collect::<Vec<_>>(),
                "sparse nodes quality={quality}"
            );
            assert_eq!(
                sparse.requested_nodes,
                dense
                    .requested_nodes
                    .iter()
                    .map(|node| LodNodeId(node.0 * 2))
                    .collect::<Vec<_>>(),
                "sparse requests quality={quality}"
            );
            assert!(dense.nodes.windows(2).all(|nodes| nodes[0] < nodes[1]));
            assert!(
                dense
                    .requested_nodes
                    .windows(2)
                    .all(|nodes| nodes[0] < nodes[1])
            );
        }

        settings.quality = 0.0;
        let coarse = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert_eq!(coarse.nodes, built.manifest.roots);
        assert_eq!(
            coarse.status.active_gaussians,
            built.manifest.quality.coarsest_gaussian_count
        );

        settings.quality = 1.0;
        let fine = select_frontier(&hierarchy, &AllResident, view(), &settings).unwrap();
        assert!(fine.nodes.iter().all(|node| {
            hierarchy
                .node(*node)
                .is_some_and(|manifest_node| manifest_node.is_leaf())
        }));
        assert_eq!(
            fine.status.active_gaussians,
            built.manifest.header.source_gaussian_count
        );
    }

    #[test]
    #[ignore = "manual stable-cut selector throughput benchmark"]
    fn benchmark_large_stable_cut_selection() {
        const ITERATIONS: u32 = 40;
        let hierarchy = DenseBenchmarkHierarchy::binary(14);
        let previous = hierarchy.leaves();
        let mut settings = GaussianLodSettings {
            quality: 0.65,
            hysteresis: 0.1,
            frustum_culling: false,
            ..Default::default()
        };
        settings.budgets.max_active_gaussians = 2_000_000;
        settings.budgets.max_traversal_nodes_per_view = 1_000_000;

        let started = Instant::now();
        for _ in 0..ITERATIONS {
            let selected = select_frontier_with_previous(
                black_box(&hierarchy),
                &AllResident,
                black_box(view()),
                black_box(&settings),
                black_box(&previous),
            )
            .unwrap();
            assert_eq!(selected.nodes, previous);
            black_box(selected);
        }
        let elapsed = started.elapsed();
        eprintln!(
            "large stable-cut selection: {ITERATIONS} iterations in {elapsed:?} ({:?}/selection)",
            elapsed / ITERATIONS,
        );
    }

    #[test]
    #[ignore = "manual compiled-manifest dense-versus-sparse throughput benchmark"]
    fn benchmark_compiled_manifest_moving_cut_selection() {
        const ITERATIONS: u32 = 20;
        let source = PlanarGaussian3d::from(
            LodTestScene::nested_octants(5)
                .gaussians
                .into_iter()
                .map(|entry| entry.gaussian)
                .collect::<Vec<_>>(),
        );
        let built = build_planar_3d_lod(
            &source,
            GaussianLodBuildSettings {
                leaf_capacity: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let dense = CompiledManifestLodHierarchy::new(built.manifest).unwrap();
        let sparse = force_sparse_compiled_index(dense.clone());
        let views = [
            view(),
            LodView::perspective(
                Vec3::new(0.25, 0.0, 0.0),
                1080.0,
                std::f32::consts::FRAC_PI_2,
                0.1,
            ),
            LodView::perspective(
                Vec3::new(0.5, 0.25, 0.0),
                1080.0,
                std::f32::consts::FRAC_PI_2,
                0.1,
            ),
            LodView::perspective(
                Vec3::new(0.25, 0.5, 0.0),
                1080.0,
                std::f32::consts::FRAC_PI_2,
                0.1,
            ),
        ];
        let mut settings = GaussianLodSettings {
            quality: 0.65,
            hysteresis: 0.1,
            frustum_culling: false,
            ..Default::default()
        };
        settings.budgets.max_active_gaussians = 2_000_000;
        settings.budgets.max_traversal_nodes_per_view = 1_000_000;

        let mut previous = Vec::new();
        let expected = views
            .iter()
            .map(|view| {
                let selected = select_frontier_with_previous(
                    &dense,
                    &AllResident,
                    *view,
                    &settings,
                    &previous,
                )
                .unwrap();
                previous.clone_from(&selected.nodes);
                selected
            })
            .collect::<Vec<_>>();
        let selections = ITERATIONS * u32::try_from(views.len()).unwrap();

        for (label, hierarchy) in [("dense", &dense), ("sparse", &sparse)] {
            let started = Instant::now();
            for _ in 0..ITERATIONS {
                previous.clear();
                for (view, expected) in views.iter().zip(&expected) {
                    let selected = select_frontier_with_previous(
                        black_box(hierarchy),
                        &AllResident,
                        black_box(*view),
                        black_box(&settings),
                        black_box(&previous),
                    )
                    .unwrap();
                    assert_eq!(&selected, expected);
                    previous.clone_from(&selected.nodes);
                    black_box(selected);
                }
            }
            let elapsed = started.elapsed();
            eprintln!(
                "compiled-manifest {label} moving cut: {} nodes, {selections} selections in {elapsed:?} ({:?}/selection)",
                hierarchy.manifest.nodes.len(),
                elapsed / selections
            );
        }
    }
}
