//! Deterministic CPU reference traversal for a Gaussian LoD hierarchy.
//!
//! The renderer is expected to execute the same policy on the GPU. Keeping this
//! allocation-light oracle independent of a concrete manifest gives tests and
//! offline tools a precise source of truth for endpoint, budget, and residency
//! behavior.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    fmt::Debug,
    hash::Hash,
    sync::Arc,
};

use bevy::math::{Mat4, Vec3, Vec4};

use crate::{
    gaussian::{
        formats::{
            planar_3d_chunked::{LodNodeId, LodPageId, LodPageRange},
            planar_3d_lod::GaussianLodManifest,
        },
        lod_settings::{
            GaussianLodSettings, LodDegradation, LodEffectiveStatus, LodQualityEndpoint,
            LodSettingsError,
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
    children: Vec<Vec<LodNodeId>>,
}

/// Owned, shareable form used by long-lived streaming/runtime state.
/// Topology is compiled once when the manifest is opened, so camera traversal
/// never rebuilds child vectors.
#[derive(Clone, Debug)]
pub struct CompiledManifestLodHierarchy {
    manifest: Arc<GaussianLodManifest>,
    node_indices: BTreeMap<LodNodeId, usize>,
    children: Vec<Vec<LodNodeId>>,
}

impl CompiledManifestLodHierarchy {
    pub fn new(manifest: GaussianLodManifest) -> Result<Self, ManifestHierarchyError> {
        let manifest = Arc::new(manifest);
        let borrowed = ManifestLodHierarchy::new(&manifest)?;
        let node_indices = borrowed.node_indices;
        let children = borrowed.children;
        Ok(Self {
            manifest: Arc::clone(&manifest),
            node_indices,
            children,
        })
    }

    pub fn manifest(&self) -> &GaussianLodManifest {
        &self.manifest
    }

    pub fn node(
        &self,
        node: LodNodeId,
    ) -> Option<&crate::gaussian::formats::planar_3d_lod::GaussianLodNode> {
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

impl LodHierarchy for CompiledManifestLodHierarchy {
    type NodeId = LodNodeId;

    fn roots(&self) -> &[Self::NodeId] {
        &self.manifest.roots
    }

    fn parent(&self, node: Self::NodeId) -> Option<Self::NodeId> {
        self.node(node).and_then(|node| node.parent)
    }

    fn children(&self, node: Self::NodeId) -> &[Self::NodeId] {
        self.node_indices
            .get(&node)
            .and_then(|index| self.children.get(*index))
            .map(Vec::as_slice)
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
        let mut children = Vec::with_capacity(manifest.nodes.len());
        for node in &manifest.nodes {
            let start = node.children.start as usize;
            let end = node
                .children
                .end()
                .ok_or(ManifestHierarchyError::ChildRangeOverflow)? as usize;
            let child_nodes = manifest
                .nodes
                .get(start..end)
                .ok_or(ManifestHierarchyError::ChildRangeOutOfBounds(node.id))?;
            children.push(child_nodes.iter().map(|child| child.id).collect());
        }
        Ok(Self {
            manifest,
            node_indices,
            children,
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
        self.node_indices
            .get(&node)
            .and_then(|index| self.children.get(*index))
            .map(Vec::as_slice)
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

    fn projected_node(self, metrics: LodNodeMetrics) -> LodProjectedNode {
        let transform_scale = self.world_scale_upper_bound();
        let center = self.world_from_local.transform_point3(metrics.center);
        let radius = metrics.radius.max(0.0) * transform_scale;
        let projection_scale_px_per_world = match self.projection {
            LodViewProjection::Perspective {
                vertical_fov_radians,
            } => {
                let focal_length_px =
                    0.5 * self.viewport_height_px / (0.5 * vertical_fov_radians).tan();
                let distance_to_surface =
                    (self.camera_position.distance(center) - radius).max(self.near_plane);
                focal_length_px / distance_to_surface
            }
            LodViewProjection::Orthographic {
                vertical_world_size,
            } => self.viewport_height_px / vertical_world_size,
        };
        let support_radius_px = radius * projection_scale_px_per_world;
        LodProjectedNode {
            error_px: metrics.geometric_error * transform_scale * projection_scale_px_per_world,
            support_radius_px,
            coverage: (2.0 * support_radius_px / self.viewport_height_px).clamp(0.0, 1.0),
        }
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

#[derive(Clone, Debug, PartialEq)]
pub struct LodFrontier<NodeId> {
    /// A stable node-id-ordered complete cut.
    pub nodes: Vec<NodeId>,
    /// Missing desired nodes, deduplicated and ordered for deterministic requests.
    pub requested_nodes: Vec<NodeId>,
    pub status: LodEffectiveStatus,
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
    accepted: BTreeSet<NodeId>,
    refined: BTreeSet<NodeId>,
    traversal_nodes_visited: u32,
}

impl<NodeId> Default for PreviousCut<NodeId> {
    fn default() -> Self {
        Self {
            accepted: BTreeSet::new(),
            refined: BTreeSet::new(),
            traversal_nodes_visited: 0,
        }
    }
}

impl<NodeId: Copy + Ord> PreviousCut<NodeId> {
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
            let mut chain = BTreeSet::new();
            while remaining_work > 0 {
                let Some(parent) = hierarchy.parent(cursor) else {
                    break;
                };
                if !chain.insert(parent) {
                    return Err(LodSelectionError::HierarchyCycle(parent));
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

/// Selects a camera-aware hierarchy cut with a conservative caller-supplied
/// visibility predicate. Missing descendants never replace a resident ancestor.
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

/// Stateful camera-aware selector with caller-supplied conservative visibility.
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
    view.validate().map_err(|error| match error {
        LodSelectionError::InvalidView(field) => LodSelectionError::InvalidView(field),
        _ => unreachable!(),
    })?;

    let endpoint = settings.quality_endpoint();
    let traversal_limit = settings.budgets.max_traversal_nodes_per_view;
    let mut state = SelectionState::<H::NodeId>::new(previous.traversal_nodes_visited);

    for &root in hierarchy.roots() {
        state.visit(traversal_limit)?;
        let metrics = checked_metrics(hierarchy, root)?;
        if !visible(root, metrics) {
            continue;
        }
        if residency.is_resident(root) {
            state.insert_frontier(root, metrics)?;
            state.maybe_queue_candidate(hierarchy, view, settings, previous, root, metrics);
        } else {
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

            let mut visible_children = Vec::new();
            let mut missing_children = Vec::new();
            let mut child_count = 0_u64;
            for &child in hierarchy.children(candidate.node) {
                if state.visited_nodes >= traversal_limit {
                    state.degradation = state.degradation.merge(LodDegradation::TraversalBudget);
                    break;
                }
                let metrics = checked_metrics(hierarchy, child)?;
                state.visit(traversal_limit)?;
                if !visible(child, metrics) {
                    continue;
                }
                if residency.is_resident(child) {
                    child_count = child_count
                        .checked_add(u64::from(metrics.representative_count))
                        .ok_or(LodSelectionError::CountOverflow)?;
                    visible_children.push((child, metrics));
                } else {
                    missing_children.push(child);
                }
            }

            if state.degradation == LodDegradation::TraversalBudget
                || state.degradation == LodDegradation::Multiple
                    && state.visited_nodes >= traversal_limit
            {
                // An incomplete child enumeration cannot safely replace its ancestor.
                continue;
            }

            if !missing_children.is_empty() {
                state.requested.extend(missing_children.iter().copied());
                state.degradation = state.degradation.merge(LodDegradation::Residency);
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

            state.frontier.remove(&candidate.node);
            state.active_gaussians = next_count;
            for (child, metrics) in visible_children {
                state.frontier.insert(child, metrics.representative_count);
                state.maybe_queue_candidate(hierarchy, view, settings, previous, child, metrics);
            }
        }
    }

    if state.active_gaussians > settings.budgets.max_active_gaussians {
        state.degradation = state.degradation.merge(LodDegradation::ActiveBudget);
    }

    let mut achieved_max_error_px = 0.0_f32;
    let mut achieved_max_target_ratio = 0.0_f32;
    let requested_target = settings.quality_target();
    for &node in state.frontier.keys() {
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

    Ok(LodFrontier {
        nodes: state.frontier.into_keys().collect(),
        requested_nodes: state.requested.into_iter().collect(),
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

struct SelectionState<NodeId: Copy + Ord> {
    frontier: BTreeMap<NodeId, u32>,
    candidates: BinaryHeap<Candidate<NodeId>>,
    requested: BTreeSet<NodeId>,
    expanded: BTreeSet<NodeId>,
    active_gaussians: u64,
    visited_nodes: u32,
    degradation: LodDegradation,
}

impl<NodeId: Copy + Ord> SelectionState<NodeId> {
    fn new(visited_nodes: u32) -> Self {
        Self {
            frontier: BTreeMap::new(),
            candidates: BinaryHeap::new(),
            requested: BTreeSet::new(),
            expanded: BTreeSet::new(),
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
        Ok(())
    }

    fn maybe_queue_candidate<H: LodHierarchy<NodeId = NodeId>>(
        &mut self,
        hierarchy: &H,
        view: LodView,
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
            LodQualityEndpoint::Continuous => pressure > 1.0,
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
    fn visibility_filters_whole_branches() {
        let hierarchy = hierarchy();
        let mut settings = GaussianLodSettings::default();
        settings.quality = 1.0;
        settings.budgets.max_active_gaussians = 100;
        let selected = select_frontier_with_visibility(
            &hierarchy,
            &AllResident,
            view(),
            &settings,
            |node, _| !matches!(node, 2 | 5 | 6),
        )
        .unwrap();
        assert_eq!(selected.nodes, vec![3, 4]);
        assert_eq!(selected.status.active_gaussians, 8);
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
        assert_eq!(rows[5], [16, 16, 16]);
        assert_eq!(rows[6], [16, 16, 16]);
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
        assert_eq!(cut.accepted, BTreeSet::from([3]));
        assert_eq!(cut.refined, BTreeSet::from([1]));
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
        settings.quality = 0.5;
        settings.hysteresis = 0.0;
        settings.budgets.max_active_gaussians = 100;

        let base = view();
        let scaled = view().with_world_from_local(Mat4::from_scale(Vec3::splat(10.0)));
        let base_selection = select_frontier(&hierarchy, &AllResident, base, &settings).unwrap();
        let scaled_selection =
            select_frontier(&hierarchy, &AllResident, scaled, &settings).unwrap();

        assert_eq!(base_selection.nodes, vec![3, 4, 5, 6]);
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
        let mut settings = GaussianLodSettings::default();
        settings.budgets.max_active_gaussians = 1_000;

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
}
