# LODGE active-set LoD

The crate supports two deliberately separate 3D Gaussian LoD representations:

| Strategy | Artifact | Selection unit | Transition |
| --- | --- | --- | --- |
| Finest-source hierarchy | `.gsplatlod` | A complete Morton/MomentMerge frontier selected from projected error and coverage | ABI 16 immediate parent/child optical-depth morph, or a categorical fallback |
| External LODGE active sets | `.gslodge` plus its bound `.gsplatlod` and Gaussian pages | The deduplicated union of the two nearest authored camera-cluster sets | Opacity scaling on the sets' symmetric difference |

The second strategy implements the presentation method in
[LODGE](https://arxiv.org/abs/2505.23158) without pretending that independently
trained Gaussian levels or overlapping camera-cluster memberships form a
source-partition tree. The existing hierarchy format, selector, and public
components remain source- and byte-compatible.

## Presentation contract

An external producer supplies independently trained levels, stable Gaussian
record IDs, camera-cluster centers, and one sorted active-set membership per
cluster. Runtime selection is deterministic:

1. Select the two nearest cluster centers, with the cluster ID as the distance
   tie-breaker. This is the default; a positive `pair_hysteresis` is an
   explicit, non-paper extension for applications which prefer fewer
   second-neighbor swaps.
2. Keep the pair identity in canonical ID order, even when nearest rank swaps.
3. Merge the two sorted memberships once, emitting every shared Gaussian once
   and classifying the rest as `FirstOnly` or `SecondOnly`.
4. Project the camera center onto the line between the two cluster centers and
   clamp the scalar coefficient to `[0, 1]`.
5. Draw one globally depth-sorted union. Shared records retain authored
   opacity; first-only records use `1 - t`; second-only records use `t`.

Only peak opacity is scaled. Position, covariance, spherical-harmonic color,
visibility, support, and the fixed three-sigma LoD cutoff remain authored. The
union is compacted and radix-sorted once; rendering the two sets as separate
clouds would duplicate their intersection and produce a different alpha order.

Camera-only motion inside a stable pair updates a small per-view presentation
header. It does not rebuild the classified ranges, compaction, or radix output.
`Frozen` selection captures both the pair and its coefficient. A pair change is
an atomic candidate replacement: the last complete union remains drawable
until the replacement is complete and render-proven.

## `.gslodge` v1

`.gslodge` is an authenticated companion, not a replacement for `.gsplatlod`.
Its fixed 120-byte outer header gates allocation-driving counts before JSON or
Flexbuffers decoding and authenticates the payload plus the exact base manifest
identity with SHA-256. The semantic payload records:

- discrete level thresholds and depth-aware filter/prune/fine-tune metadata;
- dense stable Gaussian ID runs mapped to ordinary Gaussian pages;
- camera-cluster centers, radii, neighbor lists, and membership descriptors;
- exact SHA-256 identities for the base manifest, every page, the membership
  object/index, and each independently fetched membership stream.

The normative byte layout, Serde representation, URI-resolution roots, hash
domains, membership codec, and compatibility rules are specified in the
[LODGE active-set sidecar format](lodge_active_set_format.md).

Membership streams are strictly increasing stable IDs encoded as canonical
delta unsigned LEB128. Decoding rejects zero or overlong deltas, duplicates,
unknown IDs, count/endpoint mismatches, trailing bytes, and non-canonical
encodings. Page bytes are SHA-256 checked before the existing page descriptor,
decoded FNV/content, bounds, and Gaussian-value checks run.

Native `lod_build` users can author the membership dependency through
`build_canonical_lodge_membership_artifact`. Its bounded, replayable input API
assigns the crate-owned `BGSLMEM` header/directory, rejects replay drift, and
returns a manifest-ready `LodgeMembershipIndexDescriptor`. Gaussian level-page
authoring remains explicit: upstream LODGE publishes no canonical checkpoint or
portable output format for this crate to parse, so an exporter must normalize
its trained Gaussian levels into ordinary authenticated page descriptors and
stable record runs.

Hashes provide dependency integrity relative to the loaded sidecar. They do
not identify a publisher; an application still chooses which local path,
origin, signature, or pinned sidecar hash it trusts.

## Instantiation

The public surface is additive:

- `GaussianLodgeAsset` and `GaussianLodgeHandle` load `.gslodge` manifests;
- `GaussianLodgeSettings` owns active-set selection/residency policy without
  inheriting the hierarchy quality slider or claiming an `Original` endpoint;
- the LODGE resident-catalog component binds an authenticated stable-ID catalog
  and cluster memberships to an ordinary `PlanarGaussian3d` GPU asset;
- `GaussianLodgeStatus` reports the selected pair, coefficient, classified
  counts, required residency, stale retention, render satisfaction, and a
  typed failure code;
- `GaussianLodRepresentationKind`/`GaussianLodStrategy` lets generic UI code
  distinguish `FinestHierarchy` from `LodgeActiveSets`.

Use the authenticated materializer when constructing the resident catalog from
external bytes. Arbitrary in-memory membership lists are intentionally not
promoted to a renderable package without an explicit validation proof. The
materializer resolves stable IDs in run order, validates every page slice and
catalog count, and keeps one class per coalesced physical range.

The current ECS integration is deliberately fully resident. The
`GaussianLodgeManifestLoader` loads and validates only the sidecar; the
application resolves its authenticated dependency closure before attaching a
cloud:

1. Call `GaussianLodgeResidentCatalog::validate_manifest_budget` before any
   page allocation. Pass the exact manifest `Arc` returned by
   `GaussianLodgeAsset::shared_manifest()` through the remaining steps; this
   binds the materialized catalog to that loaded sidecar asset identity.
2. Fetch `base_manifest.uri`, then construct an
   `AuthenticatedLodgeBaseManifest` from its exact bytes.
3. Fetch every declared base/extra page and decode it through
   `AuthenticatedLodgePage` (SHA-256 first, ordinary page validation second).
4. Fetch and authenticate the complete membership object and index through
   `AuthenticatedLodgeMembershipObject`, then decode each cluster entry from
   that proof.
5. Call `GaussianLodgeResidentCatalog::from_authenticated_pages`, passing the
   same settings used by the entity, and attach the result with the sidecar
   handle:

```rust,ignore
commands.spawn((
    GaussianLodgeHandle(lodge_asset_handle),
    resident_catalog,
    GaussianLodgeSettings::default(),
));
```

`GaussianSplattingPlugin` supplies the required cloud/transform/visibility and
resident render systems when `lod_render_path_is_supported()` is true. The
normal `lod_render` and WebGPU `web` bundles satisfy that storage-buffer radix
contract; `buffer_texture` and WebGL2 builds retain the codec and CPU planner
but cannot instantiate the resident renderer. Attaching both
`GaussianLodgeHandle` and `GaussianLodHandle` is rejected; the two strategies
never race for one cloud.

The generated `PlanarGaussian3d` catalog is an authenticated, immutable
stable-ID mapping. Modifying or removing that asset invalidates the cloud and
requires materializing a fresh catalog; a same-length replacement is not
accepted as equivalent.

## Lifecycle and failure policy

Cold start remains non-drawable until one complete pair is available. A
replacement pair is never published from a partial membership or page set.
Invalid hashes, malformed memberships, non-finite/coincident centers, source
index overflow, unsupported render configurations, and candidate/header generation
mismatches fail closed. An existing complete pair is retained and reported as
stale/degraded whenever possible.

LODGE status transitions use the same public `LodOrchestrationTransition`
message as hierarchy packages, with `source == ExternalActiveSets`. Repeated
identical failures do not churn revisions or messages; recovery is emitted only
after an exact replacement becomes active again.

Per-view pair selection and coefficients are independent; decoded pages and
GPU assets may be shared. The presentation buffer's mode is mutually exclusive
with ABI 16 hierarchy morphing, and the two-bit entry class is interpreted only
under that mode.

This first integration materializes the complete stable-ID catalog and all
cluster memberships before drawing. The format, page-demand planner, retained
candidate state, and transport-authentication primitives are suitable for a
future out-of-core orchestrator, but the `.gslodge` asset loader does not yet
perform LODGE's background chunk reload automatically. The resident path still
reduces submitted/rasterized work to the selected pair union; it does not yet
reduce CPU catalog residency.

## Method boundary

LODGE's two-nearest-set method cannot guarantee continuity when the identity of
the second-nearest cluster changes at a nonzero exclusive weight. The runtime
retains a complete prior pair while a replacement is prepared, but it does not
invent nested weights or silently reinterpret the imported memberships.
Producers should provide spatially coherent neighboring clusters and active
sets, and applications should treat a delayed nonzero neighbor swap as explicit
stale/degraded presentation.

`.gslodge` is this crate's authenticated adapter format for LODGE-style output,
not a file format defined by the paper or its reference implementation. An
offline exporter must provide the levels, stable IDs, cluster centers, and
memberships described above.

The paper's reported mobile FPS is not a transferable benchmark for this
renderer. It uses a different web renderer and reports rasterization separately
from asynchronous sorting and chunk reload. This crate's real GPU/browser
qualification must be run on the target adapter and artifact.
