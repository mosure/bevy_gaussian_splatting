# Global ordering for Gaussian quads

`GaussianGlobalOrderSettings` selects an experimental camera backend that
projects participating clouds into one bounded stream, sorts that stream by
full 32-bit forward camera depth (`-view-space Z`), and submits one alpha-over draw.
This handles interleaved clouds within the shared stream. The bounded image
fixture has software and NVIDIA Vulkan results from earlier renderer versions;
those results do not qualify the changed depth metric. Broader performance and
hardware qualification remain open.

## Supported profile

The initial profile uses planar 3D storage-buffer clouds, color rasterization,
non-additive blending, `Msaa::Off`, no resolution override, and no LoD debug or
bounding-box visualization. Participating clouds use `SortMode::Radix` or
`SortMode::None`. Flat sources, complete discrete CPU-selected LoD cuts, and GPU hierarchy
snapshots are supported. AABB/OBB support and selected/highlighted draw modes retain their
existing cloud settings.

CPU candidate morphs, external active sets, additive clouds and other raster modes remain
outside its shared order. The viewer rejects simultaneous global ordering and
GPS, and selects discrete LoD presentation without changing authored quality.
`--global-order --lod-gpu-traversal --input-lod path/to/scene.gsplatlod` selects
GPU traversal with ordered quads. Applications should avoid composing independently transparent
backends when a single global transparency order is required.

## Use

The default viewer profile includes the backend:

```sh
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-cloud assets/scene.ply --global-order \
  --global-order-max-gaussians 1048576 \
  --global-order-max-gpu-bytes 268435456
```

Use an existing source path, or `--input-lod path/to/scene.gsplatlod` for a
package with CPU-selected discrete cuts; add `--lod-gpu-traversal` for GPU
selection. Applications can attach
`GaussianGlobalOrderSettings::default()` and `Msaa::Off` to a camera containing
`GaussianCamera`; `GaussianSplattingPlugin` installs the backend. The settings
and `GaussianGlobalOrderDiagnostics` are exported from `render::ordered`.

## Current-camera spatial transitions

For authored ABI16/17 packages, the optional camera component
`render::spatial_morph::GaussianLodSpatialTransitionSettings` enables bounded
adjacent parent/child transitions with GPU traversal and ordered quads. The
package uses `LodPresentationMode::ContinuousMorph`. The defaults admit at most
256 fractional edges and 65,536 participating child records, with 32 MiB of
immutable correspondence. These limits supplement existing traversal, residency
and renderer memory limits. GPS does not yet implement this representation.

Quality policy determines which splits are eligible. A separate finite projected
size/error estimate orders their fixed refinement costs; child scores are at most
half their parent's score. The first excluded score defines the current-camera
budget cutoff. Strictly admitted edges in its open factor-two band interpolate;
ties retain actual parents and an unconstrained cut retains actual children.
Inside the band, the complete child cohort interpolates position and covariance and
uses the existing projected-area-corrected optical-depth and linear-color
mixture. Parent pages join the same-submission residency receipt. A conservative
envelope encloses interpolated means and convex covariance, including authored
support-sigma and bounds tolerance. A near-plane intersection bypasses the whole
edge, retaining the discrete child cut. Dynamic Gaussian scale magnitudes above
one also retain discrete drawing because they enlarge the authored envelope.

Missing correspondence, missing parents, optional pipeline compilation and
transition admission failures retain a complete discrete image. An active band
that exceeds either transition limit falls back as a whole for that source;
no nonzero-weight suffix is silently discarded. No elapsed-time
blend or acknowledgement history modifies the weight. Children arriving after
the camera has crossed their band therefore make a categorical handoff; this
path does not claim to hide late loading or repair inaccurate representatives.
This finite budget coordinate remains defined when quality 0.95 requires
uncertified proxies to refine. It does not certify those proxies' image quality.
Continuity requires an unchanged eligible graph and successful full-band
admission; visibility/eligibility boundaries, missing pages and safety bypasses
remain explicit categorical cases.

Ordinary projected records remain 64 bytes; the optional spatial variant uses 96
bytes and rechecks device limits before admission. Run correspondence is four
bytes per authored parent run plus 16 bytes per node and a 32-byte header. Per-view
range metadata is 32 bytes per frontier slot rather than per Gaussian. Diagnostics
report actual and required band records/edges, with separate flags for invalid
pressure, near-plane bypass, missing parents, full-band capacity failure and
unavailable exact selection. To remove the transition limits as an independent
boundary, provision them for the admitted frontier and active record ceiling.
They do not substitute
for measured image quality. The new spatial path still requires the bounded GPU
endpoint/adjacent-camera fixture and scene qualification.

## Invariants and limits

- Each source is bound for projection separately; sorting and drawing consume
  the shared projected stream without an unbounded source-binding array.
- Stable gathering and radix sorting preserve a deterministic tie order based
  on cloud entity and each source's stable record order. Reordering the entity query does not change
  that order.
- Projected-record and GPU-byte limits apply to the entire view. Owned
  projection, prefix, sort, input and feedback buffers enter the shared memory
  ledger; replaced allocations remain charged through their GPU fences.
- Eligible clouds suppress their separate per-cloud draw and radix workspace.
  Admission failure reports an error and suspends the shared draw instead of
  launching an unbounded fallback. Existing flat identity-index storage remains
  a compatibility cost.
- Cold package preparation only admits resources. A discrete cut is
  acknowledged from completed draw feedback carrying its physical candidate
  identity; pipeline readiness alone cannot activate it. GPU snapshots use
  renderer-neutral receipts containing source identity, residency generation
  and completed submission. A failed traversal suppresses the shared image
  and its receipts.

The order uses forward camera depth, with center-depth testing against opaque
geometry. Parallel lateral camera motion therefore preserves the order of
distinct center depths. Equal depths retain the existing deterministic tie
order. A scalar Gaussian order does not
provide exact ray ordering for extended overlapping ellipsoids, and does not
order other transparent renderer passes. The GPS sampling/view-time controller
does not govern this quad backend. No speedup, representative-quality pass,
browser parity or large-scene qualification is claimed here.

## Qualification

The following recorded runs used squared radial camera distance. They remain
historical evidence; their images and timings have not been recomputed or promoted
to qualification of forward-depth ordering.

On 2026-09-08, the [single render fixture](../tests/global_order.rs) passed on
llvmpipe Vulkan in 4.56 seconds, including the cold-package extension. It projects 304 admitted records across multiple
prefix groups, renders four visible interleaved splats from two clouds, and
matches the merged-cloud 32-bit radix reference within two 8-bit quantization
levels. It also checks shared-capacity rejection without duplicate per-cloud draws,
recovery, and zero drawn records after the final source disappears. A four-record
native CPU package then loads authenticated pages, activates its current discrete
candidate against the same atlas identity, and produces a nonempty image; camera
movement requires twelve fresh completed draw submissions and a changed image.
The focused
allocation, viewer configuration and per-cloud radix-demand checks pass.
[Command, log and binary hash](../target/lod-roadmap/2026-09-08/completion/global-order-package.json)
record this software correctness result. Broader projection variants and
hardware performance need separate measured coverage.

The 2026-09-08 closeout also passes on NVIDIA RTX PRO 6000 Blackwell Vulkan
(driver 610.43.02), including two GPU-streamed packages mixed with one flat
record. It requires coarse and exact generation-fenced receipts from both
packages, nine exact projected records under the shared cap, and a black image
when the complete roots cannot fit.
[Current command and binary identity](../target/lod-roadmap/2026-09-08/production-closeout/gpu-global-order-final.json)
record the 4.99-second bounded execution; this is a correctness fixture, not a
scene-performance measurement.
