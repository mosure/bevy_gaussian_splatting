# Gaussian level of detail

The LoD system selects a camera-aware, globally covering hierarchy cut, streams
or materializes the required pages into a bounded atlas, compacts visible
records on the GPU, and renders them through the ordinary Gaussian raster path.

The promoted product surface is deliberately small:

- `GaussianLodSettings::quality` controls detail from `0` (coarsest) to `1`
  (exact original).
- `GaussianLodSettings::selection_mode` is `Dynamic` or `Frozen`.
- `LodDebugPreset` selects one of the documented debug views.
- `GaussianLodStatus` reports lifecycle, selected records, quality pressure,
  residency, and typed failure state.

These controls describe the crate's finest-source MomentMerge hierarchy.
Externally trained discrete levels and camera-cluster active sets use the
separate `GaussianLodgeSettings`/`GaussianLodgeStatus` surface and authenticated
`.gslodge` companion format; see the [LODGE guide](lodge.md). The two strategies
share Gaussian storage, compaction, radix, and raster infrastructure, but not a
selector or transition ABI.

Budgets, transport limits, and selector hysteresis remain programmatic safety
controls. They are not additional quality sliders. The reusable library keeps
its temporal hysteresis default, while the standalone viewer deliberately uses
zero hysteresis so an identical camera and quality always reproduce one
canonical logical cut.

## Runtime paths

### Resident flat cloud

Attaching `GaussianLodSettings` to a `PlanarGaussian3dHandle` enables the
automatic bridge. For an interior quality it builds a bounded progressive
MomentMerge hierarchy and publishes complete cuts through the same atlas and
render handshake used by packages. During cold initialization, the already
loaded source remains drawable until the requested cut reaches a drained,
stable publication point. Residency-degraded intermediate cuts are never
presented page by page. Interior-quality rendering then remains on bounded
atlas cuts: camera motion, residency pressure, and active-count pressure never
promote the whole source as a fallback. Quality `1` intentionally stays on the
exact flat path and allocates no LoD compaction state. Debug presets keep the
atlas path so annotations still match hierarchy records.

This path is convenient for ordinary PLY, gcloud, and glTF/GLB assets. On
native targets, the bridge copies one globally bounded source batch per
application update, keeps drawing the resident flat cloud, then builds and
encodes the hierarchy on a bounded worker pool. Only one large transient build
is admitted at a time; source or structural changes invalidate stale results
before publication. Atlas capacity is derived from the configured resident
page, record, and byte budgets rather than the source count. Guard pages stream
first and remain reserved for recovery, but the normal cold handoff waits for
one quiescent target or terminal cut rather than displaying progressive coarse
ancestors. If cold streaming saturates the configured resident capacity with
unresolved demand, the bridge may present a resident guard only when that guard
actually meets the requested screen-space target. A correctness-only but
visually useless coarse guard never replaces a valid source or ACTIVE cut. The
flat source can therefore remain active when a transient target is impossible;
production-scale scenes should use a prebuilt package with a deliberately
authored bounded bootstrap tier.

Transient construction still starts from an already resident flat source and
stores the generated hierarchy in memory, so its build storage and work remain
proportional to source size. Use a prebuilt package when the source itself must
be out of core or is too large for that one-time construction footprint.

WebAssembly has no background CPU worker guarantee. To keep the browser event
loop responsive, automatic synchronous hierarchy construction is therefore
limited to 1,024 source Gaussians. Use a prebuilt package for larger Web scenes.

### Prebuilt package

An entity with `GaussianLodHandle`, `GaussianLodPackageSource`, and
`GaussianLodSettings` streams independently addressable pages from standalone
`.gspage` objects or range-packed `.bgslodpack` shards. Package startup reserves
the bounded GPU address space but keeps CPU storage sparse: only materialized
slots in the bounded runtime cache own planar payloads, including reusable warm
slots retained between cuts. A package does not allocate or zero a source-sized
CPU cloud before its first useful page.
Native directories and immutable HTTP(S) roots share the same bounded runtime.
For bounded-refinement MomentMerge packages (builder/reducer pairs `(14, 3)`,
`(15, 3)`, or `(16, 4)`), cold startup first requests one deterministic
whole-level bootstrap antichain capped at 8 pages, 8,192 active records, and
2 MiB each of encoded, decoded, and GPU staging payload. It is published only
after the whole antichain is generation-current, then retained unchanged until
the camera target reaches a quiescent fixed point. Legacy
external ABI 5/6 packages do not qualify for this presentation path. Ordinary
resident ancestor waves are never exposed. When the bootstrap, root recovery,
and all-resident camera target can coexist for one atomic handoff, a stationary
package therefore produces at most the bootstrap and final target cuts. If that
handoff footprint exceeds the configured cache, the package keeps the useful
bootstrap ACTIVE, reports a capacity degradation, and suspends unchanged detail
demand instead of repeatedly downloading, evicting, and publishing page waves.
Camera, policy, or budget changes re-evaluate that admission proof.
Persistent caching is opt-in through `GaussianStreamingSettings` and requires
an explicit namespace; native caching also requires an explicit cache root.

One `.gsplatlod` manifest describes one Gaussian cloud. A composed scene keeps
its entity transforms, cameras, and non-Gaussian content in the ordinary scene
format and references one package per Gaussian primitive; `.gsplatlod` is not a
replacement for a multi-entity scene container.

The viewer exposes both paths:

```bash
# Resident scene with a transient hierarchy
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-scene=https://mitchell.mosure.me/trellis.glb \
  --lod-quality=0.75

# Prebuilt out-of-core package; page URIs resolve beside the manifest
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-lod=file:///data/city/scene.gsplatlod \
  --lod-quality=0.75
```

The viewer allows up to its 8,000,000-record resident capacity per active LoD
cut by default, while the reusable library policy remains conservatively capped
at 2,000,000. Use `--lod-max-active-gaussians=<count>` to trade GPU compaction,
sorting, and render work for finer spatial fidelity. The viewer clamps this
startup setting to that fixed resident capacity, and rejects zero rather than
treating it as unbounded. For example, an explicit 4,000,000-record ceiling is
a lower-cost override rather than the default:

```bash
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-lod=file:///data/garden/scene.gsplatlod \
  --lod-quality=0.65 --lod-max-active-gaussians=4000000
```

Standalone package loading uses up to 64 concurrent page transport requests by
default; the reusable `GaussianStreamingSettings` default remains 8. Use
`--lod-max-concurrent-requests=<1..256>` to tune that transport concurrency,
and lower it when an HTTP origin or intermediary enforces a smaller practical
connection/request limit.

In the viewer, a relative `--input-lod` path is relative to Bevy's configured
`assets/` root; page shards resolve beside that physical manifest. On the Web,
the same root is fetched relative to the document URL. Absolute HTTP(S)
manifests use the hardened URL subset in the
[format contract](lod_format.md); signed query URLs and percent-escaped package
paths are intentionally rejected today.

For a deterministic local smoke scene:

```bash
cargo run --release --bin bevy_gaussian_splatting -- \
  --gaussian-count=65536 --gaussian-seed=42 --lod-quality=0.65
```

The editor and the compact Gaussian LoD panel are enabled by default. Press
`F` to freeze the current cut, then move the camera closer to inspect it.

## Quality contract

The slider is screen-space and scene-scale independent. Perspective projection
already supplies distance response, so there is no second world-distance curve.
For interior quality `q`, projected coverage `p`, node threshold `t`, projected
error `e`, and the fixed nominal target `L(q) = 16 * 64^-q`:

```text
f = smoothstep(.90, .99, q)
S = q * (p + (1 - p) * f) / t
E = e / L(q)
a = min(q / .99, 1)^3
P_error = max(min(S, E), a * E)
```

A builder-authored high-fidelity certificate `c` adds a morphology guard:

```text
n = min(q / .95, 1)
g = smoothstep(.90, .95, q)
D = g * q * n * (p + (1 - p) * n^3)
P_certificate = D / c
P = max(P_error, P_certificate)
```

Pressure `P <= 1` meets the requested target. Certificate pressure is exactly
zero through quality `.90`, then reaches its full morphology guard at `.95`.
Quality `.95` is intentionally a strong safety gate and may select exact leaves
when no safe approximation exists. Quality `1` is categorical exact-original,
not a zero-error approximation. The mapping is monotonic in requested detail,
but hierarchy cuts are discrete and PSNR is not expected to be linear.

The pinned Trellis graph, near/mid/far PSNR table, continuity gates, and exact
selector equations live in [the canonical quality report](lod_quality_report.md).

## Raster filtering and support

Imported 3D Gaussians use the renderer's intended, corrected `+0.3 px²`
physical projected-covariance footprint, and the rasterizer scales peak alpha by
`sqrt(det(C) / det(C + 0.3 I))`. This determinant normalization preserves the
Gaussian's integrated screen-space alpha instead of brightening small splats as
the low-pass footprint widens them. WGSL uses two projection-coordinate units
per physical pixel, so its covariance is scaled by four and the equivalent
shader-space diagonal addition is `+1.2`. It is a Mip-Splatting-style post-hoc
screen filter for already-authored assets; it is not the paper's
training-frequency 3D filter and does not claim its training-time aliasing
guarantees.

Flat clouds keep the historical opacity-adaptive cutoff. Every LoD candidate
instead uses its fixed authored 3-sigma support for raster and compaction. This
also covers portable finite opacity values above one without introducing a
raster-only tail beyond the compaction sphere. A MomentMerge record can have a
low peak opacity while still representing substantial unmodeled mass, so
shrinking its quad from peak opacity alone would truncate valid support.

## Hierarchy and file contracts

The CPU builder creates deterministic Morton-ordered logical nodes and packs
same-depth, same-kind node slices into physical pages. Leaf payloads contain
original records. Interior payloads use risk-aware contiguous MomentMerge
representatives with conservative error, support, and fidelity metadata.
Progressive CPU builder ABI 14 calibrates representative opacity against a
conservative all-view projected alpha-mass bound so thin anisotropic surfaces
cannot become oversized bright representatives. External-memory ABI 15 remains
readable as the reducer-3 bounded multi-representative format. The current
external writer emits ABI 16 / MomentMerge v4: it retains ABI 15's
original-source accumulation and bounded amplification, adds bounded spatial
fitting, and serializes a compact monotone immediate-child-record to
parent-record map for temporal morphing. ABI 15 remains readable but is not
relabeled or mutated. Readable legacy external CPU/GPU ABIs 5/6 remain on
reducer v2 and do not claim the v3/v4 proof.

The v4 spatial fitter evaluates every authored-support-touching node pair
inside one same-depth future-parent cohort. The validated branching limit of 32
bounds this to 496 node-pair checks and a deterministic 3x3 tangential grid
bounds it to 4,464 seam probes per cohort. Probe references use the flat
renderer's opacity-adaptive cutoff; emitted representatives use the LoD
candidate's fixed authored 3-sigma cutoff. Each probe is evaluated over the
fixed projection-direction set and a normalized `0.0625x..4x` pixel-scale
ladder so the `0.3 px²` mip footprint cannot bless a fit at only one zoom.

Accepted edits keep representative positions fixed, widen tangent covariance
only inside the touching pair's authored support envelope, rerun the all-view
opacity ceiling, improve the target seam, and do not regress any affected
sampled seam or cohort composited error at any sampled scale. If a measured
seam cannot be repaired, v4 raises ordinary selection-visible geometric error
so a screen-visible unsafe representation refines at normal quality. A
source-less coarse pair is left unchanged and reported as unmeasured; it is not
assigned infinite error, which preserves useful distance response. These are
within-cohort guarantees: same-depth boundaries split across different future
parents and mixed-depth selected-cut boundaries are qualified by the Garden
image oracle rather than claimed to be jointly fit.

The manifest records:

- source, node, page, and stored-record counts;
- topology and per-node page slices;
- bounds, quality interval, error, and high-fidelity certificate;
- page encoding, length, checksum, and optional storage location;
- builder ABI, reducer version, and required format features;
- for ABI 16, morph-map schema 1 plus one validated positive `u16` run per
  parent-local record, ordered by manifest child and page-local record order.

Validation rejects cycles, invalid ranges, heterogeneous shared pages,
unsupported features/ABIs, non-finite metadata, count overflow, and hierarchy
amplification/storage violations before allocation.

The readable format axes are deliberately independent:

| Layer | Magic / version | Purpose |
| --- | --- | --- |
| manifest container | `BGSLODC`, v2 | bounded serialized manifest envelope |
| semantic manifest | `BGSLOD3`, v3 | topology, quality, page descriptors, ABI |
| page schema | v2 | decoded Gaussian page contract |
| page container | `BGSPAGE`, v1/v2 | independently verified encoded page |
| shard container | `BGSSHARD`, v1 | immutable range-addressable page pack |

The exact byte layouts, compatibility rules, and native/HTTP transport
requirements are specified in the [LoD package format](lod_format.md).

`F32Planar` is the historical semantic-manifest-v3 encoding identifier; its
payload is record-major canonical data, not a zero-copy GPU plane layout.
Readers must not reinterpret it as plane-major storage. Any future directly
uploadable layout needs a new encoding identifier with explicit endianness,
plane offsets, alignment, scalar widths, SH count, and content digest; semantic
manifest v3 remains readable and normalizes through the same validated runtime
model.

HTTP package hosting must support byte ranges with exact lengths and identity
content encoding, expose the relevant range/length/validator headers to browser
CORS, and provide an immutable strong `ETag`. The default package loader pins
that identity and fails closed if an object changes during a session.

Build a package with:

```bash
cargo run --release --no-default-features --features lod_build_sh3 \
  --bin build_lod -- \
  --input scene.ply --output out/city --gpu-preprocess
```

This writes the manifest to `out/city/scene.gsplatlod` and its page payloads
beside it. On Linux, Apple platforms, and Windows, publication uses the native
atomic no-replace rename operation. If any filesystem entry already occupies
the output name—or appears there while the build is in progress—the builder
returns `OutputExists` and leaves that entry untouched. It never replaces an
empty directory or dangling symbolic link.

The SH degree is an explicit package ABI choice. Use `lod_build_sh3` for the
canonical desktop PLY pipeline, or `lod_build_sh0` to produce packages for the
official `web` feature profile. A loader rejects a package whose SH layout does
not match the compiled renderer rather than silently truncating coefficients.
The builder profiles are mutually exclusive; `lod_build_sh0` must be paired
with `--no-default-features` because the crate's default renderer enables SH3.
An input PLY containing coefficients above the selected build profile is
rejected by default. Pass `--allow-sh-truncation` to opt into dropping those
higher-order `f_rest_*` coefficients; an SH0 package built this way is exact
only in the selected SH0 rendering ABI. Use SH3 when those coefficients must
be preserved.

The builder is replayable and bounded. Canonical Morton quantization and the
source-index total-order tie breaker are authored on the host for both external
preprocessors. The CPU path uses deterministic Rayon batches; the explicit GPU
path uploads those exact keys and sorts them without re-deriving Morton values
from adapter floating-point arithmetic. Publication bytes and topology
therefore remain stable at quantization boundaries. Library callers can wrap
an already loaded `PlanarGaussian3d` in `PlanarGaussianSource`, which
reconstructs only one configured batch at a time and works with either the CPU
or GPU external preprocessor. This is the conversion path for resident
`.gcloud`, glTF/GLB, and procedural assets without a second full-source clone.
The pure build plan rejects manifest byte limits below the current format's
guaranteed node/page-vector minimum before scanning the source. Because
Flexbuffers metadata and shard URI widths are data-dependent, the exact
`--max-manifest-bytes` limit is still checked after hierarchy construction and
before atomic publication; it is an output limit, not a promise that the final
serializer never temporarily reaches that size.

`--gpu-preprocess` explicitly enables bounded GPU canonical preprocessing and
run sorting; parsing, external merge, the ABI 16 MomentMerge v4 hierarchy and
morph map, encoding, verification, and atomic publication remain on the CPU.
GPU setup is never implicit. The old `--gpu-hierarchy` spelling remains a CLI
alias, but it no longer selects the visually unsafe legacy GPU v2 hierarchy
reducer. A future GPU hierarchy path must match the CPU v4 topology, opacity,
spatial-fit, morph, certificate, and render-quality oracles before it can
replace this conservative stage.

## Selection, residency, and commit

Selection retains every root for global coverage, then uses the camera
projection to decide which visible subtrees merit refinement. A parent is
replaced only when every child in that substitution is resident, including
off-screen siblings, so every published frontier remains a complete global
antichain. GPU compaction applies the live frustum to that global cut before
drawing. Missing descendants retain the nearest resident ancestor; the
renderer never publishes holes as a successful cut. Active, traversal,
resident-page, resident-byte, request, preprocessing, and upload budgets fail
closed to the previous complete cut or a complete ancestor fallback.

Streaming may produce many complete ancestor cuts internally. Completeness is
a spatial safety invariant, not a presentation signal: those partial cuts stay
hidden while requests are queued, in flight, preprocessing, or capacity
blocked. The bridge performs one atomic replacement only after the target is
drained, or after a bounded terminal cut has reached a stable fixed point.
For a categorical legacy package, one bounded authored substitution is not
itself that fixed point; an unchanged request continues through as many
complete cohorts as required and ownership follows a stable no-transition
update. ABI 16 ownership instead follows the coherent drawable edge table: a
stable fractional presentation may own the stationary request when every
expected consumer is published and no edge has recovery lag or invalid
pressure. A quiescent rendered target has no missing consumers or requested
pages, reports the target satisfied without degradation, and agrees with the
public count and quality status.

Selector hysteresis suppresses small cut oscillations. For readable packages
without the ABI 16 map, a categorical topology demand must persist for two
consecutive selection frames before admission. Each legacy categorical frame
admits at most 256 authored substitutions and targets changed-record work of
`min(ceil(active_records / 24), 256 * 1024 records)`. One indivisible hierarchy
cohort may overshoot that record budget so the parent/children substitution
remains atomic.

ABI 16 presentation has no serialized cohort clock. Its immutable adjacent
parent/children edges persist across compatible cut replacements, and every
retained render view owns independent displayed and desired weights for those
edges. Multiple edges can therefore overlap and respond independently. In
`Dynamic` mode, each view recomputes desired weights from its current camera
and detail target; an ordinary resident edge follows that value exactly in the
next radix-proven drawable publication, including after an abrupt pose change.
A newly prefetched edge first draws its authored retained endpoint exactly.
Bounded `1/12`-per-drawable-update slew is reserved for exceptional continuity
recovery: late-residency activation, resuming `Dynamic` after `Frozen`, or the
first valid evaluations after an invalid-pressure hold. `Frozen` retains the
last drawable weights. Invalid pressure also retains them, reports explicit
degradation, and cannot satisfy the live target. A stable fractional edge
table can itself be the stationary view-conditioned fixed point; it need not
run to an endpoint merely because frames elapsed. Every presentation range set
remains a bounded, globally covering antichain. Quality zero and one remain
categorical endpoints.

For a direct ABI 16 parent/children edge, the runtime expands the validated
monotone runs into a bounded destination-cardinality GPU lookup. Refinement
draws the target child records; coarsening temporarily retains the old child
records until their split-parent endpoint is exact. Positions and covariance
interpolate between mapped endpoints. Each endpoint's SH radiance is evaluated
independently with its own view ray and color-space conversion, then combined
in linear light by the same optical-depth terms used for fragment alpha. The
parent's optical depth is divided across its mapped child run. Thus both
filtered endpoints are exact without drawing an unfiltered parent/child union.

Each retained view compacts and radix-sorts its private presentation. Render
`Cleanup`, after the ordered view graph, reduces every expected consumer into
one coherent aggregate snapshot and only then permits a prepared morph-capable
candidate to become ACTIVE. An ACTIVE table with a missing consumer remains an
explicit degraded hold with conservative fractional retirement evidence; a
prepared table simply waits. Neither state claims target satisfaction.

Morphing is presentation capability, not a correctness prerequisite. A
complete categorical `BoundedHardCohort` handoff is used for readable pre-ABI16
packages, a transaction without a usable direct map, a morph payload above the
adapter's buffer or storage-binding limit, or a package-authored sticky capacity
downgrade when the exact target fits the atlas but the temporary morph union
does not. Render code must not upgrade that capacity decision. Multi-subview
rendering is not by itself a hard-fallback condition; it uses the all-consumer
`Cleanup` barrier above. If exact topology or predecessor evidence becomes
stale, activation fails closed and the package is replanned; an incomplete
aggregate remains the explicit hold described above. If even the exact target
cannot fit, the runtime retains the prior valid cut and reports capacity
degradation; it does not present an incomplete morph or page wave.

Each runtime permanently pins a bounded, globally covering emergency guard.
It uses the deepest complete hierarchy level that still occupies no more atlas
slots than the root guard and remains within the configured byte, Gaussian, and
active-count budgets; it is not normally the single scene-wide root
representative. A dynamic camera-view change may make the previous cut's
quality stale, but not its spatial coverage, so that cut remains drawable while
a replacement streams. The guard is reserved for recovery when no better
draw capability exists; its view-local quality status reports degradation
instead of claiming to meet the requested target. It replaces a valid flat
source or ACTIVE cut only when it satisfies the requested target. Camera,
policy, coarsening, or residency changes invalidate cached selection results
without making a stale global cut spatially incomplete.

The package-only presentation bootstrap is different from this permanent
emergency guard. Its runtime-owned startup pins are released after the first
ACTIVE package transaction acquires independent page leases. The bootstrap
remains drawable through those package leases until the final atomic handoff,
or indefinitely with an explicit capacity-degraded status when that handoff
cannot fit.

When current and replacement leases create cache pressure, the bridge first
looks for a stable, complete resident cut that will release at least one
old-only atlas slot. That relieving cut follows the same generation-checked
PREPARED-to-ACTIVE handoff as an ordinary replacement. A presentation-qualified
guard can be used when no such cut stabilizes; otherwise the last valid draw
capability remains in place, preventing both intermediate-cut flicker and a
visually destructive guard/detail oscillation.

`GaussianLodStatus::resident_pages` counts decoded pages admitted to the bounded
page cache and assigned physical atlas slots; it is not the number of pages
visible in the current view, the selected-frontier page count, or proof that a
page generation is already GPU-current. It includes permanently pinned guard
pages and completed detail pages retained as warm cache entries. The count may
therefore rise with a stationary camera while already-requested transport and
preprocessing finish. With unchanged demand it must settle; across camera
history it may retain useful pages until eviction pressure, but it remains
bounded by the configured resident page, byte, and Gaussian limits.
`selected_gaussians` is the global frontier size, while the renderer
frustum-compacts that frontier to the actually submitted draw. The viewer names
these counters `Scene-wide selected` and `Pre-cull candidates`: neither is the
post-frustum indirect draw count. Camera direction still controls which visible
subtrees refine, but a complete global antichain retains coarse off-frustum
representatives so it remains spatially valid after motion. The exact compacted
draw count remains GPU-resident and is not synchronously read back into the UI.

Package replacements upload full page slots incrementally under the global
per-frame limits. Existing camera views keep recomputing and drawing their last
complete candidate while its leased pages remain non-evictable. Only after all
required target—or morph-union—generations are GPU-current does the renderer
commit candidate descriptors, compact, radix-sort, and advance the candidate's
two-phase publication token. Retired slots are cleared only after the endpoint
switch. New views without a prior output remain in the loading state until
their first complete ancestor candidate is ready; a package page-cache atlas is
never drawn as an unfiltered parent/child union. Device loss invalidates GPU
generations and rebuilds atlas, compaction, radix, morph bindings, bind groups,
and candidates from retained CPU state. The package re-enqueues each
materialized slot in the leased current cut once; the ordinary per-frame
uploader drains that replay before compaction and radix republish it.

On Wasm, package page verification and fixed-record decoding advance
cooperatively with a bounded Gaussian budget per application frame. Native
package preprocessing and transient flat-cloud hierarchy construction use
bounded worker backends. Both paths preserve checksum, codec, validation, and
support-bound error order.

## Debugging

Use named presets rather than raw scalar tuning:

```bash
--lod-debug=level
--lod-debug=page
--lod-debug=residency
--lod-debug=boundaries
--lod-debug=selection-pressure
```

Debug metadata uses the ordinary Gaussian raster path. The panel reports when
metadata is ready and when a flat exact-original source has no hierarchy to
annotate. Prebuilt packages retain their exact leaf hierarchy at quality one.
The canonical Garden gate changes `Off -> Page -> Off` without changing the
ACTIVE logical cut. It requires nonblack draws throughout, the exact Page
pipeline after binding readiness, stable consecutive Page frames, categorical
Page color diversity, and exact restoration of the Off alpha silhouette. RGB
continuity is deliberately not required across Page's diagnostic color
boundaries.

See [LoD debugging](lod_debug.md) for interpretation.

## Benchmarks

The portable CPU benchmark target covers hierarchy construction and traversal,
manifest validation/compilation versus validated `Arc` sharing, package-runtime
initialization, page codec throughput, steady-state runtime updates, candidate
range construction, upload-queue coalescing, and lazy external-build planning
through the default 234,881,024-record ceiling:

```bash
cargo bench --locked --no-default-features \
  --features "lod_build_sh3,lod,testing" --bench lod
```

The opt-in GPU target compares canonical CPU preprocessing plus Morton sorting
with the complete production GPU-assisted external preprocessor, including the
CPU support-bound reconstruction after GPU readback. It also labels and
measures the legacy ABI 6/v2 leaf and internal reduction primitives as
experimental microbenchmarks; those are not the ABI 16/v4 hierarchy path and are
not an end-to-end CPU/GPU quality or speed comparison. Merely compiling this
target never opens an adapter; execution requires an explicit environment flag:

```bash
RUN_GPU_LOD_BENCHMARKS=1 cargo bench --locked --no-default-features \
  --features "lod_build_sh3,testing" --bench lod_gpu
```

These Criterion cases isolate deterministic stages. They do not relabel the
current semantic-manifest-v3 record-major decoder/transposition path as
zero-copy, nor do they use microbenchmarks as a substitute for the headless
first-ACTIVE, cache, recovery, and visual-quality integration gates below.
Browser/CDN request latency and peak RSS remain deployment measurements because
stable thresholds depend on the host and transport.

The [canonical Garden preprocessing report](lod_garden_report.md) identifies
the real 5.83-million-record ABI 16/v4 host-Morton artifact, records its hashes,
bounded spatial-fit coverage, selector distance sweep, and completed
authenticated 1920x1080 boundary oracle. On 2026-08-22, the complete
persistent-edge temporal gate passed the final shader tree in 433.57 s for the
standard profile and 426.69 s with `precompute_covariance_3d`. Frozen refining
and coarsening controls retained 4,258,853 records/507 peak edges and 5,146,656
records/621 peak edges, respectively, with zero lag, uploads, writes,
allocations, or hard frames. The 48/48-fractional Dynamic traces refined
4,258,853 -> 5,074,997 records through 11 endpoint changes and coarsened
5,146,656 -> 4,416,507 through 13. They used 13/14 authored publications with a
maximum two-frame hold, 11/12 immutable-table uploads, 34/34 weight writes, and
zero allocation, lagging, or hard-fallback events. The reversible roundtrip
covered 141 samples, 48 poses, and six keyframes with 619 peak edges, 37 table
uploads, 104 weight writes, zero allocations, and 38 authored publications with
a maximum two-frame hold.

The endpoint-radiance K2 shader gate passed in both covariance profiles, and
the generic GPU qualification script passed on an RTX PRO 6000 Blackwell using
Vulkan driver 610.43.02. Superseded shared-frame-clock measurements remain in
the report only as historical provenance. The current evidence covers one
canonical fixed-key temporal workload and the generic GPU-script surface, not
the broader ignored Garden static, interactive, debug, and automatic/native
matrix. Arbitrary shader hot reload, previously unseen simultaneous ACTIVE
raster-key churn, browser/CDN latency, and cross-adapter real-time behavior also
remain separate qualifications.

## Validation

The non-GPU release checks are collected in:

```bash
tools/qualify_lod_release.sh
```

GPU validation is explicit and never runs implicitly:

```bash
BGS_RUN_GPU_QUALIFICATION=1 tools/run_lod_gpu_qualification.sh
```

Deterministic CPU camera traces measure motion-cancelled temporal residuals,
`(LoD_t - LoD_{t-1}) - (exact_t - exact_{t-1})`, plus temporal curvature.
They distinguish frontier identity rather than count alone, bound settled-cut
noise and cut-event spikes separately. A separate compatibility trace verifies
that pre-ABI16 packages without morph payloads bound each categorical cohort's
event energy relative to an immediate coarsening jump. ABI 16 instead evaluates
persistent per-view edge weights directly from the current camera; its monotone
parent map and optical-depth morph are an authored hierarchy transition, not the
paper's per-ray depth-sort ordering, and do not claim StopThePop's ordering
guarantee.

The canonical Trellis quality workflow verifies the exact artifact length and
SHA-256, runs matched-resolution foreground PSNR/SSIM/IoU/alpha and morphology
checks across near/mid/far cameras, then uploads the report even on failure.
Locally, provide the already-verified fixture:

```bash
BGS_TRELLIS_GLB=/absolute/path/to/trellis.glb \
BGS_TRELLIS_AUDIT_PROFILE=full \
BGS_LOD_REPORT_PATH=/tmp/trellis-lod-quality.md \
cargo +1.95.0 test --locked --test lod_real_scene_quality \
  --features "lod_build_sh3 testing headless" \
  canonical_trellis_high_quality_color_and_covariance_audit -- \
  --ignored --nocapture --test-threads=1
```

Package data is untrusted input. Size/count/allocation bounds, version checks,
checksums, path confinement, request caps, and cache validation are part of the
runtime contract; see [SECURITY.md](../SECURITY.md).
