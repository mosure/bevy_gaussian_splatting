# LoD frame capture contract

This opt-in interchange is available under the `testing` feature as
`testing::lod_capture`. Reading or validating records does not start a GPU.
The native `capture_lod` tool described below explicitly runs the renderer and
emits observations. Ordinary applications do not install its instrumentation.
The producer must supply actual observations and retain the identity of the work
that generated them. The checker validates supplied data; it cannot prove that a
producer obtained a value from hardware.

Run the small CPU fixture and the validator tests:

```sh
python3 tools/check_lod_capture.py tools/fixtures/lod_capture_synthetic.jsonl
python3 -m unittest discover -s tools -p 'test_check_lod_capture.py'
```

The checked-in fixture is explicitly `synthetic`. Its numbers, hashes, camera and
image name are schema examples and have no performance or quality meaning. It
passes diagnostic validation and fails `--require-gpu-evidence`.

## Capturing the native renderer

The focused native harness streams a real package through the production package,
compaction, radix and Gaussian draw paths. Its small synthetic scene is generated
as saved source bytes and an authenticated package before opening the renderer;
the resulting GPU observations have `native_gpu` provenance.

```sh
cargo +1.95.0 run --locked --no-default-features --features 'headless testing' \
  --bin capture_lod -- --config tools/fixtures/lod_runtime_synthetic.json
python3 tools/check_lod_capture.py target/lod-roadmap/captures/synthetic/capture.jsonl \
  --verify-images
```

The output directory must be new. Copy the Garden or Trellis configuration under
`tools/fixtures/`, set the actual package manifest, original source and builder
revision, and choose a new output directory. Paths are relative to the configuration
file. Package loading supports the renderer's current manifest/page ABI; the
original source is hashed for attribution and need not be loaded into GPU memory.
`request_timestamps: true` requests the device timestamp features; set it to false
for devices without them. Such a run retains empty GPU timing maps.

`frame_period_ms` optionally sets a minimum headless schedule period, for example
`33.333333333333336` for a 30 Hz camera path. The default zero runs without pacing.
This gives streaming real time between poses and prevents an unpaced logical path
from being mistaken for interactive motion. The resolved value participates in
the settings identity. Pacing is not a frame-time cap: slow work can exceed the
period. It does not measure display presentation or input latency, and its wait
must not be interpreted as renderer cost.

Camera segments optionally accept `"up": [x, y, z]` (default `[0, 1, 0]`).
This preserves authored camera roll; for a COLMAP camera-to-world rotation with
columns right/down/forward, use `up = -column1`, `target = position + column2`.
Validation rejects nonfinite/zero up vectors and paths that meet the target or
become parallel to up, including an interior point of a linear segment. Up is
included in the saved camera path/settings identity and submitted view matrix.

`presentation_mode: "Discrete"` selects permanent categorical cut presentation.
The omitted/default value is `"ContinuousMorph"`. Discrete mode uses the same
quality target, conservative projected error, culling, and complete-cut
publication, but does not construct persistent, late-residency, or predictive
parent/child blends. This is useful for measuring the selected representation's
actual quality independently of a child-count morph. The mode is saved in
`settings.json` and participates in run identity. It has no effect on flat-source
rendering. See `tools/fixtures/lod_runtime_synthetic_discrete.json` for an example.

GPU hierarchy captures can opt into current-camera spatial transitions with
`render_mode: "hierarchy_ordered"`, `presentation_mode: "ContinuousMorph"`, and
`spatial_transitions: {"max_transition_nodes": 256,
"max_transition_records": 65536, "max_mapping_bytes": 33554432}`. The policy
bounds adjacent fractional edges and their correspondence storage. It is saved
in the settings identity. GPS does not accept this transition policy. Mapping or
residency limits must be checked in the capture diagnostics before interpreting
a discrete fallback image as a transition result.

Ordered captures write `device_preflight.json` before streaming any pages. It
records the negotiated device limits and checks the same renderer allocation
plan used for drawing. An oversized projected buffer fails immediately with a
`capture_status.json` reason. The preflight covers this single-cloud ordered
allocation, not atlas residency or the whole device's memory use. Submission
evidence separately records actual/requested spatial pipeline use, emitted and
required transition counts, and categorical fallback flags.

Calibrated captures can render a physical tile with
`camera_crop: {"full_viewport": [960, 638], "origin": [0, 319]}` and
`viewport: [480, 319]`. Every segment must select an authored `camera_frame`.
The full viewport retains the camera calibration; the output viewport specifies
the tile size. The tile must lie wholly inside the full viewport. This uses
Bevy's sub-camera projection and preserves physical pixel rays. Cropped captures
have their own camera and settings identities: a tile is not interchangeable
with a resized full-frame image. Full-frame versus tiled rendering must pass
an image comparison before stitched tiles serve as a reference, because view
culling can change the contributing records.

`render_mode: "flat_source"` loads the original PLY or a bounded, self-contained
single-primitive GLB through the production decoder and renders the original
ordered stream. Its count pipeline is `flat_source`: `compacted` is null because
there is no compaction stage, while `drawn` comes from the actual draw buffer and
command attestation. Generation zero explicitly means no hierarchy publication.
`max_source_gaussians` bounds this opt-in fully resident teacher. The saved
`flat_source.json` descriptor is its manifest identity when no package is supplied.

For GLB package building, first convert without opening a GPU:

```sh
capture_lod --convert-glb scene.glb --output scene.ply --max-gaussians 500000
```

The converter writes binary PLY and `scene.metadata.json`, preserving the decoded
instance transform and color convention and measuring the opacity/log-scale float
roundtrip. Point the external package builder at the PLY; point capture `source`
at the original GLB and `source_metadata` at that sidecar so the package and flat
teacher use the same world coordinates and color convention. Conversion accepts
one self-contained primitive and one instance and refuses existing outputs.

Set `capture_images: false` for counts/timestamps without texture readback or PNG
encoding. Each GPU staging slot then occupies 512 bytes plus its timestamp resolve
buffer. These runs remain instrumented; they are not uninstrumented throughput.

`capture_start_path_frame` defaults to zero. Set it to the warmup length when
capturing a dense motion window: regular image/count readbacks start at that
logical frame, while bounded startup probes still establish renderer readiness.
This avoids writing an image for every warmup frame when `capture_every` is one.
Image-bearing runs currently encode PNG on the main thread, included in frame
cadence. The settings identity records this distinction.

Set `capture_mode: "cadence_only"` to run the same camera path without installing
GPU instrumentation, extracted capture requests or the draw probe. This mode
forces images and timestamps off and writes `frame_cadence.jsonl` plus
`cadence_summary.json`, with CPU main-loop frame timing and exact intended logical
poses. It emits no count/image capture file and makes no draw or GPU completion
claim. Samples stay in bounded memory until exit; no per-frame file or PNG writes
occur. The final JSON reports the real adapter and the mode explicitly.

Hierarchy cadence waits at the initial stationary boundary for the main-world
package's `Active` status. Flat cadence waits for the main-world source asset plus
`cadence_minimum_warmup_seconds` (default two seconds after the frame loop starts).
The latter is a fixed warmup, not proof of GPU readiness. Qualify the same scene
with an instrumented run before using cadence for performance comparison. Startup
pauses retain at most four cadence samples per second; logical paths are limited
to 100,000 frames and timeouts to 3,600 seconds. Scenario summaries use nearest-rank
p50/p95/p99; startup and steady-state scenarios remain separate.

Every image-enabled capture copies the rendered RGBA8 image, the actual indirect argument buffer
and four GPU timestamps in the same camera submission, after Bevy's final upscaling
write into the camera target. This final write also runs at a 1:1 resolution.
The render command separately attests that it issued `draw_indirect` using that
exact buffer and generation. `drawn` remains null without that attestation.
Image files and JSONL records retain the submission's frame/view/generation through
asynchronous completion. Publication generations, atlas epochs, fingerprints,
candidate hits, overflow counts and draw attestations are saved in
`submission_evidence.jsonl` for checking retained-output/successor transitions.

The GPU ring has 1–8 slots and a separate 256 MiB admission ceiling. Mapping uses
nonblocking polling. A full ring drops a sample rather than stalling camera motion;
`capture_status.json` reports dropped, missing, unattested and failed observations.
The initial stationary camera segment waits at its final pose until the first
completed GPU readback of an attested command with nonzero vertex and instance
counts, bounded by `timeout_seconds`; subsequent poses advance by logical
`path_frame`. This prevents asynchronous startup from consuming the entire path.
The first segment's `from` and `to` should therefore be identical. Startup attempts
are limited to four per second, and at most four missing-drawable image samples
are emitted. The last candidate/atlas/handshake diagnostic remains in the status
file. Sampling follows logical path-frame indices after readiness, allowing exact
camera-pose pairing despite different startup durations.

Native `capture_lod` status includes `stats.startup_timing`. Its clock begins at
`run_capture` entry, before config/source loading, identity hashing and Bevy/GPU
initialization; process launch before that function is excluded. The first
sampled nonzero attested draw retains its original frame/generation, selected and
drawn counts, frame-start elapsed seconds, and readback-observed elapsed seconds.
A separate field requires a complete package frontier and stays absent for flat
references. The frame start and readback completion bound that sampled draw;
four-per-second startup sampling does not identify the exact first GPU draw or
certify image quality. Missing run-entry origins remain null, including the
separate procedural virtual-city runner. Older artifacts cannot reconstruct this
timing.

Both native and procedural runs also report `renderer_loop_timing`, starting at
the first capture `Core3d` execution with a `RenderDevice`. This is a renderer
loop proxy after initialization, not the exact device-initialization instant.
Its complete-package readback elapsed time supplies a separate sampled upper
bound after that origin. If the corresponding main-world frame began before
this origin, its frame-start lower bound is zero and
`frame_started_before_origin` is true. Run-entry and renderer-loop elapsed times
must not be compared as if they had the same origin.

Captures from executable SHA-256
`376354424d326d41a9c416de67ce3185ca307e442dc0a360eaec021fba2ea042`
predate this final-target ordering fix. Their draw-command/count evidence remains
separate from image evidence: a first image after a pose or publication change may
contain the preceding frame. Those images cannot qualify exact frame/pose pairing;
stable stationary images need separate verification.

GPU timings bracket compaction, sorting, and raster/postprocessing; `view_total`
overlaps those stages and must not be added to their sum. Frame wall time is
main-loop start-to-start cadence, attached to the preceding frame. The recorded
`main_to_render_encode` CPU time ends at command encoding. Asset hashing, synthetic
package construction, PNG encoding and capture allocation overhead are not claims
about uninstrumented engine throughput. Compare a matching uninstrumented run.

Memory observations currently include actual private LoD buffer allocations,
retirements, the capture ring and process RSS. Unmeasured atlas/CPU/backend
categories and independent device memory remain absent. Consequently these native
captures are **partial diagnostics** and intentionally fail strict memory evidence
completeness. The producer reports `release_qualified: false`; running it alone
does not qualify visual quality, streaming resilience or performance.

## Exporting an authored rung without a GPU

The CPU-only exporter selects every node at absolute depth `D`, retaining any
original leaf ending above that depth. It validates complete canonical-source
coverage and disjoint page ranges before allocating payload buffers or creating
outputs. This selects a representation independently of the runtime error policy.

```sh
target/debug/capture_lod --export-rung path/to/scene.gsplatlod \
  --output path/to/rung.ply --depth 2 --max-gaussians 1500000
```

Both the PLY and adjacent `rung.rung.json` must be new. The exporter handles local
individual pages and manifest-described packed byte ranges, rejects escaping
paths/symlinks, and validates page IDs, counts, encodings and versioned payload
checksums with the production decoder. It retains one decoded page at a time;
assigned output offsets preserve canonical source-node order for shared pages.
Failure removes only partial files created by that invocation.

Admission ceilings are 64 MiB encoded manifest, 262,144 nodes, 65,536 pages,
16 MiB per encoded/decoded page, 65,535 records per page, and 2 GiB each for total
encoded reads, decoded page work and output PLY bytes. The explicit output count
limit must be in `1..=8,000,000`. Page counts/lengths are checked before allocation;
this export does not materialize the source scene or the whole decoded rung.

The sidecar contains manifest/output/encoded-page SHA-256, the manifest's canonical
source fingerprint and source count, selected node/source/output ownership ranges,
page-local offsets, conservative bounds, unencoded per-owner record SHA-256 and
build provenance. The source fingerprint is not the original PLY/GLB file SHA-256.
No input-file hash, cloud transform, color space, image quality certificate or
measured PLY scalar roundtrip is inferred. PLY opacity/log-scale conversion uses
the same writer as the bounded GLB converter.

The native `capture_lod` harness sets `opacity_adaptive_radius: false` for both
flat-source and package clouds, so its exported-rung screen uses the same fixed
three-sigma support convention. An ordinary flat viewer or another harness using
the default adaptive cutoff has different raster conditions. Check that setting
and explicit derived-artifact lineage when comparing an export against the
original teacher. The ordinary capture comparator rejects unrelated source hashes;
do not replace its identity fields to make an exported PLY appear to be the
original source.

## Producing records

Construct `LodFrameCapture` and call `capture.write_jsonl(&mut output)` for every
frame/view. The writer validates before serializing a line. The JSONL fixture
contains every field and can serve as an interchange example for other languages.
Unknown fields and unsupported schema versions are rejected; coordinate schema
changes in the Rust producer and Python checker together.
Both implementations reject duplicate evidence-map entries, boolean numbers,
non-finite values and counts outside their integer widths. The reader limits each
JSONL line to 4 MiB before parsing it.

Each record includes:

- Exact source and manifest SHA-256; builder/renderer revisions; renderer executable
  or Wasm SHA-256; sorted feature flags; backend/adapter/driver; camera-path and
  settings hashes; instrumentation mode.
- A run, view, submitted frame and residency/publication generation; camera matrices,
  viewport and physical pixel scale; scenario such as `cold_start`,
  `warm_stationary`, `motion`, or `return_after_eviction`.
- Selected and candidate records, explicit transition additions, output capacity,
  and optional observed compacted/drawn records with a declared origin.
- Optional CPU stage times, GPU timestamps, whole-frame wall time, disjoint memory
  categories, independent RSS/device memory observations, and an image identity.

Hash the **actual manifest bytes**; its authenticated shard references identify
payloads transitively. Hash the actual source bytes and executable/module. Hash a
canonical, saved settings document containing quality, work/memory limits, sorting,
filtering, coordinate/SH/alpha/color conventions and reference mode. Record that
serialization convention alongside the run. A different artifact, renderer build,
feature set, configuration or instrumentation mode requires a different run ID.
The validator rejects identity changes within a run.

The generation is the generation actually consumed for rendering. Attach a
`LodCaptureStamp` when submitting work and carry it through readback and screenshot
completion. The checker rejects an image, timing, memory or count stamp from a
different view, frame or generation. It accepts out-of-order completion but rejects
duplicate run/view/frame entries, including conflicting generations.

`null` compacted/drawn values mean **unobserved**. They must remain null for
`cpu_selection` evidence. CPU oracles use `cpu_oracle` mode/source; GPU captures use
`native_gpu`/`web_gpu` mode and `gpu_readback` only after reading the actual count and
indirect draw buffers. The default `hierarchy` count pipeline requires drawn Gaussian records to equal
observed compacted records. The explicit `flat_source` pipeline requires null
compaction and bounds actual draw instances by the source/output capacity. Old
schema-version-one records without `pipeline` retain the hierarchy contract. Candidate records
may exceed selected records only by the separately recorded transition additions.
Backend names are canonical: `synthetic` for synthetic mode, `cpu` for CPU oracles,
`vulkan`/`metal`/`dx12`/`gl` for native GPU captures, and `webgpu` for browser WebGPU.
Conflicting mode/backend declarations are rejected.

CPU/GPU allocations are disjoint categories with reserved and used bytes. Assign
shared allocations one owner. Record explicit zero entries for measured unused
categories, and omit categories that were not audited. These omissions keep GPU
measurement completeness false. Include staging, transitions and retired resources;
do not equate the resulting allocation sum with independently observed RSS or
whole-device memory. Optional timing stages contain nonnegative milliseconds;
`gpu_ms` contains device timestamps, never CPU submission elapsed time.

The render-world `render::lod::LodCompactionBuffers<R>::memory_snapshot()` API
provides CPU bookkeeping for current private LoD compaction/radix buffers and
retired buffers awaiting queue completion, plus their tracked sum and view/cloud
state count. Sampling performs no GPU readback or wait. Its live and retired
categories are disjoint; do not also add their sum as another allocation category.
This covers one planar representation and excludes atlas/cloud storage, debug,
per-cloud sort, CPU mirrors, Bevy/wgpu staging and driver allocations. It is not a
whole-device memory observation; completed resources may remain in backend pools.

## Checking a later measured run

These commands inspect existing evidence. They do not execute experiments:

```sh
python3 tools/check_lod_capture.py /path/to/capture.jsonl \
  --require-gpu-evidence --verify-images \
  --require-scenario cold_start --require-scenario warm_stationary \
  --require-scenario motion --require-scenario return_after_eviction \
  --minimum-frames 120
```

Image paths resolve relative to the JSONL file. `--verify-images` hashes existing
files without decoding them. Store bounded image samples when full frame sequences
would exceed the capture budget; records without matching image identity remain
partial diagnostics under this strict completeness check.

Output separates every run/view/scenario and reports observed count peaks, recorded
memory peaks, timed-frame count, nearest-rank p50/p95/p99, and frames over 50 ms.
Missing values stay null. Exit code 0 means the requested checks passed, 1 means
invalid/unreadable input, and 2 means requested evidence/scenario/sample coverage is
incomplete. The CLI always reports `release_qualified: false`.

An evidence-complete capture still needs independently checked matching reference
images, radiance/alpha/temporal metrics, agreed thresholds, representative adapter
and browser runs, sufficient samples, and an uninstrumented timing comparison.
Neither file validation nor field completeness establishes useful LoD quality or
performance. Existing CPU/GPU oracle producers can adopt this writer when their
actual observations are connected; no runtime observation is fabricated here.
