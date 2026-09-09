# LoD implementation and acceptance status

This tracks the [branch plan](lod_engine_plan.md) against `feat/lod`. The engine
remains experimental: implementation progress does not override failed quality
or performance gates. Historical runs and their exact scope remain in the
[qualification report](lod_runtime_qualification.md).

The [2026-09-09 production refinement](lod_production_refinement.md) records the
latest ordering, residency and capture-contract changes, fresh Poland motion and
aerial captures, and ordered/GPS facade comparisons. It separates preliminary
timing observations from captures with explicit producer-input receipts; release
qualification remains false. Earlier evidence below retains its recorded scope.

## Current closeout (2026-09-09)

The latest [temporal qualification](lod_temporal_quality.md) supersedes the
earlier motion and church observations below. Shared page leases, fixed-slot
publication, pending-demand retention and eviction-aware capacity checks address
CPU overhead and residency churn. Conservative view omission reduces physical
expansion after complete-cohort logical selection; it does not skip residency
dependencies or reclaim logical selection budgets. A shared smooth finite support kernel closes the
small GPU fixture's clipped-tail endpoint discontinuity without a timing-based
transition. These changes do not establish seamless streamed motion or qualify
the existing aerial proxies. The new bounded local fit improves its training
views but does not improve the separate development view; no package is promoted.
The linked report distinguishes historical captures from the final renderer's
route, image and timing evidence. Whole-frame responsiveness, GPS/automatic
budget quality and production acceptance remain open.

The earlier [Poland scene follow-up](lod_poland.md) covers the full 106,447,647-record
SH0 source and its 600 calibrated cameras. Writer 17 removes support bounds from
un-emitted candidates; runtime refinement uses conservative boxes. Full-source
conversion takes about nine minutes with roughly one GiB peak RSS on the
recorded machine. Both GPU renderers completed the 600-pose route with sampled
image/count receipts. Resident-first traversal admission fixes held-camera page
churn. The ordered camera-300 revisit settles at 14,935,787 selected records and
3,831,900 drawn quads, with a 68.07 dB / .99998 single-pose reference comparison.
Those reduced cuts failed badly at the church; GPS timing-driven cap and sampling
changes do not qualify image quality. Point capture also retains incomplete
cadence coverage. Consult the report for the exact settings and evidence limits.
The [Poland quality plan](lod_poland_quality_plan.md) tracks the next gated work:
authenticated local substitutions, calibrated/context-aware fitting, an adaptive
representation decision, prioritized GPU admission, GPS quality qualification,
and quality-aware budget control. Its implementation ledger distinguishes the
new cut attribution, local supervision, GPS expectation and sampling-floor work
from unpassed representation and production gates. The corrected local seed now
reproduces the captured cut's CPU RGB exactly at one church pixel; replacing two
8:1 proxy owners with originals removes their false-positive opacity there.
The [bounded fit screen](lod_poland_quality_results.md) then completed both
24-step candidates. Geometry fitting improves the church training crops and
survives export/reload; the selected candidate still fails validation at
32.31 dB / .895 SSIM. Its previously unused far crop passes 41.07 dB / .989,
which does not qualify the whole representation. No fitted package is promoted.

GPU hierarchy outputs now feed both GPS and globally ordered quads. Flat and
CPU-compacted sources reserve their prepared capacities before hierarchy roots
share the remaining view budget. A failed traversal suppresses the shared draw
and its renderer-neutral, generation-fenced residency receipts. The viewer
supports `--global-order --lod-gpu-traversal --input-lod ...`.

Camera-motion fixes separate current-frame selection from residency publication:
deterministic GPU admission prevents held-cut races, compatible publications reuse
traversal workspace and retain in-flight feedback, and pending atlas uploads keep
the previous authenticated hierarchy traversable with the current camera. GPS
renders even when telemetry buffers are busy. See the
[GPU camera-update contract](lod_discrete_residency.md#gpu-selection-during-camera-movement).
The viewer also offers actual `bevy_flycam` controls through
`--camera-controller flycam` and the matching config/query fields. These changes
do not qualify continuous transitions, cold-loading latency, or Poland proxy quality.
Focused validation passes: seven CPU regressions and three bounded native GPU
fixtures. The traversal oracle checks identical held/returned cuts and immediate
camera-dependent selection. The GPS fixture moves the image with a full telemetry
ring. The package fixture blocks replacement uploads, verifies current-camera
rendering from resident data, then verifies buffer reuse through refinement.

Camera sort slices are device-aligned and isolated across radix, Std and Rayon;
slots follow camera identity independently of render order and survive source
shrinkage and camera removal. Streaming now reuses unchanged cohort topology,
deduplicates ancestor walks and prioritizes visible replacement work.

The four small native GPU fixtures pass on NVIDIA RTX PRO 6000 Blackwell Vulkan,
driver 610.43.02: camera isolation, global ordering, GPS and GPU package lifecycle.
The ordered fixture includes two streamed hierarchies plus a flat source, exact
nine-record admission, coarse/exact residency receipts and whole-image rejection
when roots do not fit. GPS fixtures exercise actual timestamp-driven sampling
and hierarchy-budget reduction. These establish functional behavior on this
adapter, not performance at production scene sizes.

Twenty-three focused CPU checks, 34 Python protocol checks and three JavaScript
observer checks pass; affected native library/test/capture targets pass Clippy
with warnings denied. New GPU and protocol fixtures are wired into CI. The
[closeout record](../target/lod-roadmap/2026-09-08/production-closeout/summary.json)
preserves commands and executable hashes. No broad covariance suite was repeated.
The standard default library/viewer also pass Clippy with warnings denied.
Package listing resolves all 127 checked literal and embedded-shader dependencies;
archive compilation and publication remain pending acceptance. Capture telemetry now exposes its completed sample to callers, while native
frame helpers and point-view exports compile only for their consumers. This
also closes the standalone `default + testing` instrumentation lint failures.

The existing native-pixel fourfold Icecream rung completed its missing hardware
screen. All eight views still fail 35 dB/.98, despite improvement over the seed
(29.237–34.961 dB, .931243–.958659). The
[report](../target/lod-roadmap/2026-09-08/production-closeout/quality-report.json)
qualifies neither representative quality nor full-frame speed. No new fit sweep
or weakened gate was used.

The bounded 100M closeout now proves sustained recovery of the stationary
10,115-instance draw after return and completes zero-ledger unload/reload. Final
movement/eviction package-update p95 remains 1.427/1.424 ms, above the unchanged
1 ms gate. Physical process/adapter observations and workload limits are in the
[large-scene report](lod_virtual_city_results.md#2026-09-08-scheduler-closeout).
This diagnostic uses CPU-selected quads; it does not qualify large-scene GPU
hierarchy traversal.

The current SH3 browser fixture passes on the actual non-fallback NVIDIA
WebGPU device in Chrome, including zero-ledger unload/reload at the requested
160×90 viewport. It records fourteen same-submission GPU observations with no
mapping/count errors or dropped readbacks. See the
[browser report](lod_browser_qualification.md#current-gps-hardware-execution-2026-09-08)
for source identity and the distinction between root reload and restored leaf detail.

Unused renderer backends, reducer scaffolding and inert APIs were removed
without aliases. See the [migration guide](lod_migration.md) for public API and
feature changes. Browser and large-scene results below must be tied to their
actual capture artifacts.

| Phase | Implemented or executed | Remaining acceptance requirement |
| --- | --- | --- |
| 0: baseline and measurement | Isolated main/branch references; source/package hashes; authored Garden cameras; same-submission indirect counts, images, GPU timestamps, CPU scopes and startup clocks; native/browser capture tools. | Complete uninstrumented comparisons, broader scene coverage, and classification of residual flat-render differences. |
| 1: allocation and CPU overhead | Independent descriptor capacity; package draws without dense per-cloud sort assets; cache fast path; indexed retirement; shared staging; bounded asynchronous/cooperative preparation; native HTTP checksums on existing workers; package memory admission; separately charged capacity successors. | Full physical-memory and before/after performance qualification. Initial AssetServer parsing, debug/flat storage, capture/render targets and driver pools remain outside the package ledger. |
| 2: representation and error | Conservative matrix projection; discrete endpoints; authenticated rung export; fixed-count, geometry-aware teacher fitter with per-owner feasibility; tiny CPU/GPU raster comparison; corrected OBB density; real Garden/Trellis/Icecream screens. | Useful reduced representatives and calibrated appearance residuals. Garden and the fourfold Icecream rung fail the locked image-quality gates. Fitting remains diagnostic. |
| 3: demand paging | Complete replacements; joint logical/physical admission; moving-camera request retention; visible-demand priority; pinned-cut and packed-page recovery; view fairness; successful native 10M/100M and browser unload/reload lifecycles. | Matched dynamic quality/work, motion/latency gates, complete memory accounting, throttled-network qualification and incremental selection budgets. Synthetic coarse coverage does not establish image quality. |
| 4: GPU work generation | Bounded wavefront hierarchy traversal, complete-cohort admission, physical-record expansion, GPU page requests and fenced package snapshots. Native traversal oracle and software Vulkan package/GPS integration checks pass. A bounded [global quad order](global_quad_order.md) projects flat, CPU-selected and GPU-selected clouds into one shared sorted stream; split/merged images and mixed-source residency checks pass on NVIDIA Vulkan. | Broader global-order image/performance, motion and large-scene qualification. Participating GPS clouds share one visibility stream, with broader center-depth qualification outstanding. |
| 5: temporal control | GPS whole-layer sampling plus optional view GPU timing/controller, bounded hierarchy-cap trials, stale-feedback rejection, worse-cost rollback and unmet-target diagnostics. Authored quality remains unchanged. | Hardware control stability under motion, latency and mixed workloads; transitions after useful endpoints exist. Controllers do not replace the quality gate. |
| 6: raster and compression | GPS corrected sampling, shared two-pass visibility, admitted sampling-layer storage and optional LoD radix workspace. Completed bounded real-source image/time comparison; small opaque-mesh/equal-depth software checks pass. | Equal-quality complete-frame gains, broader visibility qualification and browser/hardware matrix. Flat identity-index storage remains; tile/compression alternatives need measured gains. |
| 7: integration | Main additive/emissive behavior restored; native/WebGPU checks, browser lifecycle tooling and build/capture documentation. | Qualified defaults, broader hardware/browser matrix, visual review and measured integration onto main. No merge or publication has occurred. |

The procedural runs use real unique source records and authenticated on-demand
pages. Both 10M and 100M complete cold start, movement, return, eviction,
zero-ledger unload and reload under declared CPU/GPU reservations. At quality
.9 and a 65K active ceiling they render different visible work, even with 256
page slots; that pair cannot support a scaling claim. A separate exact-visible
profile produces matching 10,115-instance stationary draws at both source sizes.
Its instrumented timing is a paging diagnostic, not an accepted LoD operating
point or proof of quality for 100M unique records.

The original two branch commits are reviewed in the engine plan. Follow-up
implementation preserves their history; working-tree changes remain available
for review. Earlier CPU-only status is archived with the local experiment
artifacts rather than presented as current state.

Related contracts:

- [Capture and provenance](lod_capture.md)
- [Package preparation](lod_package_preparation.md)
- [Discrete residency](lod_discrete_residency.md)
- [CPU timing scopes](lod_cpu_telemetry.md)
- [Projection parity](lod_projection_parity.md)
- [Representative fitting](lod_representative_fitting.md)
- [Browser qualification](lod_browser_qualification.md)
- [Large-scene qualification](lod_virtual_city_qualification.md)
- [Gaussian Point Splatting](gaussian_point_splatting.md)
