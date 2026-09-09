# LoD runtime qualification — 2026-09-07

Implementation and diagnostic qualification on `feat/lod`. This report records
fresh GPU/system observations, with implementation and acceptance kept separate.
The roadmap remains incomplete; failed quality gates are not accepted phases.

## Frozen baseline and gates

The branch begins at `7ef9bb1c918a297e35338dd7687dac55ee0db496`; local `main` is
`67cde03783a60c2cdccc8f877da973665cddafc2`. An isolated baseline includes the earlier
CPU-only improvements. Its tracked-file snapshot SHA-256 is
`4cd9c7dbfaeba064767bbbfef6ae65eeaead362358cb4c9dcdf6167a2ca743cb`.
The baseline patch, snapshot metadata, fixed gates, commands, binary hashes and
logs are under `target/lod-roadmap/2026-09-07/`. Generated artifacts are local,
not repository fixtures.

Hardware is an NVIDIA RTX PRO 6000 Blackwell Workstation Edition, Vulkan,
driver 610.43.02, approximately 96 GiB VRAM, and 94 GiB system RAM. Desktop
applications remain running. Logical 4 GiB engine/2 GiB payload limits on this
adapter do not qualify physical 8 GiB hardware or another vendor.

The locked target remains 1080p p95 <=16.7 ms, p99 <=25 ms, foreground teacher
PSNR >=35 dB and SSIM >=.98, ordinary-view post-cull reduction >=4x with >=2x
complete-frame improvement, and <=5% small-scene overhead. Fixed-visible-work
10M-to-100M stress must meet the <=10% p95 growth gate. These are acceptance
targets, not measured results. Exact full gate definitions are saved in
`target/lod-roadmap/2026-09-07/gates.json`.

## Fresh source and package provenance

| Source | SHA-256 | Records |
| --- | --- | ---: |
| Garden PLY | `16701d5e0630dfaca74f8794ed7ce2aa23fa922f87dc09a7e37484e8d3f82d5a` | 5,834,784 |
| Trellis GLB | `fbe9d96b6689a78228c121e5f1bc8c5ccc32cef1941294d25f1db66f4a901dc1` | 478,368 |
| Icecream PLY | `131205a37bfb30c90ddb8a5a686a67c27729c69c5ca9cec591c79d9b5fdc202a` | 84,348 |

Garden was rebuilt from its hash-verified source using the isolated baseline's
bounded CPU external builder, SH3, branching 8, leaf capacity 1024, batches 65,536,
merge fan-in 32, support sigma 3. The new package is at
`target/lod-roadmap/packages/garden-baseline/scene.gsplatlod`; manifest SHA-256
`67b9119222e1435fb88755698dcd916e608c9cd21c1417b687a7cce663729600` matches the
historical canonical artifact. It contains 6,517 nodes/pages, 6,668,314 stored
records and three shards. This is a fresh determinism observation.

The build took 650.0 seconds with 80,488 KiB peak RSS (`/usr/bin/time -v`), using
the dev profile with root opt-level 1, dependency opt-level 3 and debug info off.
Hierarchy/encoding accounted for 617.1 seconds. Other CPU builds ran concurrently,
so this is provenance and a bounded-build observation, not an isolated throughput
comparison. Icecream's corresponding build completed in 2.32 seconds.

## Completed GPU checks

The production embedded OBB conic correction passes all 22 CPU/GPU image cases
in each covariance layout, including a nonzero viewport origin. Full floating
point and RGBA8 outputs use the same unchanged quantization/contributor error
bounds. These are actual Vulkan executions, not shader parsing or startup tests.
The frozen v9/v10 native results and exact executable hashes are indexed in
`gpu-suite-v9-v10-index.json` under the dated artifact directory.

| Check | Native result |
| --- | --- |
| Hidden cloud and additive/emissive toggles | Passed |
| Debug presets and offscreen-center support overlap | Passed; both covariance layouts |
| Automatic bridge camera/quality sweep | Passed; both covariance layouts |
| Bounded native package cuts | Passed; both covariance layouts |
| Denied capacity successors with two moving views | Passed; both covariance layouts |
| Injected device-loss recovery | Passed; both covariance layouts |
| Fragmented production physical-range lookup | Passed |
| Authenticated atlas GPU copy | Passed; both covariance layouts |
| GPU collision sorting and external multi-run package parity | Passed |

The authenticated K=2 morph test initially timed out before a transition existed.
Its old fixed quality/camera pair did not produce a finite parent-side interval
under the corrected conservative projection. The fixture now derives that
interval from authenticated node metrics and the live view. Actual execution
also exposed a categorical bootstrap handoff and stale selector history after
direct publication. The repaired non-precomputed path passes parent, child and
fractional radiance checks, including selection/highlight modes, with unchanged
image tolerances (`gpu-v19-morph.json`). The separate delayed-delivery test
exposed missing pre-I/O demand history, which is now retained per view. Its
fixture explicitly proves an adjacent cold transition before the original deep
near/far sweep; a deep target missing intermediate pages legitimately uses
categorical fallback. The repaired GPU test passes with six late edges and
twelve authored publications (`gpu-v19-package.json`).

The quality fixture expected cuts were still using the orientation-free
perspective constructor. Correcting only the fixture camera to match production
changed near cuts from `[1,3,7,11]` to `[1,3,46,214]`, and the near/far/distant
probe from `[7,2,1]` to `[46,2,1]`. Both GPU paths then matched. This confirms
camera agreement, and exposes the conservative bound's additional near-support
refinement. It does not establish an efficient approximation frontier.

The range test executes the production lookup functions on the GPU and compares
every output source ID/residency/presentation value with an independent CPU list.
Cases include zero, 255/256/257 boundaries, 65K one-record descriptors, more than
one million candidates, noncontiguous physical storage and invalid descriptor
gaps. Full compaction/radix/render composition also passed in the quality tests.
An experimental workgroup range window also passed membership parity, but its
timestamp benchmark was slower than the original lookup. Five ABBA blocks,
eight dispatches per sample, independent CPU output validation, and four
fragmentation cases measured median regressions of 4.8%, 27.1%, 46.0%, and 74.2%.
The production shader was restored. `range-lookup-abba.json` and
`rejected-range-window.wgsl` retain the experiment; the optional
`LOD_RANGE_CANDIDATE_WGSL` benchmark input reproduces it without changing the
renderer. These are lookup/write dispatch timings, not complete frame timings.

The capacity-successor GPU regression now passes with two moving views: denied
overlap retains both old draws and refreshes their radix order; admitted
successors remain separately charged and publish together after both are ready.
Injected device-loss recovery also passes on this adapter. The executed test
logs are `successor-v4-executed.log` and `recovery-v4-executed.log` (one enabled
GPU test each). The earlier `*-v4.log` invocations selected zero ignored tests
and are explicitly not execution evidence. The latest completed CPU library
suite at integration v19 reports 874 passed, zero failed, eight ignored
(`native-v19-cpu-lib.json`). This includes complete-cut physical admission,
packed-page recovery, memory leases, frame-associated CPU telemetry, independent
geometry derivatives and per-representative feasibility backtracking. The test
executable SHA-256 is
`58b0bb9902d1b3c7baaa4177d8c87cd833607b05e026af8621e328a46191caa2`.
The non-precomputed GPU package, morph, capacity-successor and recovery checks
pass at v19; its camera sweep passed at v18. The final precomputed-covariance
library suite reports 876 passed, zero failed, eight ignored
(`covariance-v19-cpu-lib.json`), executable SHA-256
`cd777caf934e4fda80e877ec50be2b0a189005e7c15dbd9b94125c5819f1fe3e`.
All five covariance GPU checks pass at v19: package delayed delivery, morph,
camera sweep, capacity successors and recovery (`gpu-covariance-v19-*.json`).
Clippy with warnings denied, default/minimal library builds, the wasm32 WebGPU
library build, formatting, 30 Python tool tests and three JavaScript observer
tests also pass. Final result records and diagnostic reports are indexed with
SHA-256 hashes in `qualification-wrap-up-index.json` under the dated artifact
directory. These are functional checks, not image-quality or performance
acceptance; earlier frozen browser and performance captures retain their
original source epochs.

## Capture validation and measured representation failures

The standalone capture producer now renders without a flat cloud to prewarm
pipelines. A real startup deadlock was fixed: the private radix variant was
queued only after candidates were staged, while staging itself waited for atlas
preparation that required that pipeline. The synthetic rerun attested 91 draws,
with zero dropped samples or unattested draw commands. Earlier blank runs retain
null counters and are rejected as render evidence.

The first Garden quality-.35 run still failed its 120-second deadline after
loading 4,222 pages without publishing any drawable cut. This is a separate
bootstrap-policy defect: the ABI16 spatial reducer uses version 4, while the
bounded bootstrap admission path accepted only version 3. The redundant version
check has been removed in favor of the authenticated bounded-amplification
contract. Fresh Garden runs now publish complete cuts. The failure is preserved
under `garden-q35/`; it does not count as a performance or quality sample.

The paired-image analyzer checks source identity, exact camera matrices, logical
camera sample identity, image hashes and actual draw attestations. Metrics use
RGBA8 sRGB output converted to linear premultiplied RGB, including output
quantization, with foreground-union PSNR, fixed 11x11 Gaussian SSIM, alpha error
and silhouette IoU. Empty images are rejected. Eighteen Python schema and metric
tests pass, including analytically known PSNR/SSIM cases. Samples dropped because
readback slots were busy remain recorded; matched intersections are diagnostic
screens, not complete camera-route qualification.

Fresh Icecream and Trellis captures used binary SHA-256
`376354424d326d41a9c416de67ce3185ca307e442dc0a360eaec021fba2ea042`.
They cover fixed near, mid and held-out poses at 1920x1080, FOV .7853982 radians,
near plane .01, with identical source transforms between each flat/package pair.
Trellis's authored transform and color space come from the GLB conversion
sidecar; scalar round-trip error was at most 1.1920928955078125e-7.
That older capture binary copied images after post-processing but did not
explicitly order the copy after final upscaling. Its settled stationary images
remain diagnostic; moving images cannot establish frame-accurate temporal
quality. The producer now orders image/count capture after final upscaling and
has a typed render-schedule ordering regression.

The following are the last matched stationary mid-view samples, with the
existing continuous morph mode still enabled. Counts are actual post-cull draw
readbacks. These results **do not establish useful coarse endpoints**.

| Source / quality | Drawn | Foreground PSNR dB | Foreground SSIM |
| --- | ---: | ---: | ---: |
| Icecream / .05 | 6,655 | 19.83 | .62292 |
| Icecream / .15 | 37,228 | 23.91 | .83055 |
| Icecream / .25 | 84,348 | 36.20 | .95965 |
| Icecream / .35 | 84,348 | 42.91 | .97773 |
| Icecream / .65 | 84,348 | 51.21 | .99527 |
| Trellis / .35 | 478,368 | 30.02 | .89004 |
| Trellis / .65 | 478,368 | 44.32 | .99543 |

The selector's supplied geometric field is the full source-support extent,
not a representation residual. On Icecream's deepest reduced rung, those errors
are 1.165–1.424 world units. Conservative projection therefore requires near-full
refinement. Persistent morph edges can then retain full child cardinality while
showing a coarse appearance. Discrete-cut measurement and a residual-based
production authoring policy are required before choosing temporal behavior.

Fresh Discrete captures and a fixed 4x exported rung isolate those two issues.
They use the same corrected producer, binary SHA-256
`f95e778ca41b22d1569a3284a4ac5f8b066867b4f70178673912a712a2930430`, and fixed
three-sigma support for both flat source and candidate. The branching-4 package
exports a complete depth-3 antichain with 21,087 representatives from 84,348
originals. Export validation checks exact source-domain coverage, all 21 page
hashes, and distinct original/export identities; no source hashes are rewritten
to make the comparison pass.

| Icecream endpoint | Near PSNR / SSIM | Mid PSNR / SSIM | Held-out PSNR / SSIM |
| --- | --- | --- | --- |
| Fixed 4x rung | 26.18 / .9150 | 25.75 / .8755 | 25.14 / .8976 |
| Discrete .35 and .65 | Pixel-exact | 89.93 / .9999995 | 78.96 / .9999897 |

The fixed rung draws exactly 21,087 records in each view and fails the quality
gate. Discrete .35/.65 draw 84,278 near and 84,348 in the other views, preserving
quality without useful reduction. The fixed rung's alpha maximum errors are
.725, .690 and .569, so fitting color alone cannot correct its compositing
residual. The validated result is
`icecream-v5-authored-rung-screen.json`. Changing the selector alone cannot meet
this gate.

The subsequent bounded teacher fitter preserves record count and authenticated
owner support, with independent geometry derivatives and per-representative
feasibility backtracking. The v15 run completes 33 accepted updates within its
120-second training budget and all 12 evaluation views, reducing training loss
by 95.96%. The unchanged predeclared eight-pose 1080p GPU screen improves every
fitted view over the seed, but all eight still fail both 35 dB and .98 SSIM.
The six reused orbit evaluations reach 28.1245–29.7119 dB and .91916–.93220.
Each pair attests exactly 84,348 versus 21,087 submitted instances; flat-source
submission counts are not a post-frustum survivor ratio or a speedup result.
The [fitting report](lod_representative_fitting.md) preserves the complete RGB,
alpha, local-error, camera-domain and artifact evidence, including earlier
failed fits. This is diagnostic authoring progress, not an accepted replacement
representation or integration into the package hierarchy.

Garden's authored cameras were recovered using bounded authenticated HTTP ZIP
range reads from the original pretrained-model archive. The local PLY matches
the archive entry's exact size and CRC32 in addition to its recorded local
SHA-256. Camera metadata SHA-256 is
`60986440d8fdc73b9e8e2493a9b745016e20bf5275963f1c49ee64f8e616a391`.
Train IDs 24/104 and held-out IDs 0/4/8/12/16/20 are frozen before fitting.
The capture camera now supports authored roll through an explicit up vector.
At 1080p the square-pixel Bevy perspective uses a central vertical crop;
this is a teacher-relative engine comparison, not exact photographic intrinsics
or real-photo PSNR. Earlier arbitrary Garden poses are stress views, not the
representative camera set. Provenance and the split are saved under
`garden-authored-cameras/`.

On the five held-out views with two repeated stable cut/count observations,
Discrete .35 and .65 are identical at the 2M active ceiling: foreground PSNR
16.12–21.62 dB, SSIM .4896–.6427, alpha MAE .0697–.1237, and actual draw counts
393,524–489,303. Both runs report an unsatisfied active-budget target. Held-out
ID 0 is excluded because its final sample changes from the bootstrap to the
deep cut without a second settled observation. These are clear quality failures,
with incomplete held-out coverage; `garden-authored-heldout-screen.json`
preserves the exact inclusion checks. Nominally stationary interpolation
produced tiny inter-frame camera rounding differences in these older captures;
each teacher/candidate pair has an exactly matching matrix. The producer now
retains the exact endpoint when a segment is stationary.

The separately rendered `main` flat reference differs from branch flat by
24.31–26.73 dB / .7784–.8211 across the eight authored views. These are static
image-alignment diagnostics without main draw/timing attestation. The branch
changes the screen-space filter from .075 to .3 physical-pixel variance and adds
determinant-based opacity compensation. An isolated shader ablation restores
only main's filter variance and opacity factor: agreement improves from
26.26 dB / .8125 to 42.54 dB / .9901 on train 24, and from 25.84 dB / .8148 to
37.06 dB / .9905 on held-out 12. The same baseline binary rendered both variants;
the override WGSL is copied into the output. This attributes most discrepancy
to filtering, while remaining differences persist in the image interior and
are not claimed resolved. No production filter changed, and neither renderer
is treated as photographic truth.
The result is `garden-authored-main-branch-flat-alignment.json`.
The controlled experiment is `garden-main-filter-ablation-screen.json`.

An independent CPU authoring screen used pinned
[`@playcanvas/splat-transform` 3.3.3](https://github.com/playcanvas/splat-transform),
revision `d092ae9`, with `--gpu cpu --max-workers 2 --decimate 25%`. It reduced
Icecream from 84,348 to 21,087 records. Both source and output were rendered using
the same isolated `main` flat-render binary. Foreground PSNR was 25.45–25.76 dB,
SSIM .872–.908 across the three poses: **rejected against the locked 35 dB/.98
quality gate**. Counts in this external screen are file record counts, not GPU
readbacks. This is an external reducer comparison, not an integrated hierarchy.
The exact tool lockfile, hashes, settings and results are retained locally.

GPU timestamp brackets in stationary captures can surround cached/no-op
compaction or sorting. Image encoding also perturbs CPU frame cadence. Neither
these sparse timestamp samples nor process wall time constitutes the required
complete-frame speedup result. Dedicated moving, count-only and uninstrumented
cadence runs remain separate evidence tasks.

## Browser and lifecycle work

The actual WebGPU package harness compiles for `wasm32-unknown-unknown`, SH3,
with the production renderer and a fixed three-slot indirect readback ring.
It checks actual draw submission, adapter identity, load/move/unload/reload and
reservation retirement. The final v10 headed Chrome 152 runs passed all four
lifecycle phases at quality 0 and 1, with 35 and 359 GPU observations respectively,
zero dropped readbacks, zero mapping errors, and zero scene-owned reservations
after unload. The actual renderer adapter/device request chain attests NVIDIA
Blackwell, `is_fallback_adapter=false`. Wasm SHA-256 is
`b93d455434b6b5226286a09a197653e0a2794cec15a10893559ea95449a9a5d8`.
Quality 1 additionally reaches exactly 84,348 selected, candidate, hit, compacted
and drawn records in stationary, moving and reloaded phases; all 120 moving
observations retain that exact count. Quality 0 draws 165 proxies. These are
960x540 lifecycle and original-count checks, not image-quality or performance
acceptance. Headless Chrome returned no WebGPU adapter on this system. The
frozen browser build predates the subsequent continuous bootstrap-handoff
repair; that new path is not browser-qualified. See the exact settings, device
limits, observations and retained earlier runs in
[browser qualification](lod_browser_qualification.md).

The procedural 10M/100M fixtures contain genuinely distinct spatial records,
authenticate every encoded page, and serve pages lazily through the production
HTTP/package path. CPU prehash completed in 8.34s / 71.75s with peak RSS about
48 / 194 MiB respectively; these are preparation costs outside runtime timing.
The first 10M GPU attempt failed admission: a 136-page deep target exceeded its
64-page cache despite fitting the record/byte limits. The old bootstrap remained
complete, but this attempt produced no attested image and is not a scaling
result. Incremental complete-cut publication was integrated before rerun.
The next run exercised that path, then stalled with all 64 cache pages resident,
62 intermediate representatives, zero post-cull draws, and no pending I/O.
It was stopped after 177s with the last complete lifecycle observation saved in
`virtual-city/10m-capture-v2/runner_stop.json`. Breadth-first intermediate demand
had consumed the capacity needed for the next replacement. The repair budgets
the destination by physical residency, prioritizes visible refinement, and
reserves a concrete replacement footprint; this second attempt is also not
scaling evidence.

The next run reached real source leaves and slot reuse, then exposed a harness
bug: successfully persisted readbacks never incremented the completed counter,
so unload could not drain despite the package ledger reaching zero. That run
was stopped and the consumer acknowledgement was corrected.

Fresh v4 and v5 runs now complete all six nonzero-draw phases, authenticated
on-demand page access, eviction, zero-ledger unload and reload for both 10M and
100M sources. The 64-slot profile and a matched 256-slot profile both fail the
fixed-visible-work comparison: stationary draws are 10,115 versus 7,725. Increasing
the slots alone does not fix this. The balanced hierarchy changes with source
extent, and the active-budget selector chooses different source leaves.

A separate v6 exact-visible profile requested quality 1 with a 1,048,576 active
ceiling, 256 page slots, and the same 1 GiB CPU / 512 MiB GPU ledger limits.
Both runs complete their lifecycles and draw exactly 10,115 stationary instances.
Across 150 stable stationary samples each, instrumented frame-wall p95 is
2.678023 / 2.940904 ms (+9.82%), GPU-view p95 is .197632 / .210720 ms (+6.62%),
and package-update p95 is .015243 / .014897 ms. Pre-cull candidates still differ:
119,571 / 153,861. This is one instrumented paging screen, not an accepted
dynamic-LoD performance point. The first roughly .70-second nonzero attestation
is a synthetic proxy draw, not original-detail or useful-quality readiness.
The [large-scene results](lod_virtual_city_results.md) retain sample windows,
source-page intersections, motion gaps, ledger peaks, RSS and artifact identities.

These historical v6 runs use renderer binary SHA-256
`a19710993ff351fc75b41f63aa8e44611f9048fa268f62ace02ecfefe81f16f1`, before the
fragment-density correction. Old virtual run IDs omit settings; those artifacts
must be identified by the complete identity/settings hash and directory, never
concatenated by stamp alone. New runs include settings in the run ID and separate
payload-generator identity from capture orchestration.

The final v11 pair uses the corrected fragment density and frame-associated CPU
work counters, executable SHA-256
`327dbb2c4e7d096062f7bcc98816395db9d9e9331be49b9f5ac56354aff7e20d`.
Both runs again contain 150 stable, attested 10,115-instance stationary draws.
All stationary observations explicitly record zero canonical visits, destination
compilations and cache lookups: settled planning is bypassed. Instrumented
frame-wall p95 is 2.926844/2.955962 ms (+0.99%), but GPU-view p95 grows 10.92%.
Movement visits at most 375/813 hierarchy nodes while full package-update p95 is
1.801497/2.851226 ms, exceeding the 1 ms CPU target. Existing scopes leave a
substantial part of that update tail unattributed. Rapid return still has zero
draw gaps and ends at proxy coverage with pending demand. First sampled complete
proxy frame starts are 1.817539/1.628594 seconds after renderer initialization,
not first useful source imagery. The detailed report retains every profile;
none of the favorable single-window timing changes establishes accepted scaling.

The subsequent 100M v16/v17 diagnostic moves native HTTP payload checksum work
into the existing bounded worker. Inclusive polling p95 drops from .850139 to
.020956 ms during movement and 1.623471 to .020159 ms during eviction. Full
package-update p95 remains 1.972186/2.015716 ms, above the 1 ms target.
Movement frame-wall p95 increases 6.61% with changed rendered work. Both
zero-ledger lifecycles pass; this does not establish an overall speedup.
The [CPU telemetry report](lod_cpu_telemetry.md) records exact populations,
identities, scope nesting and comparison limits.

Capture startup telemetry now distinguishes run entry (before config/source
loading and initialization) from the first renderer-loop execution with a
RenderDevice. Each records the earliest sampled, nonzero, attested frame and
readback-observed upper bound, with a separate complete-package field. The
renderer-loop clock is an initialization proxy, and sampling is bounded; neither
clock is an exact first-draw timestamp or proof of a useful-quality image.

The shared ledger now admits CPU morph construction scratch and retains a lease
with each immutable batch; Arc reuse is charged once and final render ownership
controls release. Metadata admission includes morph run-vector capacities.
The remaining exclusions are explicit: initial AssetServer manifest parsing,
debug overlays, ordinary flat-cloud storage, capture buffers under their own
limit, render targets, wgpu staging/driver pools, allocator overhead, and fixture
server state. Declared package/compaction reservations are bounded; this ledger
does not claim to cap process RSS or total adapter memory.

## Acceptance status

The complete roadmap remains unqualified. Native 100M and actual WebGPU lifecycle
checks now provide execution evidence for paging, retirement and recovery;
representation quality, complete-frame gains, total physical memory, network
latency and broader hardware remain open gates. GPU hierarchy traversal, global
cross-cloud ordering and temporal budget control are not implemented. Conditional
raster/compression work must earn its place through profiles. The
[implementation status](lod_implementation_status.md) maps each phase to concrete
work and outstanding requirements. Failed quality gates are not marked complete.
