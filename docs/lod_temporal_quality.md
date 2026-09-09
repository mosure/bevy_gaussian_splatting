# Poland temporal rendering qualification

September 9 update: **full-scene visual qualification remains failed**. The
[partial blocker closeout](#blocker-closeout-september-9-partial) below records
new support and streaming fixes, physical omission after complete-cohort
selection, and final matched-camera image evidence. Return images recover exactly;
local motion differences and aerial proxy quality still prevent production acceptance.
The [closeout validation](../target/lod-roadmap/2026-09-08/blocker-closeout/validation.json)
pins the final renderer and checks.

September 8 baseline. The held church image is substantially improved,
and the final sampled held images are stable. The completed 16M route and 1080p
baseline fail streaming qualification. After the streaming fixes, the 8M route
meets the measured GPU timing target and recovers its initial image. **Full-scene
visual qualification fails:** settled aerial proxies remain blurry against the
Original reference, with further error from interpolation. This is bounded native
evidence, not production qualification.
The [final validation summary](../target/lod-roadmap/2026-09-08/temporal-quality/validation.json)
pins the renderer, measured gates and validation logs.
Historical failed cuts and fitting experiments remain documented in
[Poland quality results](lod_poland_quality_results.md).

## Captured configuration and mechanism

The [spatial configuration](../target/lod-roadmap/2026-09-08/temporal-quality/final-spatial-held-config.json)
and [discrete control](../target/lod-roadmap/2026-09-08/temporal-quality/final-discrete-held-config.json)
use the same 106,447,647-record SH0 v17 package, calibrated camera 0, 960×638,
near 0.1, quality 0.95, 16M selected/projected and 24M resident record ceilings.
The camera is held for 900 warmup frames and 120 measurement frames; images
are requested every 12 frames with a 33.333ms frame period. The device is NVIDIA
RTX PRO 6000 Blackwell, Vulkan, driver 610.43.02. Both runs identify renderer
SHA-256 `c8a40df64210c21f700e3006b121ef2cf4aecabd6eb011a0d6c807c37fc5d97d`
and manifest `54da07e3cdc1a8e58c00c9691cd4bc7cfdc0e30b6eaffd6031dc7ed9d96a176f`.

The ordered spatial path evaluates finite current-view refinement scores,
constrains child scores by ancestry and derives a cutoff from the first
budget-excluded candidate. An eligible adjacent edge uses
`weight = clamp((score - cutoff) / cutoff, 0, 1)`. No exclusion produces the
actual child endpoint; ties remain actual parents. Authenticated monotone
correspondence maps child records to parents. Projection interpolates positions
and covariances and evaluates the parent/child density contributions in the
current camera. Elapsed time and delayed draw acknowledgements do not set
weights. Correspondence describes transition ownership, not exact reducer
lineage for each representative.

Complete resident parent fallback, bounded sibling demand and publication
leases preserve a drawable cut while pages arrive. Mapping, candidate-graph and
transition-band limits have explicit categorical fallback diagnostics. This
profile raises spatial limits to 16,384 edges, 16M participating records and
128MiB mapping bytes; it does not qualify the smaller defaults for this scene.
These captures precede the cache-index and obsolete-request cancellation fixes
described below.

The shared OBB projection helper also received a numerical fix: lengths and
orientation now use one stable eigensystem. Cancellation in the previous
eigenvector formula could rotate a diagonal support box after a tiny camera
change. The [GPU regression](../target/lod-roadmap/2026-09-08/temporal-quality/gpu-ordered-obb-fix.log)
passes in 8.39s, including the tight infinitesimal fractional-projection check,
actual coarse/Original endpoint parity, padded 2D dispatch and small perspective
full-frame/half-tile parity. Its orthographic motion preserves score ratios;
that check alone does not establish continuity while cutoff weights change.

The final [eight-pose perspective regression](../target/lod-roadmap/2026-09-08/temporal-quality/perspective-regression/receipt.json)
extends the same small integration app and passes in 11.31s. It reads actual GPU
weights: a 0.0001 depth change moves the weight from 0.49999994 to 0.49992877,
with a maximum image change of one RGBA8 level. Returning to the initial pose
reproduces the image exactly. The parent endpoint differs by at most two levels:
split optical depth uses two separately rounded alpha blends, while the actual
parent uses one. Descriptor/header copies share a producer submission; screenshots
use the unchanged held camera after acknowledgement, not an asserted identical
submission to those copies.

The child endpoint exposes hard support clipping: low-intensity tail pixels jump
by up to 14 encoded levels; brighter pixels change by at most one. The test keeps
a four-level gate outside the opacity-one three-sigma tail bound. Only pixels
whose before/after RGB values both remain below that analytical bound, with
unchanged alpha, receive the tail allowance. Earlier stricter failed assertions
are preserved. This checks bounded endpoint behavior, not continuity at clipped
tails; it does not change the frozen Poland renderer or qualify arbitrary motion.

## Native viewer profile

From the repository root, this opens camera 0 paused with the measured scene,
record, memory and spatial limits. The package and camera file must already exist:

```sh
BEVY_ASSET_ROOT="$PWD" cargo run --release --locked --no-default-features \
  --features 'planar lod_render sh0 viewer io_flexbuffers io_ply file_asset' \
  --bin bevy_gaussian_splatting -- \
  --input-lod target/lod-packages/poland-sh0-v17/scene.gsplatlod \
  --lod-max-manifest-bytes 268435456 \
  --camera-path 'assets/Jastrzębia_Góra_camera_path.json' \
  --camera-path-index 0 --camera-path-fps 0 --width 960 --height 638 \
  --camera-controller flycam --camera-speed 200 --match-ground-plane \
  --lod-quality 0.95 --lod-max-active-gaussians 16000000 \
  --lod-max-resident-gaussians 24000000 \
  --lod-max-resident-bytes 4294967296 \
  --lod-max-cpu-bytes 16107127360 --lod-max-gpu-bytes 8589934592 \
  --lod-max-concurrent-requests 64 --lod-max-page-requests 4096 \
  --global-order --lod-gpu-traversal \
  --global-order-max-gaussians 16000000 \
  --global-order-max-gpu-bytes 4294967296 \
  --lod-spatial-transitions --lod-max-transition-nodes 16384 \
  --lod-max-transition-records 16000000 --lod-max-mapping-bytes 134217728
```

Spatial transitions require the explicit opt-in and raised limits above. The
viewer derives 23,437 resident pages from `floor(24,000,000 / 1,024)`; GPU
traversal defaults to 16,384 frontier nodes, 262,144 visits and 256MiB scratch.
The command explicitly raises the default page-reference limit from **1,024**
to the capture's **4,096**. The same setting is available as
`lod_max_page_requests` in query strings and JSON configs. This command is not
a claim that the 24M residency profile sustains route motion; the linked capture
config also specifies its measurement cadence and readback instrumentation.

For bounded diagnostic review, the **8M profile passes the measured GPU timing
target but fails visual qualification**: change only `--lod-max-active-gaussians` and
`--global-order-max-gaussians` above to `8000000`. Keep the 24M residency and
16M transition-record limits, matching the
[8M held configuration](../target/lod-roadmap/2026-09-08/temporal-quality/spatial-8m-held-config.json).
This is a reproducible diagnostic profile, not a production recommendation or a
new default.

The authored pose stays intact until manual input. Press Tab to capture/release
flycam input; see [camera controls](camera_paths.md). Ground matching affects
navigation on takeover and does not rotate the scene or change its calibration.

## Held measurements

The [spatial analysis](../target/lod-roadmap/2026-09-08/temporal-quality/final-spatial-held-analysis.json)
and [discrete analysis](../target/lod-roadmap/2026-09-08/temporal-quality/final-discrete-held-analysis.json)
contain ten held samples, frames 900–1008. GPU times below use those same ten
timestamp samples. The p99 values are derived from their capture rows with
NumPy's linear percentile interpolation; ten samples do not qualify workload
tails. Each pass is summarized independently, so percentile columns need not sum.

| GPU milliseconds | Spatial p50 | p95 | p99 | Discrete p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hierarchy | 1.070208 | 2.5947568 | 2.61495136 | 3.287712 | 7.45688 | 7.4777312 |
| Ordered backend | 12.05272 | 17.0096192 | 17.04961664 | 11.160976 | 18.0804832 | 18.12681664 |
| Postprocess | 0.112384 | 0.15024 | 0.1518528 | 0.15048 | 0.1564544 | 0.15880448 |
| View total | 13.30608 | 19.6159984 | 19.65135328 | 16.196608 | 24.0152304 | 25.39519968 |

These are instrumented headless GPU timings, not interactive FPS or a controlled
performance attribution to morphing. The modes use different traversal and
residency work, and the paced loop includes CPU work and capture encoding.

| Final frame 1008 | Spatial | Discrete |
| --- | ---: | ---: |
| Residency generation | 346 | 282 |
| Selected nodes | 15,622 | 15,622 |
| Selected records | 15,996,580 | 15,996,608 |
| Projected, compacted and drawn | 10,462,156 | 10,676,809 |
| Actual / required fractional edges | 703 / 703 | 0 / 0 |
| Actual / required fractional records | 5,758,944 / 5,758,944 | 0 / 0 |
| Spatial flags | 0 | 0 |
| Resident / snapshot pages | 21,954 / 21,954 | 17,858 / 17,858 |
| Shared mapping bytes | 62,728,020 | 0 |
| Owned GPU capacity reservations | 3,613,506,548 bytes | 3,036,848,460 bytes |
| Owned CPU capacity reservations | 7,961,710,860 bytes | 7,897,578,648 bytes |

The spatial [same-submission evidence](../target/lod-roadmap/2026-09-08/temporal-quality/final-spatial-held/submission_evidence.jsonl)
attests both requested and actual spatial pipeline use. The discrete
[receipts](../target/lod-roadmap/2026-09-08/temporal-quality/final-discrete-held/submission_evidence.jsonl)
attest neither. All nine adjacent held comparisons have zero changed cut nodes
and zero changed RGB pixels. Final package snapshots have one acknowledged view,
zero queued/in-flight/capacity-blocked requests, zero pending-publication pages
and zero terminal failures.

Spatial traversal flags are **10**: record-limited plus request overflow, with
4,096 raw page references. Discrete traversal flags are **2**, record-limited,
with zero requests. Both cuts cover the complete canonical source domain and
report no cutoff-unavailable fallback. Stable held output does not remove the
spatial request-budget limitation or prove motion streaming can keep up.
The owned-capacity ledger is neither measured RSS nor device VRAM and overlaps
private allocation counters; the complete memory audit remains open.

Both [spatial status](../target/lod-roadmap/2026-09-08/temporal-quality/final-spatial-held/capture_status.json)
and [discrete status](../target/lod-roadmap/2026-09-08/temporal-quality/final-discrete-held/capture_status.json)
account for 86 requests: 84 attested drawable captures and two startup missing
drawables, with no unresolved requests, ring drops or mapping errors. They still
set `attested_counts_complete`, `memory_audit_complete` and `release_qualified`
to false. Sampling every twelfth frame does not establish every-frame cadence.

## Current reference comparison

The [attested tiled Original comparison](../target/lod-roadmap/2026-09-08/temporal-quality/original-tiled-camera0-vs-final-spatial-held.json)
reports **65.837600789 dB PSNR / 0.9996114118 SSIM** for the final spatial image
against five physical camera-0 tiles. Metrics use linearized RGBA8 and are
teacher-relative, not photographic ground truth. Alpha MAE is 0.000222804,
maximum alpha error 0.125490196 and silhouette IoU 1.0. All tiles have complete
canonical coverage, zero visible proxy nodes under the conservative AABB audit,
zero traversal flags and no missing demand at the selected reference frame.

The reference change does not explain away the historical defect. Independent
comparisons using the existing image metric tool find the
[historical raw subset](../target/lod-roadmap/2026-09-08/poland/reference-0-capture/frame-00000206.png)
and [new tiled reference](../target/lod-roadmap/2026-09-08/temporal-quality/original-tiled-camera0.png)
agree at 74.322614207 dB / 0.9999131328 SSIM. The
[historical church cut](../target/lod-roadmap/2026-09-08/poland-quality/church-cut-capture/frame-00002100.png)
still scores only 18.549113903 dB / 0.2766756274 against the new reference.
Current selection replaces the problematic owners 8484 and 8435 despite the
unchanged aggregate count of 15,622 selected nodes. The package was not refitted
or rebuilt for these renderer measurements.

Physical ray parity and the small GPU tile regression are established, but
whole-Poland full-frame/tile contributor parity remains unproven. Per-tile
frustum rejection can change which projected supports contribute. Tiled timings
cannot substitute for full-frame Original rendering cost; the actual wgpu
storage binding limit is 2,147,483,644 bytes and startup now preflights impossible
single-buffer profiles. No broad scene-quality acceptance follows from one pose.

## Completed 16M route and 1080p baseline

The [960×638 image route](../target/lod-roadmap/2026-09-08/temporal-quality/final-spatial-motion-analysis.json)
contains 30 held, 599 forward, 599 reverse and 90 returned attested frames.
The [1920×1080 count route](../target/lod-roadmap/2026-09-08/temporal-quality/final-spatial-1080-counts-analysis.json)
contains 30 held, 599 forward and 90 returned attested frames. Both use the
16M/24M limits and renderer `c8a40df6…` above, before the streaming fixes.
All requested outcomes terminate: 1,321 and 722 completed captures respectively,
including two startup missing drawables each; 1,319 and 720 have drawable
attestations. There are no unresolved requests, ring drops or mapping errors.

| Baseline scenario | GPU total p50 / p95 / p99, ms | Package CPU p50 / p95 / p99, ms | Frame wall p50 / p95 / p99, ms |
| --- | ---: | ---: | ---: |
| 960×638 forward | 9.853 / 17.120 / 19.522 | 27.478 / 49.558 / 54.798 | 49.514 / 73.701 / 79.262 |
| 960×638 reverse | 11.028 / 17.922 / 20.500 | 31.801 / 43.790 / 47.505 | 55.842 / 70.833 / 78.109 |
| 960×638 returned | 9.907 / 16.215 / 18.562 | 30.080 / 43.501 / 48.099 | 54.328 / 72.372 / 76.055 |
| 1080p held | 19.696 / 23.752 / 23.817 | 5.188 / 6.437 / 6.470 | 33.394 / 33.483 / 33.494 |
| 1080p forward | 11.372 / 18.097 / 20.810 | 33.672 / 37.636 / 41.871 | 44.656 / 50.545 / 53.912 |
| 1080p returned | 11.703 / 18.202 / 19.861 | 33.450 / 47.913 / 56.086 | 45.211 / 65.633 / 71.612 |

These are the analyzer's sampled percentiles, not interactive FPS. The 960×638
reverse phase reaches 170.004ms frame wall and 126.005ms package CPU maxima.
Both routes reach all 23,437 resident slots and continue with 64 requests in
flight. Returned queue medians are 3,935 and 3,901; neither settles during the
90-frame return window. Terminal-failure and capacity-blocked gauges remain zero:
those gauges do not establish that residency keeps up.

The initial 29 equal-pose image pairs are identical. After the route, **all 89
adjacent returned cuts differ**, and only seven corresponding image pairs are
identical; maximum pixel delta reaches 54 RGBA8 levels. Across 598 matched
forward/reverse poses, no cut or image pair is identical; RGB MAE p50/p95 is
4.839/12.360 levels, with maximum pixel delta 235. These equal-pose differences
include residency evolution. Different-pose image changes are not independently
classified as LoD flicker. The final returned image still scores
[65.462491090 dB / 0.9995170016 SSIM](../target/lod-roadmap/2026-09-08/temporal-quality/original-tiled-camera0-vs-final-spatial-motion.json)
against the tiled teacher; one good final image does not pass temporal recovery.

All recorded spatial flags are zero and actual/required transition counts fit
the configured band; no cutoff-unavailable fallback occurs. Request-overflow
traversal flags recur during motion. The 1080p run has no images and no reverse
phase; its changing returned cuts and persistent demand establish a streaming
failure, not a measured pixel-flicker magnitude. Counts and completed requests
alone do not qualify image continuity, memory or display cadence.

## 8M held candidate and streaming fixes

The [8M held analysis](../target/lod-roadmap/2026-09-08/temporal-quality/spatial-8m-held-analysis.json)
selects **7,999,830** records and actually projects/draws **3,599,410** at frame
1008, generation 180. It retains 11,299 resident/snapshot pages, with zero queued,
in-flight, blocked or pending-publication work and no terminal failure. All nine
adjacent held comparisons have zero cut and pixel deltas. Actual and required
spatial participation both equal **138 edges / 1,130,496 records**. Spatial flag
**4** records near-plane bypass, so this is not an all-fractional-path result;
traversal flag **2** indicates the record limit without request overflow.

The [tiled-reference comparison](../target/lod-roadmap/2026-09-08/temporal-quality/original-tiled-camera0-vs-spatial-8m-held.json)
is **57.801555725 dB PSNR / 0.9981253439 SSIM**. Ten held GPU samples give view
total p50/p95 **8.011520 / 11.227912ms** and ordered backend
**6.313712 / 9.509570ms**. The same teacher, memory and instrumented-timing
limitations apply. This hold alone does not establish route or 1080p image quality.

Three concrete changes are implemented for the next measurements:

- Cache eviction maintains an index of unpinned pages and cached totals, avoiding
  repeated whole-cache sorting and accounting scans while preserving deterministic
  LRU and transactional admission.
- Retargeted GPU demand cancels obsolete queued work before starting new I/O,
  preserving current priorities and avoiding fetch/decode work immediately
  discarded by end-of-frame cancellation.
- `--lod-max-page-requests` exposes the existing traversal limit consistently in
  CLI, query and JSON configuration; the reproduction profile explicitly uses
  4,096 instead of the unchanged 1,024 default.

The [focused CPU checks](../target/lod-roadmap/2026-09-08/temporal-quality/cpu-streaming-final.json)
pass: nine cache tests, one demand-lifecycle regression and two CLI regressions.
The frozen `capture-streaming-final` executable is
`faf0fdedec847b55a16e6964f9a6e35b9c17975478dacd9a23f2b7ff1d494e23`.
The following measurements use that executable.

## Measurements after the streaming fixes

The [matched 16M 1080p count route](../target/lod-roadmap/2026-09-08/temporal-quality/fixed-spatial-1080-counts-analysis.json)
keeps the baseline camera schedule and limits. Forward page-commit CPU p50/p95
falls from **17.6857/23.4251ms to 0.0701/0.1112ms**, and package-update CPU
p50/p95 falls from **33.6717/37.6364ms to 17.5909/24.9724ms**. Frame-wall p95
falls from 50.545 to 37.943ms. This fixes substantial CPU overhead, but the 16M
profile still fails recovery: all 89 adjacent returned cuts differ, resident
slots remain full, and the returned queue median is 3,916 with 64 in flight.

The [8M 1080p count route](../target/lod-roadmap/2026-09-08/temporal-quality/fixed-spatial-8m-1080-counts-analysis.json)
uses 8M selected/projected records, 24M residency, unchanged 16M spatial-record
limits and 4,096 page references. It records 30 held, 599 forward, 599 reverse
and **300 returned** frames; the longer return hold differs from the 16M run.

| Fixed 8M scenario | GPU total p50 / p95 / p99, ms | Package CPU p95, ms | Frame wall p95, ms |
| --- | ---: | ---: | ---: |
| Held | 7.820416 / 8.262880 / 8.824384 | 3.133070 | 33.561589 |
| Forward | 9.928416 / 13.248096 / 13.901760 | 18.816314 | 33.498452 |
| Reverse | 9.141312 / 15.181440 / 17.199552 | 18.637111 | 33.484078 |
| Returned | 7.894592 / 12.926368 / 13.467648 | 3.808305 | 33.523280 |

This fixed profile passes the **GPU timing portion** of the proposed p95 ≤16.7ms
and p99 ≤25ms target. It does not pass a whole-frame 60 FPS or the 1ms LoD CPU
objective: the headless loop is paced at 33.333ms, and CPU motion work remains
substantial. At 1080p the held cut selects 7,999,830 and draws 3,593,306 records.
After return, generation advances from 1369 to 1376, the final cut matches the
initial cut, and 296 of 299 adjacent returned cuts are identical. Returned
queue/in-flight medians are zero; the cache retains all 23,437 resident slots.

Near-plane spatial bypass flag **4** occurs in all held/returned samples and
425/599 forward plus 449/599 reverse samples. No cutoff-unavailable fallback is
reported. These receipts prove bounded rendered work and cut recovery for this
schedule, not continuous interpolation everywhere. Count-only results do not
establish pixel stability, photographic quality, interactive latency or complete
memory qualification.

## Completed image route and aerial quality rejection

The [8M 1080p image route](../target/lod-roadmap/2026-09-08/temporal-quality/fixed-spatial-8m-1080-images-analysis.json)
captures every fourth frame. All **74 adjacent returned image pairs are exactly
identical**, and both the first and last return images exactly match the initial
held image. Of 149 equal-pose forward/reverse pairs, ten are identical and 139
differ. Mean RGB error across each image has median **0.082256/255** and maximum
**1.990665/255**, while individual RGBA8 pixel differences reach **209** levels.
This demonstrates sampled return-image recovery; it does not establish universally
seamless motion or what happened between captured frames.

Visual review of [camera 271](../target/lod-roadmap/2026-09-08/temporal-quality/fixed-spatial-8m-1080-images/frame-00001200.png)
and [camera 471](../target/lod-roadmap/2026-09-08/temporal-quality/fixed-spatial-8m-1080-images/frame-00001400.png)
reveals large blurred patches. A separate
[settled camera-471 capture](../target/lod-roadmap/2026-09-08/temporal-quality/fixed-spatial-8m-aerial-held-analysis.json)
remains blurry at generation 207, with 7,998,024 selected and 3,833,277 drawn
records, 13,027 resident/snapshot pages and no queued, in-flight or pending
publication work. Its sampled held cuts and images are stable.

The [Original center crop](../target/lod-roadmap/2026-09-08/temporal-quality/aerial-original-center/frame-00001008.png)
is sharp. It covers physical pixels `[640,360]` through a 640×360 region of the
same calibrated 1920×1080 view. Its
[cut audit](../target/lod-roadmap/2026-09-08/temporal-quality/aerial-original-center-analysis.json)
has complete canonical source coverage, zero visible proxy nodes, zero traversal
flags and no pending requests; it selects 9,146,710 and draws 7,155,125 records.

| Camera-471 center crop vs Original | PSNR | SSIM |
| --- | ---: | ---: |
| [Settled spatial cut](../target/lod-roadmap/2026-09-08/temporal-quality/aerial-original-center-vs-fixed-spatial-8m-aerial-held.json) | 20.144704 dB | 0.4857911 |
| [Same cut with the whole transition band disabled](../target/lod-roadmap/2026-09-08/temporal-quality/aerial-original-center-vs-fixed-spatial-8m-aerial-discrete-cut.json) | 20.727551 dB | 0.5418875 |

The control sets `max_transition_records=1`, triggering categorical fallback;
the actual selected node/output pairs exactly match the spatial capture.
Both runs are settled at the same generation. Thus the static proxy cut itself
fails quality without late page arrival or interpolation. Spatial interpolation
adds approximately **0.583 dB** of degradation for this crop. Whole-frame/crop
contributor parity remains unproven, and these are teacher-relative image metrics,
not photographic ground truth. Within that evidence boundary, the clear Original
crop and failed settled controls reject full-scene visual qualification.

## Blocker closeout (September 9; partial)

Streaming now shares one runtime page lease across overlapping snapshots, unions
camera demand, and builds fixed-slot snapshots with bounded occupancy validation.
Required page priority survives preprocessing and staging. View-scoped omission
keeps the complete resident logical source antichain while emitting zero physical
records outside conservative world support. Zero-count nodes still require their
complete cohort and page pins; omission does not skip residency requirements.
[Capture receipts](lod_point_runtime_capture.md) distinguish these
counts and preserve the exact producer frustum and omission parameters.

The shared C1 support taper preserves the inner `q <= 8` density and reaches zero
at `q = 9`. It changes the renderer/teacher kernel, so earlier image metrics do
not automatically qualify it. The
[strict eight-pose GPU seam comparison](../target/lod-roadmap/2026-09-08/blocker-closeout/smooth-support-gpu-transition-comparison.json)
passes its unchanged gates: tiny fractional step **1/4**, parent endpoint **2/2**,
child endpoint **1/4**, returned pose **0/0** maximum RGBA8 difference/allowance.
This is a fully resident tiny fixture, not a full-scene continuity result.
[15 focused CPU checks](../target/lod-roadmap/2026-09-08/blocker-closeout/cpu-focused-recovery.json)
and the [ordered](../target/lod-roadmap/2026-09-08/blocker-closeout/gpu-ordered-recovery.log),
[traversal](../target/lod-roadmap/2026-09-08/blocker-closeout/gpu-traversal-recovery.log),
[package](../target/lod-roadmap/2026-09-08/blocker-closeout/gpu-package-recovery.log),
and [GPS](../target/lod-roadmap/2026-09-08/blocker-closeout/gpu-points-recovery.log)
fixtures pass on the recovery build. The final complete-cohort correction is
validated separately below.

The [initial omission route](../target/lod-roadmap/2026-09-08/blocker-closeout/streaming-cpu-comparison-final.json)
is **rejected**: a twelve-frame residency cycle prevented all 300 return samples
from matching the initial 3,593,306 drawn records. Its lower timings are not a
performance win. After demand-lease and exact-capacity pressure fixes, the
[30 Hz recovery route](../target/lod-roadmap/2026-09-08/blocker-closeout/motion-recovery-timing.json)
matches that count in 281/300 return samples and ends with no pending work.
Package CPU p95 is **11.60/12.75 ms** forward/reverse; GPU view p95 is
**14.49/15.15 ms**. These are changing cuts, not equal-image throughput.
Forward median drawn records remain **1,867,157**, versus **2,747,567** in the
[pre-omission comparator](../target/lod-roadmap/2026-09-08/blocker-closeout/streaming-cpu-comparison-v1.json).
Motion detail therefore remains a regression to investigate; count recovery is
not image recovery.

The subsequent [navigation image run](../target/lod-roadmap/2026-09-08/blocker-closeout/navigation-images-analysis.json)
is also **rejected**. At camera 27, the reverse traversal's coarse fallback
(owner 8438) produces nearly whole-image haze: equal-pose forward/reverse RGB8
MAE reaches **16.409**, versus the previous maximum **1.991**. Returned adjacent
images differ by at most **2** RGBA8 levels and the final returned image is exact,
but those endpoints do not excuse the worse motion images. The
[count replay](../target/lod-roadmap/2026-09-08/blocker-closeout/navigation-motion-comparison.json)
recovers its held draw count in 291/300 returned samples; this is not visual
acceptance. The missing-node navigation policy is withdrawn. Complete logical
cohort residency is restored before physical-only omission; the final matched
route and image results follow below. The
[local aerial fit](lod_poland_quality_results.md#blocker-closeout-local-aerial-authoring)
has not been promoted into the package.

## Final complete-cohort correction

The final renderer (`bd8613a91dbcfdc672cdab2c4d2d1dd79c524c048eac40e510d52a9fc856e3f7`)
restores complete logical record charging, root requirements, sibling residency
and page demand before physical omission. Classification runs after spatial
annotation; omitted endpoint records still retain their logical pages. The failed
virtual-navigation suffix and its prefix API were removed. Omission saves
projection/expansion work, not residency or logical selection budget.

The [final image replay](../target/lod-roadmap/2026-09-08/blocker-closeout/cohort-images-analysis.json)
uses the same 30 Hz paced 1080p route and 8M/24M limits. All 386 requests complete,
including 383 attested draws and three startup observations, with no dropped
readbacks, mapping errors or unresolved requests. Both initial-to-returned image
comparisons and **all 74 adjacent returned-image pairs are pixel-identical**.
Logical offscreen cuts can still differ without changing the image. The
[near-camera regression](../target/lod-roadmap/2026-09-08/blocker-closeout/frame27-residency-diagnosis.json)
is removed: camera-27 forward/reverse RGB8 MAE falls from **16.4093** in the rejected
navigation run to **0.002578**, matching the historical **0.002571** scale.

Local motion differences remain: **138/149** equal-camera forward/reverse image
pairs differ. RGB8 MAE is median **0.06782**, p95 **1.71654**, maximum **1.99647**;
the largest individual RGBA8 difference is **213**. The corresponding historical
maximum mean error was **1.99066**. Residency changes and poor proxy endpoints
remain visible; this is recovery of the baseline motion behavior, not seamless
streaming qualification.

The [matched route comparison](../target/lod-roadmap/2026-09-08/blocker-closeout/cohort-motion-comparison.json)
records the same camera matrices and matching median post-projection draw counts:

| Image-route samples | Previous package CPU p95 | Final package CPU p95 | Final GPU view p95 | Median drawn, both runs |
| --- | ---: | ---: | ---: | ---: |
| Forward, 150 | 18.36 ms | 13.45 ms | 14.75 ms | 2,753,507 |
| Reverse, 149 | 17.60 ms | 15.18 ms | 15.19 ms | 2,866,508 |

Only 75/150 forward and 85/149 reverse samples have exactly matching drawn counts;
asynchronous cuts and the support kernel differ. These single instrumented runs
are not equal-image throughput or interactive FPS qualification. The held view
expands **4,593,269** records rather than **7,999,830** (42.58% fewer), with the same
**3,593,306** post-projection draws. Full-frame timing includes CPU and presentation
work beyond the GPU timestamps.

[All 17 focused CPU checks](../target/lod-roadmap/2026-09-08/blocker-closeout/cpu-focused-cohort.json),
the final [traversal](../target/lod-roadmap/2026-09-08/blocker-closeout/gpu-traversal-cohort.log),
[ordered](../target/lod-roadmap/2026-09-08/blocker-closeout/gpu-ordered-cohort.log),
[GPS](../target/lod-roadmap/2026-09-08/blocker-closeout/gpu-points-cohort.log) and
[package](../target/lod-roadmap/2026-09-08/blocker-closeout/gpu-package-cohort.log) fixtures pass.
[SH0/SH3 Clippy and the viewer/capture builds](../target/lod-roadmap/2026-09-08/blocker-closeout/build-cohort-results.json)
and formatting pass. No broad covariance test suite was repeated.

The [final aerial capture](../target/lod-roadmap/2026-09-08/blocker-closeout/fixed-spatial-8m-aerial-held-cohort-analysis.json)
has nine identical adjacent held-image comparisons, yet remains visibly hazy.
The [matched-renderer crop diagnostic](../target/lod-roadmap/2026-09-08/blocker-closeout/aerial-original-center-vs-fixed-spatial-8m-aerial-held-cohort.json)
reports **20.13162 dB / 0.485724 SSIM**. Its strict Original reference gate is
**unqualified**: physically expanded proxy ranges outside the crop remain in the
reference cut. The independent
[prior equivalent-image crop audit](../target/lod-roadmap/2026-09-08/blocker-closeout/aerial-original-navigation-proxy-crop-audit.json)
excludes 3,089 ranges conservatively but leaves nine uncertain; no contributor
parity or Original promotion is inferred from the draw count. This diagnostic
does not replace the historical reference gate or qualify the representation.

## Remaining qualification

- **Representation and interpolation quality:** failed at the settled aerial
  crop. Improve proxy fidelity and measure the additional spatial error before
  promoting the renderer; the historical local fits remain unqualified.
- **Route/dense motion/reversal:** 16M still fails recovery under 24M residency
  pressure. The 8M route recovers its sampled image but has local equal-pose
  differences up to 213 levels on the final renderer; seamless motion is not
  established. The smooth-support fixture passes its strict gate and complete
  cohort residency removes the introduced motion-haze regression. Dense motion,
  cold views and larger-budget recovery remain unqualified.
- **1080p:** fixed 8M meets the measured GPU timing target. Full-scene visual
  qualification fails, and whole-frame/CPU responsiveness targets remain open.
- **Automatic budgets and broader deployment:** matched quality/performance
  targets, controller response, complete memory accounting and native/browser
  hardware coverage remain open. GPS does not inherit ordered spatial acceptance.
- Preserve rejected routes, historical fits and held captures as separate
  evidence; acceptance requires matched detail and images on the final renderer.
