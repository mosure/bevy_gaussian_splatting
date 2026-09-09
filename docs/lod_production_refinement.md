# Production refinement: 2026-09-09

`release_qualified` remains **false**. This follow-up records concrete correctness
and maintenance improvements after the [temporal qualification](lod_temporal_quality.md).
It does not resolve aerial representative quality, GPS quality, whole-frame
responsiveness, or platform acceptance.

The subsequent [near-camera follow-up](lod_near_camera.md) adds support-based
near clipping and a smooth approach to the spatial child endpoint. Focused SH0
and SH3 checks pass; the substantial Poland brightness change is still being
reviewed. The measurements below retain their original renderer
identities and do not qualify that later policy.

The implementation now:

- Uses signed forward camera depth consistently for flat radix, CPU sorting,
  LoD compaction and global quad ordering. Camera rotation invalidates cached
  ordering; flat CPU sorting also reacts to cloud transforms, same-handle asset
  changes and sort-mode changes. Forced CPU sorts refresh the camera even during
  the throttle interval; removed sources no longer repeatedly sort unrelated
  resident clouds. Hidden records retain their invalid sentinel with 16-, 24-
  and 32-bit radix keys. CPU backends retain their configured throttling; these
  changes do not make them suitable for current-frame large-scene ordering.
- Retains active transition-parent pages independently of whether their own
  records survive view culling. Complete child cohorts, source ownership and
  snapshot leases remain required; physical omission does not remove residency
  dependencies.
- Reuses unchanged demand footprints, ancestor closures and publication state.
  Empty request queues avoid reprioritization. Residency changes still trigger
  checked admission, publication and lease retirement.
- Records the hierarchy inputs actually consumed by each renderer submission:
  cloud/source identity, residency generation, allocation generation and producer
  submission. The capture checks these against the traversal output before
  copying counters with the image. Renderer and traversal counters are independent;
  numerical equality is not their association proof.
- Supports explicit view mapping and strict complete pairing in capture
  comparisons. Browser validation bounds child-page requests by request capacity,
  independently of visited nodes: an unvisited child cohort can still be requested.
  The [core GPU runner](../tools/run_lod_gpu_qualification.sh)
  independently covers SH0 and SH3, serializing the multi-camera, global-order,
  GPS, package-lifecycle and traversal fixtures. Select it with
  `BGS_GPU_QUALIFICATION_SUITE=core BGS_RUN_GPU_QUALIFICATION=1`.

The [initial route comparison](../target/lod-roadmap/2026-09-09/production-refinement/route-comparison.json)
uses the unchanged Poland v17 package, 8M active/24M resident records, 1920×1080,
and a paced 30 Hz headless path on NVIDIA RTX PRO 6000 Blackwell Vulkan. Its
package-update CPU observations are:

| Scope | Earlier binary | Refined binary |
| --- | ---: | ---: |
| Held p50, 8 samples each | 1.970 ms | 1.194 ms |
| Forward motion p95, 150 samples each | 13.356 ms | 13.764 ms |
| Reverse motion p95, 149 samples each | 15.129 ms | 15.405 ms |

Held work decreases; moving-camera CPU cost shows **no material improvement**.
Returned images match the initial held image exactly, but matched forward/reverse
poses still differ. Ordering changed between binaries, and moving draw counts
also differ, so this is not a fixed-image speedup or an interactive FPS result.
Both captures predate explicit producer-input receipts and cannot qualify the
strengthened protocol. Warmup images without attested counts remain identified.

Fresh captures use explicit producer-input receipts and the pinned executable
`b2464bc5de9a154a75e22f59981770eecb4dac2e38e3b0e59c0f97762255c32f`.
The [commands and configs](../target/lod-roadmap/2026-09-09/production-refinement/run-verified-captures.sh)
preserve each run. Later CPU-sort lifecycle and browser-validator fixes do not
change the GPU hierarchy pipeline used by these captures.

| Capture | Result | Scope |
| --- | --- | --- |
| [Ordered facade](../target/lod-roadmap/2026-09-09/production-refinement/facade-ordered-verified-comparison.json) | Minimum full-image 62.008 dB / .999707 SSIM; foreground 51.785 dB / .997183 | 90 source-relative paired frames across held, motion and return segments, 320×180. |
| [GPS facade](../target/lod-roadmap/2026-09-09/production-refinement/facade-point-verified-comparison.json) | Minimum full-image 32.019 dB / .905941 SSIM; foreground 23.673 dB / .419501 | Same route, configured 8 samples/pixel with a floor of 4; fails the stricter quality target. |
| [Poland route](../target/lod-roadmap/2026-09-09/production-refinement/verified-motion-qualified-receipts.json) | Exact held/returned images; forward/reverse RGB8 MAE p95 1.724, maximum 2.007 | 149 identical-camera pairs. Local differences remain; this single-run analysis is not a speedup comparison. |
| [Poland aerial crop](../target/lod-roadmap/2026-09-09/production-refinement/aerial-verified-comparison.json) | 20.119 dB / .484957 SSIM | Same-renderer calibrated 640×360 crop. Reference remains unqualified: 3,098 physical proxy ranges remain. |

Facade metrics linearize RGBA8 output and compare against the source renderer,
not photographs. Full-image metrics include the background; foreground coverage
can be only 9.5%. Both comparisons draw all 1,024 records, so they test renderer
parity rather than reduced representatives. Both retain one unpaired startup
sample on each side; complete pairing was not required. The tiny facade's
Gaussian budget is already at its floor, so it does not qualify adaptive
large-scene budgeting.

The fresh Poland replay records 385 completed capture requests with no recorded
drops or errors: 382 steady samples, one attested warmup image and two startup
images without GPU counts. All seven held and 74 returned adjacent image pairs
match exactly. Record-budget flags persist, request-capacity flags occur during
motion, and near-plane transition bypasses remain. These bounded fallbacks are
not renderer failures, but they limit detail and do not prove seamless motion.
Package-update CPU p95 is 14.284/15.126 ms forward/reverse; GPU view p95 is
13.729/14.172 ms. These separate scopes are neither whole-frame latency nor
interactive delivered FPS.

The [SH0](../target/lod-roadmap/2026-09-09/production-refinement/sh0-gpu-fixtures.json)
and [final SH3](../target/lod-roadmap/2026-09-09/production-refinement/sh3-fixtures.json)
fixtures pass on the recorded native adapter. They cover multi-camera sorting,
hidden records at three key precisions, orthographic pan, pure rotation, spatial
motion/zoom/return, GPS visibility and work overflow, package upload delays, and
bounded hierarchy traversal. The final CPU-sort lifecycle regression passes in
both layouts. Three initial oracle witnesses still asserted the old finite-support
behavior; those assertions were corrected without relaxing their numerical bounds.

The [validation record](../target/lod-roadmap/2026-09-09/production-refinement/validation.json)
pins executable identities, source files, configs, reports and logs. Clippy passes
with warnings denied for both native layouts and the default library/viewer;
both actual browser-capture binaries compile for Wasm. The portable CPU LoD
profile, formatting, package listing, 39 Python protocol/metric tests and three
JavaScript observer tests also pass. Package listing is not archive compilation,
and browser compilation is not browser GPU qualification. Earlier SH0 GPU runs
precede the final CPU-only lifecycle changes; final SH3 GPU and SH0 lifecycle
checks cover those changes. FPS display remains FPS only.

Production acceptance still requires reduced representatives that pass the locked
quality gates, continuous detail during streamed camera motion and near crossings,
quality-preserving automatic budgets/GPS sampling, and measured interactive
latency. The [scenario inventory](../tools/fixtures/lod_gpu_qualification_matrix.json)
also identifies cold teleports, perspective zoom, shared hierarchy multiview,
cache-pressure recovery, browser execution and other adapters that small native
fixtures do not qualify. No fitted package was promoted and no quality threshold
was weakened.
