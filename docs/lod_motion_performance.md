# Poland camera-motion performance follow-up

September 8, 2026. This pass reduces work on the main thread and compute-pass
encoding while retaining current-camera selection, complete sibling replacement
and the configured quality and resource limits. It does not qualify Poland's
proxy quality or interactive input-to-display latency.

This report preserves the earlier optimization baseline. The later
[temporal qualification](lod_temporal_quality.md) includes indexed eviction,
shared publication leases, fixed-slot snapshots, required-request priority,
view-scoped physical omission and paced 1080p routes with image comparisons.
Its September 9 closeout explicitly rejects faster captures that lost detail.
GPU timings alone do not qualify whole-frame FPS, input latency or visual quality.

## Changes

- GPU packages retain normalized page-demand sets until membership, generation
  or camera ownership changes. Runtime demand epochs still refresh every frame;
  terminal failures, retries, newly resident pages and removed views remain live
  inputs to admission and cancellation. Reconciliation checks pending work
  against the retained view sets instead of constructing another large union.
- GPU traversal batches dependent dispatches into one compute pass per level.
  Copies into a separate indirect-argument buffer preserve storage/indirect
  usage boundaries. Dispatch order, bounds and selection are unchanged.
- Global quad ordering batches its three gather kernels into one pass and its
  fifteen radix dispatches into another. Four stable radix digits and the
  resulting draw order are unchanged.
- GPS automatic sampling rejects delayed timings and overflow observations from
  a superseded sample count. Successful image acknowledgements are independent
  of that decision, so stale feedback cannot cascade into repeated reductions.

## Bounded observation

The [raw comparison](../target/lod-roadmap/2026-09-08/motion-performance/comparison.json)
records executable identities, configs, sample counts and all phase statistics.
The optimized native SH0 builds use Vulkan on an RTX PRO 6000 Blackwell with
driver 610.43.02. Both runs use writer-17 Poland, 960×638, quality 0.95, a 16M
active cap and 24M resident cap. A 1,200-frame logical path warms for 600 frames,
holds for 120, moves laterally 30 world units over 180, returns over 180 and
holds for 120. The path starts at camera 0's position/orientation with a standard
perspective projection derived from its vertical focal length.

The final pair uses eight readback slots, samples every 24 logical frames and
captures counts/timestamps without images. Both runs complete all 49 requests,
including 48 attested draws and one startup observation; both have zero ring
drops, mapping errors and unresolved requests. All requested scenarios have
complete samples. The capture checker passes diagnostic schema/receipt checks;
full memory and image qualification remain false.

| Movement and return, 15 samples per run | Before median | After median |
| --- | ---: | ---: |
| Package update CPU | 13.87 ms | 2.22 ms |
| GPU hierarchy | 1.58 ms | 0.92 ms |
| Ordered GPU backend | 11.23 ms | 13.20 ms |
| Total GPU view | 13.66 ms | 14.15 ms |
| Selected Gaussians | 15,995,705 | 15,640,848 |
| Post-projection drawn Gaussians | 10,825,235 | 10,707,842 |

The sampled camera matrices match at every corresponding logical frame, but
asynchronous residency and selected cuts differ. A logical path also advances
at different wall-clock speeds when the producer becomes faster. This is a
single before/after diagnostic, not fixed-residency or equal-image throughput
qualification. Medians of individual stages are not additive. Ordered rendering
remains the dominant GPU cost; the run establishes no total GPU improvement.

The headless runner has no presentation surface and runs without pacing.
`frame_wall_ms` measures main-loop cadence, not displayed FPS or completion
latency. An initial three-slot/every-eight-frame after run dropped 38 samples;
those artifacts remain preserved and are excluded from the table. Increasing
the diagnostic ring fixed sample loss, not GPU queue latency. The actual viewer
uses AutoVsync and Bevy's default surface frame-latency hint of two. A paced
motion comparison and interactive latency review remain separate work.

## Verification and remaining work

Seven focused CPU regressions pass for demand epochs, retries/residency, shared
view cancellation, stable in-flight demand, cohort invalidation, delayed GPS
feedback and flycam input. Four small GPU fixtures pass: traversal compares
separate-pass and batched held/moved/returned cuts; global ordering preserves
split/merged image equivalence; the package fixture covers pending uploads,
camera changes and generation acknowledgements; GPS checks image/overflow and
sampling behavior. Targeted Clippy with warnings denied, formatting and whitespace
checks pass. The viewer and capture executable are rebuilt.

Further work should target measured projection/sorting/raster costs, residency
publication tails during startup and fast movement, and paced input-to-display
qualification. Useful proxy fitting and smooth parent/child transitions remain
image-quality blockers. No proxy fit, package rebuild, lower quality preset or
browser qualification was performed for this pass.
