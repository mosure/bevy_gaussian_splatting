# Jastrzębia Góra / Poland scene

This local scene exercises package construction and calibrated rendering beyond
100 million splats. Writer 17 with GPU traversal and globally ordered quads
successfully settled at camera 300. Whole-route quality, reduced-detail rendering,
GPS, and production readiness remain unqualified. See the
[acceptance gates](lod_implementation_status.md) and the
[remaining quality implementation plan](lod_poland_quality_plan.md).
The [camera-motion performance follow-up](lod_motion_performance.md) records
the bounded CPU/GPU comparison and its interactive-latency limitations.

Place the original files in `assets/`:

- `20K-Photo-103Mspats-4x2KM-Andrii_Shramko_Poland-JG.ply`
- `Jastrzębia_Góra_camera_path.json`

The PLY contains **106,447,647 SH0 Gaussians**, despite the approximate count in
its filename, and occupies 5,961,068,597 bytes. It stores positions, RGB DC
coefficients, log scales, opacity logits, and quaternions. These local inputs and
generated packages are not included in the crate.

| Recorded input / artifact | SHA-256 |
| --- | --- |
| Original PLY | `c62edd860c6dbf94effd35ff31b3deeaf3fec55e6fa14edae31ddbf983fa9f37` |
| Camera JSON | `de3db614cd63ce278de6ccc4c248d4be64bc2e6df5a741521f44cc4b86e71f1d` |
| SH0 package manifest | `54da07e3cdc1a8e58c00c9691cd4bc7cfdc0e30b6eaffd6031dc7ed9d96a176f` |
| Recorded builder executable | `647de5d136afcbdacce7239cd683a1ece81003ffdcc90e3341b12ffedeba7eff` |

The camera file contains 600 standard 3DGS camera-to-world poses. All frames use
4946×3286 images, `fx=4649.505859375`, and `fy=4627.30029296875`.
The [camera importer](camera_paths.md) preserves authored roll and independent
focal lengths. A 960×638 target approximately preserves the source aspect ratio.
Indices `0, 150, 300, 450, 599` are useful route checkpoints; index 300 is the
single held pose used below.

## Build the SH0 package

Run from the repository root. This is a substantial CPU/disk operation; `--plan`
can inspect the declared admission bounds before the build. The destination must
not already exist. The command uses CPU preprocessing and CPU hierarchy fitting.

```sh
CARGO_PROFILE_DEV_OPT_LEVEL=3 CARGO_PROFILE_DEV_DEBUG=0 \
  cargo build --locked --no-default-features \
  --features 'headless lod_build_sh0' --bin build_lod

target/debug/build_lod \
  --input 'assets/20K-Photo-103Mspats-4x2KM-Andrii_Shramko_Poland-JG.ply' \
  --output target/lod-packages/poland-sh0-v17 \
  --batch-size 262144 --max-temporary-bytes 34359738368 \
  --max-manifest-bytes 268435456 --max-shard-bytes 268435456 \
  --pipeline-depth 2 --hierarchy-workers 16 \
  --max-hierarchy-working-bytes 2147483648
```

The September 8, 2026 writer-17 build produced 118,808 nodes/pages in 30 shards,
121,654,408 stored records including representatives, and 7,943,530,563 package
bytes. Its manifest is **148,617,843 bytes**, exceeding the library's default
64 MiB manifest admission. The commands and capture fixture explicitly allow
256 MiB for this asset; the library default remains unchanged.

Writer 17 removes bounds from temporary candidates that were never emitted.
It preserves actual representative/descendant support and the existing
conservative approximation errors. All 30 output shards are byte-identical to
the writer-16 package; only the manifest changed. Runtime refinement now tests
the stored boxes against the frustum while retaining spherical error estimates.

The recorded optimized development build took 556.704 seconds and reached
1,039,794,176 bytes of process peak RSS. It admitted and observed 16 concurrent
hierarchy workers. These are one machine's build observations, not renderer
memory or throughput. The builder reported 5,359 unmeasured within-cohort touching
pairs, unqualified cross-cohort interactions, and no jointly fitted mixed-depth
pairs; package validation does not certify their image quality.

## View and capture the calibrated pose

This viewer command holds camera 300 at `Original` quality with GPU hierarchy
traversal, discrete cuts, and globally ordered quads:

```sh
BEVY_ASSET_ROOT="$PWD" cargo run --release --locked --no-default-features \
  --features 'planar lod_render sh0 viewer io_flexbuffers io_ply file_asset' \
  --bin bevy_gaussian_splatting -- \
  --input-lod target/lod-packages/poland-sh0-v17/scene.gsplatlod \
  --lod-max-manifest-bytes 268435456 \
  --camera-path 'assets/Jastrzębia_Góra_camera_path.json' \
  --camera-controller flycam --camera-speed 200 \
  --camera-path-index 300 --width 960 --height 638 \
  --lod-quality 1 --lod-max-active-gaussians 16000000 \
  --lod-max-resident-gaussians 24000000 \
  --lod-max-resident-bytes 4294967296 \
  --lod-max-cpu-bytes 16107127360 --lod-max-gpu-bytes 8589934592 \
  --lod-max-concurrent-requests 64 \
  --global-order --lod-gpu-traversal \
  --global-order-max-gaussians 16000000 \
  --global-order-max-gpu-bytes 4294967296
```

`--camera-path-fps 30` advances through the remaining authored frames; this does
not qualify route quality. The resident budget provides refinement headroom above
the active cut and yields 23,437 transient page slots. CPU/GPU ledger limits apply
across owned LoD allocations; the atlas and renderer also retain their own caps.

The portable [capture fixture](../tools/fixtures/lod_runtime_poland.json) reproduces
the held-pose diagnostic settings. Its traversal workspace allows 32,768 frontier
nodes and 1 GiB; the viewer retains its 16,384-node / 256 MiB traversal defaults.
Capture therefore supplies the reproducible measured profile, not a measurement
of the interactive command:

```sh
CARGO_PROFILE_DEV_OPT_LEVEL=3 CARGO_PROFILE_DEV_DEBUG=0 \
  cargo build --locked --no-default-features \
  --features 'headless testing lod_build_sh0' --bin capture_lod
timeout 240 target/debug/capture_lod \
  --config tools/fixtures/lod_runtime_poland.json
python3 tools/check_lod_capture.py \
  target/lod-roadmap/captures/poland-camera-300-ordered/capture.jsonl \
  --verify-images
```

Paths resolve relative to the fixture. Choose a new output directory for every
run. Its `builder_revision` identifies the recorded executable above; update it
to the actual builder revision or executable digest after rebuilding the package.
The source PLY is hashed for attribution; hierarchy capture does not load all
106 million records into GPU memory. `camera_frame` supplies the calibrated
projection, superseding the fixture's fallback `vertical_fov_radians` value.

## Measured camera-300 result

`v17-ordered-original-fast-300-capture` used Vulkan on an NVIDIA RTX PRO 6000
Blackwell Workstation Edition, driver 610.43.02, at 960×638. Eight sampled frames
from 1,800 through 8,100 retained generation 473 and identical counts:

| Settled observation | Result |
| --- | --- |
| GPU-selected hierarchy records | **14,935,787** |
| Compacted and drawn quads | **3,831,900** |
| Selected pages / resident snapshot pages | 14,587 / 16,674 |
| Median timestamped GPU view work | **8.21728 ms** across eight samples |
| Frame-6,300 diagnostic reference comparison | **68.07051 dB PSNR / 0.99998289 SSIM** |

The final same-submission traversal/draw receipt reports zero traversal flags,
complete coverage, no pending requests, and a package acknowledgement for the
resident snapshot. This demonstrates settled rendering at this pose. The GPU
timing covers hierarchy, ordered rendering, and postprocessing in an instrumented
headless capture; it is not viewer FPS. CPU frame time includes capture work.

The reference retained 3,969,504 byte-preserved original records using a
conservative four-sigma plus filter-margin prefilter, without occlusion culling.
Camera matrices and viewport match, but source identities, renderer binaries, and
capture schedules differ. This is a strong single-view diagnostic, not strict
full-source or whole-route parity. The subset is not a certified GPS reference;
filtering also changes index-based sampling identity.

The final ledger reports 7,842,703,796 CPU bytes and 3,036,022,120 GPU bytes of
owned capacity reservations. These are not measured RSS or VRAM and overlap
private buffer accounting; do not add them together with those buffers.
`device_used_bytes` is unavailable. The capture correctly retains
`release_qualified: false` and `memory_audit_complete: false`.

## Completed route diagnostics and remaining qualification

The final ordered and GPS route runs exited successfully and passed image/schema
validation. Both applied all 600 authored poses, with receipts sampled at their
configured cadence. This establishes functional route execution, not image
quality at every pose. The [qualification summary](../target/lod-roadmap/2026-09-08/poland/qualification-summary.json)
retains configuration, identities, counts, timing, and incomplete evidence gates.

The ordered route at quality 0.95 produced 54 attested images from 56 samples,
including 10 during motion. Resident-first refinement settled camera 0 at
generation 425 for 22 samples: **15,996,538 selected / 10,960,624 drawn**, 17,858
snapshot pages, an acknowledgement, and zero queued/in-flight requests. The
16M record limit still applied. Camera 0 is a near-ground view of the church;
its image quality still fails despite stable residency.

On returning to camera 300, the ordered renderer recovered **14,935,787 selected /
3,831,900 drawn** with no traversal flags or pending requests. Thirteen settled
samples had a median GPU view time of **7.127616 ms**. This later route result is
separate from the held-`Original` diagnostic above.

| Final diagnostic against the raw-source subset | PSNR | SSIM |
| --- | --- | --- |
| [Ordered, camera 0](../target/lod-roadmap/2026-09-08/poland/final-ordered-route-last-quality-0.json) | 18.54747 dB | 0.276158 |
| [Ordered, camera 300 revisit](../target/lod-roadmap/2026-09-08/poland/final-ordered-route-last-quality-300.json) | 68.06842 dB | 0.99998288 |
| [GPS, camera 0](../target/lod-roadmap/2026-09-08/poland/final-point-route-last-quality-0.json) | 24.36764 dB | 0.222164 |
| [GPS, camera 300 revisit](../target/lod-roadmap/2026-09-08/poland/final-point-route-last-quality-300.json) | 10.32705 dB | 0.114360 |

The adaptive GPS run produced 64 attested images from 68 samples, including 14
during motion. Its final camera-300 image selected 896,034 records, projected
48,127 Gaussians, and dispatched 18,838,957 point attempts without point overflow.
Twenty-five samples in the final acknowledged residency generation had median
GPU view work of 3.187584 ms. Controller observations ranged over 1–7 sampling
layers and active caps of 897,592–2,000,000; those lagged observations do not prove
stable control or acceptable quality. The run's `scenario_samples_complete`
remains false, so completion does not imply every scheduled capture landed.
GPS comparisons use the same flat-quad subset teachers and retain their stated
source, scheduling, and sampling limitations.

The current 8:1 proxies preserve additive alpha mass but do not constrain
perspective-composited occlusion. The next quality step is a bounded cohort fit
and evaluation against perspective RGB and transmittance before another full
scene rebuild. No such fit or improvement is claimed here. Whole-route reduced
LoD quality, motion/rapid-return quality, quality-preserving automatic control,
and complete device/residency memory accounting remain release blockers. Use the
[native GPU capture guide](lod_point_runtime_capture.md) to retain actual
image/count receipts and matched reference evidence for each configuration.

The current traversal still maintains a global cut: refining a visible parent
admits offscreen siblings. The earlier camera-300 global `Original` cut cost
31.04 million records; writer-17 bounds plus AABB refinement now measure
14.94 million. Further savings require a separate per-view visible-cohort budget.
That change must prove conservative support bounds for authored sigma, runtime
splat scale, transforms, projection/filter margin, and near-plane crossings.
It must retain the parent until the entire visible child cohort is resident,
reserve pending refinement capacity, handle child counts smaller than the parent,
charge visibility visits, and preserve root pins, snapshot leases, and exact
source/generation image acknowledgements. Hard pruning is not implemented.

Evidence is local under `target/lod-roadmap/2026-09-08/poland/`:
`source_inventory.json`, `package-v17.json`, `v17-shard-parity.json`,
`full-build-v17.log`, `v17-ordered-original-fast-300-capture/`,
`v17-ordered-original-fast-300-quality.json`,
[final ordered route](../target/lod-roadmap/2026-09-08/poland/final-ordered-route-capture/capture.jsonl),
and [final GPS route](../target/lod-roadmap/2026-09-08/poland/final-point-route-capture/capture.jsonl).
Preserve settings, hashes, images, and submission evidence for fresh conclusions;
these large local artifacts are not bundled with the crate.
