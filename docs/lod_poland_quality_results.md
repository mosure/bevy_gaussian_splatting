# Poland quality investigation: measured results

September 9 update: [local aerial authoring](#blocker-closeout-local-aerial-authoring)
improves training crops but does not repair the development validation crop or
promote a package. [Runtime closeout](lod_temporal_quality.md#blocker-closeout-september-9-partial)
remains partial, with recovered counts but a motion-detail regression.

September 8 baseline. The renderer's held church result is substantially
improved: **65.8376 dB PSNR / 0.9996114 SSIM** against the attested tiled Original
reference, with unchanged sampled held cuts and pixels. See
[temporal rendering results](lod_temporal_quality.md) for the new selection,
spatial transitions, projection fix and reference limitations. The fixed 8M/24M
profile meets the measured 1080p GPU timing target and recovers the initial image
after the route. **Full-scene visual qualification fails:** a settled aerial crop
scores 20.14 dB / .486 SSIM against Original, and the identical cut without
interpolation still fails at 20.73 dB / .542. Static proxy quality remains a
blocker, with additional interpolation error and local route differences. The
scoped GPU result is not production acceptance.

The investigation below preserves the **historical failed cut and fitting
experiments**. Its bounded geometry fit improves training crops but still fails
the validation target; no fitted Poland representation or useful GPS quality
profile was accepted. The historical cut also fails against the new reference,
so its defect is not merely an artifact of switching teachers. The
[phase plan](lod_poland_quality_plan.md) defines the remaining gates; the
[fitting guide](lod_representative_fitting.md) documents the diagnostic tools.
Artifact links below refer to local, untracked evidence under `target/`.

## Frozen church capture

The [configuration](../target/lod-roadmap/2026-09-08/poland-quality/church-cut-config.json)
uses calibrated camera-path frame 0 at 960×638, near 0.1, with GPU hierarchy
traversal and globally ordered quads. It ran on NVIDIA RTX PRO 6000 Blackwell,
Vulkan, driver 610.43.02. The frozen image is capture frame 2100, generation 324,
view `4294967105`:

| Identity | Value |
| --- | --- |
| Run | `2828d7b4f4bc0b4e3b1919b28444807e7ce96cca2d4adf613b6b8c56559a2504` |
| Manifest SHA-256 | `54da07e3cdc1a8e58c00c9691cd4bc7cfdc0e30b6eaffd6031dc7ed9d96a176f` |
| Renderer executable SHA-256 | `317c43243b80a796dcfcaf0da68cbb1756582a2cae675fefb3d1d6e600381cad` |
| Image SHA-256 | `8b0ebdf6f30a047872dfc0245eb979762ea826864b2ea9e13884c132b353166e` |

The [capture rows](../target/lod-roadmap/2026-09-08/poland-quality/church-cut-capture/capture.jsonl)
contain source, camera, settings and build identities. The
[submission receipts](../target/lod-roadmap/2026-09-08/poland-quality/church-cut-capture/submission_evidence.jsonl)
and [hierarchy index](../target/lod-roadmap/2026-09-08/poland-quality/church-cut-capture/hierarchy_index.json)
pin the selected node/output ranges from the same submission as the image.

| Frame 2100 observation | Result |
| --- | ---: |
| Selected cut | 15,622 nodes; 15,996,538 records |
| Post-projection compacted and drawn records | 10,822,649 |
| Output capacity | 16,000,000 |
| Resident pages | 17,858 |
| GPU hierarchy / ordered backend / postprocess | 3.929856 / 12.514976 / 0.425760 ms |
| Total measured GPU view time | 16.870592 ms |
| Process RSS at this frame | 4,464,226,304 bytes |
| Partially instrumented GPU allocations | 1,368,642,992 bytes |

The [integrity check](../target/lod-roadmap/2026-09-08/poland-quality/church-cut-integrity.json)
finds exact, nonoverlapping coverage of all 106,447,647 canonical source records
and all selected output records. This establishes cut completeness, not image
quality. The [image comparison](../target/lod-roadmap/2026-09-08/poland-quality/church-cut-quality.json)
still fails visually: **18.547857 dB PSNR / 0.276310 SSIM** against the historical
camera-0 conservative raw-source subset. That reference has no newly attested
support-profile match and is not a full-source quality qualification.

The [run summary](../target/lod-roadmap/2026-09-08/poland-quality/church-cut-run.json)
records exit 0, 47.007 seconds, sampled peak RSS 4,474,052,608 bytes and peak
whole-device usage 5,683 MiB. RSS was sampled once per second; device usage is
not process-exclusive. The allocation ledger and partial renderer allocations
are separate accounting scopes and must not be added as independent VRAM usage.
Instrumented headless timings do not establish interactive viewer FPS.

The [request outcomes](../target/lod-roadmap/2026-09-08/poland-quality/church-cut-capture/request_outcomes.jsonl)
and [status](../target/lod-roadmap/2026-09-08/poland-quality/church-cut-capture/capture_status.json)
account for all 10 requests exactly once: 10 completed readbacks, eight attested
drawable captures and two startup captures with missing drawable diagnostics.
There are zero unresolved requests, accounting errors, ring drops or mapping
errors. Request accounting and scenario sampling are complete;
`attested_counts_complete`, `memory_audit_complete` and `release_qualified` remain
false. Recording startup skips does not establish complete rendered cadence.

## Corrected native-value substitution

The [corrected check](../target/lod-roadmap/2026-09-08/poland-quality/church-substitution-native-check.json)
uses physical pixel `[480,319]`, owned nodes 8484 and 8435, all 16,384 owned
original records, 2,048 seed representatives and 24,574 immutable context records.
Source intervals identify the canonical original-leaf domain, not raw PLY row
numbers or proven per-representative reducer lineage.

| Rendered input | Linear RGB | Owned-layer transmittance |
| --- | --- | ---: |
| Full captured cut, CPU | `[0.1408044845, 0.1853571236, 0.0737909824]` | — |
| Native seed + frozen context | `[0.1408044845, 0.1853571236, 0.0737909824]` | 0.1704709530 |
| Native originals + frozen context | `[0.0914092958, 0.0965567529, 0.0502529182]` | 1.0 |

Seed RGB matches the CPU full-cut attribution exactly at this pixel. The owned
originals contribute no opacity here under deployment world-support culling;
the coarse owners introduce false opacity. Substitution reduces RGB MSE against
the old subset pixel by 69.3%, but context still contributes haze. Final owner
compositing weights include context attenuation and are not `1 - owned T`.
This is a one-pixel causal diagnostic with fixed three-sigma source/seed/context
support, not a fit, whole-crop acceptance, or equivalence to adaptive flat support.

The [three-pixel GPU comparison](../target/lod-roadmap/2026-09-08/poland-quality/church-attribution-pixel-check.json)
has maximum linear-channel differences of 0.00180–0.00246 against decoded RGBA8
capture pixels. Quantization is included; whole-frame CPU/GPU parity is unproven.

The old [substitution diagnostic](../target/lod-roadmap/2026-09-08/poland-quality/church-substitution/diagnostic.json)
and its artifacts remain preserved as **invalid supervision**. They omitted the
deployment world-support frustum gate, and PLY re-import could reclamp anisotropic
context scales. The corrected path retains authenticated native page values,
validates pinned PLY sidecars, and preserves the full deployment frustum when
evaluating physical crops. The two regional candidates below pass the separate
export/reload image gate for their admitted views.

## Local GPS expectation check

The [GPU fixture log](../target/lod-roadmap/2026-09-08/poland-quality/gps-expectation-gpu.log)
records 16 independent seeds at 8 SPP on the same NVIDIA adapter, with two
projected Gaussians and a bounded pixel-integrated expectation oracle. Ensemble
bias RMSE is 0.0089807 against estimated Monte Carlo standard-error RMSE 0.0091030,
with quantization allowance 0.0005590 and oracle allowance 0.0000139. Individual
displayed-frame RMSE remains 0.0350–0.0384. This local stochastic-process check
passes its declared comparison; it establishes no Poland image acceptance or
universal agreement between GPS and ordered alpha compositing.

## Regional diagnostics, fitting and validation

The initial five 32×32 regions were
[rejected at the record cap](../target/lod-roadmap/2026-09-08/poland-quality/church-regional-export.log).
The declared fallback used five 16×16 physical crops at the same centers, without
resizing the 960×638 calibration or dropping a context tail. The
[admitted cohort](../target/lod-roadmap/2026-09-08/poland-quality/church-regional-cohort-16/cohort.json)
retains all 16,384 owned originals, all 2,048 seed representatives and **12,015
immutable context records**. Its complete support union scanned 15,996,538 cut
records, 1,024,465,800 encoded bytes and 1,023,778,432 decoded bytes, one page at a
time. Completeness applies to this frozen cut and these crops, not a new runtime
cut at every evaluation pose.

The [regional seed comparison](../target/lod-roadmap/2026-09-08/poland-quality/church-regional-cpu-gpu-seed-parity.json)
matches CPU seed/context images against the captured GPU image in the two camera-0
crops: 56.64 dB / 0.999556 SSIM and 54.56 dB / 0.998996 SSIM. Maximum linear error
outside RGBA8 quantization intervals is 0.00185 and 0.00213. This does not attest
GPU source substitution, other poses or a fitted payload.

Exactly two 24-step fits were run. The
[appearance fit](../target/lod-roadmap/2026-09-08/poland-quality/church-fit-appearance-16/fit.json)
reduces training RGB-plus-owned-transmittance loss from 0.27540174 to 0.24259949;
the [geometry fit](../target/lod-roadmap/2026-09-08/poland-quality/church-fit-geometry-16/fit.json)
reaches **0.00896794**, with representative-local mean coordinates and bounded
geometry feasibility. Validation loss changes from 0.04130476 to 0.04031749 and
0.03885789 respectively. Both PLY reload gates pass their 0.0001 maximum RGB and
transmittance error limits. Their `artifact_accepted` fields describe these local
artifact gates, not the separate image-quality target.

Whole-command wall time was
[4.91 seconds for appearance](../target/lod-roadmap/2026-09-08/poland-quality/church-fit-appearance-16-time.txt)
and [5.01 seconds for geometry](../target/lod-roadmap/2026-09-08/poland-quality/church-fit-geometry-16-time.txt),
including approximately 4.36–4.37 seconds of admission/authentication each. Peak
RSS was 272,980 and 272,576 KiB respectively.

The [selection receipt](../target/lod-roadmap/2026-09-08/poland-quality/church-candidate-selection.json)
chooses geometry before rendering final evaluation camera 540. Both permitted fit
invocations are consumed. The
[reloaded candidate evaluation](../target/lod-roadmap/2026-09-08/poland-quality/church-selected-evaluation-16-quality.json)
compares against authenticated original owners in the same native frozen context:

| Crop and camera | Role | Seed PSNR / SSIM | Selected PSNR / SSIM | ≥35 dB and ≥0.98 SSIM |
| --- | --- | --- | --- | --- |
| Church false opacity, 0 | Training | 24.40 / 0.8880 | 83.89 / 1.0000 | Pass |
| Church owned surface, 0 | Training | 26.78 / 0.9483 | 65.06 / 1.0000 | Pass |
| Aerial owned surface, 300 | Training | 31.77 / 0.9207 | 35.03 / 0.9681 | Fail |
| Separated window, 360 | Validation/selection | 32.11 / 0.8811 | 32.31 / 0.8951 | Fail |
| Separated window, 540 | Final evaluation | Not measured | 41.07 / 0.9890 | Pass |

Mean owned transmittance in the false-opacity crop is 1.0 for the originals and
0.99472265 for the selected fit. This repairs that local opacity error, while the
remaining failed crops prevent regional acceptance. Each image is only 16×16;
11-tap SSIM uses a **6×6 valid interior**, so these scores have limited spatial
coverage. Camera 540 was unused for this candidate selection, not globally unseen
in the earlier route investigation. No additional optimizer run, full rebuild or
fitted GPU evaluation was performed.

The [24 focused CPU checks](../target/lod-roadmap/2026-09-08/poland-quality/regional-cpu-tests.json)
pass, covering calibrated support, exact context/forgery/order, PLY image drift
and budget/outcome contracts. The
[GPU traversal regression](../target/lod-roadmap/2026-09-08/poland-quality/traversal-visit-refund-gpu.log)
passes in 1.23 seconds; the GPS fixture passes in about 4.5 seconds. Focused
[SH0 Clippy](../target/lod-roadmap/2026-09-08/poland-quality/clippy-focused-final.log)
and [default SH3 Clippy](../target/lod-roadmap/2026-09-08/poland-quality/clippy-default-sh3-final.log)
pass. [SH0 Clippy including all test targets](../target/lod-roadmap/2026-09-08/poland-quality/clippy-sh0-final.log)
also passes with warnings denied. This compiles and lints those targets; it does
not run the broad covariance suite. Formatting and whitespace checks pass. The
[validation summary](../target/lod-roadmap/2026-09-08/poland-quality/validation-summary.json)
records the executable identity and the remaining failed quality gate.

## Navigation and near-camera refinement follow-up

GPU traversal now ranks eligible splits within each breadth-first frontier by
near-plane AABB intersection, then projected error and footprint. Eight stable
priority buckets precede the bounded prefix allocation for resident refinement
and missing-page demand. This prevents topology order alone from giving a distant
coarse branch priority over a nearby severe proxy. Complete sibling replacement,
parent fallback and record/frontier/visit/request limits remain enforced.
Priority is per frontier, not a global best-first search across tree depths.

The [GPU traversal check](../target/lod-roadmap/2026-09-08/viewer-controls/traversal-ground-priority.log)
passes in 1.16 seconds, including near-plane, projected-error, zero-error footprint,
bounded page demand and held/moved/returned camera cases. The
[package regression](../target/lod-roadmap/2026-09-08/viewer-controls/package-ground-priority.log)
passes in 2.06 seconds, covering current-camera selection during pending uploads,
residency generation changes and reusable traversal buffers. The additional rank
workspace is included in allocation admission; required workgroup storage is
13 KiB. These small fixtures do not measure full-Poland traversal overhead.

Optional [ground-matched flycam](camera_paths.md) uses a bounded asynchronous
estimate and a fixed navigation up direction. A CPU check of the actual viewer
estimator on 4,096 stratified Poland positions finds up approximately
`[-0.00691, -0.99996, -0.00549]`, with 2,445 inliers and RMS distance 4.85 world
units. Reading that sample used 49,152 position payload bytes, not a full PLY
scan. This qualifies the estimator on that sample; automatic resident sampling
and camera navigation still need a full-scene interactive review.

The [close-proxy metadata audit](../target/lod-roadmap/2026-09-08/viewer-controls/near-proxy-metadata.json)
also finds authored coarse representatives whose scales are large relative to
their camera depth. The earlier original-owner substitution establishes false
opacity in these representatives. Refinement priority does not repair their
payloads or make discrete parent/child transitions continuous. No new proxy fit,
package rebuild, full-scene image comparison or motion-quality acceptance was
performed for this follow-up.

## Blocker closeout: local aerial authoring

The [new authenticated aerial cohort](../target/lod-roadmap/2026-09-08/blocker-closeout/aerial-cohort-1052/cohort.json)
contains 65,536 owned originals, 1,024 fixed seed representatives and 6,514 frozen
context records. Evaluation uses native 32×32 crops and the new smooth support
kernel. An area-weighted initialization worsened camera-471 RGB PSNR from
24.17 to 21.83 dB and increased owned-transmittance error; it was rejected before
optimization. The [bounded protocol](../target/lod-roadmap/2026-09-08/blocker-closeout/aerial-fit-protocol.json)
then permits one 64-step geometry fit from the authored seed, with a 60-second
training and 90-second shared compute ceiling. Camera 481 is development
validation: its baseline was already inspected, so it is not a blind holdout.

| Camera / role | Seed RGB PSNR | Reloaded fit RGB PSNR |
| --- | ---: | ---: |
| 471 / training | 24.17 dB | 32.33 dB |
| 461 / training | 25.85 dB | 36.31 dB |
| 481 / development validation | 30.62 dB | 30.60 dB |

The [source-relative comparison](../target/lod-roadmap/2026-09-08/blocker-closeout/aerial-authoring-comparison.json)
shows useful training improvement and essentially unchanged validation error.
The [fit receipt](../target/lod-roadmap/2026-09-08/blocker-closeout/aerial-cohort-1052-geometry-fit/fit.json)
records training RGB-plus-owned-transmittance loss **0.024128 → 0.001526** and a
passed PLY reload image-fidelity gate on all three views. Its `artifact_accepted`
flag means this local reload gate passed; `gpu_forward_parity_qualified`,
`quality_certificate` and `package_or_error_policy_modified` remain false.
There is no fitted GPU/package image acceptance, and this small cohort does not
close the settled aerial full-scene quality failure. No package promotion follows.

The final complete-cohort renderer restores the sampled return image exactly
and removes the introduced near-camera haze regression; local motion differences
remain. Its settled aerial crop is still visibly hazy. The
[20.13162 dB / 0.485724 SSIM comparison](../target/lod-roadmap/2026-09-08/blocker-closeout/aerial-original-center-vs-fixed-spatial-8m-aerial-held-cohort.json)
is explicitly diagnostic: the stricter reference audit cannot exclude every
physically expanded proxy from the cropped image. See
[final temporal evidence](lod_temporal_quality.md#final-complete-cohort-correction)
for the exact renderer, runtime results and remaining qualification boundaries.

## Implementation and remaining gates

Implemented facilities include captured-cut attribution, authenticated native
cohorts, calibrated crops, frozen-context RGB/owned-transmittance objectives,
bounded context record extraction, stable captured tie order, reload checks and
explicit capture outcomes. Their existence does not close the image-quality gates.

The [phase ledger](lod_poland_quality_plan.md#implementation-and-acceptance-ledger)
tracks the dependencies: broader forward parity and a fit passing regional quality;
capacity or partition changes if that fit requires them; bounded selective
authoring; quality-driven traversal priorities; useful Poland GPS noise/work
profiles; representation-aware automatic controls; and full-source, dense-motion,
native/browser and physical-memory qualification. No full rebuild or production
acceptance follows from these bounded local improvements.
