# Bounded representative fitting diagnostic

`capture_lod --fit-rung FIT.json` fits an exported authored rung against the original PLY, or an authenticated local cohort against its original leaves, using the existing CPU production oracle. It does not start Bevy rendering, load a GPU, change a package, update error thresholds, or create a quality certificate. CPU/GPU forward parity must be qualified before interpreting its loss as the renderer's loss.

## Captured-cut attribution and calibrated cohorts

Use an instrumented `hierarchy_ordered` capture with `capture_hierarchy_cut: true`
to identify contributors before choosing a cohort. Wait for the capture to finish,
then pin `capture.jsonl`, `submission_evidence.jsonl`, the manifest and the imported
camera file by SHA256. The commands below are CPU diagnostics; use a binary whose
compiled SH layout matches the package. Replace illustrative paths, hashes and
node IDs with authenticated inputs; output files/directories must be new.

```sh
target/debug/capture_lod --attribute-cut attribution.json
target/debug/capture_lod --export-cohort export.json
target/debug/capture_lod --fit-rung diagnostic.json
```

`attribution.json` selects one actual capture frame and physical pixel centers:

```json
{
  "manifest": {"path": "package/scene.gsplatlod", "sha256": "<manifest SHA256>"},
  "capture": {"path": "capture/capture.jsonl", "sha256": "<capture SHA256>"},
  "receipt": {"path": "capture/submission_evidence.jsonl", "sha256": "<receipt SHA256>"},
  "frame": 2100,
  "camera_path": {"path": "camera_path.json", "sha256": "<camera SHA256>"},
  "camera_frame": 0,
  "pixels": [[480,319], [700,400], [240,200]],
  "output": "attribution.json.out"
}
```

Attribution paths resolve relative to its config. It checks the image, camera,
settings and complete canonical-source cut, resolving GPU root aliases through
the pinned manifest. Equal forward camera-depth ties use captured expanded output offsets.
One authenticated page is decoded at a time: limits are 30M selected records,
32,768 nodes, 2 GiB encoded/decoded page work, eight probes, 100k retained hits
total and 180 seconds. Overflow fails without a prefix report. Each probe reports
`owners_by_final_alpha_weight`, source counts and composed linear RGBA. This CPU
attribution is diagnostic; it does not establish GPU image parity.

Use those owner IDs in `export.json`, with disjoint neighboring context domains:

```json
{
  "manifest": {"path": "/data/package/scene.gsplatlod", "sha256": "<manifest SHA256>"},
  "output_directory": "/data/cohort",
  "owned_nodes": [123], "context_nodes": [124],
  "include_original_context": false
}
```

Export produces `source.ply`, `seed.ply`, `context.ply` and `cohort.json`;
`include_original_context: true` also exports `original-context.ply`. Owned source
records come from a complete original-leaf cover of each chosen node. Source
intervals address canonical decoded records, **not raw PLY row numbers**. Morph
transition ownership does not prove exact reducer contributors per representative.
Owned source and seed exports preserve whole domains; owned/context ancestor
overlap is rejected. Fitting retains authenticated native page values exactly.
The PLYs remain pinned, byte-checked export artifacts: reimport can clamp log-scale
outliers or normalize rotation, so exported/reloaded image fidelity is a separate
gate. Both native original leaves and proxies use fixed three-sigma support here;
this is not an adaptive-support flat-reference comparison.
Context counts toward the 100k source-plus-context and 30k seed-plus-context caps;
original context must fit too when requested. Export page work is capped at 512 MiB.
Export and fit paths resolve from the process working directory; absolute paths
avoid ambiguity.

Manual `context_nodes` preserve whole neighboring nodes, but provide no proof that
a crop's context is complete. For that proof, replace `context_nodes` with this
`captured_context` fragment in the export config, freezing every training and
evaluation crop before extraction:

```json
{
  "include_original_context": false,
  "captured_context": {
    "capture": {"path": "/data/capture/capture.jsonl", "sha256": "<capture SHA256>"},
    "receipt": {"path": "/data/capture/submission_evidence.jsonl", "sha256": "<receipt SHA256>"},
    "frame": 2100,
    "camera_path": {"path": "/data/camera_path.json", "sha256": "<camera SHA256>"},
    "views": [{
      "frame_index": 0, "calibration_viewport": [960,638],
      "crop": {"origin": [416,255], "size": [128,128]},
      "near": 0.1, "far": 100000.0
    }]
  }
}
```

This mode requires an attested ordered cut containing every `owned_node` and is
exclusive with manual or original context. It reprojects that **same frozen cut**
over the union of one to eight calibrated regions; it does not claim the runtime
selected that cut at every pose. The full deployment frustum and Mip filter are
retained. A conservative OBB/rectangle test includes every possible pixel-center
support intersection, with rounding allowance. It can retain extra records; it
does not drop records by source alpha, accumulated opacity, visibility behind
another splat, contribution rank or an arbitrary tail threshold. Frozen context
remains sufficient as owned parameters move because the evaluated regions stay fixed.

The `context_selection` sidecar records typed consecutive page-local `runs`,
their parent node domains and captured output indices. Those domains are not
exact original-contributor subranges of individual proxies. Loader admission
repeats the complete scan and compares the derived runs, rejecting forged omissions.
Fits also reject cameras/crops outside the frozen region list. Stable camera-depth ties
use `(captured node output start << 32) | record offset`; original-leaf replacement
uses the owner's prefix plus canonical within-owner source offset. This preserves
unchanged seed/context order without relying on owned-before-context array layout.

Each scan admits at most 30M cut records, 32,768 nodes and 2 GiB each of encoded
and decoded page work, with a 180-second deadline checked between pages. It decodes
one page at a time and retains record runs rather than the full cut's records.
Export and subsequent load each perform this scan. Their separately reported
payload page work remains capped at 512 MiB, and retained context still counts
toward both fit record ceilings. These are admitted work/owned-memory bounds,
not RSS or GPU-memory measurements. Exceeding a cap fails without a partial context.
The [five-region Poland config](../target/lod-roadmap/2026-09-08/poland-quality/church-regional-cohort-config.json)
is a geometry-selected diagnostic input, not a quality result.

Hash the resulting `cohort.json` for `diagnostic.json`:

```json
{
  "cohort_sidecar": {"path": "/data/cohort/cohort.json", "sha256": "<cohort SHA256>"},
  "context_representation": "selected", "diagnostic_only": true,
  "output_directory": "/data/cohort-diagnostic", "viewport": [64,64],
  "training": [{
    "id": "development_camera0",
    "camera_path": {"path": "/data/camera_path.json", "sha256": "<camera SHA256>"},
    "frame_index": 0, "calibration_viewport": [960,638],
    "crop": {"origin": [416,255], "size": [128,128]}, "near": 0.1, "far": 100000.0
  }],
  "heldout": [],
  "options": {"reference_pixels": {"viewport": [128,128], "offset": [0,0]}}
}
```

Independent fx/fy and imported rotation survive projection. The physical crop
subtracts its origin from the principal point; it preserves native focal lengths
and Mip filtering. Here the 64×64 grid samples every second pixel of that crop.
`diagnostic_only` renders unchanged source/seed substitutions with zero optimizer
steps, publishing `diagnostic.json`, RGB/owned-transmittance PNGs and `.rgbt32`
arrays containing little-endian f32 `[R,G,B,owned T]`. Context stays in the shared
forward camera-depth order and attenuates RGB gradients; it has no optimizer state or owned-T
contribution. `context_representation: "original"` uses the optional original
context export in manual-node mode. Region-complete context certifies the declared
CPU support profile and frozen cut/crops; it does not establish GPU parity,
original full-scene occlusion, or another runtime cut's coverage.

To fit, set `diagnostic_only: false`, supply distinct training/heldout views and
bound `options.max_steps`/time. Camera-frame reuse across splits, even with another
crop, is rejected; already inspected views remain development evidence. Optional
`fit_geometry: true` with `mean_coordinates: {"mode":"representative_local",
"trust_region_fraction":0.05}` uses frozen seed rotation/scale axes and a bounded
dimensionless mean step; its fraction must be in `(0,0.25]`. The default mode is
`world`. Both retain owner-support checks and fresh complete-image acceptance.

Calibrated-cohort fits reserve ten seconds inside `options.max_seconds` for a
PLY export/reload image check. The same deployment PLY decoder can normalize
rotations and clamp anisotropic scales; every configured view must retain
composed RGB and owned transmittance within the declared absolute tolerances
(both default to `1e-4`). `ply_roundtrip` configures the reserve and tolerances.
Failed or incomplete checks retain `unaccepted-candidate.ply` and a report, and
return failure. A passing `fitted.ply` establishes artifact fidelity on those
views, not the source-relative image-quality target.

Evaluate a fitted artifact with `diagnostic_only: true` and
`candidate_ply: {"path":"/data/fit/fitted.ply","sha256":"<candidate SHA256>"}`.
Its count and owned index order must match the original seed. The diagnostic
reloads the pinned candidate, retains native original/context values, and enforces
the authenticated crop union. Candidate images are named `view-NN-candidate`;
their source comparisons also permit both-blank regions, which can represent a
successfully removed false-positive contribution. A pinned candidate can also
initialize a bounded training run; cardinality, owner bounds and input hashes
are checked again, and the fitted output must pass the same deployment reload
check. Freeze candidate selection before evaluating final test poses.

## Complete-rung screen

The first screen config is [`tools/fixtures/lod_fit_icecream_24.json`](../tools/fixtures/lod_fit_icecream_24.json). It pins the 84,348-record Icecream teacher and complete 21,087-record branch-four depth-three export by distinct SHA256 identities, together with the export sidecar and successful builder command record. Those local fixtures must exist at the paths in the config. The config freezes two training cameras and six heldout cameras before optimization. Its 24-step invocation is:

```sh
target/debug/capture_lod --fit-rung tools/fixtures/lod_fit_icecream_24.json
```

This command is an experiment, not a validation prerequisite. Coordinate its execution with other CPU/GPU qualification workloads. The example output directory must not already exist. The first bounded attempt stopped before optimization at the default 16-million-visit ceiling; a separately recorded 128-million-visit attempt reached the original whole-image tape ceiling before publishing an output. The example now uses 128 million visits; tape and time ceilings remain unchanged. Failed admission attempts establish no fitted quality result.

The objective is mean linear premultiplied composed RGB squared error plus `alpha_weight` times mean owned-only transmittance squared error, averaged over training views. With no context, the transmittance term equals the previous alpha error. Rendering uses identity cloud transforms, fixed authored three-sigma OBB support, sRGB display SH decoded to linear color, Mip covariance filtering, forward camera-depth (`-view-space Z`) far-to-near sorting and f32 source-over compositing. Recorded fits and captures from the earlier squared-radial-distance renderer remain historical evidence; they do not qualify the changed ordering objective. The native flat capture teacher must also use `opacity_adaptive_radius: false`. This differs from the ordinary flat renderer's adaptive-support default. The diagnostic admits PLY inputs only, at most SH degree three; it does not fit exposure, tonemapping, cloud transforms, camera intrinsics or the renderer's background.

`options.reference_pixels` optionally specifies a deployment `viewport` and a
physical-pixel `offset`. Its viewport must be the same integer multiple of both
allocated viewport axes, with stride 1–256 and reference dimensions at most
8192. Each offset must be smaller than that stride. Projection, Mip covariance
filtering and determinant opacity compensation are evaluated at the reference
resolution, then the conic is mapped into the allocated sample grid. Teachers,
gradients and fresh proposal-acceptance renders use those same fixed physical
pixel centers. No full reference image is allocated. Omitting the option
preserves ordinary rendering at the allocated resolution.

For example, a 240×135 allocated grid with reference viewport 1920×1080 and
offset `[4,4]` supervises physical centers `(8x+4.5, 8y+4.5)`. This samples the
native image; it does not apply a low-resolution blur or integrate an 8×8 pixel
cell. The fixed sparse grid can miss detail between its samples, so full-image
GPU qualification remains necessary. Losses from different grids or reference
resolutions are different objectives and must not be compared as the same loss.

The optimizer reverses complete source-over compositing using a bounded contribution tape for each pixel tile. Each training view projects and sorts its representatives once, then processes disjoint tiles of at most 32×32 pixels. Every pixel receives the full Gaussian depth order. A conservative support-rectangle count splits a tile before it could exceed the tape allocation ceiling; even a single pixel must fit its complete contribution list or the invocation fails. One tape allocation is reused, with the old allocation released before any growth. Per-owner gradients accumulate in f64 across tiles, normalized by the full image size. Tiling changes only the order of f64 reductions over independent pixels; it does not subsample supervision, truncate overlap, or reset the full-view pixel-visit counter. Teacher and proposal-acceptance renders retain the complete untiled forward path.

The optimizer differentiates SH coefficients and logit opacity analytically. Setting `options.fit_geometry` to `true` additionally fits world means, log scales and local rotation increments through finite differences of each Gaussian's local projected parameters, once per representative after tile gradients are accumulated. It does not rerender an entire image per geometry parameter. Discontinuous support edges and depth-order changes are excluded from the local derivative, so every proposed parameter update is accepted only after a fresh complete training render decreases or preserves the objective. Four global step-halving attempts are allowed per update. Adam moments advance for complete training-gradient evaluations, including rejected updates. An independent complete-image regression checks all nine geometry coordinates for two overlapping rotated anisotropic splats at three finite-difference steps, with unchanged support/order. This exposed f32 conic/opacity cancellation at the original dimensionless step; the log-scale/rotation Jacobian now uses .004, with unchanged assertion bounds.

`options.max_fitted_sh_degree` limits the appearance coordinates the optimizer
may change. It defaults to the compiled SH degree and must not exceed that degree
or three. Higher coefficients remain active during every render and are preserved
exactly from the seed, rather than being zeroed. The report records this subspace;
the output layout, renderer ABI, ownership and full-image acceptance stay unchanged.

Every record keeps its original index and node owner. The tool authenticates the seed manifest and validates complete disjoint source/output intervals against its nodes. The default `options.geometry_feasibility: "reject_whole_proposal"` retains the original behavior: any escaped three-sigma AABB rejects the entire proposal. The explicit `"per_representative_backtracking"` policy tries each representative's geometry at factors 1, 1/2, 1/4, 1/8, 1/16 and 1/32, always starting from its unchanged current geometry. If all six fail, only that representative's geometry freezes; its SH/opacity proposal and other representatives' feasible geometry remain eligible. Every resulting support still passes the original owner check, followed by the same complete training-image acceptance. Owners never expand, and the tool never splits, prunes, duplicates or transfers representatives. These bounds establish ownership, not a radiance-error certificate. The original-to-package association uses the pinned builder command record and manifest fingerprint; the tool explicitly does not claim to recompute the builder's canonical source fingerprint from the original PLY.

The hard ceilings are 256 steps, 180 seconds for teacher rendering/optimization/heldout evaluation, a 128 MiB contribution tape, 100,000 source records, 30,000 representatives and 512×512 pixels. Up to 16 training and 16 heldout views are admitted, with checked summed training RGBA allocations capped at 8 MiB: the previous two-512×512-view memory envelope is preserved. Twelve 256×144 training images use 7,077,888 bytes; twelve 384×216 images are rejected. Training teachers are released before sequential heldout evaluation. Each forward pass also has a pixel-visit ceiling: 16 million by default, at most 128 million. Exceeding a tape, teacher-image or visit limit fails the invocation without publishing a partial output; no contribution tail is omitted. A time limit during optimization retains the last completely accepted candidate, and any missing heldout evaluations are reported. A deadline before the first complete teacher/objective evaluation fails without an export. Bounded input hashing/decoding and output encoding are outside the fit timer; input admission time is reported separately, while export duration is not measured. The total working set includes source/candidate records, parameter/moment arrays, teacher images and projection data in addition to the tape.

Two optional controls preserve the original behavior when zero. `max_training_seconds`
limits teacher rendering and optimization inside the existing total deadline,
leaving remaining time for heldout evaluation. `training_stagnation_patience`
stops after the declared number of complete updates without a strict training
loss decrease. Consecutive steps that fail ownership at all four backtracking
scales report `ownership_stagnation`; this is an authoring-admission failure to
diagnose, not evidence of improved representatives. Neither control observes
heldout images or scores.

Each completed step records at most four global `trials`, with explicit
`accepted`, `training_loss_rejected` or `ownership_rejected` outcomes and counts
of requested, limited and frozen geometry. Counts belong to that particular
trial; interventions in rejected trials are not published updates. A `RejectWholeProposal`
ownership rejection may stop before considering all representatives. Final
deadline-interrupted trials appear separately in `interrupted_step_trials`
with `training_deadline`, preserving their unpublished status.

The new output directory contains `fitted.ply`, `fit.json` and the exact frozen `config.json`. The report records source/seed/manifest/output/executable hashes, admitted options, training losses, accepted/rejected steps, work counters, timer scope and any incomplete heldout evaluation. Work counters distinguish logical full-view passes, projection passes, completed gradient tiles, tile splits, rectangle-bound tests, pixel visits, actual contributions and peak owned tape capacity. The tape capacity is not a process RSS measurement. Heldout rendering occurs only after all training updates stop; heldout losses cannot select a checkpoint, modify learning rates or alter optimizer acceptance. Complete-rung mode does not measure PLY roundtrip image error, so its final acceptance requires reloading and rendering the exported PLY. Calibrated-cohort mode performs the explicit artifact check described above.

The tiled implementation completed the 24-step screen with all six heldout
evaluations: 27.85 seconds process wall time, 158,096 KiB peak RSS, and 62,343,232
bytes peak contribution tape. Training loss fell from .0036649433 to .0027832786;
five of six heldout CPU losses increased. All 21,087 record owners and source
intervals are retained. The output SHA256 is
`0e79b8915aef9e1e075a2d2b1e3fbc495d7b40c37a5f04f3b38d151e72f3dcf0`.
The exact config, failed attempts, successful run and output are under
`target/lod-roadmap/2026-09-07/icecream-fit-rung-screen-24*`.

The subsequent native GPU screen reloaded all three PLYs and completed all eight
frozen cameras at 1920×1080. Every capture used the same production `capture_lod_v9`
executable, fixed three-sigma support, 32-bit radix sorting, SH3, no MSAA and no
tonemapping. Each pose ran for 600 stationary frames with capture requests every
150 frames. Analysis required the last two common late samples to have identical
camera matrices and PNG hashes within each source, and exactly matching matrices
between sources. All configuration, image-hash, same-submission draw/count/image,
adapter and executable checks passed; no zero-instance samples were excluded.

The fitted output **fails the image-quality screen**. Its two training views
improve, while five of six heldout views worsen in both foreground PSNR and SSIM.
The far-front heldout view improves. Every seed and fitted view fails both the
35 dB foreground PSNR threshold and the 0.98 foreground SSIM threshold:

| Frozen camera | Split | Seed PSNR (dB) | Fitted PSNR (dB) | Seed SSIM | Fitted SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| `train_front` | Training | 26.1832 | 32.8149 | 0.91503 | 0.94546 |
| `train_rear` | Training | 24.1378 | 30.0638 | 0.89484 | 0.93354 |
| `heldout_front_far` | Heldout | 25.7477 | 27.9127 | 0.87556 | 0.89764 |
| `heldout_front_right` | Heldout | 25.1369 | 24.4808 | 0.89760 | 0.89458 |
| `heldout_front_left` | Heldout | 24.3579 | 23.6538 | 0.88503 | 0.88094 |
| `heldout_rear_right` | Heldout | 24.3994 | 23.6611 | 0.88572 | 0.87940 |
| `heldout_rear_left` | Heldout | 25.2131 | 24.3877 | 0.90316 | 0.89611 |
| `heldout_top` | Heldout | 23.6149 | 22.8718 | 0.89521 | 0.88757 |

Metrics compare sRGB RGBA8 captures decoded to linear premultiplied RGB. Alpha
MAE decreases on all eight fitted views, including heldout views whose RGB
worsens. Fitted heldout full-image alpha MAE is 0.00210–0.00980; foreground-only
alpha MAE is 0.03035–0.03924. Heldout silhouette IoU is 0.97146–0.97926, with mixed
changes from the seed. Fixed 128×128 local tiles expose larger residuals: the
worst fitted heldout tile per camera has foreground PSNR 16.21–20.66 dB, and the
minimum tile SSIM per camera is 0.4497–0.6083. Alpha and local diagnostics have no
invented pass thresholds; the complete tile records are retained in the report.

Every selected source capture submits 84,348 indirect instances; every seed and
fitted capture submits 21,087. The attested submitted-instance reduction is
exactly **4×** on every view. Flat shaders can discard instances after submission,
so these counts establish neither a post-frustum survivor ratio nor a GPU-time
speedup. This screen does not exercise package streaming, hierarchy selection,
mixed cuts, continuous camera motion or photographic supervision. The quality
and complete Phase 2 gates remain unsatisfied. No heldout-driven follow-up search
is authorized by the example config.

The local evidence is
[`icecream-fit24-gpu-v9-report.json`](../target/lod-roadmap/2026-09-07/icecream-fit24-gpu-v9-report.json)
(SHA256 `4b7a9301793d125c7e6bc4dfbbde19f9b40b7012a888a72dc0a32f8bb288f79c`)
and its
[`screen-plan.json`](../target/lod-roadmap/2026-09-07/icecream-fit24-gpu-v9/screen-plan.json).
The report pins every launch config, run record, capture JSONL, image hash and
submission record, and preserves distinct original/seed/fitted source identities:

| Artifact | SHA256 |
| --- | --- |
| Original Icecream PLY | `131205a37bfb30c90ddb8a5a686a67c27729c69c5ca9cec591c79d9b5fdc202a` |
| Complete branch-four depth-three seed PLY | `c301cd8021f4901510205984c383fe58b82c70fb59461e390f99da099eb6913a` |
| Reloaded fitted PLY | `0e79b8915aef9e1e075a2d2b1e3fbc495d7b40c37a5f04f3b38d151e72f3dcf0` |
| Shared `capture_lod_v9` executable | `43925a5dc52880285d216d506a2b6901ae73118be78bdfc999d88ee6c7ffa300` |
| Successful 24-step / 128M-visit input config | `daae701c8120f3274a39681502cd9714b4dbdc1ac38a9b03e7d4227d9c4a5ee2` |

Lineage validation rehashed all 21 exported package pages and checked the pinned
successful builder command, manifest, complete source/output partition, rung
sidecar and fit report. The authenticated canonical package fingerprint is
`78797d3a532ac3d4`; the original PLY's canonical fingerprint was not independently
recomputed. The actual GPU screen reused no earlier v5 teacher images. Fit-report
configuration comparison normalizes only fields declared as Rust f32 before
exact comparison; input config bytes remain SHA256-pinned, and f64 optimizer
values and integer budgets remain exact.

A subsequent twelve-camera protocol was frozen at
[`icecream-fit-orbit12-geometry128-256-protocol.json`](../target/lod-roadmap/2026-09-07/icecream-fit-orbit12-geometry128-256-protocol.json),
with config SHA256 `ee678a5c9a89748db989a8a65143aab006d9730522ac6cee5739bfe76eb6624a`.
It starts from the authenticated seed and enables the mean, log-scale
and rotation derivatives. Twelve closed-form orbital poses span two elevations
and two distances at 256×144; six new deterministic withheld poses are frozen
before fitting. The previously observed six heldouts are retained only as
explicitly exploratory final diagnostics. Each declared run is limited to 128
steps, 120 seconds of training, the unchanged 180-second total deadline, and
eight stagnant updates. No heldout-based checkpoint selection or follow-up sweep
is permitted.

The first `capture_lod_v13` invocation completed in 34.75 seconds process wall
time with 159,912 KiB peak RSS. Two accepted updates reduced training loss from
.00235705 to .000781974; eight subsequent updates failed ownership at all four
global backtracking scales, producing `ownership_stagnation`. All twelve CPU
evaluations completed. The training trace alone identifies an authoring
feasibility blocker: one constrained representative can reject the entire
scene's update. The GPU-screen preparer rejects this outcome before any capture;
it is not a fourfold image-quality result.

The explicit per-representative policy is a correction for that demonstrated
training blocker. Its separately frozen config is
[`icecream-fit-orbit12-geometry128-256-feasible.json`](../target/lod-roadmap/2026-09-07/icecream-fit-orbit12-geometry128-256-feasible.json),
SHA256 `19db318d7468ff581f572e75273c8537f6511c713bbf7e58f9dd73e2910ad578`.
Its generator verifies that only `output_directory` and
`options.geometry_feasibility` change from the frozen orbital config. The
companion protocol records the eight ownership-blocked training steps; no
heldout score chooses the correction, a hyperparameter or a checkpoint. The six
orbital withheld CPU results have now been observed and must be described as
reused evaluation after this correction, not a second fresh blind test. The
predeclared GPU subset remains training indices 0 and 6 plus those six orbital
withheld poses, with the original exploratory six omitted.

That single `capture_lod_v15` invocation completed in 123.326 seconds process
wall time with 159,160 KiB peak RSS. It accepted 33 updates before the declared
120-second training deadline, reducing the twelve-view training loss from
.002357050229 to .0000951489233 (95.96%). All twelve final CPU evaluations
completed inside the unchanged 180-second total limit. The contribution tape
peaked at 63,550,956 bytes; retained training images occupied 7,077,888 bytes.
No proposal failed ownership. Across the accepted trials, geometry was limited
14 times and frozen 41 times; these are intervention observations summed across
updates, not counts of distinct representatives. Nine additional global trials
were rejected by the training objective and did not publish their geometry or
appearance updates. The last accepted trial froze one representative's geometry.
This resolves the demonstrated all-or-nothing training-admission failure without
changing owners, cardinality, camera selection, learning rates or resource limits.

The subsequent GPU screen reloaded the original, seed and fitted PLYs using that
same v15 executable. All three 1920×1080 captures completed, with the original
600-frame stationary pose duration and one capture request every 150 frames.
The strict analyzer passed every config/source/executable/adapter identity,
same-submission image/count/draw attestation, exact camera-matrix equality and
two-late-sample image stability check. No zero-instance sample was excluded.
The fitted output improves foreground RGB PSNR and SSIM over the seed on all
eight poses, but **every seed and fitted pose still fails both the 35 dB and
0.98 thresholds**:

| Frozen camera | Evaluation role | Seed PSNR (dB) | Fitted PSNR (dB) | Seed SSIM | Fitted SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| `train_orbit_0_00` | Training index 0 | 25.8886 | 34.7471 | 0.91188 | 0.95504 |
| `train_orbit_1_00` | Training index 6 | 23.7140 | 31.7919 | 0.85401 | 0.92864 |
| `withheld_orbit_00` | Reused orbital evaluation | 24.8981 | 29.6840 | 0.88474 | 0.92829 |
| `withheld_orbit_01` | Reused orbital evaluation | 24.4383 | 28.5040 | 0.87870 | 0.92166 |
| `withheld_orbit_02` | Reused orbital evaluation | 24.4997 | 29.2144 | 0.87328 | 0.92413 |
| `withheld_orbit_03` | Reused orbital evaluation | 24.5813 | 29.4201 | 0.88099 | 0.93220 |
| `withheld_orbit_04` | Reused orbital evaluation | 24.1199 | 28.1245 | 0.87078 | 0.91916 |
| `withheld_orbit_05` | Reused orbital evaluation | 24.5802 | 29.7119 | 0.87841 | 0.92796 |

Alpha MAE decreases and silhouette IoU increases on all eight poses. The fixed
128×128 tile diagnostics also improve over the seed in each camera's worst
foreground PSNR and minimum foreground SSIM, while retaining substantial local
errors. The following values describe the fitted output; alpha and local
diagnostics have no added pass thresholds:

| Frozen camera | Full-image alpha MAE | Foreground alpha MAE | Silhouette IoU | Worst local PSNR (dB) | Minimum local SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_orbit_0_00` | 0.001340 | 0.004677 | 0.98895 | 31.8240 | 0.81096 |
| `train_orbit_1_00` | 0.000491 | 0.007284 | 0.98312 | 30.1216 | 0.87338 |
| `withheld_orbit_00` | 0.000933 | 0.007509 | 0.98420 | 25.0107 | 0.63178 |
| `withheld_orbit_01` | 0.001120 | 0.008952 | 0.98213 | 22.8934 | 0.87548 |
| `withheld_orbit_02` | 0.001035 | 0.008237 | 0.98417 | 25.0930 | 0.83893 |
| `withheld_orbit_03` | 0.000961 | 0.007372 | 0.98639 | 26.4664 | 0.90413 |
| `withheld_orbit_04` | 0.001293 | 0.009846 | 0.97995 | 25.4621 | 0.81137 |
| `withheld_orbit_05` | 0.001064 | 0.008564 | 0.98321 | 26.3346 | 0.80388 |

All selected captures again attest 84,348 original versus 21,087 seed/fitted
indirect instances, an exact 4× submitted-instance reduction. The same limits
apply: this does not measure post-frustum survivors, whole-frame speedup,
streaming, mixed cuts or continuous-camera quality. The six orbital evaluation
poses were predeclared before the first orbital fit, but their CPU results had
been observed before this training-only feasibility correction; this is reused
evaluation, not fresh blind qualification. The 35 dB/.98 and complete Phase 2
gates remain unsatisfied. The earlier 24-step failure and v13 ownership-stagnation
result remain separate evidence and were not replaced by this result.

The complete v15 evidence is
[`icecream-orbit12-feasible-gpu-v15-report.json`](../target/lod-roadmap/2026-09-07/icecream-orbit12-feasible-gpu-v15-report.json)
(SHA256 `a7a520123cd4e749fe83088e37f7cd07fba447989c5efe80740639a2f666b39d`),
with the authenticated
[`screen-plan.json`](../target/lod-roadmap/2026-09-07/icecream-orbit12-feasible-gpu-v15/screen-plan.json)
and preserved input/export lineage. The original and seed hashes above are
unchanged; the new artifacts have distinct identities:

| Artifact | SHA256 |
| --- | --- |
| Feasibility-policy input config | `19db318d7468ff581f572e75273c8537f6511c713bbf7e58f9dd73e2910ad578` |
| Completed fit report | `35b6cced81691bb49c3c18fdf713090195d3b5e831c948ce054cdf69c124e1aa` |
| Reloaded fitted PLY | `a4e885b010fd90299f5d6e563fb854f133be9eb687fb3becb49aae856542a4be` |
| Shared fit/capture v15 executable | `a737c2e599fe1e39c6f072fa54b42fbcd9e3e58ceab6e17f74008f7d86b51a4c` |
| Prepared GPU screen plan | `14cfcb60c96ed44b20e9b24648960d23b4281d284d8f3a20cd2e9f310a47e1ec` |

A single degree-zero ablation is now prepared, not executed. Inspection of the
authenticated source found no higher SH fields, and every seed higher coefficient
is zero. The previous fit nevertheless introduced higher SH in all 21,087
representatives (coefficient RMS .02615). Their sphere-average per-channel
directional standard deviation has p90 .05162 and p99 .09354 in sRGB before
clamping. This identifies an avoidable angular degree of freedom under twelve
training views; it does not prove that directional appearance is always harmful,
since a reduced representative can use it to approximate changing occlusion.
The last accepted training updates were still improving at the deadline, so the
failed screen also does not establish an inherent fourfold representation limit.

The [paired config](../target/lod-roadmap/2026-09-07/icecream-fit-orbit12-geometry128-256-feasible-sh0.json)
has SHA256 `0bd1a1924dae9eaac9efe05aa88d95a4a9dcad0ad141cb604861756f6e7473bc`.
Only the output directory and explicit fitted degree zero differ from the pinned
feasibility config. Its [protocol](../target/lod-roadmap/2026-09-07/icecream-fit-orbit12-geometry128-256-feasible-sh0-protocol.json)
preserves the seed, twelve training cameras, geometry/optimizer policy, 120-second
training and 180-second total limits, and original memory/visit ceilings. Root
qualification must reload the result and apply the same 4×, 35 dB/.98 GPU screen;
the orbital withheld views remain reused evaluation. No sweep or heldout-based
checkpoint choice is authorized by this protocol. Training at 256×144 while
screening at 1920×1080 remains a separate filtering/detail mismatch that this
ablation does not resolve.

The single degree-zero invocation completed 43 accepted updates before the
unchanged training deadline; the prior degree-three fit completed 33. Its
training loss fell from .002357050229 to .0000906544787 (96.15%), versus
.0000951489233 for the prior fit. The fit timer, including final CPU evaluation,
reported 123.137 seconds. All twelve evaluation images completed. Peak owned
tape was 63,890,736 bytes and training teachers occupied the unchanged 7,077,888
bytes. No trial failed ownership; the accepted trials contain 18 limited and
40 frozen geometry interventions, summed across updates. Thirteen other trials
were rejected by training loss and did not publish their proposals.

All six reused orbital CPU losses decrease relative to the prior fit:

| Reused evaluation pose | Prior degree-three loss | Degree-zero loss | Decrease |
| --- | ---: | ---: | ---: |
| `withheld_orbit_00` | .0001433961 | .0000920068 | 35.84% |
| `withheld_orbit_01` | .0002084990 | .0001558731 | 25.24% |
| `withheld_orbit_02` | .0001708942 | .0001155211 | 32.40% |
| `withheld_orbit_03` | .0001594626 | .0001018295 | 36.14% |
| `withheld_orbit_04` | .0002395259 | .0001499599 | 37.39% |
| `withheld_orbit_05` | .0001556611 | .0001072696 | 31.09% |

The six exploratory reused losses also decrease. These are the low-resolution
CPU RGB-plus-alpha objective, not GPU PSNR or SSIM. The differing accepted update
counts under the same time ceiling prevent attributing the entire difference
to SH degree alone. The [fit report](../target/lod-roadmap/2026-09-07/icecream-fit-orbit12-geometry128-256-feasible-sh0/fit.json)
pins exported PLY SHA256
`dddf6dd795657ad9379eb382a758375883e1ff28870edb829a0c2eb4f33340eb`.
Its [execution record](../target/lod-roadmap/2026-09-07/icecream-fit-orbit12-geometry128-256-feasible-sh0-run.json)
was recorded after completion from observed tool arguments and exit status;
it preserves that provenance and the actual invocation.

The [single GPU screen plan](../target/lod-roadmap/2026-09-07/icecream-sh0-gpu-screen/screen-plan.json)
completed all three runs and **fails the unchanged GPU quality gate**. It reloaded original, seed
and degree-zero fitted PLYs using the same frozen `capture_lod_gps_sh0` executable
(SHA256 `3a65e96b1ad1e1f2066b744cb9c97c3b7bdd9da14995eaa239f83022460ebb18`).
The unchanged eight poses, 1920×1080 viewport, stationary duration, capture
cadence, 35 dB/.98 image thresholds and 4× attested instance gate are retained.
The [preparer/analyzer](../target/lod-roadmap/2026-09-07/icecream_sh0_gpu_screen.py)
reuses the prior capture, metric and identity checks with explicit ablation
lineage validation. Every selected capture attests the original 84,348 versus
21,087 fitted instances, preserving the fourfold submitted-instance reduction.
The [completed report](../target/lod-roadmap/2026-09-07/icecream-sh0-gpu-screen-report.json)
records fitted foreground PSNR of 29.693–34.748 dB and SSIM of .927876–.956912.
All eight fitted views improve over the seed, but every view still fails both
35 dB and .98. The three capture processes took 16.56, 10.55 and 10.40 seconds,
respectively; these process times are not frame-performance measurements.
This closes the single ablation with a negative acceptance result. It does not
justify a quality certificate or another fit sweep. The training-resolution
mismatch, useful reduced representatives and broader LoD quality remain open.

## Native-pixel filter correction on 2026-09-08

The completed SH0 screen identifies a concrete forward-objective mismatch: the
fit used +0.3 physical-pixel Mip variance at 256×144, while qualification used
1920×1080. Scaling covariance after filtering at the training resolution does
not preserve the native footprint or its determinant opacity compensation.
The optional reference-pixel path above corrects that mismatch within the
existing allocation and work ceilings. One focused regression compares its
sampled RGBA against the corresponding pixels of a complete production-oracle
image, then checks a translation derivative against native-image finite
differences. This is a CPU contract check, not GPU image-quality acceptance.

One [follow-up configuration](../target/lod-roadmap/2026-09-08/icecream-fit-native-pixels-sh0.json)
and [frozen protocol](../target/lod-roadmap/2026-09-08/icecream-fit-native-pixels-sh0-protocol.json)
were frozen before the single completed run. They retain the same original and complete fourfold
seed identities, twelve training cameras, six reused orbital evaluation cameras
and six exploratory reused views, SH0 restriction, optimizer, ownership bounds
and 120-second training/180-second
total fit limits. The allocated grid is 240×135 with stride eight into the
1920×1080 reference viewport and offset `[4,4]`; retained training RGBA occupies
6,220,800 bytes. One invocation is allowed, with no automatic retry or sweep.

The acceptance requirements remain fourfold attested submitted-instance
reduction, foreground PSNR ≥35 dB and foreground SSIM ≥0.98 at every predeclared
1920×1080 GPU pose after reloading the exported PLY. The previously observed
heldouts remain reused evaluation. Correcting the objective does not establish
that 21,087 representatives satisfy these requirements; that remains an
empirical result, and the last completed quality screen still fails.

The native-pixel projection/translation-gradient regression passes. The single
CPU fit completed in 122.91 seconds process wall time with 50 accepted updates
before the unchanged training deadline. Sparse training loss fell from
0.0023028635 to 0.0000776490 (96.63%); all twelve reused evaluation views improved
against their own seed losses. Training images occupied 6,220,800 bytes and the
contribution tape peaked at 66,486,140 bytes. These losses use a different pixel
grid from earlier runs and cannot establish a cross-run image-quality gain.

The [execution record](../target/lod-roadmap/2026-09-08/icecream-fit-native-pixels-sh0-run.json)
pins executable, config, protocol, report and exported PLY identities; the
[analysis](../target/lod-roadmap/2026-09-08/icecream-fit-native-pixels-sh0-analysis.json)
records the scope. The new export has not received the full 1080p hardware GPU
screen: the resumed environment exposes only software Vulkan. No second fit or
parameter sweep was run, and no quality gate was relaxed.

### Completed native-pixel hardware screen (2026-09-08)

The previously fitted native-pixel SH0 rung has now completed the unchanged
eight-view 1080p NVIDIA Vulkan screen. All views improve over the seed, but
**all eight fail the 35 dB/.98 gate**: PSNR ranges from 29.237 to 34.961 dB and
SSIM from .931243 to .958659. The original submits 84,348 records and the fitted
rung 21,087 (exactly 4x fewer); this is a fixed-support, flat-rung comparison,
not a post-frustum reduction or complete-frame performance result.

The [report](../target/lod-roadmap/2026-09-08/production-closeout/quality-report.json)
records camera/image/count attestations and artifact hashes. The frozen capture
executable SHA-256 is `e0f65800e97b5c4a3e1c6cd310fccc75fef3230f062d8b1ae7d59d8d40800ac2`;
the fitted PLY is `43e053a281610ffd39d9eb1396ada8b74633e5768aeaf9b98f29c2f8e4f2d00a`.
No additional fit, parameter sweep, certificate, or relaxed gate was used. This
closes the missing hardware screen and leaves representative quality unresolved.
