# Canonical Garden ABI 16 preprocessing and qualification record

This report identifies the 2026-08-20 external-memory Garden package and
records its artifact identity, CPU selector evidence, fixed-view boundary
oracle, and runtime GPU qualification. The package lives under
`target/` and is not a repository fixture. Its ABI 16 bytes, build envelope,
manifest validation, CPU selector distance response, and authenticated
1920x1080 boundary evidence are recorded below. On 2026-08-22, the complete
persistent-edge temporal gate passed against the final shader tree in both the
standard and `precompute_covariance_3d` profiles. The endpoint-radiance K2 gate
also passed in both profiles, and the generic GPU qualification script passed
on an RTX PRO 6000 Blackwell through Vulkan driver 610.43.02. Measurements from
the superseded shared-frame-clock renderer remain below as historical
provenance. The broader current-tree ignored Garden matrix, browser/CDN, and
cross-adapter qualifications remain bounded separately.

## Source and build

| Property | Value |
| --- | --- |
| Source | external canonical `garden.ply`, not tracked in this checkout |
| Source bytes | 1,447,027,964 |
| Source records | 5,834,784 SH3 Gaussians |
| Source SHA-256 | `16701d5e0630dfaca74f8794ed7ce2aa23fa922f87dc09a7e37484e8d3f82d5a` |
| Output | `target/lod-packages/garden-sh3-spatial-v4-host-morton` |
| Builder/reducer ABI | 16 / MomentMerge reducer 4 |
| Morph sidecar | schema 1 |

The build used explicit GPU preprocessing and bounded run sorting. Canonical
Morton keys and source-index tie breakers were authored on the host and
uploaded to the GPU sorter. External merge, the v4 hierarchy and spatial
fitter, morph-map construction, page encoding, verification, and atomic
publication remained on the CPU. The equivalent invocation is:

```bash
BGS_GARDEN_PLY=/absolute/path/to/hash-matched/garden.ply

cargo +1.95.0 build --locked --release --no-default-features \
  --features lod_build_sh3 --bin build_lod

/usr/bin/time -v target/release/build_lod \
  --input "$BGS_GARDEN_PLY" \
  --output target/lod-packages/garden-sh3-spatial-v4-host-morton \
  --gpu-preprocess
```

The builder reported 503.103 seconds total. The enclosing command took
8:24.48 wall time and reached 380,828 KiB maximum resident memory. These are
whole-build measurements; the superseded ABI 15 stage timings are not reused.

This package supersedes the earlier spatial-v4 build whose GPU preprocessing
shader derived Morton keys from floating-point coordinates. Values on Morton
quantization boundaries could then make package ordering and fingerprints
adapter-dependent. The repaired path applies the canonical host quantizer
once, uploads the exact 64-bit key plus source-index total-order tie breaker,
and leaves bounded ordering and readback validation to the GPU. The shader no
longer recomputes a key, so adapter arithmetic cannot change package identity.

## Artifact identity

The validated manifest contains 6,517 nodes, 6,517 pages, and 6,668,314 stored
records in three range-addressable shards. Stored records include original
leaves and internal representations; they are not an active-cut or draw count.
The four package objects total 1,608,978,507 bytes, approximately 1.50 GiB.

| Object | Bytes | SHA-256 |
| --- | ---: | --- |
| `scene.gsplatlod` | 8,087,735 | `67b9119222e1435fb88755698dcd916e608c9cd21c1417b687a7cce663729600` |
| `pages/shard-000000.bgslodpack` | 536,660,028 | `d8884945ff558d8a231d48511900f9cc97df407c9bd442d1a8ab35bc9a0766ea` |
| `pages/shard-000001.bgslodpack` | 536,660,028 | `1232414ca7f0addbd4524516d06c205832468f685eb897c60e53412e24608504` |
| `pages/shard-000002.bgslodpack` | 527,570,716 | `cdc3c896fba1f1aae469c09e913ba075c824fb6b8e0434b08206b48a03c9a8b2` |

ABI 16 retains semantic manifest v3 and page schema v2. Its required feature
bit adds a fail-closed morph sidecar without relabeling page bytes or mutating
readable ABI 15 packages. Morph schema 1 stores one monotone run length per
parent-local record; child records are concatenated in manifest-child then
page-local record order.

## Raster and spatial-fit contract

The renderer adds `0.3 px²` of physical-pixel variance to projected covariance
and scales peak alpha by `sqrt(det(C) / det(C + 0.3 I))`, preserving integrated
screen-space alpha for valid projected covariance. The WGSL projection uses
two coordinate units per physical pixel, so covariance is four times larger
there and the equivalent shader-space diagonal addition is `1.2`. Flat source
records retain their opacity-adaptive support cutoff. LoD candidates retain at
least their authored 3-sigma support. The v4 fitter evaluates source references
with the former rule and emitted representatives with the latter.

Within each same-depth future-parent cohort, the fitter considers every pair of
nodes whose authored support bounds touch, not only adjacent Morton entries.
The validated branching bound of 32 limits this to 496 node-pair checks. A
deterministic 3x3 tangential grid and normalized `0.0625x..4x` projection-scale
ladder cover at most 4,464 seam probes per cohort. An accepted edit keeps record
positions fixed, widens tangent covariance only within the exact touching
pair's authored support envelope, reapplies the all-view opacity ceiling,
improves the target seam, and does not regress any affected sampled seam or
cohort composited error at any sampled scale. An unsafe measured seam instead
raises selection-visible error so it can refine at ordinary quality.

The fresh Garden build reported:

| Boundary class | Count | Jointly fit by v4? |
| --- | ---: | --- |
| Within-cohort authored-support-touching pairs | 2,260 | scope total |
| Within-cohort pairs with retained source partitions | 1,982 | yes, measured |
| Within-cohort source-less pairs | 278 | no; reported unmeasured |
| Potential cross-future-parent pairs | at most 255,104 | no; conservative upper bound |
| Mixed-depth selected-cut boundaries | not enumerated as one build count | no |

The cross-future-parent value is deliberately an upper bound, not a claim that
all 255,104 pairs touch. Source-less coarse pairs are not assigned artificial
infinite error, avoiding blanket refinement when no comparison source is
available. Same-depth boundaries across future parents and mixed-depth cut
boundaries are qualified by the image oracle rather than claimed to be jointly
fit by the builder. The completed authenticated result follows.

## Authenticated 1920x1080 boundary oracle

`headless::canonical_garden_abi16_node_boundary_oracle` authenticated the
canonical PLY byte hash and record count plus the manifest and all three shard
hashes above before evaluating the host-Morton package. The 1920x1080 run
passed in 36.44 seconds at approximately 4.82 GB maximum resident memory.

Its mandatory aggregate endpoint and local-jump gates retained this coverage:

| Boundary class | Endpoint coverage (px) | Jump coverage (px) |
| --- | ---: | ---: |
| Same-depth | 1,650 | 1,258 |
| Mixed-depth | 252 | 208 |
| Same-parent | 606 | 468 |
| Cross-parent | 1,296 | 998 |

Across those mandatory aggregate gates, the maximum RGB enrichment was 1.022,
the maximum alpha enrichment was 1.057, the maximum RGB-jump enrichment was
1.022, and the maximum alpha-jump enrichment was 1.027. Every signed
endpoint-control gap was well under 0.02. The matched full-image results were:

| View | PSNR (dB) | Alpha MAE |
| --- | ---: | ---: |
| Viewer auto-frame, q=.65 | 39.765 | 0.000059 |
| Far, q=.65 | 35.477 | 0.000048 |

These are the authoritative aggregate values available from the final run; no
more granular per-class metric tuple is inferred. They qualify the sampled
boundary classes and fixed views. Current temporal runtime evidence is recorded
below. Static, interactive, and debug measurements from the former presentation
implementation are preserved separately as historical data.

## All-resident CPU selector distance response

The authenticated scene bounds are:

- minimum `[-118.729537964, -130.432022095, -121.283477783]`;
- maximum `[137.847320557, 109.880554199, 136.600799561]`;
- center `[9.558891296, -10.275733948, 7.658660889]`;
- radius `R = 217.994338989`;
- viewer auto-frame distance `474.641113281 = 2.177309275R`.

The deterministic sweep uses a 1920x1080-equivalent perspective, 45-degree
vertical field of view, 0.1 near plane, the normalized `(0, 1.5, 5)` view
direction, zero hysteresis, an eight-million active-record ceiling, and an
all-resident manifest. Counts are complete scene-wide frontier records before
GPU frustum compaction, not post-cull indirect draw counts.

| Distance | q=.35 | q=.50 | q=.65 |
| --- | ---: | ---: | ---: |
| auto-frame (`2.177309275R`) | 11,398 (0.195%) | 191,868 (3.288%) | 3,556,571 (60.955%) |
| `2.4R` | 11,398 (0.195%) | 127,356 (2.183%) | 3,298,607 (56.533%) |
| `4R` | 1,425 (0.024%) | 39,172 (0.671%) | 1,774,789 (30.417%) |
| `6R` | 1,425 (0.024%) | 1,425 (0.024%) | 994,565 (17.045%) |
| `10R` | 179 (0.003%) | 1,425 (0.024%) | 611,154 (10.474%) |

Percentages are relative to the 5,834,784-record source. At q=.65 the selected
frontier falls 82.816% from auto-frame to `10R`, while remaining below the exact
leaf count at every sampled distance. This is evidence of useful quality and
distance response rather than a policy that merely selects nearly all leaves.
It is a selector/manifest result, not evidence of rendered fidelity.

Reproduce the authenticated count table without opening a GPU adapter:

```bash
BGS_GARDEN_LOD="$PWD/target/lod-packages/garden-sh3-spatial-v4-host-morton/scene.gsplatlod" \
cargo +1.95.0 test --locked --no-default-features \
  --features "headless testing" --test lod_quality_render \
  canonical_garden_manifest_cpu_selector_has_useful_distance_response -- \
  --ignored --nocapture --test-threads=1
```

## Recorded host qualification (historical snapshot)

The host/core matrix recorded with the former presentation implementation was:

| Slice | Result |
| --- | ---: |
| Full `headless testing` library suite | 608 passed, 6 ignored, 0 failed |
| Exact multi-step request-ownership and retained-status regressions | passed |
| Default `cargo check` and library Clippy | passed |
| Minimal SH0 and `precompute_covariance_3d` check/library-test Clippy | passed |
| `lod_build_sh3` check and all-targets Clippy | passed |
| Formatting and diff checks | passed |

These gates cover deterministic hierarchy, format, sidecar, selector, package,
bridge, and CPU raster/morph contracts. They do not substitute for executing
the current package through a real GPU renderer.

## Runtime presentation and bounded fallback

For a direct ABI 16 substitution on a capable adapter, the implementation uses
the validated monotone sidecar to expose immutable adjacent parent/children
edges. Each retained render view owns independent persistent displayed and
desired weights for those edges; there is no shared frame clock. Ordinary
`Dynamic` updates evaluate the current view and target statelessly and follow
that desired value exactly in the next radix-proven drawable publication.
Independent edges may overlap and a fractional edge table may be a stationary
ACTIVE fixed point. Positions and covariance map between exact endpoints. Each
endpoint's SH radiance is evaluated independently and combined in linear light
by the same optical-depth terms used for alpha. Parent optical depth is divided
across each mapped child run, and an unfiltered parent-plus-child union is never
presented.

Only exceptional recovery uses the bounded per-update slew: late-residency
activation, `Frozen`-to-`Dynamic` resumption, and the first valid evaluation
after an invalid-pressure hold. Invalid evaluation retains the last drawable
weights and reports degradation. Each private view compacts and radix-sorts its
own suffix; render `Cleanup` aggregates all expected retained-view consumers
before a prepared table becomes ACTIVE. A missing consumer keeps a prepared
table unactivated or an ACTIVE table in an explicit degraded hold, and cannot
satisfy or retire the live target.

Morphing is optional presentation capability. The implementation uses a
complete `BoundedHardCohort` transition for a readable pre-ABI16 package, a
transaction without a usable direct map, a morph payload beyond the adapter's
buffer or storage-binding limit, or a package-authored sticky bounded-hard
decision when the exact target fits the atlas but the temporary morph union
does not. Multi-subview rendering uses the aggregate barrier and is not itself
a categorical-fallback condition. The render stage must not upgrade a capacity
decision. Stale topology or predecessor evidence requests a fail-closed replan.
If the exact target itself cannot fit, the runtime keeps the prior valid cut and
reports capacity degradation instead of exposing an incomplete cut.

Package request ownership follows drawable presentation. A categorical legacy
cohort remains an intermediate topology step until selector convergence. An
ABI 16 ACTIVE fractional table may itself own the stationary request once the
selector is stable, every expected consumer has coherent aggregate evidence,
and no edge is in late-residency, `Frozen`-resume, or invalid-pressure recovery.
While a replacement is pending, public status retains the prior committed view
and submitted-candidate counts without misreporting the replacement as ACTIVE.

A current-runtime settlement gate must require an ACTIVE candidate, a coherent
all-consumer aggregate, zero recovery lag, invalid pressure, missing consumers,
and rendered requested pages, plus `target_satisfied = true`, no degradation,
and exact agreement with public count and quality status. A stable fractional
ABI 16 table may meet that contract; it is not an intermediate merely because
its edges are fractional. Categorical legacy work must additionally have no
remaining topology-transition provenance. A debug gate is a presentation and
cut-preservation contract during preset changes; it does not independently
redefine package settlement.

## Current persistent-edge temporal qualification

On 2026-08-22, the complete authenticated Garden temporal gate passed against
the final endpoint-radiance, visibility, and fixed candidate-support shader
tree in both covariance profiles:

| Profile | Result | Time |
| --- | ---: | ---: |
| Standard | passed | 433.57 s |
| `precompute_covariance_3d` | passed | 426.69 s |

Both profiles reproduced the same Frozen controls. Each retained one fixed
logical endpoint, used no immutable-table upload or weight write during the
measured camera motion, allocated no replacement blend buffer, and observed no
lagging or bounded-hard frame:

| Direction | Active records | Peak blend edges | Lagging frames | Table uploads | Weight writes | Buffer allocations | Hard frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Refining | 4,258,853 | 507 | 0 | 0 | 0 | 0 | 0 |
| Coarsening | 5,146,656 | 621 | 0 | 0 | 0 | 0 | 0 |

The ordinary all-resident `Dynamic` traces were fractional on every measured
sample. They changed logical endpoints while retaining zero lag, allocation,
and hard-fallback events:

| Direction | Fractional samples | Active records | Endpoint changes | Table uploads | Weight writes | Buffer allocations | Lagging frames | Hard frames | Authored publications | Maximum authored hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Refining | 48/48 | 4,258,853 -> 5,074,997 | 11 | 11 | 34 | 0 | 0 | 0 | 13 | 2 frames |
| Coarsening | 48/48 | 5,146,656 -> 4,416,507 | 13 | 12 | 34 | 0 | 0 | 0 | 14 | 2 frames |

The camera-conditioned reversal and fixed-pose proof completed 141 moving
samples over 48 distinct poses and compared six matched-pose keyframes. It
reached 619 peak blend edges, 37 immutable-table uploads, 104 weight writes,
zero buffer allocations, and 38 authored publications with a maximum
two-frame authored hold.

The shader endpoint-radiance and visibility test
`headless::authenticated_abi16_k2_morph_radiance_visibility` passed in the
standard and `precompute_covariance_3d` profiles. The complete generic
`tools/run_lod_gpu_qualification.sh` suite also passed on an RTX PRO 6000
Blackwell with the Vulkan backend and driver 610.43.02. The supplied execution
did not establish the optional cross-adapter branch. These results qualify the
current temporal and generic GPU-script surfaces; they do not imply that the
broader ignored Garden static, interactive, debug, or automatic/native matrix
was rerun.

## Historical Garden GPU qualification (superseded runtime)

The numbers in this section are preserved exactly from the renderer that
preceded persistent per-view independent edges. The host-Morton package and
hash-matched PLY completed the recorded bracketed headless matrix with the
standard renderer and with `precompute_covariance_3d`. Each profile ran the
automatic-bridge and native preprocessed-package smoke gates before and after
the Garden gates. These timings and outcomes are historical provenance, not a
current pass claim. The replacement temporal qualification is recorded above;
the broader current-tree ignored Garden matrix has not been established by the
supplied executions.

| Gate | Standard | `precompute_covariance_3d` |
| --- | ---: | ---: |
| Leading automatic/native smoke | passed / passed | passed / passed |
| Frozen controls plus bidirectional dynamic temporal | 49.43 s | 48.02 s |
| Static viewer-auto frame | 35.46 s | 28.52 s |
| Eight-scenario interactive trace | 69.09 s | 62.39 s |
| Debug sparse upload and pixel transition | 10.79 s | 10.73 s |
| Trailing automatic/native smoke | 4.01 s / 1.59 s | 4.24 s / 1.69 s |

In both recorded temporal reruns, the 48-frame flat and Frozen controls produced
zero logical endpoint changes, zero morphs, and zero hard cuts in both
directions. The dynamic
trace refined from 851,219 to 1,051,881 selected records with one ACTIVE
endpoint backed by one completed authored morph, then coarsened from 1,538,294
to 1,495,293 with two endpoints backed by two completed authored morphs. The
refinement trace observed 14 morph frames and 12 positive advances; coarsening
observed 39 and 34. Neither profile used a hard fallback, and the unchanged
temporal image-jump thresholds passed.

The recorded static viewer-auto endpoint contained 3,556,571 records in 3,474
ranges. Against the matched flat rendering, the standard profile measured
61.17478 dB full-image PSNR, 38.56104 dB foreground PSNR, 0.999607872
luminance SSIM, and
0.994534556 foreground IoU. The precomputed-covariance profile measured
61.17503 dB, 38.56129 dB, 0.999607941, and 0.994534556 respectively. The same
spatial-fidelity thresholds were used in both profiles.

All eight interactive scenarios reached the strict fixed point without a hard
cut. The initial, recovered, and returned high-quality overview reproduced the
same logical-cut digest `c1faf034752a7417`, 3,291,439 selected records, and
0.9983079 achieved ratio. The cold and returned low-quality overview reproduced
digest `04a579059d9d024d`, 11,398 records, and ratio 0.7. Fixed-pose refinement
completed only authored refinement substitutions, and the fixed-pose return
completed only authored coarsening substitutions; camera-moving scenarios may
legitimately mix directions. The longest return required 2,245 request frames
in the standard profile and 2,252 in the precomputed profile. Each completed
129 authored endpoint identities and matched all 129 to completed morphs; the
standard/precomputed runs observed 1,804/1,805 morph frames and 1,542 positive
advances. The gate's 3,600-frame limit was a test-only cross-adapter allowance
derived from the branch-eight operational budget-filling estimate
`172 * 17 + 120 + 10 = 3,054` frames. It is not a strict
scheduler bound. Its `1/24` morph-energy policy and 256-substitution cohort cap
describe the superseded scheduler; current ABI 16 presentation has no shared
cohort clock, while those bounds remain relevant to categorical legacy work.

The debug gate changed `Off -> Page -> Off` on one unchanged ACTIVE cut. Every
captured frame was nonblack and nonempty, retained the logical cut and a
nonzero indirect draw, and queued the `LOD_DEBUG` raster variant whenever the
Page binding was ready. Two consecutive Page frames were identical, and the
restored Off endpoint reproduced the original alpha silhouette exactly
(alpha MAE 0, foreground IoU 1). Page exposed all 12 supported hue bins and
changed 7,640 foreground pixels, or 96.7946% of the comparison mask, relative
to Off. The gate deliberately does not require RGB continuity across Page's
categorical color boundaries. Standard/precomputed sparse telemetry used
7,999,488/7,895,040 atlas records, 72,417,280/72,294,400 Page-camera upload
bytes, and remained inside the 67,108,864-byte, 256-slot per-frame cap.

The bracketed automatic smoke retained the deterministic 1/3/7/11 quality
sweep, the native package smoke passed, and the retained-status regression
passed. Earlier ABI 15 image, timing, and debug measurements remain excluded
because they identify different package bytes and a different reducer.

## Remaining bounded caveats

The 2026-08-22 temporal qualification authenticated the canonical fixed-key
headless Garden workload on one RTX PRO 6000 Blackwell/Vulkan configuration in
both covariance profiles. Together with the passing endpoint-radiance K2 gate
and generic GPU qualification script, it qualifies the final tree's measured
temporal path on that host. It does not establish a real-time latency SLA,
browser/CDN behavior, cross-adapter behavior, or seamless presentation during
arbitrary shader hot reload. The broader current-tree ignored Garden static,
interactive, debug, and automatic/native matrix remains pending until those
gates are explicitly rerun in both covariance profiles.
