# Canonical Trellis LoD quality report

This report records a passing full-profile qualification of the current
production-default LoD selector on the pinned Trellis scene. It is evidence for
this artifact, renderer convention, and camera contract. It is **not** a
universal mapping from the quality slider to PSNR, nor a promise that unrelated
scenes will reach the same savings at the same quality.

## Provenance and method

- Artifact size: 112,899,460 bytes
- Artifact SHA-256:
  `fbe9d96b6689a78228c121e5f1bc8c5ccc32cef1941294d25f1db66f4a901dc1`
- Source count: 478,368 Gaussians
- Hierarchy: production-default CPU builder ABI 13 and reducer settings
- Full rendered graph: quality `.00` through `1.00` in `.01` increments, with
  `.99` present explicitly
- Image oracle: matched 192px selection height and 192x192 deterministic raster
- Metric: linear-RGB foreground-union PSNR against quality `1` at the same camera
- Deployment continuity graph: 1080px selection height in `.005` quality steps;
  these counts are evaluated separately and are never paired with 192px PSNR

The near, mid, and far cameras retain the asset's authored viewing direction
and up vector. Their distances are derived from projected root coverage rather
than scene-unit thresholds:

| Scale | Root coverage of viewport height | Center distance / root radius |
|---|---:|---:|
| near | 80% | 4.018 R |
| mid | 40% | 7.036 R |
| far | 20% | 13.071 R |

## Measured quality graph

Each active entry is `selected Gaussians (percent of source)`. PSNR uses the
matched 192px cut and raster described above. Infinite PSNR means the rendered
RGB foreground-union metric has zero mean-squared error against the quality-one
oracle. Quality `1` separately requires the exact full image and source count.

| q | near active | near PSNR dB | mid active | mid PSNR dB | far active | far PSNR dB |
|---:|---:|---:|---:|---:|---:|---:|
| .00 | 1 (0.00%) | 15.93 | 1 (0.00%) | 14.72 | 1 (0.00%) | 13.34 |
| .10 | 361 (0.08%) | 16.23 | 342 (0.07%) | 15.15 | 2 (0.00%) | 13.61 |
| .25 | 3,902 (0.82%) | 17.24 | 1,674 (0.35%) | 16.59 | 933 (0.20%) | 16.57 |
| .50 | 44,777 (9.36%) | 22.08 | 36,521 (7.63%) | 21.22 | 33,205 (6.94%) | 22.97 |
| .60 | 158,196 (33.07%) | 28.16 | 150,264 (31.41%) | 26.48 | 143,285 (29.95%) | 27.20 |
| .70 | 323,087 (67.54%) | 34.38 | 319,029 (66.69%) | 32.88 | 317,511 (66.37%) | 30.13 |
| .73 | 365,450 (76.40%) | 37.86 | 359,493 (75.15%) | 36.88 | 358,274 (74.90%) | 35.97 |
| .74 | 377,824 (78.98%) | 40.22 | 371,629 (77.69%) | 41.19 | 370,649 (77.48%) | 39.11 |
| .75 | 389,714 (81.47%) | 44.55 | 381,428 (79.74%) | 42.17 | 380,432 (79.53%) | 39.53 |
| .80 | 420,856 (87.98%) | 47.01 | 412,729 (86.28%) | 44.46 | 412,303 (86.19%) | 44.89 |
| .90 | 471,115 (98.48%) | 54.51 | 431,981 (90.30%) | 46.47 | 418,387 (87.46%) | 45.97 |
| .95 | 478,368 (100.00%) | inf | 478,352 (100.00%) | inf | 477,971 (99.92%) | 98.88 |
| .99 | 478,368 (100.00%) | inf | 478,368 (100.00%) | inf | 478,368 (100.00%) | inf |
| 1.00 | 478,368 (100.00%) | inf | 478,368 (100.00%) | inf | 478,368 (100.00%) | inf |

The useful knee for this scene is broad enough to inspect rather than being a
late slider cliff. The first common sampled anchor with at least 10% savings and
at least 33 dB at every distance is `.73`; it actually retains about 75% of the
source in the 192px oracle. The same `.73` cut is also the first common anchor
with at least 5% savings and at least 35 dB. At that anchor the audit found
zero spill outside the dilated reference, no extreme projected splats, and a
maximum projected aspect ratio no greater than 1.467. Quality `.95` is exact or
near-exact at all three distances, `.99` is exact, and the authored camera is
exact at both `.95` and `.99`. Six additional deterministic orbit views also
reported zero new visible elongated splats and zero opacity-visible spill.

The graph is intentionally not described as linear in PSNR. Hierarchy cuts are
discrete, and foreground-mask changes can cause small local metric reversals.
The regression instead constrains those reversals while requiring active counts
to be monotonic and the high-quality region to meet explicit image gates.

## 1080p workload continuity

The deployment graph contains 201 samples per distance. `q20`, `q50`, and
`q80` are the first quality values reaching 20%, 50%, and 80% of the source
count. The widest step is the largest source-count activation between adjacent
`.005` samples.

| Scale | Distinct cuts | q20 | q50 | q80 | Widest `.005` step | Mean active |
|---|---:|---:|---:|---:|---:|---:|
| near | 140 | .520 | .565 | .600 | 24,022 (5.02%, `.560 -> .565`) | 213,352.0 (44.60%) |
| mid | 149 | .555 | .615 | .660 | 19,640 (4.11%, `.610 -> .615`) | 189,929.7 (39.70%) |
| far | 151 | .570 | .650 | .715 | 13,401 (2.80%, `.655 -> .660`) | 171,589.5 (35.87%) |

The workload crossings move later as distance increases, and mean active count
decreases from near to far. This demonstrates projection-dependent LoD behavior
without a scene-tuned world-distance curve. It does not imply that PSNR should
be computed by combining these 1080p counts with the separate 192px images.

## Regression gates

The qualification fails unless all of the following remain true:

- Active counts are monotonic with quality at both 192px and 1080p, and for a
  common quality are ordered `near >= mid >= far`.
- Every interior `.005` deployment step activates at most 5% of the source plus
  one 128-record two-leaf domain for discrete node refinement, and the
  `q20 -> q80` span is at least `.075` at every distance.
- Average 1080p near-to-mid and mid-to-far separation is at least 2% of the
  source, with strict `near > mid > far` selection on at least 20% of interior
  samples.
- Foreground PSNR stays within 1 dB of its running best below `.5`, within
  1.25 dB from `.5` to `.7`, and within 0.5 dB from `.7` upward. From `.7`,
  SSIM, foreground IoU, alpha MAE, and covariance spill also stay inside bounded
  running-best envelopes.
- A common interior cut exists at every distance for both utility goals:
  10% savings at at least 33 dB, and 5% savings at at least 35 dB.
- At `.95`, foreground PSNR is at least 38 dB, SSIM and IoU are at least `.99`,
  alpha MAE and spill are at most `.005`; at `.99`, the corresponding gates are
  40 dB, `.995`, `.995`, `.003`, and `.003`.
- Utility anchors and `.95`/`.99` satisfy projected covariance size and
  morphology bounds, preventing elongated coarse splats from passing on PSNR
  alone. Quality `1` restores the exact source count and image.

## Selector contract qualified by this report

For an interior quality `q`, projected coverage `p`, node threshold `t`,
projected error `e`, and nominal error target `L(q) = 16 * 64^-q`:

```text
f = smoothstep(.90, .99, q)
S = q * (p + (1 - p) * f) / t
E = e / L(q)
a = min(q / .99, 1)^3
P_error = max(min(S, E), a * E)
```

For a usable builder-authored high-fidelity certificate `c`:

```text
n = min(q / .95, 1)
D_certificate = q * n * (p + (1 - p) * n^3)
P_certificate = D_certificate / c
P_final = max(P_error, P_certificate)
```

Thus the certificate base is quadratic below `.95`, while cubic authority
smoothly removes its projected-coverage relaxation. A zero, tiny (`<=1/65535`),
non-finite, or out-of-range certificate is ignored below `.95` for legacy
compatibility and fails closed for non-original representations at `.95` and
above. Quality `0` and quality `1` remain categorical coarsest and exact-original
endpoints.

## Reproducing and publishing the report

The test consumes a local fixture and does not download it. After independently
provisioning the pinned artifact:

```bash
sha256sum /absolute/path/to/trellis.glb
BGS_TRELLIS_GLB=/absolute/path/to/trellis.glb \
  BGS_TRELLIS_AUDIT_PROFILE=full \
  BGS_LOD_REPORT_PATH=/tmp/trellis-lod-quality.md \
  cargo +1.95.0 test --locked --test lod_real_scene_quality \
  --features "lod_build testing headless" \
  canonical_trellis_high_quality_color_and_covariance_audit -- \
  --ignored --nocapture --test-threads=1
```

The `pr` profile renders every `.05`; the `full` profile renders every `.01`.
Both run the 201-sample 1080p selection graph and always include `.99` and the
quality-one reference. The dedicated
[`lod-quality.yml`](../.github/workflows/lod-quality.yml) workflow validates or
fetches the exact fixture through
[`fetch_trellis_fixture.sh`](../tools/fetch_trellis_fixture.sh), uses `pr` on
pull requests and main pushes, uses `full` on the weekly schedule by default,
and publishes the generated Markdown plus the captured log as CI artifacts.
The report is deliberately emitted before assertions so a failure still has a
complete diagnostic table. Treat the test and workflow conclusion as the pass
or fail result; an uploaded report is not by itself evidence that its gates
passed.
