# Near-camera rendering: 2026-09-09

`release_qualified` remains **false**. Two current-camera fixes address specific
near-plane discontinuities. They do not repair authored floaters or establish
full-scene representative quality.

The shared 3D projection now scales peak opacity by a C1 near-clipping weight.
For center clearance `d` and covariance variance along the near-plane normal,
`r = 3 * sqrt(variance)` and `w = smoothstep(0, r, d)`. Opacity reaches zero at
the center plane and remains unchanged when the directional three-sigma support
is fully inside. Covariance, projected geometry and depth-order keys are
unchanged. This stateless camera clipping policy uses neither a screen-radius
cap nor a timed fade; it is not exact clipped-volume integration or source cleanup.

The shader evaluates clearance in view space and covariance in world space,
including the cloud transform and dynamic Gaussian scale. Morphs apply the same
weight from their current interpolated geometry to both endpoint opacity terms,
once. GPS derives its sampling rate from the adjusted peak. The CPU production
oracle and fitter use the corresponding scalar policy. Conditional 4D projection
is unchanged; the [projection contract](lod_projection_parity.md) gives details.

Ordered spatial transitions also approach the child endpoint before their
conservative near envelope becomes unsafe. With budget weight `t` and a smooth
envelope-clearance factor `s`, the adjusted weight is
`t / (t + (1 - t) * s)`. This removes the previous abrupt fractional-to-child
switch at that boundary while preserving the budget endpoints. Complete resident
cohorts and existing capacity limits remain required. Missing data, invalid
safety evidence, and the simultaneous zero-weight/unsafe corner remain
categorical limits. Spatial interpolation still requires `abs(global_scale) <= 1`.

The [final SH0](../target/lod-roadmap/2026-09-09/near-camera/sh0-fixtures.json)
and [SH3 fixture results](../target/lod-roadmap/2026-09-09/near-camera/sh3-fixtures.json)
record successful focused CPU and GPU checks, including:

- [Near clipping](../tests/near_clip_render.rs): one 48×48 application, ordinary
  quads, ordered quads and fixed-seed eight-sample GPS; six approach/cross/return
  depths under both perspective and orthographic cameras. It checks fading,
  bounded near-plane image peaks and exact returns, without hierarchy inputs.
- [Ordered spatial motion](../tests/global_order.rs): a fixed resident cut
  crosses the former near-envelope switch and approaches the exact child image.
- The production CPU/GPU oracle, support-overlap images, scalar clipping
  endpoints, current-camera directional covariance and fitting-gradient checks.

The first GPS analytical fixture failed because its 100× isotropic support now
entered the near-clipping band. Its corrected fixture changes only depth scale;
XY covariance and the analytical intensity target remain unchanged. The
[initial result](../target/lod-roadmap/2026-09-09/near-camera/sh0-initial-fixtures.json)
and [passing rerun](../target/lod-roadmap/2026-09-09/near-camera/sh0-gaussian_point_splatting.log)
are retained. The shared core runner includes the new near-clipping fixture for
both SH0 and SH3; both profiles pass these focused checks.

The same near-clipping GPU fixture also passes with SH3 and precomputed 3D
covariance, covering the alternate covariance storage path. Its
[command and result](../target/lod-roadmap/2026-09-09/near-camera/precompute-check.json)
are retained.

Warnings-denied Clippy and the actual browser-capture Wasm build pass for both
SH profiles. Formatting, diff checks and all 39 Python protocol/metric tests
also pass. The [validation record](../target/lod-roadmap/2026-09-09/near-camera/validation.json)
pins the tested binaries, source snapshot, commands and logs. Wasm compilation
does not qualify browser GPU execution.

The [bounded Poland comparison](../target/lod-roadmap/2026-09-09/near-camera/poland-comparison.json)
uses the first 61 calibrated camera poses, 1920×1080 ordered rendering, and
8M active/24M resident records. Configurations differ only in output directory;
the report pins both renderer identities, source/package identities and images.
Within **each** run, all 29 matched forward/reverse images are byte-identical,
as are held observations and return endpoints. Both binaries already pass that
short-route repeatability check; it demonstrates neither improved quality nor a
speedup or longer-route recovery.

Before/after images differ substantially. The
[image-change report](../target/lod-roadmap/2026-09-09/near-camera/poland-image-changes.json)
records a brighter result, including frame 960 changing by 83.72 RGB8 MAE from
the earlier dark image. The [bounded diagnosis](../target/lod-roadmap/2026-09-09/near-camera/frame-960-diagnosis.json)
verifies identical camera matrices and all 7,813 selected cut rows at that frame.
It authenticates 147 nearby original-leaf pages, reading 9.64 MB. One original
Gaussian has black RGB, opacity .9843, center clearance 1.17 and three-sigma
depth extent 32.73 in scene units; its new near weight is .00376. Attenuating such
camera-straddling source Gaussians strongly explains the removed dark veil.
This is not an exact per-pixel decomposition: Mip and compositing were not
replayed separately, and the spatial-weight change was not isolated.

Brightness is not a source-quality reference. Prior near-plane image evidence
needs recapture under the changed rendering policy. Browser, broader adapter
and real-source near-camera quality acceptance remain open.
