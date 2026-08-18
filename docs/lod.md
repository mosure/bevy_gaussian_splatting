# Gaussian level of detail

The LoD system selects a complete hierarchy cut per camera, streams or
materializes the required pages into a bounded atlas, compacts visible records
on the GPU, and renders them through the ordinary Gaussian raster path.

The promoted product surface is deliberately small:

- `GaussianLodSettings::quality` controls detail from `0` (coarsest) to `1`
  (exact original).
- `GaussianLodSettings::selection_mode` is `Dynamic` or `Frozen`.
- `LodDebugPreset` selects one of the documented debug views.
- `GaussianLodStatus` reports lifecycle, selected records, quality pressure,
  residency, and typed failure state.

Budgets, transport limits, and selector hysteresis remain programmatic safety
controls. They are not additional quality sliders.

## Runtime paths

### Resident flat cloud

Attaching `GaussianLodSettings` to a `PlanarGaussian3dHandle` enables the
automatic bridge. For an interior quality it builds a bounded progressive
MomentMerge hierarchy and publishes complete cuts through the same atlas and
render handshake used by packages. When every active view would retain at
least 95% of the source, the bridge draws the exact retained source instead;
the marginal reduction cannot repay compaction and a second radix pass. The
status then reports the source count and exact rendered quality. Quality `1`
always stays on that flat path and allocates no LoD compaction state. Debug
presets keep the atlas path so annotations still match hierarchy records.

This path is convenient for ordinary PLY, gcloud, and glTF/GLB assets, but the
hierarchy build is synchronous. Large production scenes should use a prebuilt
package.

### Prebuilt package

An entity with `GaussianLodHandle`, `GaussianLodPackageSource`, and
`GaussianLodSettings` streams independently addressable pages from standalone
`.gspage` objects or range-packed `.bgslodpack` shards.
Native directories and immutable HTTP(S) roots share the same bounded runtime.
Persistent caching is opt-in through `GaussianStreamingSettings` and requires
an explicit namespace; native caching also requires an explicit cache root.

The viewer exposes both paths:

```bash
# Resident scene with a transient hierarchy
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-scene=https://mitchell.mosure.me/trellis.glb \
  --lod-quality=0.75

# Prebuilt out-of-core package; page URIs resolve beside the manifest
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-lod=file:///data/city/scene.gsplatlod \
  --lod-quality=0.75
```

For a deterministic local smoke scene:

```bash
cargo run --release --bin bevy_gaussian_splatting -- \
  --gaussian-count=65536 --gaussian-seed=42 --lod-quality=0.65
```

The editor and the compact Gaussian LoD panel are enabled by default. Press
`F` to freeze the current cut, then move the camera closer to inspect it.

## Quality contract

The slider is screen-space and scene-scale independent. Perspective projection
already supplies distance response, so there is no second world-distance curve.
For interior quality `q`, projected coverage `p`, node threshold `t`, projected
error `e`, and the fixed nominal target `L(q) = 16 * 64^-q`:

```text
f = smoothstep(.90, .99, q)
S = q * (p + (1 - p) * f) / t
E = e / L(q)
a = min(q / .99, 1)^3
P_error = max(min(S, E), a * E)
```

A builder-authored high-fidelity certificate `c` adds a morphology guard:

```text
n = min(q / .95, 1)
D = q * n * (p + (1 - p) * n^3)
P_certificate = D / c
P = max(P_error, P_certificate)
```

Pressure `P <= 1` meets the requested target. Quality `.95` is intentionally a
strong safety gate and may select exact leaves when no safe approximation
exists. Quality `1` is categorical exact-original, not a zero-error
approximation. The mapping is monotonic in requested detail, but hierarchy cuts
are discrete and PSNR is not expected to be linear.

The pinned Trellis graph, near/mid/far PSNR table, continuity gates, and exact
selector equations live in [the canonical quality report](lod_quality_report.md).

## Hierarchy and file contracts

The CPU builder creates deterministic Morton-ordered logical nodes and packs
same-depth, same-kind node slices into physical pages. Leaf payloads contain
original records. Interior payloads use risk-aware contiguous MomentMerge
representatives with conservative error, support, and fidelity metadata.

The manifest records:

- source, node, page, and stored-record counts;
- topology and per-node page slices;
- bounds, quality interval, error, and high-fidelity certificate;
- page encoding, length, checksum, and optional storage location;
- builder ABI, reducer version, and required format features.

Validation rejects cycles, invalid ranges, heterogeneous shared pages,
unsupported features/ABIs, non-finite metadata, count overflow, and hierarchy
amplification/storage violations before allocation.

Build a package with:

```bash
cargo run --release --no-default-features --features "lod_build sh3" \
  --bin build_lod -- \
  --input scene.ply --output out/city
```

This writes the manifest to `out/city/scene.gsplatlod` and its page payloads
beside it.

The builder is replayable and bounded. `--gpu-hierarchy` accelerates the
promoted GPU sorting/reduction primitives; the output remains validated against
the same manifest/page contracts.

## Selection, residency, and commit

Selection traverses visible roots against the camera projection and returns a
complete node cut. Missing descendants retain the nearest resident ancestor;
the renderer never publishes holes as a successful cut. Active, traversal,
resident-page, resident-byte, request, preprocessing, and upload budgets fail
closed to the previous complete cut or a complete ancestor fallback.

Selector hysteresis suppresses small cut oscillations, and an accepted
replacement is committed atomically as another complete cut. Device loss
invalidates GPU generations and rebuilds atlas, compaction, radix, bind groups,
and candidates from retained CPU state.

On Wasm, page verification and fixed-record decoding advance cooperatively with
a bounded Gaussian budget per application frame. Native preprocessing uses a
bounded worker backend. Both paths preserve checksum, codec, validation, and
support-bound error order.

## Debugging

Use named presets rather than raw scalar tuning:

```bash
--lod-debug=level
--lod-debug=page
--lod-debug=residency
--lod-debug=boundaries
--lod-debug=selection-pressure
```

Debug metadata uses the ordinary Gaussian raster path. The panel reports when
metadata is ready and when a flat exact-original source has no hierarchy to
annotate. Prebuilt packages retain their exact leaf hierarchy at quality one.
See [LoD debugging](lod_debug.md) for interpretation.

## Validation

The non-GPU release checks are collected in:

```bash
tools/qualify_lod_release.sh
```

GPU validation is explicit and never runs implicitly:

```bash
BGS_RUN_GPU_QUALIFICATION=1 tools/run_lod_gpu_qualification.sh
```

The canonical Trellis quality workflow verifies the exact artifact length and
SHA-256, runs matched-resolution foreground PSNR/SSIM/IoU/alpha and morphology
checks across near/mid/far cameras, then uploads the report even on failure.
Locally, provide the already-verified fixture:

```bash
BGS_TRELLIS_GLB=/absolute/path/to/trellis.glb \
BGS_TRELLIS_AUDIT_PROFILE=full \
BGS_LOD_REPORT_PATH=/tmp/trellis-lod-quality.md \
cargo +1.95.0 test --locked --test lod_real_scene_quality \
  --features "lod_build testing headless" \
  canonical_trellis_high_quality_color_and_covariance_audit -- \
  --ignored --nocapture --test-threads=1
```

Package data is untrusted input. Size/count/allocation bounds, version checks,
checksums, path confinement, request caps, and cache validation are part of the
runtime contract; see [SECURITY.md](../SECURITY.md).
