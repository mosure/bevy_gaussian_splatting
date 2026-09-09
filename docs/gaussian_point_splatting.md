# Gaussian Point Splatting

Gaussian Point Splatting (GPS) is an experimental, opt-in camera backend for
rendering the currently admitted Gaussian set without a Gaussian depth sort.
The implementation has focused CPU, native GPU traversal/image, and software
Vulkan package-integration coverage. This is limited qualification, with no accepted performance result or claim
of parity with the authors' CUDA renderer.

The method follows Rijsdijk et al., *Gaussian Point Splatting* (SIGGRAPH 2026):
sample opaque points from an opacity-corrected distribution, resolve visibility,
then average samples. See the [project site](https://jorisar.nl/gaussian_point_splatting/),
[paper](https://jorisar.nl/gaussian_point_splatting/gaussian_point_splatting.pdf),
and [reference implementation pinned to `54d1308ede68662c9129db5b4af61c26f69499a6`](https://github.com/JorisAR/gaussian-point-splatting/tree/54d1308ede68662c9129db5b4af61c26f69499a6).
Reported performance of that implementation does not transfer to this WGSL path.

## Supported integration

Attach `GaussianPointSplattingSettings` to a Gaussian camera. The initial profile
uses planar 3D Gaussians, storage buffers, `RasterizeMode::Color`, non-additive
clouds, disabled LoD debug/bounding-box visualization, and `Msaa::Off`. Eligible
clouds use `CloudSettings.sort_mode` of `SortMode::Radix` (the default) or
`SortMode::None`; `Std` and `Rayon` retain the per-cloud quad renderer. LoD clouds
must present discrete complete cuts; continuous parent/child morphing and external
active-set inputs are outside this initial GPS profile. The per-cloud quad backend supports those rendering profiles.

```rust
use bevy::prelude::*;
use bevy_gaussian_splatting::{
    GaussianCamera, GaussianLodSettings, LodPresentationMode,
    render::point::GaussianPointSplattingSettings,
};

fn spawn_point_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        GaussianCamera::default(),
        Msaa::Off,
        GaussianPointSplattingSettings {
            samples_per_pixel: 4,
            temporal_sampling: true,
            target_gpu_ms: None,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn configure_lod_presentation(settings: &mut GaussianLodSettings) {
    settings.presentation_mode = LodPresentationMode::Discrete;
}
```

Configure the cloud's existing LoD settings so its selected quality and residency
budgets are preserved. GPS is available in the crate's `lod_render` profile; its
plugin is installed by `GaussianSplattingPlugin`. Set `temporal_sampling: false`
to freeze the frame RNG for repeatable images. Changing samples per pixel changes
stochastic noise, not the LoD quality target. Temporal sampling itself provides
neither temporal accumulation nor denoising.
GPU hierarchy sampling uses immutable node and representative identities, so
relocating unchanged pages within an atlas does not change their frozen samples.

The standalone viewer exposes the same opt-in policy without application code:

```sh
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-cloud assets/scene.ply --point-splatting \
  --point-samples-per-pixel 4 --point-max-points 16777216 \
  --lod-quality 0.5 --lod-max-active-gaussians 500000
```

Use an existing PLY path in place of `assets/scene.ply`, or replace `--input-cloud`
with `--input-lod path/to/scene.gsplatlod` for a built package. The default crate
features include this viewer profile. `--point-splatting` attaches the component
only to the viewer's main camera, sets MSAA off, and uses discrete LoD cuts while
preserving the requested quality and record budget. It rejects non-3D or non-color
modes, LoD debug presets other than `off`, and `--input-cloud-target`. Imported
clouds with unsupported material/sort settings retain the per-cloud quad path.
`--point-max-projected-gaussians` and `--point-max-gpu-bytes` expose the
1,048,576-record and 256 MiB defaults; a requested cut must fit both limits.

Add `--point-target-gpu-ms 6.5` to enable automatic sampling. Bevy's default
`Functionality` policy negotiates timestamp features only when the adapter
supports them; the viewer does not force a required feature that would fail on
unsupported devices. A stricter feature policy can leave timestamps unavailable,
which is reported at startup. GPS leaves the viewer's device-feature policy
unchanged. It allocates timestamp queries only when automatic sampling is enabled
and the device feature is available; fixed sampling allocates no GPS timestamp
queries. Without `--point-splatting`, the viewer also retains its existing camera
policy.

## Rendering contract

The backend gathers participating clouds into one bounded projected-record
stream per view. It evaluates projection and radiance once per input Gaussian,
then counts and distributes point work. GPS and globally ordered quads share a
conservative world-support sphere/frustum gate, including the mip-filter margin,
before covariance projection. This matches the per-cloud quad support policy and
excludes off-axis perspective-linearization artifacts whose entire world support
misses the view. Large ellipses whose support intersects the view remain admitted.
The covariance projection itself is shared and does not clamp off-axis coordinates.
The numerical reference in
[math.rs](../src/render/point/math.rs) uses the corrected full-process expectation
`SPP * 2π * sqrt(det(screen_covariance)) * Li₂(opacity)`. For each Gaussian it
chooses the cheaper of the full radial process and a homogeneous Poisson envelope
over the viewport-clipped three-sigma ellipse bounding rectangle. The rectangle
rate per sample is `area * rectangle_peak`. The peak conservatively bounds the
intensity inside that rectangle using its distance from the Gaussian mean and
the marginal screen variances. Uniform proposals are thinned with probability
`-log(1-opacity*exp(-r²/2)) / rectangle_peak` and rejected outside
`r ≤ 3`. Both choices therefore preserve the same corrected spatial intensity.
This bounds proposal work for huge or off-axis ellipses by their visible rectangle,
without adding to the 64-byte projected record.

Rates above 1,048,576 are sampled as sums of independently keyed Poisson
components, each at or below that numerical bound, with at most 1,024 components.
Once a partial count already exceeds the configured point ceiling, the entire
image is rejected through the existing overflow path. Rejected proposals are
never replaced and accepted counts are never clipped. Finite precision, bounded
samplers, filtering and support truncation still require their own numerical and
image qualification.
All renderers use the same finite radial density: the Gaussian is unchanged for
Mahalanobis squared radius `q <= 8`, then multiplied by
`(1-u)^2 (1+2u)`, `u=q-8`, until it reaches zero at `q=9`. Both the density and
its first derivative are continuous. Adaptive flat-cloud cutoffs scale this band
to the existing radius; no support is extended and opacity is not renormalized.
GPS independently thins its radial or rectangle proposal to this target. CPU
references and fitting use the same density and analytic derivative.

At three sigma this removes 0.3281% of the previous disk kernel's integrated
mass, or 0.9010% of the previous OBB square's mass. These are isolated kernel
figures, not bounds on composited image error. Measurements from the earlier
hard support renderer require a new matched reference. Finite pixel integration,
filtering and depth ordering can still make converged GPS differ from quads.

Portable WGSL exposes 32-bit atomic integer types. This adaptation separates
nearest-depth reduction from winner selection into two passes, with deterministic
depth ties, instead of assuming the reference renderer's 64-bit atomic packing.
The passes replay the same keyed samples. Resolve averages premultiplied samples
and tests supported opaque mesh depth before composition; the initial mesh-depth
profile has MSAA disabled. [WGSL atomic types](https://www.w3.org/TR/WGSL/#atomic-types).

All participating clouds on the same view share visibility resolution, so their
relative draw order no longer requires Gaussian radix sorting. This statement
does not cover a mixture of GPS clouds and separately composited transparent
quad clouds. Each sampled point still uses its Gaussian's center depth; removing
sorting does not make extended overlapping Gaussians exact ray geometry.
GPS compares projected center depth in reverse-Z. Quad sort backends use forward
camera depth (`-view-space Z`), so ordinary perspective/orthographic projections
agree on front-to-back center order. Sampling, finite precision and equal-depth
tie handling still have separate contracts.

The optional [global quad order](global_quad_order.md) backend provides one shared
forward camera-depth order for supported flat, discrete CPU-selected and GPU-traversed
quad clouds. Both renderers consume the same hierarchy input and fenced residency
receipts. Mixing the two rendering backends does not create a shared transparency
order.

Quad comparisons recorded before this depth-metric change used squared radial
camera distance. Those images remain historical evidence and require fresh
matched captures before qualifying the current quad/GPS comparison.

An eligible GPS camera suppresses Gaussian radix dispatch and LoD radix scratch
allocation. Flat assets release radix workspaces when every active Gaussian
camera can use GPS. The existing flat identity-index allocation remains a
compatibility cost. Unsupported clouds and cameras can still require radix work.

`GpuLodTraversalSettings` selects an additional input path: immutable hierarchy
topology plus a versioned resident-page snapshot, bounded GPU wavefront traversal,
and physical record expansion directly into the shared GPS stream. Refinement
admits every child together; missing pages or exhausted work budgets retain the
resident parent. Deduplicated page requests return asynchronously to the package
loader. Snapshot ownership and atlas generation checks protect pages while GPU
work remains in flight. This path does not construct a CPU list of selected
Gaussians.

GPU hierarchy selection currently requires dynamic, discrete presentation;
frozen camera selection is rejected explicitly. All GPU-hierarchy clouds in a
view share one selected-record ceiling, with every root cut reserved before
remaining capacity is distributed deterministically. Eligible flat and
CPU-compacted inputs reserve their prepared capacities first; hierarchy outputs
share the remaining projected capacity. The hierarchy selected-record limit
applies only to hierarchy output. Insufficient capacity suppresses the entire
shared image and its residency acknowledgements.

## Separate capacity and sampling budgets

| Control | Default | What it bounds |
| --- | --- | --- |
| `samples_per_pixel` | 4, maximum 8 | Complete image sampling layers and their visibility storage |
| `min_samples_per_pixel` | 1 | Sampling floor for automatic timing, overflow and allocation decisions; must not exceed the initial sample count |
| `max_projected_gaussians` | 1,048,576 | Projected input records; portable scan ceiling 16,777,216 |
| `max_points_per_frame` | 16,777,216 | Generated point work, before support/viewport rejection; hard ceiling `1 << 30` for safe prefix/dispatch arithmetic |
| `max_gpu_bytes` | 256 MiB | Accounted allocations admitted for this camera |
| `target_gpu_ms` | `None` | Enables automatic whole-layer control from actual GPU timing or confirmed point overflow |

Record counts bound projection, storage and streaming. Projected opacity mass
determines stochastic point work: a few large opaque splats can cost more than
many small faint ones. Both must be measured. GPS does not improve poor LoD
representatives or remove the bandwidth needed to fetch/decode their pages.

Allocation checks include `width * height * SPP`, depth/winner storage, projected
records, count/scan scratch, output and feedback resources. The renderer admits
these through its bounded memory ledger and checks device limits; that ledger is
an owned-allocation account, not measured driver memory or process RSS. Diagnostic
feedback is asynchronous and bounded, with the originating submission identity.
`projected_gaussians` counts valid projected screen records. `requested_points`
counts generated work before support/viewport rejection; `dispatched_points`
uses the same pre-rejection scope and is zero on overflow. Neither point counter
is a count of accepted visible samples.

Preparation failures are diagnosed. An explicitly selected, eligible GPS camera
keeps its backend claim during allocation or record-admission failure; a completed
image can be retained when its target and extent remain compatible. It does not
silently launch an expensive quad draw. Unsupported cloud material/sort profiles
use the compatibility renderer; invalid GPS camera settings report unavailable
output. Diagnostics distinguish ready, retained and
unavailable output, and report the actually allocated sampling-layer capacity.
Automatic sampling allocates admitted layers rather than eight layers up front.
Asynchronous timing and overflow feedback changes the sampling count only when
it was rendered at the current count. Delayed measurements from older counts
still acknowledge complete images but cannot cascade sampling changes.

An admitted GPS view projects and draws using the current camera every frame,
even while all three asynchronous telemetry slots are busy. Only feedback and
LoD draw acknowledgements wait for a free slot; a busy telemetry queue does not
replay the previous camera image.

After GPS admission, point-work overflow or a sampling failure retains and
composites the last complete GPS image and does not acknowledge a new LoD cut.
With no previous complete image, there is no successful GPS presentation to
claim. Overflow must never silently discard a tail of points or Gaussians.

The optional controller changes at most one whole layer per fresh observation:
above 110% of target it reduces layers; eight consecutive observations below 80%
permit one increase. Stale or invalid timestamps are ignored. Without supported
timestamp feedback there are no timing-driven adjustments. Automatic mode can
still remove one whole layer after a confirmed point-budget overflow, sharing
the same stale-submission guard and resetting the headroom streak. Fixed mode
(`target_gpu_ms: None`) retains its configured layers and reports overflow.
At the configured sampling floor, an overflow remains a failure. Memory
admission cannot lower that floor; inability to allocate it is reported. The
floor is an application noise policy and requires independent image qualification.
Feedback from a previous sampling policy cannot change the current controller.
The backend uses timestamps only
when the optional device feature was requested and is supported. Every increase
still needs memory/work admission. This controller never thins per-Gaussian samples,
changes the LoD quality slider, or treats missing timestamps as zero GPU time.

### GPU package selection and view budget

For a discrete package, add `GaussianGpuLodPackage` to the cloud and
`GpuLodTraversalSettings` to the GPS camera. The viewer exposes this combination:

```sh
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-lod path/to/scene.gsplatlod --point-splatting --lod-gpu-traversal \
  --point-target-gpu-ms 16 --lod-max-active-gaussians 500000
```

The viewer caps selected records by both the LoD active-record budget and GPS
projection capacity across the view. `GaussianPointSplattingViewBudget` is an optional outer
camera policy. With GPU traversal, the viewer's time flag enables both this
policy and the inner sampling controller. Application code may configure the two
targets separately.

The outer interval uses actual GPU timestamps from before this view's traversal
and compaction through its final upscaling. It includes intervening view passes;
it excludes CPU work, shadows and other cameras. It is reported as `view_gpu_ms`,
not whole-application frame time. Three asynchronous readback slots carry the
same-submission point and traversal counters. Missing timestamps leave timing
control unavailable.

The controller first lets sampling reach its configured floor. Measured excess cost or
hard work overflow can reduce the selected-record cap by one eighth; 32 low-cost
observations permit recovery. A coarsening trial with a valid image baseline is
evaluated over eight fresh observations. If larger representatives increase time
or point mass by over 25%, it restores the prior cap and enters a cooldown. A
trial that overflows point work or loses root coverage rolls back immediately.
Cold overflow at the sampling floor has no successful baseline to restore; record-cap
reduction cannot guarantee recovery because coarser proxies can cost more.
Stale feedback cannot
change a newer cap. Authored quality settings remain intact, and diagnostics
report an unmet target when the bounded controller cannot satisfy it. Record
coarsening from a valid image is therefore a measured trial.
Visible root records also bound coarsening before GPU feedback arrives, so an
overly small cap cannot starve the very feedback needed for recovery. Device
recovery clears old successful-image acknowledgements and timing history.

## Recorded qualification and remaining limits

The subsequent 2026-09-08 integration pass adds opaque mesh occlusion behind,
ahead of and between Gaussian centers, equal-depth cross-cloud stability after
ECS reordering, package camera motion/revisit, and unload retirement of snapshots,
atlas assets and accounted leases. The extended GPS and GPU-package fixtures
pass on llvmpipe Vulkan in 3.13 and 1.91 seconds. The new globally ordered quad
fixture, extended to cold native CPU-package activation and movement, also passes
in 4.56 seconds. Nine focused CPU checks pass, including the
native-pixel fitter projection/gradient regression. These execution checks are
recorded with exact binaries in
[the new test record](../target/lod-roadmap/2026-09-08/completion/gpu-tests.json);
they do not qualify hardware performance or large-scene visual quality.

The 2026-09-08 completion pass passed 25 focused CPU checks covering sampling,
controllers, aggregate admission, allocation, snapshot validation, output proofs,
CLI policy and the restricted fitter. Native library/render-test and viewer
Clippy passed with warnings denied, and the WebGPU library compiled.

Three small rendering checks passed on **llvmpipe software Vulkan** in the resumed
environment: traversal oracle (.41 s), GPS visibility/retention (2.22 s), and
[package integration](../tests/gpu_lod_package.rs) (1.58 s). The package test loads
authenticated native pages, refines to an exact 32-record cut, adds a second
independently resident package, requires both snapshot acknowledgements from the
same successful GPS image within one shared 32-record ceiling, and exercises
timestamp-driven cap reduction with both root covers retained. These are
execution/correctness checks, not hardware performance measurements. Logs,
commands and binary hashes are in the
[test records](../target/point-splatting/2026-09-08/test-runs.json).
The [completion record](../target/point-splatting/2026-09-08/summary.json) also
records the stopped optional viewer-hotkey test build and a final formatting-only
module-order adjustment. The viewer passed native Clippy; its separate hotkey
test executable was not run in this pass.

Before the interruption, the actual traversal shader also passed its tiny
transformed/budget/fallback oracle on NVIDIA Vulkan in 1.25 s, and the updated
GPS image/allocation test passed in 2.10 s. The final multi-package/controller
integration still needs that hardware run. Device-reset bookkeeping is covered
by implementation and focused checks; forced physical device loss, large-scene
motion and browser execution are not qualified by these tests.

The earlier initial backend screen is retained below for provenance.

On 2026-09-07, fifteen focused CPU tests passed: ten math/settings, three LoD
output-proof and two viewer configuration tests. These cover the sampling
reference, bounded settings/allocation, the whole-layer controller, exact
candidate identity for asynchronous activation, and CLI/config compatibility.
The native [GPU test](../tests/gaussian_point_splatting.rs) passed in 2.01 seconds
on one Vulkan NVIDIA RTX PRO 6000 Blackwell, driver 610.43.02. It uses a 96×80
render target with a 64×64 viewport and checks opacity over two overlapping clouds,
frozen-sample repeatability, zero opacity, retention of the last complete image on
point overflow, and discrete-cut activation while the camera moves. Its automatic
phase received actual GPU timestamps, reduced eight layers to one, and confirmed
that point overflow also reduces whole layers, dispatches no partial process,
and preserves the complete image. The timing phase explicitly skips on devices
without timestamp support; it ran on this adapter.

The default native viewer and `wasm32-unknown-unknown` WebGPU library compile
checks passed. Targeted native library/GPU-test Clippy passed with warnings denied;
formatting and diff checks also passed. Local logs and source/binary hashes are
stored under `target/point-splatting/2026-09-07/`; these are a correctness screen,
not a benchmark result. A final formatting-only delta is recorded separately.

That test does not measure performance or establish exact per-cloud-renderer parity,
opaque-mesh occlusion correctness, GPU hierarchy traversal, browser execution or
multi-vendor support. Further qualification must cover mesh intersections, ties,
memory and
feedback bounds, and evidence that admitted GPS frames omit Gaussian radix
dispatch. Matched quality/frame-time comparisons must include projection,
sampling, both visibility passes, resolve and composition. Fixed seeds aid
debugging; multiple seeds and motion sequences are needed to assess stochastic
error.

The [LoD status](lod_implementation_status.md) retains the failed representative
quality and package-update gates. GPS is a separate raster alternative; it does
not close those gates or establish CUDA, browser or multi-vendor parity.

## Bounded real-scene comparison tool

`capture_lod --compare-points CONFIG.json` loads one SHA256-pinned PLY once and
uses one stationary camera for sequential quad, GPS one-layer and GPS four-layer
cases. Each case warms up for the configured number of encoded rendering frames,
then records sixteen submissions. The prepared local Icecream configuration is
[`point-compare-icecream.json`](../target/lod-roadmap/2026-09-07/point-compare-icecream.json):

```sh
target/debug/capture_lod --compare-points \
  target/lod-roadmap/2026-09-07/point-compare-icecream.json
```

This is an opt-in experiment. Its completed and failed screens are recorded below.
Configuration fields pin `source`, `source_sha256`, `output`, the source-record
ceiling and camera `from`/`target`/`up` (`camera_from`, `camera_target`, `camera_up`).
The default viewport is 1280×720, warmup is 32 frames and the total application
deadline is 120 seconds. Optional `point_settings` keep their bounded allocation
and point-work limits, with automatic layer control disabled for this comparison.
The hard limits are three readback slots, sixteen measured frames per case, at
most three PNGs per case, 720p, eight million source records and 300 seconds.
Existing output directories are rejected.

### Real-source screen, 2026-09-07

The pinned 84,348-record Icecream source at 1280×720 exceeded the default
16,777,216 generated-point limit at both one and four layers. All GPS frames
reported overflow and zero dispatched points. That
[failed report](../target/lod-roadmap/2026-09-07/point-compare-icecream/report.json)
is retained; its short GPU intervals are not successful rendering performance.

One bounded follow-up used the same source/camera at 480×270, a 67,108,864-point
ceiling, eight warmup frames and sixteen measured submissions per case. Every
measured frame completed on Vulkan RTX PRO 6000 Blackwell, driver 610.43.02.
The [report](../target/lod-roadmap/2026-09-07/point-compare-icecream-480/report.json)
pins its executable and the [image analysis](../target/lod-roadmap/2026-09-07/point-compare-icecream-480/analysis.json)
pins that report. Later integration edits are outside this artifact's identity.

| Backend | Median view GPU interval | Median generated points | RGB PSNR against quad, three images | Pairwise temporal RGB RMS |
| --- | ---: | ---: | ---: | ---: |
| Quad | 0.493 ms | not applicable | reference | 0 |
| GPS, one layer | 0.507 ms | 5.19M | 21.70–21.79 dB | .1040 |
| GPS, four layers | 1.586 ms | 20.74M | 26.78–26.83 dB | .0589 |

Four layers reduce noise and cost more in this screen. This is not an equal-quality
speedup or LoD acceptance result. Display-space PSNR includes stochastic error
and the documented support/depth differences; three images do not establish
convergence. The small scene does not qualify large-scene scaling, moving-camera
quality, paging or browser execution.

`report.json` preserves the executable, source, configuration, actual camera,
adapter and image identities. Quad counts come from a copied indirect buffer
and an actual draw-command attestation. GPS counts come from its newly encoded
submission's 32-byte feedback copied beside the target image; delayed diagnostic
counters are not image identities. Both encoder timestamps enclose Gaussian work
through final upscaling, before feedback/image copies. Missing timestamp support
is reported as null, never replaced with CPU timing. CPU PNG encoding and GPU
readback instrumentation still affect scheduling, so these are instrumented
measurements. The three temporal-noise images are a screen, not a convergence or
LoD-quality certificate. Support and depth-order differences described above
remain part of any image comparison.

The current closeout also passes the small native GPS/package fixtures on
NVIDIA Vulkan and the [SH3 browser lifecycle](lod_browser_qualification.md#current-gps-hardware-execution-2026-09-08)
on an actual non-fallback NVIDIA WebGPU device. Browser proof includes
same-submission point/traversal headers and zero-ledger unload/reload at 160×90.
It does not establish equal-quality gains or large-scene controller stability.
