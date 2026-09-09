# Gaussian projection and density parity

The OBB Gaussian3d/Gaussian4d fragment path now evaluates its Gaussian density from the fragment's framebuffer position and a projected mean/inverse covariance. Quad positions and raster coverage retain their existing calculation. This removes density's dependence on interpolated unit-quad UVs, which produced measurable errors even inside a splat's support.

For a projected mean `m` in absolute physical framebuffer pixels and inverse filtered covariance `Q` in the shader's doubled-pixel coordinates, the fragment uses:

```text
d = 2 * (fragment_position.xy - m)
power = -0.5 * (Q.xx*d.x*d.x + 2*Q.xy*d.x*d.y + Q.yy*d.y*d.y)
q = -2 * power
u = clamp(q - 8, 0, 1)                 # fixed three-sigma support
weight = exp(power) * (1-u)^2 * (1+2*u) # zero for q >= 9
alpha = min(weight * peak_opacity, 0.999)
```

The vertex computes `m = viewport.xy + (ndc.xy * (0.5, -0.5) + 0.5) * viewport.zw`. The viewport origin is included. The covariance already uses framebuffer-down Y through the projection Jacobian; reflecting the fragment displacement again would reverse the off-diagonal covariance. The OBB mean and conic use flat interpolation on the supported native and WebGPU paths. Location 3 is named `conic_position`: it carries this absolute mean for OBB and preserves the interpolated doubled-coordinate offset for AABB.

The current shared finite kernel leaves density unchanged through `q=8` and tapers it continuously to zero at `q=9`. The OBB remains a conservative raster envelope: covered corners beyond the ellipse emit zero density. This later support change applies consistently to ordinary, ordered, morph and GPS rendering and their CPU density oracles; it does not extend authored support. The analytic parity witnesses separately check covariance, OBB geometry, the elliptical boundary and the cubic tail.

The original conic correction retained color/SH evaluation, opacity clamping, Mip filtering, morph coefficients, quad vertices, depth testing and blend modes. The CPU production oracle also received two independently identified corrections: reflect the shader's OBB axes into framebuffer coordinates, and clamp vertex peak opacity to one while retaining the fragment's `.999` alpha cap.

## Current 3D near clipping

The shared 3D projection retains center-plane clipping and smoothly reduces peak opacity while the Gaussian's depth support crosses the camera's near plane. For current world-space mean `m`, covariance `Sigma` and inward near plane `(n, d)`, it uses:

```text
clearance = dot(n, m) + d
radius = 3 * sqrt(dot(n, Sigma * n))
x = clamp(clearance / radius, 0, 1)
near_weight = x*x*(3 - 2*x)
```

The weight is zero at or behind the plane and one when the directional three-sigma support is wholly inside. Zero depth extent has full weight for positive clearance. Plane normalization cancels. This C1 camera clipping filter changes peak opacity; covariance, projected geometry and depth-order keys remain unchanged. It uses no frame history or screen-radius cap.

Morphs compute the weight from the current interpolated mean and covariance and apply that same weight to both endpoint opacity terms. GPS derives its process rate from the resulting peak opacity. The CPU production oracle and fitter use the same scalar policy. This is a rendering policy, not source cleanup or physically exact integration of a clipped Gaussian volume; conditional 4D projection is unchanged. Prior near-plane image evidence needs recapture with the current renderer identity.

The bounded [near-clipping fixture](../tests/near_clip_render.rs) covers ordinary quads, ordered quads and GPS with perspective and orthographic cameras. The separate [ordered spatial fixture](../tests/global_order.rs) checks the near-envelope transition at a fixed resident cut. These contracts do not qualify real-source proxy quality or changing-residency recovery.

## Measured native diagnostic

On 2026-09-07, the same compiled `lod_cpu_gpu_oracle` test executable ran first with its embedded production Gaussian shader and then with a pinned replacement of that shader asset. Imported shader modules and test tolerances were identical. The replacement evaluated fragment-position conics; it was subsequently promoted to [gaussian.wgsl](../src/render/gaussian.wgsl), with field renaming and comments.

These historical measurements predate the finite elliptical tail and the later forward-depth ordering correction. They establish the original conic fix, not qualification of the current renderer binary.

The [GPU parity test](../tests/lod_cpu_gpu_oracle.rs) renders ten cases into each of `Rgba16Float` and `Rgba8UnormSrgb`, for 20 image comparisons at 128×128. Cases include rotated anisotropic support, isolated Gaussians, overlapping equal-depth colors, a footprint clipped by the viewport, the same footprint translated inward, additive blending and restoration of ordinary alpha-over blending. The cameras use identity cloud transforms, fixed three-sigma support, full 32-bit forward camera-depth sorting, no MSAA and no tonemapping. The unit-opacity case also exercises vertex versus fragment clamping.

| Comparison | Embedded UV-density shader | Fragment-position conic shader |
| --- | ---: | ---: |
| Image cases within the unchanged bound | 13 / 20 | 20 / 20 |
| Violating channel comparisons across all cases | 1,438 | 0 |
| Violations in the mixed alpha-over Float16 case | 322 | 0 |
| Largest error / allowed bound | 3.342782 | 0.999380 |

Channel totals include repeated scene states; they are not unique affected pixels. The frozen bound propagates target-storage quantization plus a `3e-6` arithmetic allowance through each actual contributing blend. Every pixel is checked, so black background cannot dilute a missed-support error. The bound was not increased after observing the production failure.

The production failure also appears when the clipped Gaussian is translated inward. This rules out viewport clipping alone as its cause. Direct f32 evaluation of the old shader's ellipse formula agrees algebraically with the CPU conic; the isolated shader replacement removes the dependency on raster UV interpolation. The experiment establishes the effectiveness of that change under these cases without identifying a universal hardware interpolation-error bound.

Evidence is under `target/lod-roadmap/2026-09-07/`:

| Artifact | SHA256 |
| --- | --- |
| `cpu-gpu-oracle-v8-production.log` | `8ab312579d6b276f7ab2708a57d7e5602d2c80359f7bdac7186eee12aeebda4c` |
| `cpu-gpu-oracle-v8-fragment-conic.log` | `62210325ecf884e084fe5bb7b9615c839f2e2fa30a6125a0addfe84ce4300974` |
| Original embedded `gaussian.wgsl` | `38c4e404ae523cb4096bcfcedb4089c175d899a2a1a65b664eaf4998e2961ed2` |
| `gaussian-fragment-conic-diagnostic.wgsl` | `e206a43460d6c0e333a0fb1d9a015805bf81c38d3873ced22f9a25fc5a84c462` |

The adjacent `gaussian-fragment-conic-diagnostic.json` records the original source hash, replacement hash and exact patch. `BGS_ORACLE_GAUSSIAN_WGSL` is the test-only override hook; it freezes the replacement bytes before rendering and reports the installed shader identity.

A subsequent 22-case run also passes the unchanged bounds with the current
production shader installed through that explicit hook. It adds viewport origin
`(13, 7)` and size `96x104` within a `128x128` target in both formats, using a
matching local CPU projection and requiring exactly transparent pixels outside
the viewport. `cpu-gpu-oracle-v9-viewport-diagnostic.json` records the executable
and shader identity; the compiled supporting runtime remains v8 in this run.

## Limits and remaining checks

This is a numerical density/projection correction. It does not establish faster rendering, useful LoD reduction, representative quality, temporal stability, or a universal renderer-exact CPU fitting oracle. Triangle edge coverage remains subject to rasterization rules; the rotated witness deliberately avoids samples exactly on a quad edge. The change does not remove that distinction.

The measured A/B uses native Gaussian3d fixtures. The broader covariance-storage/morph feature matrix, nonuniform cloud transforms, orthographic/asymmetric projections, 4D images, WebGPU behavior and MSAA require their own checks. Compiling a shader variant is not equivalent to qualifying its images. The low-resolution CPU fitter therefore continues to report GPU forward parity as unqualified for arbitrary scenes and camera domains.
