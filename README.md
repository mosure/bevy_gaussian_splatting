# bevy_gaussian_splatting 🌌

[![test](https://github.com/mosure/bevy_gaussian_splatting/workflows/test/badge.svg)](https://github.com/Mosure/bevy_gaussian_splatting/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/bevy_gaussian_splatting)](https://raw.githubusercontent.com/mosure/bevy_gaussian_splatting/main/LICENSE-MIT)
[![crates.io](https://img.shields.io/crates/v/bevy_gaussian_splatting.svg)](https://crates.io/crates/bevy_gaussian_splatting)

bevy gaussian splatting render pipeline plugin. view the [live demo gallery](https://mosure.github.io/bevy_gaussian_splatting/examples/) or open [`trellis.glb`](https://mosure.github.io/bevy_gaussian_splatting/index.html?input_scene=https%3A%2F%2Fmitchell.mosure.me%2Ftrellis.glb&rasterization_mode=Color) directly.

![Gaussian splatting demo](https://raw.githubusercontent.com/mosure/bevy_gaussian_splatting/main/docs/bevy_gaussian_splatting_demo.webp)
![Animated Gaussian splatting example](https://raw.githubusercontent.com/mosure/bevy_gaussian_splatting/main/docs/go.gif)


## install

```bash
cargo install bevy_gaussian_splatting
bevy_gaussian_splatting --input-cloud [file://gaussian.ply | https://mitchell.mosure.me/go_trimmed.ply]
bevy_gaussian_splatting --input-scene [file://scene.glb | https://mitchell.mosure.me/trellis.glb]
bevy_gaussian_splatting --input-lod [directory/scene.gsplatlod | https://cdn.example/scene.gsplatlod]
```

Prebuilt packages use bounded sparse residency and native/HTTP range
streaming. See the [LoD guide](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lod.md)
and the normative [package-format contract](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lod_format.md).
Externally trained camera-cluster active sets use the additive
[LODGE sidecar and runtime](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lodge.md)
and its normative
[active-set format](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lodge_active_set_format.md)
without changing the existing `.gsplatlod` hierarchy ABI.

## viewer hotkeys

- `esc`: close viewer
- `s`: save screenshot to `screenshots/`
- `g`: export the loaded gaussian scene to `exports/gaussian_scene_<frame>.glb` (cloud transforms + active camera)
- `f`: freeze/unfreeze the current LoD camera selection for close inspection

For a reproducible native LoD review scene (editor and LoD panel are on by
default):

```bash
cargo run --release --bin bevy_gaussian_splatting -- \
  --gaussian-count=65536 --gaussian-seed=42 --lod-quality=0.65
```

Append `--lod-debug=selection-pressure` to visualize the effective guarded
selection pressure: structural detail and projected error, plus the guarded
high-fidelity certificate demand. Structural coverage strengthens toward the
high-quality end; certificate pressure is intentionally inactive through `.90`
and reaches full authority at `.95`.

For a larger native review, open Trellis at the guarded high-fidelity setting:

```bash
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-scene=https://mitchell.mosure.me/trellis.glb \
  --lod-quality=0.95
```

The editor and Gaussian LoD panel are on by default. The GLB is loaded as a
resident scene and receives a transient CPU-built hierarchy; this is not the
out-of-core `.gsplatlod` package path. Quality `.95` is a safety gate, not a
promise that a coarser representation exists, so exact source leaves are a
valid result. Compare authored color at quality `1`, inspect `.95`, then sweep
`.90` through `.95` when evaluating useful savings. The panel's selected-splat
count confirms whether the selected cut actually saves records. `level` and
`selection-pressure` explain the logical cut; `page` shows physical packing,
and `residency` is mainly useful for streamed packages. Use `F` to freeze a
distant cut and move closer to inspect exactly what it contains. The URL is a
review convenience; use the
[hash-pinned quality audit](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lod_quality_report.md)
for regression results.


## capabilities

- [X] ply to gcloud converter
- [X] gcloud and ply asset loaders
- [X] bevy gaussian cloud render pipeline
- [X] gaussian cloud particle effects
- [X] wasm support /w [live demo](https://mosure.github.io/bevy_gaussian_splatting/index.html)
- [X] depth colorization
- [X] normal rendering
- [X] f16 and f32 gcloud
- [X] wgl2 and webgpu
- [X] multi-format scenes
- [X] 2dgs
- [X] 3dgs
- [x] 4dgs
- [X] [glTF `KHR_gaussian_splatting`](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting) scene load/save
- [ ] 4dgs motion blur
- [ ] [deformable radial kernel](https://github.com/VAST-AI-Research/Deformable-Radial-Kernel-Splatting)
- [ ] implicit mlp node (isotropic rotation, color)
- [ ] temporal gaussian hierarchy
- [ ] gcloud, spherical harmonic coefficients Huffman encoding
- [ ] [spz](https://github.com/nianticlabs/spz) format io
- [ ] spherical harmonic coefficients clustering
- [ ] 4D gaussian cloud wavelet compression
- [ ] accelerated spatial queries
- [X] rotation-stable world-distance sorting and sort-cache reuse
- [ ] per-ray hierarchical [StopThePop](https://arxiv.org/abs/2402.00525) compositing
- [ ] temporal depth sorting
- [ ] skeletons
- [ ] volume masks
- [X] [bounded CPU/GPU LoD construction, globally covering guard cuts, native/HTTP package streaming with persistent caches, atomic complete-cut commits, automatic GPU-atlas bridges, exact compaction/radix, and device recovery](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lod.md)
- [X] [authenticated external LODGE levels and camera-cluster active sets with one deduplicated, opacity-blended global draw](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lodge.md)
- [X] [named LoD level, page, residency, pressure, and boundary debug views](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lod_debug.md)
- [ ] lighting and shadows
- [ ] bevy_openxr support
- [ ] bevy 3D camera to gaussian cloud pipeline


## usage

```rust
use bevy::prelude::*;
use bevy_gaussian_splatting::{
    CloudSettings,
    GaussianSplattingPlugin,
    PlanarGaussian3dHandle,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup_gaussian_cloud)
        .run();
}

fn setup_gaussian_cloud(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    // CloudSettings and Visibility are automatically added
    commands.spawn((
        PlanarGaussian3dHandle(asset_server.load("scenes/icecream.gcloud")),
        CloudSettings::default(),
    ));

    commands.spawn(Camera3d::default());
}
```


## tools

- [LoD architecture, package builder, fixtures, tests, and benchmarks](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lod.md)
- [External LODGE active-set format, instantiation, and presentation contract](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lodge.md)
- [Normative `.gslodge` container, manifest, dependency, and membership format](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lodge_active_set_format.md)
- [Pinned Trellis LoD quality report and regression protocol](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lod_quality_report.md)
- [Named LoD level, page, residency, pressure, and boundary debug views](https://github.com/mosure/bevy_gaussian_splatting/blob/main/docs/lod_debug.md)
- [Security policy and LoD package trust boundary](https://github.com/mosure/bevy_gaussian_splatting/blob/main/SECURITY.md)
- [ply to gcloud converter](https://github.com/mosure/bevy_gaussian_splatting/blob/main/tools/README.md#ply-to-gcloud-converter)
- [gaussian cloud training pipeline](https://github.com/mosure/burn_gaussian_splatting)
- aabb vs. obb gaussian comparison via `cargo run --bin compare_aabb_obb`


### creating gaussian clouds

the following tools are compatible with `bevy_gaussian_splatting`:

- [X] 2d gaussian clouds:
    - [gsplat](https://docs.gsplat.studio/main/)

- [X] 3d gaussian clouds:
    - [brush](https://github.com/ArthurBrussee/brush)
    - [gsplat](https://docs.gsplat.studio/main/)
    - [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

- [X] 4d gaussian clouds:
    - [4d-gaussian-splatting](https://fudan-zvg.github.io/4d-gaussian-splatting/)
        - [4dgs ply-export](https://gist.github.com/mosure/d9d4d271e05a106157ce39db62ec4f84)
    - [easy-volcap](https://github.com/zju3dv/EasyVolcap)


## compatible bevy versions

| `bevy_gaussian_splatting` | `bevy` |
| :--                       | :--    |
| `9.0`                     | `0.19` |
| `8.0`                     | `0.19` |
| `7.0`                     | `0.18` |
| `6.0`                     | `0.17` |
| `5.0`                     | `0.16` |
| `3.0`                     | `0.15` |
| `2.3`                     | `0.14` |
| `2.1`                     | `0.13` |
| `0.4 - 2.0`               | `0.12` |
| `0.1 - 0.3`               | `0.11` |


## license
licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


## contribution

unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.


## analytics
![alt](https://repobeats.axiom.co/api/embed/4f273f05f00ec57e90be34727e85952039e1a712.svg "analytics")
