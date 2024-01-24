# bevy_gaussian_splatting 🌌

[![test](https://github.com/mosure/bevy_gaussian_splatting/workflows/test/badge.svg)](https://github.com/Mosure/bevy_gaussian_splatting/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/bevy_gaussian_splatting)](https://raw.githubusercontent.com/mosure/bevy_gaussian_splatting/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/mosure/bevy_gaussian_splatting)](https://github.com/mosure/bevy_gaussian_splatting)
[![GitHub Releases](https://img.shields.io/github/v/release/mosure/bevy_gaussian_splatting?include_prereleases&sort=semver)](https://github.com/mosure/bevy_gaussian_splatting/releases)
[![GitHub Issues](https://img.shields.io/github/issues/mosure/bevy_gaussian_splatting)](https://github.com/mosure/bevy_gaussian_splatting/issues)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/mosure/bevy_gaussian_splatting.svg)](http://isitmaintained.com/project/mosure/bevy_gaussian_splatting)
[![crates.io](https://img.shields.io/crates/v/bevy_gaussian_splatting.svg)](https://crates.io/crates/bevy_gaussian_splatting)

bevy gaussian splatting render pipeline plugin. view the [live demo](https://mosure.github.io/bevy_gaussian_splatting/index.html?arg1=cactus.gcloud)

![alt text](docs/assets/bevy_gaussian_splatting_demo.webp)
![alt text](docs/assets/go.gif)


## capabilities

- [X] ply to gcloud converter
- [X] gcloud and ply asset loaders
- [X] bevy gaussian cloud render pipeline
- [X] gaussian cloud particle effects
- [X] wasm support /w [live demo](https://mosure.github.io/bevy_gaussian_splatting/index.html?arg1=cactus.gcloud)
- [X] depth colorization
- [X] f16 and f32 gcloud
- [X] wgl2 and webgpu
- [ ] spherical harmonic coefficients clustering
- [ ] 4D gaussian cloud wavelet compression
- [ ] accelerated spatial queries
- [ ] temporal depth sorting
- [ ] skeletons
- [ ] volume masks
- [ ] level of detail
- [ ] lighting and shadows
- [ ] bevy_openxr support
- [ ] bevy 3D camera to gaussian cloud pipeline

## usage

```rust
use bevy::prelude::*;
use bevy_gaussian_splatting::GaussianSplattingPlugin;

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, setup_gaussian_cloud)
        .run();
}

fn setup_gaussian_cloud(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    commands.spawn(GaussianSplattingBundle {
        cloud: asset_server.load("scenes/icecream.gcloud"),
        ..Default::default()
    });

    commands.spawn(Camera3dBundle::default());
}
```

## tools

- [ply to gcloud converter](tools/README.md#ply-to-gcloud-converter)
- [gaussian cloud training pipeline](https://github.com/mosure/burn_gaussian_splatting)
- aabb vs. obb gaussian comparison via `cargo run --bin compare_aabb_obb`


### creating gaussian clouds

- [X] 3d gaussian clouds: [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [X] 4d gaussian clouds: [4d-gaussian-splatting](https://fudan-zvg.github.io/4d-gaussian-splatting/)
- [ ] edge-device training pipeline: [burn_gaussian_splatting](https://github.com/mosure/burn_gaussian_splatting)


## compatible bevy versions

| `bevy_gaussian_splatting` | `bevy` |
| :--                       | :--    |
| `0.4 - 1.0`               | `0.12` |
| `0.1 - 0.3`               | `0.11` |


## projects using this plugin
- [kitt2](https://github.com/cs50victor/kitt2)
