# bevy_gaussian_splatting 🌌

[![test](https://github.com/mosure/bevy_gaussian_splatting/workflows/test/badge.svg)](https://github.com/Mosure/bevy_gaussian_splatting/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/bevy_gaussian_splatting)](https://raw.githubusercontent.com/mosure/bevy_gaussian_splatting/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/mosure/bevy_gaussian_splatting)](https://github.com/mosure/bevy_gaussian_splatting)
[![GitHub Releases](https://img.shields.io/github/v/release/mosure/bevy_gaussian_splatting?include_prereleases&sort=semver)](https://github.com/mosure/bevy_gaussian_splatting/releases)
[![GitHub Issues](https://img.shields.io/github/issues/mosure/bevy_gaussian_splatting)](https://github.com/mosure/bevy_gaussian_splatting/issues)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/mosure/bevy_gaussian_splatting.svg)](http://isitmaintained.com/project/mosure/bevy_gaussian_splatting)
[![crates.io](https://img.shields.io/crates/v/bevy_gaussian_splatting.svg)](https://crates.io/crates/bevy_gaussian_splatting)

bevy gaussian splatting render pipeline plugin. view the [live demo](https://mosure.github.io/bevy_gaussian_splatting?input_file=scenes/go_board.gcloud)

![Alt text](docs/bevy_gaussian_splatting_demo.webp)
![Alt text](docs/go.gif)


## capabilities

- [X] ply to gcloud converter
- [X] gcloud and ply asset loaders
- [X] bevy gaussian cloud render pipeline
- [X] gaussian cloud particle effects
- [X] wasm support /w [live demo](https://mosure.github.io/bevy_gaussian_splatting/index.html?arg1=cactus.gcloud)
- [X] depth colorization
- [X] f16 and f32 gcloud
- [X] wgl2 and webgpu
- [ ] 4dgs
- [X] 3dgs
- [X] 2dgs
- [ ] temporal gaussian hierarchy
- [ ] gcloud, spherical harmonic coefficients Huffman encoding
- [ ] [spz](https://github.com/nianticlabs/spz) format io
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
    // GaussianCloudSettings and Visibility are automatically added
    commands.spawn(
        GaussianCloudHandle(asset_server.load("scenes/icecream.gcloud")),
    );

    commands.spawn(Camera3dBundle::default());
}
```

## tools

- [ply to gcloud converter](tools/README.md#ply-to-gcloud-converter)
- [gaussian cloud training pipeline](https://github.com/mosure/burn_gaussian_splatting)
- aabb vs. obb gaussian comparison via `cargo run --bin compare_aabb_obb`


## compatible bevy versions

| `bevy_gaussian_splatting` | `bevy` |
| :--                       | :--    |
| `2.8`                     | `0.15` |
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
