# bevy_gaussian_splatting tools

## inspect LoD frame captures

Validate existing JSONL evidence and summarize per-view/scenario counts, timing
percentiles and memory without running the renderer:

```sh
python3 tools/check_lod_capture.py path/to/capture.jsonl
```

See [the capture contract](../docs/lod_capture.md) for the Rust writer, synthetic
fixture, provenance rules and optional evidence completeness checks.

## build a LoD package

Build bounded external runs, merge them, fit the spatial hierarchy, and publish
a verified ABI 17 package:

```bash
cargo run --release --no-default-features --features lod_build_sh3 \
  --bin build_lod -- --input scene.ply --output out/scene
```

Add `--gpu-preprocess` to sort bounded source runs on the GPU. Hierarchy
construction, representative fitting, and package publication use the CPU.
GPU limits are `--gpu-max-input-bytes`, `--gpu-max-sort-commands`, and
`--gpu-max-readback-bytes`. Library integrations use `GpuLodBatchSorter` and
`GpuExternalLodBatchPreprocessor`. See [LoD construction](../docs/lod.md) for
SH profiles, byte bounds, and opt-in preprocessing benchmarks.

## ply to gcloud converter

convert ply files into bevy_gaussian_splatting gcloud file format (more efficient)

```bash
cargo run --bin ply_to_gcloud -- assets/scenes/icecream.ply
```

## render trellis thumbnails

render local example thumbnails from `trellis.ply` render modes.

```bash
cargo run --bin render_trellis_thumbnails --features io_ply
```

## build web output

build wasm, generate wasm-bindgen output, and regenerate `www/examples/thumbnails/*`.

```bash
bash ./tools/build_www.sh
```

on Windows:

```powershell
pwsh ./tools/build_www.ps1
```
