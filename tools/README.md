# bevy_gaussian_splatting tools

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
