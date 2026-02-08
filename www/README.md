# bevy_gaussian_splatting for web

## wasm support

to build wasm run:

```bash
cargo build --target wasm32-unknown-unknown --release --no-default-features --features "web"
```

to generate bindings:
> `wasm-bindgen --out-dir ./www/out/ --target web ./target/wasm32-unknown-unknown/release/bevy_gaussian_splatting.wasm`

to build the web output (wasm + bindings + thumbnails):
> macOS/Linux/CI: `bash ./tools/build_www.sh`
> Windows: `pwsh ./tools/build_www.ps1`

examples page:
- `www/examples/index.html`
- config manifest: `www/examples/examples.json`

manifest notes:
- `args`: viewer query args
- `input_scene` or `input_cloud`: scene/cloud opened when example card is clicked
- `thumbnail_input_scene` or `thumbnail_input_cloud`: optional thumbnail capture override

the build script renders thumbnails via:
> `RENDER_EXAMPLE_THUMBNAILS=1 cargo test --test headless_examples render_example_thumbnails -- --nocapture`

open a live server for `www/index.html` and append args, for example:
> `?input_scene=https%3A%2F%2Fmitchell.mosure.me%2Ftrellis.glb&rasterization_mode=Color`
