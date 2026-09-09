#!/usr/bin/env bash
# Build only. This deliberately does not invoke build_www's GPU thumbnails.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
degree="${BGS_BROWSER_SH_DEGREE:-3}"
case "$degree" in 0|3) ;; *) echo "BGS_BROWSER_SH_DEGREE must be 0 or 3" >&2; exit 2 ;; esac
version="$(python3 -c 'import tomllib; d=tomllib.load(open("Cargo.lock","rb")); print(next(p["version"] for p in d["package"] if p["name"]=="wasm-bindgen"))')"
actual="$(wasm-bindgen --version)"
if [[ "$actual" != "wasm-bindgen $version" ]]; then
  echo "Install matching bindings: cargo install wasm-bindgen-cli --version $version --locked" >&2
  exit 2
fi
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS RUSTDOCFLAGS || true
features="planar lod_render sh${degree} io_flexbuffers web_asset webgpu testing"
out="www/out/lod-browser"
mkdir -p "$out"
identity="$(mktemp)"
trap 'rm -f "$identity"' EXIT
python3 tools/serve_lod_browser_capture.py --source-identity > "$identity"
CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" cargo build --locked --release \
  --target wasm32-unknown-unknown --bin capture_lod_browser --no-default-features --features "$features"
wasm-bindgen --out-dir "$out" --target web \
  target/wasm32-unknown-unknown/release/capture_lod_browser.wasm
python3 tools/serve_lod_browser_capture.py --write-build-identity "$identity" --features "$features"
echo "Browser capture build ready. See docs/lod_browser_qualification.md for the bounded run."
