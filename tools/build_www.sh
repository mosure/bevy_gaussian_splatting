#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

resolve_cmd() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    echo "${name}"
    return 0
  fi
  if command -v "${name}.exe" >/dev/null 2>&1; then
    echo "${name}.exe"
    return 0
  fi
  if [[ -x "${HOME}/.cargo/bin/${name}" ]]; then
    echo "${HOME}/.cargo/bin/${name}"
    return 0
  fi
  if [[ -x "${HOME}/.cargo/bin/${name}.exe" ]]; then
    echo "${HOME}/.cargo/bin/${name}.exe"
    return 0
  fi
  return 1
}

cargo_cmd="$(resolve_cmd cargo)" || {
  echo "cargo not found on PATH" >&2
  exit 1
}
wasm_bindgen_cmd="$(resolve_cmd wasm-bindgen)" || {
  echo "wasm-bindgen not found on PATH" >&2
  exit 1
}

# Ensure wasm builds ignore host-specific rust flag overrides from outer env.
unset RUSTFLAGS || true
unset CARGO_ENCODED_RUSTFLAGS || true
unset RUSTDOCFLAGS || true

echo "Building wasm (web feature set)..."
"${cargo_cmd}" build --target wasm32-unknown-unknown --release --no-default-features --features web

wasm_path="./target/wasm32-unknown-unknown/release/bevy_gaussian_splatting.wasm"
if [[ ! -f "${wasm_path}" ]]; then
  echo "wasm output not found at ${wasm_path}" >&2
  exit 1
fi

echo "Generating wasm bindings..."
"${wasm_bindgen_cmd}" --out-dir ./www/out --target web "${wasm_path}"

echo "Rendering example thumbnails from manifest..."
RENDER_EXAMPLE_THUMBNAILS=1 "${cargo_cmd}" test --test headless_examples render_example_thumbnails -- --nocapture

echo "www build complete."
