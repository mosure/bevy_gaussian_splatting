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
wasm_bindgen_lock_version="$(
  awk '
    $0 == "name = \"wasm-bindgen\"" { in_wasm_bindgen = 1; next }
    in_wasm_bindgen && $1 == "version" {
      gsub(/"/, "", $3)
      print $3
      exit
    }
  ' Cargo.lock
)"
if [[ -z "${wasm_bindgen_lock_version}" ]]; then
  echo "failed to resolve wasm-bindgen version from Cargo.lock" >&2
  exit 1
fi
wasm_bindgen_cli_version="$("${wasm_bindgen_cmd}" --version | awk '{ print $2 }')"
if [[ "${wasm_bindgen_cli_version}" != "${wasm_bindgen_lock_version}" ]]; then
  echo "wasm-bindgen CLI version ${wasm_bindgen_cli_version} does not match Cargo.lock wasm-bindgen ${wasm_bindgen_lock_version}" >&2
  echo "install the matching CLI with: cargo install wasm-bindgen-cli --version ${wasm_bindgen_lock_version} --locked --force" >&2
  exit 1
fi

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

scene_cache_dir="./assets/.thumbnail_cache"
mkdir -p "${scene_cache_dir}"

if command -v curl >/dev/null 2>&1; then
  remote_scene_list="$(
    grep -Eo '"(thumbnail_input_scene|input_scene)"[[:space:]]*:[[:space:]]*"https?://[^"]+"' ./www/examples/examples.json \
      | sed -E 's/.*"(https?:\/\/[^"]+)".*/\1/' \
      | sort -u || true
  )"

  if [[ -n "${remote_scene_list}" ]]; then
    echo "Caching remote scenes for thumbnails..."
    while IFS= read -r scene_url; do
      if [[ -z "${scene_url}" ]]; then
        continue
      fi

      url_without_query="${scene_url%%\?*}"
      scene_file="${url_without_query##*/}"
      if [[ -z "${scene_file}" ]]; then
        continue
      fi

      scene_path="${scene_cache_dir}/${scene_file}"
      echo "  ${scene_url} -> ${scene_path}"
      curl \
        --fail \
        --location \
        --retry 4 \
        --retry-delay 2 \
        --connect-timeout 10 \
        --max-time 120 \
        "${scene_url}" \
        --output "${scene_path}"
    done <<< "${remote_scene_list}"

    export THUMBNAIL_SCENE_CACHE_DIR="${scene_cache_dir}"
    export THUMBNAIL_SCENE_CACHE_STRICT="1"
  fi
fi

echo "Rendering example thumbnails from manifest..."
RENDER_EXAMPLE_THUMBNAILS=1 THUMBNAIL_SORT_MODE=std "${cargo_cmd}" test --test headless_examples render_example_thumbnails -- --nocapture

if [[ "${THUMBNAIL_SCENE_CACHE_CLEANUP:-0}" == "1" ]]; then
  echo "Cleaning thumbnail scene cache..."
  rm -rf "${scene_cache_dir}"
fi

echo "www build complete."
