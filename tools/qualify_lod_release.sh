#!/usr/bin/env bash
set -euo pipefail

toolchain="${BGS_RUST_TOOLCHAIN:-1.95.0}"
cargo_cmd=(cargo "+${toolchain}")
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
if [[ -n "${BGS_TMPDIR:-}" ]]; then
  export TMPDIR="${BGS_TMPDIR}"
fi

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

run "${cargo_cmd[@]}" fmt --all -- --check
run git diff --check

run "${cargo_cmd[@]}" check --locked --lib
run "${cargo_cmd[@]}" check --locked --no-default-features \
  --bin build_lod --features 'lod_build'
run "${cargo_cmd[@]}" test --locked
run "${cargo_cmd[@]}" test --locked --lib --features 'lod_build testing'
run "${cargo_cmd[@]}" clippy --locked --all-targets \
  --features 'lod_build testing' -- -D warnings

supported='planar lod_render sh0 testing io_flexbuffers'
precomputed="${supported} precompute_covariance_3d"
portable='planar buffer_storage lod sh0 sort_std io_flexbuffers'

run "${cargo_cmd[@]}" clippy --locked --lib --tests --no-default-features \
  --features "${supported}" -- -D warnings
run "${cargo_cmd[@]}" clippy --locked --lib --tests --no-default-features \
  --features "${precomputed}" -- -D warnings
run "${cargo_cmd[@]}" test --locked --lib --no-default-features \
  --features "${portable}" \
  lod_render_path_support_matches_the_functional_handshake_cfg

if ! rustup target list --toolchain "${toolchain}" --installed \
  | grep -qx wasm32-unknown-unknown; then
  printf 'wasm32-unknown-unknown is required for %s release qualification\n' \
    "${toolchain}" >&2
  exit 1
fi
run "${cargo_cmd[@]}" clippy --locked --lib --target wasm32-unknown-unknown \
  --no-default-features --features web -- -D warnings
run "${cargo_cmd[@]}" clippy --locked --lib --target wasm32-unknown-unknown \
  --no-default-features --features 'web precompute_covariance_3d' -- -D warnings
run "${cargo_cmd[@]}" test --locked --lib --target wasm32-unknown-unknown \
  --no-run --no-default-features \
  --features 'planar lod_render sh0 io_flexbuffers'
run "${cargo_cmd[@]}" check --locked --target wasm32-unknown-unknown \
  --no-default-features --features 'lod_build' --bin build_lod

run cargo +nightly check --locked --manifest-path fuzz/Cargo.toml --bins

run "${cargo_cmd[@]}" doc --locked --lib --no-deps --features 'lod_build testing'
run "${cargo_cmd[@]}" package --list --locked --allow-dirty
run "${cargo_cmd[@]}" publish --dry-run --locked --allow-dirty
