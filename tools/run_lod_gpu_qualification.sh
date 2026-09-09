#!/usr/bin/env bash
set -euo pipefail

if [[ "${BGS_RUN_GPU_QUALIFICATION:-0}" != 1 ]]; then
  echo 'set BGS_RUN_GPU_QUALIFICATION=1 to launch the GPU qualification suite' >&2
  exit 2
fi

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

# Compact current-renderer contracts, shared with CI. Each ABI is built and run
# independently; headless deliberately does not select spherical harmonics.
suite="${BGS_GPU_QUALIFICATION_SUITE:-full}"
if [[ "${suite}" != full && "${suite}" != core ]]; then
  echo 'BGS_GPU_QUALIFICATION_SUITE must be core or full' >&2
  exit 2
fi
for harmonics in sh0 sh3; do
  features="headless testing lod_build_${harmonics}"
  RUN_GPU_RENDER_TESTS=1 run "${cargo_cmd[@]}" test --locked \
    --no-default-features --features "${features}" \
    --test multi_camera_sort --test global_order \
    --test gaussian_point_splatting --test gpu_lod_package --test near_clip_render \
    -- --include-ignored --nocapture --test-threads=1
  RUN_GPU_RENDER_TESTS=1 run "${cargo_cmd[@]}" test --locked --lib \
    --no-default-features --features "${features}" \
    gpu_traversal_matches_cpu_cuts_and_preserves_bounded_parent_fallback \
    -- --ignored --nocapture --test-threads=1
done
if [[ "${suite}" == core ]]; then
  exit 0
fi

RUN_GPU_RENDER_TESTS=1 run "${cargo_cmd[@]}" test --locked \
  --no-default-features --features 'headless sh3' --test visibility_render \
  -- --nocapture --test-threads=1
RUN_GPU_RENDER_TESTS=1 run "${cargo_cmd[@]}" test --locked \
  --no-default-features --features 'headless sh3 testing' --test lod_range_expansion \
  -- --nocapture --test-threads=1
for covariance in '' 'precompute_covariance_3d'; do
  RUN_GPU_RENDER_TESTS=1 run "${cargo_cmd[@]}" test --locked \
    --no-default-features --features "headless sh3 testing ${covariance}" \
    --test lod_cpu_gpu_oracle -- --nocapture --test-threads=1
  RUN_GPU_RENDER_TESTS=1 run "${cargo_cmd[@]}" test --locked \
    --no-default-features --features "headless sh3 ${covariance}" \
    --test support_overlap_render -- --nocapture --test-threads=1
  RUN_GPU_RENDER_TESTS=1 run "${cargo_cmd[@]}" test --locked \
    --no-default-features --features "headless sh3 ${covariance}" \
    --test lod_debug_render -- --nocapture --test-threads=1
  RUN_GPU_RENDER_TESTS=1 run "${cargo_cmd[@]}" test --locked \
    --no-default-features --features "headless sh3 testing ${covariance}" \
    --test lod_quality_render -- --nocapture --test-threads=1
  RUN_GPU_RENDER_TESTS=1 run "${cargo_cmd[@]}" test --locked \
    --no-default-features --features "headless testing lod_build_sh3 ${covariance}" \
    --test lod_morph_radiance_render \
    headless::authenticated_abi16_k2_morph_radiance_visibility \
    -- --exact --nocapture --test-threads=1
done

RUN_GPU_LOD_ATLAS_TESTS=1 run "${cargo_cmd[@]}" test --locked --lib \
  --no-default-features \
  --features 'planar buffer_storage lod sh0 sort_std io_flexbuffers' \
  gpu_atlas_copy_matches_cpu_oracle -- --ignored --nocapture
RUN_GPU_LOD_ATLAS_TESTS=1 run "${cargo_cmd[@]}" test --locked --lib \
  --no-default-features \
  --features 'planar buffer_storage lod sh0 sort_std io_flexbuffers precompute_covariance_3d' \
  gpu_atlas_copy_matches_cpu_oracle -- --ignored --nocapture

RUN_GPU_DEVICE_LOSS_TESTS=1 run "${cargo_cmd[@]}" test --locked \
  --no-default-features --features 'headless sh3 testing' \
  --test lod_device_recovery \
  headless::active_lod_render_recovers_after_injected_device_loss \
  -- --exact --nocapture --test-threads=1
RUN_GPU_DEVICE_LOSS_TESTS=1 run "${cargo_cmd[@]}" test --locked \
  --no-default-features --features 'headless sh3 testing precompute_covariance_3d' \
  --test lod_device_recovery \
  headless::active_lod_render_recovers_after_injected_device_loss \
  -- --exact --nocapture --test-threads=1

RUN_GPU_LOD_PREPROCESS_TESTS=1 run "${cargo_cmd[@]}" test --locked --lib \
  --no-default-features \
  --features 'lod_build_sh3 lod testing' gpu_collision_sort_matches_cpu_canonical_order \
  -- --ignored --nocapture --test-threads=1
RUN_GPU_LOD_PREPROCESS_TESTS=1 run "${cargo_cmd[@]}" test --locked --lib \
  --no-default-features \
  --features 'lod_build_sh3 lod testing' gpu_sorted_external_multi_run_matches_cpu_package \
  -- --ignored --nocapture --test-threads=1

if [[ "${BGS_RUN_CROSS_ADAPTER:-0}" == 1 ]]; then
  for covariance in '' 'precompute_covariance_3d'; do
    RUN_GPU_CROSS_ADAPTER_TESTS=1 LOD_MIN_ADAPTER_GOLDENS="${LOD_MIN_ADAPTER_GOLDENS:-2}" \
      run "${cargo_cmd[@]}" test --locked --no-default-features \
        --features "headless sh3 testing ${covariance}" --test lod_device_recovery \
        headless::cross_adapter_goldens -- --exact --nocapture --test-threads=1
  done
fi
