# Migrating to the current LoD branch

This branch removes unfinished backends and changes APIs without compatibility
aliases. The engine remains experimental; the [acceptance status](lod_implementation_status.md)
records the quality, performance and platform gates still required before a
production release.

| Previous feature or API | Current replacement |
| --- | --- |
| `buffer_texture`, `packed` renderer features and `render::{texture, packed}` | Use planar storage buffers: `planar` + `buffer_storage`. Packed CPU/IO representations remain separate from renderer selection. |
| `webgl2` | Use `webgpu` for browser rendering. The `web` profile includes it; build that SH0 profile with `--no-default-features --features web`. |
| Empty `sort_bitonic` feature/module | Use `sort_radix`, `sort_std` or `sort_rayon`. `lod_render` includes the required storage-buffer and radix features. |
| SH3 implicitly enabled by `headless` | Select the asset ABI separately: `--no-default-features --features 'headless sh0'` or `'headless sh3'`. Builder profiles `lod_build_sh0` and `lod_build_sh3` also select that ABI. Normal default builds retain SH3. |
| `gaussian::lod_build_gpu::hierarchy::{GpuLodHierarchyBuilder, GpuLodHierarchyLimits, GpuLodHierarchyError}` | Use `gaussian::lod_build_gpu::sort::{GpuLodBatchSorter, GpuLodSortLimits, GpuLodSortError}`. GPU preprocessing sorts canonical batches; CPU code constructs the hierarchy and its representatives. |
| `GpuHierarchyExternalLodBatchPreprocessor` | Use `GpuExternalLodBatchPreprocessor` with the external package builder. |
| Internal dense `LodTransientAtlas::new(...)` or sparse `new_empty(capacity)` | Use `LodTransientAtlas::new(capacity)` followed by `write_slot(...)`. The constructor allocates no dense CPU Gaussian mirror. Internal registry `register` also drops its former `source_count` argument. |
| GPS-specific package image receipts | Use `render::traversal::{GpuLodDrawAcknowledgement, GpuLodDrawAcknowledgements, GpuLodDrawRenderer}`. Match source and residency generation; submission counters belong to the identified renderer and are not comparable across renderer switches. |
| Radial-distance quad keys and `SortTrigger::last_camera_position` | Quad sorters use forward camera depth (`-view-space Z`). `SortTrigger::last_camera_depth` stores its world-space plane; rotation invalidates order. CPU sorters use the same unsigned key/invalid-visibility convention as radix, with stable source-index ties. Re-capture image references made with radial ordering. |
| Sort storage indexed by `Camera.order` or square-padded record counts | Camera slots are assigned independently of render order. CPU slices use retained `SortedEntries::entry_count`; GPU slices use device-aligned `GpuSortedEntry::camera_stride`. The redundant `GpuSortedEntry::count` field is removed. |

Ordinary per-cloud quads remain the default. Select shared ordered quads with
`GaussianGlobalOrderSettings` or GPS with `GaussianPointSplattingSettings` on a
`GaussianCamera`; both use `Msaa::Off` in their supported 3D color profile. For
package traversal, also attach `GaussianGpuLodPackage` to the cloud and
`GpuLodTraversalSettings` to the camera. GPU selection requires dynamic,
discrete cuts. See [shared quads](global_quad_order.md) and [GPS](gaussian_point_splatting.md)
for material, composition and budget limits.

For an existing package matching the default SH3 profile:

```sh
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-lod path/to/scene.gsplatlod --global-order --lod-gpu-traversal \
  --lod-max-active-gaussians 65536
```

Use `--point-splatting` in place of `--global-order` to select GPS. Its optional
sampling and view-budget controllers are separate from shared quad ordering.
