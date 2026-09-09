# Native GPU hierarchy capture

`capture_lod --config PATH.json` supports `render_mode: "hierarchy_point"` for
one native package and one camera. It uses the production GPU hierarchy and
Gaussian point renderer. `capture_mode` must be `instrumented` and
`presentation_mode` must be `Discrete`.

`render_mode: "hierarchy_ordered"` instead uses the production globally ordered
quad renderer. Replace `point_gpu` with `ordered_gpu`, retaining these fields:
`max_projected_gaussians`, `max_gpu_bytes`, `max_traversal_gpu_bytes`,
`max_frontier_nodes`, `max_visited_nodes`, and `max_page_requests`. The two backend
configurations are mutually exclusive. Both share the package, calibrated camera,
residency limits, bounded readback ring and capture protocol. Ordered capture has
fixed record budgets; point sampling controls apply only to point capture.

Add this bounded backend configuration to an ordinary native capture config:

```json
{
  "render_mode": "hierarchy_point",
  "presentation_mode": "Discrete",
  "max_manifest_bytes": 268435456,
  "max_cpu_bytes": 4294967296,
  "max_gpu_bytes": 4294967296,
  "point_gpu": {
    "samples_per_pixel": 1,
    "max_projected_gaussians": 2097152,
    "max_points_per_frame": 67108864,
    "max_gpu_bytes": 536870912,
    "max_traversal_gpu_bytes": 536870912,
    "max_frontier_nodes": 16384,
    "max_visited_nodes": 262144,
    "max_page_requests": 1024
  },
  "camera_path": "Jastrzębia_Góra_camera_path.json",
  "segments": [
    {"scenario": "camera_300_cold", "frames": 600, "camera_frame": 300},
    {"scenario": "camera_300_stationary", "frames": 240, "camera_frame": 300}
  ]
}
```

This fragment does not replace the required source, manifest, output, camera,
residency and sampling fields. Paths are relative to the config. The decoder
override is bounded to 256 MiB and applies only to this capture. Normal asset
loading already accepts per-asset `GaussianLodManifestLoaderSettings`; neither
override changes the library default.

`max_concurrent_requests` controls package transport concurrency and is saved in
the resolved `settings.json`. It defaults to the library's eight requests and
accepts 1–256. Set it explicitly to 64 when comparing loading with the viewer's
default; earlier captures without this field used eight.

An initial SH0 screen can use a 640×425 target, quality 0.5, 2,097,152 active
records, a 4,194,304-record atlas, 512 MiB resident/atlas ceiling, 4096 resident
pages, 16 MiB uploads per frame, three readback slots, capture every eight
frames and a 120-second runtime timeout. These are proposed admission limits,
not measured memory or performance results. The actual package's root cover,
page stride, metadata and simultaneous retained generations must fit. An
outer process timeout should also bound source hashing and application setup.

`camera_frame` is the zero-based file row, independent of its authored ID. It
uses the exact camera-to-world rotation and independent fx/fy calibration from
the [standard 3DGS camera path](camera_paths.md). Frames hold that pose; a sequence
of one-frame segments reproduces a selected route without look-at interpolation.
The original camera JSON is copied into `camera_path_source.json` and its bytes
participate in the camera-path identity alongside the segment schedule: SHA-256
of the exact `camera_path.json` bytes concatenated with
`camera_path_source.json` bytes. Without an imported path, only the segment
file is hashed.

Each completed sample copies the point completion header, traversal counters,
optional timestamps and final image in the same view submission. The receipt
must identify the current hierarchy source and residency generation. Evidence
records selected and projected Gaussians, requested/dispatched point attempts,
visited nodes, page requests, selected pages, overflow and traversal-limit flags.
`counts.compacted` and `counts.drawn` in this pipeline mean admitted projected
Gaussians; they do not mean stochastic point attempts or covered pixels.
Overflow or invalid bounds leave those counts unobserved and do not release
the startup gate. CPU package state is separately labeled as an extraction-time
snapshot; asynchronous acknowledgements can lag the rendered submission.

Ordered capture instead copies the 72-byte draw header and 64-byte traversal
header. Its receipt is recorded after issuing the shared indirect quad draw and
cleared before each frame's preparation. Only a completed header with the expected
four-vertex draw, bounded instance count and no renderer/root failure attests the
image. `counts.compacted` and `counts.drawn` then equal the actual indirect quad
instance count. Safe traversal limits can retain a coarser complete cut; they are
reported separately from image validity. The receipt uses the current draw even
when the renderer's separate diagnostic readback ring is full.

`capture_hierarchy_cut: true` additionally copies the logical selected ranges from
one traversal submission. It defaults to false. `hierarchy_index.json` pins the
manifest hash and defines indices as root aliases in roots order, followed by
nodes in manifest order. The schema-2 `hierarchy_cut` receipt records 16-byte rows:
`[gpu_node_index, physical_output_start, physical_record_count, omission_flags]`.
Flag `0` expands the node's complete representation. `OMITTED_OUTSIDE_VIEW = 1`
requires zero physical records and the enabled conservative current-view world
support policy. Omitted rows retain logical source ownership without expanding
records. Resident pages of the complete logical cut remain pinned for camera
navigation, including siblings of a transition bypassed for near-plane safety.
Logical selection, record admission and complete-child-cohort demand precede
physical omission. An absent offscreen child retains the complete resident parent
fallback until its cohort arrives, exactly as for a visible child. This preserves
the navigation structure across camera motion. Zero-count nodes still require
their complete resident logical cohort; omission provides no residency-skipping
benefit. It saves expansion and rendering work without claiming additional
logical detail budget. Physical starts
can therefore repeat. Raw request counts can repeat shared pages and must not be
interpreted as unique pending I/O. Complete cut and request accounting evidence
does not establish camera-motion detail or residency recovery; those require
matched route measurements.

The physical count sum matches traversal output and is distinct from the number
of logical cut nodes. Readback validates flags and contiguous physical offsets;
[attribution](lod_representative_fitting.md#captured-cut-attribution-and-calibrated-cohorts)
independently checks representation counts and exact full-source coverage against
the pinned manifest before selecting owners. It skips omitted payloads.

The extra ring storage/copy is 16 bytes per admitted frontier slot. Historical
8-byte, two-word receipts remain readable and imply complete representation
expansion. For cuts containing omissions, context export rejects omitted owners
and permits only contained crops of the identical uncropped captured camera,
including exact matrices, calibration viewport, and near/far policy. It rejects
other-camera reprojection because the omitted nodes could contribute there.
The context sidecar records this restriction explicitly.

Instrumented native captures write a separate `request_outcomes.jsonl` at clean
shutdown, with one terminal row per registered request: `schema_version`,
`run_id`, `frame`, `path_frame`, `scenario`, `readback_encoded` and `outcome`.
The outcome `kind` is `completed`, `missing_drawable`, `ring_full`,
`mapping_failure`, `delivery_failure`, `write_failure`, `capture_failure`,
`timeout`, `unsubmitted` or `aborted`. Completed rows separately record
`image_written` and `draw_attested`; an early diagnostic image can lack a draw.
Failure rows retain their reason/error when available. External process termination
can prevent shutdown output and must be treated as missing evidence.

`capture_status.json` includes `request_outcomes`, `request_outcomes_file` and
`request_accounting_complete`. The summary counts terminal outcomes, accounting
errors, `unresolved` and requests still pending when shutdown began
(`unresolved_at_shutdown`). Accounting for a skip or timeout does not produce an
image, attest a draw, or complete capture cadence. `scenario_samples_complete`
and `attested_counts_complete` retain their stricter requested-sample requirements;
the existing version-one image receipt remains unchanged.

GPU timing intervals are `gpu_hierarchy`, `point_backend`, `postprocess` and
`view_total`. They are encoder-boundary intervals and can include other view
work between those boundaries. They are not isolated kernel timings. The point
backend uses a fixed seed. Optional `point_gpu.target_gpu_ms` enables the
existing sample-layer controller; `point_gpu.target_view_gpu_ms` installs the
existing outer record-budget controller. Both default to `null`, preserving
fixed reference settings, and either target requires `request_timestamps: true`.
The outer policy retains the runtime's visible-root floor and is capped by the
configured active/projected record limit. Its configured minimum is the smaller
of 16,384 and that limit. Time-based coarsening requires the inner controller to
reach `point_gpu.min_samples_per_pixel` (default 1). That floor applies to timing,
overflow and allocation admission and must not exceed `samples_per_pixel`.
It expresses a sampling policy; it does not certify representation quality.

Ordered capture reports `ordered_backend` in place of `point_backend`; that
interval includes projection, gathering, radix ordering and quad rendering, plus
any other view work scheduled between the same boundaries.

Point image evidence records the actual layer count. Controller observations
retain their own completed submission IDs and are explicitly separate from
the current image's counters. A healthy fixed run should precede an adaptive
diagnostic; admission, convergence and image quality still need measured checks.

`python3 tools/check_lod_capture.py OUTPUT/capture.jsonl` checks the extended
schema. The adjacent `submission_evidence.jsonl` supplies the point/traversal
proof and limits; the schema check alone does not validate those proofs. Memory
records combine partial buffer observations with a separate shared ownership
ledger. Driver memory, a complete residency audit, image quality and hardware
budget qualification require independent evidence. A spatially filtered flat
reference is not automatically a full-scene quality reference.
