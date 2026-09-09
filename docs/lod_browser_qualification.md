# Browser package execution qualification

This opt-in harness runs the real Bevy package, HTTP, preprocessing, atlas and
Gaussian render systems in browser WebGPU. Its default quad profile uses
compaction and radix sorting; the explicit GPS profile uses GPU hierarchy
traversal and Gaussian point splatting. It shares the ordinary canvas/frontend and
renderer; there is no substitute JavaScript shader.
The existing browser unit suite remains useful for HTTP/cache contracts, but does
not provide this renderer execution evidence.

The runner exercises cold package instantiation, stationary rendering, a fixed
camera movement, package despawn, queue-fenced allocation retirement, and package
reload in the same session. Reload may reuse Bevy's manifest asset and HTTP state;
it is not a claim about a browser with empty caches. Persistent Cache Storage is
disabled in this focused lifecycle profile and has its own existing browser tests.

Do not run the commands below until a browser/GPU experiment slot is authorized.
`tools/build_www.sh` also renders native example thumbnails; the dedicated build
script below only compiles and generates bindings.

```sh
# Install/use the toolchain and wasm-bindgen CLI matching this checkout first.
# SH degree is part of the package ABI: existing SH3 packages need this profile.
BGS_BROWSER_SH_DEGREE=3 CARGO_BUILD_JOBS=2 bash tools/build_lod_browser_capture.sh

# Serve one already-built, bounded package. This does not build source scenes.
python3 tools/serve_lod_browser_capture.py \
  --package /absolute/path/to/package \
  --manifest scene.gsplatlod \
  --config tools/fixtures/lod_browser_capture.json

# In a separate terminal, use a new output directory for each operating point.
python3 tools/run_lod_browser_capture.py \
  --output target/lod-browser-qualification/run-001 \
  --chromedriver /absolute/path/to/chromedriver

python3 tools/check_lod_browser_capture.py \
  target/lod-browser-qualification/run-001/evidence.jsonl
```

The browser can also be opened manually at
`http://127.0.0.1:8765/lod-qualification.html`; it enables the evidence download
after completion or failure. For a visible automation run, add `--headed`.
ChromeDriver must match the installed Chrome version. The launcher does not force
a software adapter or disable GPU rendering. The result reports Bevy's actual
adapter/backend/driver; validation rejects a reported software adapter by default.
An explicit `--allow-software` on the launcher and on subsequent checker calls
permits a distinctly labeled software diagnostic. It does not select an adapter
or relax artifact, request-chain, count, snapshot, lifecycle or memory checks.
Such results report `qualification: "software_diagnostic"` and
`hardware_execution_verified: false`; they cannot qualify hardware execution.
Before Wasm starts, the frontend also observes the renderer's original
`requestAdapter` and returned adapter's `requestDevice` promises. It does not make
an independent adapter/device request. Their linked IDs, exact Wasm hash,
requested options, adapter info/fallback flag, and actual device limits are
exported as `browser_adapter`/`browser_device` records. The checker reports
`hardware_adapter_attested` only when this complete request chain succeeds with
nonempty identity and an explicit false fallback flag. Older captures with only
wgpu's generic `BrowserWebGpu` identity retain their execution evidence and
explicitly report no hardware adapter attestation. An unavailable WebGPU device
is a failed/incomplete run; a successful headed run does not qualify headless
execution.

The current v2 profile also supports GPS with GPU package traversal. Use an
already-built tiny package, such as the 32-source-record fixture, with:

```sh
python3 tools/serve_lod_browser_capture.py \
  --package /absolute/path/to/tiny-package \
  --manifest scene.gsplatlod \
  --config tools/fixtures/lod_browser_point_gpu_capture.json

python3 tools/run_lod_browser_capture.py \
  --output target/lod-browser-qualification/point-gpu-001 \
  --chromedriver /absolute/path/to/chromedriver \
  --timeout-seconds 90

python3 tools/check_lod_browser_capture.py \
  target/lod-browser-qualification/point-gpu-001/evidence.jsonl
```

For an intentionally permitted software run, append `--allow-software` to both
the launcher and checker commands. GPS browser execution is still pending until
an actual capture passes these checks; the native tests and historical NVIDIA
quad captures below do not establish this profile's browser execution.

The `point_gpu` configuration object is the explicit backend opt-in. It installs
GPS and GPU traversal on the camera with MSAA disabled, marks the package for GPU
selection and uses dynamic, discrete LoD. The supplied fixture uses 160×90 pixels,
one sampling layer, a 32-Gaussian projection/selection cap, at most 262,144 point attempts,
a 256-record atlas, one observation every four frames, 24-frame motion/hold stages
and a 90-second application timeout. These small limits require a compatible tiny
package; they are not a large-scene operating point. Unload removes both camera
backend components so their buffers can retire before the zero-reservation check;
reload reinstalls them.

Adjust the three camera coordinates in a copied JSON configuration to the actual
scene bounds. The supplied quad configuration is a small origin-centered diagnostic:
960×540, 120 frames per motion/hold stage, one observation every 10 frames,
256 metadata records per application frame, and a 180-second whole-run timeout.
Its 256K physical record cap and 64 MiB atlas byte cap must admit the package's
largest physical page and complete root cover. Choose the matching SH0 build
with `BGS_BROWSER_SH_DEGREE=0` for a package actually encoded with the SH0 ABI.

The page hashes the served manifest and exact Wasm before startup. Build identity
includes the Git revision and a hash of current Rust, WGSL, Cargo manifests and
lockfiles, including untracked sources. A source change during the build rejects
publication of that identity. The local server provides authenticated immutable
ETags and exact byte ranges; its first request for an object computes an ETag hash.
That server-side hashing can affect cold latency, so this local diagnostic is not
a CDN throughput benchmark.

Each quad GPU observation carries its application frame, lifecycle phase, view,
camera/projection, atlas epoch, compaction/radix generation and exact candidate
fingerprints. It copies indirect arguments after rendering in the same view
submission. A draw command probe attests the precise buffer/generation used by
the real renderer. A compacted count alone never becomes a claimed drawn count.

The GPS profile emits `gpu_point_frame` observations instead of quad draw proof.
After point composition, the harness copies the 32-byte point work header and
the 64-byte traversal feedback header into one readback slot in the same command
submission. It borrows the exact input receipt retained with the newly encoded
point image and matches its cloud, source asset and residency generation to the
current traversal output and main-world snapshot. A retained previous image or
an unrelated/latest asynchronous acknowledgement cannot attest new work.

GPS rows report projected Gaussians, full requested/dispatched point attempts,
sampling layers, hierarchy-selected Gaussians, visited nodes, page demand and
limit flags. Point flags must indicate complete successful work; hierarchy flags
may report bounded parent fallback, but cannot report an incomplete root cover.
Selected/projected counts and point attempts have separate meanings: attempts
precede viewport/support rejection and do not measure final visible pixels.
CPU candidate counts and quad indirect draws are never substituted for GPS
execution evidence.

The v2 schema uses three 96-byte readback buffers (288 bytes total) for both
profiles. Historical v1 quad captures retain their three 64-byte buffers
(192 bytes total) and remain accepted by the checker. Mapping callbacks are
asynchronous; Bevy's systems never wait for GPU completion. Full rings, callback failures, overflow,
stamp mismatches or missing lifecycle evidence prevent successful validation.

Main-world observations include package residency/selection, queue and
preprocessing state, frame cadence and the shared CPU/GPU reservation ledger.
Unload must reach zero scene-owned reservations before reload. These are capacity
reservations, not process RSS, browser heap or driver-memory measurements. The
instrumentation ring is reported separately. The exported screenshot is
an actual browser screenshot for inspection, not a flat-reference quality test.

The validator reports `execution_verified` only after all required phases have
attested nonzero quad draws or completed point work, and memory/count/stamp checks
pass. `hardware_execution_verified` additionally requires a successful hardware
adapter/device request chain and normal hardware validation mode. It always reports
`release_qualified: false`: image quality, temporal quality, realistic memory
pressure, different adapters, large-scene timing and native/browser parity need
their own measured operating points. Its lifecycle readiness condition accepts
positive attested work, including bootstrap. A q=1 lifecycle pass alone does
not prove the original leaf cut was reached; require separately attested source
counts in stationary, moving and reloaded phases.

On 2026-09-07, the frozen v10 SH3 build completed two actual headed Chrome
152.0.7977.64 runs of the 84,348-record Icecream package. Both used the renderer's
observed NVIDIA Blackwell, non-fallback `BrowserWebGpu` adapter/device request
chain, with the recorded Vulkan/ForceEnableWebGpuInterop browser flags. The
Wasm SHA256 was
`b93d455434b6b5226286a09a197653e0a2794cec15a10893559ea95449a9a5d8`;
the authenticated manifest SHA256 was
`5dbe0d8b531414b26d03c59e6d144e7c68f45da3b0c2aada0e29f3cc37265e27`.

The q=0 run recorded 35 GPU and 39 main-world observations, with 165 actually
drawn records in every attested phase. The q=1 run recorded 359 GPU and 363
main-world observations. Its frozen protocol required
`selected == candidates == candidate_hits == compacted == drawn == 84,348`,
with draw-command attestation, valid counts and zero overflow, in each of the
stationary, moving and reloaded phases. The evidence satisfies that requirement:

| q=1 phase | Attested samples | Drawn count, min–max | Samples with all five counts exactly 84,348 | First exact-count frame |
| --- | ---: | ---: | ---: | ---: |
| Cold load | 1 | 1,319–1,319 | 0 | — |
| Stationary | 120 | 1,319–84,348 | 104 | 190 |
| Move | 120 | 84,348–84,348 | 120 | 1,230 |
| Reload | 118 | 1,319–84,348 | 101 | 2,620 |

The corresponding application-clock timestamps for those first full-count rows
were 3,455.2, 12,284.7 and 25,779.3 ms. These identify captured observations;
they are not an uninstrumented latency or throughput benchmark. Unlike the
120-frame q=0 profile, q=1 used 1,200 frames per stationary/movement/reload phase,
a 240-second application timeout and a 270-second automation timeout. Sampling
remained once every ten frames, with the same three-slot, 192-byte readback ring.

Both runs had zero mapping errors, invalid counts and dropped readbacks. The
scene-owned CPU/GPU ledger and allocation count reached zero before reload at
frame 273 for q=0 and frame 2,433 for q=1. Peak observed reservations were
234,688,474 CPU bytes in both runs and 66,013,760 / 67,068,512 GPU bytes for
q=0 / q=1, within the configured 512 MiB CPU and 256 MiB GPU limits. The atlas
remained subject to the 262,144-record / 64 MiB caps. These observations retain
the reservation-versus-RSS distinction above.

The detailed local evidence and first qualifying frame identities are preserved
in [`browser-icecream-v10-leaf-qualification.json`](../target/lod-roadmap/2026-09-07/browser-icecream-v10-leaf-qualification.json)
(SHA256 `18f60de9292f908b6bd1f066d6cc3c751bd0ac458a7f0c971a814900cbeb89e3`).
The analysis reran the standard validator on both raw captures, pinned their
hashes and the frozen q=1 protocol, and retained matched camera/main/GPU stamps. The raw captures
are [`browser-icecream-v10/evidence.jsonl`](../target/lod-roadmap/2026-09-07/browser-icecream-v10/evidence.jsonl)
and [`browser-icecream-q1-v10/evidence.jsonl`](../target/lod-roadmap/2026-09-07/browser-icecream-q1-v10/evidence.jsonl).

This qualifies the measured original-count lifecycle on that browser/adapter.
The requested and returned device limits included 16 storage buffers per shader
stage, 2,147,483,644 bytes per storage-buffer binding and a 4,294,967,292-byte
maximum buffer. These exceed the portable WebGPU minimum profile; the result
does not establish operation at minimum limits, headless browser support or
support across GPU vendors. Counts and authenticated runtime ownership also do
not replace an independent per-record source-identity audit, image-quality
comparison, native/browser parity check, pressure test or performance benchmark.
Both results retain `release_qualified: false`.

Lightweight protocol checks, with no browser/GPU activity:

```sh
cd tools
python3 -m unittest -v test_lod_browser_capture.py
node --test test_lod_adapter_observer.mjs
```

The 2026-09-08 GPS/GPUT v2 SH3 WebGPU build and wasm-bindgen 0.2.125 bindings
completed successfully. Eleven protocol tests pass, but the actual ChromeDriver
run was rejected before launch by the execution sandbox: local/private network
access to 127.0.0.1 is prohibited. No server or browser started, and no GPS browser
execution result is claimed. The [build identity](../target/lod-roadmap/2026-09-08/completion/browser-build-identity.json)
and [blocked-run record](../target/lod-roadmap/2026-09-08/completion/browser-run-blocked.json)
preserve this distinction. The historical NVIDIA quad runs above remain separate
evidence. Run the prepared tiny GPS profile in an environment allowing its local
HTTP/ChromeDriver endpoint before promoting browser support.

## Current GPS hardware execution (2026-09-08)

The current SH3 GPS/GPU-hierarchy fixture passes the strict checker in headed
Chrome 152.0.7977.64 on the renderer's observed non-fallback NVIDIA Blackwell
WebGPU adapter/device chain. It uses the configured 160×90 viewport, one sample
layer, 32 projected-record cap, 256 atlas records, and 64 MiB CPU / 32 MiB GPU
owned-memory limits. The [validation](../target/lod-roadmap/2026-09-08/production-closeout/browser-gps-clean/validation.json)
reports fourteen same-submission GPU observations, zero mapping/count errors, zero
dropped readbacks, and completed cold/stationary/move/unload/reload stages.
Unload reaches zero owned reservations before the new package loads.
Stationary and movement attest 32 selected/projected records; the reload proof
attests the complete four-record root representation, not restored leaf detail.

Wasm SHA-256 is `545fea3ce893b162c87f4038e8c60b8d6de0fad9217832a1405e5bc62e607090`;
[source/build identity](../target/lod-roadmap/2026-09-08/production-closeout/browser-gps-clean/build_identity.json)
also pins the entire Rust/WGSL/Cargo source set. This supersedes the earlier
sandbox execution block. It establishes actual hardware lifecycle execution,
not browser image parity, complete-frame speed, controller stability, or
large-scene qualification.

Execution exposed and fixed three harness issues: the fixture's atlas budget
now bounds upload staging instead of reserving a desktop-sized 64 MiB buffer;
an empty reload image is pending and cannot attest a hierarchy; and window
minimum dimensions no longer enlarge small requested viewports. A failed
package terminates the capture immediately. Earlier failed runs remain under
the same dated closeout directory; their evidence is not relabeled as passing.
