# Virtual-city execution screens — 2026-09-07

The final v11 10M and 100M synthetic packages completed native GPU streaming,
movement, eviction, zero-ledger unload, and reload. Both stationary windows
contain **150 stable, attested samples drawing 10,115 instances**. The CPU
capture checker and same-stamp CPU-work checks pass. Candidate work and
source-page sets differ, and motion
still has visible response gaps; these single instrumented screens do not
qualify representative quality or establish general performance scaling.

## Final v11: measured planning work

The final pair is under
`target/lod-roadmap/virtual-city/{10m,100m}-capture-v11-exact-visible`.
Its renderer executable SHA-256 is
`327dbb2c4e7d096062f7bcc98816395db9d9e9331be49b9f5ac56354aff7e20d`.
Both runs use the same hardware, route, quality/capacity settings, and prepared
payload-v2 sources described for v9 below. They ran serially without Cargo
concurrency. Settings hashes, distinct `renderer:settings:manifest` run IDs,
and generator identities were independently recomputed and matched; see
`target/lod-roadmap/2026-09-07/virtual-city-v11-identity-verification.json`.

Each stationary window has 150 unchanged, nonzero, attested draws of 10,115
instances, with zero queued/in-flight work in its interior lifecycle samples.
Selected counts remain 119,571 versus 153,861. The original-page sets remain
28 versus 36 pages, with the same sixteen common source indices listed below.
These are matched drawn counts, not identical candidate work or visible IDs.

Every stationary capture has a valid `package_cpu_work` stamp matching its
capture and submission evidence. All 150 observations per run explicitly
record **zero canonical visits, zero destination compilations, and zero
destination-cache hits**. Only `lod_package_update_system` executes. This is
settled-plan bypass; with no destination lookup, a cache-hit rate is undefined.
Its median/p95 is 0.011554/0.015612 ms at 10M and 0.011093/0.014960 ms at 100M.

Changed cameras do execute additional work. The visited-node distributions
below include only samples in which canonical selection actually executes.
Each ordinary movement/eviction phase has 150 captured CPU-work observations.

| Sampled work | 10M movement | 100M movement | 10M eviction | 100M eviction |
|---|---:|---:|---:|---:|
| Canonical selection samples | 70 | 66 | 63 | 57 |
| Visited nodes, median / p95 / maximum | 350 / 373 / 375 | 771 / 811 / 813 | 339 / 371 / 375 | 749 / 811 / 813 |
| Destination misses / hits | 70 / 19 | 66 / 24 | 63 / 30 | 57 / 18 |
| Misses / observed destination lookups | 78.65% | 73.33% | 67.74% | 76.00% |
| Captured updates containing a compilation | 46.67% | 44.00% | 42.00% | 38.00% |
| Canonical selection median / p95, ms | 0.099426 / 0.117594 | 0.175028 / 0.250470 | 0.101453 / 0.119362 | 0.183906 / 0.258207 |
| Destination compilation median / p95, ms | 0.139178 / 0.202154 | 0.409671 / 0.613232 | 0.140858 / 0.179245 | 0.406898 / 0.631665 |
| Full package-update median / p95, ms | 0.591947 / 1.801497 | 0.519046 / 2.851226 | 0.397676 / 2.436709 | 0.486585 / 2.899762 |

The 100M hierarchy contains 48,829 nodes; observed selection misses visit at
most 813. These samples show increased changed-view cost without a full
hierarchy traversal in canonical selection. They do not establish asymptotic
bounds for destination compilation or every other package-update operation.
In movement, the median visited count grows 2.20× and compilation p95 grows
3.03×. Cached stationary cost cannot stand in for this work.

Miss fractions use `compilations / (compilations + cache_hits)`. One update can
contain multiple lookups, including both a miss and a hit. The separate update
fraction counts frames with at least one compilation. Both denominators cover
captured observations only; every-eighth-frame sampling can alias pipeline
cadence. Absent evidence stays unobserved. These system-wide counters are
associated with the capture camera/frame, not attributed exclusively to it.

Rapid return records eight and sixteen destination hits with no compilation
or canonical visits in the ten and twenty-four sampled updates. Yet nine and
twenty-three draws are zero, respectively. The final nonzero samples draw
2,390 and 256 instances, and both phases finish with pending page demand.
Cached planning therefore does not imply immediate visible-detail recovery.
Cold-start captures do not observe a canonical-selection scope; this sampling
must not be reported as startup having no selection work.
Reload's 100M samples include one destination compilation with zero canonical
visits, demonstrating that destination and canonical-cache misses are distinct.

### v11 stationary timings and remaining transients

| Scope, ms, all 150 stationary samples | 10M median / p95 | 100M median / p95 | p95 change |
|---|---:|---:|---:|
| Instrumented frame wall | 2.151875 / 2.926844 | 2.043003 / 2.955962 | +0.99% |
| Main-to-render encode elapsed | 1.826355 / 2.586907 | 1.683626 / 2.597021 | +0.39% |
| GPU compaction | 0.003824 / 0.025120 | 0.003808 / 0.025312 | +0.76% |
| GPU sort | 0.003872 / 0.025184 | 0.003808 / 0.019712 | −21.73% |
| GPU raster and postprocess | 0.139872 / 0.305408 | 0.131264 / 0.343328 | +12.42% |
| GPU view total | 0.150448 / 0.355328 | 0.138608 / 0.394144 | +10.92% |

Stationary spans are 2.47/2.36 seconds. Frame-wall p95 stays within the 10%
screen threshold, while GPU-view p95 exceeds it. Neither the favorable v9
GPU-view result nor this single instrumented pair establishes locked scaling.
CPU scopes overlap; GPU work includes instrumentation. No uninstrumented
control or repeated statistical qualification was performed.

Movement has 150/150 versus 146/150 nonzero draws and 75/150 versus 135/150
busy lifecycle samples. Eviction has 150/150 nonzero draws in both runs but
93/150 versus 149/150 busy samples. At the 100M eviction endpoint, two pages
remain queued and two in flight. These are changing workloads, not settled
windows. Reload settles to the same original stationary counts with 75/57
stable samples. Rapid-return gates still stop at the first nonzero draw;
endpoint recovery to sustained original detail remains unqualified.

The first complete-package nonzero readbacks draw 256 proxies. Their frame
starts are 1.817539/1.628594 seconds after the renderer-initialized proxy, with
readback observed by 1.822784/1.633743 seconds. Neighboring lifecycle cuts are
complete but contain zero original leaves. These omit source prehash and app
setup and are not first useful source-detail or total startup measurements.

Sampled CPU ledger peaks are 683.71/755.47 MiB and GPU peaks are 273.07 MiB in
both runs. Process RSS peaks are 1,173.29/1,125.55 MiB. Both unload to zero scene
ledger and reload, with zero ring drops, mapping errors, or unattested draws.
They complete 732/732 and 766/766 submitted readbacks, observe 129/122 distinct
original leaf pages and 687/1,117 slot replacements, and finish their lifecycle
in 19.48/21.22 seconds excluding prehash/app setup. Scope limits for memory,
proxy quality, synthetic transport, and source-range reconstruction below
apply equally to v11.

## Prior v9 evidence and identities

The pair ran serially without Cargo overlap on NVIDIA RTX PRO 6000
Blackwell/Vulkan with NVIDIA 610.43.02. Both use 960×540, the same route, Discrete
presentation, quality 1.0, a 1,048,576 active-record cap, 256 resident pages,
4,096 source records per leaf, and two bounded loopback handlers. Ordinary
phases last 1,200 frames, sampled every eight. Admission ceilings are 1 GiB
CPU and 512 MiB GPU. No capture images were requested.

Raw v9 artifacts are under
`target/lod-roadmap/virtual-city/{10m,100m}-capture-v9-exact-visible`.
Each directory includes settings, preparation, capture JSONL, same-submission
evidence, lifecycle JSONL, and final status. The renderer executable SHA-256 is
`05504e759bdb4f8fe54741cf2b915129b6fdea9bcd6ae7b3a2a23cb955a23805`.
The authenticated manifest identities are
`bb5cdef95b882a9c1f172360acde1fc371a8d1fa2da62d30e7f9ebb3c3c7f571`
(10M) and
`0e792d46c2288f465b4ac85c88d2035ff240a437a85b26bacb82a1bfb01d6ca4`
(100M). These results identify the captured shader/build pair; subsequent builds
need their own evidence.

The analyzer independently recomputed the settings hashes, payload-v2 generator
identities, and `renderer:settings:manifest` run IDs. All match. The new payload
identity hashes source inputs/SH degree and the generation modules without
hashing lifecycle orchestration. The two run IDs are distinct. Full identities
and input-file hashes are preserved in
`target/lod-roadmap/2026-09-07/virtual-city-v9-identity-verification.json` and the
per-run analysis JSON files. Historical v4–v6 IDs reused stamps across settings;
those artifacts must remain separated by full identity and file hashes.

## Full stationary window in v9

Each run's stationary camera, selected/candidate/drawn counts, and both candidate
fingerprints are unchanged. All interior lifecycle samples have zero queued and
in-flight work. The sampled spans are 2.51 and 2.31 seconds. Drawn counts agree
with same-submission indirect readback and draw-command attestation, with zero
overflow. The p95 uses nearest rank, `ceil(0.95 × n)`, over all 150 samples.

| Measured scope, milliseconds | 10M median / p95 | 100M median / p95 | p95 change |
|---|---:|---:|---:|
| Instrumented frame wall | 2.165417 / 3.144916 | 1.987850 / 2.778476 | −11.65% |
| Package update function | 0.011355 / 0.014780 | 0.011317 / 0.014513 | −1.81% |
| Main-to-render encode elapsed | 1.812225 / 2.806526 | 1.585803 / 2.259029 | −19.51% |
| GPU compaction scope | 0.003920 / 0.019872 | 0.003744 / 0.012672 | −36.23% |
| GPU sort scope | 0.003936 / 0.021440 | 0.003744 / 0.012640 | −41.04% |
| GPU raster and postprocess | 0.148864 / 0.306496 | 0.129888 / 0.198304 | −35.30% |
| GPU view total | 0.157488 / 0.355744 | 0.137312 / 0.243520 | −31.55% |

This pair shows no stationary p95 growth above the 10% screen threshold, but the
reductions do not establish a source-size speedup. The earlier v6 screen varied
in the opposite direction. These are single instrumented runs without repeated
or uninstrumented controls, and candidate counts increase from 119,571 to
153,861 (+28.68%). Cached stationary scopes also do not measure fresh selection
or compaction cost during motion. CPU scopes overlap and must not be added;
main-to-render is elapsed latency, not isolated CPU execution time.

Reload reaches the same stationary counts, with 75 and 56 stable captures.
Its frame-wall p95 is 2.895840 versus 3.267574 ms (+12.84%), emphasizing that one
favorable stationary window is insufficient for a locked performance claim.
The checker compares only the last ten stationary/reload draws and confirms
count matching; the full-window distributions above come from the separate
analyzer and must not be confused with that checker's ten-sample medians.

## v9 coverage, motion, and return

Stationary cuts contain 28 original pages at 10M and 36 at 100M. Sixteen
source-page indices are common:
`[0, 1, 2, 128, 129, 256, 257, 383, 511, 895, 1151, 2046, 2047, 2048, 2303, 2304]`.
Equal drawn counts therefore do not imply identical selected source-page sets.
The capture does not read back per-Gaussian visible identities or image quality.
Diagnostic reconstruction of the generator's source intervals finds complete,
nonoverlapping coverage in every recorded nonempty lifecycle cut. This is not
an independent authenticated-manifest decoder or a same-frame source-range
certificate; publication relies on the production complete-cut contract.

| v9 phase observation | 10M | 100M |
|---|---:|---:|
| Movement nonzero / captured draws | 150 / 150 | 145 / 150 |
| Movement busy lifecycle samples | 74 / 150 | 133 / 150 |
| Rapid-return zero / captured draws | 9 / 10 | 23 / 24 |
| Rapid-return sampled span | 0.217 s | 0.607 s |
| First nonzero rapid-return draw | 2,390 | 256 |
| Eviction nonzero / captured draws | 150 / 150 | 150 / 150 |
| Eviction busy lifecycle samples | 69 / 150 | 149 / 150 |

Movement and eviction remain changing workloads, not settled comparison
windows. The 100M eviction phase ends with four queued and two in-flight pages;
rapid return ends with pending demand in both runs. Its phase gate accepts the
first nonzero sample, not settled original detail. A future quality gate should
hold return/movement endpoints until a bounded stable detail window or timeout,
and distinguish complete proxy coverage from visible original detail.

The uncached CPU cost also remains source-sensitive. During movement,
canonical-selection median/p95 is 0.096266/0.121617 ms at 10M (65 captured scope
samples) and 0.174327/0.247025 ms at 100M (55). Destination-compilation
median/p95 is 0.130476/0.159037 versus 0.404053/0.614432 ms with the same sample
counts. Stationary caching is cheap, while planning under motion costs more.
Scope counts represent captured executions only; an absent scope is not a
zero-duration execution.

First nonzero cold draws contain 256 synthetic proxies at frames 704 and 600.
Frame starts occur 1.899683 and 1.607714 seconds after the renderer-initialized
proxy; readback is observed by 1.907928 and 1.615117 seconds. Neighboring
lifecycle cuts cover the source but contain zero original leaves. The origin
is the first capture Core3d execution with a RenderDevice, excluding earlier
source preparation/loading and app setup. These are not total application
startup measurements or first useful source-detail images. The evidence's
`complete_package_frontier` flag derives from package-required candidates.

## v9 memory and lifecycle

| v9 observation | 10M | 100M |
|---|---:|---:|
| Peak sampled admitted CPU capacity | 683.71 MiB | 755.47 MiB |
| Peak sampled admitted GPU capacity | 273.07 MiB | 273.07 MiB |
| Peak sampled process RSS | 1,131.68 MiB | 1,141.50 MiB |
| Distinct original leaf pages observed in render ranges | 129 | 121 |
| Observed slot replacements | 704 | 1,115 |
| Completed / submitted readbacks | 730 / 730 | 767 / 767 |
| Lifecycle runtime, excluding prehash/app setup | 19.56 s | 21.15 s |

Both runs observe zero scene ledger during unload, reload successfully, and
report zero ring drops, mapping errors, or unattested draws. Ledger capacities
and RSS have different scopes: fixture server/manifest ownership and backend
allocations are not all charged to the scene ledger. The CPU admission ceiling
is not a process-RSS ceiling. Sampled peaks can miss between-sample peaks, and
individual category maxima must not be summed. These runs do not qualify
constrained physical VRAM behavior or total device memory accounting.

The generator has authenticated payloads and distinct source extents; CPU
prehash is a separate O(source records) step. On-demand payload generation is
included in runtime. Synthetic loopback does not qualify external file/CDN
deployment, and synthetic proxy centers/colors do not qualify unique-asset
quality. Observing a page in render ranges does not establish that all of its
records contributed visible pixels.

## Historical profiles

| Profile | Quality / active cap / resident pages | Stationary selected, 10M / 100M | Stationary drawn, 10M / 100M | Interpretation |
|---|---|---:|---:|---|
| v4 | 0.9 / 65,536 / 64 | 55,345 / 39,969 | 10,115 / 7,725 | Lifecycle passes; visible-work comparison fails |
| v5 | 0.9 / 65,536 / 256 | 63,523 / 65,077 | 10,115 / 7,725 | More slots preserve the mismatch; 64-slot pressure alone does not explain it |
| v6 exact-visible | 1.0 / 1,048,576 / 256 | 119,571 / 153,861 | 10,115 / 10,115 | Draw-count screen passes; candidate work remains unequal |
| v9 shader pair | 1.0 / 1,048,576 / 256 | 119,571 / 153,861 | 10,115 / 10,115 | Prior captured shader evidence; planning counters unavailable |
| v11 final pair | 1.0 / 1,048,576 / 256 | 119,571 / 153,861 | 10,115 / 10,115 | Lifecycle and count checks pass; measured planning work and timing limits above |

The generator rebuilds its balanced tree over total source-page count. Increasing
source size changes near-region ancestors, bounds, and proxy counts. The same
continuous quality/active cap can therefore choose different local detail.
v6/v9/v11 use the Original endpoint to obtain matching drawn counts; none
qualifies useful dynamic LoD reduction or fitted representative quality.

Historical v4–v6 renderer SHA-256 is
`a19710993ff351fc75b41f63aa8e44611f9048fa268f62ace02ecfefe81f16f1`.
v6 stationary frame-wall p95 was 2.678023/2.940904 ms (+9.82%) and GPU-view p95
was 0.197632/0.210720 ms (+6.62%), each with 150 stable samples. These remain
historical observations and are not substituted for the final v11 pair. Full
v4/v5/v6 analyses and the prior v6 results text remain in the dated artifact
directory. The interrupted 180-frame v3 run supplied only about 13 stable
stationary samples over 0.2 seconds and is not used for these comparisons.

## Reproduce the CPU analysis

The read-only standard-library analyzers and all JSON reports are in
`target/lod-roadmap/2026-09-07/`. These local artifacts are not bundled with
the source tree. Input SHA-256, complete identities, scope sample counts,
quantiles, source intervals, and sampled ledger peaks are retained.

```sh
python3 tools/check_lod_virtual_city.py target/lod-roadmap/virtual-city/10m-capture-v11-exact-visible target/lod-roadmap/virtual-city/100m-capture-v11-exact-visible
python3 target/lod-roadmap/2026-09-07/analyze_virtual_city_capture.py target/lod-roadmap/virtual-city/10m-capture-v11-exact-visible --output target/lod-roadmap/2026-09-07/virtual-city-10m-v11-analysis.json
python3 target/lod-roadmap/2026-09-07/analyze_virtual_city_capture.py target/lod-roadmap/virtual-city/100m-capture-v11-exact-visible --output target/lod-roadmap/2026-09-07/virtual-city-100m-v11-analysis.json
python3 target/lod-roadmap/2026-09-07/compare_virtual_city_work.py target/lod-roadmap/2026-09-07/virtual-city-10m-v11-analysis.json target/lod-roadmap/2026-09-07/virtual-city-100m-v11-analysis.json --output target/lod-roadmap/2026-09-07/virtual-city-v11-work-comparison.json
```

`virtual-city-v11-checker.json` records the checker result;
`virtual-city-v11-identity-verification.json` records the independent identity
recipe checks. No GPU work is performed by these analysis commands.

## 2026-09-08 scheduler closeout

Two bounded 100M runs reused the v9 prepared manifest and v17 route/configuration
without repeating the large source prehash. The first validates cached cohort
readiness and visible-work priority; the second also removes the async executor
and atomic preparation budget from synchronous page-footprint accumulation.
Both complete cold load, movement, eviction, return, zero-ledger unload and
reload. The capture now requires ten consecutive return observations matching
the stationary camera and selected/candidate/compacted/drawn counts.

| Observation | v17 | Cached/visible scheduler | Direct footprint closeout |
| --- | ---: | ---: | ---: |
| Movement package-update p95 ms | 1.972 | 1.394 | 1.427 |
| Eviction package-update p95 ms | 2.016 | 1.387 | 1.424 |
| Initial zero-draw return observations | 22 | 2 | 2 |
| Sustained returned draw | Unobserved | 10,115 | 10,115 |

The final return first draws 2,390 instances at path frame 16, then matches the
stationary 10,115-instance work for the ten observations at path frames 272–344.
The reference has 153,861 selected/candidate records; those are not its rendered
count. Movement has no observed zero draws; eviction still has five. Full-image
or visible-ID equivalence is not measured. The stronger endpoint and changed
visible work prevent treating these timings as an accepted equal-work speedup.
**The 1 ms movement/eviction update gate still fails.** Removing the synchronous
async wrapper does not establish an additional p95 gain.

The final diagnostic took 23.03 seconds. Process RSS/HWM peaked at
1,223,987,200 bytes; half-second adapter-wide observations peaked at 1,769 MiB
against a 1,217 MiB desktop baseline. GPU values include other clients and can
miss between-sample peaks. These are separate from the 1 GiB CPU / 512 MiB GPU
owned-reservation limits and do not qualify physical 8 GiB hardware.

[First comparison](../target/lod-roadmap/2026-09-08/production-closeout/100m-profile/comparison.json),
[final comparison](../target/lod-roadmap/2026-09-08/production-closeout/100m-footprint-profile/comparison.json),
and [final process/memory record](../target/lod-roadmap/2026-09-08/production-closeout/100m-footprint-profile/run.json)
retain exact binary, manifest, configuration and source-generator identities.
These are CPU-selected quad paging diagnostics; the tiny native/browser GPU
traversal fixtures provide separate correctness evidence. Large-scene GPU
traversal and stable control under deployment workloads remain qualification
requirements. The remaining CPU work is to make destination planning and staged
materialization fit the per-frame allowance; another unchanged run is not that fix.
