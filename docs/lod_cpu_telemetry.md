# Package CPU qualification scopes

Instrumented native and virtual-city captures install `LodPackageCpuTelemetry`
through their shared render-extraction hook. Ordinary applications and cadence
captures do not install the collector. No per-node clocks are read.

Each captured `timings.cpu_ms` map may contain:

- `lod_package_update_system`: elapsed time inside `update_lod_packages`, across
  every package updated by that system invocation.
- `lod_package_canonical_selection_miss`: actual uncached all-resident selection.
- `lod_package_destination_plan`: constrained destination lookup or planning.
- `lod_package_destination_compile`: physical destination compilation on a miss.
- `lod_package_discrete_wave_plan`: adjacent transaction planning/admission.
- `lod_package_discrete_resident_plan`: readiness resolution for admitted cohorts.
- `lod_package_publish_staged_cut`: main-world staged-cut publication function.

These are elapsed caller-thread function scopes. They can include synchronization
waits, overlap through nesting, and do not include all renderer or worker CPU
work. Do not sum them as total LoD CPU. `main_to_render_encode` remains the
separate existing main-to-render latency observation.

Extraction consumes exactly one completed update and assigns the same camera and
frame identity as the extracted capture request. The GPU copy accepts the CPU
snapshot only when both identities match; asynchronous readback carries that
record unchanged. Unsampled updates are drained, preventing stale reuse. Startup
before the collector is installed omits CPU fields instead of emitting zeros.

The same submission's evidence also includes `package_cpu_work`: its full capture
stamp, calls per scope, destination cache hits and compilations, and nodes visited
by canonical selection misses. These counters describe the entire package update,
including other packages/views processed by that invocation; the stamp associates
them with the extraction, not with exclusive work for that camera. Cached
stationary updates must not be used to claim constant-time uncached traversal.

The collector stores fixed-size accumulators and one completed sample. Its CPU
unit tests cover disabled recording, update isolation, counter saturation, and
frame/camera stamping without stale reuse. This instrumentation prepares actual
measurements; its presence alone does not qualify performance.

## Observed virtual-city work, 2026-09-07

The final v11 10M/100M native pair provides CPU-work observations with exact
capture/submission stamps. Its raw artifacts are
`target/lod-roadmap/virtual-city/{10m,100m}-capture-v11-exact-visible`;
[the execution report](lod_virtual_city_results.md) records identities and limits.

| Observed scope | 10M | 100M |
|---|---:|---:|
| Stationary updates with CPU-work evidence | 150 / 150 | 150 / 150 |
| Stationary canonical visits / destination compilations / hits | 0 / 0 / 0 | 0 / 0 / 0 |
| Stationary package-update median / p95, ms | 0.011554 / 0.015612 | 0.011093 / 0.014960 |
| Movement selection samples | 70 / 150 captures | 66 / 150 captures |
| Movement visited nodes per selection, median / p95 / max | 350 / 373 / 375 | 771 / 811 / 813 |
| Movement destination misses / hits | 70 / 19 | 66 / 24 |
| Movement destination miss fraction among sampled lookups | 78.65% | 73.33% |
| Movement compilation median / p95, ms | 0.139178 / 0.202154 | 0.409671 / 0.613232 |

Stationary updates bypass planning entirely; zero lookups make a cache-hit rate
undefined. During movement, 46.67%/44.00% of captured updates compile a
destination. The 100M canonical selector visits at most 813 of 48,829 hierarchy
nodes in these samples, while destination compilation is materially slower
than at 10M. This does not bound the complexity of other update operations.

Fractions use captured lookup counts, not whole-run totals. An update can
perform both a hit and a miss, and sampling every eight frames can alias
cadence. Conditional selection distributions omit unexecuted scopes; missing
evidence is unobserved, never zero. Destination compilation can reuse a cached
canonical selection: the 100M reload records one such compilation with zero
canonical visits. Cached rapid-return planning also coexists with zero drawn
instances, so these counters alone do not qualify visual recovery or quality.

## Additional scope contract after v11

Four optional function scopes are installed at the boundaries below. They are
absent from the frozen v11 evidence and remain unavailable in historical reports.
The subsequent v16 diagnostic supplies observations for these hooks.

| Field | Exact function boundary | Includes / excludes |
|---|---|---|
| `lod_package_runtime_poll_pages` | `LodStreamingRuntime::poll_pages`, entry through return | Transport polls, preprocessing submission, and bounded preprocessor advancement; excludes native worker execution outside the call |
| `lod_package_runtime_commit_preprocessed_pages` | `LodStreamingRuntime::commit_preprocessed_pages`, entry through return | Ready-page sorting/admission, cache eviction/insertion, decoded ownership and pin changes; excludes decode worker time and later atlas packing |
| `lod_package_target_candidates` | `LodStreamingRuntime::package_target_candidates`, entry through return | Readiness, fresh camera/policy quality, fallback provenance and physical candidate construction, including nested destination lookup/compilation |
| `lod_package_advance_staged_cut` | `advance_package_staged_cut`, entry through return | Slot admission, padded canonical CPU payload construction, enqueue, debug-target discovery and range validation; excludes separate debug-record preparation and RenderWorld GPU upload |

All four are nested in `lod_package_update_system`. Runtime polling and cache
commit are disjoint sibling calls. Target-candidate resolution is disjoint from
those calls and the wave/resident planners, but may contain `destination_plan`,
which contains canonical-selection misses and destination compilation. Other
`destination_plan` calls occur outside target-candidate resolution. Aggregated
per-frame durations therefore do not allow blindly subtracting or summing every
field. Atlas advance is a separate main-world staging call; publication remains
its small ownership-swap scope. Each field aggregates all calls in the sampled
package-system update, including additional views/packages.

The v11 source layout permits a diagnostic remainder per frame: package update
minus destination planning, wave planning, resident planning and publication.
Their canonical/compile child scopes are not subtracted again. Movement remainder
median/p95 is 0.283/1.280 ms at 10M and 0.350/1.890 ms at 100M (150 samples each).
This is unattributed caller-thread elapsed time, including scheduling/waits, not
an isolated function measurement. The additional scopes can distinguish polling,
cache commits, candidate construction and slot packing; they do not retroactively
identify that remainder or establish which optimization will improve its p95.


## Observed 100M update tail, v16

One subsequent 100M run uses the v11 configuration with only its output path
changed and the same prepared payload. The frozen executable hash is
`86ab41dc9fb3fcdcea6a082efe9c3fb13f4a533248a0d5f1262abfda0817f274`;
raw evidence is `target/lod-roadmap/virtual-city/100m-capture-v16-exact-visible`.
Lifecycle/schema checks pass, and 743 of 744 captures have exact-stamp CPU-work
evidence. The first cold sample remains unobserved. This is a scope diagnostic,
not a new scaling pair or a measurement of the later HTTP checksum change.

| Inclusive function | Movement samples | Movement median / p95, ms | Eviction samples | Eviction median / p95, ms |
|---|---:|---:|---:|---:|
| Package update | 150 | 0.622777 / 2.040006 | 150 | 0.566622 / 3.054144 |
| Runtime polling | 88 | 0.008845 / 0.850139 | 78 | 0.015645 / 1.623471 |
| Ready-page commit | 88 | 0.006324 / 0.020784 | 78 | 0.011013 / 0.019386 |
| Target candidates, including destination planning | 40 | 0.653282 / 1.014839 | 41 | 0.477995 / 1.011632 |
| Atlas staging advance | 26 | 0.277320 / 0.606229 | 38 | 0.201223 / 0.857059 |

Polling is the largest observed outer function in five of the eight slowest
movement updates and all eight slowest eviction updates. Those eviction frames
spend 1.618–1.643 ms inside polling. Candidate timings include planning and must
not be added to destination timings. The new analyzer bounds their partially
overlapping union per frame; movement's time outside observed scopes has p95
lower/upper bounds of 0.323/0.538 ms. This unexplained interval includes missing
scopes, other work and scheduling, rather than identifying another function.

Code inspection finds a caller-thread payload-sized FNV scan in HTTP response
handling: `handle_response` constructs `PagePayload` after fetch. The native
worker-checksum change moves that scan into the existing bounded worker while
preserving HTTP checks and downstream verification. The v16 timings precede that
change and cannot establish its benefit.

Stationary remains 150 stable draws of 10,115 with explicit zero planning-work
counters and update p95 0.014617 ms. Movement has 17/150 zero draws; rapid return
has 25/26, ending on 256 proxies. Full lifecycle success and complete source
interval coverage therefore do not qualify useful-detail recovery. The detailed
scope report, full identities and checker outputs are in
`target/lod-roadmap/2026-09-07/virtual-city-100m-v16-scope-analysis.md` and its
neighboring `*-v16-*.json` artifacts. Historical v11 measurements are unchanged.


## Native worker checksum after-check, v17

The same 100M configuration was run once after moving FNV checksum generation
into the existing bounded native HTTP worker. Only the output setting changes.
The frozen executable hash is
`75271f13417d62083534aa3a30250fe101b2f8e0c089b39e9f9ab4756c180291`;
raw artifacts are `target/lod-roadmap/virtual-city/100m-capture-v17-exact-visible`.
Identity, lifecycle, schema and exact-stamp checks pass: 743 captures, 742 CPU-work
observations, with the first cold sample unobserved. The source also includes
continuous-presentation error/retention changes; this Discrete profile is not a
one-function whole-binary A/B.

| Inclusive scope | Movement n, v16 / v17 | Movement p95, ms | Eviction n, v16 / v17 | Eviction p95, ms |
|---|---:|---:|---:|---:|
| Runtime polling | 88 / 82 | 0.850139 / 0.020956 | 78 / 79 | 1.623471 / 0.020159 |
| Full package update | 150 / 150 | 2.040006 / 1.972186 | 150 / 150 | 3.054144 / 2.015716 |
| Target candidates, including planning | 40 / 46 | 1.014839 / 0.986207 | 41 / 51 | 1.011632 / 1.000271 |
| Staging advance | 26 / 26 | 0.606229 / 0.867265 | 38 / 36 | 0.857059 / 0.597862 |

The large polling tail is absent in these after-run samples: observed movement
and eviction maxima fall from 1.625/1.643 to 0.087/0.078 ms. Native-client tests
verify that the worker supplies the checksum without copying the body, while
actual downstream preprocessing still rejects a corrupt hint. HTTP metadata
and authenticated descriptor validation remain intact. The generic/browser
fallback is unchanged. This supports retaining the removal of the caller-thread
byte scan; the inclusive poll timer does not measure checksum work exclusively.

**Changed-view package-update p95 still exceeds the 1 ms gate.** Movement
frame-wall p95 grows 5.565242→5.933056 ms (+6.61%) while selected/drawn work changes;
eviction frame-wall p95 changes 6.432067→6.382677 ms (−0.77%). No overall frame
speedup or causal quality improvement is established. Destination planning and
candidate timings overlap and remain substantial. Missing observations are not
zeros, and every-eighth-frame sampling can alias stage cadence.

Both stationary windows retain 150 stable draws of 10,115 from 153,861 selected
records, with explicit zero planning counters. Movement nonzero samples change
133/150→148/150; rapid return remains 25/26→22/23 zero draws, ending on 256 proxies
with pending demand. Both lifecycles unload to zero scene ledger and reload,
with identical sampled CPU/GPU ledger peaks and no readback errors or ring drops.
These are bounded synthetic execution observations, not useful-detail recovery,
browser performance or release qualification.

Full populations, frame/GPU/work changes, provenance and limits are recorded in
`target/lod-roadmap/2026-09-07/virtual-city-100m-v16-v17-checksum-analysis.md` and
`virtual-city-100m-v16-v17-cpu-comparison.json`. The CPU-only analyzer suite has
24 passing synthetic tests, including lifecycle-only unload with unobserved CPU
scopes. Historical v11 and v16 evidence remains unchanged.

Applications using the `testing` feature can consume the latest update with
`LodPackageCpuTelemetry::take_completed()`. Its public `PackageCpuSample`
contains elapsed scopes and work counters; a missing sample remains unobserved,
and consumers must supply their own frame identity when associating it with
rendered work.
