# Package preparation and memory admission

Opening a validated `.gsplatlod` asset now stages immutable metadata before
publishing a physical atlas handle. Native builds perform this work on Bevy's
asynchronous compute pool. Browser builds poll the same compiler cooperatively;
each application frame supplies a shared record allowance across packages and
first-use debug indexing.

The compiler covers hierarchy indices, transport URI/range checks, cache page
identities, shared-page node ranges, page size preflight, and complete bootstrap
coverage. Final main-world instantiation consumes those prepared objects without
revalidating every manifest node or page. Dense canonical identifiers avoid
redundant maps, and immutable page indices and URI strings share storage.

| Configuration | Default | Contract |
| --- | ---: | --- |
| `GaussianLodPackageConfig::max_preparation_jobs` | 2 | Running, completed-but-unpublished, and canceled-but-unacknowledged jobs count toward this limit. |
| `max_preparation_bytes` | 512 MiB | Aggregate conservative input/index admission charge for those jobs. |
| `preparation_records_per_frame` | 4096 | Aggregate cooperative node/page work allowance; a record can require bounded lookup/insertion work. |
| `LodMemoryLimits::max_cpu_bytes` | 4 GiB | Shared reservation ceiling for owned CPU capacities. |
| `LodMemoryLimits::max_gpu_bytes` | 4 GiB | Shared reservation ceiling for owned GPU capacities. |

Native tasks check cancellation between 4096-record slices. Replacing a source,
manifest, or structural setting cancels the old job; its admission charge remains
until the worker acknowledges cancellation. An obsolete result cannot publish an
atlas. Invalid immutable preparation results remain cached until relevant inputs
change, while temporary global memory pressure retries the already prepared
result without recompiling it.

Debug-Off startup creates no debug manifest index. First enable compiles it with
the same cooperative allowance. Disabling during compilation drops the partial
index; after successful compilation, toggling Off/On reuses the shared index.
Debug readiness remains false while that index is incomplete.

Before compilation, the package reserves metadata capacity from the shared
`LodMemoryLedger`. Before creating transport/runtime/atlas owners, it atomically
reserves atlas GPU capacity (including its indirect draw buffer), decoded-page
capacity, canonical CPU recovery capacity, and bounded transport/preprocess
capacity. Existing local limits still apply. Lease clones preserve one allocation
identity, so main/render sharing does not charge the same reservation twice.
Replacing a physical GPU allocation requires a separate reservation while the old
allocation remains charged through queue completion.

CPU recovery reservations follow extracted and retried upload payloads after
package teardown. Detached native file, HTTP and preprocessing workers retain CPU
reservation clones until completion, and browser Fetch promises retain them until
settlement after cancellation. A package disappearing from the main world does
not immediately make this still-owned capacity available to its replacement.

These numbers describe admission reservations, not observed RSS or GPU memory.
The metadata estimate is checked arithmetic over validated node/page counts and
URI storage; it is not an allocator measurement. Record allowances bound compiler
visits, not elapsed milliseconds or allocator latency. Manifest parsing and
authoritative format validation still belong to asset loading. This change does
not establish a browser frame-time or large-scene performance result; those
require the stamped runtime captures described in [the capture guide](lod_capture.md).

Focused CPU tests cover one-record progress/equivalence, cancellation accounting,
signature changes before publication, lazy debug indexing, global admission
retries, and reservation lifetime through worker cancellation and upload retries.
Run these alongside the package, preprocessing, transport and HTTP regression
suites under the relevant native/Wasm feature configurations.
