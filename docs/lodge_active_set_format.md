# LODGE active-set sidecar format v1

This document defines the `.gslodge` bytes accepted by this crate. The words
**MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. Offsets are
zero-based, fixed-width integers are little-endian, and a byte range is the
pair `(start, length)`, not an inclusive end.

`.gslodge` is an authenticated companion to an ordinary `.gsplatlod` package.
It adds independently authored levels, a dense stable-record catalog, camera
clusters, and compressed active-set memberships without changing the existing
page containers. It is this crate's adapter for LODGE-style output. It is
**not** a file format specified by the
[LODGE paper](https://arxiv.org/abs/2505.23158) or by an upstream reference
implementation, and an upstream artifact is not a drop-in `.gslodge` file.

| Layer | Magic or discriminator | Version |
| --- | --- | ---: |
| sidecar container | `BGSLODGE` | 1 |
| semantic manifest | `BGSLOG1\0` | 1 |
| membership schema | `DeltaUleb128StableIdsV1` | 1 |
| decoded Gaussian page schema | inherited from `.gsplatlod` | 2 |

## Sidecar container

Every v1 sidecar starts with this exact 120-byte header. SHA-256 fields in the
header are raw 32-byte digests, not text or hexadecimal.

| Offset | Width | Field |
| ---: | ---: | --- |
| 0 | 8 | magic `BGSLODGE` |
| 8 | 2 | container version, `1` |
| 10 | 2 | semantic manifest version, `1` |
| 12 | 1 | payload encoding: Flexbuffers `1`, JSON `2` |
| 13 | 1 | flags, zero |
| 14 | 2 | reserved, zero |
| 16 | 8 | serialized payload length |
| 24 | 32 | SHA-256 of exactly the serialized payload bytes |
| 56 | 32 | SHA-256 of exactly the encoded base `.gsplatlod` manifest |
| 88 | 4 | level count |
| 92 | 4 | camera-cluster count |
| 96 | 4 | record-run count |
| 100 | 4 | extra-page count |
| 104 | 4 | page-authentication count |
| 108 | 4 | flattened-neighbor count |
| 112 | 4 | membership-entry count |
| 116 | 4 | reserved, zero |
| 120 | variable | serialized semantic payload |

The file length MUST equal `120 + payload_length`; trailing bytes are invalid.
Readers authenticate the payload before deserializing it. Every envelope count
MUST equal both its semantic-header count, where one exists, and the decoded
collection length. The membership-entry count MUST equal the cluster count.
The page-authentication count MUST equal `base_page_count + extra_page_count`.
The base digest at offset 56 MUST equal `base_manifest.sha256` in the payload.

## Serde payload representation

The logical payload is the Serde data model of `GaussianLodgeManifest`. Both
encodings use the exact Rust field and variant names shown below.

| Rust shape | JSON and Flexbuffers representation |
| --- | --- |
| struct | map keyed by field name |
| `Vec<T>` or fixed array | sequence |
| `(u64, u64)` byte range | two-element sequence `[start, length]` |
| transparent ID newtype | its unsigned integer value |
| `[u8; 32]` SHA-256 | sequence of 32 byte-valued integers, not a hex string |
| unit enum variant | string, for example `"Original"` |
| struct enum variant | externally tagged map, for example `{"DepthAware3dV1": {...}}` |
| `Option<T>` | the value or `null` |

JSON encoding `2` is the compact output of `serde_json::to_vec` when written by
the crate. JSON has no additional canonical-byte rule: changing whitespace or
map order changes the payload digest and therefore requires a new digest at
offset 24. JSON support is always present when the LODGE codec module is
compiled.

Flexbuffers encoding `1` requires the `io_flexbuffers` feature. Its root MUST
be a map. The current bounded preflight additionally requires the top-level
map keys and the `membership_index` map keys to be strictly increasing and
unique. Producers SHOULD use `encode_lodge_manifest_with_encoding` rather than
construct Flexbuffers manually. The default encoder writes Flexbuffers when
that feature is enabled and JSON otherwise.

The semantic payload has these top-level fields:

| Field | Type and meaning |
| --- | --- |
| `header` | semantic versions, feature bits, and aggregate counts |
| `base_manifest` | authenticated identity of the companion `.gsplatlod` |
| `extra_pages` | ordinary page descriptors for coarse authored records |
| `page_authentication` | encoded SHA-256 identity of every base and extra page |
| `levels` | distance thresholds, run ranges, and producer filter metadata |
| `record_runs` | dense stable IDs mapped to page-local record ranges |
| `clusters` | camera-cluster geometry and flattened-table ranges |
| `neighbors` | flattened, per-cluster neighbor IDs |
| `membership_index` | authenticated membership object and stream directory |

### Semantic header

`header` contains, in the Serde representation:

| Field | Type | Required value or invariant |
| --- | --- | --- |
| `magic` | `[u8; 8]` | byte sequence `BGSLOG1\0` |
| `manifest_version` | `u16` | `1` |
| `page_schema_version` | `u16` | `2` |
| `required_features` | `u64` | all five v1 bits below and no unknown bits |
| `base_page_count` | `u32` | nonzero and equal to authenticated base page count |
| `extra_page_count` | `u32` | equal to `extra_pages.len()` |
| `level_count` | `u32` | equal to `levels.len()` |
| `cluster_count` | `u32` | equal to `clusters.len()` and membership-entry count |
| `record_run_count` | `u32` | equal to `record_runs.len()` |
| `neighbor_count` | `u32` | equal to `neighbors.len()` |
| `stable_gaussian_count` | `u64` | last dense stable ID |
| `total_membership_ids` | `u64` | sum of all membership entry counts |

The v1 feature bits are mandatory rather than optional capabilities:

| Bit | Value | Meaning |
| ---: | ---: | --- |
| 0 | `0x01` | dense stable Gaussian IDs |
| 1 | `0x02` | depth-filter authoring metadata |
| 2 | `0x04` | camera clusters |
| 3 | `0x08` | authenticated dependencies |
| 4 | `0x10` | delta-uLEB128 memberships |

Readers reject both missing mandatory bits and unknown required bits.

### Shared structures

The remaining maps use these exact fields:

- `LodgeAuthenticatedObject`: `uri: String`, `encoded_len: u64`, and
  `sha256: [u8; 32]`.
- `LodgePageAuthentication`: `page: LodPageId` and
  `encoded_sha256: [u8; 32]`.
- `LodgeLevelDescriptor`: `id: LodgeLevelId`, `distance_min: f32`,
  `records: LodIndexRange`, and `filter: LodgeLevelFilter`.
- `LodgeRecordRun`: `first_id: LodgeGaussianId`, `count: u32`,
  `page: LodPageId`, and `page_offset: u32`.
- `LodgeCameraCluster`: `id: LodgeClusterId`, `center: [f32; 3]`,
  `radius: f32`, `neighbors: LodIndexRange`, and `membership_entry: u32`.
- `LodIndexRange`: `start: u32` and `count: u32`, indexing a manifest-owned
  vector.

`LodgeLevelFilter` is either the string `"Original"` or the externally tagged
variant `{"DepthAware3dV1": {"reference_depth": f32,
"reference_focal_length_px": f32, "smoothing_scale": f32,
"importance_threshold": f32, "fine_tune_steps": u32}}`.

Each entry of `extra_pages` is an unchanged `LodPageDescriptor` from the
ordinary package format: `id`, `kind`, `encoding`, `gaussian_count`,
`decoded_len`, `content_hash`, `bounds`, and `storage`. Its `storage`, when
present, contains `uri`, optional `byte_range`, and `encoded_len`. See the
[LoD package format](lod_format.md#page-containers) for page-container bytes,
decoded FNV-1a content hashes, and page descriptor semantics.

## Stable IDs, levels, runs, and pages

Stable Gaussian IDs are dense over the complete imported catalog:
`1..=stable_gaussian_count`. Zero is invalid. They are assigned by
concatenating levels in level-vector order and record runs in each level's run
range. Every nonempty run MUST begin at the next expected stable ID, so each
stable ID maps to exactly one `(page, page_offset)` record. Membership order
does not affect IDs. In the current resident materializer, stable ID `N` is
placed at physical catalog index `N - 1`.

There MUST be at least two levels. Level IDs equal their zero-based vector
indexes. `levels[*].records` are nonempty contiguous ranges which partition
the entire `record_runs` vector. Distances are finite, nonnegative, and
strictly increasing. Level zero has distance `+0.0` and filter `Original`.
Every coarser level uses `DepthAware3dV1`, whose:

- `reference_depth` has exactly the same `f32` bits as `distance_min`;
- `reference_focal_length_px` is finite and positive;
- `smoothing_scale` is finite and nonnegative; and
- `importance_threshold` is finite and in `[0, 1]`.

Level zero MUST reproduce the base manifest's original source leaves in exact
canonical source-leaf order and count. Its runs reference only base-package
pages. Coarse runs reference only `extra_pages`. Extra page IDs are strictly
increasing, do not collide with base page IDs, have kind `Representatives`,
and carry storage metadata. Across all coarse runs, every extra page record is
covered exactly once without a gap or overlap.

Each run's page-local half-open range
`page_offset..page_offset + count` MUST remain within the named page. The
sorted `page_authentication` vector MUST contain exactly one record for every
base and extra page and no other page.

The `.gslodge` sidecar does not change page schema 2 or its page container.
After SHA-256 authentication of the exact encoded page bytes, readers still
perform all ordinary page checks: page/container identities, bounds, lengths,
decoded content hash, and Gaussian-value validity.

## Camera clusters

There MUST be at least two clusters. Cluster IDs are nonzero, unique, and
strictly increasing. Centers contain three finite values and are distinct;
positive and negative zero compare as the same center. Radii are finite and
nonnegative.

`clusters[*].neighbors` are contiguous ranges which partition the flattened
`neighbors` vector. Each cluster has at least one neighbor, and each slice is
strictly increasing, unique, refers to a known cluster, and excludes the
cluster itself. `membership_entry` equals the cluster's vector index. The
membership entry at that index names the same cluster ID.

These fields describe producer-authored geometry and adjacency. They do not
turn overlapping active sets into a source-partition hierarchy. Runtime pair
selection and presentation are specified in the [LODGE runtime guide](lodge.md).

## Membership object

`membership_index` has these exact fields:

| Field | Type | v1 rule |
| --- | --- | --- |
| `schema_version` | `u16` | `1` |
| `encoding` | enum | string `"DeltaUleb128StableIdsV1"` |
| `object` | `LodgeAuthenticatedObject` | identity of the complete blob |
| `index_byte_range` | `(u64, u64)` | exactly `(0, L)` with `L > 0` |
| `index_sha256` | `[u8; 32]` | SHA-256 of `object[0..L]` |
| `entries` | vector | one entry per cluster, in cluster order |

Each `LodgeMembershipEntry` contains `cluster`, `byte_range`, `member_count`,
`first_id`, `last_id`, and `encoded_sha256`. Every membership is nonempty.
Endpoints are valid catalog IDs, `first_id <= last_id`, and the declared count
can fit in that inclusive endpoint interval.

The blob layout is exact and contiguous:

| Blob range | Meaning |
| --- | --- |
| `[0, L)` | authenticated index/directory prefix |
| `[L, L + E0)` | cluster 0 membership stream |
| next `E1` bytes | cluster 1 membership stream |
| ... | remaining streams in cluster order |

The first entry starts at `L`; every later entry starts exactly where the
previous one ends; every length is positive; and the final entry ends at
`object.encoded_len`. Gaps, overlaps, and trailing bytes are invalid. The
generic v1 runtime authenticates the prefix but treats its contents as opaque,
so authenticated third-party producer prefixes remain readable. The canonical
stream directory used at runtime is the payload's `entries` vector. Producers
MUST NOT require the runtime to parse an additional byte schema from the
prefix.

### Canonical `BGSLMEM` v1 prefix

The native `lod_build` API
`build_canonical_lodge_membership_artifact` emits the crate-owned canonical
producer profile. Its prefix has `L = 40 + 80 * N`, where `N` is the cluster
and directory-entry count. All integers are little-endian.

| Prefix offset | Width | Field |
| ---: | ---: | --- |
| 0 | 8 | magic `BGSLMEM\0` |
| 8 | 2 | membership-object container version, `1` |
| 10 | 2 | membership directory schema version, `1` |
| 12 | 2 | flags, zero |
| 14 | 2 | reserved, zero |
| 16 | 4 | directory-entry count `N` |
| 20 | 4 | directory-entry width, `80` |
| 24 | 8 | exact prefix length `L` |
| 32 | 8 | exact complete membership-object length |
| 40 | `80 * N` | fixed-width directory entries in cluster order |

Each directory entry starts at `40 + 80 * i` and duplicates the corresponding
semantic `membership_index.entries[i]` fields exactly:

| Entry offset | Width | Field |
| ---: | ---: | --- |
| 0 | 4 | nonzero cluster ID |
| 4 | 4 | reserved, zero |
| 8 | 8 | membership stream offset |
| 16 | 8 | membership stream length |
| 24 | 8 | decoded member count |
| 32 | 8 | first stable Gaussian ID |
| 40 | 8 | last stable Gaussian ID |
| 48 | 32 | SHA-256 of the exact encoded stream bytes |

The header object length, prefix length, directory count/width, every directory
field, payload `entries`, and actual contiguous stream ranges MUST agree. The
prefix SHA-256 is still `membership_index.index_sha256`, and the SHA-256 of the
complete prefix plus streams is `membership_index.object.sha256`.

The canonical builder scans each nonempty, strictly ordered cluster source
under `LodgeCodecLimits`, then replays it into one exactly reserved object.
Count, endpoints, encoded length, and encoded SHA-256 MUST be identical on both
passes; replay drift fails the build. This defines a deterministic crate-owned
adapter artifact, not an upstream LODGE checkpoint or file format.

### Delta-uLEB128 stream

For a strictly increasing membership `id[0], id[1], ...`, set `previous = 0`
and encode each positive delta `id[i] - previous` as unsigned LEB128. Each byte
carries the next low seven bits; bit 7 is set when more groups follow. Then set
`previous = id[i]`. Thus the first value is a delta from zero, not from one.

The representation MUST be shortest-form canonical uLEB128. A decoder rejects:

- an empty stream or a zero delta;
- a truncated value;
- a multi-byte value whose terminal seven-bit payload is zero;
- more than ten bytes for one `u64`, or a tenth-byte payload greater than one;
- integer addition overflow, a non-increasing result, or an ID above
  `stable_gaussian_count`;
- a decoded count different from `member_count`;
- decoded endpoints different from `first_id` and `last_id`; or
- bytes left outside the entry's exact range.

Each stream's SHA-256 is checked before decoding it.

## URI roots and dependency loading

Every URI carried directly by the LODGE sidecar is a nonempty canonical
relative path. It MUST NOT contain a scheme, leading slash, backslash, query,
fragment, percent escape, whitespace or control character, an empty path
segment, or a `.` or `..` segment.

Resolution has two deliberately different roots:

1. `base_manifest.uri`, `membership_index.object.uri`, and every
   `extra_pages[*].storage.uri` resolve relative to the directory containing
   the `.gslodge` sidecar.
2. Storage URIs in the decoded, authenticated base `.gsplatlod` manifest
   resolve relative to the directory containing that resolved `.gsplatlod`
   manifest, as specified by the [ordinary package format](lod_format.md#storage-uris-and-http).

For example, if `scene.gslodge` names `base/scene.gsplatlod`, a base page URI
`pages/0.gspage` resolves below `base/`, while a LODGE extra-page URI
`extras/1.gspage` resolves beside `scene.gslodge` under `extras/`.

For a standalone object, the fetched object length MUST equal `encoded_len`.
For a page descriptor with `byte_range = [start, length]`, the transport MUST
return exactly that range and `encoded_len` MUST equal `length`. Content
transformation or decompression that changes the authenticated bytes is
invalid. Native resolvers MUST also prevent canonicalized paths or symlinks
from escaping the selected package root. Remote transports SHOULD apply the
range, immutable-validator, redirect, and content-encoding rules in the
[ordinary package format](lod_format.md#storage-uris-and-http).

The Bevy `.gslodge` asset loader validates only the sidecar. It does not fetch
the dependency closure. The application is responsible for resolving the two
roots above, fetching exact bytes, constructing the authenticated base/page/
membership proofs, and then materializing the resident catalog. The current
integration requires the complete stable-ID catalog and every membership
before it draws; the format primitives do not imply automatic background
streaming.

### Feature and target availability

The container and membership codecs are public with either the `lod` or
`lod_build` feature. Flexbuffers additionally requires `io_flexbuffers`; the
`lod` feature enables it, so a normal runtime build accepts both defined
payload encodings. The `.gslodge` Bevy asset loader and CPU pair-planning APIs
require `lod`.

Authenticated resident ECS materialization is available only when the crate's
`lod_render_path` capability is present: `lod`, storage buffers, and radix sort
must be enabled, while `buffer_texture` and `webgl2` must be absent. The
standard `lod_render` and WebGPU `web` bundles meet that compile-time contract;
a WebGL2 build retains portable format/planning APIs but cannot instantiate the
resident LODGE renderer. This is a software availability contract, not evidence
of successful GPU or browser qualification on a particular adapter. Native and
Wasm applications remain responsible for the authenticated dependency-loading
steps above.

## Exact hash domains

All SHA fields use standard SHA-256. Except for the container payload digest,
semantic SHA fields MUST be nonzero. A digest authenticates byte identity
relative to the trusted sidecar; it is not a publisher signature.

| Digest field | Exact input bytes |
| --- | --- |
| outer header `[24, 56)` | serialized semantic payload only, beginning at byte 120 |
| outer header `[56, 88)` | complete encoded base `.gsplatlod` manifest |
| `base_manifest.sha256` | same complete encoded base manifest |
| `page_authentication[i].encoded_sha256` | exact encoded page object, or exact descriptor storage range, before page decoding |
| `membership_index.object.sha256` | complete membership object, prefix and every stream |
| `membership_index.index_sha256` | exact prefix selected by `index_byte_range`, which is `object[0..L]` in v1 |
| `membership_index.entries[i].encoded_sha256` | exact compressed stream selected by that entry's `byte_range` |

These checks are layered with, not substituted for, the ordinary package's
FNV-1a manifest/page checks. In particular, an encoded page passes its LODGE
SHA-256 before it is decoded and checked against the descriptor's canonical
decoded `content_hash`.

The trusted root is the `.gslodge` itself: its local path, immutable origin,
external signature, or independently pinned file hash is an application
decision. A self-consistent malicious sidecar and dependency set remains
malicious. Readers MUST enforce declared lengths and configured limits before
allocating or parsing dependency-controlled data.

## Reader limits

The format does not promise that every representable count is accepted.
`LodgeCodecLimits` provides application-configurable safety ceilings. Current
defaults are:

| Limit | Default |
| --- | ---: |
| complete `.gslodge` bytes | 64 MiB |
| levels | 64 |
| clusters / membership entries | 65,536 |
| record runs | 1,048,576 |
| extra pages | 262,144 |
| page authentications | 524,288 |
| flattened neighbors | 4,194,304 |
| stable Gaussians | 4,000,000,000 |
| total membership IDs | 16,000,000,000 |
| members in one cluster | 1,000,000,000 |
| one dependency and checked declared dependency aggregate | 1 TiB |
| one compressed membership stream | 512 MiB |

The declared dependency aggregate is the base-manifest length plus membership
object length plus all extra-page encoded lengths. Base-package page lengths
are checked after the base manifest itself is authenticated. Header counts are
bounded before payload deserialization; JSON and Flexbuffers receive a bounded
collection-shape preflight; decoded scalar counts, ranges, object lengths, and
the aggregate are checked again before dependency allocation.

These defaults are denial-of-service ceilings, not authoring targets.
Applications SHOULD lower them to match their deployment and memory budget.

## Compatibility and producer boundary

Readers fail closed on an unknown container, semantic, membership, or page
version; an unknown payload encoding; a nonzero reserved field; a missing
mandatory feature; or an unknown required feature. A new byte layout requires
a new container version or encoding discriminator. New semantic behavior
requires a new semantic/membership/page version or a required feature bit.
JSON and Flexbuffers can encode the same logical manifest but do not share a
byte identity or payload digest.

An exporter adapting external LODGE output MUST explicitly construct:

- page-schema-2 records for the original and independently authored levels;
- the dense run-ordered stable-ID catalog;
- sorted cluster IDs, finite distinct centers, and valid neighbor ranges;
- one nonempty, sorted membership per cluster using those stable IDs; and
- the exact object lengths and SHA-256 digests described above.

The crate does not infer this mapping from an upstream training checkpoint or
reinterpret a paper/reference-implementation artifact as `.gslodge`.

The canonical implementation is
[`gaussian/formats/lodge.rs`](../src/gaussian/formats/lodge.rs) for semantic
validation, [`io/lodge.rs`](../src/io/lodge.rs) for the container and
membership codecs, and
[`stream/lodge_resident.rs`](../src/stream/lodge_resident.rs) for authenticated
resident materialization.
