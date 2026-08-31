# LoD package format (semantic manifest v3)

This document defines the package bytes accepted by the current LoD reader.
The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. A
reader must reject incompatible versions and features; it must not reinterpret
them as a best-effort older or newer layout.

All fixed-width integers and floating-point bit patterns are little-endian.
Offsets are zero-based. A byte range is the pair `(start, length)`, not an
inclusive end. Declared lengths are exact: trailing bytes are invalid unless a
container explicitly assigns them to a page payload.

| Layer | Magic | Version | Conventional suffix |
| --- | --- | ---: | --- |
| manifest container | `BGSLODC\0` | 2 | `.gsplatlod` |
| semantic manifest | `BGSLOD3\0` | 3 | inside the manifest container |
| decoded page schema | -- | 2 | inside each page container |
| page container | `BGSPAGE\0` | 1 or 2 | `.gspage` or a shard range |
| shard container | `BGSSHARD` | 1 | `.bgslodpack` |

## Common checksum

Checksums use 64-bit FNV-1a with offset basis `0xcbf29ce484222325`, prime
`0x00000100000001b3`, and wrapping multiplication after each XOR. The byte
scope differs by field and is specified below.

## Manifest container v2

The manifest starts with this exact 40-byte envelope:

| Offset | Width | Field |
| ---: | ---: | --- |
| 0 | 8 | magic `BGSLODC\0` |
| 8 | 2 | container version, `2` |
| 10 | 1 | payload encoding: Flexbuffers `1`, JSON `2` |
| 11 | 5 | reserved, all zero |
| 16 | 4 | encoded node count |
| 20 | 4 | encoded page count |
| 24 | 8 | payload byte length |
| 32 | 8 | FNV-1a checksum of the serialized payload bytes |
| 40 | variable | serialized `GaussianLodManifest` payload |

The file length MUST equal `40 + payload byte length`. Both envelope counts
MUST equal the corresponding semantic header count and decoded vector length.
Readers MUST apply configured byte and count limits before deserializing or
allocating from untrusted declarations.

### Semantic manifest v3

The serialized payload contains these top-level fields:

- `header`: magic, semantic and page-schema versions, required feature bits,
  source/stored Gaussian counts, and node/page counts;
- `scene_bounds`: absent only for an empty scene;
- `roots`: stable root node IDs;
- `nodes`: breadth-first nodes with parent, depth, bounds, contiguous child
  range, Morton/source ranges, page slice, error, quality interval, and
  high-fidelity certificate;
- `pages`: page descriptors with ID, kind, encoding, counts, decoded length,
  decoded content hash, bounds, and optional storage location;
- `build`: settings, reducer, builder/reducer ABI versions, and source/config
  fingerprints;
- `quality`: depth, finest/coarsest counts, and maximum error;
- `morph_map`: the ABI 16 monotone immediate-child-record to parent-record
  correspondence described below; absent for every older readable ABI.

The semantic header MUST use magic `BGSLOD3\0`, manifest version `3`, and page
schema version `2`. Known `required_features` bits are:

| Bit | Value | Meaning |
| ---: | ---: | --- |
| 0 | `0x01` | SH0 decoded layout |
| 1 | `0x02` | SH1 decoded layout |
| 2 | `0x04` | SH2 decoded layout |
| 3 | `0x08` | SH3 decoded layout |
| 4 | `0x10` | SH4 decoded layout |
| 5 | `0x20` | monotonic high-fidelity certificates |
| 6 | `0x40` | same-depth, same-kind nodes may share physical pages |
| 7 | `0x80` | ABI 16 monotone parent/child morph map |

Exactly one SH bit and the certificate bit MUST be set. The SH bit MUST match
the reader's compiled SH degree. Bit 6 is optional. Bit 7 and `morph_map` MUST
both be present for ABI 16 and MUST both be absent for older ABIs. Unknown
required bits are rejected. The decoded padded SH coefficient counts for SH0
through SH4 are `4`, `12`, `28`, `48`, and `76`, respectively.

The currently recognized MomentMerge builder/reducer ABI pairs are legacy
external CPU `(5, 2)`, legacy external GPU `(6, 2)`, progressive in-memory CPU
`(14, 3)`, progressive external-memory CPU `(15, 3)`, and spatial progressive
external-memory CPU `(16, 4)`. ABI 15 remains readable and retains the
configured wide topology while bounding every parent-to-children
representation-count amplification and accumulating each representative from
an original canonical source interval. ABI 16 retains those contracts, adds
renderer-consistent bounded spatial fitting, and requires the monotone morph
map. Other pairs are incompatible with semantic v3.

### ABI 16 morph map schema 1

`morph_map.schema_version` MUST be `1`. `node_runs` is index-aligned with the
manifest's breadth-first `nodes` vector and contains one range into the flat
`child_run_lengths: Vec<u16>` array per node. Those ranges MUST be contiguous,
non-overlapping, start at zero, and cover the flat array exactly.

A leaf has an empty run range. For an internal node, the run slice MUST contain
exactly one positive value per parent-local representation record. Immediate
children are concatenated in manifest child-range order; records within each
child retain page-local representation order. Run `p` maps the next
`child_run_lengths[p]` concatenated child records to parent-local record `p`.
The run sum MUST equal the total immediate-child representation count. The
implicit parent indexes are therefore monotone and surjective without storing
one parent index per child record. ABI 16 also constrains the configured leaf
capacity to the portable `u16` run ABI.

The sidecar changes neither page schema nor page encoding. Runtime morphing is
optional presentation behavior: a reader that accepts ABI 16 MUST validate the
sidecar fail-closed, while a renderer that cannot stage or bind a particular
bounded morph transaction may publish the already-valid complete target cut as
a categorical bounded-hard transition.

A valid non-empty hierarchy MUST have unique, nonzero node/page IDs; a rooted,
acyclic tree in which every node is reached exactly once; contiguous
breadth-first child ranges; finite nested bounds and monotonic error/quality
metadata; and page slices that stay within their descriptors. Leaves MUST
partition the canonical Morton-sorted source exactly and use full-precision
source records. Shared pages require bit 6 and may combine only non-overlapping
slices of nodes with the same depth and page kind. Header, quality, node, page,
source, and stored-record counts MUST agree.

ABI 16's spatial fitter is a builder guarantee, not additional serialized
topology. It jointly evaluates authored-support-touching nodes only within each
same-depth future-parent cohort (at most the validated branching factor of 32).
Same-depth boundaries split across different future parents and mixed-depth
boundaries in a selected cut are not claimed to be jointly fit; they remain an
image-oracle qualification surface. Unsafe measured within-cohort seams are
carried through ordinary selection-visible error metadata. Coarse cohorts that
do not retain source partitions are reported as unmeasured rather than assigned
an artificial infinite error.

The fitter models the production screen filter. For a positive projected
covariance `C`, the renderer uses `C' = C + 0.3 I` in pixel units and multiplies
peak alpha by `sqrt(clamp(det(C) / det(C'), 0, 1))`; invalid determinant cases
contribute zero. Flat source references retain their opacity-adaptive support
cutoff, while emitted LoD representatives retain at least authored 3-sigma
support. Fitting checks a normalized `0.0625x..4x` projection-scale ladder so
the fixed pixel-space mip term cannot validate a change at only one zoom.

With the maximum branching factor, all sibling-node pairs require at most 496
pair checks and the deterministic 3x3 tangential grid requires at most 4,464
seam probes per cohort. An accepted edit keeps positions fixed, widens tangent
covariance only within the exact touching pair's authored support envelope,
reapplies the all-view opacity ceiling, improves the target seam, and does not
regress any affected sampled seam or cohort composited error at any sampled
scale. A rejected measured edit is represented through selection-visible error
metadata rather than by silently increasing opacity or support.

## Page containers

Every page is an independently decodable container with this exact 44-byte
header:

| Offset | Width | Field |
| ---: | ---: | --- |
| 0 | 8 | magic `BGSPAGE\0` |
| 8 | 2 | page container version, `1` or `2` |
| 10 | 2 | decoded page schema version, `2` |
| 12 | 8 | nonzero page ID |
| 20 | 4 | nonzero Gaussian count |
| 24 | 4 | encoded SH coefficient count per Gaussian |
| 28 | 8 | payload byte length |
| 36 | 8 | decoded canonical page content hash |
| 44 | variable | record-major payload |

The container length MUST equal `44 + payload byte length`. The historical
manifest name `F32Planar` does **not** mean that the payload is plane-major or
directly GPU-uploadable. Both current payloads are record-major.

Let `D` be the reader's compiled SH degree and
`C = round_up_to_4(3 * (D + 1)^2)`. Let `d` be a reduced degree and
`c = 3 * (d + 1)^2`.

| Container | Descriptor encoding | SH header count | Bytes per record |
| --- | --- | ---: | ---: |
| v1 | `F32Planar` | exactly `C` | `4 * (12 + C)` |
| v2 | `F16Sh { degree: d }` | exactly `c`, with `d <= D` | `48 + 2 * c` |

Each record stores, in order:

1. position `x, y, z` and visibility as four `f32` values;
2. the SH coefficient sequence: `C` `f32` values in v1, or the first `c`
   interleaved RGB coefficients as IEEE binary16 values in v2;
3. rotation as four `f32` values;
4. scale `x, y, z` and opacity as four `f32` values.

V2 decoding expands binary16 values to `f32` and fills every remaining padded
or higher-degree coefficient with positive zero. Only `Representatives` pages
may use v2; `SourceLeaves` and `Mixed` pages MUST use v1. Every decoded record
must contain finite fields, nonnegative scales, and a nondegenerate rotation.
The descriptor's ID, encoding, Gaussian count, and content hash MUST match the
decoded page. Its `decoded_len` MUST equal `count * 4 * (12 + C)`, regardless
of container encoding. Descriptor and referenced-node bounds MUST
conservatively contain the decoded Gaussian support.

The page content hash is FNV-1a over the following decoded canonical stream:

1. schema version as `u16`, page ID as `u64`, and Gaussian count as `u64`;
2. for every Gaussian in order, the `f32` bits of position, visibility, all
   `C` SH coefficients, rotation, scale, and opacity.

Each `f32` is little-endian and both signed zeros hash as positive zero; all
other bit patterns are unchanged. For v2 this hash is computed **after** f16
rounding and zero filling. It is not a checksum of the encoded payload bytes.

## Shard container v1

A shard concatenates complete page containers after an exact 40-byte header
and fixed-width range table:

| Offset | Width | Field |
| ---: | ---: | --- |
| 0 | 8 | magic `BGSSHARD` |
| 8 | 2 | shard version, `1` |
| 10 | 2 | reserved, zero |
| 12 | 4 | nonzero entry count `N` |
| 16 | 8 | table byte length, exactly `32 * N` |
| 24 | 8 | exact total shard byte length |
| 32 | 8 | FNV-1a checksum of the table bytes only |
| 40 | `32 * N` | range table |

Each table entry is page ID `u64`, absolute byte offset `u64`, encoded length
`u64`, and decoded page content hash `u64`. Page IDs MUST be nonzero and
strictly increasing. The first page starts at `40 + 32 * N`; all ranges MUST be
positive, contiguous, non-overlapping, and end exactly at the declared file
length. Each range contains one complete `BGSPAGE` container.

For a packed page, its descriptor storage range and encoded length MUST equal
the shard entry, and its descriptor content hash MUST equal the entry. For a
standalone page, `byte_range` is absent and the named object length MUST equal
`encoded_len`.

## Storage URIs and HTTP

`storage.uri` is resolved relative to the directory containing the manifest.
For a package portable across native and Web loaders it MUST be a nonempty
relative path with no scheme, leading slash, backslash, query, fragment,
percent escape, control/space character, empty segment, or `.`/`..` segment.
The current native reader accepts a slightly broader platform-relative path,
but still rejects absolute paths, URI schemes, parent/root/prefix components,
and any canonicalized target that escapes the canonical package root,
including through a symlink.

HTTP manifest URLs and derived package base URLs MUST use `http` or `https`
with a nonempty authority, and the manifest URL MUST name a file so its parent
can become the package base. Queries, fragments, percent escapes, backslashes,
spaces/controls, and user information are rejected; consequently signed query
URLs and redirect-based object locations are not supported by this transport.

The default package HTTP transport enforces:

- standalone objects return `200`; range-backed pages return `206` and an
  exact `Content-Range` for `start..start + length - 1`;
- redirects are rejected;
- `Content-Length` is present and equals `encoded_len`, and the received body
  has exactly that length;
- `Content-Encoding` is absent or `identity`;
- every response carries a non-weak, immutable `ETag`; the first validator is
  pinned per object, later requests send `If-Match`, and mismatches fail;
- range arithmetic does not overflow and range length equals `encoded_len`.

A custom `HttpRangePageTransport` may use a configured immutable version
header instead of an ETag, but the high-level native and browser package APIs
currently require the ETag path. `Last-Modified` alone is not sufficient for
those APIs.

For cross-origin browser hosting, the server MUST allow `GET`, `Range`, and
`If-Match` as applicable, and expose `Content-Length`, `Content-Range`,
`Content-Encoding`, `ETag`, and `Retry-After`. It SHOULD advertise
`Accept-Ranges: bytes`. CDN compression or content transformation MUST be
disabled for page and shard objects so stored offsets remain valid.

## Compatibility rule

New byte layouts require a new container version or encoding discriminator;
new decoded semantics require a new page schema, semantic version, or required
feature bit. In particular, a plane-major/direct-upload page must not reuse
`F32Planar`, and an ABI 16 writer must not omit or relabel the required morph
feature/sidecar. Writers SHOULD emit Flexbuffers manifests and readers MAY also
accept JSON when that codec feature is compiled.

The canonical validation and codec implementations are
[`planar_3d_lod.rs`](../src/gaussian/formats/planar_3d_lod.rs),
[`planar_3d_chunked.rs`](../src/gaussian/formats/planar_3d_chunked.rs),
[`io/lod.rs`](../src/io/lod.rs),
[`stream/transport.rs`](../src/stream/transport.rs), and
[`stream/http.rs`](../src/stream/http.rs).
