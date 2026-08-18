# Security policy

## Supported versions

Security fixes are applied to the latest published major release. Older major
versions may receive a fix when the affected code is shared and the patch can
be applied safely, but they are not guaranteed support.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not
open a public issue for an unpatched vulnerability. Include the affected crate
version or commit, target platform, feature set, reproduction input, expected
impact, and whether the input crossed a local, filesystem, HTTP, or browser
boundary.

The maintainers will coordinate disclosure after a fix or mitigation is
available.

## LoD package trust boundary

LoD manifests, pages, shard indexes, HTTP responses, and persistent-cache
records are treated as untrusted structured data. Their decoders enforce
explicit byte, count, allocation, checksum, and version bounds. PLY conversion
enforces vertex, batch, and normalized-allocation bounds, but the upstream PLY
parser does not bound an arbitrarily long header or ASCII line; applications
handling hostile PLY bytes should impose an outer input/line limit. Stable
package checksums detect accidental corruption; they are not digital
signatures. A deployment that requires publisher authenticity must
authenticate its manifest and transport independently until a signed-manifest
policy is part of the file format.

Native package roots should be application-controlled. Browser packages should
use immutable HTTPS objects with strong validators and an appropriate Content
Security Policy.
