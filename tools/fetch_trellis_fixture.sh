#!/usr/bin/env bash
set -euo pipefail

readonly TRELLIS_URL="https://mitchell.mosure.me/trellis.glb"
readonly TRELLIS_BYTE_LEN="112899460"
readonly TRELLIS_SHA256="fbe9d96b6689a78228c121e5f1bc8c5ccc32cef1941294d25f1db66f4a901dc1"

usage() {
  echo "usage: $0 OUTPUT_PATH" >&2
}

if [[ "$#" -ne 1 || -z "$1" ]]; then
  usage
  exit 2
fi

readonly output_path="$1"
if [[ -d "${output_path}" ]]; then
  echo "Trellis fixture output is a directory: ${output_path}" >&2
  exit 2
fi

if command -v sha256sum >/dev/null 2>&1; then
  readonly checksum_tool="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  readonly checksum_tool="shasum"
else
  echo "sha256sum or shasum is required to verify the Trellis fixture" >&2
  exit 2
fi

checksum() {
  local path="$1"
  if [[ "${checksum_tool}" == "sha256sum" ]]; then
    sha256sum -- "${path}" | awk '{ print $1 }'
  else
    shasum -a 256 -- "${path}" | awk '{ print $1 }'
  fi
}

fixture_matches() {
  local path="$1"
  local observed_byte_len
  local observed_sha256

  [[ -f "${path}" ]] || return 1
  observed_byte_len="$(wc -c < "${path}")"
  observed_byte_len="${observed_byte_len//[[:space:]]/}"
  [[ "${observed_byte_len}" == "${TRELLIS_BYTE_LEN}" ]] || return 1
  observed_sha256="$(checksum "${path}")"
  [[ "${observed_sha256}" == "${TRELLIS_SHA256}" ]]
}

if fixture_matches "${output_path}"; then
  echo "Using verified Trellis fixture: ${output_path}"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Trellis fixture is absent or invalid and curl is unavailable: ${output_path}" >&2
  exit 1
fi

output_dir="$(dirname -- "${output_path}")"
mkdir -p -- "${output_dir}"
temporary_path="$(mktemp "${output_path}.part.XXXXXX")"
cleanup() {
  rm -f -- "${temporary_path}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Fetching canonical Trellis fixture: ${TRELLIS_URL}"
curl \
  --fail \
  --location \
  --proto '=https' \
  --retry 5 \
  --retry-all-errors \
  --retry-delay 2 \
  --connect-timeout 15 \
  --max-time 600 \
  --output "${temporary_path}" \
  "${TRELLIS_URL}"

if ! fixture_matches "${temporary_path}"; then
  observed_byte_len="$(wc -c < "${temporary_path}")"
  observed_byte_len="${observed_byte_len//[[:space:]]/}"
  observed_sha256="$(checksum "${temporary_path}")"
  echo "Downloaded Trellis fixture failed provenance validation" >&2
  echo "  expected bytes:  ${TRELLIS_BYTE_LEN}" >&2
  echo "  observed bytes:  ${observed_byte_len}" >&2
  echo "  expected sha256: ${TRELLIS_SHA256}" >&2
  echo "  observed sha256: ${observed_sha256}" >&2
  exit 1
fi

chmod 0644 "${temporary_path}"
mv -f -- "${temporary_path}" "${output_path}"
if ! fixture_matches "${output_path}"; then
  echo "Installed Trellis fixture failed final provenance validation: ${output_path}" >&2
  exit 1
fi

echo "Installed verified Trellis fixture: ${output_path}"
