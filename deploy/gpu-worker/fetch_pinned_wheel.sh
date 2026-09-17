#!/bin/sh
# Download one wheel and fail closed unless the SHA-256 pin matches.
set -eu
url="${1:?wheel url required}"
dest="${2:?destination path required}"
sha="${3:?sha256 required}"
mkdir -p "$(dirname "$dest")"
curl -fsSL --retry 5 --retry-delay 2 -o "$dest" "$url"
echo "${sha}  ${dest}" | sha256sum --check --strict
