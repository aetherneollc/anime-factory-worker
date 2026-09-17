#!/bin/sh
# Download one wheel into DEST_DIR using the URL basename (PEP 427 name).
# pip rejects dest names like fouroversix.whl ("not a valid wheel filename").
set -eu
url="${1:?wheel url required}"
dest_dir="${2:?destination directory required}"
sha="${3:?sha256 required}"
mkdir -p "$dest_dir"
name="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.unquote(sys.argv[1].rsplit("/", 1)[-1]))' "$url")"
case "$name" in
  *-*-*.whl) ;;
  *)
    echo "refusing non-wheel basename: $name" >&2
    exit 1
    ;;
esac
dest="${dest_dir}/${name}"
curl -fsSL --retry 5 --retry-all-errors --retry-delay 2 \
  -A "AnimeFactoryWorker/fetch_pinned_wheel" \
  -o "$dest" "$url"
echo "${sha}  ${dest}" | sha256sum --check --strict
