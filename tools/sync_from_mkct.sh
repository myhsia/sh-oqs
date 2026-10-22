#!/usr/bin/env bash
# Sync the shared pieces from the canonical ../mkct-oqs checkout.
# mkct-oqs owns heom.py, the configs, and qutils/fsutils/pltools; this repository
# keeps synced copies so that a public clone and the CI work without a token.
# Run after changing them in mkct-oqs, then commit and push here.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/../mkct-oqs/source"
[ -d "$src" ] || { echo "missing canonical checkout at $src"; exit 1; }
mkdir -p "$root/source/config" "$root/source/library"
rsync -a --delete "$src/config/" "$root/source/config/"
cp "$src/heom.py" "$root/source/heom.py"
cp "$src/library/qutils.py" "$src/library/fsutils.py" "$src/library/pltools.py" "$root/source/library/"
echo "synced config/, heom.py and qutils/fsutils/pltools.py from $src"
