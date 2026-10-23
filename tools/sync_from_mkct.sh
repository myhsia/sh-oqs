#!/usr/bin/env bash
# Sync the shared pieces from the canonical ../mkct4oqs checkout.
# mkct4oqs owns reference.py, the configs, and qutils/fsutils/pltools/mre/extended_deom;
# this repository keeps synced copies so that a public clone and the CI work without a token.
# Run after changing them in mkct4oqs, then commit and push here.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/../mkct4oqs/source"
[ -d "$src" ] || { echo "missing canonical checkout at $src"; exit 1; }
mkdir -p "$root/source/config" "$root/source/library"
rsync -a --delete "$src/config/" "$root/source/config/"
cp "$src/reference.py" "$root/source/reference.py"
cp "$src/library/qutils.py" "$src/library/fsutils.py" "$src/library/pltools.py" "$root/source/library/"
cp "$src/library/extended_deom.py" "$src/library/mre.py" "$root/source/library/"
echo "synced config/, reference.py and qutils/fsutils/pltools/extended_deom/mre.py from $src"
