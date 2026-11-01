#!/usr/bin/env bash
# Fetch a model version from the shared SSH host into services/mlseg/weights/last.pt (sha256-verified).
#   scripts/fetch_model.sh [VERSION|latest] [-l]      -l lists available versions
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/model_common.sh

DEST=services/mlseg/weights/last.pt
VERSION=latest
while [ $# -gt 0 ]; do
  case "$1" in
    -l) ssh "$MODEL_HOST" "cd '$MODEL_DIR' && ls -1 | grep -v '^\.' ; echo; readlink latest | sed 's/^/latest -> /'"; exit 0 ;;
    -h|--help) sed -n '2,3p' "$0"; exit 0 ;;
    *) VERSION="$1"; shift ;;
  esac
done

mkdir -p "$(dirname "$DEST")"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
rsync -a "$MODEL_HOST:$MODEL_DIR/$VERSION/manifest.json" "$TMP/manifest.json"
read -r SHA REAL < <(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["sha256"], m["version"])' "$TMP/manifest.json")

if [ -f "$DEST" ] && [ "$(sha256_of "$DEST")" = "$SHA" ]; then
  echo "weights already at $REAL"; exit 0
fi
rsync -a --progress "$MODEL_HOST:$MODEL_DIR/$VERSION/$MODEL_FILE" "$TMP/$MODEL_FILE"
[ "$(sha256_of "$TMP/$MODEL_FILE")" = "$SHA" ] || { echo "sha256 mismatch for $REAL" >&2; exit 1; }
mv "$TMP/$MODEL_FILE" "$DEST"
cp "$TMP/manifest.json" "$(dirname "$DEST")/manifest.json"
echo "fetched $REAL -> $DEST"
