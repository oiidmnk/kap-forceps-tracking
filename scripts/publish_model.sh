#!/usr/bin/env bash
# Publish a weights file as a new immutable version on the shared SSH host.
#   scripts/publish_model.sh [weights.pt] [-v VERSION] [-m "note / metrics"]
# Layout on the host:  $MODEL_DIR/<version>/{model.pt,manifest.json}  and  $MODEL_DIR/latest -> <version>
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/model_common.sh

WEIGHTS=services/mlseg/weights/last.pt
VERSION="v$(date +%Y%m%d-%H%M)"
NOTE=""
while [ $# -gt 0 ]; do
  case "$1" in
    -v) VERSION="$2"; shift 2 ;;
    -m) NOTE="$2"; shift 2 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) WEIGHTS="$1"; shift ;;
  esac
done
[ -f "$WEIGHTS" ] || { echo "no such file: $WEIGHTS" >&2; exit 1; }

SHA=$(sha256_of "$WEIGHTS")
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
cp "$WEIGHTS" "$STAGE/$MODEL_FILE"
VERSION="$VERSION" SHA="$SHA" NOTE="$NOTE" SIZE=$(wc -c <"$WEIGHTS" | tr -d ' ') \
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo -dirty)" \
AUTHOR="$(git config user.name 2>/dev/null || whoami)" \
python3 - >"$STAGE/manifest.json" <<'PY'
import json, os, datetime
e = os.environ
print(json.dumps({"version": e["VERSION"], "file": "model.pt", "sha256": e["SHA"], "bytes": int(e["SIZE"]),
                  "published": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                  "author": e["AUTHOR"], "git_commit": e["COMMIT"], "note": e["NOTE"]}, indent=2))
PY

# refuse to overwrite; upload to a hidden dir, then rename, then flip `latest` last
ssh "$MODEL_HOST" "mkdir -p '$MODEL_DIR' && chmod g+rwxs '$MODEL_DIR' 2>/dev/null; test ! -e '$MODEL_DIR/$VERSION'" \
  || { echo "version $VERSION already exists on $MODEL_HOST" >&2; exit 1; }
rsync -a --chmod=Dg+rwxs,Fg+rw "$STAGE/" "$MODEL_HOST:$MODEL_DIR/.incoming-$VERSION/"
ssh "$MODEL_HOST" "cd '$MODEL_DIR' && mv '.incoming-$VERSION' '$VERSION' && ln -sfn '$VERSION' latest"
echo "published $VERSION ($SHA) -> $MODEL_HOST:$MODEL_DIR/$VERSION (latest)"
