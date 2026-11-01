# shared by publish_model.sh / fetch_model.sh (sourced)
MODEL_HOST="${MODEL_HOST:-luis.carilla@musicanalyzer.cc}"
MODEL_DIR="${MODEL_DIR:-mlseg-models}"            # relative to the remote home
MODEL_FILE="model.pt"

sha256_of() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
