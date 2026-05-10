#!/usr/bin/env bash
# Helper: copy generated TFLite files into Flutter assets and commit.
# Usage: ./tools/install_tflite_assets.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
MASAUSTU_DIR="$ROOT_DIR/masaustu"
ASSETS_DIR="$ROOT_DIR/mobile_app/assets"

mkdir -p "$ASSETS_DIR"

COPIED=0
for f in model_cnn.tflite model_lstm.tflite; do
  if [ -f "$MASAUSTU_DIR/$f" ]; then
    cp -v "$MASAUSTU_DIR/$f" "$ASSETS_DIR/$f"
    git add "$ASSETS_DIR/$f"
    COPIED=$((COPIED+1))
  fi
done

if [ "$COPIED" -eq 0 ]; then
  echo "No TFLite files found in $MASAUSTU_DIR. Nothing to do."
  exit 0
fi

git config user.name "CI-helper"
git config user.email "ci-helper@example.com"
git commit -m "Add generated TFLite assets" || echo "Nothing to commit"
echo "Committed $COPIED TFLite files to repo. Push manually if desired."
