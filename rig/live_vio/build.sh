#!/usr/bin/env bash
# Build the tars-live-vio image: the binary from the Basalt build volume, then the runtime image.
set -euo pipefail
cd "$(dirname "$0")"
docker run --rm --platform linux/arm64 -v slam-basalt-build:/build -v "$PWD":/out slam-basalt-deps \
  cp /build/tars_live_vio /out/tars_live_vio
docker build -q --platform linux/arm64 -t tars-live-vio:latest . >/dev/null
echo "tars-live-vio:latest built"
