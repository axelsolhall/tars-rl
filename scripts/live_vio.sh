#!/usr/bin/env bash
# Basalt on a live stream (tars_live_vio, built by ~/dev/slam/build-basalt.sh into the slam-basalt-build
# volume) in Docker on this Mac, for replaying sessions through tars_scan.live_map:
#   scripts/live_vio.sh          start (port 5077 on localhost)      scripts/live_vio.sh stop
# The stereo-hypothesis patch settings are the ones the walk2 maps used (night_log.md).
set -euo pipefail
docker rm -f tars-live-vio >/dev/null 2>&1 || true
[[ ${1:-} == stop ]] && exit 0
docker run -d --rm --name tars-live-vio -p 127.0.0.1:5077:5077 \
  -e TARS_STEREO_DEPTHS=0.25,0.4,0.65,1.0,2,5 -e TARS_STEREO_AMBIGUITY=1.6 \
  -v slam-basalt-build:/build slam-basalt-deps /build/tars_live_vio --bind 0.0.0.0 --num-threads 3
