#!/usr/bin/env bash
# Replay a recorded session through the live pipeline in Docker on this Mac, laid out as on the Pi:
# tars-live-vio (Basalt) and tars-rig (tars_scan.live_map) on one host network, the map streamed to
# a Rerun viewer on this Mac. Replays run at recording speed, so it is a fair test of the live path.
#
#   scripts/live_replay.sh runs/rig/<session> [out_dir] [live_map flags...]
#
# Both containers share the Docker VM's network, as on the Pi (host networking there).
# Needs the images from rig/deploy.sh --build.
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=$(cd "$1" && pwd)
OUT=${2:-runs/live_view/$(basename "$SESSION")}
shift $(( $# >= 2 ? 2 : 1 ))
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
VIEWER_PORT=9876

if ! nc -z 127.0.0.1 "$VIEWER_PORT" 2>/dev/null; then
  echo "starting the Rerun viewer on port $VIEWER_PORT"
  .venv/bin/rerun --port "$VIEWER_PORT" >/dev/null 2>&1 &
  for _ in $(seq 50); do nc -z 127.0.0.1 "$VIEWER_PORT" 2>/dev/null && break; sleep 0.1; done
fi

docker rm -f tars-live-vio tars-live-replay >/dev/null 2>&1 || true
docker run -d --rm --name tars-live-vio --network host tars-live-vio:latest \
  --port 5077 --num-threads 3 >/dev/null
trap 'docker rm -f tars-live-vio >/dev/null 2>&1 || true' EXIT

# host.lima.internal is this Mac as seen from Colima's VM
docker run --rm --name tars-live-replay --network host \
  -v "$SESSION:/session:ro" -v "$OUT:/out" tars-rig:latest \
  python3 -m tars_scan.live_map /session --out /out --recording-id "$(basename "$SESSION")-live" \
  --connect "rerun+http://host.lima.internal:$VIEWER_PORT/proxy" "$@"
docker logs tars-live-vio 2>&1 | grep -c "dropped frame" | sed 's/^/Basalt dropped frames: /'
echo "trajectory, summary and recording in $OUT"
