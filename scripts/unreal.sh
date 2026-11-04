#!/usr/bin/env bash
# Build and run the TarsRl Unreal project on Linux (the GPU box). Unreal 5.5 for Linux lives in
# $UE_ROOT (default ~/UnrealEngine: the folder that holds Engine/).
#
#   scripts/unreal.sh build             compile the C++ (first time ~10 min, then seconds)
#   scripts/unreal.sh view [PORT]       a window on this screen; TARS waits for Python
#   scripts/unreal.sh fast [N]          N headless Unreals (no rendering) on ports 5577..
#   scripts/unreal.sh stop              stop the headless ones
#   scripts/unreal.sh sequence          T10: the scripted body comparison, in a window
#   scripts/unreal.sh sequence-fast     T10 headless: records the CSV (~40 s) and exits
#
# Then drive it from Python, e.g.
#   uv run python -m tars_rl.unreal.verify --policy runs/... --level flat --realtime   (view)
#   uv run python -m tars_rl.unreal.verify --policy runs/... --ports 5577 5578 5579 5580 (fast 4)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
project="$root/unreal/TarsRl/TarsRl.uproject"
ue=${UE_ROOT:-$HOME/UnrealEngine}
editor="$ue/Engine/Binaries/Linux/UnrealEditor"
pids="$root/unreal/TarsRl/Saved/bridge.pids"
logs="$root/unreal/TarsRl/Saved/Logs"

need_engine() {
  [ -x "$editor" ] || { echo "No Unreal at $ue (looked for $editor). Unzip Unreal 5.5 for Linux there, or set UE_ROOT." >&2; exit 1; }
}

case "${1:-}" in
  build)
    need_engine
    "$ue/Engine/Build/BatchFiles/Linux/Build.sh" TarsRlEditor Linux Development -project="$project" -waitmutex
    ;;
  view)
    need_engine
    exec "$editor" "$project" -game -windowed -ResX=1600 -ResY=900 -TarsPort="${2:-5577}" -log
    ;;
  sequence)
    need_engine
    exec "$editor" "$project" -game -windowed -ResX=1600 -ResY=900 -TarsMode=sequence -log
    ;;
  sequence-fast)
    need_engine
    csv="$root/unreal/TarsRl/Saved/TarsRuns/unreal_poses.csv"
    mkdir -p "$logs"
    rm -f "$csv"
    "$editor" "$project" -game -nullrhi -nosound -unattended -TarsMode=sequence \
      -abslog="$logs/sequence.log" > /dev/null 2>&1 &
    pid=$!
    # The actor writes the CSV once the sequence ends; Unreal itself keeps running, so stop it.
    for _ in $(seq 600); do
      [ -s "$csv" ] && break
      kill -0 $pid 2>/dev/null || break
      sleep 1
    done
    kill $pid 2>/dev/null || true
    [ -s "$csv" ] || { echo "no recording; see $logs/sequence.log" >&2; exit 1; }
    echo "recorded $csv"
    ;;
  fast)
    need_engine
    n=${2:-1}
    mkdir -p "$logs"
    : > "$pids"
    for i in $(seq 0 $((n - 1))); do
      port=$((5577 + i))
      rm -f "$logs/bridge-$port.log"  # an old log's "listening" line would pass the wait below
      "$editor" "$project" -game -nullrhi -nosound -unattended -TarsPort=$port \
        -abslog="$logs/bridge-$port.log" > /dev/null 2>&1 &
      echo $! >> "$pids"
      echo "headless Unreal on port $port (log $logs/bridge-$port.log)"
    done
    # Ready when each log says the bridge listens (the first start loads shaders: up to minutes).
    for i in $(seq 0 $((n - 1))); do
      port=$((5577 + i))
      for _ in $(seq 600); do
        grep -q "TARS bridge listening" "$logs/bridge-$port.log" 2>/dev/null && break
        sleep 1
      done
      grep -q "TARS bridge listening" "$logs/bridge-$port.log" 2>/dev/null \
        || { echo "port $port never came up; see $logs/bridge-$port.log" >&2; exit 1; }
    done
    echo "ready: ports $(seq -s ' ' 5577 $((5577 + n - 1)))"
    ;;
  stop)
    [ -f "$pids" ] && xargs -r kill < "$pids" && rm -f "$pids" && echo stopped || echo "nothing running"
    ;;
  *)
    sed -n '2,16p' "$0"
    exit 1
    ;;
esac
