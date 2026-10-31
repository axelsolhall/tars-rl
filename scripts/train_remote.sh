#!/usr/bin/env bash
# Full training run on the GPU box, started from the Mac over SSH (A62).
#
#   TARS_HOST=user@host scripts/train_remote.sh start [seed] [name]   # sync code, start in background
#   TARS_HOST=user@host scripts/train_remote.sh status [name]         # tail the run's log
#   TARS_HOST=user@host scripts/train_remote.sh fetch  [name]         # copy runs/<name> back here
#
# TARS_HOST is never stored in the repo. TARS_REMOTE_DIR (default ~/tars-rl) is where the code
# lives on the box. The run refuses to start if the GPU has less than TARS_MIN_FREE_MIB free
# (default 12000): stop whatever else holds the GPU first.
set -euo pipefail

cmd=${1:-}
host=${TARS_HOST:?set TARS_HOST=user@host}
dir=${TARS_REMOTE_DIR:-tars-rl}
min_free=${TARS_MIN_FREE_MIB:-12000}
root=$(cd "$(dirname "$0")/.." && pwd)

case "$cmd" in
  start)
    seed=${2:-0}
    name=${3:-full-seed$seed}
    rsync -az --delete --exclude .git --exclude .venv --exclude runs --exclude .loopdesk \
      --exclude '.env*' "$root/" "$host:$dir/"
    ssh "$host" bash -s -- "$dir" "$seed" "$name" "$min_free" <<'EOF'
set -euo pipefail
dir=$1 seed=$2 name=$3 min_free=$4
export PATH="$HOME/.local/bin:$PATH"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt "$min_free" ]; then
  echo "GPU has ${free} MiB free (< ${min_free}); free it and retry." >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 1
fi
cd "$dir"
uv sync --extra cuda
mkdir -p "runs/$name"
nohup uv run python -m tars_rl.train.run --profile full --seed "$seed" --out "runs/$name" \
  > "runs/$name/train.log" 2>&1 < /dev/null &
echo "started runs/$name (pid $!) with ${free} MiB GPU free"
EOF
    ;;
  status)
    name=${2:-full-seed0}
    ssh "$host" "tail -n 20 $dir/runs/$name/train.log"
    ;;
  fetch)
    name=${2:-full-seed0}
    mkdir -p "$root/runs"
    rsync -az "$host:$dir/runs/$name" "$root/runs/"
    echo "fetched runs/$name; score it: uv run python -m tars_rl.train.eval --policy runs/$name"
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 2
    ;;
esac
