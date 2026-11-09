#!/usr/bin/env bash
# Build the rig image on this Mac and run it on the Pi.
#
#   rig/deploy.sh            build, send to the Pi, replace the running container
#   rig/deploy.sh --force    the same, even while a session is recording (it is stopped cleanly)
#   rig/deploy.sh --build    build only (checks the Dockerfile without touching the Pi)
#   rig/deploy.sh --save     send the whole image with docker save | docker load (no registry)
#
# The image goes through a registry container on this Mac, reached from the Pi over an SSH reverse
# tunnel, so the Pi pulls only the layers it lacks: the ~700 MB camera-stack layer once, then about
# 400 kB per code change. Docker trusts registries on localhost without TLS, so the Pi needs no
# settings. RIG_HOST overrides the SSH target. The Pi needs Docker (rig/pi_setup.sh).
set -euo pipefail

HOST=${RIG_HOST:-axel@tars-rig.local}
IMAGE=tars-rig:latest
PORT=5055  # registry port, on localhost at both ends of the tunnel
REGISTRY_IMAGE=registry:2.8.3
MODE=${1:-}
cd "$(dirname "$0")"

if ! docker info >/dev/null 2>&1; then
  echo "starting colima (the Docker VM on this Mac)"
  colima start --cpu 4 --memory 6
fi

docker build --platform linux/arm64 -t "$IMAGE" .
[[ $MODE == --build ]] && exit 0

if [[ $MODE != --force ]] && ssh "$HOST" 'docker exec tars-rig rig status 2>/dev/null' | grep -q 'state: recording'; then
  echo "a session is recording on the Pi; stop it (scripts/rig stop) or pass --force" >&2
  exit 1
fi

if [[ $MODE == --save ]]; then
  echo "copying the whole image to $HOST"
  docker save "$IMAGE" | gzip -1 | ssh "$HOST" 'gunzip | docker load'
else
  if ! docker ps --format '{{.Names}}' | grep -qx tars-rig-registry; then
    docker run -d --rm --name tars-rig-registry -p "127.0.0.1:$PORT:5000" "$REGISTRY_IMAGE" >/dev/null
  fi
  docker tag "$IMAGE" "localhost:$PORT/$IMAGE"
  docker push -q "localhost:$PORT/$IMAGE" >/dev/null
  echo "pulling on $HOST (only new layers travel)"
  ssh -o ExitOnForwardFailure=yes -R "$PORT:localhost:$PORT" "$HOST" \
    "docker pull -q localhost:$PORT/$IMAGE >/dev/null && docker tag localhost:$PORT/$IMAGE $IMAGE"
fi

# --privileged + /dev: cameras (/dev/media*, /dev/video*, /dev/dma_heap, the sensor's
# v4l-subdev for line timing), I2C and GPIO chips. /run/udev: libcamera finds the cameras through
# udev. --init: forwards SIGTERM to the daemon, which then stops a running session and writes its
# done.json (hence the long stop timeout).
ssh "$HOST" "docker rm -f tars-rig >/dev/null 2>&1 || true
  docker run -d --name tars-rig --restart unless-stopped --init --stop-timeout 45 \
    --privileged --network host -v /dev:/dev -v /run/udev:/run/udev:ro -v /srv/rig:/data \
    $IMAGE >/dev/null
  docker image prune -f >/dev/null
  sleep 2; docker exec tars-rig rig status"
