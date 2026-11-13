# tars_live_vio: Basalt on a live TCP stream, for tars_scan.live_map on the Pi (rig/deploy.sh runs it
# as container tars-live-vio, host network, listening on 127.0.0.1:5077). The binary is our patched
# Basalt (~/dev/slam/basalt, src/tars_live_vio.cpp), built for the Pi 5 by ~/dev/slam/build-basalt.sh;
# rig/live_vio/build.sh copies it out of the build volume next to this file.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y \
    libopencv-imgcodecs406t64 libopencv-features2d406t64 libtbb12 libfmt9 liblz4-1 libepoxy0 \
    libyaml-cpp0.8 libsqlite3-0 libegl1 libgl1 libglx0 && rm -rf /var/lib/apt/lists/*
COPY tars_live_vio /usr/local/bin/tars_live_vio
# the stereo-hypothesis patch, as on walk2 (runs/night/night_log.md)
ENV TARS_STEREO_DEPTHS=0.25,0.4,0.65,1.0,2,5 TARS_STEREO_AMBIGUITY=1.6
ENTRYPOINT ["/usr/local/bin/tars_live_vio"]
CMD ["--port", "5077", "--num-threads", "3"]
