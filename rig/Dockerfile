# The rig's container for the Raspberry Pi 5. Built on the Mac (Apple Silicon is arm64 too, so no
# emulation) by rig/deploy.sh, which streams the image to the Pi over SSH.
#
# Camera stack (libcamera with the Pi's PiSP pipeline, picamera2) and GPIO (gpiozero, lgpio) come
# from the Raspberry Pi apt archive, the same packages Raspberry Pi OS uses. The suite must match the
# Pi's OS release, since libcamera talks to the Pi kernel's camera drivers.
FROM debian:trixie-slim

ARG RPI_SUITE=trixie
# The keyring package, not the old raspberrypi.gpg.key: trixie's apt rejects that key's SHA1 binding.
ARG RPI_KEYRING=2025.1+rpt1
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSLo /tmp/keyring.deb "https://archive.raspberrypi.com/debian/pool/main/r/raspberrypi-archive-keyring/raspberrypi-archive-keyring_${RPI_KEYRING}_all.deb" \
 && dpkg -i /tmp/keyring.deb && rm /tmp/keyring.deb \
 && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.pgp] http://archive.raspberrypi.com/debian ${RPI_SUITE} main" \
    > /etc/apt/sources.list.d/raspi.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    python3-picamera2 python3-libcamera libcamera-ipa python3-simplejpeg python3-numpy \
    python3-gpiozero python3-lgpio python3-pip i2c-tools v4l-utils \
 && pip install --break-system-packages --no-cache-dir smbus2==0.5.0 \
 && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY config.toml /app/config.toml
COPY src /app/src
RUN printf '#!/bin/sh\nexec python3 -m tars_rig "$@"\n' > /usr/local/bin/rig \
 && chmod +x /usr/local/bin/rig \
 && python3 -c "import picamera2, gpiozero, lgpio, smbus2, simplejpeg, tars_rig.cli"

CMD ["rig", "daemon"]
