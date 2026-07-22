#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${1:-.env.jetson}
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

IMAGE_NAME=${IMAGE_NAME:-yolo-capsule:jetson}
CONTAINER_NAME=${CONTAINER_NAME:-yolo-capsule}
CAPSULE_MODEL=${CAPSULE_MODEL:-/app/models/trained/capsule_yolo11s_obb_best.pt}
CAPSULE_SOURCE=${CAPSULE_SOURCE:-0}
CAPSULE_DEVICE=${CAPSULE_DEVICE:-0}
CAPSULE_IMGSZ=${CAPSULE_IMGSZ:-1280}
CAPSULE_CAPTURE_WIDTH=${CAPSULE_CAPTURE_WIDTH:-1920}
CAPSULE_CAPTURE_HEIGHT=${CAPSULE_CAPTURE_HEIGHT:-1080}
CAPSULE_CAPTURE_FPS=${CAPSULE_CAPTURE_FPS:-30}
CAPSULE_JPEG_QUALITY=${CAPSULE_JPEG_QUALITY:-95}
CAPSULE_EXPOSURE_US=${CAPSULE_EXPOSURE_US:-8000}
CAPSULE_ANALOG_GAIN=${CAPSULE_ANALOG_GAIN:-1.0}
CAPSULE_DIGITAL_GAIN=${CAPSULE_DIGITAL_GAIN:-1.0}
CAPSULE_HALF=${CAPSULE_HALF:-true}
CAPSULE_CONF=${CAPSULE_CONF:-0.25}
CAPSULE_IOU=${CAPSULE_IOU:-0.7}
CAPSULE_CAMERA_DEVICE=${CAPSULE_CAMERA_DEVICE:-/dev/video${CAPSULE_SOURCE}}

DEVICE_ARGS=()
ARGUS_ARGS=()
GROUP_ARGS=()
if [[ -e /tmp/argus_socket ]]; then
  ARGUS_ARGS=(-v /tmp/argus_socket:/tmp/argus_socket)
fi

for group_name in video render; do
  group_entry=$(getent group "$group_name" || true)
  if [[ -n "$group_entry" ]]; then
    IFS=: read -r _ _ group_id _ <<< "$group_entry"
    GROUP_ARGS+=(--group-add "$group_id")
  fi
done

if [[ "$CAPSULE_SOURCE" =~ ^[0-9]+$ ]]; then
  if [[ ! -e "$CAPSULE_CAMERA_DEVICE" ]]; then
    echo "Camera device not found: $CAPSULE_CAMERA_DEVICE" >&2
    echo "Set CAPSULE_CAMERA_DEVICE in $ENV_FILE, or set CAPSULE_SOURCE to a mounted video file." >&2
    exit 1
  fi
  DEVICE_ARGS=(--device "$CAPSULE_CAMERA_DEVICE:$CAPSULE_CAMERA_DEVICE")
fi

docker run --rm -it \
  --name "$CONTAINER_NAME" \
  --runtime nvidia \
  --network host \
  --ipc host \
  "${DEVICE_ARGS[@]}" \
  "${ARGUS_ARGS[@]}" \
  "${GROUP_ARGS[@]}" \
  -e CAPSULE_MODEL="$CAPSULE_MODEL" \
  -e CAPSULE_SOURCE="$CAPSULE_SOURCE" \
  -e CAPSULE_DEVICE="$CAPSULE_DEVICE" \
  -e CAPSULE_IMGSZ="$CAPSULE_IMGSZ" \
  -e CAPSULE_CAPTURE_WIDTH="$CAPSULE_CAPTURE_WIDTH" \
  -e CAPSULE_CAPTURE_HEIGHT="$CAPSULE_CAPTURE_HEIGHT" \
  -e CAPSULE_CAPTURE_FPS="$CAPSULE_CAPTURE_FPS" \
  -e CAPSULE_JPEG_QUALITY="$CAPSULE_JPEG_QUALITY" \
  -e CAPSULE_EXPOSURE_US="$CAPSULE_EXPOSURE_US" \
  -e CAPSULE_ANALOG_GAIN="$CAPSULE_ANALOG_GAIN" \
  -e CAPSULE_DIGITAL_GAIN="$CAPSULE_DIGITAL_GAIN" \
  -e CAPSULE_HALF="$CAPSULE_HALF" \
  -e CAPSULE_CONF="$CAPSULE_CONF" \
  -e CAPSULE_IOU="$CAPSULE_IOU" \
  -e YOLO_CONFIG_DIR=/tmp/ultralytics \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -v "$PWD/src:/app/src:ro" \
  -v "$PWD/configs:/app/configs:ro" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/runs:/app/runs" \
  "$IMAGE_NAME"
