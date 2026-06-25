#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${1:-.env.jetson}
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

: "${JETSON_BASE_IMAGE:?Set JETSON_BASE_IMAGE in $ENV_FILE to an NVIDIA l4t-pytorch image matching this Jetson L4T release}"
if [[ "$JETSON_BASE_IMAGE" == REPLACE_WITH_* ]]; then
  echo "JETSON_BASE_IMAGE still contains the placeholder value in $ENV_FILE" >&2
  echo "Set it to an NVIDIA l4t-pytorch tag matching /etc/nv_tegra_release." >&2
  exit 1
fi
IMAGE_NAME=${IMAGE_NAME:-yolo-capsule:jetson}

docker build \
  --build-arg BASE_IMAGE="$JETSON_BASE_IMAGE" \
  -f Dockerfile.jetson \
  -t "$IMAGE_NAME" \
  .
