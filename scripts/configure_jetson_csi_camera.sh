#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run with sudo: sudo $0 [imx219|imx477] [A|C|dual]" >&2
  exit 1
fi

SENSOR=${1:-imx219}
PORT=${2:-A}

case "$SENSOR:$PORT" in
  imx219:A) OVERLAY=/boot/tegra234-p3767-camera-p3768-imx219-A.dtbo ;;
  imx219:C) OVERLAY=/boot/tegra234-p3767-camera-p3768-imx219-C.dtbo ;;
  imx219:dual) OVERLAY=/boot/tegra234-p3767-camera-p3768-imx219-dual.dtbo ;;
  imx477:A) OVERLAY=/boot/tegra234-p3767-camera-p3768-imx477-A.dtbo ;;
  imx477:C) OVERLAY=/boot/tegra234-p3767-camera-p3768-imx477-C.dtbo ;;
  imx477:dual) OVERLAY=/boot/tegra234-p3767-camera-p3768-imx477-dual.dtbo ;;
  *)
    echo "Unsupported camera selection: $SENSOR $PORT" >&2
    echo "Supported: imx219 A, imx219 C, imx219 dual, imx477 A, imx477 C, imx477 dual" >&2
    exit 1
    ;;
esac

if [[ ! -f "$OVERLAY" ]]; then
  echo "Overlay not found: $OVERLAY" >&2
  exit 1
fi

CONF=/boot/extlinux/extlinux.conf
DTB=/boot/tegra234-p3768-0000+p3767-0005-nv.dtb
LABEL="camera-${SENSOR}-${PORT}"
BACKUP="$CONF.pre-camera-$(date +%Y%m%d-%H%M%S)"
cp "$CONF" "$BACKUP"

python3 - "$CONF" "$DTB" "$OVERLAY" "$LABEL" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

conf_path = Path(sys.argv[1])
dtb = sys.argv[2]
overlay = sys.argv[3]
label = sys.argv[4]
append_line = None
linux_line = None
initrd_line = None

lines = conf_path.read_text().splitlines()
default_label = None
for line in lines:
    stripped = line.strip()
    if stripped.startswith("DEFAULT "):
        default_label = stripped.split(maxsplit=1)[1]
        break
if default_label is None:
    raise SystemExit("Could not find DEFAULT entry in boot config")

inside_default = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("LABEL "):
        inside_default = stripped.split(maxsplit=1)[1] == default_label
        continue
    if not inside_default:
        continue
    if stripped.startswith("APPEND "):
        append_line = "      " + stripped
    elif stripped.startswith("LINUX "):
        linux_line = "      " + stripped
    elif stripped.startswith("INITRD "):
        initrd_line = "      " + stripped

missing = [
    name
    for name, value in (("LINUX", linux_line), ("INITRD", initrd_line), ("APPEND", append_line))
    if value is None
]
if missing:
    raise SystemExit(
        f"Default boot entry {default_label!r} is missing: {', '.join(missing)}"
    )

out: list[str] = []
skip = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("DEFAULT "):
        out.append(f"DEFAULT {label}")
        continue
    if stripped == f"LABEL {label}":
        skip = True
        continue
    if skip and stripped.startswith("LABEL "):
        skip = False
    if skip:
        continue
    out.append(line)

while out and not out[-1].strip():
    out.pop()

out.extend([
    "",
    f"LABEL {label}",
    f"      MENU LABEL Camera {label} kernel",
    linux_line,
    f"      FDT {dtb}",
    initrd_line,
    append_line,
    f"      OVERLAYS {overlay}",
])

conf_path.write_text("\n".join(out) + "\n")
print(f"Configured DEFAULT {label} with OVERLAYS {overlay}")
PY

echo "Backed up previous boot config to $BACKUP"
echo "Reboot required. After reboot, test with:"
echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink"
