#!/usr/bin/env bash
set -euo pipefail

# Raspberry Pi Camera Module 3 autofocus support for the exact Jetson Linux
# release installed on this project device. The IMX708 sensor driver must
# already be installed; this adds the missing DW9817/DW9807 lens driver and DT.

EXPECTED_KERNEL="5.15.185-tegra"
EXPECTED_L4T_CORE="36.5.0-20260115194252"
EXPECTED_ARDUCAM="5.15.185-tegra-36.5.0-20260207171113"
EXPECTED_BOARD="nvidia,p3768-0000+p3767-0005-super"
SOURCE_SHA256="741f9bcb01e6afacc7325d4bff7cf00d1f6e202036d9b75b83a0c66216840e0e"
AF_LABEL="JetsonIO-IMX708-AF"
BASE_LABEL="JetsonIO"
OVERLAY_NAME="tegra234-p3767-imx708-a-autofocus.dtbo"
MODULE_NAME="dw9807-vcm.ko"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DTS="$SCRIPT_DIR/imx708_autofocus/tegra234-p3767-imx708-a-autofocus.dts"
VCM_SOURCE="$SCRIPT_DIR/imx708_autofocus/dw9807-vcm.c"
OVERLAY_DEST="/boot/arducam/dts/$OVERLAY_NAME"
MODULE_DEST="/lib/modules/$EXPECTED_KERNEL/updates/drivers/media/i2c/$MODULE_NAME"
EXTLINUX="/boot/extlinux/extlinux.conf"

usage() {
  cat <<'EOF'
Usage:
  scripts/install_imx708_autofocus.sh check
  sudo scripts/install_imx708_autofocus.sh install
  sudo scripts/install_imx708_autofocus.sh remove
  scripts/install_imx708_autofocus.sh verify

This installer is intentionally locked to JetPack 6.2.2 / Jetson Linux 36.5,
kernel 5.15.185-tegra, and an IMX708 camera connected to CAM A.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

package_version() {
  dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing command '$1'. Install the documented host dependencies first."
}

check_platform() {
  local kernel l4t arducam compatible vermagic
  kernel=$(uname -r)
  l4t=$(package_version nvidia-l4t-core)
  arducam=$(package_version arducam-nvidia-l4t-kernel)
  compatible=$(tr '\0' '\n' </proc/device-tree/compatible 2>/dev/null || true)

  [[ $(uname -m) == "aarch64" ]] || fail "This installer requires a Jetson aarch64 host."
  [[ $kernel == "$EXPECTED_KERNEL" ]] || fail "Kernel is '$kernel'; expected '$EXPECTED_KERNEL'. Rebuild for the running kernel instead of forcing this module."
  [[ $l4t == "$EXPECTED_L4T_CORE" ]] || fail "nvidia-l4t-core is '$l4t'; expected '$EXPECTED_L4T_CORE' (JetPack 6.2.2 / L4T 36.5)."
  grep -Fxq "$EXPECTED_BOARD" <<<"$compatible" || fail "Expected Orin Nano Super board compatibility '$EXPECTED_BOARD'."
  [[ $arducam == "$EXPECTED_ARDUCAM" ]] || fail "Arducam IMX708 package is '$arducam'; expected '$EXPECTED_ARDUCAM'."

  require_command dtc
  require_command fdtoverlay
  require_command gcc
  require_command make
  require_command modinfo
  require_command python3

  [[ -d /lib/modules/$kernel/build ]] || fail "Matching kernel headers are missing. Install nvidia-l4t-kernel-headers."
  [[ -f $DTS ]] || fail "Overlay source not found: $DTS"
  [[ -f $VCM_SOURCE ]] || fail "DW9807 source not found: $VCM_SOURCE"
  [[ -f $EXTLINUX ]] || fail "Boot configuration not found: $EXTLINUX"
  [[ -f /boot/arducam/Image ]] || fail "Arducam kernel image is missing."
  [[ -f /boot/arducam/dts/tegra234-p3767-camera-p3768-imx708-dual.dtbo ]] || fail "The active IMX708 base overlay is missing."

  vermagic=$(modinfo -F vermagic imx708 2>/dev/null || true)
  [[ $vermagic == "$EXPECTED_KERNEL "* ]] || fail "The installed imx708 module does not match '$EXPECTED_KERNEL': $vermagic"

  echo "Platform check passed: JetPack 6.2.2 / L4T 36.5, $kernel, Orin Nano Super."
  echo "IMX708 sensor package check passed: $arducam."
}

build_artifacts() {
  local build_dir=$1 actual_sha vermagic
  mkdir -p "$build_dir"

  actual_sha=$(sha256sum "$VCM_SOURCE" | awk '{print $1}')
  [[ $actual_sha == "$SOURCE_SHA256" ]] || fail "Unexpected DW9807 source checksum: $actual_sha"
  cp "$VCM_SOURCE" "$build_dir/dw9807-vcm.c"

  printf 'obj-m += dw9807-vcm.o\n' >"$build_dir/Makefile"
  make -C "/lib/modules/$EXPECTED_KERNEL/build" M="$build_dir" modules
  vermagic=$(modinfo -F vermagic "$build_dir/$MODULE_NAME")
  [[ $vermagic == "$EXPECTED_KERNEL "* ]] || fail "Built module ABI mismatch: $vermagic"

  dtc -@ -I dts -O dtb -o "$build_dir/$OVERLAY_NAME" "$DTS"
  fdtoverlay \
    -i /boot/arducam/dts/dtb/tegra234-p3768-0000+p3767-0005-nv-super.dtb \
    -o "$build_dir/merged-test.dtb" \
    /boot/arducam/dts/tegra234-p3767-camera-p3768-imx708-dual.dtbo \
    "$build_dir/$OVERLAY_NAME"

  echo "Module and overlay build passed."
}

configure_boot_entry() {
  local backup_dir=$1
  cp -a "$EXTLINUX" "$backup_dir/extlinux.conf"

  python3 - "$EXTLINUX" "$BASE_LABEL" "$AF_LABEL" "$OVERLAY_DEST" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
base_label = sys.argv[2]
af_label = sys.argv[3]
af_overlay = sys.argv[4]
lines = path.read_text().splitlines()

sections: dict[str, list[str]] = {}
starts: list[tuple[int, str]] = []
for index, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("LABEL "):
        starts.append((index, stripped.split(maxsplit=1)[1]))

for offset, (start, label) in enumerate(starts):
    end = starts[offset + 1][0] if offset + 1 < len(starts) else len(lines)
    sections[label] = lines[start:end]

if base_label not in sections:
    raise SystemExit(f"Boot label {base_label!r} was not found")

base = sections[base_label]
if not any(line.strip().startswith("OVERLAYS ") for line in base):
    raise SystemExit(f"Boot label {base_label!r} has no OVERLAYS line")

new_section: list[str] = []
for line in base:
    stripped = line.strip()
    indent = line[: len(line) - len(line.lstrip())]
    if stripped == f"LABEL {base_label}":
        new_section.append(f"LABEL {af_label}")
    elif stripped.startswith("MENU LABEL "):
        new_section.append(f"{indent}MENU LABEL IMX708 CAM A with autofocus")
    elif stripped.startswith("OVERLAYS "):
        # Jetson-IO/L4T 36.x serializes multiple OVERLAYS as a comma-separated
        # list (see /opt/nvidia/jetson-io/Jetson/board.py).
        overlays = [item for item in stripped.split(maxsplit=1)[1].split(",") if item]
        if af_overlay not in overlays:
            overlays.append(af_overlay)
        new_section.append(f"{indent}OVERLAYS {','.join(overlays)}")
    else:
        new_section.append(line)

out: list[str] = []
skip = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("DEFAULT "):
        out.append(f"DEFAULT {af_label}")
        continue
    if stripped == f"LABEL {af_label}":
        skip = True
        continue
    if skip and stripped.startswith("LABEL "):
        skip = False
    if not skip:
        out.append(line)

while out and not out[-1].strip():
    out.pop()
out.extend(["", *new_section])
path.write_text("\n".join(out).rstrip() + "\n")
PY
}

install_driver() {
  local build_dir backup_dir stamp
  [[ ${EUID} -eq 0 ]] || fail "Install requires root. Run: sudo $0 install"
  check_platform

  build_dir=$(mktemp -d /tmp/imx708-autofocus-build.XXXXXX)
  # Expand the local path while it is still in scope; EXIT runs after the
  # function returns and `set -u` would otherwise treat build_dir as unset.
  trap "rm -rf '$build_dir'" EXIT
  build_artifacts "$build_dir"

  stamp=$(date +%Y%m%d-%H%M%S)
  backup_dir="/boot/imx708-autofocus-backup-$stamp"
  mkdir -p "$backup_dir"
  [[ ! -e $MODULE_DEST ]] || cp -a "$MODULE_DEST" "$backup_dir/"
  [[ ! -e $OVERLAY_DEST ]] || cp -a "$OVERLAY_DEST" "$backup_dir/"

  install -D -o root -g root -m 0644 "$build_dir/$MODULE_NAME" "$MODULE_DEST"
  install -D -o root -g root -m 0644 "$build_dir/$OVERLAY_NAME" "$OVERLAY_DEST"
  depmod "$EXPECTED_KERNEL"
  configure_boot_entry "$backup_dir"

  echo
  echo "Installed the exact-kernel lens module and CAM A autofocus overlay."
  echo "Backup: $backup_dir"
  echo "New default boot entry: $AF_LABEL"
  echo "Fallback boot entry retained: $BASE_LABEL"
  echo "NOTE: JetPack 6.2.2 Argus does not expose this IMX708 lens as a focus control."
  echo "This installs experimental VCM plumbing; it does not enable Argus autofocus."
  echo "Reboot is required; this script does not reboot automatically."
}

remove_driver() {
  [[ ${EUID} -eq 0 ]] || fail "Remove requires root. Run: sudo $0 remove"
  [[ -f $EXTLINUX ]] || fail "Boot configuration not found: $EXTLINUX"
  cp -a "$EXTLINUX" "$EXTLINUX.pre-imx708-af-remove-$(date +%Y%m%d-%H%M%S)"

  python3 - "$EXTLINUX" "$BASE_LABEL" "$AF_LABEL" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
base_label = sys.argv[2]
af_label = sys.argv[3]
lines = path.read_text().splitlines()
out = []
skip = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("DEFAULT "):
        out.append(f"DEFAULT {base_label}")
        continue
    if stripped == f"LABEL {af_label}":
        skip = True
        continue
    if skip and stripped.startswith("LABEL "):
        skip = False
    if not skip:
        out.append(line)
path.write_text("\n".join(out).rstrip() + "\n")
PY

  rm -f "$MODULE_DEST" "$OVERLAY_DEST"
  depmod "$EXPECTED_KERNEL"
  echo "Autofocus files removed and default boot entry restored to $BASE_LABEL. Reboot required."
}

verify_driver() {
  check_platform
  echo
  echo "Installed module:"
  modinfo dw9807_vcm 2>/dev/null | grep -E '^(filename|description|alias|vermagic):' || true
  echo
  echo "Live device-tree lens node:"
  if [[ -d /sys/firmware/devicetree/base/bus@0/cam_i2cmux/i2c@0/dw9817@c ]]; then
    echo "present"
  else
    echo "not present (install and reboot first)"
  fi
  echo
  echo "Camera and lens media nodes:"
  ls -l /dev/video* /dev/v4l-subdev* 2>/dev/null || true
  echo
  echo "Recent IMX708/DW9807 kernel messages:"
  journalctl -k -b --no-pager 2>/dev/null | grep -Ei 'imx708|dw9807|dw9817' | tail -40 || true
}

action=${1:-}
case "$action" in
  check)
    check_platform
    build_dir=$(mktemp -d /tmp/imx708-autofocus-check.XXXXXX)
    trap 'rm -rf "$build_dir"' EXIT
    build_artifacts "$build_dir"
    ;;
  install) install_driver ;;
  remove) remove_driver ;;
  verify) verify_driver ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
