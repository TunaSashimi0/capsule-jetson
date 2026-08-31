#!/usr/bin/env bash
set -euo pipefail

# Configure both Orin Nano CSI connectors for native Sony IMX219 sensors.
# The existing boot entries are retained as recovery choices.

EXPECTED_KERNEL="5.15.185-tegra"
EXPECTED_L4T="36.5.0-20260115194252"
EXPECTED_BOARD="nvidia,p3768-0000+p3767-0005-super"
ENTRY_LABEL="JetsonIO-IMX219-DUAL"
OVERLAY="/boot/arducam/dts/tegra234-p3767-camera-p3768-imx219-dual.dtbo"
FDT="/boot/arducam/dts/dtb/tegra234-p3768-0000+p3767-0005-nv-super.dtb"
KERNEL="/boot/arducam/Image"
EXTLINUX="/boot/extlinux/extlinux.conf"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

package_version() {
  dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true
}

check_platform() {
  local compatible
  [[ $(uname -m) == "aarch64" ]] || fail "This configurator must run on the Jetson."
  [[ $(uname -r) == "$EXPECTED_KERNEL" ]] || fail "Running kernel is '$(uname -r)', expected '$EXPECTED_KERNEL'."
  [[ $(package_version nvidia-l4t-core) == "$EXPECTED_L4T" ]] || fail "This script is locked to L4T $EXPECTED_L4T."
  compatible=$(tr '\0' '\n' </proc/device-tree/compatible 2>/dev/null || true)
  grep -Fxq "$EXPECTED_BOARD" <<<"$compatible" || fail "Expected Orin Nano Super compatibility '$EXPECTED_BOARD'."
  [[ -f $KERNEL ]] || fail "Arducam kernel image is missing: $KERNEL"
  [[ -f $FDT ]] || fail "Base device tree is missing: $FDT"
  [[ -f $OVERLAY ]] || fail "IMX219 dual overlay is missing: $OVERLAY"
  [[ -f $EXTLINUX ]] || fail "Boot configuration is missing: $EXTLINUX"

  modinfo nv_imx219 >/dev/null 2>&1 || fail "The nv_imx219 driver is not installed for the running kernel."
  [[ $(modinfo -F vermagic nv_imx219) == "$EXPECTED_KERNEL "* ]] || fail "The nv_imx219 driver ABI does not match the running kernel."

  # Validate that the overlay can be applied to this exact base DT before
  # touching the boot configuration.
  local decoded merged
  merged=$(mktemp /tmp/imx219-dual-merged.XXXXXX.dtb)
  decoded=$(mktemp /tmp/imx219-dual-merged.XXXXXX.dts)
  trap 'rm -f "$merged" "$decoded"; trap - RETURN' RETURN
  fdtoverlay -i "$FDT" -o "$merged" "$OVERLAY"
  dtc -I dtb -O dts -o "$decoded" "$merged" 2>/dev/null
  grep -q 'rbpcv2_imx219_a@10' "$decoded" || fail "Merged DT lacks CAM A IMX219 node."
  grep -q 'rbpcv2_imx219_c@10' "$decoded" || fail "Merged DT lacks CAM C IMX219 node."

  echo "Platform check passed: Orin Nano Super, L4T 36.5, kernel $EXPECTED_KERNEL."
  echo "Dual IMX219 overlay and nv_imx219 module check passed."
}

install_config() {
  [[ $EUID -eq 0 ]] || fail "Installation requires root: sudo $0 install"
  check_platform

  local backup_dir stamp
  stamp=$(date +%Y%m%d-%H%M%S)
  backup_dir="/boot/imx219-dual-backup-$stamp"
  mkdir -p "$backup_dir"
  cp -a "$EXTLINUX" "$backup_dir/extlinux.conf"

  python3 - "$EXTLINUX" "$ENTRY_LABEL" "$KERNEL" "$FDT" "$OVERLAY" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
label, kernel, fdt, overlay = sys.argv[2:]
lines = path.read_text().splitlines()

# Clone the APPEND arguments from the current default entry. Root devices and
# boot arguments are machine-specific and must never be synthesized here.
default_label = None
for line in lines:
    stripped = line.strip()
    if stripped.startswith("DEFAULT "):
        default_label = stripped.split(maxsplit=1)[1]
        break
if default_label is None:
    raise SystemExit("extlinux.conf has no DEFAULT line")

append_args = None
inside_default = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("LABEL "):
        inside_default = stripped.split(maxsplit=1)[1] == default_label
        continue
    if inside_default and stripped.startswith("APPEND "):
        append_args = stripped.split(maxsplit=1)[1]
        break
if append_args is None:
    raise SystemExit(f"default boot entry {default_label!r} has no APPEND line")

# Remove an older copy of our managed section, if one exists.
out: list[str] = []
skip = False
for line in lines:
    stripped = line.strip()
    if stripped == f"LABEL {label}":
        skip = True
        continue
    if skip and stripped.startswith("LABEL "):
        skip = False
    if not skip:
        out.append(line)

# Switch only DEFAULT; every pre-existing entry remains available in the menu.
for index, line in enumerate(out):
    if line.strip().startswith("DEFAULT "):
        out[index] = f"DEFAULT {label}"
        break
else:
    raise SystemExit("extlinux.conf has no DEFAULT line")

while out and not out[-1].strip():
    out.pop()
out.extend(
    [
        "",
        f"LABEL {label}",
        "\tMENU LABEL Arducam IMX219-AF dual (CAM A + CAM C)",
        f"\tLINUX {kernel}",
        f"\tFDT {fdt}",
        "\tINITRD /boot/initrd",
        f"\tAPPEND {append_args}",
        f"\tOVERLAYS {overlay}",
    ]
)
path.write_text("\n".join(out).rstrip() + "\n")
PY

  grep -Fxq "DEFAULT $ENTRY_LABEL" "$EXTLINUX" || fail "Failed to set the new default boot entry."
  grep -Fq "OVERLAYS $OVERLAY" "$EXTLINUX" || fail "Failed to install the IMX219 overlay entry."

  echo "Installed default boot entry: $ENTRY_LABEL"
  echo "Backup: $backup_dir/extlinux.conf"
  echo "Existing IMX708 and primary entries were retained as fallbacks."
  echo "Reboot is required."
}

verify_config() {
  check_platform
  echo
  grep -E '^(DEFAULT|LABEL)|^[[:space:]]+(MENU LABEL|OVERLAYS)' "$EXTLINUX"
  echo
  if grep -Fxq "DEFAULT $ENTRY_LABEL" "$EXTLINUX"; then
    echo "The dual IMX219 entry is configured as default."
  else
    fail "The dual IMX219 entry is not the default."
  fi
}

case ${1:-} in
  check) check_platform ;;
  install) install_config ;;
  verify) verify_config ;;
  *)
    echo "Usage: $0 {check|install|verify}" >&2
    exit 2
    ;;
esac
