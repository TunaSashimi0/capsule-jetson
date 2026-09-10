#!/usr/bin/env python3
"""Continuously pulse the eight outputs on an Adafruit I2C solenoid driver.

This is a supervised hardware test, not the inference-controlled production
sequence.  It owns MCP23017 port A (the board outputs labelled 0/A0 through
7/A7) while it is running and leaves every port inactive when it exits.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path
from typing import Protocol, Sequence


DEFAULT_BUS = 7
DEFAULT_ADDRESS = 0x20
DEFAULT_PORTS = tuple(range(8))


class MCP23017PortA(Protocol):
    gpioa: int
    iodira: int


def parse_int(value: str) -> int:
    """Accept decimal or Python-style hexadecimal integers."""
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc


def parse_ports(value: str) -> tuple[int, ...]:
    """Parse an ordered list such as ``0-7``, ``0,2,4``, or ``7-0``."""
    ports: list[int] = []
    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                raise ValueError
            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start, end = int(start_text), int(end_text)
                step = 1 if end >= start else -1
                ports.extend(range(start, end + step, step))
            else:
                ports.append(int(item))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ports must be a comma-separated list or range, for example 0-7 or 0,2,4"
        ) from exc

    if not ports or any(port not in DEFAULT_PORTS for port in ports):
        raise argparse.ArgumentTypeError("solenoid ports must be between 0 and 7")
    if len(set(ports)) != len(ports):
        raise argparse.ArgumentTypeError("solenoid ports must not be repeated")
    return tuple(ports)


class PortCycler:
    """Drive exactly one MCP23017 port-A output at a time."""

    def __init__(self, mcp: MCP23017PortA, *, active_high: bool) -> None:
        self._mcp = mcp
        self._active_high = active_high
        self._inactive = 0x00 if active_high else 0xFF

        # Load the inactive output latch before changing the direction.  This
        # prevents an activation glitch when port A changes from inputs to
        # outputs.  Port B is deliberately left untouched.
        self._mcp.gpioa = self._inactive
        self._mcp.iodira = 0x00
        self.all_off()

    def activate(self, port: int) -> None:
        if port not in DEFAULT_PORTS:
            raise ValueError(f"solenoid port {port} is outside 0..7")
        bit = 1 << port
        self._mcp.gpioa = self._inactive | bit if self._active_high else self._inactive & ~bit

    def all_off(self) -> None:
        self._mcp.gpioa = self._inactive


def cycle_ports(
    cycler: PortCycler,
    ports: Sequence[int],
    *,
    on_seconds: float,
    gap_seconds: float,
    cycles: int,
    stop_event: threading.Event,
) -> None:
    completed = 0
    while not stop_event.is_set() and (cycles == 0 or completed < cycles):
        print(f"Starting pass {completed + 1}", flush=True)
        for port in ports:
            if stop_event.is_set():
                break
            print(f"  port {port}: ON", flush=True)
            cycler.activate(port)
            if stop_event.wait(on_seconds):
                cycler.all_off()
                break
            cycler.all_off()
            print(f"  port {port}: off", flush=True)
            if stop_event.wait(gap_seconds):
                break
        completed += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pulse Adafruit 8-channel solenoid outputs one at a time until stopped."
    )
    parser.add_argument("--bus", type=int, default=DEFAULT_BUS, help="Linux I2C bus (default: 7)")
    parser.add_argument(
        "--address",
        type=parse_int,
        default=DEFAULT_ADDRESS,
        help="MCP23017 I2C address (default: 0x20)",
    )
    parser.add_argument(
        "--ports",
        type=parse_ports,
        default=DEFAULT_PORTS,
        help="ordered ports/ranges to cycle (default: 0-7)",
    )
    parser.add_argument(
        "--on-seconds", type=float, default=0.2, help="energized time per port (default: 0.2)"
    )
    parser.add_argument(
        "--gap-seconds", type=float, default=0.5, help="all-off time between ports (default: 0.5)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="number of complete passes; 0 runs until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--active-low",
        action="store_true",
        help="use only if the attached driver energizes on a low signal",
    )
    parser.add_argument(
        "--energize",
        action="store_true",
        help="required acknowledgement that this command will move connected solenoids",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.bus < 0:
        parser.error("--bus must be non-negative")
    if not 0x08 <= args.address <= 0x77:
        parser.error("--address must be a valid 7-bit I2C address (0x08..0x77)")
    if args.on_seconds <= 0:
        parser.error("--on-seconds must be positive")
    if args.gap_seconds < 0:
        parser.error("--gap-seconds cannot be negative")
    if args.cycles < 0:
        parser.error("--cycles cannot be negative")
    if not args.energize:
        parser.error("refusing to activate hardware without --energize")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    device = Path(f"/dev/i2c-{args.bus}")
    if not device.exists():
        parser.error(f"{device} does not exist; verify the Jetson header bus number")

    try:
        from adafruit_extended_bus import ExtendedI2C
        from adafruit_mcp230xx.mcp23017 import MCP23017
    except (ImportError, RuntimeError) as exc:
        parser.error(
            "could not load the Adafruit I2C libraries; install the project requirements "
            f"and run with I2C permissions (often via sudo): {exc}"
        )

    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    i2c = None
    cycler = None
    try:
        i2c = ExtendedI2C(args.bus)
        # reset=False avoids modifying the unused MCP23017 port B configuration.
        mcp = MCP23017(i2c, address=args.address, reset=False)
        cycler = PortCycler(mcp, active_high=not args.active_low)
        cycle_label = "forever" if args.cycles == 0 else str(args.cycles)
        print(
            f"Cycling ports {','.join(map(str, args.ports))} on {device} at "
            f"0x{args.address:02x} ({cycle_label} passes). Press Ctrl+C to stop.",
            flush=True,
        )
        cycle_ports(
            cycler,
            args.ports,
            on_seconds=args.on_seconds,
            gap_seconds=args.gap_seconds,
            cycles=args.cycles,
            stop_event=stop_event,
        )
        return 0
    except OSError as exc:
        print(f"ERROR: I2C access failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if cycler is not None:
            try:
                cycler.all_off()
                print("All solenoid ports are off.", flush=True)
            except Exception as exc:  # Cleanup must not hide an earlier I2C failure.
                print(f"WARNING: failed to switch all ports off: {exc}", file=sys.stderr)
        if i2c is not None:
            deinit = getattr(i2c, "deinit", None)
            if callable(deinit):
                try:
                    deinit()
                except Exception as exc:
                    print(f"WARNING: failed to close the I2C bus: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
