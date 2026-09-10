from __future__ import annotations

import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO

from scripts.test_solenoid_ports import PortCycler, cycle_ports, parse_ports


class FakeMCP23017:
    def __init__(self) -> None:
        self.writes: list[tuple[str, int]] = []
        self._gpioa = 0
        self._iodira = 0xFF

    @property
    def gpioa(self) -> int:
        return self._gpioa

    @gpioa.setter
    def gpioa(self, value: int) -> None:
        self._gpioa = value
        self.writes.append(("gpioa", value))

    @property
    def iodira(self) -> int:
        return self._iodira

    @iodira.setter
    def iodira(self, value: int) -> None:
        self._iodira = value
        self.writes.append(("iodira", value))


class SolenoidPortScriptTests(unittest.TestCase):
    def test_port_ranges_preserve_requested_order(self) -> None:
        self.assertEqual(parse_ports("0-2,7,5-4"), (0, 1, 2, 7, 5, 4))

    def test_active_high_cycle_energizes_only_one_port(self) -> None:
        mcp = FakeMCP23017()
        cycler = PortCycler(mcp, active_high=True)

        with redirect_stdout(StringIO()):
            cycle_ports(
                cycler,
                (0, 2),
                on_seconds=0.001,
                gap_seconds=0,
                cycles=1,
                stop_event=threading.Event(),
            )

        self.assertEqual(
            mcp.writes,
            [
                ("gpioa", 0x00),
                ("iodira", 0x00),
                ("gpioa", 0x00),
                ("gpioa", 0x01),
                ("gpioa", 0x00),
                ("gpioa", 0x04),
                ("gpioa", 0x00),
            ],
        )

    def test_active_low_uses_high_as_the_inactive_level(self) -> None:
        mcp = FakeMCP23017()
        cycler = PortCycler(mcp, active_high=False)
        cycler.activate(3)
        cycler.all_off()

        self.assertEqual(
            mcp.writes,
            [
                ("gpioa", 0xFF),
                ("iodira", 0x00),
                ("gpioa", 0xFF),
                ("gpioa", 0xF7),
                ("gpioa", 0xFF),
            ],
        )


if __name__ == "__main__":
    unittest.main()
