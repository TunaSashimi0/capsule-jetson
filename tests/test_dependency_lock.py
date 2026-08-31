from __future__ import annotations

import unittest
from pathlib import Path

from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SharedDependencyLockTests(unittest.TestCase):
    def test_every_shared_dependency_is_exactly_pinned(self) -> None:
        lines = (
            PROJECT_ROOT.joinpath("requirements.shared.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        requirements = [
            Requirement(line.strip())
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertGreater(len(requirements), 0)
        for requirement in requirements:
            specifiers = list(requirement.specifier)
            self.assertEqual(len(specifiers), 1, requirement.name)
            self.assertEqual(specifiers[0].operator, "==", requirement.name)

    def test_platform_owned_packages_are_not_in_shared_lock(self) -> None:
        contents = PROJECT_ROOT.joinpath("requirements.shared.txt").read_text(
            encoding="utf-8"
        )

        for package in ("torch", "torchvision", "opencv-python"):
            self.assertNotIn(f"{package}==", contents.lower())


if __name__ == "__main__":
    unittest.main()
