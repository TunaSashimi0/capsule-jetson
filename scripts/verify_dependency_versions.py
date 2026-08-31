#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
from pathlib import Path

from packaging.requirements import Requirement


def locked_requirements(path: Path) -> list[Requirement]:
    requirements: list[Requirement] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise ValueError(f"{path}:{line_number} is not pinned with exactly one ==")
        requirements.append(requirement)
    return requirements


def verify(path: Path) -> list[str]:
    errors: list[str] = []
    for requirement in locked_requirements(path):
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f"{requirement.name}: missing; expected {requirement.specifier}")
            continue
        if installed not in requirement.specifier:
            errors.append(
                f"{requirement.name}: installed {installed}; expected {requirement.specifier}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify exact versions from a shared requirements lock."
    )
    parser.add_argument("requirements", type=Path)
    args = parser.parse_args()

    errors = verify(args.requirements)
    if errors:
        print("Dependency version verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Verified {len(locked_requirements(args.requirements))} shared dependencies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
