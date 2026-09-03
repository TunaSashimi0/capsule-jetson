from __future__ import annotations

import argparse
import json

from .client import OceanMesClient, OceanMesError
from .settings import OceanMesConfigurationError, OceanMesSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCEANMES edge connectivity tools")
    parser.add_argument(
        "command",
        choices=("config",),
        help="Fetch and print the authoritative device configuration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = OceanMesSettings.from_env()
        with OceanMesClient(settings) as client:
            if args.command == "config":
                print(json.dumps(client.get_configuration().__dict__, indent=2))
                return 0
    except (OceanMesConfigurationError, OceanMesError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
