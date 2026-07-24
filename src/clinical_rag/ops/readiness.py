"""Readiness helpers for public workflow entrypoints."""

from __future__ import annotations

import argparse
import sys


def not_implemented_message(target: str, intended_command: str) -> str:
    return (
        f"NOT IMPLEMENTED: make {target} is advertised but not wired yet.\n"
        f"Intended command: {intended_command}\n"
        "This target exits non-zero so automation cannot mistake a placeholder "
        "for a completed clinical evaluation workflow."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail loudly for planned clinical RAG workflows."
    )
    parser.add_argument("target", help="Make target name.")
    parser.add_argument("intended_command", help="Command planned for the wired workflow.")
    args = parser.parse_args(argv)
    print(not_implemented_message(args.target, args.intended_command), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
