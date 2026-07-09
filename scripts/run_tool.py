"""Manual invocation CLI for NASA tools.

Usage: make tool name=apod args='{"date": "2026-07-01"}'
"""

from __future__ import annotations

import asyncio
import json
import sys

from app.tools.registry import execute


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: make tool name=<tool_name> args=\'{"key": "value"}\'', file=sys.stderr)
        sys.exit(1)

    name = sys.argv[1]
    raw_args = sys.argv[2] if len(sys.argv) > 2 else "{}"
    try:
        args = json.loads(raw_args) if raw_args.strip() else {}
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in args: {exc}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(execute(name, args))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
