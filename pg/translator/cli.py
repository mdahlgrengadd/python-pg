"""Command line interface for pg_translator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pg_preprocessor_pygment import convert_pg_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a .pg problem file into translated Python code."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the .pg file to translate.",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Optional destination for the generated .pyg file (defaults to alongside the source).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--emit-imports",
        dest="use_sandbox_macros",
        action="store_false",
        help="Emit macro imports instead of assuming the sandbox provides them.",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Generate standalone executable .pyg file with boilerplate for direct execution.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding to use when reading and writing files (default: utf-8).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress informational output.",
    )
    # Default: sandbox provides macros (no imports)
    parser.set_defaults(use_sandbox_macros=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --standalone implies --emit-imports
    if args.standalone:
        args.use_sandbox_macros = False

    try:
        written_path, result = convert_pg_file(
            args.source,
            output_path=args.output,
            use_sandbox_macros=args.use_sandbox_macros,
            overwrite=args.overwrite,
            encoding=args.encoding,
            standalone=args.standalone,
        )
    except (FileNotFoundError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failures bubble up
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Wrote {written_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
