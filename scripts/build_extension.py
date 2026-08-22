#!/usr/bin/env python3
"""Builds the Python extension from `affinegaps.mojo`.

Mojo refuses to emit a shared library from a file that defines `main`, and the native command-line
tool needs one, so the extension is compiled from a copy with the command-line region removed. That
region is last in the file, which makes this a truncation rather than a splice.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

MARKER = "# region Command Line"
ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "affinegaps.mojo"


def library_source(text: str) -> str:
    """The module with its command-line region removed."""
    marker = text.find(MARKER)
    if marker < 0:
        raise SystemExit(f"{SOURCE} has no '{MARKER}' marker to strip")
    return text[:marker]


def main() -> int:
    output = ROOT / "build" / "affinegaps_mojo.so"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        stripped = pathlib.Path(scratch) / "affinegaps.mojo"
        stripped.write_text(library_source(SOURCE.read_text()))
        compiler = shutil.which("mojo")
        command = [compiler] if compiler else [sys.executable, "-m", "mojo"]
        return subprocess.run(
            [*command, "build", str(stripped), "--emit", "shared-lib", "-o", str(output)],
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
