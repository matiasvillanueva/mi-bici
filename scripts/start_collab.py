#!/usr/bin/env python3
"""Start Jupyter Lab. Safe on Windows checkouts (CRLF does not break Python)."""

from __future__ import annotations

import os
import sys


def main() -> None:
    port = os.environ.get("PORT", "8888")
    root = sys.argv[1] if len(sys.argv) > 1 else "/workspace"
    if not os.path.isdir(os.path.join(root, "puntos")):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print(f"Jupyter Lab → http://127.0.0.1:{port}/lab/tree/puntos")
    print()

    os.execvp(
        "jupyter",
        [
            "jupyter",
            "lab",
            "--ip=0.0.0.0",
            f"--port={port}",
            "--no-browser",
            "--allow-root",
            "--ServerApp.token=",
            "--ServerApp.password=",
            "--ServerApp.allow_origin=*",
            f"--ServerApp.root_dir={root}",
            "--ServerApp.default_url=/lab/tree/puntos",
        ],
    )


if __name__ == "__main__":
    main()
