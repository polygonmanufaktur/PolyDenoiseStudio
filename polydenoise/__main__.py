# SPDX-License-Identifier: MIT
"""Module entry point allowing `python -m polydenoise` execution."""

from __future__ import annotations

import sys

from .constants import APP_NAME, APP_VERSION


def main(argv: list[str] | None = None) -> int:
    """Launch PolyDenoise Studio desktop GUI or print headless CLI help."""
    args = sys.argv[1:] if argv is None else argv
    if "--version" in args or "-V" in args:
        print(f"{APP_NAME} v{APP_VERSION}")
        return 0
    if "--help" in args or "-h" in args:
        print(f"{APP_NAME} v{APP_VERSION} — Launch via `python polydenoise_studio.py` or `python -m polydenoise`.")
        return 0

    from .ui_app import PolyDenoiseStudio

    app = PolyDenoiseStudio()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
