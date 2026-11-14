# SPDX-License-Identifier: MIT
"""PolyDenoise Studio launcher entrypoint (delegates to the modular `polydenoise` package)."""

from __future__ import annotations

import sys

from polydenoise.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
