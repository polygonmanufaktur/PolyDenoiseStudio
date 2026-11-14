# SPDX-License-Identifier: MIT
"""Application constants, directory discovery, DLL registration, and codec tables."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import numpy as np
import OpenEXR

APP_NAME = "PolyDenoise Studio"
APP_VERSION = "1.5.1"
TOOL_GITHUB_REPO = "polygonmanufaktur/PolyDenoiseStudio"
POLYGON_WEB_URL = "https://www.polygonmanufaktur.de/?utm_source=polydenoise_studio_app"

# Resolve workspace/application root regardless of package depth or frozen exe bundling.
BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "oidn_config.json"

BEAUTY_SUM_KEY = "[⚡ Sum All Lights (Beauty)]"
ZOOM_STEPS = [25, 33, 50, 67, 75, 100, 125, 150, 200, 250, 300, 400]
SCALE_MODE_OPTIONS = ["Scale %", "Target Width (px)", "Target Height (px)", "No Downscale (100%)"]

INSPECTOR_MODES = [
    "Off (Standard View)",
    "False-Color (EV / IRE)",
    "Clipping & Fireflies (>1.0)",
    "NaN / Inf / Neg / Albedo=0",
]

COMPRESSION_MAP = {
    "DWAB (Lossy 256-Line)": ("DWAB_COMPRESSION", "20"),
    "DWAA (Lossy 32-Line)": ("DWAA_COMPRESSION", "20"),
    "ZIP (Lossless 16-Line)": ("ZIP_COMPRESSION", "4"),
    "ZSTD (v3.5 Lossless)": ("ZSTD_COMPRESSION", "5"),
    "RLE (Lossless Flat/ID)": ("RLE_COMPRESSION", ""),
    "ZIPS (Lossless 1-Line)": ("ZIPS_COMPRESSION", "4"),
    "PIZ (Wavelet Lossless)": ("PIZ_COMPRESSION", ""),
    "LJ2K (v3.5 Lossy HTJ2K)": ("LJ2K_COMPRESSION", "110"),
    "HTJ2K256 (v3.4 Lossless)": ("HTJ2K256_COMPRESSION", ""),
    "HTJ2K32 (v3.4 Fast Lossless)": ("HTJ2K32_COMPRESSION", ""),
    "NONE (Uncompressed)": ("NO_COMPRESSION", ""),
}

AUX_COMPRESSION_OPTIONS = [
    "ZIP (Lossless 16-Line)",
    "ZSTD (v3.5 Lossless)",
    "RLE (Lossless Flat/ID)",
    "ZIPS (Lossless 1-Line)",
    "PIZ (Wavelet Lossless)",
    "HTJ2K32 (v3.4 Fast Lossless)",
]


def get_initial_oidn_path() -> Path:
    """Locate bundled or configured Intel OIDN executable/DLL cleanly."""
    for candidate in (
        BASE_DIR / "oidn" / "oidnDenoise.exe",
        BASE_DIR / "oidn" / "OpenImageDenoise.dll",
        BASE_DIR / "oidnDenoise.exe",
        BASE_DIR / "OpenImageDenoise.dll",
        BASE_DIR / "_internal" / "oidnDenoise.exe",
        BASE_DIR / "_internal" / "OpenImageDenoise.dll",
    ):
        if candidate.exists():
            return candidate
    if CONFIG_FILE.exists():
        try:
            configured = Path(json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("oidn_bin", ""))
            if configured.exists():
                return configured
        except Exception:
            pass
    return BASE_DIR / "oidn" / "oidnDenoise.exe"


INITIAL_OIDN_BIN = get_initial_oidn_path()


def register_dll_dir(directory: Path) -> None:
    """Register native library directory on Windows PATH and DLL search list."""
    if not directory.exists():
        return
    dir_str = str(directory)
    if dir_str not in os.environ.get("PATH", ""):
        os.environ["PATH"] = dir_str + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(dir_str)
        except Exception:
            pass


for _dll_dir in (BASE_DIR, BASE_DIR / "oidn", BASE_DIR / "OpenEXR.libs", BASE_DIR / "_internal", INITIAL_OIDN_BIN.parent):
    register_dll_dir(_dll_dir)

CPU_THREADS = os.cpu_count() or 8
if hasattr(OpenEXR, "set_global_thread_count"):
    OpenEXR.set_global_thread_count(CPU_THREADS)

LUT_STEPS = 4096
SRGB_LUT_12BIT = (np.power(np.linspace(0.0, 1.0, LUT_STEPS, dtype=np.float32), 1.0 / 2.2) * 255.0).astype(np.float32)
