# SPDX-License-Identifier: MIT
"""Viewport tonemapping, IGN dithering, False-Color EV/Clipping/NaN inspectors, and ROI helpers."""

from __future__ import annotations

from collections import OrderedDict
import threading

import cv2
import numpy as np

from .constants import INSPECTOR_MODES, LUT_STEPS, SRGB_LUT_12BIT
from .exr_pipeline import DIAG_BIT_NAN_INF, DIAG_BIT_NEGATIVE

_FP16_RAW_U16 = np.arange(65536, dtype=np.uint16)
with np.errstate(all="ignore"):
    _FP16_F32 = _FP16_RAW_U16.view(np.float16).astype(np.float32)
_FP16_LUT_BASE = np.where(
    np.isnan(_FP16_F32) | (_FP16_F32 <= 0.0),
    0.0,
    np.where(np.isposinf(_FP16_F32), 65500.0, _FP16_F32),
).astype(np.float32)
_FP16_ACES_CACHE: OrderedDict[float, np.ndarray] = OrderedDict()
_FP16_CACHE_LOCK = threading.Lock()


def _get_fp16_srgb_lut(ev_val: float) -> np.ndarray:
    """Cached 256 KB float16->sRGB[0..255] LUT including exposure gain + ACES filmic curve."""
    key = round(float(ev_val), 4)
    with _FP16_CACHE_LOCK:
        cached = _FP16_ACES_CACHE.get(key)
        if cached is not None:
            _FP16_ACES_CACHE.move_to_end(key)
            return cached
    gain = np.float32(2.0 ** key)
    x = _FP16_LUT_BASE * gain
    mapped = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    lut = SRGB_LUT_12BIT[(mapped * np.float32(LUT_STEPS - 1)).astype(np.int32)]
    with _FP16_CACHE_LOCK:
        _FP16_ACES_CACHE[key] = lut
        _FP16_ACES_CACHE.move_to_end(key)
        while len(_FP16_ACES_CACHE) > 64:
            _FP16_ACES_CACHE.popitem(last=False)
    return lut


def build_fp16_lut_key(linear_rgb: np.ndarray) -> np.ndarray:
    """Pack float32 HDR RGB array into uint16 fp16 bit-indices for rapid CPU LUT tonemapping."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return np.ascontiguousarray(np.asarray(linear_rgb, dtype=np.float32), dtype=np.float16).view(np.uint16)


def compute_ign_dither(h: int, w: int) -> np.ndarray:
    """Compute Interleaved Gradient Noise triangular PDF dither matrix in [-0.5, +0.5]."""
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    ign1 = np.mod(52.9829189 * np.mod(0.06711056 * x + 0.00583715 * y, 1.0), 1.0)
    ign2 = np.mod(52.9829189 * np.mod(0.06711056 * (x + 17.0) + 0.00583715 * (y + 31.0), 1.0), 1.0)
    return ((ign1 - ign2) * 0.5)[:, :, None].astype(np.float32)


def _muted_luma_bg(linear_rgb: np.ndarray, ev_val: float) -> np.ndarray:
    """Muted neutral backdrop so bright diagnostic alert overlays read unmistakably."""
    img = np.maximum(np.nan_to_num(linear_rgb, nan=0.0, posinf=65500.0, neginf=0.0), 0.0)
    if abs(ev_val) > 1e-4:
        img = img * np.float32(2.0 ** ev_val)
    y = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    mapped = y / (1.0 + y)
    srgb_gray = np.clip(np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2) * 145.0, 0.0, 180.0)
    return np.repeat(srgb_gray[:, :, None], 3, axis=2)


def apply_false_color_ev(linear_rgb: np.ndarray, ev_val: float = 0.0) -> np.ndarray:
    """
    Studio False-Color exposure LUT mapped in EV stops relative to 18% middle gray (0.18 -> green).
    """
    img = np.maximum(np.nan_to_num(linear_rgb, nan=0.0, posinf=65500.0, neginf=0.0), 0.0)
    if abs(ev_val) > 1e-4:
        img = img * np.float32(2.0 ** ev_val)
    lum = np.maximum(0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2], 1e-6)
    ev_rel = np.log2(lum / 0.18)

    # Stop stops centered around 0 EV (18% middle gray):
    stops = np.array([-5.0, -3.5, -2.0, -0.7, 0.0, 0.9, 1.8, 2.8, 4.2, 6.0], dtype=np.float32)
    palette = np.array([
        [35, 10, 80],     # < -5 EV crushed shadow (deep purple)
        [30, 45, 150],    # -3.5 EV shadow blue
        [0, 145, 225],    # -2.0 EV cyan fill
        [0, 200, 175],    # -0.7 EV low midtone teal
        [45, 205, 70],    #  0.0 EV 18% middle gray green
        [235, 220, 45],   # +0.9 EV bright diffuse yellow
        [255, 140, 20],   # +1.8 EV key / skin highlight orange
        [250, 45, 30],    # +2.8 EV approaching white clip red
        [255, 0, 190],    # +4.2 EV hot specular magenta
        [255, 240, 255],  # > +6 EV extreme peak
    ], dtype=np.float32)

    out = np.empty((img.shape[0], img.shape[1], 3), dtype=np.float32)
    for c in range(3):
        out[:, :, c] = np.interp(ev_rel, stops, palette[:, c])
    # Mark strictly black pixels deep charcoal so background zero stands apart cleanly
    zero_mask = lum <= 1.1e-6
    out[zero_mask] = (14.0, 14.0, 22.0)
    return np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)


def apply_clipping_fireflies(linear_rgb: np.ndarray, ev_val: float = 0.0) -> np.ndarray:
    """Highlight SDR clipping (>1.0), strong HDR (>4.0), and localized single-pixel fireflies."""
    img = np.maximum(np.nan_to_num(linear_rgb, nan=0.0, posinf=65500.0, neginf=0.0), 0.0)
    if abs(ev_val) > 1e-4:
        img = img * np.float32(2.0 ** ev_val)
    max_c = np.max(img, axis=2)
    out = _muted_luma_bg(linear_rgb, ev_val)

    # Localized firefly detector: spike > 6x 3x3 blurred neighborhood and max > 2.0
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    med_l = cv2.medianBlur(np.minimum(lum, 1000.0).astype(np.float32), 5)
    firefly = (lum > np.maximum(med_l * 6.0, 2.0)) | (max_c > 16.0)

    clip_sdr = (max_c > 1.0) & (max_c <= 4.0)
    clip_hdr = (max_c > 4.0) & ~firefly

    out[clip_sdr] = (255.0, 175.0, 0.0)   # Amber: > 1.0
    out[clip_hdr] = (255.0, 55.0, 15.0)   # Red-Orange: > 4.0
    out[firefly] = (255.0, 0.0, 255.0)    # Magenta: extreme / firefly spikes
    return np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)


def apply_invalid_pixel_inspector(
    linear_rgb: np.ndarray,
    diag_mask: np.ndarray | None = None,
    albedo_rgb: np.ndarray | None = None,
    ev_val: float = 0.0,
) -> np.ndarray:
    """
    Highlight invalid pixel hazards:
    - Neon Magenta (#ff00cc): NaN / Inf (directly or preserved in diag_mask)
    - Electric Cyan (#00e5ff): Negative RGB values (< -1e-6)
    - Bright Yellow (#ffe600): Albedo == 0 on active scene pixels
    """
    out = _muted_luma_bg(linear_rgb, ev_val)
    arr = np.asarray(linear_rgb, dtype=np.float32)

    nan_mask = ~np.all(np.isfinite(arr), axis=2)
    finite_arr = np.where(np.isfinite(arr), arr, 0.0)
    neg_mask = np.min(finite_arr, axis=2) < -1e-6

    if diag_mask is not None and diag_mask.shape[:2] == arr.shape[:2]:
        nan_mask = nan_mask | ((diag_mask & DIAG_BIT_NAN_INF) != 0)
        neg_mask = neg_mask | ((diag_mask & DIAG_BIT_NEGATIVE) != 0)

    if albedo_rgb is not None and albedo_rgb.shape[:2] == arr.shape[:2]:
        alb_max = np.max(np.nan_to_num(albedo_rgb, nan=0.0), axis=2)
        lum = np.max(finite_arr, axis=2)
        alb_zero_mask = (alb_max <= 1e-6) & (lum > 1e-4)
        out[alb_zero_mask] = (255.0, 230.0, 0.0)

    out[neg_mask] = (0.0, 229.0, 255.0)
    out[nan_mask] = (255.0, 0.0, 204.0)
    return np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)


def tonemap_and_dither_fast(
    linear_rgb: np.ndarray,
    ev_val: float,
    ch_name: str = "",
    ign_dither: np.ndarray | None = None,
    xray_mask_active: bool = False,
    fp16_key: np.ndarray | None = None,
) -> np.ndarray:
    """Fast ACES filmic tonemapping + 12-bit sRGB LUT + IGN TPDF dithering for viewport."""
    if xray_mask_active:
        return np.clip(linear_rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    ch_low = ch_name.lower()
    if ch_low == "depth":
        valid = linear_rgb[np.isfinite(linear_rgb)]
        if valid.size > 0:
            d_min, d_max = float(np.percentile(valid, 1.0)), float(np.percentile(valid, 99.0))
            norm_d = np.clip((linear_rgb - d_min) / max(d_max - d_min, 1e-5), 0.0, 1.0)
        else:
            norm_d = np.clip(linear_rgb, 0.0, 1.0)
        if abs(ev_val) > 1e-4:
            norm_d = np.clip(norm_d * float(2.0 ** ev_val), 0.0, 1.0)
        return (norm_d * 255.0 + 0.5).astype(np.uint8)

    if "normal" in ch_low:
        n_vis = np.clip(linear_rgb * 0.5 + 0.5, 0.0, 1.0)
        return (n_vis * 255.0 + 0.5).astype(np.uint8)

    key_u16 = fp16_key if (fp16_key is not None and fp16_key.shape == linear_rgb.shape) else build_fp16_lut_key(linear_rgb)
    srgb_float = _get_fp16_srgb_lut(ev_val)[key_u16]
    h, w, _ = srgb_float.shape
    dith = ign_dither if (ign_dither is not None and ign_dither.shape[:2] == (h, w)) else compute_ign_dither(h, w)
    return np.clip(srgb_float + dith + 0.5, 0.0, 255.0).astype(np.uint8)


def render_viewport_rgb(
    linear_rgb: np.ndarray,
    ev_val: float,
    ch_name: str = "",
    inspector_mode: str = INSPECTOR_MODES[0],
    ign_dither: np.ndarray | None = None,
    xray_mask_active: bool = False,
    diag_mask: np.ndarray | None = None,
    albedo_rgb: np.ndarray | None = None,
    fp16_key: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch viewport linear array to standard ACES or active diagnostic Inspector overlay."""
    if xray_mask_active:
        return np.clip(linear_rgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    if inspector_mode == INSPECTOR_MODES[1]:
        return apply_false_color_ev(linear_rgb, ev_val)
    if inspector_mode == INSPECTOR_MODES[2]:
        return apply_clipping_fireflies(linear_rgb, ev_val)
    if inspector_mode == INSPECTOR_MODES[3]:
        return apply_invalid_pixel_inspector(linear_rgb, diag_mask=diag_mask, albedo_rgb=albedo_rgb, ev_val=ev_val)
    return tonemap_and_dither_fast(
        linear_rgb, ev_val, ch_name=ch_name, ign_dither=ign_dither, xray_mask_active=False, fp16_key=fp16_key
    )


def clamp_roi_box(x0: float, y0: float, x1: float, y1: float, full_w: int, full_h: int, min_px: int = 16) -> tuple[int, int, int, int]:
    """Clamp & snap fractional/pixel ROI coordinates to even-aligned full-res bounds."""
    lx0, lx1 = sorted((int(round(min(x0, x1))), int(round(max(x0, x1)))))
    ly0, ly1 = sorted((int(round(min(y0, y1))), int(round(max(y0, y1)))))
    lx0 = max(0, min(lx0 & ~1, max(0, full_w - min_px)))
    ly0 = max(0, min(ly0 & ~1, max(0, full_h - min_px)))
    lx1 = min(full_w, max(lx0 + min_px, (lx1 + 1) & ~1))
    ly1 = min(full_h, max(ly0 + min_px, (ly1 + 1) & ~1))
    return lx0, ly0, lx1, ly1


def probe_pixel_summary(linear_rgb: np.ndarray, ix: int, iy: int, ch_name: str = "", diag_bit: int = 0) -> str:
    """Format crisp HDR pixel probe text for viewport status bar."""
    h, w, _ = linear_rgb.shape
    cx = max(0, min(w - 1, int(ix)))
    cy = max(0, min(h - 1, int(iy)))
    r, g, b = [float(v) for v in linear_rgb[cy, cx, :3]]
    flags = []
    if (diag_bit & DIAG_BIT_NAN_INF) or not (np.isfinite(r) and np.isfinite(g) and np.isfinite(b)):
        flags.append("NaN/Inf!")
    if (diag_bit & DIAG_BIT_NEGATIVE) or min(r, g, b) < -1e-6:
        flags.append("NEG")
    if max(r, g, b) > 1.0:
        flags.append("HDR>1")
    lum = max(0.0, 0.2126 * r + 0.7152 * g + 0.0722 * b)
    ev_stop = np.log2(lum / 0.18) if lum > 1e-6 else -99.0
    flag_txt = f" [{' '.join(flags)}]" if flags else ""
    if ch_name.lower() == "depth":
        return f"XY({cx},{cy}) Depth={r:.4f}{flag_txt}"
    ev_txt = f"{ev_stop:+.1f}EV" if ev_stop > -50.0 else "-inf EV"
    return f"XY({cx},{cy}) RGB({r:.3f}, {g:.3f}, {b:.3f}) Y={lum:.3f} ({ev_txt}){flag_txt}"
