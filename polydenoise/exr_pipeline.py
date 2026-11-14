# SPDX-License-Identifier: MIT
"""OpenEXR 3.5+ I/O, Multi-Layer EXR assembly, AOV pass parser, and depth normalization."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import OpenEXR

from .constants import CPU_THREADS
from .downscale_engine import DownscaleEngine

AUX_ALIASES: dict[str, str] = {
    "normals_pre_bump": "normals_pre_bump",
    "normal_pre_bump": "normals_pre_bump",
    "normals_shading": "normals",
    "shading_normal": "normals",
    "normals": "normals",
    "normal": "normals",
    "albedo": "albedo",
    "diffuse_albedo": "albedo",
    "alpha": "alpha",
    "depth": "depth",
    "zdepth": "depth",
    "z_depth": "depth",
    "roughness": "roughness",
    "specular_roughness": "roughness",
    "material_id": "material_id",
    "mat_id": "material_id",
    "object_id": "object_id",
    "obj_id": "object_id",
    "position": "position",
    "world_position": "position",
}

AUX_BLACKLIST: set[str] = {
    "albedo", "alpha", "depth", "roughness", "material_id", "normals", "normals_pre_bump", "object_id", "position",
}

KNOWN_STD_LIGHT_PASSES: list[str] = [
    "direct_lighting",
    "indirect_lighting",
    "emission_lighting",
    "specular_reflection_lighting",
    "refraction_lighting",
    "transmission_lighting",
    "sss_lighting",
    "participating_media_lighting",
    "beauty",
    "combined",
]

COMPONENT_AOV_PASSES: set[str] = set(KNOWN_STD_LIGHT_PASSES)

# Bitmask flags preserved before NaN/Inf/Negative sanitization so Inspectors see raw defects
DIAG_BIT_NAN_INF = 1
DIAG_BIT_NEGATIVE = 2
DIAG_BIT_ALBEDO_ZERO = 4


def normalize_depth_guide(raw_depth: np.ndarray | None) -> np.ndarray | None:
    """Log-compress and robust-normalize Z-depth (P1..P99) into [0, 1] float32 (H, W) for shader guides."""
    if raw_depth is None:
        return None
    d = raw_depth[:, :, 0] if raw_depth.ndim == 3 else raw_depth
    d = np.nan_to_num(np.asarray(d, dtype=np.float32), nan=0.0, posinf=65500.0, neginf=0.0)
    d = np.log1p(np.maximum(d, 0.0))
    sample = d[::4, ::4] if min(d.shape[:2]) >= 16 else d
    lo, hi = float(np.percentile(sample, 1.0)), float(np.percentile(sample, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 1e-5:
        lo, hi = float(np.min(d)), float(np.max(d))
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 1e-5:
            return None
    return np.ascontiguousarray(np.clip((d - lo) / (hi - lo), 0.0, 1.0), dtype=np.float32)


def compute_raw_diagnostics_mask(raw_rgb: np.ndarray) -> np.ndarray:
    """Build uint8 per-pixel bitmask recording raw NaN/Inf (bit0) and negative values (bit1)."""
    if raw_rgb.dtype == np.uint32:
        return np.zeros(raw_rgb.shape[:2], dtype=np.uint8)
    arr = np.asarray(raw_rgb, dtype=np.float32)
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    bad = ~np.all(np.isfinite(arr), axis=2)
    if np.any(bad):
        mask[bad] |= DIAG_BIT_NAN_INF
    finite_arr = np.where(np.isfinite(arr), arr, 0.0)
    neg = np.min(finite_arr, axis=2) < -1e-6
    if np.any(neg):
        mask[neg] |= DIAG_BIT_NEGATIVE
    return mask


def read_exr_native(exr_path: Path, preserve_uint32: bool = False):
    """Read single-layer or multi-channel OpenEXR file into contiguous numpy array + alpha + keys."""
    with OpenEXR.File(str(exr_path)) as f:
        ch_dict = f.channels()
        alpha = None
        keys = list(ch_dict.keys())
        first_pix = ch_dict[keys[0]].pixels
        target_dt = np.uint32 if (preserve_uint32 and first_pix.dtype == np.uint32) else np.float32

        if "RGBA" in ch_dict:
            rgba = np.asarray(ch_dict["RGBA"].pixels, dtype=target_dt)
            return (
                np.ascontiguousarray(rgba[:, :, :3], dtype=target_dt),
                np.ascontiguousarray(rgba[:, :, 3], dtype=np.float32),
                keys,
            )
        if "RGB" in ch_dict:
            rgb = np.ascontiguousarray(np.asarray(ch_dict["RGB"].pixels, dtype=target_dt)[:, :, :3], dtype=target_dt)
        elif all(k in ch_dict for k in ("R", "G", "B")):
            rgb = np.ascontiguousarray(
                np.dstack((ch_dict["R"].pixels, ch_dict["G"].pixels, ch_dict["B"].pixels)), dtype=target_dt
            )
        elif all(k in ch_dict for k in ("X", "Y", "Z")):
            rgb = np.ascontiguousarray(
                np.dstack((ch_dict["X"].pixels, ch_dict["Y"].pixels, ch_dict["Z"].pixels)), dtype=target_dt
            )
        else:
            gray = np.asarray(first_pix, dtype=target_dt)
            rgb = np.ascontiguousarray(np.dstack((gray, gray, gray)) if gray.ndim == 2 else gray[:, :, :3], dtype=target_dt)
        if "A" in ch_dict:
            alpha = np.ascontiguousarray(ch_dict["A"].pixels, dtype=np.float32)
        return rgb, alpha, keys


def is_exr_black_fast(exr_path: Path) -> bool:
    """Fast strided check for 100% empty black light passes."""
    try:
        with OpenEXR.File(str(exr_path)) as f:
            ch_dict = f.channels()
            for k in ("RGB", "RGBA", "R", "G", "B", "Y"):
                if k in ch_dict:
                    pix = ch_dict[k].pixels
                    if float(np.max(pix[::8, ::8])) > 1e-6:
                        return False
                    return float(np.max(pix)) <= 1e-6
            first_k = next(iter(ch_dict.keys()))
            return float(np.max(ch_dict[first_k].pixels)) <= 1e-6
    except Exception:
        return False


def _apply_compression_header(header: dict, comp_attr_name: str, level_str: str) -> None:
    if not level_str:
        return
    try:
        vf, vi = float(level_str), int(float(level_str))
        if "DWA" in comp_attr_name:
            header["dwaCompressionLevel"] = vf
        elif "ZSTD" in comp_attr_name:
            header["zstdCompressionLevel"] = max(1, min(22, vi))
        elif "LJ2K" in comp_attr_name:
            header["lossyHTJ2KQuality"] = max(1, min(150, vi))
        elif "ZIP" in comp_attr_name:
            header["zipCompressionLevel"] = max(1, min(9, vi))
    except ValueError:
        pass


def write_exr_static(
    out_exr_path: Path,
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    comp_attr_name: str,
    level_str: str,
    use_half: bool,
    worker_threads: int = CPU_THREADS,
    is_data_pass: bool = False,
) -> None:
    """Write a single OpenEXR image with configurable compression and data/color interop attribute."""
    header = {"compression": getattr(OpenEXR, comp_attr_name), "type": OpenEXR.scanlineimage}
    if is_data_pass:
        header["colorInteropID"] = "data"
    _apply_compression_header(header, comp_attr_name, level_str)

    if rgb.dtype == np.uint32:
        target_dtype, rgb_safe = np.uint32, rgb
    elif use_half:
        target_dtype = np.float16
        rgb_safe = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=65500.0, neginf=-65500.0), -65500.0, 65500.0)
    else:
        target_dtype = np.float32
        rgb_safe = np.nan_to_num(rgb, nan=0.0, posinf=1e30, neginf=-1e30)

    channels = {
        "R": np.ascontiguousarray(rgb_safe[:, :, 0], dtype=target_dtype),
        "G": np.ascontiguousarray(rgb_safe[:, :, 1], dtype=target_dtype),
        "B": np.ascontiguousarray(rgb_safe[:, :, 2], dtype=target_dtype),
    }
    if alpha is not None:
        a_dt = np.float16 if use_half else np.float32
        channels["A"] = np.ascontiguousarray(np.clip(alpha, 0.0, 65500.0) if use_half else alpha, dtype=a_dt)

    try:
        with OpenEXR.File(header, channels, num_threads=worker_threads) as outfile:
            outfile.write(str(out_exr_path))
    except TypeError:
        with OpenEXR.File(header, channels) as outfile:
            outfile.write(str(out_exr_path))


def _prep_layer_plane(arr: np.ndarray, use_half: bool) -> np.ndarray:
    if use_half:
        return np.ascontiguousarray(
            np.clip(np.nan_to_num(arr, nan=0.0, posinf=65500.0, neginf=-65500.0), -65500.0, 65500.0),
            dtype=np.float16,
        )
    return np.ascontiguousarray(np.nan_to_num(arr, nan=0.0, posinf=1e30, neginf=-1e30), dtype=np.float32)


def write_multilayer_exr(
    out_exr_path: Path,
    light_paths: dict[str, Path],
    aux_paths: dict[str, Path],
    target_w: int,
    target_h: int,
    ds_kernel: str,
    comp_attr_name: str,
    level_str: str,
    use_half: bool,
    worker_threads: int = CPU_THREADS,
) -> list[str]:
    """
    Assemble one consolidated Nuke/Resolve-compatible multi-layer EXR for a completed frame.
    Reads already-written passes sequentially so memory stays bounded regardless of layer count.
    Returns list of exported layer names.
    """
    header = {"compression": getattr(OpenEXR, comp_attr_name), "type": OpenEXR.scanlineimage}
    _apply_compression_header(header, comp_attr_name, level_str)
    is_dwa = "DWA" in comp_attr_name

    channels: dict[str, np.ndarray] = {}
    exported_layers: list[str] = []
    beauty_sum = np.zeros((target_h, target_w, 3), dtype=np.float32)
    alpha_out = None

    # Determine non-double-counted Beauty light layers for top-level R/G/B composite
    sum_keys = set(get_sum_light_channels(light_paths, set()))

    for pass_name, l_path in sorted(light_paths.items()):
        if not l_path.exists():
            continue
        rgb, alpha_ch, _ = read_exr_native(l_path)
        rgb = np.maximum(np.nan_to_num(rgb, nan=0.0, posinf=65500.0, neginf=0.0), 0.0)
        if rgb.shape[:2] != (target_h, target_w):
            rgb = DownscaleEngine.downscale_aux_pass(rgb, target_w, target_h, pass_name, ds_kernel)
        if alpha_ch is not None and alpha_out is None:
            if alpha_ch.shape[:2] != (target_h, target_w):
                alpha_ch = DownscaleEngine.downscale_aux_pass(
                    np.dstack([alpha_ch] * 3), target_w, target_h, "alpha", ds_kernel
                )[:, :, 0]
            alpha_out = alpha_ch

        if pass_name in sum_keys:
            beauty_sum += rgb
        safe_name = pass_name.replace(" ", "_")
        channels[f"{safe_name}.R"] = _prep_layer_plane(rgb[:, :, 0], use_half)
        channels[f"{safe_name}.G"] = _prep_layer_plane(rgb[:, :, 1], use_half)
        channels[f"{safe_name}.B"] = _prep_layer_plane(rgb[:, :, 2], use_half)
        exported_layers.append(safe_name)

    if light_paths:
        channels["R"] = _prep_layer_plane(beauty_sum[:, :, 0], use_half)
        channels["G"] = _prep_layer_plane(beauty_sum[:, :, 1], use_half)
        channels["B"] = _prep_layer_plane(beauty_sum[:, :, 2], use_half)

    for aux_name, a_path in sorted(aux_paths.items()):
        if not a_path.exists():
            continue
        is_id = aux_name in ("material_id", "object_id")
        if is_id and is_dwa:
            # OpenEXR DWA compressors do not support integer UINT channels.
            continue
        a_rgb, a_alpha, _ = read_exr_native(a_path, preserve_uint32=is_id)
        if a_rgb.shape[:2] != (target_h, target_w):
            a_rgb = DownscaleEngine.downscale_aux_pass(a_rgb, target_w, target_h, aux_name, ds_kernel)
        if a_alpha is not None and alpha_out is None:
            if a_alpha.shape[:2] != (target_h, target_w):
                a_alpha = DownscaleEngine.downscale_aux_pass(
                    np.dstack([a_alpha] * 3), target_w, target_h, "alpha", ds_kernel
                )[:, :, 0]
            alpha_out = a_alpha

        safe_aux = aux_name.replace(" ", "_")
        if aux_name == "depth":
            channels[f"{safe_aux}.Z"] = _prep_layer_plane(a_rgb[:, :, 0], use_half=False)
        elif aux_name in ("normals", "normals_pre_bump", "position"):
            force_fp32 = aux_name == "position"
            channels[f"{safe_aux}.X"] = _prep_layer_plane(a_rgb[:, :, 0], use_half=(use_half and not force_fp32))
            channels[f"{safe_aux}.Y"] = _prep_layer_plane(a_rgb[:, :, 1], use_half=(use_half and not force_fp32))
            channels[f"{safe_aux}.Z"] = _prep_layer_plane(a_rgb[:, :, 2], use_half=(use_half and not force_fp32))
        elif aux_name == "alpha":
            if alpha_out is None:
                alpha_out = a_rgb[:, :, 0]
        elif is_id:
            if a_rgb.dtype == np.uint32:
                channels[f"{safe_aux}.ID"] = np.ascontiguousarray(a_rgb[:, :, 0], dtype=np.uint32)
            else:
                channels[f"{safe_aux}.R"] = _prep_layer_plane(a_rgb[:, :, 0], use_half=False)
        else:
            channels[f"{safe_aux}.R"] = _prep_layer_plane(a_rgb[:, :, 0], use_half)
            channels[f"{safe_aux}.G"] = _prep_layer_plane(a_rgb[:, :, 1], use_half)
            channels[f"{safe_aux}.B"] = _prep_layer_plane(a_rgb[:, :, 2], use_half)
        exported_layers.append(safe_aux)

    if alpha_out is not None:
        channels["A"] = _prep_layer_plane(alpha_out, use_half)

    if not channels:
        return []

    try:
        with OpenEXR.File(header, channels, num_threads=worker_threads) as outfile:
            outfile.write(str(out_exr_path))
    except TypeError:
        with OpenEXR.File(header, channels) as outfile:
            outfile.write(str(out_exr_path))
    return exported_layers


def extract_frame_and_channel(stem: str, discovered_prefixes: list[str]) -> tuple[str, str]:
    """Parse universal render-pass filename stems into (frame_id, canonical_pass_name)."""
    if "_channels_" in stem:
        frame_id, raw_pass = stem.split("_channels_", 1)
        return frame_id, AUX_ALIASES.get(raw_pass.lower(), raw_pass)

    all_known = sorted(list(AUX_ALIASES.keys()) + KNOWN_STD_LIGHT_PASSES, key=len, reverse=True)
    stem_low = stem.lower()
    for kw in all_known:
        for sep in ("_", ".", "-", " "):
            token = f"{sep}{kw}"
            if stem_low.endswith(token):
                prefix = stem[: -len(token)].rstrip("_.- ") or "Scene"
                return prefix, AUX_ALIASES.get(kw, kw)

    for pref in discovered_prefixes:
        if stem.startswith(pref) and len(stem) > len(pref):
            rem = stem[len(pref) :]
            if rem[0] in ("_", ".", "-", " "):
                raw_pass = rem.lstrip("_.- ")
                if raw_pass:
                    return pref, AUX_ALIASES.get(raw_pass.lower(), raw_pass)

    m = re.match(r"^(.*?)[_.\-]([^_.\-]+)$", stem)
    if m and m.group(1):
        return m.group(1), AUX_ALIASES.get(m.group(2).lower(), m.group(2))
    return "Scene", stem


def get_sum_light_channels(frame_dict: dict, empty_set: set[str]) -> list[str]:
    """Return active non-overlapping light channels suitable for additive Beauty summation."""
    candidates = [c for c in frame_dict.keys() if c not in AUX_BLACKLIST and c not in empty_set]
    custom_lights = [c for c in candidates if c.lower() not in COMPONENT_AOV_PASSES and not c.lower().endswith("_lighting")]
    return custom_lights if custom_lights else candidates
