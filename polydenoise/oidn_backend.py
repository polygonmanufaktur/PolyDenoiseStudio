# SPDX-License-Identifier: MIT
"""Intel Open Image Denoise (OIDN) direct RAM/VRAM C-API wrapper and PFM/CLI utilities."""

from __future__ import annotations

import ctypes
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np

from .constants import register_dll_dir


class OIDNNativeDLL:
    """Zero-disk-I/O ctypes wrapper around OpenImageDenoise.dll (OIDN v2.x / v3.x C-API)."""

    DEVICE_MAP = {"default": 0, "cpu": 1, "sycl": 2, "cuda": 3, "hip": 4, "metal": 5}
    FORMAT_FLOAT3, FORMAT_HALF3, STORAGE_DEVICE = 3, 259, 2
    QUALITY_MAP = {"fast": 4, "balanced": 5, "high": 6}

    def __init__(self, dll_path: Path):
        register_dll_dir(dll_path.parent)
        self.dll_path = dll_path
        self.lib = ctypes.CDLL(str(dll_path))
        self._bind_signatures()
        self.device = self.current_device_key = self.buf_shape_key = None
        self.buf_cache: dict[str, ctypes.c_void_p] = {}
        self.version_str = self._query_dll_version()

    def _bind_signatures(self):
        vp, cp, ci, cb, cs = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_bool, ctypes.c_size_t
        self.lib.oidnNewDevice.argtypes, self.lib.oidnNewDevice.restype = [ci], vp
        self.lib.oidnSetDeviceInt.argtypes, self.lib.oidnSetDeviceInt.restype = [vp, cp, ci], None
        self.lib.oidnGetDeviceInt.argtypes, self.lib.oidnGetDeviceInt.restype = [vp, cp], ci
        self.lib.oidnCommitDevice.argtypes, self.lib.oidnCommitDevice.restype = [vp], None
        self.lib.oidnReleaseDevice.argtypes, self.lib.oidnReleaseDevice.restype = [vp], None
        self.lib.oidnGetDeviceError.argtypes, self.lib.oidnGetDeviceError.restype = [vp, ctypes.POINTER(cp)], ci
        self.lib.oidnNewBuffer.argtypes, self.lib.oidnNewBuffer.restype = [vp, cs], vp
        if hasattr(self.lib, "oidnNewBufferWithStorage"):
            self.lib.oidnNewBufferWithStorage.argtypes, self.lib.oidnNewBufferWithStorage.restype = [vp, cs, ci], vp
        self.lib.oidnWriteBuffer.argtypes, self.lib.oidnWriteBuffer.restype = [vp, cs, cs, vp], None
        self.lib.oidnReadBuffer.argtypes, self.lib.oidnReadBuffer.restype = [vp, cs, cs, vp], None
        self.lib.oidnReleaseBuffer.argtypes, self.lib.oidnReleaseBuffer.restype = [vp], None
        self.lib.oidnNewFilter.argtypes, self.lib.oidnNewFilter.restype = [vp, cp], vp
        self.lib.oidnSetFilterImage.argtypes, self.lib.oidnSetFilterImage.restype = [vp, cp, vp, ci, cs, cs, cs, cs, cs], None
        self.lib.oidnSetFilterBool.argtypes, self.lib.oidnSetFilterBool.restype = [vp, cp, cb], None
        self.lib.oidnSetFilterInt.argtypes, self.lib.oidnSetFilterInt.restype = [vp, cp, ci], None
        self.lib.oidnCommitFilter.argtypes, self.lib.oidnCommitFilter.restype = [vp], None
        self.lib.oidnExecuteFilter.argtypes, self.lib.oidnExecuteFilter.restype = [vp], None
        self.lib.oidnReleaseFilter.argtypes, self.lib.oidnReleaseFilter.restype = [vp], None

    def _query_dll_version(self) -> str:
        try:
            dev = self.lib.oidnNewDevice(1)
            if dev:
                self.lib.oidnCommitDevice(dev)
                maj = self.lib.oidnGetDeviceInt(dev, b"versionMajor")
                min_v = self.lib.oidnGetDeviceInt(dev, b"versionMinor")
                pat = self.lib.oidnGetDeviceInt(dev, b"versionPatch")
                self.lib.oidnReleaseDevice(dev)
                if maj > 0:
                    return f"{maj}.{min_v}.{pat}"
        except Exception:
            pass
        return "2.x"

    def _check_error(self):
        if not self.device:
            return
        err_msg = ctypes.c_char_p()
        code = self.lib.oidnGetDeviceError(self.device, ctypes.byref(err_msg))
        if code != 0:
            msg = err_msg.value.decode("utf-8", errors="replace") if err_msg.value else f"Error {code}"
            raise RuntimeError(f"OIDN DLL Error: {msg}")

    def release_buffers(self):
        for b in self.buf_cache.values():
            if b:
                try:
                    self.lib.oidnReleaseBuffer(b)
                except Exception:
                    pass
        self.buf_cache.clear()
        self.buf_shape_key = None

    def shutdown(self):
        self.release_buffers()
        if self.device is not None:
            try:
                self.lib.oidnReleaseDevice(self.device)
            except Exception:
                pass
            self.device = None

    def ensure_device(self, device_str: str, verbose: int):
        key = (device_str, verbose)
        if self.device is not None and self.current_device_key == key:
            return
        self.shutdown()
        if device_str == "default":
            for fb in ("cuda", "hip", "cpu"):
                dev_try = self.lib.oidnNewDevice(self.DEVICE_MAP[fb])
                if dev_try:
                    self.device = dev_try
                    break
        else:
            self.device = self.lib.oidnNewDevice(self.DEVICE_MAP.get(device_str, 1))
        if not self.device:
            raise RuntimeError(f"Failed to initialize OIDN device '{device_str}'.")
        self.lib.oidnSetDeviceInt(self.device, b"verbose", int(verbose))
        self.lib.oidnCommitDevice(self.device)
        self._check_error()
        self.current_device_key = key

    def ensure_buffers(self, h: int, w: int, use_half: bool):
        key = (h, w, use_half)
        if self.buf_shape_key == key and "color" in self.buf_cache:
            return
        self.release_buffers()
        byte_size = int(h) * int(w) * 3 * (2 if use_half else 4)
        has_storage = hasattr(self.lib, "oidnNewBufferWithStorage")
        for name in ("color", "out", "albedo", "normal"):
            buf = self.lib.oidnNewBufferWithStorage(self.device, byte_size, self.STORAGE_DEVICE) if has_storage else None
            if not buf:
                buf = self.lib.oidnNewBuffer(self.device, byte_size)
            if not buf:
                self._check_error()
                raise RuntimeError(f"Failed to allocate OIDN VRAM buffer ({byte_size} B).")
            self.buf_cache[name] = buf
        self.buf_shape_key = key

    @staticmethod
    def _prep_array(arr: np.ndarray, use_half: bool) -> np.ndarray:
        return (
            np.ascontiguousarray(np.clip(arr, -65500.0, 65500.0), dtype=np.float16)
            if use_half
            else np.ascontiguousarray(arr, dtype=np.float32)
        )

    def upload_aux_to_vram(self, alb_rgb: np.ndarray | None, nrm_rgb: np.ndarray | None, use_half: bool):
        ref = alb_rgb if alb_rgb is not None else nrm_rgb
        if ref is None:
            return
        self.ensure_buffers(ref.shape[0], ref.shape[1], use_half)
        if alb_rgb is not None:
            arr_a = self._prep_array(alb_rgb, use_half)
            self.lib.oidnWriteBuffer(self.buf_cache["albedo"], 0, arr_a.nbytes, arr_a.ctypes.data_as(ctypes.c_void_p))
        if nrm_rgb is not None:
            arr_n = self._prep_array(nrm_rgb, use_half)
            self.lib.oidnWriteBuffer(self.buf_cache["normal"], 0, arr_n.nbytes, arr_n.ctypes.data_as(ctypes.c_void_p))
        self._check_error()

    def denoise_in_vram(
        self,
        color_rgb: np.ndarray,
        has_alb: bool,
        has_nrm: bool,
        filter_type: str = "RT",
        quality: str = "high",
        use_half: bool = True,
        is_hdr: bool = True,
        clean_aux: bool = True,
        inplace: bool = False,
        maxmem_mb: int = -1,
    ) -> np.ndarray:
        h, w, _ = color_rgb.shape
        self.ensure_buffers(h, w, use_half)
        dtype, fmt = (np.float16, self.FORMAT_HALF3) if use_half else (np.float32, self.FORMAT_FLOAT3)
        arr_in = self._prep_array(color_rgb, use_half)
        nbytes = arr_in.nbytes
        color_buf = self.buf_cache["color"]
        out_buf = color_buf if inplace else self.buf_cache["out"]
        self.lib.oidnWriteBuffer(color_buf, 0, nbytes, arr_in.ctypes.data_as(ctypes.c_void_p))

        flt = self.lib.oidnNewFilter(self.device, filter_type.encode("ascii"))
        try:
            self.lib.oidnSetFilterImage(flt, b"color", color_buf, fmt, w, h, 0, 0, 0)
            if has_alb:
                self.lib.oidnSetFilterImage(flt, b"albedo", self.buf_cache["albedo"], fmt, w, h, 0, 0, 0)
            if has_nrm:
                self.lib.oidnSetFilterImage(flt, b"normal", self.buf_cache["normal"], fmt, w, h, 0, 0, 0)
            self.lib.oidnSetFilterImage(flt, b"output", out_buf, fmt, w, h, 0, 0, 0)
            self.lib.oidnSetFilterBool(flt, b"hdr", bool(is_hdr))
            self.lib.oidnSetFilterBool(flt, b"cleanAux", bool(clean_aux))
            self.lib.oidnSetFilterInt(flt, b"quality", self.QUALITY_MAP.get(quality, 6))
            if maxmem_mb > 0:
                self.lib.oidnSetFilterInt(flt, b"maxMemoryMB", int(maxmem_mb))
            self.lib.oidnCommitFilter(flt)
            self._check_error()
            self.lib.oidnExecuteFilter(flt)
            self._check_error()
            arr_out = np.empty((h, w, 3), dtype=dtype)
            self.lib.oidnReadBuffer(out_buf, 0, nbytes, arr_out.ctypes.data_as(ctypes.c_void_p))
            self._check_error()
            return arr_out.astype(np.float32)
        finally:
            self.lib.oidnReleaseFilter(flt)


def write_pfm(rgb: np.ndarray, pfm_path: Path) -> None:
    """Serialize float32 RGB image into Portable FloatMap format."""
    clean_rgb = np.nan_to_num(rgb, nan=0.0, posinf=65500.0, neginf=-65500.0)
    img_flipped = np.ascontiguousarray(clean_rgb[::-1, :, :], dtype=np.float32)
    h, w, _ = img_flipped.shape
    with open(pfm_path, "wb") as f:
        f.write(f"PF\n{w} {h}\n-1.0\n".encode("ascii"))
        f.write(img_flipped.tobytes())


def read_pfm(pfm_path: Path) -> np.ndarray:
    """Load Portable FloatMap back into top-down float32 contiguous array."""
    with open(pfm_path, "rb") as f:
        if f.readline().decode("ascii").strip() != "PF":
            raise RuntimeError("Invalid PFM header.")
        w, h = map(int, f.readline().decode("ascii").strip().split())
        _ = float(f.readline().decode("ascii").strip())
        data = np.frombuffer(f.read(), dtype=np.float32)
    return np.ascontiguousarray(data.reshape((h, w, 3))[::-1, :, :], dtype=np.float32)


def estimate_noise_fast_roi(orig_rgb: np.ndarray, den_rgb: np.ndarray) -> float:
    """Estimate % high-frequency noise reduction across the central ROI patch."""
    h, w, _ = den_rgb.shape
    cy, cx = h // 2, w // 2
    r = min(200, cy, cx)
    if r < 8:
        return 0.0
    d_patch = den_rgb[cy - r : cy + r, cx - r : cx + r, :]
    oh, ow = orig_rgb.shape[:2]
    if (oh, ow) == (h, w):
        o_patch = orig_rgb[cy - r : cy + r, cx - r : cx + r, :]
    else:
        sy, sx = oh / float(h), ow / float(w)
        y0, x0 = int((cy - r) * sy), int((cx - r) * sx)
        y1, x1 = max(y0 + 1, int((cy + r) * sy)), max(x0 + 1, int((cx + r) * sx))
        o_patch = cv2.resize(np.ascontiguousarray(orig_rgb[y0:y1, x0:x1, :]), (2 * r, 2 * r), interpolation=cv2.INTER_AREA)

    def _hf(p: np.ndarray) -> float:
        lum = np.maximum(0.2126 * p[:, :, 0] + 0.7152 * p[:, :, 1] + 0.0722 * p[:, :, 2], 0.0)
        tm = lum / (1.0 + lum)
        return float(np.mean(np.abs(tm[:, 1:] - tm[:, :-1])) + np.mean(np.abs(tm[1:, :] - tm[:-1, :])))

    n_b, n_a = _hf(o_patch), _hf(d_patch)
    return ((n_b - n_a) / max(n_b, 1e-7)) * 100.0


def stride_pack(img: np.ndarray, stride: int, pad: int = 16):
    """Sub-sample correlated supersampled pixel lattice into decorrelated sub-tiles."""
    h, w, _ = img.shape
    sh, sw = (h // stride) * stride, (w // stride) * stride
    cropped = img[:sh, :sw, :]
    rows = [
        np.concatenate(
            [np.pad(cropped[dy::stride, dx::stride, :], ((pad, pad), (pad, pad), (0, 0)), mode="reflect") for dx in range(stride)],
            axis=1,
        )
        for dy in range(stride)
    ]
    return np.concatenate(rows, axis=0), sh, sw


def stride_unpack(packed: np.ndarray, orig_img: np.ndarray, stride: int, pad: int = 16) -> np.ndarray:
    """Reassemble decorrelated OIDN output sub-tiles into original image layout."""
    h, w, _ = orig_img.shape
    sh, sw = (h // stride) * stride, (w // stride) * stride
    sub_h, sub_w = sh // stride, sw // stride
    tile_h, tile_w = sub_h + 2 * pad, sub_w + 2 * pad
    out = orig_img.copy()
    for dy in range(stride):
        for dx in range(stride):
            y0, x0 = dy * tile_h + pad, dx * tile_w + pad
            out[dy:sh:stride, dx:sw:stride, :] = packed[y0 : y0 + sub_h, x0 : x0 + sub_w, :]
    return out


def build_oidn_cmd(
    oidn_bin: Path,
    hdr_pfm: Path,
    out_pfm: Path,
    alb_pfm: Path | None,
    nrm_pfm: Path | None,
    *,
    device: str,
    filter_type: str,
    quality: str,
    precision: str,
    verbose: str | int,
    use_ldr: bool,
    clean_aux: bool,
    inplace: bool,
    maxmem: str = "",
    future_args: str = "",
) -> list[str]:
    """Assemble oidnDenoise CLI argument list cleanly for fallback / custom CLI flags."""
    cmd = [
        str(oidn_bin),
        "-d", device,
        "-f", filter_type,
        "-q", quality,
        "-t", precision,
        "-v", str(verbose),
        "--ldr" if use_ldr else "--hdr",
        str(hdr_pfm),
        "-o",
        str(out_pfm),
    ]
    if alb_pfm is not None:
        cmd.extend(["--alb", str(alb_pfm)])
    if nrm_pfm is not None:
        cmd.extend(["--nrm", str(nrm_pfm)])
    if clean_aux:
        cmd.append("--clean_aux")
    if inplace:
        cmd.append("--inplace")
    if maxmem.strip().isdigit():
        cmd.extend(["--maxmem", maxmem.strip()])
    if future_args.strip():
        cmd.extend(future_args.strip().split())
    return cmd


_BENCH_RES_LABELS = {
    "3840x2160": "4K",
    "1920x1080": "1080p",
    "1280x720": "720p",
    "4096x4096": "4K",
    "2048x2048": "2K",
    "1024x1024": "1K",
}


def sanitize_benchmark_log_line(line: str) -> str:
    """Strip parenthesized compiler repo URLs so benchmark headers read cleanly in log views."""
    return re.sub(r"\s*\(https?://[^)]*\)", "", line).rstrip()


def format_benchmark_test_name(raw_name: str) -> str:
    """Format cryptic OIDN benchmark test IDs (e.g. RT.hdr_calb_cnrm.1280x720) concisely."""
    parts = (raw_name or "").strip().split(".")
    if len(parts) == 3 and parts[0] == "RT":
        _, variant, res = parts
        res_tag = _BENCH_RES_LABELS.get(res, res)
        range_tag = "HDR" if variant.startswith("hdr") else ("LDR" if variant.startswith("ldr") else variant.upper())
        if "calb" in variant or "cnrm" in variant:
            aux_tag = "NoisyAux"
        elif "alb" in variant or "nrm" in variant:
            aux_tag = "CleanAux"
        else:
            aux_tag = "ColorOnly"
        return f"{res_tag} {range_tag} • {aux_tag}"
    if len(parts) == 3 and parts[0] == "RTLightmap":
        _, variant, res = parts
        res_tag = _BENCH_RES_LABELS.get(res, res)
        return f"{res_tag} Lightmap {variant.upper()}"
    return raw_name
