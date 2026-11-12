# SPDX-License-Identifier: MIT
# PolyDenoise Studio — High-Performance Direct RAM/VRAM AI Batch Denoiser & HDR Downscaler
# Copyright (c) 2026 Arthur Liebnau / Polygon Manufaktur (https://www.polygonmanufaktur.de)
#
# Third-Party Notices:
# - Intel(R) Open Image Denoise (OIDN): Licensed under Apache License 2.0 (Copyright Intel Corp.)
# - OpenEXR: Licensed under BSD-3-Clause License (Copyright Contributors to the OpenEXR Project)
# - Magic Kernel Sharp algorithm: Mathematics by John Costella

import os, sys, json, re, time, ctypes, tempfile, threading, subprocess, webbrowser
import urllib.request, urllib.error
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import OpenEXR

APP_NAME = "PolyDenoise Studio"
APP_VERSION = "1.2.0"
TOOL_GITHUB_REPO = "polygonmanufaktur/PolyDenoiseStudio"
POLYGON_WEB_URL = "https://www.polygonmanufaktur.de/?utm_source=polydenoise_studio_app"

BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
CONFIG_FILE = BASE_DIR / "oidn_config.json"


def get_initial_oidn_path() -> Path:
    for c in [
        BASE_DIR / "oidn" / "oidnDenoise.exe", BASE_DIR / "oidn" / "OpenImageDenoise.dll",
        BASE_DIR / "oidnDenoise.exe", BASE_DIR / "OpenImageDenoise.dll",
        BASE_DIR / "_internal" / "oidnDenoise.exe", BASE_DIR / "_internal" / "OpenImageDenoise.dll"
    ]:
        if c.exists():
            return c
    if CONFIG_FILE.exists():
        try:
            p = Path(json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("oidn_bin", ""))
            if p.exists():
                return p
        except Exception:
            pass
    return BASE_DIR / "oidn" / "oidnDenoise.exe"


INITIAL_OIDN_BIN = get_initial_oidn_path()


def register_dll_dir(directory: Path):
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


for d in [BASE_DIR, BASE_DIR / "oidn", BASE_DIR / "OpenEXR.libs", BASE_DIR / "_internal", INITIAL_OIDN_BIN.parent]:
    register_dll_dir(d)

CPU_THREADS = os.cpu_count() or 8
if hasattr(OpenEXR, "set_global_thread_count"):
    OpenEXR.set_global_thread_count(CPU_THREADS)

_LUT_STEPS = 4096
_SRGB_LUT_12BIT = (np.power(np.linspace(0.0, 1.0, _LUT_STEPS, dtype=np.float32), 1.0 / 2.2) * 255.0).astype(np.float32)


def create_minimal_icon(kind: str, size: int = 18, color: str = "#dcdcdc") -> ctk.CTkImage:
    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    lw = max(2, int(1.6 * scale))
    if kind == "stopwatch":
        cx, cy, r = s // 2, int(s * 0.54), int(s * 0.34)
        draw.line([(cx - int(s * 0.12), int(s * 0.10)), (cx + int(s * 0.12), int(s * 0.10))], fill=color, width=lw)
        draw.line([(cx, int(s * 0.10)), (cx, cy - r)], fill=color, width=lw)
        draw.line([(cx + int(r * 0.65), cy - int(r * 0.65)), (cx + int(r * 0.92), cy - int(r * 0.92))], fill=color, width=lw)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=lw)
        draw.line([(cx, cy), (cx + int(r * 0.48), cy - int(r * 0.42))], fill=color, width=lw)
        dr = max(2, lw // 2)
        draw.ellipse((cx - dr, cy - dr, cx + dr, cy + dr), fill=color)
    elif kind == "sliders":
        px, kr = int(s * 0.14), int(s * 0.09)
        for y, kx in zip([int(s * 0.24), int(s * 0.50), int(s * 0.76)], [int(s * 0.64), int(s * 0.36), int(s * 0.58)]):
            draw.line([(px, y), (kx - kr - 2, y)], fill=color, width=lw)
            draw.line([(kx + kr + 2, y), (s - px, y)], fill=color, width=lw)
            draw.ellipse((kx - kr, y - kr, kx + kr, y + kr), outline=color, width=lw)
    smooth = img.resize((size, size), Image.Resampling.LANCZOS)
    return ctk.CTkImage(light_image=smooth, dark_image=smooth, size=(size, size))


# ================= HOVER TOOLTIP WIDGET =================
class CTkToolTip:
    def __init__(self, widget, text: str, delay: int = 320):
        self.widget, self.text, self.delay = widget, text, delay
        self.tip_window = self.after_id = None
        self.widget.bind("<Enter>", self._schedule, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _=None):
        self._unschedule()
        self.after_id = self.widget.after(self.delay, self._show)

    def _unschedule(self):
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def _show(self):
        if self.tip_window or not self.text:
            return
        x, y = self.widget.winfo_pointerx() + 14, self.widget.winfo_pointery() + 14
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        frame = tk.Frame(tw, background="#1e1e1e", highlightbackground="#5a5a5a", highlightthickness=1, bd=0)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text=self.text, justify="left", background="#1e1e1e", foreground="#e8e8e8", font=("Segoe UI", 9), wraplength=330, padx=8, pady=6).pack()

    def _hide(self, _=None):
        self._unschedule()
        if self.tip_window is not None:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None


# ================= SCROLLABLE DROPDOWN WITH SEARCH & STEP BUTTONS =================
class ScrollableDropdown(ctk.CTkFrame):
    def __init__(self, master, variable: ctk.StringVar, values: list, command=None, width: int = 185, display_formatter=None):
        super().__init__(master, fg_color="transparent")
        self.variable, self.values, self.command = variable, list(values), command
        self.display_formatter, self.popup, self.btn_width = display_formatter, None, width
        self.prev_btn = ctk.CTkButton(self, text="◀", width=24, height=28, fg_color="#3a3a3a", hover_color="#505050", command=lambda: self.step(-1))
        self.prev_btn.pack(side="left", padx=(0, 2))
        self.main_btn = ctk.CTkButton(self, text=self._format_btn_text(self.variable.get()), width=width, height=28, anchor="w", fg_color="#1f538d", hover_color="#14375e", command=self.toggle_popup)
        self.main_btn.pack(side="left", padx=0)
        self.next_btn = ctk.CTkButton(self, text="▶", width=24, height=28, fg_color="#3a3a3a", hover_color="#505050", command=lambda: self.step(1))
        self.next_btn.pack(side="left", padx=(2, 0))
        self.variable.trace_add("write", lambda *_: self.refresh_label())

    def _get_label(self, val: str) -> str:
        return self.display_formatter(val) if self.display_formatter else val

    def _format_btn_text(self, val: str) -> str:
        disp = self._get_label(val)
        mc = max(10, int(self.btn_width / 7.2))
        return (disp[: mc - 2] + "… ▾") if len(disp) > mc else f"{disp}  ▾"

    def configure_values(self, new_values: list):
        self.values = list(new_values)
        self.refresh_label()

    def refresh_label(self):
        self.main_btn.configure(text=self._format_btn_text(self.variable.get()))

    def step(self, delta: int):
        if not self.values:
            return
        cur = self.variable.get()
        idx = self.values.index(cur) if cur in self.values else 0
        new_val = self.values[(idx + delta) % len(self.values)]
        if new_val != cur:
            self.variable.set(new_val)
            if self.command:
                self.command(new_val)

    def toggle_popup(self):
        if self.popup is not None and self.popup.winfo_exists():
            self.close_popup()
        else:
            self.open_popup()

    def close_popup(self, _=None):
        if self.popup is not None:
            try:
                self.popup.destroy()
            except Exception:
                pass
            self.popup = None

    def open_popup(self):
        if not self.values:
            return
        self.close_popup()
        self.popup = pop = tk.Toplevel(self)
        pop.wm_overrideredirect(True)
        pop.attributes("-topmost", True)
        x, y = self.main_btn.winfo_rootx(), self.main_btn.winfo_rooty() + self.main_btn.winfo_height() + 2
        pop_w, pop_h = max(self.btn_width + 75, 300), min(max(len(self.values) * 22 + 45, 130), 380)
        pop.wm_geometry(f"{pop_w}x{pop_h}+{x}+{y}")

        outer = tk.Frame(pop, bg="#242424", highlightbackground="#3b8ed0", highlightthickness=1)
        outer.pack(fill="both", expand=True)
        search_var = tk.StringVar()
        search_entry = ctk.CTkEntry(outer, textvariable=search_var, placeholder_text="Search / Filter...", height=26)
        search_entry.pack(fill="x", padx=6, pady=(6, 4))

        list_frame = tk.Frame(outer, bg="#242424")
        list_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        scrollbar = ctk.CTkScrollbar(list_frame, orientation="vertical")
        scrollbar.pack(side="right", fill="y")
        listbox = tk.Listbox(list_frame, bg="#1e1e1e", fg="#e8e8e8", selectbackground="#1f6aa5", selectforeground="#ffffff", font=("Consolas", 10), highlightthickness=0, bd=0, activestyle="none", yscrollcommand=scrollbar.set)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=listbox.yview)

        filtered_items = []

        def populate(filter_str: str = ""):
            listbox.delete(0, "end")
            filtered_items.clear()
            f_low, cur_val, sel_idx = filter_str.lower(), self.variable.get(), None
            for item in self.values:
                disp = self._get_label(item)
                if f_low in item.lower() or f_low in disp.lower():
                    filtered_items.append(item)
                    listbox.insert("end", f" {disp}")
                    if item == cur_val:
                        sel_idx = len(filtered_items) - 1
            if sel_idx is not None:
                listbox.selection_set(sel_idx)
                listbox.see(sel_idx)

        populate()
        search_var.trace_add("write", lambda *_: populate(search_var.get()))

        def on_select(_=None):
            sel = listbox.curselection()
            if sel:
                chosen = filtered_items[sel[0]]
                self.close_popup()
                if chosen != self.variable.get():
                    self.variable.set(chosen)
                    if self.command:
                        self.command(chosen)

        listbox.bind("<ButtonRelease-1>", on_select)
        listbox.bind("<Return>", on_select)
        pop.bind("<Escape>", self.close_popup)
        pop.bind("<FocusOut>", lambda _: self.close_popup() if (self.popup and self.popup.winfo_exists() and self.popup.focus_get() is None) else None)
        search_entry.focus_set()


# ================= DIRECT RAM/VRAM OIDN C-API ENGINE =================
class OIDNNativeDLL:
    DEVICE_MAP = {"default": 0, "cpu": 1, "sycl": 2, "cuda": 3, "hip": 4, "metal": 5}
    FORMAT_FLOAT3, FORMAT_HALF3, STORAGE_DEVICE = 3, 259, 2
    QUALITY_MAP = {"fast": 4, "balanced": 5, "high": 6}

    def __init__(self, dll_path: Path):
        register_dll_dir(dll_path.parent)
        self.dll_path = dll_path
        self.lib = ctypes.CDLL(str(dll_path))
        self._bind_signatures()
        self.device = self.current_device_key = self.buf_shape_key = None
        self.buf_cache = {}
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
                self.lib.oidnReleaseBuffer(b)
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
        return np.ascontiguousarray(np.clip(arr, -65500.0, 65500.0), dtype=np.float16) if use_half else np.ascontiguousarray(arr, dtype=np.float32)

    def upload_aux_to_vram(self, alb_rgb: np.ndarray, nrm_rgb: np.ndarray, use_half: bool):
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
        self, color_rgb: np.ndarray, has_alb: bool, has_nrm: bool, filter_type: str = "RT",
        quality: str = "high", use_half: bool = True, is_hdr: bool = True, clean_aux: bool = True,
        inplace: bool = False, maxmem_mb: int = -1
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


# ================= ARBITRARY-RATIO POLYPHASE & SIMD DOWNSCALER ENGINE =================
def _compute_lanczos_2x(lobes: int) -> np.ndarray:
    taps = lobes * 4
    x = (np.arange(taps, dtype=np.float64) - (taps / 2.0 - 0.5)) * 0.5
    w = np.sinc(x) * np.sinc(x / float(lobes))
    return (w / np.sum(w)).astype(np.float32)


def _compute_bicubic_2x(b: float, c: float) -> np.ndarray:
    x = np.abs((np.arange(8, dtype=np.float64) - 3.5) * 0.5)
    w = np.zeros_like(x)
    m1, m2 = x < 1.0, (x >= 1.0) & (x < 2.0)
    x1, x2 = x[m1], x[m2]
    w[m1] = ((12.0 - 9.0 * b - 6.0 * c) * (x1 ** 3) + (-18.0 + 12.0 * b + 6.0 * c) * (x1 ** 2) + (6.0 - 2.0 * b)) / 6.0
    w[m2] = ((-b - 6.0 * c) * (x2 ** 3) + (6.0 * b + 30.0 * c) * (x2 ** 2) + (-12.0 * b - 48.0 * c) * x2 + (8.0 * b + 24.0 * c)) / 6.0
    return (w / np.sum(w)).astype(np.float32)


class DownscaleEngine:
    K_MKS_DOWN = np.array([1.0, 9.0, 22.0, 22.0, 9.0, 1.0], dtype=np.float32) / 64.0
    K_MKS2021_SHARP = np.array([-1.0, 6.0, -35.0, 204.0, -35.0, 6.0, -1.0], dtype=np.float32) / 144.0
    K_MKS2013_SHARP = np.array([-1.0, 6.0, -1.0], dtype=np.float32) / 4.0
    K_LANCZOS3_DOWN = _compute_lanczos_2x(3)
    K_LANCZOS4_DOWN = _compute_lanczos_2x(4)
    K_MITCHELL_DOWN = _compute_bicubic_2x(1.0 / 3.0, 1.0 / 3.0)
    K_CATROM_DOWN = _compute_bicubic_2x(0.0, 0.5)
    _POLYPHASE_CACHE = OrderedDict()

    @staticmethod
    def compute_target_size(orig_w: int, orig_h: int, scale_mode: str, scale_val_str: str):
        if scale_mode == "No Downscale (100%)":
            return orig_w, orig_h
        try:
            val = float(scale_val_str.strip().replace(",", "."))
        except ValueError:
            val = 50.0
        if scale_mode == "Scale %":
            factor = np.clip(val / 100.0, 0.05, 1.0)
            if abs(factor - 1.0) < 1e-4:
                return orig_w, orig_h
            if abs(factor - 0.5) < 1e-4:
                return max(2, orig_w // 2), max(2, orig_h // 2)
            tw, th = max(2, int(round(orig_w * factor))), max(2, int(round(orig_h * factor)))
        elif scale_mode == "Target Width (px)":
            tw = max(2, min(orig_w, int(round(val))))
            th = max(2, int(round(orig_h * (tw / float(orig_w)))))
        elif scale_mode == "Target Height (px)":
            th = max(2, min(orig_h, int(round(val))))
            tw = max(2, int(round(orig_w * (th / float(orig_h)))))
        else:
            return orig_w, orig_h
        if tw > 4 and (tw % 2 != 0):
            tw += 1
        if th > 4 and (th % 2 != 0):
            th += 1
        return min(orig_w, tw), min(orig_h, th)

    @staticmethod
    def _eval_continuous_kernel(x: np.ndarray, k_type: str) -> np.ndarray:
        ax = np.abs(x)
        if k_type == "mks":
            w = np.zeros_like(ax)
            m1, m2 = ax < 0.5, (ax >= 0.5) & (ax < 1.5)
            w[m1], w[m2] = 0.75 - (ax[m1] ** 2), 0.5 * ((1.5 - ax[m2]) ** 2)
            return w
        if k_type == "lanczos3":
            return np.where(ax < 3.0, np.sinc(x) * np.sinc(x / 3.0), 0.0)
        if k_type == "lanczos4":
            return np.where(ax < 4.0, np.sinc(x) * np.sinc(x / 4.0), 0.0)
        if k_type in ("mitchell", "catrom"):
            b, c = (1.0 / 3.0, 1.0 / 3.0) if k_type == "mitchell" else (0.0, 0.5)
            w = np.zeros_like(ax)
            m1, m2 = ax < 1.0, (ax >= 1.0) & (ax < 2.0)
            x1, x2 = ax[m1], ax[m2]
            w[m1] = ((12.0 - 9.0 * b - 6.0 * c) * (x1 ** 3) + (-18.0 + 12.0 * b + 6.0 * c) * (x1 ** 2) + (6.0 - 2.0 * b)) / 6.0
            w[m2] = ((-b - 6.0 * c) * (x2 ** 3) + (6.0 * b + 30.0 * c) * (x2 ** 2) + (-12.0 * b - 48.0 * c) * x2 + (8.0 * b + 24.0 * c)) / 6.0
            return w
        return np.zeros_like(ax)

    @classmethod
    def _get_polyphase_table(cls, n_in: int, n_out: int, k_type: str):
        key = (n_in, n_out, k_type)
        if key in cls._POLYPHASE_CACHE:
            cls._POLYPHASE_CACHE.move_to_end(key)
            return cls._POLYPHASE_CACHE[key]
        R = {"mks": 1.5, "mitchell": 2.0, "catrom": 2.0, "lanczos3": 3.0, "lanczos4": 4.0}.get(k_type, 1.5)
        scale = float(n_out) / float(n_in)
        f_scale = min(scale, 1.0)
        support = R / f_scale
        num_taps = int(np.ceil(support * 2.0)) + 2
        centers = (np.arange(n_out, dtype=np.float64) + 0.5) / scale - 0.5
        left = np.floor(centers - support).astype(np.int32)
        raw_indices = left[:, None] + np.arange(num_taps, dtype=np.int32)[None, :]
        dist = (raw_indices.astype(np.float64) - centers[:, None]) * f_scale
        weights = cls._eval_continuous_kernel(dist, k_type)
        weights = (weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-8)).astype(np.float32)
        indices = np.clip(raw_indices, 0, n_in - 1).astype(np.int32)
        cls._POLYPHASE_CACHE[key] = (indices, weights)
        if len(cls._POLYPHASE_CACHE) > 32:
            cls._POLYPHASE_CACHE.popitem(last=False)
        return indices, weights

    @classmethod
    def _resample_polyphase_2d(cls, arr: np.ndarray, target_w: int, target_h: int, k_type: str) -> np.ndarray:
        h_in, w_in, ch = arr.shape
        idx_y, w_y = cls._get_polyphase_table(h_in, target_h, k_type)
        tmp = np.zeros((target_h, w_in, ch), dtype=np.float32)
        for t in range(idx_y.shape[1]):
            tmp += arr[idx_y[:, t], :, :] * w_y[:, t : t + 1, None]
        idx_x, w_x = cls._get_polyphase_table(w_in, target_w, k_type)
        out = np.zeros((target_h, target_w, ch), dtype=np.float32)
        for t in range(idx_x.shape[1]):
            out += tmp[:, idx_x[:, t], :] * w_x[None, :, t : t + 1]
        return out

    @classmethod
    def downscale_aux_pass(cls, rgb: np.ndarray, target_w: int, target_h: int, pass_type: str, ds_kernel: str) -> np.ndarray:
        h, w, _ = rgb.shape
        if (target_w, target_h) == (w, h):
            return rgb
        p_low = pass_type.lower()
        if "material_id" in p_low or "object_id" in p_low or p_low.endswith("_id"):
            if rgb.dtype == np.uint32:
                yi = (np.arange(target_h, dtype=np.float64) * (h / float(target_h))).astype(np.int32)
                xi = (np.arange(target_w, dtype=np.float64) * (w / float(target_w))).astype(np.int32)
                return np.ascontiguousarray(rgb[yi[:, None], xi[None, :], :], dtype=np.uint32)
            res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            return np.ascontiguousarray(res[:, :, None] if res.ndim == 2 else res, dtype=np.float32)
        if "mask" in p_low or p_low == "alpha":
            res = cls.downscale(rgb, target_w, target_h, "Mitchell-Netravali (Smooth VFX)")
            return np.ascontiguousarray(np.clip(res, 0.0, 1.0), dtype=np.float32)
        if "depth" in p_low or "position" in p_low or "normal" in p_low:
            return np.ascontiguousarray(cls._resample_polyphase_2d(np.ascontiguousarray(rgb, dtype=np.float32), target_w, target_h, "mitchell"), dtype=np.float32)
        return cls.downscale(rgb, target_w, target_h, ds_kernel)

    @classmethod
    def _apply_mks_sharp(cls, res: np.ndarray, mode_label: str) -> np.ndarray:
        if "2013" in mode_label:
            return cv2.sepFilter2D(res, cv2.CV_32F, cls.K_MKS2013_SHARP, cls.K_MKS2013_SHARP, borderType=cv2.BORDER_REPLICATE)
        if "2021" in mode_label:
            return cv2.sepFilter2D(res, cv2.CV_32F, cls.K_MKS2021_SHARP, cls.K_MKS2021_SHARP, borderType=cv2.BORDER_REPLICATE)
        return res

    @classmethod
    def downscale(cls, rgb: np.ndarray, target_w: int, target_h: int, mode_label: str) -> np.ndarray:
        h, w, c = rgb.shape
        if target_w >= w and target_h >= h:
            return rgb
        exact_2x = (target_w == (w // 2)) and (target_h == (h // 2))
        src = np.ascontiguousarray(rgb[: target_h * 2, : target_w * 2, :] if exact_2x else rgb, dtype=np.float32)
        is_unclamped = ("Unclamped" in mode_label) or ("Catmull-Rom" in mode_label)

        k_type = "mks" if ("MKS" in mode_label or "Magic Kernel" in mode_label) else (
            "lanczos4" if "Lanczos-4" in mode_label else ("lanczos3" if "Lanczos-3" in mode_label else ("catrom" if "Catmull-Rom" in mode_label else "mitchell"))
        )
        k_2x = {"mks": cls.K_MKS_DOWN, "lanczos4": cls.K_LANCZOS4_DOWN, "lanczos3": cls.K_LANCZOS3_DOWN, "catrom": cls.K_CATROM_DOWN, "mitchell": cls.K_MITCHELL_DOWN}[k_type]
        pad = (len(k_2x) // 2) - 1

        if is_unclamped:
            if exact_2x:
                res = cv2.sepFilter2D(src, cv2.CV_32F, k_2x, k_2x, anchor=(pad, pad), borderType=cv2.BORDER_REPLICATE)[0::2, 0::2, :]
            else:
                res = cls._resample_polyphase_2d(src, target_w, target_h, k_type)
            return np.ascontiguousarray(np.maximum(cls._apply_mks_sharp(res, mode_label), 0.0), dtype=np.float32)

        if exact_2x:
            b00, b01, b10, b11 = src[0::2, 0::2, :], src[0::2, 1::2, :], src[1::2, 0::2, :], src[1::2, 1::2, :]
            local_min = np.minimum(np.minimum(b00, b01), np.minimum(b10, b11))
            local_max = np.maximum(np.maximum(b00, b01), np.maximum(b10, b11))
        else:
            area_ref = cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_AREA)
            if area_ref.ndim == 2:
                area_ref = area_ref[:, :, None]
            k3 = np.ones((3, 3), dtype=np.uint8)
            local_min, local_max = cv2.erode(area_ref, k3), cv2.dilate(area_ref, k3)
            if local_min.ndim == 2:
                local_min, local_max = local_min[:, :, None], local_max[:, :, None]

        lum = (0.2126 * src[:, :, 0] + 0.7152 * src[:, :, 1] + 0.0722 * src[:, :, 2]) if c >= 3 else src[:, :, 0]
        w_karis = (1.0 / (1.0 + np.maximum(lum, 0.0)))[:, :, None]
        packed = np.concatenate([src * w_karis, w_karis], axis=2)

        if exact_2x:
            dec = cv2.sepFilter2D(packed, cv2.CV_32F, k_2x, k_2x, anchor=(pad, pad), borderType=cv2.BORDER_REPLICATE)[0::2, 0::2, :]
        else:
            dec = cls._resample_polyphase_2d(packed, target_w, target_h, k_type)

        res = cls._apply_mks_sharp(dec[:, :, :c] / np.maximum(dec[:, :, c : c + 1], 1e-7), mode_label)
        if "Lanczos-3" in mode_label:
            res = np.clip(res, local_min, local_max)
        else:
            span = (local_max - local_min) * 0.08
            res = np.clip(res, np.maximum(local_min - span, 0.0), local_max + span)
        return np.ascontiguousarray(np.maximum(res, 0.0), dtype=np.float32)
# === END OF PART 1 (PASTE PART 2 DIRECTLY BELOW) ===
# === START OF PART 2 ===
class PolyDenoiseStudio(ctk.CTk):
    BEAUTY_SUM_KEY = "[⚡ Sum All Lights (Beauty)]"
    ZOOM_STEPS = [25, 33, 50, 67, 75, 100, 125, 150, 200, 250, 300, 400]

    AUX_ALIASES = {
        "normals_pre_bump": "normals_pre_bump", "normal_pre_bump": "normals_pre_bump",
        "normals_shading": "normals", "shading_normal": "normals", "normals": "normals", "normal": "normals",
        "albedo": "albedo", "diffuse_albedo": "albedo", "alpha": "alpha",
        "depth": "depth", "zdepth": "depth", "z_depth": "depth",
        "material_id": "material_id", "mat_id": "material_id",
        "object_id": "object_id", "obj_id": "object_id",
        "position": "position", "world_position": "position"
    }

    KNOWN_STD_LIGHT_PASSES = [
        "direct_lighting", "indirect_lighting", "emission_lighting", "specular_reflection_lighting",
        "refraction_lighting", "transmission_lighting", "sss_lighting", "participating_media_lighting",
        "beauty", "combined"
    ]

    def __init__(self):
        super().__init__()
        exr_ver = getattr(OpenEXR, "__version__", "3.5+")
        self.title(f"{APP_NAME} v{APP_VERSION} — Direct RAM/VRAM Engine (OpenEXR v{exr_ver})")
        self.geometry("1440x940")
        self._icon_photo_ref = None
        self._apply_window_icon(self)

        self.icon_stopwatch = create_minimal_icon("stopwatch", size=17, color="#dcdcdc")
        self.icon_sliders = create_minimal_icon("sliders", size=17, color="#dcdcdc")

        self.oidn_bin = INITIAL_OIDN_BIN
        self.native_oidn = None
        self._try_init_native_dll()

        self.blacklist = {"albedo", "alpha", "depth", "material_id", "normals", "normals_pre_bump", "object_id", "position"}

        self.compression_map = {
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
            "NONE (Uncompressed)": ("NO_COMPRESSION", "")
        }

        self.aux_compression_options = [
            "ZIP (Lossless 16-Line)", "ZSTD (v3.5 Lossless)", "RLE (Lossless Flat/ID)",
            "ZIPS (Lossless 1-Line)", "PIZ (Wavelet Lossless)", "HTJ2K32 (v3.4 Fast Lossless)"
        ]

        self.downscale_options = [
            "Magic Kernel Sharp 2013 (Crisp / HDR-Safe)",
            "MKS 2013 Unclamped (Ultra Crisp / Pure Linear)",
            "Lanczos-4 Unclamped (16-Tap / Extreme Acuity)",
            "Catmull-Rom Bicubic (ArchViz Sharp / Unclamped)",
            "Magic Kernel Sharp 2021 (Balanced SOTA)",
            "Lanczos-3 + Anti-Ringing (Max HDR Edge)",
            "Mitchell-Netravali (Smooth VFX)"
        ]

        self.scale_mode_options = ["Scale %", "Target Width (px)", "Target Height (px)", "No Downscale (100%)"]

        self.input_folder = None
        self.render_queue, self.aux_mask_files = [], []
        self.frames_data, self.denoised_data, self.empty_passes = {}, {}, {}
        self.last_known_orig_dims = (4096, 3072)

        self.linear_lru_cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.primary_linear = self.alternate_linear = self.primary_rgb = self.alternate_rgb = None
        self.primary_tag = self.alternate_tag = ""
        self.tk_img_primary = self.tk_img_alternate = self.canvas_img_id = None
        self._last_canvas_dims = (0, 0)
        self._img_offset_x = self._img_offset_y = 0
        self._dither_cache = {}
        self.is_holding_compare = self.show_denoised_default = False
        self.config_window = self.bench_window = self.bench_process = None
        self.bench_running = self.bench_cancel_flag = False

        self._scan_generation = 0
        self._spinner_active = False
        self._spinner_angle = 0
        self._spinner_text = ""
        self._spinner_after_id = None

        self.is_running = False
        self.cancel_event = threading.Event()
        self.active_process = None

        self.setup_ui()
        self.bind("<KeyPress-space>", self._on_space_press)
        self.bind("<KeyRelease-space>", self._on_space_release)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self._scan_generation += 1
        self.cancel_event.set()
        self.bench_cancel_flag = True
        if self.bench_process and self.bench_process.poll() is None:
            try:
                self.bench_process.terminate()
            except Exception:
                pass
        if self.native_oidn is not None:
            self.native_oidn.shutdown()
        self.destroy()

    def _apply_window_icon(self, target_win):
        ico_candidates = [BASE_DIR / "polydenoise.ico", BASE_DIR / "_internal" / "polydenoise.ico"]
        png_candidates = [BASE_DIR / "logo.png", BASE_DIR / "icon_source.png", BASE_DIR / "_internal" / "logo.png"]

        def _set():
            if not target_win.winfo_exists():
                return
            for ico_p in ico_candidates:
                if ico_p.exists():
                    try:
                        target_win.iconbitmap(str(ico_p))
                        return
                    except Exception:
                        pass
            for png_p in png_candidates:
                if png_p.exists():
                    try:
                        if self._icon_photo_ref is None:
                            pil_ic = Image.open(png_p).convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)
                            self._icon_photo_ref = ImageTk.PhotoImage(pil_ic)
                        target_win.iconphoto(False, self._icon_photo_ref)
                        return
                    except Exception:
                        pass

        _set()
        target_win.after(250, _set)

    def _try_init_native_dll(self):
        for dll_p in [
            BASE_DIR / "oidn" / "OpenImageDenoise.dll", BASE_DIR / "OpenImageDenoise.dll",
            BASE_DIR / "_internal" / "OpenImageDenoise.dll", self.oidn_bin.parent / "OpenImageDenoise.dll"
        ]:
            if dll_p.exists():
                try:
                    self.native_oidn = OIDNNativeDLL(dll_p)
                    return
                except Exception:
                    pass
        self.native_oidn = None

    def get_oidn_benchmark_exe(self) -> Path:
        search_dirs = [BASE_DIR / "oidn", BASE_DIR, BASE_DIR / "_internal", self.oidn_bin.parent]
        if self.native_oidn is not None:
            search_dirs.insert(0, self.native_oidn.dll_path.parent)
        for d in search_dirs:
            candidate = d / "oidnBenchmark.exe"
            if candidate.exists():
                return candidate
        return BASE_DIR / "oidn" / "oidnBenchmark.exe"

    def get_oidn_version_str(self) -> str:
        return self.native_oidn.version_str if self.native_oidn is not None else "Unknown"

    def save_config(self):
        try:
            CONFIG_FILE.write_text(json.dumps({"oidn_bin": str(self.oidn_bin)}, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _format_channel_label(self, ch_name: str) -> str:
        if ch_name == self.BEAUTY_SUM_KEY:
            return ch_name
        frame = self.frame_var.get()
        if frame in self.empty_passes and ch_name in self.empty_passes[frame]:
            return f"⚫ {ch_name} [Empty]"
        if frame in self.denoised_data and ch_name in self.denoised_data[frame]:
            return f"✓ {ch_name}"
        return f"  {ch_name}"

    # ================= ANIMATED VIEWPORT SPINNER OVERLAY =================
    def start_viewport_spinner(self, text: str = "Loading in background..."):
        self._spinner_text = text
        if not self._spinner_active:
            self._spinner_active = True
            self._animate_spinner_step()
        else:
            self._draw_spinner_overlay()

    def stop_viewport_spinner(self):
        self._spinner_active = False
        if self._spinner_after_id is not None:
            try:
                self.after_cancel(self._spinner_after_id)
            except Exception:
                pass
            self._spinner_after_id = None
        self.preview_canvas.delete("spinner_overlay")

    def _animate_spinner_step(self):
        if not self._spinner_active or not self.preview_canvas.winfo_exists():
            return
        self._spinner_angle = (self._spinner_angle - 28) % 360
        self._draw_spinner_overlay()
        self._spinner_after_id = self.after(33, self._animate_spinner_step)

    def _draw_spinner_overlay(self):
        self.preview_canvas.delete("spinner_overlay")
        if not self._spinner_active:
            return
        cx, cy = self.preview_canvas.canvasx(16), self.preview_canvas.canvasy(16)
        box_w, box_h = max(210, len(self._spinner_text) * 7 + 52), 34
        self.preview_canvas.create_rectangle(cx, cy, cx + box_w, cy + box_h, fill="#141414", outline="#3b8ed0", width=1, tags="spinner_overlay")
        ax0, ay0, ax1, ay1 = cx + 9, cy + 7, cx + 29, cy + 27
        self.preview_canvas.create_oval(ax0, ay0, ax1, ay1, outline="#2c3e50", width=3, tags="spinner_overlay")
        self.preview_canvas.create_arc(ax0, ay0, ax1, ay1, start=self._spinner_angle, extent=110, style="arc", outline="#4da3ff", width=3, tags="spinner_overlay")
        self.preview_canvas.create_text(cx + 38, cy + 17, text=self._spinner_text, fill="#e8e8e8", anchor="w", font=("Segoe UI", 9, "bold"), tags="spinner_overlay")

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=375)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar.grid_propagate(False)

        bold_font = ctk.CTkFont(size=14, weight="bold")
        lbl_font = ctk.CTkFont(size=11)

        ctk.CTkLabel(self.sidebar, text="1. Source Folder & Settings", font=bold_font).pack(pady=(8, 3), anchor="w", padx=10)
        p_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        p_row.pack(fill="x", padx=8, pady=2)

        self.folder_btn = ctk.CTkButton(p_row, text="Select Render Folder", height=28, command=self.select_folder)
        self.folder_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        CTkToolTip(self.folder_btn, "Select folder with EXR render passes.\nDisplays the first light pass immediately while scanning the rest in the background!")

        self.cfg_btn = ctk.CTkButton(p_row, text="", image=self.icon_sliders, width=32, height=28, fg_color="#2b2b2b", hover_color="#3f3f3f", corner_radius=6, command=self.open_config_popup)
        self.cfg_btn.pack(side="right")
        CTkToolTip(self.cfg_btn, "Configuration, OIDN Path, Version Check & Credits")

        self.bench_open_btn = ctk.CTkButton(p_row, text="", image=self.icon_stopwatch, width=32, height=28, fg_color="#2b2b2b", hover_color="#3f3f3f", corner_radius=6, command=self.open_benchmark_popup)
        self.bench_open_btn.pack(side="right", padx=(0, 4))
        CTkToolTip(self.bench_open_btn, "OIDN Hardware Benchmark (GPU vs. CPU)")

        self.folder_label = ctk.CTkLabel(self.sidebar, text=f"OIDN v{self.get_oidn_version_str()} | Folder: None", font=lbl_font, text_color="gray", wraplength=350)
        self.folder_label.pack(padx=10, pady=(2, 4), anchor="w")

        ctk.CTkLabel(self.sidebar, text="2. Downscaling & OpenEXR v3.5", font=bold_font).pack(pady=(4, 2), anchor="w", padx=10)
        ctk.CTkLabel(self.sidebar, text="Downscale Filter Kernel", font=lbl_font).pack(anchor="w", padx=10)
        self.downscale_var = ctk.StringVar(value=self.downscale_options[0])
        self.downscale_menu = ctk.CTkOptionMenu(self.sidebar, variable=self.downscale_var, values=self.downscale_options, height=26, command=self.on_downscale_mode_change)
        self.downscale_menu.pack(fill="x", padx=8, pady=(0, 4))
        CTkToolTip(self.downscale_menu, "Filter kernel applied AFTER full-res OIDN denoising.\nMaterial/Object IDs automatically use Nearest-Neighbor & preserve uint32!")

        scale_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        scale_row.pack(fill="x", padx=8, pady=(0, 4))
        s_col1 = ctk.CTkFrame(scale_row, fg_color="transparent")
        s_col1.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(s_col1, text="Scale Mode", font=lbl_font).pack(anchor="w")
        self.scale_mode_var = ctk.StringVar(value="Scale %")
        self.scale_mode_menu = ctk.CTkOptionMenu(s_col1, variable=self.scale_mode_var, values=self.scale_mode_options, width=145, height=26, command=self.on_scale_mode_select)
        self.scale_mode_menu.pack()

        s_col2 = ctk.CTkFrame(scale_row, fg_color="transparent")
        s_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(s_col2, text="Value", font=lbl_font).pack(anchor="w")
        self.scale_val_entry = ctk.CTkEntry(s_col2, width=68, height=26)
        self.scale_val_entry.insert(0, "50")
        self.scale_val_entry.pack()
        self.scale_val_entry.bind("<Return>", lambda _: self.on_downscale_mode_change())
        self.scale_val_entry.bind("<FocusOut>", lambda _: self.on_downscale_mode_change())

        s_col3 = ctk.CTkFrame(scale_row, fg_color="transparent")
        s_col3.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(s_col3, text="Output Size", font=lbl_font).pack(anchor="w")
        self.scale_out_label = ctk.CTkLabel(s_col3, text="50.0%", font=lbl_font, text_color="#8ab4f8")
        self.scale_out_label.pack(anchor="w", pady=(2, 0))

        exr_grid = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        exr_grid.pack(fill="x", padx=8, pady=(0, 4))
        c_col1 = ctk.CTkFrame(exr_grid, fg_color="transparent")
        c_col1.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkLabel(c_col1, text="Light EXR Codec", font=lbl_font).pack(anchor="w")
        self.comp_var = ctk.StringVar(value="DWAB (Lossy 256-Line)")
        self.comp_menu = ctk.CTkOptionMenu(c_col1, variable=self.comp_var, values=list(self.compression_map.keys()), width=155, height=26, command=self.on_comp_change)
        self.comp_menu.pack(fill="x")

        c_col2 = ctk.CTkFrame(exr_grid, fg_color="transparent")
        c_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(c_col2, text="Level", font=lbl_font).pack(anchor="w")
        self.comp_level_entry = ctk.CTkEntry(c_col2, width=52, height=26)
        self.comp_level_entry.insert(0, "20")
        self.comp_level_entry.pack()

        c_col3 = ctk.CTkFrame(exr_grid, fg_color="transparent")
        c_col3.pack(side="right")
        ctk.CTkLabel(c_col3, text="Bit Depth", font=lbl_font).pack(anchor="w")
        self.exr_depth_var = ctk.StringVar(value="HALF (16-bit)")
        self.exr_depth_menu = ctk.CTkOptionMenu(c_col3, variable=self.exr_depth_var, values=["HALF (16-bit)", "FLOAT (32-bit)"], width=125, height=26)
        self.exr_depth_menu.pack()

        aux_codec_row = ctk.CTkFrame(self.sidebar, fg_color="#222222", corner_radius=6)
        aux_codec_row.pack(fill="x", padx=8, pady=(0, 6))
        self.sep_aux_comp_var = ctk.BooleanVar(value=True)
        self.sep_aux_cb = ctk.CTkCheckBox(aux_codec_row, text="Separate Aux/Mask/ID Codec:", variable=self.sep_aux_comp_var, font=lbl_font, checkbox_width=18, checkbox_height=18, command=self._on_toggle_aux_codec)
        self.sep_aux_cb.pack(side="left", padx=(8, 4), pady=5)

        self.aux_comp_var = ctk.StringVar(value="ZIP (Lossless 16-Line)")
        self.aux_comp_menu = ctk.CTkOptionMenu(aux_codec_row, variable=self.aux_comp_var, values=self.aux_compression_options, width=165, height=24)
        self.aux_comp_menu.pack(side="right", padx=(0, 6), pady=5)

        ctk.CTkLabel(self.sidebar, text="3. OIDN & Guide Parameters", font=bold_font).pack(pady=(2, 2), anchor="w", padx=10)
        row_aux = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row_aux.pack(fill="x", padx=8, pady=(0, 4))

        a_col1 = ctk.CTkFrame(row_aux, fg_color="transparent")
        a_col1.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkLabel(a_col1, text="Aux Guide Passes", font=lbl_font).pack(anchor="w")
        self.aux_mode_var = ctk.StringVar(value="Albedo + Normals (Bump)")
        self.aux_mode_menu = ctk.CTkOptionMenu(a_col1, variable=self.aux_mode_var, width=170, height=26, values=["Albedo + Normals (Bump)", "Albedo + Normals (Pre-Bump)", "Albedo Only", "None (Color Only)"])
        self.aux_mode_menu.pack(fill="x")

        a_col2 = ctk.CTkFrame(row_aux, fg_color="transparent")
        a_col2.pack(side="right", fill="x", expand=True)
        ctk.CTkLabel(a_col2, text="Noise De-Correlation", font=lbl_font).pack(anchor="w")
        self.decorr_var = ctk.StringVar(value="Standard 1x (1px Noise)")
        self.decorr_menu = ctk.CTkOptionMenu(a_col2, variable=self.decorr_var, width=165, height=26, values=["Standard 1x (1px Noise)", "2x2 Stride + 1x Polish", "3x3 Stride + 1x Polish"])
        self.decorr_menu.pack(fill="x")

        row_dqp = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row_dqp.pack(fill="x", padx=8, pady=(0, 4))
        d_col1 = ctk.CTkFrame(row_dqp, fg_color="transparent")
        d_col1.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(d_col1, text="Device (-d)", font=lbl_font).pack(anchor="w")
        self.device_var = ctk.StringVar(value="cuda")
        self.device_menu = ctk.CTkOptionMenu(d_col1, variable=self.device_var, values=["cuda", "cpu", "default", "sycl", "hip", "metal"], width=105, height=26)
        self.device_menu.pack()

        d_col2 = ctk.CTkFrame(row_dqp, fg_color="transparent")
        d_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(d_col2, text="Quality (-q)", font=lbl_font).pack(anchor="w")
        self.quality_var = ctk.StringVar(value="high")
        self.quality_menu = ctk.CTkOptionMenu(d_col2, variable=self.quality_var, values=["high", "balanced", "fast"], width=115, height=26)
        self.quality_menu.pack()

        d_col3 = ctk.CTkFrame(row_dqp, fg_color="transparent")
        d_col3.pack(side="right")
        ctk.CTkLabel(d_col3, text="Precision (-t)", font=lbl_font).pack(anchor="w")
        self.precision_var = ctk.StringVar(value="half")
        self.precision_menu = ctk.CTkOptionMenu(d_col3, variable=self.precision_var, values=["half", "float"], width=110, height=26)
        self.precision_menu.pack()

        row_hfv = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row_hfv.pack(fill="x", padx=8, pady=(0, 4))
        h_col1 = ctk.CTkFrame(row_hfv, fg_color="transparent")
        h_col1.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(h_col1, text="HDR Mode", font=lbl_font).pack(anchor="w")
        self.hdr_mode_var = ctk.StringVar(value="HDR Linear")
        self.hdr_mode_menu = ctk.CTkOptionMenu(h_col1, variable=self.hdr_mode_var, values=["HDR Linear", "HDR Clamp Fireflies", "LDR Normalized"], width=150, height=26)
        self.hdr_mode_menu.pack()

        h_col2 = ctk.CTkFrame(row_hfv, fg_color="transparent")
        h_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(h_col2, text="Filter (-f)", font=lbl_font).pack(anchor="w")
        self.filter_var = ctk.StringVar(value="RT")
        self.filter_menu = ctk.CTkOptionMenu(h_col2, variable=self.filter_var, values=["RT", "RTLightmap"], width=105, height=26)
        self.filter_menu.pack()

        h_col3 = ctk.CTkFrame(row_hfv, fg_color="transparent")
        h_col3.pack(side="right")
        ctk.CTkLabel(h_col3, text="Verbose", font=lbl_font).pack(anchor="w")
        self.verbose_var = ctk.StringVar(value="2")
        self.verbose_menu = ctk.CTkOptionMenu(h_col3, variable=self.verbose_var, values=["0", "1", "2", "3", "4"], width=75, height=26)
        self.verbose_menu.pack()

        cb_grid = ctk.CTkFrame(self.sidebar, fg_color="#222222", corner_radius=6)
        cb_grid.pack(fill="x", padx=8, pady=(2, 5))
        cb_grid.grid_columnconfigure(0, weight=1)
        cb_grid.grid_columnconfigure(1, weight=1)

        self.clean_aux_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(cb_grid, text="CleanAux (--clean_aux)", variable=self.clean_aux_var, font=lbl_font, checkbox_width=18, checkbox_height=18).grid(row=0, column=0, sticky="w", padx=8, pady=(5, 2))

        self.inplace_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(cb_grid, text="Inplace VRAM", variable=self.inplace_var, font=lbl_font, checkbox_width=18, checkbox_height=18).grid(row=0, column=1, sticky="w", padx=8, pady=(5, 2))

        self.downscale_aux_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(cb_grid, text="Scale Aux & Masks", variable=self.downscale_aux_var, font=lbl_font, checkbox_width=18, checkbox_height=18).grid(row=1, column=0, sticky="w", padx=8, pady=2)

        self.async_save_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(cb_grid, text="Async RAM Save", variable=self.async_save_var, font=lbl_font, checkbox_width=18, checkbox_height=18).grid(row=1, column=1, sticky="w", padx=8, pady=2)

        self.skip_empty_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(cb_grid, text="Skip Empty (Black)", variable=self.skip_empty_var, font=lbl_font, checkbox_width=18, checkbox_height=18, command=lambda: self.parse_indigo_passes() if self.input_folder else None).grid(row=2, column=0, sticky="w", padx=8, pady=(2, 5))

        self.skip_existing_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(cb_grid, text="Skip Done (Resume)", variable=self.skip_existing_var, font=lbl_font, checkbox_width=18, checkbox_height=18, command=lambda: self.parse_indigo_passes() if self.input_folder else None).grid(row=2, column=1, sticky="w", padx=8, pady=(2, 5))

        extra_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        extra_frame.pack(fill="x", padx=8, pady=(0, 5))
        self.maxmem_entry = ctk.CTkEntry(extra_frame, placeholder_text="MaxMem MB", width=110, height=26)
        self.maxmem_entry.pack(side="left", padx=(0, 5))

        self.future_args = ctk.CTkEntry(extra_frame, placeholder_text="Custom CLI Args (e.g. OIDN v3)", height=26)
        self.future_args.pack(side="right", fill="x", expand=True)

        self.test_single_btn = ctk.CTkButton(self.sidebar, text="Test Single Pass (Current Channel)", command=self.start_single_test_thread, fg_color="#1f6aa5", height=34, state="disabled")
        self.test_single_btn.pack(fill="x", padx=8, pady=(2, 4))

        self.start_btn = ctk.CTkButton(self.sidebar, text="Start Full Batch Denoise", command=self.on_start_or_cancel_click, fg_color="#2b7a2b", hover_color="#1f5c1f", height=38, state="disabled")
        self.start_btn.pack(fill="x", padx=8, pady=(2, 8))

        # ================= MAIN AREA =================
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.main_frame.grid_rowconfigure(1, weight=5)
        self.main_frame.grid_rowconfigure(3, weight=2)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.top_bar = ctk.CTkFrame(self.main_frame)
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))

        ctk.CTkLabel(self.top_bar, text="Frame:").pack(side="left", padx=(8, 3), pady=6)
        self.frame_var = ctk.StringVar(value="-")
        self.frame_selector = ScrollableDropdown(self.top_bar, variable=self.frame_var, values=["-"], command=self.on_frame_change, width=135)
        self.frame_selector.pack(side="left", padx=(0, 6), pady=6)

        ctk.CTkLabel(self.top_bar, text="Channel:").pack(side="left", padx=(2, 3), pady=6)
        self.channel_var = ctk.StringVar(value="-")
        self.channel_selector = ScrollableDropdown(self.top_bar, variable=self.channel_var, values=["-"], command=self.on_channel_change, width=195, display_formatter=self._format_channel_label)
        self.channel_selector.pack(side="left", padx=(0, 6), pady=6)

        self.swap_default_btn = ctk.CTkButton(self.top_bar, text="Hold: Denoised", width=105, height=28, fg_color="#3b5998", hover_color="#2d4373", command=self.toggle_default_view_mode)
        self.swap_default_btn.pack(side="left", padx=(0, 6), pady=6)

        ctk.CTkLabel(self.top_bar, text="Zoom:").pack(side="left", padx=(2, 2), pady=6)
        self.zoom_var = ctk.StringVar(value="100%")
        self.zoom_menu = ctk.CTkOptionMenu(
            self.top_bar, variable=self.zoom_var,
            values=["Fit"] + [f"{z}%" for z in self.ZOOM_STEPS],
            width=78, height=28, command=self.on_zoom_change
        )
        self.zoom_menu.pack(side="left", padx=(0, 6), pady=6)
        CTkToolTip(self.zoom_menu, "Scroll MouseWheel directly on the image to zoom in/out at the mouse cursor position!")

        ctk.CTkLabel(self.top_bar, text="EV:").pack(side="left", padx=(2, 2), pady=6)
        self.ev_var = ctk.DoubleVar(value=0.0)
        self.ev_slider = ctk.CTkSlider(self.top_bar, from_=-5.0, to=5.0, number_of_steps=100, variable=self.ev_var, width=80, command=self.on_ev_change)
        self.ev_slider.pack(side="left", padx=(0, 3), pady=6)
        self.ev_reset_btn = ctk.CTkButton(self.top_bar, text="0.0", width=36, height=24, fg_color="#3a3a3a", command=self.reset_ev)
        self.ev_reset_btn.pack(side="left", padx=(0, 6), pady=6)

        self.res_label = ctk.CTkLabel(self.top_bar, text="", text_color="#8ab4f8", font=lbl_font)
        self.res_label.pack(side="right", padx=8)

        self.canvas_container = ctk.CTkFrame(self.main_frame)
        self.canvas_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(self.canvas_container, bg="#1a1a1a", highlightthickness=0, cursor="crosshair")
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")

        self.v_scroll = ctk.CTkScrollbar(self.canvas_container, orientation="vertical", command=self.preview_canvas.yview)
        self.h_scroll = ctk.CTkScrollbar(self.canvas_container, orientation="horizontal", command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        self.preview_canvas.bind("<ButtonPress-1>", self.on_canvas_press_compare)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_canvas_release_compare)
        self.preview_canvas.bind("<Leave>", self.on_canvas_release_compare)
        self.preview_canvas.bind("<ButtonPress-2>", lambda e: self.preview_canvas.scan_mark(e.x, e.y))
        self.preview_canvas.bind("<B2-Motion>", lambda e: self.preview_canvas.scan_dragto(e.x, e.y, gain=1))
        self.preview_canvas.bind("<ButtonPress-3>", lambda e: self.preview_canvas.scan_mark(e.x, e.y))
        self.preview_canvas.bind("<B3-Motion>", lambda e: self.preview_canvas.scan_dragto(e.x, e.y, gain=1))
        self.preview_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.preview_canvas.bind("<Configure>", self._on_canvas_configure)

        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=0)
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=12)
        self.progress_bar.set(0.0)
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="", width=140, anchor="e", font=lbl_font)

        self.clear_log_btn = ctk.CTkButton(self.progress_frame, text="Clear Log", width=85, height=22, fg_color="#444444", hover_color="#5a5a5a", command=self.clear_log)
        self.clear_log_btn.grid(row=0, column=2, sticky="e", padx=(0, 2), pady=2)

        self.log_textbox = ctk.CTkTextbox(self.main_frame, height=170, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_textbox.grid(row=3, column=0, sticky="nsew", padx=10, pady=(2, 8))

        self.log(f"{APP_NAME} v{APP_VERSION} | OpenEXR v{getattr(OpenEXR, '__version__', 'Unknown')} ({CPU_THREADS} Threads).")
        if self.native_oidn is not None:
            self.log(f"DIRECT RAM/VRAM ENGINE ACTIVE: OIDN v{self.native_oidn.version_str} ({self.native_oidn.dll_path})")
        elif self.oidn_bin.exists():
            self.log(f"OIDN CLI fallback loaded: {self.oidn_bin}")
        else:
            self.log("WARNING: OIDN not found. Place OIDN files in './oidn/' or click Settings to select OpenImageDenoise.dll.")

    # ================= OIDN HARDWARE BENCHMARK POPUP (CPU & GPU) =================
    def open_benchmark_popup(self):
        if self.bench_window is not None and self.bench_window.winfo_exists():
            self.bench_window.focus_force()
            return

        self.bench_window = win = ctk.CTkToplevel(self)
        win.title(f"{APP_NAME} — OIDN Hardware Benchmark (CPU & GPU)")
        win.geometry("650x530")
        win.minsize(600, 460)
        win.attributes("-topmost", True)
        self._apply_window_icon(win)

        bold_f, lbl_f, mono_f = ctk.CTkFont(size=14, weight="bold"), ctk.CTkFont(size=11), ctk.CTkFont(family="Consolas", size=12)
        top_box = ctk.CTkFrame(win, corner_radius=8)
        top_box.pack(fill="x", padx=12, pady=(12, 6))

        bench_exe = self.get_oidn_benchmark_exe()
        status_txt = f"Binary: {bench_exe}" if bench_exe.exists() else f"MISSING: {bench_exe} (Put oidnBenchmark.exe in ./oidn/)"
        status_col = "#8ab4f8" if bench_exe.exists() else "#e57373"

        ctk.CTkLabel(top_box, text="Intel® OIDN Standalone Benchmark (oidnBenchmark.exe)", font=bold_f).pack(anchor="w", padx=12, pady=(8, 2))
        self.bench_bin_lbl = ctk.CTkLabel(top_box, text=status_txt, font=lbl_f, text_color=status_col, wraplength=600, justify="left")
        self.bench_bin_lbl.pack(anchor="w", padx=12, pady=(0, 8))

        ctrl_row = ctk.CTkFrame(top_box, fg_color="transparent")
        ctrl_row.pack(fill="x", padx=12, pady=(0, 10))

        cur_gpu = self.device_var.get() if self.device_var.get() != "cpu" else "cuda"
        self.bench_mode_var = ctk.StringVar(value=f"GPU ({cur_gpu}) + CPU Comparison")
        ctk.CTkOptionMenu(ctrl_row, variable=self.bench_mode_var, width=215, height=28, values=[f"GPU ({cur_gpu}) + CPU Comparison", f"GPU Only ({cur_gpu})", "CPU Only (cpu)"]).pack(side="left", padx=(0, 8))

        self.bench_scope_var = ctk.StringVar(value="Quick Production (6 Tests/Device)")
        scope_menu = ctk.CTkOptionMenu(ctrl_row, variable=self.bench_scope_var, width=255, height=28, values=["Quick Production (6 Tests/Device)", "Quick CleanAux Only (3 Tests/Device)", "Ultra-Fast 1080p (1 Test/Device)", "Full Suite (All 15 Tests/Device)"])
        scope_menu.pack(side="left", padx=(0, 8))

        self.bench_run_btn = ctk.CTkButton(ctrl_row, text="▶ Run", width=105, height=28, fg_color="#2b7a2b", hover_color="#1f5c1f", state="normal" if bench_exe.exists() else "disabled", command=self._on_bench_run_or_stop)
        self.bench_run_btn.pack(side="right")

        self.bench_prog_frame = ctk.CTkFrame(win, fg_color="transparent")
        self.bench_prog_lbl = ctk.CTkLabel(self.bench_prog_frame, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color="#4da3ff", anchor="w")
        self.bench_prog_lbl.pack(fill="x", padx=2, pady=(0, 2))
        self.bench_pbar = ctk.CTkProgressBar(self.bench_prog_frame, height=10)
        self.bench_pbar.set(0.0)
        self.bench_pbar.pack(fill="x", padx=2, pady=(0, 4))

        self.bench_textbox = ctk.CTkTextbox(win, font=mono_f)
        self.bench_textbox.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        self._bench_log("Ready. Select mode above and click '▶ Run' to benchmark your GPU & CPU.\n")

    def _bench_log(self, text: str):
        if self.bench_window is not None and self.bench_window.winfo_exists():
            self.bench_textbox.insert("end", text + "\n")
            self.bench_textbox.see("end")

    def _bench_replace_running_line(self, final_line: str):
        if self.bench_window is not None and self.bench_window.winfo_exists():
            lines = self.bench_textbox.get("1.0", "end-1c").split("\n")
            for idx in range(len(lines) - 1, -1, -1):
                if lines[idx].endswith("running..."):
                    lines[idx] = final_line
                    break
            else:
                lines.append(final_line)
            clean_text = "\n".join(lines) + ("\n" if not lines[-1].endswith("\n") else "")
            self.bench_textbox.delete("1.0", "end")
            self.bench_textbox.insert("1.0", clean_text)
            self.bench_textbox.see("end")

    def _update_bench_progress(self, dev_idx: int, dev_total: int, status_text=None):
        if self.bench_window is not None and self.bench_window.winfo_exists():
            self.bench_pbar.set(min(1.0, max(0.0, dev_idx / float(max(dev_total, 1)))))
            if status_text is not None:
                self.bench_prog_lbl.configure(text=status_text, text_color="#4da3ff")

    def _on_bench_run_or_stop(self):
        if self.bench_running:
            self.bench_cancel_flag = True
            if self.bench_process and self.bench_process.poll() is None:
                try:
                    self.bench_process.terminate()
                except Exception:
                    pass
            self._bench_log("-> Stopping benchmark...")
        else:
            self._start_benchmark_thread()

    def _start_benchmark_thread(self):
        bench_exe = self.get_oidn_benchmark_exe()
        if not bench_exe.exists():
            self._bench_log(f"ERROR: oidnBenchmark.exe not found at {bench_exe}")
            return

        self.bench_running, self.bench_cancel_flag = True, False
        self.bench_textbox.delete("1.0", "end")
        self.bench_run_btn.configure(text="■ Stop", fg_color="#8b2626", hover_color="#a83232")
        self.bench_pbar.set(0.0)
        self.bench_prog_lbl.configure(text="Starting...", text_color="#4da3ff")
        self.bench_prog_frame.pack(fill="x", padx=12, pady=(0, 4), before=self.bench_textbox)

        mode_str, scope_str = self.bench_mode_var.get(), self.bench_scope_var.get()
        cur_gpu = self.device_var.get() if self.device_var.get() != "cpu" else "cuda"
        devices_to_run = [cur_gpu, "cpu"] if "Comparison" in mode_str else ([cur_gpu] if "GPU Only" in mode_str else ["cpu"])

        if scope_str.startswith("Quick Production"):
            regex_filter, tests_per_dev = r".*hdr_.*nrm\..*", 6
        elif scope_str.startswith("Quick CleanAux"):
            regex_filter, tests_per_dev = r"RT\.hdr_alb_nrm\..*", 3
        elif scope_str.startswith("Ultra-Fast"):
            regex_filter, tests_per_dev = r"RT\.hdr_alb_nrm\.1920x1080", 1
        else:
            regex_filter, tests_per_dev = None, 15

        threading.Thread(target=self._benchmark_worker, args=(bench_exe, devices_to_run, regex_filter, tests_per_dev), daemon=True).start()

    def _benchmark_worker(self, bench_exe: Path, devices: list, regex_filter, tests_per_dev: int):
        results_by_dev = {}
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        for dev in devices:
            if self.bench_cancel_flag:
                break
            cmd = [str(bench_exe), "-d", dev, "-v", "1"] + (["-r", regex_filter] if regex_filter else [])
            dev_test_idx, dev_upper = 0, dev.upper()
            header_msg = f"=== Running OIDN Benchmark on [{dev_upper}] ==="
            self.after(0, lambda m=header_msg: self._bench_log(m))
            self.after(0, lambda du=dev_upper, t=tests_per_dev: self._update_bench_progress(0, t, f"{du} Test 1/{t}"))
            self.log(header_msg)

            results_by_dev[dev] = {}
            try:
                self.bench_process = subprocess.Popen(cmd, cwd=str(bench_exe.parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=creation_flags)
                fd = self.bench_process.stdout.fileno()
                buf, pending_test_name = "", None

                while not self.bench_cancel_flag:
                    try:
                        chunk = os.read(fd, 512)
                    except Exception:
                        break
                    if not chunk:
                        break
                    buf += chunk.decode("utf-8", errors="replace")

                    while "\n" in buf:
                        raw_line, buf = buf.split("\n", 1)
                        line = raw_line.strip()
                        if not line:
                            continue

                        if pending_test_name is not None and "msec/image" in line:
                            m_ms = re.search(r"([\d\.]+)\s+msec/image", line)
                            if m_ms:
                                msec_val = float(m_ms.group(1))
                                ms_int, fps_val = int(round(msec_val)), 1000.0 / max(msec_val, 1e-4)
                                results_by_dev[dev][pending_test_name] = msec_val
                                fmt_line = f"  [{dev_test_idx}/{tests_per_dev}] {pending_test_name:<30} : {ms_int:>6d} ms/img  ({fps_val:>5.1f} img/s)"
                                self.after(0, lambda fl=fmt_line: self._bench_replace_running_line(fl))
                                self.after(0, lambda c=dev_test_idx, t=tests_per_dev: self._update_bench_progress(c, t, None))
                                self.log(fmt_line)
                            pending_test_name = None
                            continue

                        m_full = re.search(r"^([\w\.]+)\s+\.\.\.\s+([\d\.]+)\s+msec/image", line)
                        if m_full:
                            dev_test_idx += 1
                            test_name, msec_val = m_full.group(1), float(m_full.group(2))
                            ms_int, fps_val = int(round(msec_val)), 1000.0 / max(msec_val, 1e-4)
                            results_by_dev[dev][test_name] = msec_val
                            fmt_line = f"  [{dev_test_idx}/{tests_per_dev}] {test_name:<30} : {ms_int:>6d} ms/img  ({fps_val:>5.1f} img/s)"
                            self.after(0, lambda fl=fmt_line: self._bench_log(fl))
                            self.after(0, lambda du=dev_upper, c=dev_test_idx, t=tests_per_dev: self._update_bench_progress(c, t, f"{du} Test {c}/{t}"))
                            self.log(fmt_line)
                        else:
                            self.after(0, lambda l=line: self._bench_log(f"  {l}"))

                    m_part = re.search(r"^\s*([\w\.]+)\s+\.\.\.\s*$", buf)
                    if m_part and pending_test_name is None:
                        pending_test_name, buf = m_part.group(1), ""
                        dev_test_idx += 1
                        run_line = f"  [{dev_test_idx}/{tests_per_dev}] {pending_test_name:<30} : running..."
                        self.after(0, lambda rl=run_line: self._bench_log(rl))
                        self.after(0, lambda du=dev_upper, c=dev_test_idx, t=tests_per_dev: self._update_bench_progress(max(0, c - 1), t, f"{du} Test {c}/{t}"))

                self.bench_process.wait()
            except Exception as ex:
                self.after(0, lambda em=f"Benchmark error on {dev}: {ex}": self._bench_log(em))
            finally:
                self.bench_process = None
            self.after(0, lambda: self._bench_log(""))

        if not self.bench_cancel_flag and len(devices) == 2:
            gpu_dev, cpu_dev = devices[0], devices[1]
            gpu_res, cpu_res = results_by_dev.get(gpu_dev, {}), results_by_dev.get(cpu_dev, {})
            common = [k for k in gpu_res if k in cpu_res]
            if common:
                s_lines = ["==============================================================", f"⚡ SPEEDUP SUMMARY: {gpu_dev.upper()} vs. {cpu_dev.upper()}"]
                for tn in common:
                    g_ms, c_ms = gpu_res[tn], cpu_res[tn]
                    s_lines.append(f"  • {tn:<28}: {gpu_dev.upper()} is {c_ms / max(g_ms, 1e-4):.1f}× faster ({int(round(g_ms))} ms vs {int(round(c_ms))} ms)")
                s_lines.append("==============================================================")
                sb = "\n".join(s_lines)
                self.after(0, lambda s=sb: self._bench_log(s))
                self.log(sb)

        def _finish_ui():
            self.bench_running = False
            if self.bench_window is not None and self.bench_window.winfo_exists():
                self.bench_run_btn.configure(text="▶ Run", fg_color="#2b7a2b", hover_color="#1f5c1f")
                if self.bench_cancel_flag:
                    self.bench_prog_lbl.configure(text="Cancelled", text_color="#e57373")
                    self._bench_log("--- Benchmark Cancelled ---")
                else:
                    self.bench_pbar.set(1.0)
                    self.bench_prog_lbl.configure(text="Benchmark Complete", text_color="#4caf50")

        self.after(0, _finish_ui)

    def _open_local_license_file(self):
        for cand in [BASE_DIR / "oidn" / "LICENSE.txt", BASE_DIR / "LICENSE", BASE_DIR / "LICENSE.txt"]:
            if cand.exists():
                try:
                    os.startfile(str(cand))
                    return
                except Exception:
                    pass
        webbrowser.open("https://www.apache.org/licenses/LICENSE-2.0")

    def _upgrade_openexr_and_restart(self, target_ver: str):
        """Writes a clean temporary .bat script (avoiding subprocess quote-escaping bugs) and runs it via os.startfile."""
        if getattr(sys, "frozen", False):
            webbrowser.open("https://pypi.org/project/OpenEXR/")
            return
        script_path = Path(__file__).resolve()
        py_exe = Path(sys.executable).resolve()
        bat_path = Path(tempfile.gettempdir()) / "polydenoise_update_openexr.bat"
        bat_content = (
            "@echo off\r\n"
            f"title Updating OpenEXR to v{target_ver}\r\n"
            f"echo Closing {APP_NAME} to release Windows file lock on OpenEXR.pyd...\r\n"
            "timeout /t 2 /nobreak >NUL\r\n"
            f'"{py_exe}" -m pip install --upgrade --only-binary=:all: OpenEXR\r\n'
            "if errorlevel 1 (\r\n"
            "    echo.\r\n"
            "    echo Update failed! Press any key to close...\r\n"
            "    pause >NUL\r\n"
            "    exit /b 1\r\n"
            ")\r\n"
            "echo.\r\n"
            f"echo OpenEXR updated successfully! Restarting {APP_NAME}...\r\n"
            f'start "" "{py_exe}" "{script_path}"\r\n'
            "exit /b 0\r\n"
        )
        bat_path.write_text(bat_content, encoding="utf-8")
        os.startfile(str(bat_path))
        self._on_closing()

    # ================= CLEAN CONFIG, VERSIONS & CREDITS POPUP =================
    def open_config_popup(self):
        if self.config_window is not None and self.config_window.winfo_exists():
            self.config_window.focus_force()
            return

        self.config_window = win = ctk.CTkToplevel(self)
        win.title(f"{APP_NAME} — Configuration & About")
        win.geometry("480x525")
        win.minsize(480, 525)
        win.resizable(False, True)
        win.attributes("-topmost", True)
        self._apply_window_icon(win)

        bold_f, mono_f, small_f = ctk.CTkFont(size=14, weight="bold"), ctk.CTkFont(family="Consolas", size=12), ctk.CTkFont(size=11)

        p_frame = ctk.CTkFrame(win, corner_radius=8)
        p_frame.pack(fill="x", padx=14, pady=(12, 6))
        ctk.CTkLabel(p_frame, text="Intel OIDN Binary / DLL Path", font=bold_f).pack(anchor="w", padx=12, pady=(8, 2))

        active_p = self.native_oidn.dll_path if self.native_oidn else self.oidn_bin
        self.cfg_path_lbl = ctk.CTkLabel(p_frame, text=str(active_p), font=small_f, text_color="#a0a0a0", wraplength=430, justify="left")
        self.cfg_path_lbl.pack(anchor="w", padx=12, pady=(0, 6))

        btn_row = ctk.CTkFrame(p_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkButton(btn_row, text="Select OIDN Binary/DLL", height=28, command=self.select_binary).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(btn_row, text="Benchmark", image=self.icon_stopwatch, compound="left", height=28, width=130, fg_color="#333333", hover_color="#4a4a4a", command=self.open_benchmark_popup).pack(side="right")

        v_frame = ctk.CTkFrame(win, corner_radius=8)
        v_frame.pack(fill="x", padx=14, pady=6)
        ctk.CTkLabel(v_frame, text="Installed Versions", font=bold_f).pack(anchor="w", padx=12, pady=(8, 4))

        ver_grid = ctk.CTkFrame(v_frame, fg_color="transparent")
        ver_grid.pack(fill="x", padx=12, pady=(2, 6))
        ver_grid.grid_columnconfigure(0, weight=1)

        exr_ver = getattr(OpenEXR, "__version__", "Unknown")
        oidn_ver = self.get_oidn_version_str()
        self.ver_status_slots, self.ver_installed_labels = {}, {}

        for r_idx, (key, label_txt, ver_txt) in enumerate([("tool", APP_NAME, f"v{APP_VERSION}"), ("oidn", "Intel OIDN Engine", f"v{oidn_ver}"), ("exr", "OpenEXR Library", f"v{exr_ver}")]):
            ctk.CTkLabel(ver_grid, text=label_txt, font=mono_f, anchor="w").grid(row=r_idx, column=0, sticky="w", pady=2)
            v_lbl = ctk.CTkLabel(ver_grid, text=ver_txt, font=mono_f, text_color="#cccccc", width=75, anchor="w")
            v_lbl.grid(row=r_idx, column=1, sticky="w", padx=(8, 8), pady=2)
            self.ver_installed_labels[key] = v_lbl
            slot_btn = ctk.CTkButton(ver_grid, text="—", width=135, height=24, font=small_f, fg_color="transparent", text_color="#777777", hover=False, state="disabled")
            slot_btn.grid(row=r_idx, column=2, sticky="e", pady=2)
            self.ver_status_slots[key] = slot_btn

        self.chk_upd_btn = ctk.CTkButton(v_frame, text="Check for Updates", height=28, fg_color="#2b7a2b", hover_color="#1f5c1f", command=self._start_github_update_check)
        self.chk_upd_btn.pack(fill="x", padx=12, pady=(2, 10))

        c_frame = ctk.CTkFrame(win, corner_radius=8, fg_color="#1b1b1b")
        c_frame.pack(fill="x", padx=14, pady=(6, 12))
        ctk.CTkLabel(c_frame, text="build by", font=ctk.CTkFont(size=11), text_color="#888888").pack(pady=(8, 0))
        ctk.CTkLabel(c_frame, text="Arthur Liebnau", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(0, 4))

        poly_btn = ctk.CTkButton(c_frame, text="🌐 Polygon Manufaktur ↗", font=ctk.CTkFont(size=13, weight="bold"), fg_color="#1f538d", hover_color="#14375e", text_color="#ffffff", height=30, width=220, command=lambda: webbrowser.open(POLYGON_WEB_URL))
        poly_btn.pack(pady=(0, 6))
        CTkToolTip(poly_btn, f"Visit {POLYGON_WEB_URL}")

        lic_lbl = ctk.CTkLabel(c_frame, text="MIT License  •  Intel® OIDN © Intel Corp. (Apache-2.0)  •  OpenEXR (BSD-3)", font=ctk.CTkFont(size=10, underline=True), text_color="#777777", cursor="hand2")
        lic_lbl.pack(pady=(0, 8))
        lic_lbl.bind("<Button-1>", lambda _: self._open_local_license_file())
        CTkToolTip(lic_lbl, "Click to view full license text (./oidn/LICENSE.txt / LICENSE).")

    @staticmethod
    def _parse_ver_tuple(v_str: str):
        m = re.search(r"(\d+(?:\.\d+)+)", str(v_str))
        return tuple(int(x) for x in m.group(1).split(".")) if m else ()

    def _start_github_update_check(self):
        if hasattr(self, "chk_upd_btn") and self.chk_upd_btn.winfo_exists():
            self.chk_upd_btn.configure(state="disabled", text="Checking GitHub & PyPI...")
        for slot in self.ver_status_slots.values():
            if slot.winfo_exists():
                slot.configure(text="Checking...", fg_color="transparent", text_color="#8ab4f8", hover=False, state="disabled")
        threading.Thread(target=self._run_github_update_check, daemon=True).start()

    def _run_github_update_check(self):
        def check_repo_redirect(repo_slug: str):
            web_url = f"https://github.com/{repo_slug}/releases/latest"
            req = urllib.request.Request(web_url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=6) as resp:
                    final_url = resp.geturl()
                    if "/tag/" in final_url:
                        raw_tag = final_url.split("/tag/")[-1].split("?")[0].strip("/")
                        m = re.search(r"(\d+\.\d+(?:\.\d+)?)", raw_tag)
                        return "ok", (m.group(1) if m else raw_tag.lstrip("vV")), final_url
                    return "unpublished", "", web_url
            except urllib.error.HTTPError as he:
                return ("unpublished", "", web_url) if he.code == 404 else ("error", f"HTTP {he.code}", web_url)
            except Exception:
                return "error", "Offline", web_url

        def check_openexr_pypi():
            pypi_url = "https://pypi.org/pypi/OpenEXR/json"
            req = urllib.request.Request(pypi_url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=6) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    ver = data.get("info", {}).get("version", "")
                    if ver:
                        return "ok", ver, "https://pypi.org/project/OpenEXR/"
            except Exception:
                pass
            return check_repo_redirect("AcademySoftwareFoundation/openexr")

        cur_oidn = self.get_oidn_version_str()
        cur_exr = getattr(OpenEXR, "__version__", "Unknown")

        checks = {
            "tool": (APP_VERSION, check_repo_redirect(TOOL_GITHUB_REPO)),
            "oidn": (cur_oidn, check_repo_redirect("RenderKit/oidn")),
            "exr": (cur_exr, check_openexr_pypi())
        }

        def _apply_slots():
            if self.config_window is None or not self.config_window.winfo_exists():
                return
            for key, (installed_v, (status, latest_v, rel_url)) in checks.items():
                slot = self.ver_status_slots.get(key)
                if not slot or not slot.winfo_exists():
                    continue
                if status == "unpublished":
                    slot.configure(text="✓ Up to date", fg_color="transparent", text_color="#4caf50", hover=False, state="disabled")
                elif status == "ok":
                    t_inst, t_lat = self._parse_ver_tuple(installed_v), self._parse_ver_tuple(latest_v)
                    if t_inst and t_lat and t_lat > t_inst:
                        if key == "exr" and not getattr(sys, "frozen", False):
                            slot.configure(
                                text=f"Update: v{latest_v} ⚡",
                                fg_color="#1f6aa5", hover_color="#144870", text_color="#ffffff",
                                hover=True, state="normal",
                                command=lambda v=latest_v: self._upgrade_openexr_and_restart(v)
                            )
                            CTkToolTip(slot, f"Click to auto-update OpenEXR to v{latest_v} via pip and restart {APP_NAME}.")
                        else:
                            slot.configure(
                                text=f"Update: v{latest_v} ↗",
                                fg_color="#1f6aa5", hover_color="#144870", text_color="#ffffff",
                                hover=True, state="normal",
                                command=lambda u=rel_url: webbrowser.open(u)
                            )
                    else:
                        slot.configure(text="✓ Up to date", fg_color="transparent", text_color="#4caf50", hover=False, state="disabled")
                else:
                    slot.configure(text=latest_v, fg_color="transparent", text_color="#e57373", hover=False, state="disabled")

            self.chk_upd_btn.configure(state="normal", text="Check for Updates")

        self.after(0, _apply_slots)

    def _on_toggle_aux_codec(self):
        self.aux_comp_menu.configure(state="normal" if self.sep_aux_comp_var.get() else "disabled")

    def clear_log(self):
        self.log_textbox.delete("1.0", "end")

    def log(self, message):
        self.after(0, lambda: self._log_sync(message))

    def _log_sync(self, message):
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")

    def update_progress(self, current, total, text_prefix=""):
        ratio = current / total if total > 0 else 0.0
        self.after(0, lambda: self.progress_bar.set(ratio))
        self.after(0, lambda: self.progress_label.configure(text=f"{text_prefix} ({current} / {total})"))

    def _update_path_label(self):
        f_name = self.input_folder.name if self.input_folder else "None"
        self.folder_label.configure(text=f"OIDN v{self.get_oidn_version_str()} | Folder: {f_name}")
        if self.config_window is not None and self.config_window.winfo_exists():
            active_p = self.native_oidn.dll_path if self.native_oidn else self.oidn_bin
            self.cfg_path_lbl.configure(text=str(active_p))
            if "oidn" in self.ver_installed_labels:
                self.ver_installed_labels["oidn"].configure(text=f"v{self.get_oidn_version_str()}")

    def _update_scale_info_label(self):
        ow, oh = self.last_known_orig_dims
        tw, th = DownscaleEngine.compute_target_size(ow, oh, self.scale_mode_var.get(), self.scale_val_entry.get())
        self.scale_out_label.configure(text=f"{tw}×{th} ({(tw / float(max(ow, 1))) * 100.0:.0f}%)")

    def on_scale_mode_select(self, mode: str):
        ow, oh = self.last_known_orig_dims
        if mode in ("Scale %", "Target Width (px)", "Target Height (px)"):
            self.scale_val_entry.configure(state="normal")
            self.scale_val_entry.delete(0, "end")
            default_v = "50" if mode == "Scale %" else str(max(2, (ow if "Width" in mode else oh) // 2))
            self.scale_val_entry.insert(0, default_v)
        else:
            self.scale_val_entry.configure(state="disabled")
        self.on_downscale_mode_change()

    def on_comp_change(self, selected_label):
        _, default_level = self.compression_map.get(selected_label, ("ZIP_COMPRESSION", ""))
        self.comp_level_entry.delete(0, "end")
        if default_level:
            self.comp_level_entry.insert(0, default_level)

    def on_downscale_mode_change(self, _=None):
        self._update_scale_info_label()
        with self.cache_lock:
            self.linear_lru_cache.clear()
        self.reload_current_preview()

    def toggle_default_view_mode(self):
        self.show_denoised_default = not self.show_denoised_default
        self.swap_default_btn.configure(text="Hold: Raw" if self.show_denoised_default else "Hold: Denoised", fg_color="#2b7a2b" if self.show_denoised_default else "#3b5998")
        self.reload_current_preview()

    def on_ev_change(self, val):
        ev = round(float(val), 2)
        self.ev_reset_btn.configure(text=f"{ev:+.1f}")
        if self.primary_linear is not None:
            self.primary_rgb = self._tonemap_and_dither_fast(self.primary_linear, ev)
        if self.alternate_linear is not None:
            self.alternate_rgb = self._tonemap_and_dither_fast(self.alternate_linear, ev)
        self.render_preview_to_canvas()

    def reset_ev(self):
        if abs(self.ev_var.get()) > 1e-4:
            self.ev_var.set(0.0)
            self.on_ev_change(0.0)

    def reload_current_preview(self):
        frame, ch = self.frame_var.get(), self.channel_var.get()
        if frame in self.frames_data:
            self.load_preview_pair(frame, ch)

    # ================= DIRECT MOUSEWHEEL ZOOM TO CURSOR POSITION =================
    def on_mousewheel(self, event):
        if self.primary_rgb is None:
            return
        orig_h, orig_w, _ = self.primary_rgb.shape
        old_w, old_h = self._last_canvas_dims
        if old_w < 1 or old_h < 1:
            return

        # 1. Capture exact image coordinate under mouse pointer before zooming
        mx, my = event.x, event.y
        cx = self.preview_canvas.canvasx(mx)
        cy = self.preview_canvas.canvasy(my)
        rel_x = np.clip((cx - self._img_offset_x) / float(old_w), 0.0, 1.0)
        rel_y = np.clip((cy - self._img_offset_y) / float(old_h), 0.0, 1.0)

        # 2. Determine current percentage and step up/down
        cur_str = self.zoom_var.get()
        if cur_str == "Fit":
            cur_pct = (old_w / float(max(orig_w, 1))) * 100.0
        else:
            cur_pct = float(cur_str.replace("%", ""))

        if event.delta > 0:
            higher = [z for z in self.ZOOM_STEPS if z > cur_pct + 0.5]
            new_pct = higher[0] if higher else self.ZOOM_STEPS[-1]
        else:
            lower = [z for z in self.ZOOM_STEPS if z < cur_pct - 0.5]
            new_pct = lower[-1] if lower else self.ZOOM_STEPS[0]

        new_zoom_str = f"{new_pct}%"
        if new_zoom_str != cur_str:
            self.zoom_var.set(new_zoom_str)
            self.render_preview_to_canvas(anchor=(mx, my, rel_x, rel_y))

    def _on_canvas_configure(self, _=None):
        if self.primary_rgb is not None:
            if self.zoom_var.get() == "Fit":
                self.render_preview_to_canvas()
            else:
                h, w, _ = self.primary_rgb.shape
                scale = float(self.zoom_var.get().replace("%", "")) / 100.0
                self._update_scrollbars_and_position(max(1, int(w * scale)), max(1, int(h * scale)))

    def _update_scrollbars_and_position(self, new_w: int, new_h: int, anchor=None):
        canv_w, canv_h = max(self.preview_canvas.winfo_width(), 100), max(self.preview_canvas.winfo_height(), 100)
        is_fit = self.zoom_var.get() == "Fit"
        need_h, need_v = (not is_fit) and (new_w > canv_w), (not is_fit) and (new_h > canv_h)

        if need_v:
            self.v_scroll.grid(row=0, column=1, sticky="ns")
        else:
            self.v_scroll.grid_remove()
        if need_h:
            self.h_scroll.grid(row=1, column=0, sticky="ew")
        else:
            self.h_scroll.grid_remove()

        self._img_offset_x = 0 if need_h else max(0, (canv_w - new_w) // 2)
        self._img_offset_y = 0 if need_v else max(0, (canv_h - new_h) // 2)
        sr_w, sr_h = max(new_w, canv_w), max(new_h, canv_h)

        if self.canvas_img_id is not None:
            self.preview_canvas.coords(self.canvas_img_id, self._img_offset_x, self._img_offset_y)
        self.preview_canvas.configure(scrollregion=(0, 0, sr_w, sr_h))

        # Anchor zoom to mouse cursor position
        if anchor is not None:
            mx, my, rel_x, rel_y = anchor
            if need_h and sr_w > 0:
                target_left = (self._img_offset_x + rel_x * new_w) - mx
                self.preview_canvas.xview_moveto(float(np.clip(target_left / float(sr_w), 0.0, 1.0)))
            if need_v and sr_h > 0:
                target_top = (self._img_offset_y + rel_y * new_h) - my
                self.preview_canvas.yview_moveto(float(np.clip(target_top / float(sr_h), 0.0, 1.0)))

        if self._spinner_active:
            self._draw_spinner_overlay()

    def _on_space_press(self, event):
        if not isinstance(event.widget, (tk.Entry, ctk.CTkEntry)):
            self.on_canvas_press_compare()

    def _on_space_release(self, _=None):
        self.on_canvas_release_compare()

    def on_canvas_press_compare(self, _=None):
        if self.tk_img_alternate is not None and self.canvas_img_id is not None:
            self.is_holding_compare = True
            self.preview_canvas.itemconfig(self.canvas_img_id, image=self.tk_img_alternate)
            h, w, _ = self.alternate_rgb.shape
            self.res_label.configure(text=f"COMPARE: {self.alternate_tag} ({w}×{h} px)")
        elif self.primary_rgb is not None:
            self.res_label.configure(text=f"No _denoised file yet for '{self.channel_var.get()}'!")

    def on_canvas_release_compare(self, _=None):
        if self.is_holding_compare and self.tk_img_primary is not None:
            self.is_holding_compare = False
            if self.canvas_img_id is not None:
                self.preview_canvas.itemconfig(self.canvas_img_id, image=self.tk_img_primary)
            h, w, _ = self.primary_rgb.shape
            self.res_label.configure(text=f"{self.primary_tag} ({w}×{h} px)")

    def select_binary(self):
        file_path = filedialog.askopenfilename(title="Select oidnDenoise.exe or OpenImageDenoise.dll", filetypes=[("OIDN Binary / DLL", "*.exe *.dll"), ("All Files", "*.*")])
        if file_path:
            self.oidn_bin = Path(file_path)
            register_dll_dir(self.oidn_bin.parent)
            self.save_config()
            self._try_init_native_dll()
            self._update_path_label()
            if self.native_oidn is not None:
                self.log(f"Direct RAM/VRAM DLL Engine v{self.native_oidn.version_str} loaded: {self.native_oidn.dll_path}")
            else:
                self.log(f"OIDN binary saved: {self.oidn_bin}")

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder = Path(folder)
            self._update_path_label()
            with self.cache_lock:
                self.linear_lru_cache.clear()
            self.log(f"\n--- Scanned Folder: {self.input_folder.name} ---")
            self.parse_indigo_passes()

    @staticmethod
    def read_exr_native(exr_path: Path, preserve_uint32: bool = False):
        with OpenEXR.File(str(exr_path)) as f:
            ch_dict = f.channels()
            alpha = None
            keys = list(ch_dict.keys())
            first_pix = ch_dict[keys[0]].pixels
            target_dt = np.uint32 if (preserve_uint32 and first_pix.dtype == np.uint32) else np.float32

            if "RGBA" in ch_dict:
                rgba = np.asarray(ch_dict["RGBA"].pixels, dtype=target_dt)
                return np.ascontiguousarray(rgba[:, :, :3], dtype=target_dt), np.ascontiguousarray(rgba[:, :, 3], dtype=np.float32), keys
            if "RGB" in ch_dict:
                rgb = np.ascontiguousarray(np.asarray(ch_dict["RGB"].pixels, dtype=target_dt)[:, :, :3], dtype=target_dt)
            elif all(k in ch_dict for k in ("R", "G", "B")):
                rgb = np.ascontiguousarray(np.dstack((ch_dict["R"].pixels, ch_dict["G"].pixels, ch_dict["B"].pixels)), dtype=target_dt)
            elif all(k in ch_dict for k in ("X", "Y", "Z")):
                rgb = np.ascontiguousarray(np.dstack((ch_dict["X"].pixels, ch_dict["Y"].pixels, ch_dict["Z"].pixels)), dtype=target_dt)
            else:
                gray = np.asarray(first_pix, dtype=target_dt)
                rgb = np.ascontiguousarray(np.dstack((gray, gray, gray)) if gray.ndim == 2 else gray[:, :, :3], dtype=target_dt)
            if "A" in ch_dict:
                alpha = np.ascontiguousarray(ch_dict["A"].pixels, dtype=np.float32)
            return rgb, alpha, keys

    @staticmethod
    def is_exr_black_fast(exr_path: Path) -> bool:
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

    @staticmethod
    def write_exr_static(
        out_exr_path: Path, rgb: np.ndarray, alpha, comp_attr_name: str,
        level_str: str, use_half: bool, worker_threads: int = CPU_THREADS,
        is_data_pass: bool = False
    ):
        header = {"compression": getattr(OpenEXR, comp_attr_name), "type": OpenEXR.scanlineimage}
        if is_data_pass:
            header["colorInteropID"] = "data"
        if level_str:
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
            "B": np.ascontiguousarray(rgb_safe[:, :, 2], dtype=target_dtype)
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

    # ================= UNIVERSAL EXPORTER PASS PARSER & INSTANT FIRST-PASS LOADER =================
    def _extract_frame_and_channel(self, stem: str, discovered_prefixes: list):
        if "_channels_" in stem:
            frame_id, raw_pass = stem.split("_channels_", 1)
            return frame_id, self.AUX_ALIASES.get(raw_pass.lower(), raw_pass)

        all_known = sorted(list(self.AUX_ALIASES.keys()) + self.KNOWN_STD_LIGHT_PASSES, key=len, reverse=True)
        stem_low = stem.lower()
        for kw in all_known:
            for sep in ("_", ".", "-", " "):
                token = f"{sep}{kw}"
                if stem_low.endswith(token):
                    prefix = stem[: -len(token)].rstrip("_.- ") or "Scene"
                    return prefix, self.AUX_ALIASES.get(kw, kw)

        for pref in discovered_prefixes:
            if stem.startswith(pref) and len(stem) > len(pref):
                rem = stem[len(pref) :]
                if rem[0] in ("_", ".", "-", " "):
                    raw_pass = rem.lstrip("_.- ")
                    if raw_pass:
                        return pref, self.AUX_ALIASES.get(raw_pass.lower(), raw_pass)

        m = re.match(r"^(.*?)[_.\-]([^_.\-]+)$", stem)
        if m and m.group(1):
            return m.group(1), self.AUX_ALIASES.get(m.group(2).lower(), m.group(2))
        return "Scene", stem

    def parse_indigo_passes(self, select_channel=None):
        self._scan_generation += 1
        current_gen = self._scan_generation

        self.render_queue.clear()
        self.aux_mask_files.clear()
        self.frames_data.clear()
        self.denoised_data.clear()
        self.empty_passes.clear()
        skipped_denoised, skipped_resumed = 0, 0
        skip_done = self.skip_existing_var.get()

        all_exrs = sorted(self.input_folder.glob("*.exr"))
        anchor_prefixes = set()
        all_known = sorted(list(self.AUX_ALIASES.keys()) + self.KNOWN_STD_LIGHT_PASSES, key=len, reverse=True)

        for fp in all_exrs:
            st, st_low = fp.stem, fp.stem.lower()
            if st_low.endswith(("_denoised", "_50pct", "_scaled")) or "material_mask" in st_low or "object_mask" in st_low:
                continue
            if "_channels_" in st:
                anchor_prefixes.add(st.split("_channels_", 1)[0])
                continue
            for kw in all_known:
                for sep in ("_", ".", "-", " "):
                    token = f"{sep}{kw}"
                    if st_low.endswith(token):
                        pref = st[: -len(token)].rstrip("_.- ")
                        if pref:
                            anchor_prefixes.add(pref)
                        break

        sorted_prefixes = sorted(list(anchor_prefixes), key=len, reverse=True)
        frame_aux_lookup, raw_light_candidates = {}, []

        for file_path in all_exrs:
            name, stem = file_path.name, file_path.stem
            low_name, low_stem = name.lower(), stem.lower()
            if file_path.stat().st_size < 2048 and "_denoise" in low_name:
                continue
            if low_name.endswith(("_50pct.exr", "_scaled.exr")):
                continue
            if "material_mask" in low_stem or "object_mask" in low_stem:
                if not low_stem.endswith("_denoised"):
                    self.aux_mask_files.append({"path": file_path, "type": "mask"})
                continue

            is_den_file = low_stem.endswith("_denoised")
            work_stem = stem[:-9] if is_den_file else stem
            frame_id, pass_name = self._extract_frame_and_channel(work_stem, sorted_prefixes)

            if is_den_file:
                self.denoised_data.setdefault(frame_id, {})[pass_name] = file_path
                skipped_denoised += 1
                continue
            if "_denoise" in pass_name.lower():
                skipped_denoised += 1
                continue

            self.frames_data.setdefault(frame_id, {})[pass_name] = file_path
            if pass_name in self.blacklist:
                self.aux_mask_files.append({"path": file_path, "type": pass_name})
                frame_aux_lookup.setdefault(frame_id, {})[pass_name] = file_path
            else:
                raw_light_candidates.append((frame_id, pass_name, file_path))

        for frame_id, pass_name, file_path in raw_light_candidates:
            out_candidate = file_path.parent / f"{file_path.stem}_denoised.exr"
            if skip_done and out_candidate.exists() and out_candidate.stat().st_size > 2048:
                skipped_resumed += 1
                continue
            aux_dict = frame_aux_lookup.get(frame_id, {})
            self.render_queue.append({
                "target": file_path, "albedo": aux_dict.get("albedo"),
                "normal_bump": aux_dict.get("normals"), "normal_pre": aux_dict.get("normals_pre_bump"),
                "frame_id": frame_id, "pass_name": pass_name
            })

        if self.frames_data:
            frame_list = sorted(list(self.frames_data.keys()))
            self.frame_selector.configure_values(frame_list)
            if self.frame_var.get() not in frame_list:
                self.frame_var.set(frame_list[0])
            cur_f = self.frame_var.get()
            if select_channel and (select_channel == self.BEAUTY_SUM_KEY or select_channel in self.frames_data[cur_f]):
                self.channel_var.set(select_channel)
            self.on_frame_change(cur_f)

        btn_state = "normal" if self.render_queue else "disabled"
        self.start_btn.configure(state=btn_state)
        self.test_single_btn.configure(state=btn_state)

        if raw_light_candidates:
            self.start_viewport_spinner(f"Scanning {len(raw_light_candidates)} passes in background...")
            threading.Thread(target=self._background_folder_scan_worker, args=(current_gen, list(raw_light_candidates), skipped_denoised), daemon=True).start()

    def _background_folder_scan_worker(self, gen_id: int, candidates: list, paired_den_count: int):
        skip_empty = self.skip_empty_var.get()
        first_frame = sorted(list(self.frames_data.keys()))[0] if self.frames_data else None
        scan_all = len(candidates) <= 60
        discovered_empty = []

        cur_f, cur_ch = self.frame_var.get(), self.channel_var.get()
        if cur_f in self.denoised_data and cur_ch in self.denoised_data[cur_f]:
            den_p = self.denoised_data[cur_f][cur_ch]
            if den_p.exists():
                try:
                    self._get_scaled_linear_rgb(den_p, is_denoised=True, ch_label=cur_ch)
                except Exception:
                    pass

        total_c = len(candidates)
        for idx, (frame_id, pass_name, file_path) in enumerate(candidates, start=1):
            if gen_id != self._scan_generation or self.is_running:
                return
            if scan_all or frame_id == first_frame:
                if idx % 3 == 0:
                    self.after(0, lambda i=idx, t=total_c: self.start_viewport_spinner(f"Scanning passes ({i}/{t})..."))
                if self.is_exr_black_fast(file_path):
                    discovered_empty.append((frame_id, pass_name))

        def _finish_bg_scan():
            if gen_id != self._scan_generation:
                return
            for f_id, p_name in discovered_empty:
                self.empty_passes.setdefault(f_id, set()).add(p_name)
            if skip_empty and discovered_empty:
                empty_lookup = set(discovered_empty)
                self.render_queue = [item for item in self.render_queue if (item["frame_id"], item["pass_name"]) not in empty_lookup]
            self.channel_selector.refresh_label()
            f_now, ch_now = self.frame_var.get(), self.channel_var.get()
            if self.alternate_linear is None and f_now in self.denoised_data and ch_now in self.denoised_data[f_now]:
                self.load_preview_pair(f_now, ch_now)

            self.stop_viewport_spinner()
            btn_s = "normal" if self.render_queue else "disabled"
            if not self.is_running:
                self.start_btn.configure(state=btn_s)
                self.test_single_btn.configure(state=btn_s)
            self.log(f"Background scan complete: {len(self.render_queue)} active light passes | Flagged {len(discovered_empty)} empty (black) | {len(self.aux_mask_files)} Aux/Mask/ID | Paired {paired_den_count} _denoised.")

        self.after(0, _finish_bg_scan)

    def on_frame_change(self, selected_frame):
        if selected_frame not in self.frames_data:
            return
        base_channels = sorted(list(self.frames_data[selected_frame].keys()))
        light_channels = [c for c in base_channels if c not in self.blacklist]
        channels = ([self.BEAUTY_SUM_KEY] + base_channels) if len(light_channels) > 1 else base_channels
        self.channel_selector.configure_values(channels)

        current_ch = self.channel_var.get()
        if current_ch in channels:
            self.channel_selector.refresh_label()
            self.load_preview_pair(selected_frame, current_ch)
        elif channels:
            non_empty = [c for c in light_channels if c not in self.empty_passes.get(selected_frame, set())]
            default_ch = non_empty[0] if non_empty else (light_channels[0] if light_channels else channels[0])
            self.channel_var.set(default_ch)
            self.load_preview_pair(selected_frame, default_ch)

    def on_channel_change(self, selected_channel):
        frame = self.frame_var.get()
        if frame in self.frames_data:
            self.load_preview_pair(frame, selected_channel)

    def on_zoom_change(self, _=None):
        self.render_preview_to_canvas()

    def _get_ign_dither(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if key not in self._dither_cache:
            y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
            ign1 = np.mod(52.9829189 * np.mod(0.06711056 * x + 0.00583715 * y, 1.0), 1.0)
            ign2 = np.mod(52.9829189 * np.mod(0.06711056 * (x + 17.0) + 0.00583715 * (y + 31.0), 1.0), 1.0)
            self._dither_cache[key] = ((ign1 - ign2) * 0.5)[:, :, None].astype(np.float32)
        return self._dither_cache[key]

    def _tonemap_and_dither_fast(self, linear_rgb: np.ndarray, ev_val: float) -> np.ndarray:
        img_pos = linear_rgb * np.float32(2.0 ** ev_val) if abs(ev_val) > 1e-4 else linear_rgb
        img_mapped = np.clip((img_pos * (2.51 * img_pos + 0.03)) / (img_pos * (2.43 * img_pos + 0.59) + 0.14), 0.0, 1.0)
        srgb_float = _SRGB_LUT_12BIT[(img_mapped * np.float32(_LUT_STEPS - 1)).astype(np.int32)]
        h, w, _ = srgb_float.shape
        return np.clip(srgb_float + self._get_ign_dither(h, w) + 0.5, 0.0, 255.0).astype(np.uint8)

    def _get_scaled_linear_rgb(self, img_path: Path, is_denoised: bool, ch_label: str = ""):
        ds_kernel, sc_mode, sc_val = self.downscale_var.get(), self.scale_mode_var.get(), self.scale_val_entry.get().strip()
        cache_key = (str(img_path), img_path.stat().st_mtime_ns, ds_kernel, sc_mode, sc_val, is_denoised)
        with self.cache_lock:
            if cache_key in self.linear_lru_cache:
                self.linear_lru_cache.move_to_end(cache_key)
                return self.linear_lru_cache[cache_key]

        rgb_float, _, _ = self.read_exr_native(img_path)
        img_pos = np.maximum(np.nan_to_num(rgb_float, nan=0.0, posinf=65500.0, neginf=0.0), 0.0)
        oh, ow, _ = img_pos.shape
        ch_short = ch_label or img_path.stem

        if is_denoised:
            tag = f"{ch_short} [Denoised]"
        else:
            self.last_known_orig_dims = (ow, oh)
            self.after(0, self._update_scale_info_label)
            is_empty = float(np.max(img_pos)) <= 1e-6
            if is_empty and ch_short not in self.blacklist:
                self.empty_passes.setdefault(self.frame_var.get(), set()).add(ch_short)
                self.after(0, self.channel_selector.refresh_label)
            tw, th = DownscaleEngine.compute_target_size(ow, oh, sc_mode, sc_val)
            if (tw, th) != (ow, oh):
                img_pos = DownscaleEngine.downscale_aux_pass(img_pos, tw, th, ch_short, ds_kernel) if ch_short in self.blacklist else DownscaleEngine.downscale(img_pos, tw, th, ds_kernel)
                tag = f"{ch_short} [{'EMPTY — ' if is_empty else ''}Raw ➔ {tw}×{th} {ds_kernel.split('(')[0].strip()}]"
            else:
                tag = f"{ch_short} [{'EMPTY — ' if is_empty else ''}Raw 100%]"

        with self.cache_lock:
            self.linear_lru_cache[cache_key] = (img_pos, tag)
            if len(self.linear_lru_cache) > 16:
                self.linear_lru_cache.popitem(last=False)
        return img_pos, tag

    def _build_beauty_sum_pair(self, frame_id: str):
        frame_dict, den_dict = self.frames_data.get(frame_id, {}), self.denoised_data.get(frame_id, {})
        empty_set = self.empty_passes.get(frame_id, set())
        light_chs = [c for c in frame_dict.keys() if c not in self.blacklist and c not in empty_set]
        if not light_chs:
            return None, "", None, ""
        raw_sum, den_sum, any_den = None, None, False
        for ch in light_chs:
            r_lin, _ = self._get_scaled_linear_rgb(frame_dict[ch], is_denoised=False, ch_label=ch)
            raw_sum = r_lin.copy() if raw_sum is None else (raw_sum + r_lin)
            if ch in den_dict and den_dict[ch].exists():
                d_lin, _ = self._get_scaled_linear_rgb(den_dict[ch], is_denoised=True, ch_label=ch)
                if d_lin.shape == raw_sum.shape:
                    den_sum = d_lin.copy() if den_sum is None else (den_sum + d_lin)
                    any_den = True
            elif den_sum is not None and r_lin.shape == den_sum.shape:
                den_sum = den_sum + r_lin
        return raw_sum, f"Beauty Sum ({len(light_chs)} Active Lights) [Raw]", (den_sum if any_den else None), f"Beauty Sum ({len(light_chs)} Active Lights) [Denoised]"

    def load_preview_pair(self, frame_id: str, channel_name: str):
        ev = round(float(self.ev_var.get()), 2)

        if channel_name == self.BEAUTY_SUM_KEY:
            self.start_viewport_spinner("Summing light passes...")

            def _bg_sum():
                try:
                    raw_lin, raw_tag, den_lin, den_tag = self._build_beauty_sum_pair(frame_id)
                except Exception as e:
                    self.log(f"Beauty sum warning: {e}")
                    self.after(0, self.stop_viewport_spinner)
                    return
                if raw_lin is None:
                    self.after(0, self.stop_viewport_spinner)
                    return

                def _apply_sum():
                    self.stop_viewport_spinner()
                    if self.channel_var.get() != self.BEAUTY_SUM_KEY:
                        return
                    if self.show_denoised_default and den_lin is not None:
                        self.primary_linear, self.primary_tag = den_lin, den_tag
                        self.alternate_linear, self.alternate_tag = raw_lin, raw_tag
                    else:
                        self.primary_linear, self.primary_tag = raw_lin, raw_tag
                        self.alternate_linear, self.alternate_tag = den_lin, den_tag
                    self.primary_rgb = self._tonemap_and_dither_fast(self.primary_linear, ev)
                    self.alternate_rgb = self._tonemap_and_dither_fast(self.alternate_linear, ev) if self.alternate_linear is not None else None
                    self.render_preview_to_canvas()

                self.after(0, _apply_sum)

            threading.Thread(target=_bg_sum, daemon=True).start()
            return

        raw_path = self.frames_data.get(frame_id, {}).get(channel_name)
        if not raw_path:
            return
        den_path = self.denoised_data.get(frame_id, {}).get(channel_name)
        has_den = den_path is not None and den_path.exists()

        primary_is_den = self.show_denoised_default and has_den
        first_path = den_path if primary_is_den else raw_path
        second_path = raw_path if primary_is_den else (den_path if has_den else None)

        try:
            first_lin, first_tag = self._get_scaled_linear_rgb(first_path, is_denoised=primary_is_den, ch_label=channel_name)
        except Exception as e:
            self.log(f"Warning: Failed to read {first_path.name} ({e}).")
            return

        self.primary_linear, self.primary_tag = first_lin, first_tag
        self.alternate_linear, self.alternate_tag = None, ""
        self.primary_rgb = self._tonemap_and_dither_fast(self.primary_linear, ev)
        self.alternate_rgb = None
        self.render_preview_to_canvas()

        if second_path is not None:
            def _bg_counterpart():
                try:
                    sec_lin, sec_tag = self._get_scaled_linear_rgb(second_path, is_denoised=not primary_is_den, ch_label=channel_name)
                except Exception:
                    return

                def _attach_alt():
                    if self.frame_var.get() == frame_id and self.channel_var.get() == channel_name:
                        cur_ev = round(float(self.ev_var.get()), 2)
                        self.alternate_linear, self.alternate_tag = sec_lin, sec_tag
                        self.alternate_rgb = self._tonemap_and_dither_fast(sec_lin, cur_ev)
                        if self.primary_rgb is not None:
                            oh, ow, _ = self.primary_rgb.shape
                            z_str = self.zoom_var.get()
                            sc = min(max(self.preview_canvas.winfo_width(), 100) / ow, max(self.preview_canvas.winfo_height(), 100) / oh) if z_str == "Fit" else float(z_str.replace("%", "")) / 100.0
                            nw, nh = max(1, int(ow * sc)), max(1, int(oh * sc))
                            self.tk_img_alternate = self._scale_pil(self.alternate_rgb, nw, nh, sc)

                self.after(0, _attach_alt)

            threading.Thread(target=_bg_counterpart, daemon=True).start()

    @staticmethod
    def _scale_pil(rgb_arr: np.ndarray, new_w: int, new_h: int, scale: float):
        pil_img = Image.fromarray(rgb_arr)
        if scale != 1.0:
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST if scale > 1.0 else Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

    def render_preview_to_canvas(self, anchor=None):
        if self.primary_rgb is None:
            return
        orig_h, orig_w, _ = self.primary_rgb.shape
        zoom_str = self.zoom_var.get()
        if zoom_str == "Fit":
            self.update_idletasks()
            scale = min(max(self.preview_canvas.winfo_width(), 100) / orig_w, max(self.preview_canvas.winfo_height(), 100) / orig_h)
        else:
            scale = float(zoom_str.replace("%", "")) / 100.0

        new_w, new_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
        self.tk_img_primary = self._scale_pil(self.primary_rgb, new_w, new_h, scale)
        self.tk_img_alternate = self._scale_pil(self.alternate_rgb, new_w, new_h, scale) if self.alternate_rgb is not None else None

        active_img = self.tk_img_alternate if (self.is_holding_compare and self.tk_img_alternate is not None) else self.tk_img_primary
        active_tag = f"COMPARE: {self.alternate_tag}" if (self.is_holding_compare and self.tk_img_alternate is not None) else self.primary_tag
        self.res_label.configure(text=f"{active_tag} ({orig_w}×{orig_h} px)")

        if self.canvas_img_id is not None and self._last_canvas_dims == (new_w, new_h):
            self.preview_canvas.itemconfig(self.canvas_img_id, image=active_img)
        else:
            self.preview_canvas.delete("all")
            self.canvas_img_id = self.preview_canvas.create_image(0, 0, anchor="nw", image=active_img)
            self._last_canvas_dims = (new_w, new_h)
        self._update_scrollbars_and_position(new_w, new_h, anchor=anchor)

    @staticmethod
    def write_pfm(rgb: np.ndarray, pfm_path: Path):
        clean_rgb = np.nan_to_num(rgb, nan=0.0, posinf=65500.0, neginf=-65500.0)
        img_flipped = np.ascontiguousarray(clean_rgb[::-1, :, :], dtype=np.float32)
        h, w, _ = img_flipped.shape
        with open(pfm_path, "wb") as f:
            f.write(f"PF\n{w} {h}\n-1.0\n".encode("ascii"))
            f.write(img_flipped.tobytes())

    @staticmethod
    def read_pfm(pfm_path: Path) -> np.ndarray:
        with open(pfm_path, "rb") as f:
            if f.readline().decode("ascii").strip() != "PF":
                raise RuntimeError("Invalid PFM header.")
            w, h = map(int, f.readline().decode("ascii").strip().split())
            _ = float(f.readline().decode("ascii").strip())
            data = np.frombuffer(f.read(), dtype=np.float32)
        return np.ascontiguousarray(data.reshape((h, w, 3))[::-1, :, :], dtype=np.float32)

    @staticmethod
    def estimate_noise_fast_roi(orig_rgb: np.ndarray, den_rgb: np.ndarray) -> float:
        h, w, _ = den_rgb.shape
        cy, cx = h // 2, w // 2
        r = min(200, cy, cx)
        if r < 8:
            return 0.0
        d_patch = den_rgb[cy - r : cy + r, cx - r : cx + r, :]
        o_patch = cv2.resize(orig_rgb, (w, h), interpolation=cv2.INTER_AREA)[cy - r : cy + r, cx - r : cx + r, :]

        def _hf(p):
            lum = np.maximum(0.2126 * p[:, :, 0] + 0.7152 * p[:, :, 1] + 0.0722 * p[:, :, 2], 0.0)
            tm = lum / (1.0 + lum)
            return float(np.mean(np.abs(tm[:, 1:] - tm[:, :-1])) + np.mean(np.abs(tm[1:, :] - tm[:-1, :])))

        n_b, n_a = _hf(o_patch), _hf(d_patch)
        return ((n_b - n_a) / max(n_b, 1e-7)) * 100.0

    @staticmethod
    def stride_pack(img: np.ndarray, stride: int, pad: int = 16):
        h, w, _ = img.shape
        sh, sw = (h // stride) * stride, (w // stride) * stride
        cropped = img[:sh, :sw, :]
        rows = [
            np.concatenate([np.pad(cropped[dy::stride, dx::stride, :], ((pad, pad), (pad, pad), (0, 0)), mode="reflect") for dx in range(stride)], axis=1)
            for dy in range(stride)
        ]
        return np.concatenate(rows, axis=0), sh, sw

    @staticmethod
    def stride_unpack(packed: np.ndarray, orig_img: np.ndarray, stride: int, pad: int = 16):
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

    def run_oidn_cli(self, cmd_args: list):
        oidn_cwd = str(self.oidn_bin.parent) if self.oidn_bin.parent.exists() else None
        self.active_process = subprocess.Popen(cmd_args, cwd=oidn_cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout_str, stderr_str = self.active_process.communicate()
        ret_code = self.active_process.returncode
        self.active_process = None
        return ret_code, stdout_str, stderr_str

    def build_oidn_cmd(self, hdr_pfm, out_pfm, alb_pfm, nrm_pfm, use_ldr):
        cmd = [
            str(self.oidn_bin), "-d", self.device_var.get(), "-f", self.filter_var.get(),
            "-q", self.quality_var.get(), "-t", self.precision_var.get(), "-v", self.verbose_var.get(),
            "--ldr" if use_ldr else "--hdr", str(hdr_pfm), "-o", str(out_pfm)
        ]
        if alb_pfm is not None:
            cmd.extend(["--alb", str(alb_pfm)])
        if nrm_pfm is not None:
            cmd.extend(["--nrm", str(nrm_pfm)])
        if self.clean_aux_var.get():
            cmd.append("--clean_aux")
        if self.inplace_var.get():
            cmd.append("--inplace")
        if self.maxmem_entry.get().strip().isdigit():
            cmd.extend(["--maxmem", self.maxmem_entry.get().strip()])
        if self.future_args.get().strip():
            cmd.extend(self.future_args.get().strip().split())
        return cmd

    def _set_running_ui_state(self, running: bool):
        self.is_running = running
        if running:
            self.start_viewport_spinner("Denoising in VRAM...")
            self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(2, 8), pady=4)
            self.progress_label.grid(row=0, column=1, padx=(0, 8), pady=4)
            self.start_btn.configure(text="Cancel Batch (Running...)", fg_color="#8b2626", hover_color="#a83232", state="normal")
            self.test_single_btn.configure(state="disabled")
            self.folder_btn.configure(state="disabled")
        else:
            self.stop_viewport_spinner()
            self.progress_bar.grid_remove()
            self.progress_label.grid_remove()
            btn_state = "normal" if self.render_queue else "disabled"
            self.start_btn.configure(text="Start Full Batch Denoise", fg_color="#2b7a2b", hover_color="#1f5c1f", state=btn_state)
            self.test_single_btn.configure(state=btn_state)
            self.folder_btn.configure(state="normal")

    def on_start_or_cancel_click(self):
        if self.is_running:
            self.cancel_batch()
        else:
            self.start_batch_thread()

    def start_single_test_thread(self):
        if self.is_running:
            return
        if self.native_oidn is None and not self.oidn_bin.exists():
            self.log(f"ERROR: OIDN binary/DLL not found at '{self.oidn_bin}'!")
            return
        frame, ch = self.frame_var.get(), self.channel_var.get()
        selected_item = next((i for i in self.render_queue if i["frame_id"] == frame and i["pass_name"] == ch), None)

        if not selected_item and frame in self.frames_data and ch in self.frames_data[frame]:
            raw_p = self.frames_data[frame][ch]
            if raw_p.exists() and ch not in self.blacklist:
                f_dict = self.frames_data[frame]
                selected_item = {
                    "target": raw_p, "albedo": f_dict.get("albedo"), "normal_bump": f_dict.get("normals"),
                    "normal_pre": f_dict.get("normals_pre_bump"), "frame_id": frame, "pass_name": ch
                }
        if not selected_item and self.render_queue:
            selected_item = self.render_queue[0]
        if not selected_item:
            return

        self.cancel_event.clear()
        self._set_running_ui_state(True)
        threading.Thread(target=self.process_items, args=([selected_item], True), daemon=True).start()

    def start_batch_thread(self):
        if self.native_oidn is None and not self.oidn_bin.exists():
            self.log(f"ERROR: OIDN binary/DLL not found at '{self.oidn_bin}'!")
            return
        self.cancel_event.clear()
        self._set_running_ui_state(True)
        threading.Thread(target=self.process_items, args=(list(self.render_queue), False), daemon=True).start()

    def cancel_batch(self):
        self.cancel_event.set()
        self.log("-> Cancellation requested! Stopping active OIDN process...")
        if self.active_process and self.active_process.poll() is None:
            try:
                self.active_process.terminate()
            except Exception:
                pass

    def _save_pass_worker(
        self, idx: int, total: int, out_path: Path, orig_rgb: np.ndarray, denoised_rgb: np.ndarray,
        alpha_ch, ds_kernel: str, sc_mode: str, sc_val: str, comp_attr_name: str, level_str: str,
        use_half_exr: bool, t_denoise_elapsed: float, exr_worker_threads: int
    ):
        t_save_start = time.perf_counter()
        oh, ow, _ = denoised_rgb.shape
        tw, th = DownscaleEngine.compute_target_size(ow, oh, sc_mode, sc_val)
        out_alpha = alpha_ch
        if (tw, th) != (ow, oh):
            denoised_rgb = DownscaleEngine.downscale(denoised_rgb, tw, th, ds_kernel)
            if out_alpha is not None:
                out_alpha = DownscaleEngine.downscale_aux_pass(np.dstack([out_alpha] * 3), tw, th, "alpha", ds_kernel)[:, :, 0]

        self.write_exr_static(out_path, denoised_rgb, out_alpha, comp_attr_name, level_str, use_half_exr, worker_threads=exr_worker_threads, is_data_pass=False)
        t_save_elapsed = time.perf_counter() - t_save_start
        red_1px = self.estimate_noise_fast_roi(orig_rgb, denoised_rgb)
        size_kb = out_path.stat().st_size / 1024.0
        h_f, w_f, _ = denoised_rgb.shape
        self.log(f"[{idx}/{total}] Saved {w_f}×{h_f} ({size_kb:.1f} KB | Denoise: {t_denoise_elapsed:.1f}s | Save: {t_save_elapsed:.1f}s | Noise Reduction: {red_1px:+.1f}%): {out_path.name}")

    def process_items(self, items_to_process: list, is_diagnostic: bool):
        initial_channel_before_run = self.channel_var.get()

        total = len(items_to_process)
        aux_mode, hdr_mode, decorr_mode = self.aux_mode_var.get(), self.hdr_mode_var.get(), self.decorr_var.get()
        ds_kernel, sc_mode, sc_val = self.downscale_var.get(), self.scale_mode_var.get(), self.scale_val_entry.get().strip()
        verbose_lvl, device_str, quality_str, filter_str = int(self.verbose_var.get()), self.device_var.get(), self.quality_var.get(), self.filter_var.get()
        use_half_oidn, clean_aux, inplace = (self.precision_var.get() == "half"), self.clean_aux_var.get(), self.inplace_var.get()
        skip_empty = self.skip_empty_var.get()
        maxmem_str, future_args = self.maxmem_entry.get().strip(), self.future_args.get().strip()
        maxmem_mb = int(maxmem_str) if maxmem_str.isdigit() else -1

        comp_attr_name, _ = self.compression_map.get(self.comp_var.get(), ("DWAB_COMPRESSION", "20"))
        level_str = self.comp_level_entry.get().strip()
        use_half_exr = self.exr_depth_var.get().startswith("HALF")

        if self.sep_aux_comp_var.get():
            aux_comp_attr, aux_comp_lvl = self.compression_map.get(self.aux_comp_var.get(), ("ZIP_COMPRESSION", "4"))
        else:
            aux_comp_attr, aux_comp_lvl = comp_attr_name, level_str

        use_async_save = self.async_save_var.get() and total > 1
        exr_worker_threads = max(1, CPU_THREADS // 4) if use_async_save else CPU_THREADS

        stride = 2 if decorr_mode.startswith("2x2") else (3 if decorr_mode.startswith("3x3") else 1)
        use_direct_dll = (self.native_oidn is not None) and (not future_args)

        t_batch_start = time.perf_counter()
        self.log("\n==========================================")
        self.log(f"RUN START ({total} Pass(es) | Engine: {'Direct RAM/VRAM DLL' if use_direct_dll else 'CLI'} | Device: {device_str} ({'FP16' if use_half_oidn else 'FP32'}) | Scale: {sc_mode}={sc_val})")
        self.update_progress(0, total, "Running")

        save_futures = []
        executor = ThreadPoolExecutor(max_workers=4) if use_async_save else None

        try:
            if use_direct_dll:
                self.native_oidn.ensure_device(device_str, verbose_lvl)

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                tmp_alb, tmp_nrm = tmp_path / "albedo.pfm", tmp_path / "normal.pfm"
                tmp_alb_s, tmp_nrm_s = tmp_path / "albedo_s.pfm", tmp_path / "normal_s.pfm"
                tmp_hdr, tmp_out = tmp_path / "input.pfm", tmp_path / "output.pfm"

                cached_frame_id = None
                has_alb = has_nrm = False
                alb_rgb_ram = nrm_rgb_ram = packed_alb_ram = packed_nrm_ram = None

                for idx, item in enumerate(items_to_process, start=1):
                    if self.cancel_event.is_set():
                        self.log("--- Cancelled by user ---")
                        break
                    if executor is not None and len(save_futures) >= 6:
                        save_futures.pop(0).result()

                    target = item["target"]
                    out_path = target.parent / f"{target.stem}_denoised.exr"
                    self.update_progress(idx - 1, total, f"Pass {idx}/{total}")
                    self.after(0, lambda i=idx, t=total, n=item["pass_name"]: self.start_viewport_spinner(f"Denoising [{i}/{t}] {n}..."))

                    try:
                        t_denoise_start = time.perf_counter()
                        orig_rgb, alpha_ch, _ = self.read_exr_native(target)
                        orig_rgb = np.maximum(np.nan_to_num(orig_rgb, nan=0.0, posinf=65500.0, neginf=0.0), 0.0)

                        if skip_empty and float(np.max(orig_rgb)) <= 1e-6:
                            self.empty_passes.setdefault(item["frame_id"], set()).add(item["pass_name"])
                            self.log(f"[{idx}/{total}] ⚫ Skipped Empty (100% Black) Pass: {target.name}")
                            self.update_progress(idx, total, "Done" if idx == total else f"Pass {idx}/{total}")
                            continue

                        if cached_frame_id != item["frame_id"]:
                            has_alb = has_nrm = False
                            alb_rgb_ram = nrm_rgb_ram = None
                            if aux_mode != "None (Color Only)" and item["albedo"]:
                                alb_rgb_ram, _, _ = self.read_exr_native(item["albedo"])
                                alb_rgb_ram = np.clip(np.nan_to_num(alb_rgb_ram, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                                has_alb = True

                            chosen_nrm = (item["normal_pre"] or item["normal_bump"]) if aux_mode == "Albedo + Normals (Pre-Bump)" else (
                                (item["normal_bump"] or item["normal_pre"]) if aux_mode == "Albedo + Normals (Bump)" else None
                            )
                            if chosen_nrm:
                                nrm_rgb_ram, _, _ = self.read_exr_native(chosen_nrm)
                                nrm_rgb_ram = np.nan_to_num(nrm_rgb_ram, nan=0.0, posinf=1.0, neginf=-1.0)
                                has_nrm = True

                            if use_direct_dll:
                                if stride == 1:
                                    self.native_oidn.upload_aux_to_vram(alb_rgb_ram if has_alb else None, nrm_rgb_ram if has_nrm else None, use_half_oidn)
                                else:
                                    if has_alb:
                                        packed_alb_ram, _, _ = self.stride_pack(alb_rgb_ram, stride)
                                    if has_nrm:
                                        packed_nrm_ram, _, _ = self.stride_pack(nrm_rgb_ram, stride)
                            else:
                                if has_alb:
                                    self.write_pfm(alb_rgb_ram, tmp_alb)
                                    if stride > 1:
                                        self.write_pfm(self.stride_pack(alb_rgb_ram, stride)[0], tmp_alb_s)
                                if has_nrm:
                                    self.write_pfm(nrm_rgb_ram, tmp_nrm)
                                    if stride > 1:
                                        self.write_pfm(self.stride_pack(nrm_rgb_ram, stride)[0], tmp_nrm_s)
                            cached_frame_id = item["frame_id"]

                        work_rgb = orig_rgb.copy()
                        ldr_scale, use_ldr = 1.0, False

                        if hdr_mode == "HDR Clamp Fireflies":
                            work_rgb = np.minimum(work_rgb, max(float(np.percentile(work_rgb, 99.5)) * 3.0, 1e-4))
                        elif hdr_mode == "LDR Normalized":
                            ldr_scale = max(float(np.percentile(work_rgb, 99.0)), 1e-6)
                            work_rgb, use_ldr = np.clip(work_rgb / ldr_scale, 0.0, 1.0), True

                        if use_direct_dll:
                            if stride > 1:
                                packed_hdr, _, _ = self.stride_pack(work_rgb, stride)
                                self.native_oidn.upload_aux_to_vram(packed_alb_ram if has_alb else None, packed_nrm_ram if has_nrm else None, use_half_oidn)
                                den_packed = self.native_oidn.denoise_in_vram(packed_hdr, has_alb, has_nrm, filter_str, quality_str, use_half_oidn, not use_ldr, clean_aux, inplace, maxmem_mb)
                                work_rgb = self.stride_unpack(den_packed, work_rgb, stride)
                                self.native_oidn.upload_aux_to_vram(alb_rgb_ram if has_alb else None, nrm_rgb_ram if has_nrm else None, use_half_oidn)

                            denoised_rgb = self.native_oidn.denoise_in_vram(work_rgb, has_alb, has_nrm, filter_str, quality_str, use_half_oidn, not use_ldr, clean_aux, inplace, maxmem_mb)
                            if use_ldr:
                                denoised_rgb = denoised_rgb * ldr_scale
                            t_denoise_elapsed = time.perf_counter() - t_denoise_start
                        else:
                            if stride > 1:
                                self.write_pfm(self.stride_pack(work_rgb, stride)[0], tmp_hdr)
                                if tmp_out.exists():
                                    tmp_out.unlink()
                                rc, _, _ = self.run_oidn_cli(self.build_oidn_cmd(tmp_hdr, tmp_out, tmp_alb_s if has_alb else None, tmp_nrm_s if has_nrm else None, use_ldr))
                                if rc == 0 and tmp_out.exists():
                                    work_rgb = self.stride_unpack(self.read_pfm(tmp_out), work_rgb, stride)

                            if tmp_out.exists():
                                tmp_out.unlink()
                            self.write_pfm(work_rgb, tmp_hdr)
                            ret_code, stdout_str, stderr_str = self.run_oidn_cli(self.build_oidn_cmd(tmp_hdr, tmp_out, tmp_alb if has_alb else None, tmp_nrm if has_nrm else None, use_ldr))
                            if self.cancel_event.is_set():
                                break
                            if ret_code != 0 or not tmp_out.exists():
                                self.log(f"ERROR (Exit {ret_code}):\n{stdout_str}\n{stderr_str}")
                                continue
                            denoised_rgb = self.read_pfm(tmp_out) * (ldr_scale if use_ldr else 1.0)
                            t_denoise_elapsed = time.perf_counter() - t_denoise_start

                        args_tup = (idx, total, out_path, orig_rgb, denoised_rgb, alpha_ch, ds_kernel, sc_mode, sc_val, comp_attr_name, level_str, use_half_exr, t_denoise_elapsed, exr_worker_threads)
                        if executor is not None:
                            save_futures.append(executor.submit(self._save_pass_worker, *args_tup))
                        else:
                            self._save_pass_worker(*args_tup)

                        self.update_progress(idx, total, "Done" if idx == total else f"Pass {idx}/{total}")
                    except Exception as e:
                        self.log(f"System error on {target.name}: {str(e)}")

                for fut in save_futures:
                    if self.cancel_event.is_set():
                        break
                    fut.result()

                if not is_diagnostic and not self.cancel_event.is_set() and sc_mode != "No Downscale (100%)" and self.downscale_aux_var.get():
                    self.log(f"--- Scaling Aux/Mask/ID EXRs (Codec: {aux_comp_attr}) ---")
                    self.after(0, lambda: self.start_viewport_spinner("Scaling Aux & Mask passes..."))
                    for aux_info in self.aux_mask_files:
                        if self.cancel_event.is_set():
                            break
                        aux_path, p_type = aux_info["path"], aux_info["type"]
                        try:
                            t_aux_start = time.perf_counter()
                            is_id_map = p_type in ("material_id", "object_id")
                            a_rgb, a_alpha, _ = self.read_exr_native(aux_path, preserve_uint32=is_id_map)
                            a_clean = np.nan_to_num(a_rgb, nan=0.0, posinf=65500.0, neginf=-65500.0) if a_rgb.dtype != np.uint32 else a_rgb
                            ah, aw, _ = a_clean.shape
                            tw, th = DownscaleEngine.compute_target_size(aw, ah, sc_mode, sc_val)
                            if (tw, th) == (aw, ah):
                                continue
                            a_down = DownscaleEngine.downscale_aux_pass(a_clean, tw, th, p_type, ds_kernel)
                            a_al_down = DownscaleEngine.downscale_aux_pass(np.dstack([a_alpha] * 3), tw, th, "alpha", ds_kernel)[:, :, 0] if a_alpha is not None else None
                            use_half_this_aux = False if p_type in ("depth", "position") else use_half_exr
                            suffix = "_50pct.exr" if (tw == aw // 2 and th == ah // 2) else "_scaled.exr"
                            out_aux = aux_path.parent / f"{aux_path.stem}{suffix}"
                            self.write_exr_static(out_aux, a_down, a_al_down, aux_comp_attr, aux_comp_lvl, use_half_this_aux, worker_threads=CPU_THREADS, is_data_pass=(p_type != "albedo"))
                            self.log(f" -> Aux [{p_type}] {tw}×{th} ({out_aux.stat().st_size / 1024.0:.1f} KB | {time.perf_counter() - t_aux_start:.1f}s): {out_aux.name}")
                        except Exception as ex:
                            self.log(f"Aux scale warning ({aux_path.name}): {ex}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        if not self.cancel_event.is_set():
            self.log(f"--- Run Finished in {time.perf_counter() - t_batch_start:.1f}s ---")
        with self.cache_lock:
            self.linear_lru_cache.clear()
        self.after(0, lambda: self._set_running_ui_state(False))
        self.after(0, lambda: self.parse_indigo_passes(select_channel=initial_channel_before_run))


if __name__ == "__main__":
    app = PolyDenoiseStudio()
    app.mainloop()
# === END OF FILE ===