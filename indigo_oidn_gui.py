import os
import sys
import json
import re
import time
import ctypes
import tempfile
import threading
import subprocess
import webbrowser
import urllib.request
import urllib.error
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import OpenEXR

APP_NAME = "PolyDenoise Studio"
APP_VERSION = "1.2.0"
TOOL_GITHUB_REPO = "polygonmanufaktur/polydenoise-studio"
POLYGON_WEB_URL = "https://www.polygonmanufaktur.de"

BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
CONFIG_FILE = BASE_DIR / "oidn_config.json"


def get_initial_oidn_path() -> Path:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if "oidn_bin" in data and Path(data["oidn_bin"]).exists():
                return Path(data["oidn_bin"])
        except Exception:
            pass
    for c in [BASE_DIR / "oidnDenoise.exe", BASE_DIR / "oidn" / "oidnDenoise.exe", BASE_DIR / "_internal" / "oidnDenoise.exe"]:
        if c.exists():
            return c
    return BASE_DIR / "oidnDenoise.exe"


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


for d in [BASE_DIR, BASE_DIR / "OpenEXR.libs", BASE_DIR / "_internal", INITIAL_OIDN_BIN.parent]:
    register_dll_dir(d)

# OpenEXR 3.5 API: Set global C++ thread pool strictly ONCE at startup
CPU_THREADS = os.cpu_count() or 8
if hasattr(OpenEXR, "set_global_thread_count"):
    OpenEXR.set_global_thread_count(CPU_THREADS)

_LUT_STEPS = 4096
_SRGB_LUT_12BIT = (np.power(np.linspace(0.0, 1.0, _LUT_STEPS, dtype=np.float32), 1.0 / 2.2) * 255.0).astype(np.float32)


# ================= HOVER TOOLTIP WIDGET =================
class CTkToolTip:
    def __init__(self, widget, text: str, delay: int = 320):
        self.widget, self.text, self.delay = widget, text, delay
        self.tip_window = None
        self.after_id = None
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
        tk.Label(
            frame, text=self.text, justify="left", background="#1e1e1e", foreground="#e8e8e8",
            font=("Segoe UI", 9), wraplength=330, padx=8, pady=6
        ).pack()

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

        self.main_btn = ctk.CTkButton(
            self, text=self._format_btn_text(self.variable.get()), width=width, height=28,
            anchor="w", fg_color="#1f538d", hover_color="#14375e", command=self.toggle_popup
        )
        self.main_btn.pack(side="left", padx=0)

        self.next_btn = ctk.CTkButton(self, text="▶", width=24, height=28, fg_color="#3a3a3a", hover_color="#505050", command=lambda: self.step(1))
        self.next_btn.pack(side="left", padx=(2, 0))
        self.variable.trace_add("write", self._on_var_write)

    def _get_label(self, val: str) -> str:
        return self.display_formatter(val) if self.display_formatter else val

    def _format_btn_text(self, val: str) -> str:
        disp = self._get_label(val)
        max_chars = max(10, int(self.btn_width / 7.2))
        return (disp[: max_chars - 2] + "… ▾") if len(disp) > max_chars else f"{disp}  ▾"

    def _on_var_write(self, *_):
        self.main_btn.configure(text=self._format_btn_text(self.variable.get()))

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
        pop_w = max(self.btn_width + 75, 300)
        pop_h = min(max(len(self.values) * 22 + 45, 130), 380)
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

        listbox = tk.Listbox(
            list_frame, bg="#1e1e1e", fg="#e8e8e8", selectbackground="#1f6aa5", selectforeground="#ffffff",
            font=("Consolas", 10), highlightthickness=0, bd=0, activestyle="none", yscrollcommand=scrollbar.set
        )
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
    FORMAT_FLOAT3 = 3
    FORMAT_HALF3 = 259
    STORAGE_DEVICE = 2
    QUALITY_MAP = {"fast": 4, "balanced": 5, "high": 6}

    def __init__(self, dll_path: Path):
        register_dll_dir(dll_path.parent)
        self.dll_path = dll_path
        self.lib = ctypes.CDLL(str(dll_path))
        self._bind_signatures()
        self.device = None
        self.current_device_key = None
        self.buf_cache = {}
        self.buf_shape_key = None
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
            self.lib.oidnNewBufferWithStorage.argtypes = [vp, cs, ci]
            self.lib.oidnNewBufferWithStorage.restype = vp
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
            dev = self.lib.oidnNewDevice(1)  # Safe CPU device probe (avoids Issue #285)
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

    def ensure_device(self, device_str: str, verbose: int):
        key = (device_str, verbose)
        if self.device is not None and self.current_device_key == key:
            return
        self.release_buffers()
        if self.device is not None:
            self.lib.oidnReleaseDevice(self.device)
            self.device = None

        # Workaround for OIDN Issue #285 (Access Violation on dual Nvidia+Intel systems)
        if device_str == "default":
            for fallback_dev in ("cuda", "hip", "cpu"):
                dev_try = self.lib.oidnNewDevice(self.DEVICE_MAP[fallback_dev])
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
        has_storage_api = hasattr(self.lib, "oidnNewBufferWithStorage")
        for name in ("color", "out", "albedo", "normal"):
            buf = None
            if has_storage_api:
                buf = self.lib.oidnNewBufferWithStorage(self.device, byte_size, self.STORAGE_DEVICE)
            if not buf:
                buf = self.lib.oidnNewBuffer(self.device, byte_size)
            if not buf:
                self._check_error()
                raise RuntimeError(f"Failed to allocate OIDN VRAM buffer ({byte_size} B).")
            self.buf_cache[name] = buf
        self.buf_shape_key = key

    @staticmethod
    def _prep_array(arr: np.ndarray, use_half: bool) -> np.ndarray:
        if use_half:
            return np.ascontiguousarray(np.clip(arr, -65500.0, 65500.0), dtype=np.float16)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def upload_aux_to_vram(self, alb_rgb: np.ndarray, nrm_rgb: np.ndarray, use_half: bool):
        ref = alb_rgb if alb_rgb is not None else nrm_rgb
        if ref is None:
            return
        h, w, _ = ref.shape
        self.ensure_buffers(h, w, use_half)
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
        dtype = np.float16 if use_half else np.float32
        fmt = self.FORMAT_HALF3 if use_half else self.FORMAT_FLOAT3

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
            w[m1] = 0.75 - (ax[m1] ** 2)
            w[m2] = 0.5 * ((1.5 - ax[m2]) ** 2)
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
            orig_dtype = rgb.dtype
            # OpenCV INTER_NEAREST supports float32; if uint32 ID map, index via coordinate grid
            if orig_dtype == np.uint32:
                yi = (np.arange(target_h, dtype=np.float64) * (h / float(target_h))).astype(np.int32)
                xi = (np.arange(target_w, dtype=np.float64) * (w / float(target_w))).astype(np.int32)
                return np.ascontiguousarray(rgb[yi[:, None], xi[None, :], :], dtype=np.uint32)
            res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            return np.ascontiguousarray(res[:, :, None] if res.ndim == 2 else res, dtype=np.float32)
        if "mask" in p_low or p_low == "alpha":
            res = cls.downscale(rgb, target_w, target_h, "Mitchell-Netravali (Smooth VFX)")
            return np.ascontiguousarray(np.clip(res, 0.0, 1.0), dtype=np.float32)
        if "depth" in p_low or "position" in p_low or "normal" in p_low:
            res = cls._resample_polyphase_2d(np.ascontiguousarray(rgb, dtype=np.float32), target_w, target_h, "mitchell")
            return np.ascontiguousarray(res, dtype=np.float32)
        return cls.downscale(rgb, target_w, target_h, ds_kernel)

    @classmethod
    def downscale(cls, rgb: np.ndarray, target_w: int, target_h: int, mode_label: str) -> np.ndarray:
        h, w, c = rgb.shape
        if target_w >= w and target_h >= h:
            return rgb
        if target_w == (w // 2) and target_h == (h // 2):
            return cls._downscale_exact_2x(rgb, mode_label)

        is_unclamped = ("Unclamped" in mode_label) or ("Catmull-Rom" in mode_label)
        k_type = "mks" if ("MKS" in mode_label or "Magic Kernel" in mode_label) else (
            "lanczos4" if "Lanczos-4" in mode_label else (
                "lanczos3" if "Lanczos-3" in mode_label else ("catrom" if "Catmull-Rom" in mode_label else "mitchell")
            )
        )
        src = np.ascontiguousarray(rgb, dtype=np.float32)

        if is_unclamped:
            res = cls._resample_polyphase_2d(src, target_w, target_h, k_type)
            if "2013" in mode_label:
                res = cv2.sepFilter2D(res, cv2.CV_32F, cls.K_MKS2013_SHARP, cls.K_MKS2013_SHARP, borderType=cv2.BORDER_REPLICATE)
            elif "2021" in mode_label:
                res = cv2.sepFilter2D(res, cv2.CV_32F, cls.K_MKS2021_SHARP, cls.K_MKS2021_SHARP, borderType=cv2.BORDER_REPLICATE)
            return np.ascontiguousarray(np.maximum(res, 0.0), dtype=np.float32)

        area_ref = cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if area_ref.ndim == 2:
            area_ref = area_ref[:, :, None]
        k3 = np.ones((3, 3), dtype=np.uint8)
        local_min, local_max = cv2.erode(area_ref, k3), cv2.dilate(area_ref, k3)
        if local_min.ndim == 2:
            local_min, local_max = local_min[:, :, None], local_max[:, :, None]

        lum = (0.2126 * src[:, :, 0] + 0.7152 * src[:, :, 1] + 0.0722 * src[:, :, 2]) if c >= 3 else src[:, :, 0]
        w_karis = (1.0 / (1.0 + np.maximum(lum, 0.0)))[:, :, None]
        dec = cls._resample_polyphase_2d(np.concatenate([src * w_karis, w_karis], axis=2), target_w, target_h, k_type)
        res = dec[:, :, :c] / np.maximum(dec[:, :, c : c + 1], 1e-7)

        if "2021" in mode_label:
            res = cv2.sepFilter2D(res, cv2.CV_32F, cls.K_MKS2021_SHARP, cls.K_MKS2021_SHARP, borderType=cv2.BORDER_REPLICATE)
        elif "2013" in mode_label:
            res = cv2.sepFilter2D(res, cv2.CV_32F, cls.K_MKS2013_SHARP, cls.K_MKS2013_SHARP, borderType=cv2.BORDER_REPLICATE)

        if "Lanczos-3" in mode_label:
            res = np.clip(res, local_min, local_max)
        else:
            span = (local_max - local_min) * 0.08
            res = np.clip(res, np.maximum(local_min - span, 0.0), local_max + span)
        return np.ascontiguousarray(np.maximum(res, 0.0), dtype=np.float32)

    @classmethod
    def _downscale_exact_2x(cls, rgb: np.ndarray, mode_label: str) -> np.ndarray:
        h, w, c = rgb.shape
        h2, w2 = h // 2, w // 2
        if h2 < 1 or w2 < 1:
            return rgb
        src = np.ascontiguousarray(rgb[: h2 * 2, : w2 * 2, :], dtype=np.float32)
        is_unclamped = ("Unclamped" in mode_label) or ("Catmull-Rom" in mode_label)

        k_down = cls.K_MKS_DOWN if ("MKS" in mode_label or "Magic Kernel" in mode_label) else (
            cls.K_LANCZOS4_DOWN if "Lanczos-4" in mode_label else (
                cls.K_LANCZOS3_DOWN if "Lanczos-3" in mode_label else (cls.K_CATROM_DOWN if "Catmull-Rom" in mode_label else cls.K_MITCHELL_DOWN)
            )
        )
        pad = (len(k_down) // 2) - 1

        if is_unclamped:
            res = cv2.sepFilter2D(src, cv2.CV_32F, k_down, k_down, anchor=(pad, pad), borderType=cv2.BORDER_REPLICATE)[0::2, 0::2, :]
            if "2013" in mode_label:
                res = cv2.sepFilter2D(res, cv2.CV_32F, cls.K_MKS2013_SHARP, cls.K_MKS2013_SHARP, borderType=cv2.BORDER_REPLICATE)
            elif "2021" in mode_label:
                res = cv2.sepFilter2D(res, cv2.CV_32F, cls.K_MKS2021_SHARP, cls.K_MKS2021_SHARP, borderType=cv2.BORDER_REPLICATE)
            return np.ascontiguousarray(np.maximum(res, 0.0), dtype=np.float32)

        b00, b01, b10, b11 = src[0::2, 0::2, :], src[0::2, 1::2, :], src[1::2, 0::2, :], src[1::2, 1::2, :]
        local_min = np.minimum(np.minimum(b00, b01), np.minimum(b10, b11))
        local_max = np.maximum(np.maximum(b00, b01), np.maximum(b10, b11))

        lum = (0.2126 * src[:, :, 0] + 0.7152 * src[:, :, 1] + 0.0722 * src[:, :, 2]) if c >= 3 else src[:, :, 0]
        w_karis = (1.0 / (1.0 + np.maximum(lum, 0.0)))[:, :, None]
        packed = np.concatenate([src * w_karis, w_karis], axis=2)

        dec = cv2.sepFilter2D(packed, cv2.CV_32F, k_down, k_down, anchor=(pad, pad), borderType=cv2.BORDER_REPLICATE)[0::2, 0::2, :]
        res = dec[:, :, :c] / np.maximum(dec[:, :, c : c + 1], 1e-7)

        if "2021" in mode_label:
            res = cv2.sepFilter2D(res, cv2.CV_32F, cls.K_MKS2021_SHARP, cls.K_MKS2021_SHARP, borderType=cv2.BORDER_REPLICATE)
        elif "2013" in mode_label:
            res = cv2.sepFilter2D(res, cv2.CV_32F, cls.K_MKS2013_SHARP, cls.K_MKS2013_SHARP, borderType=cv2.BORDER_REPLICATE)

        if "Lanczos-3" in mode_label:
            res = np.clip(res, local_min, local_max)
        else:
            span = (local_max - local_min) * 0.08
            res = np.clip(res, np.maximum(local_min - span, 0.0), local_max + span)
        return np.ascontiguousarray(np.maximum(res, 0.0), dtype=np.float32)


# ================= MAIN APPLICATION GUI (POLYDENOISE STUDIO) =================
class PolyDenoiseStudio(ctk.CTk):
    BEAUTY_SUM_KEY = "[⚡ Sum All Lights (Beauty)]"

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
        self.primary_linear = self.alternate_linear = None
        self.primary_rgb = self.alternate_rgb = None
        self.primary_tag = self.alternate_tag = ""
        self.tk_img_primary = self.tk_img_alternate = None
        self.canvas_img_id = None
        self._last_canvas_dims = (0, 0)
        self._dither_cache = {}
        self.is_holding_compare = False
        self.show_denoised_default = False
        self.config_window = None

        self.is_running = False
        self.cancel_event = threading.Event()
        self.active_process = None

        self.setup_ui()
        self.bind("<KeyPress-space>", self._on_space_press)
        self.bind("<KeyRelease-space>", self._on_space_release)

    def _apply_window_icon(self, target_win):
        ico_candidates = [BASE_DIR / "polydenoise.ico", BASE_DIR / "_internal" / "polydenoise.ico"]
        png_candidates = [BASE_DIR / "icon_source.png", BASE_DIR / "_internal" / "icon_source.png"]

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
        for dll_p in [self.oidn_bin.parent / "OpenImageDenoise.dll", BASE_DIR / "OpenImageDenoise.dll", BASE_DIR / "oidn" / "OpenImageDenoise.dll", BASE_DIR / "_internal" / "OpenImageDenoise.dll"]:
            if dll_p.exists():
                try:
                    self.native_oidn = OIDNNativeDLL(dll_p)
                    return
                except Exception:
                    pass
        self.native_oidn = None

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

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= SIDEBAR =================
        self.sidebar = ctk.CTkFrame(self, width=375)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar.grid_propagate(False)

        bold_font = ctk.CTkFont(size=14, weight="bold")
        lbl_font = ctk.CTkFont(size=11)

        # --- Section 1: Source Folder & ⚙ Config Icon ---
        ctk.CTkLabel(self.sidebar, text="1. Source Folder & Settings", font=bold_font).pack(pady=(8, 3), anchor="w", padx=10)
        p_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        p_row.pack(fill="x", padx=8, pady=2)

        self.folder_btn = ctk.CTkButton(p_row, text="Select Render Folder", height=28, command=self.select_folder)
        self.folder_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        CTkToolTip(self.folder_btn, "Select folder with EXR render passes.")

        self.cfg_btn = ctk.CTkButton(
            p_row, text="⚙", width=36, height=28, fg_color="#3a3a3a", hover_color="#555555",
            font=ctk.CTkFont(size=15), command=self.open_config_popup
        )
        self.cfg_btn.pack(side="right")
        CTkToolTip(self.cfg_btn, "Open Configuration, OIDN Binary Path, Version Check & Credits.")

        self.folder_label = ctk.CTkLabel(
            self.sidebar, text=f"OIDN v{self.get_oidn_version_str()} | Folder: None",
            font=lbl_font, text_color="gray", wraplength=350
        )
        self.folder_label.pack(padx=10, pady=(2, 4), anchor="w")

        # --- Section 2: Downscaling & OpenEXR ---
        ctk.CTkLabel(self.sidebar, text="2. Downscaling & OpenEXR v3.5", font=bold_font).pack(pady=(4, 2), anchor="w", padx=10)
        ctk.CTkLabel(self.sidebar, text="Downscale Filter Kernel", font=lbl_font).pack(anchor="w", padx=10)
        self.downscale_var = ctk.StringVar(value=self.downscale_options[0])
        self.downscale_menu = ctk.CTkOptionMenu(self.sidebar, variable=self.downscale_var, values=self.downscale_options, height=26, command=self.on_downscale_mode_change)
        self.downscale_menu.pack(fill="x", padx=8, pady=(0, 4))
        CTkToolTip(
            self.downscale_menu,
            "Filter kernel applied AFTER full-res OIDN denoising:\n"
            "• MKS 2013 (Crisp / HDR-Safe): Default crisp filter with Karis anti-ringing.\n"
            "• MKS 2013 Unclamped: Pure linear deconvolution without HDR dampening.\n"
            "• Lanczos-4 Unclamped: 16-tap Sinc filter for extreme micro-texture acuity.\n"
            "• Catmull-Rom Bicubic: Sharp ArchViz bicubic filter (B=0, C=0.5).\n"
            "• MKS 2021 / Lanczos-3 AR / Mitchell: Balanced & smooth HDR-safe options.\n"
            "Note: Material/Object IDs automatically use Nearest-Neighbor & preserve uint32!"
        )

        scale_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        scale_row.pack(fill="x", padx=8, pady=(0, 4))
        s_col1 = ctk.CTkFrame(scale_row, fg_color="transparent")
        s_col1.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(s_col1, text="Scale Mode", font=lbl_font).pack(anchor="w")
        self.scale_mode_var = ctk.StringVar(value="Scale %")
        self.scale_mode_menu = ctk.CTkOptionMenu(s_col1, variable=self.scale_mode_var, values=self.scale_mode_options, width=145, height=26, command=self.on_scale_mode_select)
        self.scale_mode_menu.pack()
        CTkToolTip(self.scale_mode_menu, "Scale %, Target Width (px), Target Height (px), or No Downscale (100%).")

        s_col2 = ctk.CTkFrame(scale_row, fg_color="transparent")
        s_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(s_col2, text="Value", font=lbl_font).pack(anchor="w")
        self.scale_val_entry = ctk.CTkEntry(s_col2, width=68, height=26)
        self.scale_val_entry.insert(0, "50")
        self.scale_val_entry.pack()
        self.scale_val_entry.bind("<Return>", lambda _: self.on_downscale_mode_change())
        self.scale_val_entry.bind("<FocusOut>", lambda _: self.on_downscale_mode_change())
        CTkToolTip(self.scale_val_entry, "Enter percentage (e.g. 50) or target pixels (e.g. 1920) and press Enter.")

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
        CTkToolTip(self.comp_menu, "OpenEXR v3.5.3 codec for denoised Beauty/Light passes.\nNote: Embedded Alpha channels are always compressed losslessly.")

        c_col2 = ctk.CTkFrame(exr_grid, fg_color="transparent")
        c_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(c_col2, text="Level", font=lbl_font).pack(anchor="w")
        self.comp_level_entry = ctk.CTkEntry(c_col2, width=52, height=26)
        self.comp_level_entry.insert(0, "20")
        self.comp_level_entry.pack()
        CTkToolTip(self.comp_level_entry, "Compression Level (Default DWAB=20 for high quality, ZSTD=1-22, LJ2K=1-150).")

        c_col3 = ctk.CTkFrame(exr_grid, fg_color="transparent")
        c_col3.pack(side="right")
        ctk.CTkLabel(c_col3, text="Bit Depth", font=lbl_font).pack(anchor="w")
        self.exr_depth_var = ctk.StringVar(value="HALF (16-bit)")
        self.exr_depth_menu = ctk.CTkOptionMenu(c_col3, variable=self.exr_depth_var, values=["HALF (16-bit)", "FLOAT (32-bit)"], width=125, height=26)
        self.exr_depth_menu.pack()
        CTkToolTip(self.exr_depth_menu, "HALF (16-bit Float) or FLOAT (32-bit Float).")

        aux_codec_row = ctk.CTkFrame(self.sidebar, fg_color="#222222", corner_radius=6)
        aux_codec_row.pack(fill="x", padx=8, pady=(0, 6))
        self.sep_aux_comp_var = ctk.BooleanVar(value=True)
        self.sep_aux_cb = ctk.CTkCheckBox(
            aux_codec_row, text="Separate Aux/Mask/ID Codec:", variable=self.sep_aux_comp_var,
            font=lbl_font, checkbox_width=18, checkbox_height=18, command=self._on_toggle_aux_codec
        )
        self.sep_aux_cb.pack(side="left", padx=(8, 4), pady=5)
        CTkToolTip(
            self.sep_aux_cb,
            "Uses a dedicated lossless codec (default: ZIP) for Masks, Depth, Material ID, Object ID, Alpha, and Normals.\n"
            "Also tags non-color passes with OpenEXR v3.5 'colorInteropID = data'."
        )

        self.aux_comp_var = ctk.StringVar(value="ZIP (Lossless 16-Line)")
        self.aux_comp_menu = ctk.CTkOptionMenu(aux_codec_row, variable=self.aux_comp_var, values=self.aux_compression_options, width=165, height=24)
        self.aux_comp_menu.pack(side="right", padx=(0, 6), pady=5)
        CTkToolTip(self.aux_comp_menu, "Lossless codec for flat Masks, Depth, and ID passes (ZIP, ZSTD, RLE, ZIPS, PIZ).")

        # --- Section 3: OIDN & Guide Passes ---
        ctk.CTkLabel(self.sidebar, text="3. OIDN & Guide Parameters", font=bold_font).pack(pady=(2, 2), anchor="w", padx=10)
        row_aux = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row_aux.pack(fill="x", padx=8, pady=(0, 4))

        a_col1 = ctk.CTkFrame(row_aux, fg_color="transparent")
        a_col1.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkLabel(a_col1, text="Aux Guide Passes", font=lbl_font).pack(anchor="w")
        self.aux_mode_var = ctk.StringVar(value="Albedo + Normals (Bump)")
        self.aux_mode_menu = ctk.CTkOptionMenu(
            a_col1, variable=self.aux_mode_var, width=170, height=26,
            values=["Albedo + Normals (Bump)", "Albedo + Normals (Pre-Bump)", "Albedo Only", "None (Color Only)"]
        )
        self.aux_mode_menu.pack(fill="x")
        CTkToolTip(self.aux_mode_menu, "Selects Albedo & Normal guide passes uploaded to GPU VRAM.")

        a_col2 = ctk.CTkFrame(row_aux, fg_color="transparent")
        a_col2.pack(side="right", fill="x", expand=True)
        ctk.CTkLabel(a_col2, text="Noise De-Correlation", font=lbl_font).pack(anchor="w")
        self.decorr_var = ctk.StringVar(value="Standard 1x (1px Noise)")
        self.decorr_menu = ctk.CTkOptionMenu(
            a_col2, variable=self.decorr_var, width=165, height=26,
            values=["Standard 1x (1px Noise)", "2x2 Stride + 1x Polish", "3x3 Stride + 1x Polish"]
        )
        self.decorr_menu.pack(fill="x")
        CTkToolTip(self.decorr_menu, "Standard 1x for 200% renders without SuperSampling.\n2x2 / 3x3 Stride breaks correlated multi-pixel supersampling grain.")

        row_dqp = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row_dqp.pack(fill="x", padx=8, pady=(0, 4))
        d_col1 = ctk.CTkFrame(row_dqp, fg_color="transparent")
        d_col1.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(d_col1, text="Device (-d)", font=lbl_font).pack(anchor="w")
        self.device_var = ctk.StringVar(value="cuda")
        self.device_menu = ctk.CTkOptionMenu(d_col1, variable=self.device_var, values=["cuda", "cpu", "default", "sycl", "hip", "metal"], width=105, height=26)
        self.device_menu.pack()
        CTkToolTip(self.device_menu, "Compute backend (cuda = NVIDIA Tensor Cores).")

        d_col2 = ctk.CTkFrame(row_dqp, fg_color="transparent")
        d_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(d_col2, text="Quality (-q)", font=lbl_font).pack(anchor="w")
        self.quality_var = ctk.StringVar(value="high")
        self.quality_menu = ctk.CTkOptionMenu(d_col2, variable=self.quality_var, values=["high", "balanced", "fast"], width=115, height=26)
        self.quality_menu.pack()
        CTkToolTip(self.quality_menu, "OIDN neural network quality preset.")

        d_col3 = ctk.CTkFrame(row_dqp, fg_color="transparent")
        d_col3.pack(side="right")
        ctk.CTkLabel(d_col3, text="Precision (-t)", font=lbl_font).pack(anchor="w")
        self.precision_var = ctk.StringVar(value="half")
        self.precision_menu = ctk.CTkOptionMenu(d_col3, variable=self.precision_var, values=["half", "float"], width=110, height=26)
        self.precision_menu.pack()
        CTkToolTip(self.precision_menu, "VRAM tensor format: half (FP16 = 259, 2x faster) or float (FP32 = 3).")

        row_hfv = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row_hfv.pack(fill="x", padx=8, pady=(0, 4))
        h_col1 = ctk.CTkFrame(row_hfv, fg_color="transparent")
        h_col1.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(h_col1, text="HDR Mode", font=lbl_font).pack(anchor="w")
        self.hdr_mode_var = ctk.StringVar(value="HDR Linear")
        self.hdr_mode_menu = ctk.CTkOptionMenu(h_col1, variable=self.hdr_mode_var, values=["HDR Linear", "HDR Clamp Fireflies", "LDR Normalized"], width=150, height=26)
        self.hdr_mode_menu.pack()
        CTkToolTip(self.hdr_mode_menu, "HDR transfer curve mode.")

        h_col2 = ctk.CTkFrame(row_hfv, fg_color="transparent")
        h_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(h_col2, text="Filter (-f)", font=lbl_font).pack(anchor="w")
        self.filter_var = ctk.StringVar(value="RT")
        self.filter_menu = ctk.CTkOptionMenu(h_col2, variable=self.filter_var, values=["RT", "RTLightmap"], width=105, height=26)
        self.filter_menu.pack()
        CTkToolTip(self.filter_menu, "RT (Raytracing) or RTLightmap.")

        h_col3 = ctk.CTkFrame(row_hfv, fg_color="transparent")
        h_col3.pack(side="right")
        ctk.CTkLabel(h_col3, text="Verbose", font=lbl_font).pack(anchor="w")
        self.verbose_var = ctk.StringVar(value="2")
        self.verbose_menu = ctk.CTkOptionMenu(h_col3, variable=self.verbose_var, values=["0", "1", "2", "3", "4"], width=75, height=26)
        self.verbose_menu.pack()
        CTkToolTip(self.verbose_menu, "Console verbosity level (0=Silent to 4=Debug).")

        # --- Compact 2-Column Checkbox Grid ---
        cb_grid = ctk.CTkFrame(self.sidebar, fg_color="#222222", corner_radius=6)
        cb_grid.pack(fill="x", padx=8, pady=(2, 5))
        cb_grid.grid_columnconfigure(0, weight=1)
        cb_grid.grid_columnconfigure(1, weight=1)

        self.clean_aux_var = ctk.BooleanVar(value=True)
        cb1 = ctk.CTkCheckBox(cb_grid, text="CleanAux (--clean_aux)", variable=self.clean_aux_var, font=lbl_font, checkbox_width=18, checkbox_height=18)
        cb1.grid(row=0, column=0, sticky="w", padx=8, pady=(5, 2))
        CTkToolTip(cb1, "Assumes Albedo & Normals are noise-free to preserve micro-bumps.")

        self.inplace_var = ctk.BooleanVar(value=False)
        cb2 = ctk.CTkCheckBox(cb_grid, text="Inplace VRAM", variable=self.inplace_var, font=lbl_font, checkbox_width=18, checkbox_height=18)
        cb2.grid(row=0, column=1, sticky="w", padx=8, pady=(5, 2))
        CTkToolTip(
            cb2,
            "Shares input/output VRAM buffer to save ~75MB VRAM.\n"
            "Drawback: Overwrites input buffers in-place and adds overhead when tiling."
        )

        self.downscale_aux_var = ctk.BooleanVar(value=True)
        cb3 = ctk.CTkCheckBox(cb_grid, text="Scale Aux & Masks", variable=self.downscale_aux_var, font=lbl_font, checkbox_width=18, checkbox_height=18)
        cb3.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        CTkToolTip(cb3, "Downscales Aux, Masks, Depth & IDs to match target resolution.")

        self.async_save_var = ctk.BooleanVar(value=True)
        cb4 = ctk.CTkCheckBox(cb_grid, text="Async RAM Save", variable=self.async_save_var, font=lbl_font, checkbox_width=18, checkbox_height=18)
        cb4.grid(row=1, column=1, sticky="w", padx=8, pady=2)
        CTkToolTip(cb4, "Writes EXRs on 4 background CPU workers (with balanced OpenEXR thread allocation).")

        self.skip_empty_var = ctk.BooleanVar(value=True)
        cb5 = ctk.CTkCheckBox(
            cb_grid, text="Skip Empty (Black)", variable=self.skip_empty_var,
            font=lbl_font, checkbox_width=18, checkbox_height=18,
            command=lambda: self.parse_indigo_passes() if self.input_folder else None
        )
        cb5.grid(row=2, column=0, sticky="w", padx=8, pady=(2, 5))
        CTkToolTip(
            cb5,
            "Detects 100% black light passes (marks them with ⚫ [Empty] in the dropdown) "
            "and skips them during OIDN denoising."
        )

        self.skip_existing_var = ctk.BooleanVar(value=False)
        cb6 = ctk.CTkCheckBox(
            cb_grid, text="Skip Done (Resume)", variable=self.skip_existing_var,
            font=lbl_font, checkbox_width=18, checkbox_height=18,
            command=lambda: self.parse_indigo_passes() if self.input_folder else None
        )
        cb6.grid(row=2, column=1, sticky="w", padx=8, pady=(2, 5))
        CTkToolTip(cb6, "Skips raw passes that already have a '_denoised.exr' file on disk.")

        extra_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        extra_frame.pack(fill="x", padx=8, pady=(0, 5))
        self.maxmem_entry = ctk.CTkEntry(extra_frame, placeholder_text="MaxMem MB", width=110, height=26)
        self.maxmem_entry.pack(side="left", padx=(0, 5))
        CTkToolTip(self.maxmem_entry, "Optional VRAM limit in MB (empty = auto).")

        self.future_args = ctk.CTkEntry(extra_frame, placeholder_text="Custom CLI Args (e.g. OIDN v3)", height=26)
        self.future_args.pack(side="right", fill="x", expand=True)
        CTkToolTip(self.future_args, "Extra CLI flags for future OIDN v3.x features.")

        self.test_single_btn = ctk.CTkButton(
            self.sidebar, text="Test Single Pass (Current Channel)", command=self.start_single_test_thread,
            fg_color="#1f6aa5", height=34, state="disabled"
        )
        self.test_single_btn.pack(fill="x", padx=8, pady=(2, 4))
        CTkToolTip(self.test_single_btn, "Denoises only the currently selected Frame & Channel directly in VRAM.")

        self.start_btn = ctk.CTkButton(
            self.sidebar, text="Start Full Batch Denoise", command=self.on_start_or_cancel_click,
            fg_color="#2b7a2b", hover_color="#1f5c1f", height=38, state="disabled"
        )
        self.start_btn.pack(fill="x", padx=8, pady=(2, 8))
        CTkToolTip(self.start_btn, "Starts processing all queued passes. Turns into Cancel button while running.")

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
        CTkToolTip(self.frame_selector, "Scrollable Frame Selector with Search Filter & ◀ / ▶ step buttons.")

        ctk.CTkLabel(self.top_bar, text="Channel:").pack(side="left", padx=(2, 3), pady=6)
        self.channel_var = ctk.StringVar(value="-")
        self.channel_selector = ScrollableDropdown(
            self.top_bar, variable=self.channel_var, values=["-"], command=self.on_channel_change,
            width=195, display_formatter=self._format_channel_label
        )
        self.channel_selector.pack(side="left", padx=(0, 6), pady=6)
        CTkToolTip(
            self.channel_selector,
            "Lists clean base channels:\n"
            "• ✓ = _denoised version exists\n"
            "• ⚫ = Empty (100% black) pass\n"
            "Hold Left-Click (or Spacebar) on the image to show Denoised!"
        )

        self.swap_default_btn = ctk.CTkButton(
            self.top_bar, text="Hold: Denoised", width=105, height=28,
            fg_color="#3b5998", hover_color="#2d4373", command=self.toggle_default_view_mode
        )
        self.swap_default_btn.pack(side="left", padx=(0, 6), pady=6)
        CTkToolTip(self.swap_default_btn, "Switches whether the viewport shows Raw by default (Hold LMB = Denoised) or Denoised by default.")

        ctk.CTkLabel(self.top_bar, text="Zoom:").pack(side="left", padx=(2, 2), pady=6)
        self.zoom_var = ctk.StringVar(value="100%")
        self.zoom_menu = ctk.CTkOptionMenu(self.top_bar, variable=self.zoom_var, values=["100%", "200%", "300%", "50%", "Fit"], width=74, height=28, command=self.on_zoom_change)
        self.zoom_menu.pack(side="left", padx=(0, 6), pady=6)
        CTkToolTip(self.zoom_menu, "Viewport zoom (Ctrl + MouseWheel).")

        ctk.CTkLabel(self.top_bar, text="EV:").pack(side="left", padx=(2, 2), pady=6)
        self.ev_var = ctk.DoubleVar(value=0.0)
        self.ev_slider = ctk.CTkSlider(self.top_bar, from_=-5.0, to=5.0, number_of_steps=100, variable=self.ev_var, width=80, command=self.on_ev_change)
        self.ev_slider.pack(side="left", padx=(0, 3), pady=6)
        self.ev_reset_btn = ctk.CTkButton(self.top_bar, text="0.0", width=36, height=24, fg_color="#3a3a3a", command=self.reset_ev)
        self.ev_reset_btn.pack(side="left", padx=(0, 6), pady=6)
        CTkToolTip(self.ev_slider, "Real-time 60fps Exposure slider (-5 EV to +5 EV) via 12-bit RAM LUT (<2ms).")

        self.res_label = ctk.CTkLabel(self.top_bar, text="", text_color="#8ab4f8", font=lbl_font)
        self.res_label.pack(side="right", padx=8)

        # --- Viewport Canvas ---
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

        # --- Log Header Bar (Progress Bar hidden until active!) ---
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=0)
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=12)
        self.progress_bar.set(0.0)
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="", width=140, anchor="e", font=lbl_font)

        self.clear_log_btn = ctk.CTkButton(self.progress_frame, text="Clear Log", width=85, height=22, fg_color="#444444", hover_color="#5a5a5a", command=self.clear_log)
        self.clear_log_btn.grid(row=0, column=2, sticky="e", padx=(0, 2), pady=2)
        CTkToolTip(self.clear_log_btn, "Clears the console log output below.")

        self.log_textbox = ctk.CTkTextbox(self.main_frame, height=170, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_textbox.grid(row=3, column=0, sticky="nsew", padx=10, pady=(2, 8))

        self.log(f"{APP_NAME} v{APP_VERSION} | OpenEXR v{getattr(OpenEXR, '__version__', 'Unknown')} ({CPU_THREADS} Threads).")
        if self.native_oidn is not None:
            self.log(f"DIRECT RAM/VRAM ENGINE ACTIVE: OIDN v{self.native_oidn.version_str} ({self.native_oidn.dll_path})")
        elif self.oidn_bin.exists():
            self.log(f"OIDN CLI fallback loaded: {self.oidn_bin}")
        else:
            self.log("WARNING: OIDN not found. Click '⚙' in the top-left to select oidnDenoise.exe / OpenImageDenoise.dll.")

    # ================= ⚙ CLEAN CONFIG, VERSIONS & CREDITS POPUP =================
    def open_config_popup(self):
        if self.config_window is not None and self.config_window.winfo_exists():
            self.config_window.focus_force()
            return

        self.config_window = win = ctk.CTkToplevel(self)
        win.title(f"{APP_NAME} — Configuration & About")
        win.geometry("460x415")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        self._apply_window_icon(win)

        bold_f = ctk.CTkFont(size=14, weight="bold")
        mono_f = ctk.CTkFont(family="Consolas", size=12)
        small_f = ctk.CTkFont(size=11)

        # 1. OIDN Binary / DLL Path
        p_frame = ctk.CTkFrame(win, corner_radius=8)
        p_frame.pack(fill="x", padx=14, pady=(14, 8))
        ctk.CTkLabel(p_frame, text="Intel OIDN Binary / DLL Path", font=bold_f).pack(anchor="w", padx=12, pady=(8, 4))

        self.cfg_path_lbl = ctk.CTkLabel(p_frame, text=str(self.oidn_bin), font=small_f, text_color="#a0a0a0", wraplength=410, justify="left")
        self.cfg_path_lbl.pack(anchor="w", padx=12, pady=(0, 6))

        sel_btn = ctk.CTkButton(p_frame, text="Select oidnDenoise.exe / OpenImageDenoise.dll", height=30, command=self.select_binary)
        sel_btn.pack(fill="x", padx=12, pady=(0, 10))

        # 2. Clean Component Versions Grid (No repo clutter, inline status / release buttons)
        v_frame = ctk.CTkFrame(win, corner_radius=8)
        v_frame.pack(fill="x", padx=14, pady=6)
        ctk.CTkLabel(v_frame, text="Installed Versions", font=bold_f).pack(anchor="w", padx=12, pady=(8, 4))

        ver_grid = ctk.CTkFrame(v_frame, fg_color="transparent")
        ver_grid.pack(fill="x", padx=12, pady=(2, 8))
        ver_grid.grid_columnconfigure(0, weight=1)
        ver_grid.grid_columnconfigure(1, weight=0)
        ver_grid.grid_columnconfigure(2, weight=0)

        exr_ver = getattr(OpenEXR, "__version__", "Unknown")
        oidn_ver = self.get_oidn_version_str()

        rows_spec = [
            ("tool", APP_NAME, f"v{APP_VERSION}"),
            ("oidn", "Intel OIDN Engine", f"v{oidn_ver}"),
            ("exr", "OpenEXR Library", f"v{exr_ver}")
        ]
        self.ver_status_slots = {}
        self.ver_installed_labels = {}

        for r_idx, (key, label_txt, ver_txt) in enumerate(rows_spec):
            ctk.CTkLabel(ver_grid, text=label_txt, font=mono_f, anchor="w").grid(row=r_idx, column=0, sticky="w", pady=3)
            v_lbl = ctk.CTkLabel(ver_grid, text=ver_txt, font=mono_f, text_color="#cccccc", width=75, anchor="w")
            v_lbl.grid(row=r_idx, column=1, sticky="w", padx=(8, 8), pady=3)
            self.ver_installed_labels[key] = v_lbl

            slot_btn = ctk.CTkButton(
                ver_grid, text="—", width=135, height=24, font=small_f,
                fg_color="transparent", text_color="#777777", hover=False, state="disabled"
            )
            slot_btn.grid(row=r_idx, column=2, sticky="e", pady=3)
            self.ver_status_slots[key] = slot_btn

        self.chk_upd_btn = ctk.CTkButton(
            v_frame, text="Check for Updates", height=30,
            fg_color="#2b7a2b", hover_color="#1f5c1f", command=self._start_github_update_check
        )
        self.chk_upd_btn.pack(fill="x", padx=12, pady=(0, 10))

        # 3. Credits Section
        c_frame = ctk.CTkFrame(win, corner_radius=8, fg_color="#1b1b1b")
        c_frame.pack(fill="x", padx=14, pady=(8, 12))

        ctk.CTkLabel(c_frame, text="build by", font=ctk.CTkFont(size=11), text_color="#888888").pack(pady=(8, 0))
        ctk.CTkLabel(c_frame, text="Arthur Liebnau", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(0, 1))

        link_lbl = ctk.CTkLabel(
            c_frame, text="Polygon Manufaktur",
            font=ctk.CTkFont(size=13, weight="bold", underline=True),
            text_color="#4da3ff", cursor="hand2"
        )
        link_lbl.pack(pady=(0, 10))
        link_lbl.bind("<Button-1>", lambda _: webbrowser.open(POLYGON_WEB_URL))
        CTkToolTip(link_lbl, f"Open {POLYGON_WEB_URL}")

    @staticmethod
    def _parse_ver_tuple(v_str: str):
        m = re.search(r"(\d+(?:\.\d+)+)", str(v_str))
        if not m:
            return ()
        return tuple(int(x) for x in m.group(1).split("."))

    def _start_github_update_check(self):
        if hasattr(self, "chk_upd_btn") and self.chk_upd_btn.winfo_exists():
            self.chk_upd_btn.configure(state="disabled", text="Checking GitHub...")
        for slot in self.ver_status_slots.values():
            if slot.winfo_exists():
                slot.configure(text="Checking...", fg_color="transparent", text_color="#8ab4f8", hover=False, state="disabled")
        threading.Thread(target=self._run_github_update_check, daemon=True).start()

    def _run_github_update_check(self):
        """Rate-limit-free GitHub version check via web redirect resolution (avoids API HTTP 403)."""
        def check_repo_redirect(repo_slug: str):
            web_url = f"https://github.com/{repo_slug}/releases/latest"
            req = urllib.request.Request(web_url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=6) as resp:
                    final_url = resp.geturl()
                    if "/tag/" in final_url:
                        raw_tag = final_url.split("/tag/")[-1].split("?")[0].strip("/")
                        m = re.search(r"(\d+\.\d+(?:\.\d+)?)", raw_tag)
                        clean_ver = m.group(1) if m else raw_tag.lstrip("vV")
                        return "ok", clean_ver, final_url
                    return "unpublished", "", web_url
            except urllib.error.HTTPError as he:
                if he.code == 404:
                    return "unpublished", "", web_url
                return "error", f"HTTP {he.code}", web_url
            except Exception:
                return "error", "Offline", web_url

        cur_oidn = self.get_oidn_version_str()
        cur_exr = getattr(OpenEXR, "__version__", "Unknown")

        checks = {
            "tool": (APP_VERSION, check_repo_redirect(TOOL_GITHUB_REPO)),
            "oidn": (cur_oidn, check_repo_redirect("RenderKit/oidn")),
            "exr": (cur_exr, check_repo_redirect("AcademySoftwareFoundation/openexr"))
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
                    t_inst = self._parse_ver_tuple(installed_v)
                    t_lat = self._parse_ver_tuple(latest_v)
                    if t_inst and t_lat and t_lat > t_inst:
                        slot.configure(
                            text=f"Update: v{latest_v} ↗",
                            fg_color="#1f6aa5",
                            hover_color="#144870",
                            text_color="#ffffff",
                            hover=True,
                            state="normal",
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
            self.cfg_path_lbl.configure(text=str(self.oidn_bin))
            if "oidn" in self.ver_installed_labels:
                self.ver_installed_labels["oidn"].configure(text=f"v{self.get_oidn_version_str()}")

    def _update_scale_info_label(self):
        ow, oh = self.last_known_orig_dims
        tw, th = DownscaleEngine.compute_target_size(ow, oh, self.scale_mode_var.get(), self.scale_val_entry.get())
        pct = (tw / float(max(ow, 1))) * 100.0
        self.scale_out_label.configure(text=f"{tw}×{th} ({pct:.0f}%)")

    def on_scale_mode_select(self, mode: str):
        ow, oh = self.last_known_orig_dims
        if mode == "Scale %":
            self.scale_val_entry.configure(state="normal")
            self.scale_val_entry.delete(0, "end")
            self.scale_val_entry.insert(0, "50")
        elif mode == "Target Width (px)":
            self.scale_val_entry.configure(state="normal")
            self.scale_val_entry.delete(0, "end")
            self.scale_val_entry.insert(0, str(max(2, ow // 2)))
        elif mode == "Target Height (px)":
            self.scale_val_entry.configure(state="normal")
            self.scale_val_entry.delete(0, "end")
            self.scale_val_entry.insert(0, str(max(2, oh // 2)))
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
        self.linear_lru_cache.clear()
        self.reload_current_preview()

    def toggle_default_view_mode(self):
        self.show_denoised_default = not self.show_denoised_default
        if self.show_denoised_default:
            self.swap_default_btn.configure(text="Hold: Raw", fg_color="#2b7a2b")
        else:
            self.swap_default_btn.configure(text="Hold: Denoised", fg_color="#3b5998")
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

    def on_mousewheel(self, event):
        if event.state & 0x0004:
            levels = ["50%", "100%", "200%", "300%"]
            cur = self.zoom_var.get()
            idx = levels.index(cur) if cur in levels else 1
            new_idx = min(len(levels) - 1, idx + 1) if event.delta > 0 else max(0, idx - 1)
            if levels[new_idx] != cur:
                self.zoom_var.set(levels[new_idx])
                self.on_zoom_change()
        elif event.state & 0x0001:
            self.preview_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            self.preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_canvas_configure(self, _=None):
        if self.primary_rgb is not None:
            if self.zoom_var.get() == "Fit":
                self.render_preview_to_canvas()
            else:
                h, w, _ = self.primary_rgb.shape
                scale = float(self.zoom_var.get().replace("%", "")) / 100.0
                self._update_scrollbars_and_position(max(1, int(w * scale)), max(1, int(h * scale)))

    def _update_scrollbars_and_position(self, new_w: int, new_h: int):
        canv_w = max(self.preview_canvas.winfo_width(), 100)
        canv_h = max(self.preview_canvas.winfo_height(), 100)
        is_fit = self.zoom_var.get() == "Fit"
        need_h = (not is_fit) and (new_w > canv_w)
        need_v = (not is_fit) and (new_h > canv_h)

        if need_v:
            self.v_scroll.grid(row=0, column=1, sticky="ns")
        else:
            self.v_scroll.grid_remove()

        if need_h:
            self.h_scroll.grid(row=1, column=0, sticky="ew")
        else:
            self.h_scroll.grid_remove()

        x_off = 0 if need_h else max(0, (canv_w - new_w) // 2)
        y_off = 0 if need_v else max(0, (canv_h - new_h) // 2)
        if self.canvas_img_id is not None:
            self.preview_canvas.coords(self.canvas_img_id, x_off, y_off)
        self.preview_canvas.configure(scrollregion=(0, 0, max(new_w, canv_w), max(new_h, canv_h)))

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
        # OpenEXR v3.5 specification: tag non-color data passes with colorInteropID = "data"
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

        # Preserve native 32-bit unsigned integer (UINT) channels for discrete ID passes
        if rgb.dtype == np.uint32:
            target_dtype = np.uint32
            rgb_safe = rgb
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

    # ================= UNIVERSAL EXPORTER PASS PARSER =================
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
        self.render_queue.clear()
        self.aux_mask_files.clear()
        self.frames_data.clear()
        self.denoised_data.clear()
        self.empty_passes.clear()
        skipped_denoised, skipped_resumed, skipped_empty = 0, 0, 0
        skip_done = self.skip_existing_var.get()
        skip_empty = self.skip_empty_var.get()

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

        first_frame = sorted(list(self.frames_data.keys()))[0] if self.frames_data else None
        scan_all_frames = len(raw_light_candidates) <= 40

        for frame_id, pass_name, file_path in raw_light_candidates:
            if (scan_all_frames or frame_id == first_frame) and self.is_exr_black_fast(file_path):
                self.empty_passes.setdefault(frame_id, set()).add(pass_name)
                if skip_empty:
                    skipped_empty += 1
                    continue

            out_candidate = file_path.parent / f"{file_path.stem}_denoised.exr"
            if skip_done and out_candidate.exists() and out_candidate.stat().st_size > 2048:
                skipped_resumed += 1
                continue
            aux_dict = frame_aux_lookup.get(frame_id, {})
            self.render_queue.append({
                "target": file_path,
                "albedo": aux_dict.get("albedo"),
                "normal_bump": aux_dict.get("normals"),
                "normal_pre": aux_dict.get("normals_pre_bump"),
                "frame_id": frame_id,
                "pass_name": pass_name
            })

        self.log(
            f"Queue ready: {len(self.render_queue)} active light passes | "
            f"Skipped {skipped_empty} empty (black) | "
            f"{len(self.aux_mask_files)} Aux/Mask/ID | Paired {skipped_denoised} _denoised."
        )
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
            self._update_scale_info_label()
            is_empty = float(np.max(img_pos)) <= 1e-6
            if is_empty and ch_short not in self.blacklist:
                self.empty_passes.setdefault(self.frame_var.get(), set()).add(ch_short)
                self.channel_selector.refresh_label()
            tw, th = DownscaleEngine.compute_target_size(ow, oh, sc_mode, sc_val)
            if (tw, th) != (ow, oh):
                img_pos = DownscaleEngine.downscale_aux_pass(img_pos, tw, th, ch_short, ds_kernel) if ch_short in self.blacklist else DownscaleEngine.downscale(img_pos, tw, th, ds_kernel)
                tag = f"{ch_short} [{'EMPTY — ' if is_empty else ''}Raw ➔ {tw}×{th} {ds_kernel.split('(')[0].strip()}]"
            else:
                tag = f"{ch_short} [{'EMPTY — ' if is_empty else ''}Raw 100%]"

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
            try:
                raw_lin, raw_tag, den_lin, den_tag = self._build_beauty_sum_pair(frame_id)
            except Exception as e:
                self.log(f"Beauty sum warning: {e}")
                return
            if raw_lin is None:
                return
        else:
            raw_path = self.frames_data.get(frame_id, {}).get(channel_name)
            if not raw_path:
                return
            try:
                raw_lin, raw_tag = self._get_scaled_linear_rgb(raw_path, is_denoised=False, ch_label=channel_name)
            except Exception as e:
                self.log(f"Warning: Failed to read {raw_path.name} ({e}).")
                return
            den_path = self.denoised_data.get(frame_id, {}).get(channel_name)
            if den_path and den_path.exists():
                try:
                    den_lin, den_tag = self._get_scaled_linear_rgb(den_path, is_denoised=True, ch_label=channel_name)
                except Exception:
                    den_lin, den_tag = None, ""
            else:
                den_lin, den_tag = None, ""

        if self.show_denoised_default and den_lin is not None:
            self.primary_linear, self.primary_tag = den_lin, den_tag
            self.alternate_linear, self.alternate_tag = raw_lin, raw_tag
        else:
            self.primary_linear, self.primary_tag = raw_lin, raw_tag
            self.alternate_linear, self.alternate_tag = den_lin, den_tag

        self.primary_rgb = self._tonemap_and_dither_fast(self.primary_linear, ev)
        self.alternate_rgb = self._tonemap_and_dither_fast(self.alternate_linear, ev) if self.alternate_linear is not None else None
        self.render_preview_to_canvas()

    @staticmethod
    def _scale_pil(rgb_arr: np.ndarray, new_w: int, new_h: int, scale: float):
        pil_img = Image.fromarray(rgb_arr)
        if scale != 1.0:
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST if scale > 1.0 else Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

    def render_preview_to_canvas(self):
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
        self._update_scrollbars_and_position(new_w, new_h)

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
            self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(2, 8), pady=4)
            self.progress_label.grid(row=0, column=1, padx=(0, 8), pady=4)
            self.start_btn.configure(text="Cancel Batch (Running...)", fg_color="#8b2626", hover_color="#a83232", state="normal")
            self.test_single_btn.configure(state="disabled")
            self.folder_btn.configure(state="disabled")
        else:
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
        threading.Thread(target=self.process_items, args=(self.render_queue, False), daemon=True).start()

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

        self.write_exr_static(
            out_path, denoised_rgb, out_alpha, comp_attr_name, level_str,
            use_half_exr, worker_threads=exr_worker_threads, is_data_pass=False
        )
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
        # OpenEXR 3.5 API: Divide thread count across concurrent async workers to prevent oversubscription
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
                                den_packed = self.native_oidn.denoise_in_vram(
                                    packed_hdr, has_alb, has_nrm, filter_str, quality_str, use_half_oidn, not use_ldr, clean_aux, inplace, maxmem_mb
                                )
                                work_rgb = self.stride_unpack(den_packed, work_rgb, stride)
                                self.native_oidn.upload_aux_to_vram(alb_rgb_ram if has_alb else None, nrm_rgb_ram if has_nrm else None, use_half_oidn)

                            denoised_rgb = self.native_oidn.denoise_in_vram(
                                work_rgb, has_alb, has_nrm, filter_str, quality_str, use_half_oidn, not use_ldr, clean_aux, inplace, maxmem_mb
                            )
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

                        args_tup = (
                            idx, total, out_path, orig_rgb, denoised_rgb, alpha_ch,
                            ds_kernel, sc_mode, sc_val, comp_attr_name, level_str,
                            use_half_exr, t_denoise_elapsed, exr_worker_threads
                        )
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
                    for aux_info in self.aux_mask_files:
                        if self.cancel_event.is_set():
                            break
                        aux_path, p_type = aux_info["path"], aux_info["type"]
                        try:
                            t_aux_start = time.perf_counter()
                            is_id_map = p_type in ("material_id", "object_id")
                            a_rgb, a_alpha, _ = self.read_exr_native(aux_path, preserve_uint32=is_id_map)
                            if a_rgb.dtype != np.uint32:
                                a_clean = np.nan_to_num(a_rgb, nan=0.0, posinf=65500.0, neginf=-65500.0)
                            else:
                                a_clean = a_rgb
                            ah, aw, _ = a_clean.shape
                            tw, th = DownscaleEngine.compute_target_size(aw, ah, sc_mode, sc_val)
                            if (tw, th) == (aw, ah):
                                continue
                            a_down = DownscaleEngine.downscale_aux_pass(a_clean, tw, th, p_type, ds_kernel)
                            a_al_down = DownscaleEngine.downscale_aux_pass(np.dstack([a_alpha] * 3), tw, th, "alpha", ds_kernel)[:, :, 0] if a_alpha is not None else None
                            use_half_this_aux = False if p_type in ("depth", "position") else use_half_exr
                            is_non_color_data = p_type != "albedo"
                            suffix = "_50pct.exr" if (tw == aw // 2 and th == ah // 2) else "_scaled.exr"
                            out_aux = aux_path.parent / f"{aux_path.stem}{suffix}"
                            self.write_exr_static(
                                out_aux, a_down, a_al_down, aux_comp_attr, aux_comp_lvl,
                                use_half_this_aux, worker_threads=CPU_THREADS, is_data_pass=is_non_color_data
                            )
                            self.log(f" -> Aux [{p_type}] {tw}×{th} ({out_aux.stat().st_size / 1024.0:.1f} KB | {time.perf_counter() - t_aux_start:.1f}s): {out_aux.name}")
                        except Exception as ex:
                            self.log(f"Aux scale warning ({aux_path.name}): {ex}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        if not self.cancel_event.is_set():
            self.log(f"--- Run Finished in {time.perf_counter() - t_batch_start:.1f}s ---")
        self.linear_lru_cache.clear()
        self.after(0, lambda: self.parse_indigo_passes(select_channel=initial_channel_before_run))
        self.after(0, lambda: self._set_running_ui_state(False))


if __name__ == "__main__":
    app = PolyDenoiseStudio()
    app.mainloop()
# === END OF FILE ===