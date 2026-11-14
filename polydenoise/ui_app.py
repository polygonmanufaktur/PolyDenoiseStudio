# SPDX-License-Identifier: MIT
"""Main CustomTkinter application window and async GPU/OIDN workflow controller."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog
import urllib.error
import urllib.request
import weakref
import webbrowser

import customtkinter as ctk
import cv2
import numpy as np
import OpenEXR
from PIL import Image, ImageTk

from .constants import (
    APP_NAME,
    APP_VERSION,
    AUX_COMPRESSION_OPTIONS,
    BASE_DIR,
    BEAUTY_SUM_KEY,
    COMPRESSION_MAP,
    CONFIG_FILE,
    CPU_THREADS,
    INITIAL_OIDN_BIN,
    INSPECTOR_MODES,
    POLYGON_WEB_URL,
    SCALE_MODE_OPTIONS,
    TOOL_GITHUB_REPO,
    ZOOM_STEPS,
    register_dll_dir,
)
from .downscale_engine import DownscaleEngine, GPU_SHADER_ENGINE
from .exr_pipeline import (
    AUX_BLACKLIST,
    COMPONENT_AOV_PASSES,
    compute_raw_diagnostics_mask,
    extract_frame_and_channel,
    get_sum_light_channels,
    is_exr_black_fast,
    normalize_depth_guide,
    read_exr_native,
    write_exr_static,
    write_multilayer_exr,
)
from .filter_registry import (
    FILTER_SPECS,
    describe_filter,
    downscale_needs_guides,
    filter_supports_depth,
    get_filter_labels,
    get_filter_spec,
)
from .inspectors import (
    build_fp16_lut_key,
    clamp_roi_box,
    compute_ign_dither,
    probe_pixel_summary,
    render_viewport_rgb,
)
from .oidn_backend import (
    OIDNNativeDLL,
    build_oidn_cmd,
    estimate_noise_fast_roi,
    format_benchmark_test_name,
    read_pfm,
    sanitize_benchmark_log_line,
    stride_pack,
    stride_unpack,
    write_pfm,
)
from .ui_widgets import CTkToolTip, ScrollableDropdown, create_minimal_icon


def _downsample_diag_mask(mask: np.ndarray | None, tw: int, th: int) -> np.ndarray | None:
    """Block-max preserve NaN/Inf/Neg diagnostic flags when downscaling so 1px faults stay visible."""
    if mask is None:
        return None
    h, w = mask.shape[:2]
    if (w, h) == (tw, th):
        return mask
    if not np.any(mask):
        return np.zeros((th, tw), dtype=np.uint8)
    kx = max(1, int(np.ceil(w / max(1, tw))))
    ky = max(1, int(np.ceil(h / max(1, th))))
    dil = cv2.dilate(mask, np.ones((ky, kx), dtype=np.uint8))
    return cv2.resize(dil, (tw, th), interpolation=cv2.INTER_NEAREST)


class PolyDenoiseStudio(ctk.CTk):
    """Desktop GUI uniting Direct RAM/VRAM OIDN denoising, GPU GLSL downscaling, and EXR inspection."""

    BEAUTY_SUM_KEY = BEAUTY_SUM_KEY
    ZOOM_STEPS = ZOOM_STEPS
    blacklist = AUX_BLACKLIST
    COMPONENT_AOV_PASSES = COMPONENT_AOV_PASSES

    def __init__(self):
        super().__init__()
        exr_ver = getattr(OpenEXR, "__version__", "3.5+")
        self.title(f"{APP_NAME} v{APP_VERSION} — Direct RAM/VRAM AI Denoiser & GPU Shader Engine (OpenEXR v{exr_ver})")
        self.geometry("1440x940")
        self.minsize(1180, 720)
        self._ui_q = queue.SimpleQueue()
        self._alt_factory = None
        self._ui_ready = False
        self._icon_photo_ref = None
        self._apply_window_icon(self)

        self.icon_stopwatch = create_minimal_icon("stopwatch", size=17, color="#dcdcdc")
        self.icon_sliders = create_minimal_icon("sliders", size=17, color="#dcdcdc")

        self.oidn_bin = INITIAL_OIDN_BIN
        self.native_oidn: OIDNNativeDLL | None = None
        self._try_init_native_dll()

        self.compression_map = COMPRESSION_MAP
        self.aux_compression_options = AUX_COMPRESSION_OPTIONS
        self.downscale_options = get_filter_labels()
        self.scale_mode_options = SCALE_MODE_OPTIONS

        self.input_folder: Path | None = None
        self.render_queue: list[dict] = []
        self.aux_mask_files: list[dict] = []
        self.frames_data: dict[str, dict[str, Path]] = {}
        self.denoised_data: dict[str, dict[str, Path]] = {}
        self.empty_passes: dict[str, set[str]] = {}
        self.last_known_orig_dims = (4096, 3072)

        # Strictly bounded full-res RAM caches
        self.ram_raw_fullres: OrderedDict = OrderedDict()
        self.ram_diag_fullres: OrderedDict = OrderedDict()
        self.ram_denoised_fullres: OrderedDict = OrderedDict()
        self.ram_roi_denoised: dict[tuple[str, str], dict] = {}
        self.ram_guides_fullres: OrderedDict = OrderedDict()
        self.ram_beauty_sum_fullres: OrderedDict = OrderedDict()
        self.ram_is_empty_cache: dict[tuple[str, str], bool] = {}
        self._roi_bg_cache: dict[tuple, np.ndarray] = {}
        self.cache_lock = threading.Lock()

        self.primary_linear = self.alternate_linear = self.primary_rgb = self.alternate_rgb = None
        self.primary_diag = self.alternate_diag = None
        self.primary_albedo_scaled = None
        self.primary_tag = self.alternate_tag = ""
        self.tk_img_viewport = None
        self.canvas_img_id = None
        self._last_draw_geom = None  # (draw_x, draw_y, x0, y0, dst_w, dst_h, scale, orig_w, orig_h)
        self._fp16_key_ref = None
        self._fp16_key_u16 = None
        self._alternate_ev_dirty = False

        self.view_pan_x = 0.0
        self.view_pan_y = 0.0
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_pan_x = 0.0
        self._drag_pan_y = 0.0

        # ROI crop region stored normalized [x0, y0, x1, y1] relative to full-res image
        self.roi_norm: tuple[float, float, float, float] = (0.35, 0.35, 0.65, 0.65)
        self._roi_drawing = False
        self._roi_drag_start = (0.0, 0.0)
        self._roi_rubber_norm: tuple[float, float, float, float] | None = None

        self._dither_cache: dict[tuple[int, int], np.ndarray] = {}
        self.is_holding_compare = self.show_denoised_default = False
        self.config_window = self.bench_window = self.bench_process = None
        self.bench_running = self.bench_cancel_flag = False

        self._scan_generation = 0
        self._preview_req_id = 0
        self._slider_debounce_id = None
        self._ev_debounce_id = None
        self._resize_debounce_id = None
        self._sidebar_sb_after_id = None
        self._last_canvas_dims = (0, 0)
        self._v_scroll_shown: bool | None = None
        self._h_scroll_shown: bool | None = None
        self._sidebar_sb_visible: bool | None = None
        self._last_boost_txt = ""
        self._last_rad_txt = ""
        self._last_ev_txt = ""
        self._last_filter_ui_state = None
        self._spinner_active = False
        self._spinner_angle = 0
        self._spinner_text = ""
        self._spinner_after_id = None

        self.is_running = False
        self.cancel_event = threading.Event()
        self.active_process = None

        self.setup_ui()
        self._bind_hotkeys()
        self._load_ui_settings()
        self._ui_ready = True
        self._poll_ui_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    # Static delegates preserved for existing callers/tests
    read_exr_native = staticmethod(read_exr_native)
    is_exr_black_fast = staticmethod(is_exr_black_fast)
    write_exr_static = staticmethod(write_exr_static)
    normalize_depth_guide = staticmethod(normalize_depth_guide)
    write_pfm = staticmethod(write_pfm)
    read_pfm = staticmethod(read_pfm)
    estimate_noise_fast_roi = staticmethod(estimate_noise_fast_roi)
    stride_pack = staticmethod(stride_pack)
    stride_unpack = staticmethod(stride_unpack)
    _describe_filter = staticmethod(describe_filter)
    _filter_supports_depth = staticmethod(filter_supports_depth)

    @staticmethod
    def _downscale_needs_guides(mode_label: str, sharp_str: float) -> bool:
        return downscale_needs_guides(mode_label, sharp_str)

    def _bind_hotkeys(self):
        self.bind("<KeyPress-space>", self._on_space_press)
        self.bind("<KeyRelease-space>", self._on_space_release)
        self.bind("<KeyPress-m>", self._on_mask_key_press)
        self.bind("<KeyRelease-m>", self._on_mask_key_release)
        self.bind("<KeyPress-M>", self._on_mask_key_press)
        self.bind("<KeyRelease-M>", self._on_mask_key_release)

        for _k in ("f", "F"):
            self.bind(f"<KeyPress-{_k}>", self._on_fit_hotkey)

        for option_idx, spec in enumerate(FILTER_SPECS):
            if spec.key_sequence:
                self.bind(
                    spec.key_sequence,
                    lambda e, idx=option_idx, seq=spec.key_sequence: self._on_filter_hotkey(e, idx, seq),
                )

    def _on_fit_hotkey(self, event):
        if self._is_typing_in_entry(event) or self.primary_rgb is None:
            return
        self.zoom_var.set("Fit")
        self.on_zoom_change()

    def _is_typing_in_entry(self, event) -> bool:
        w = event.widget if event else self.focus_get()
        return isinstance(w, (tk.Entry, ctk.CTkEntry, tk.Text))

    def _on_filter_hotkey(self, event, option_idx: int, key_seq: str):
        if self._is_typing_in_entry(event):
            return
        state = getattr(event, "state", 0)
        has_ctrl = bool(state & 0x0004)
        alt_mask = 0x20000 if sys.platform == "win32" else (0x0008 | 0x20000)
        has_alt = bool(state & alt_mask)
        is_plain = not key_seq.startswith(("<Control-", "<Alt-"))
        if is_plain and (has_ctrl or has_alt):
            return
        if key_seq.startswith("<Control-") and (not has_ctrl or has_alt):
            return
        if key_seq.startswith("<Alt-") and (not has_alt or has_ctrl):
            return
        if 0 <= option_idx < len(self.downscale_options):
            chosen = self.downscale_options[option_idx]
            if self.downscale_var.get() != chosen:
                self.downscale_var.set(chosen)
                self.on_downscale_mode_change()

    def _on_mask_key_press(self, event):
        if self._is_typing_in_entry(event) or GPU_SHADER_ENGINE.show_mask_xray:
            return
        GPU_SHADER_ENGINE.show_mask_xray = True
        self.reload_current_preview()

    def _on_mask_key_release(self, _=None):
        if GPU_SHADER_ENGINE.show_mask_xray:
            GPU_SHADER_ENGINE.show_mask_xray = False
            self.reload_current_preview()

    def _purge_all_caches(self, log_msg: bool = False):
        """Completely free all RAM arrays and persistent OpenGL/OIDN GPU VRAM buffers."""
        with self.cache_lock:
            self.ram_raw_fullres.clear()
            self.ram_diag_fullres.clear()
            self.ram_denoised_fullres.clear()
            self.ram_roi_denoised.clear()
            self.ram_guides_fullres.clear()
            self.ram_beauty_sum_fullres.clear()
            self.ram_is_empty_cache.clear()
            self._roi_bg_cache.clear()
            self._dither_cache.clear()
            self._fp16_key_ref = None
            self._fp16_key_u16 = None
        GPU_SHADER_ENGINE.release_vram_pool()
        if self.native_oidn is not None:
            self.native_oidn.release_buffers()
        gc.collect()
        if hasattr(self, "channel_selector"):
            self.channel_selector.refresh_label()
        if log_msg:
            self.log("🧹 Purged all 100% RAM preview buffers and released GPU VRAM textures.")
            self.reload_current_preview()

    def _invalidate_beauty_sum(self, frame_id: str | None = None):
        with self.cache_lock:
            if frame_id is None:
                self.ram_beauty_sum_fullres.clear()
                self._roi_bg_cache.clear()
            else:
                self.ram_beauty_sum_fullres.pop(frame_id, None)
                for k in [k for k in self._roi_bg_cache if k[0] == frame_id]:
                    self._roi_bg_cache.pop(k, None)

    def _get_sum_light_channels(self, frame_id: str) -> list[str]:
        return get_sum_light_channels(self.frames_data.get(frame_id, {}), self.empty_passes.get(frame_id, set()))

    def _on_closing(self):
        self._scan_generation += 1
        self._preview_req_id += 1
        self.cancel_event.set()
        self.bench_cancel_flag = True
        if self.bench_process and self.bench_process.poll() is None:
            try:
                self.bench_process.terminate()
            except Exception:
                pass
        self.save_config()
        GPU_SHADER_ENGINE.release_vram_pool()
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
        for dll_p in (
            BASE_DIR / "oidn" / "OpenImageDenoise.dll",
            BASE_DIR / "OpenImageDenoise.dll",
            BASE_DIR / "_internal" / "OpenImageDenoise.dll",
            self.oidn_bin.parent / "OpenImageDenoise.dll",
        ):
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

    def _ui_setting_vars(self) -> dict:
        return {
            "downscale": self.downscale_var,
            "boost": self.boost_var,
            "radius": self.rad_var,
            "depth_guide": self.depth_guide_var,
            "scale_mode": self.scale_mode_var,
            "comp": self.comp_var,
            "exr_depth": self.exr_depth_var,
            "sep_aux": self.sep_aux_comp_var,
            "aux_comp": self.aux_comp_var,
            "multilayer_exr": self.multilayer_exr_var,
            "aux_mode": self.aux_mode_var,
            "decorr": self.decorr_var,
            "device": self.device_var,
            "quality": self.quality_var,
            "precision": self.precision_var,
            "hdr_mode": self.hdr_mode_var,
            "filter": self.filter_var,
            "verbose": self.verbose_var,
            "clean_aux": self.clean_aux_var,
            "inplace": self.inplace_var,
            "scale_aux": self.downscale_aux_var,
            "async_save": self.async_save_var,
            "skip_empty": self.skip_empty_var,
            "skip_existing": self.skip_existing_var,
            "roi_crop": self.roi_crop_var,
            "inspector": self.inspector_var,
        }

    def _ui_setting_entries(self) -> dict:
        return {"scale_val": self.scale_val_entry, "comp_level": self.comp_level_entry, "maxmem": self.maxmem_entry}

    def save_config(self):
        try:
            data = {}
            if CONFIG_FILE.exists():
                try:
                    data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data["oidn_bin"] = str(self.oidn_bin)
            if self._ui_ready:
                ui = {k: v.get() for k, v in self._ui_setting_vars().items()}
                ui.update({k: e.get() for k, e in self._ui_setting_entries().items()})
                ui["roi_norm"] = list(self.roi_norm)
                data["ui"] = ui
            CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _load_ui_settings(self):
        try:
            ui = json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("ui", {}) if CONFIG_FILE.exists() else {}
        except Exception:
            ui = {}
        for k, var in self._ui_setting_vars().items():
            if k in ui:
                try:
                    if k == "downscale" and ui[k] not in self.downscale_options:
                        var.set(get_filter_spec(ui[k]).label)
                        continue
                    if k == "inspector" and ui[k] not in INSPECTOR_MODES:
                        continue
                    var.set(ui[k])
                except Exception:
                    pass
        if isinstance(ui.get("roi_norm"), (list, tuple)) and len(ui["roi_norm"]) == 4:
            try:
                rn = [float(x) for x in ui["roi_norm"]]
                if 0.0 <= rn[0] < rn[2] <= 1.0 and 0.0 <= rn[1] < rn[3] <= 1.0:
                    self.roi_norm = (rn[0], rn[1], rn[2], rn[3])
            except Exception:
                pass
        for k, ent in self._ui_setting_entries().items():
            if k in ui and isinstance(ui[k], str):
                ent.delete(0, "end")
                ent.insert(0, ui[k])
        self._last_boost_txt = f"+{int(round(float(self.boost_var.get())))}%"
        self.boost_val_lbl.configure(text=self._last_boost_txt)
        self._last_rad_txt = f"{float(self.rad_var.get()):.1f}px"
        self.rad_val_lbl.configure(text=self._last_rad_txt)
        self._last_ev_txt = f"{float(self.ev_var.get()):+.1f}"
        self.ev_reset_btn.configure(text=self._last_ev_txt)
        self.scale_val_entry.configure(state="disabled" if self.scale_mode_var.get() == "No Downscale (100%)" else "normal")
        self._on_toggle_aux_codec()
        self._refresh_filter_controls()
        self._update_roi_badge()
        self._update_scale_info_label()

    def _format_channel_label(self, ch_name: str) -> str:
        if ch_name == self.BEAUTY_SUM_KEY:
            return ch_name
        frame = self.frame_var.get()
        if frame in self.empty_passes and ch_name in self.empty_passes[frame]:
            return f"⚫ {ch_name} [Empty]"
        if (frame, ch_name) in self.ram_denoised_fullres:
            return f"⚡ {ch_name} [RAM]"
        if (frame, ch_name) in self.ram_roi_denoised:
            return f"✂ {ch_name} [ROI]"
        if frame in self.denoised_data and ch_name in self.denoised_data[frame]:
            return f"✓ {ch_name}"
        return f"  {ch_name}"

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
        cx, cy = 16, 16
        box_w, box_h = max(210, len(self._spinner_text) * 7 + 52), 34
        self.preview_canvas.create_rectangle(
            cx, cy, cx + box_w, cy + box_h, fill="#141414", outline="#3b8ed0", width=1, tags="spinner_overlay"
        )
        ax0, ay0, ax1, ay1 = cx + 9, cy + 7, cx + 29, cy + 27
        self.preview_canvas.create_oval(ax0, ay0, ax1, ay1, outline="#2c3e50", width=3, tags="spinner_overlay")
        self.preview_canvas.create_arc(
            ax0, ay0, ax1, ay1, start=self._spinner_angle, extent=110, style="arc", outline="#4da3ff", width=3, tags="spinner_overlay"
        )
        self.preview_canvas.create_text(
            cx + 38, cy + 17, text=self._spinner_text, fill="#e8e8e8", anchor="w", font=("Segoe UI", 9, "bold"), tags="spinner_overlay"
        )

    def _schedule_sidebar_scrollbar_sync(self, _=None):
        if getattr(self, "_sidebar_sb_after_id", None) is not None:
            return
        try:
            self._sidebar_sb_after_id = self.after(40, self._sync_sidebar_scrollbar)
        except Exception:
            pass

    def _sync_sidebar_scrollbar(self):
        self._sidebar_sb_after_id = None
        if not hasattr(self, "sidebar"):
            return
        sb = getattr(self.sidebar, "_scrollbar", None)
        canvas = getattr(self.sidebar, "_parent_canvas", None)
        if sb is None or canvas is None:
            return
        try:
            content_h = self.sidebar.winfo_reqheight()
            avail_h = canvas.winfo_height()
        except Exception:
            return
        if avail_h <= 20:
            return
        need_sb = content_h > avail_h + 2
        if need_sb != self._sidebar_sb_visible:
            self._sidebar_sb_visible = need_sb
            if need_sb:
                sb.grid()
            else:
                sb.grid_remove()
                try:
                    canvas.yview_moveto(0.0)
                except Exception:
                    pass

    def _section_header(self, text: str, first: bool = False):
        row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=(2 if first else 6, 2))
        ctk.CTkLabel(row, text=text, font=ctk.CTkFont(size=13, weight="bold"), text_color="#8ab4f8").pack(side="left")
        ctk.CTkFrame(row, height=1, fg_color="#3a3a3a").pack(side="left", fill="x", expand=True, padx=(8, 0), pady=(3, 0))

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_outer = ctk.CTkFrame(self, width=396)
        self.sidebar_outer.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_outer.pack_propagate(False)
        self.sidebar_footer = ctk.CTkFrame(self.sidebar_outer, fg_color="transparent")
        self.sidebar_footer.pack(side="bottom", fill="x", pady=(2, 4))
        self.sidebar = ctk.CTkScrollableFrame(self.sidebar_outer, fg_color="transparent", width=350)
        self.sidebar.pack(side="top", fill="both", expand=True, padx=(2, 0), pady=(2, 0))
        self.sidebar.bind("<Configure>", self._schedule_sidebar_scrollbar_sync, add="+")
        if hasattr(self.sidebar, "_parent_canvas"):
            self.sidebar._parent_canvas.bind("<Configure>", self._schedule_sidebar_scrollbar_sync, add="+")

        lbl_font = ctk.CTkFont(size=11)

        self._section_header("1. Source Folder & Settings", first=True)
        p_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        p_row.pack(fill="x", padx=8, pady=1)

        self.folder_btn = ctk.CTkButton(p_row, text="Select Render Folder", height=28, command=self.select_folder)
        self.folder_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        CTkToolTip(self.folder_btn, "Select folder with EXR render passes.\nPurges previous RAM & VRAM caches automatically.")

        self.cfg_btn = ctk.CTkButton(
            p_row, text="", image=self.icon_sliders, width=32, height=28,
            fg_color="#2b2b2b", hover_color="#3f3f3f", corner_radius=6, command=self.open_config_popup,
        )
        self.cfg_btn.pack(side="right")
        CTkToolTip(self.cfg_btn, "Configuration, OIDN Path, Version Check & Credits")

        self.bench_open_btn = ctk.CTkButton(
            p_row, text="", image=self.icon_stopwatch, width=32, height=28,
            fg_color="#2b2b2b", hover_color="#3f3f3f", corner_radius=6, command=self.open_benchmark_popup,
        )
        self.bench_open_btn.pack(side="right", padx=(0, 4))
        CTkToolTip(self.bench_open_btn, "OIDN Hardware Benchmark (GPU vs. CPU)")

        self.folder_label = ctk.CTkLabel(
            self.sidebar, text=f"OIDN v{self.get_oidn_version_str()} | Folder: None",
            font=lbl_font, text_color="gray", wraplength=320, justify="left",
        )
        self.folder_label.pack(padx=10, pady=(1, 2), anchor="w")

        self._section_header("2. GPU Shader Downscaler & OpenEXR")
        ctk.CTkLabel(self.sidebar, text="Filter Kernel (1–0 / Ctrl+1–7 | Hold M = Mask | F = Fit)", font=lbl_font).pack(anchor="w", padx=10)
        self.downscale_var = ctk.StringVar(value=self.downscale_options[0])
        self.downscale_menu = ctk.CTkOptionMenu(
            self.sidebar, variable=self.downscale_var, values=self.downscale_options,
            height=25, dynamic_resizing=False, command=self.on_downscale_mode_change,
        )
        self.downscale_menu.pack(fill="x", padx=8, pady=(0, 3))
        self.downscale_tip = CTkToolTip(self.downscale_menu, self._describe_filter(self.downscale_var.get()))

        sr_box = ctk.CTkFrame(self.sidebar, fg_color="#222222", corner_radius=6)
        sr_box.pack(fill="x", padx=8, pady=(0, 3))

        sh_row = ctk.CTkFrame(sr_box, fg_color="transparent")
        sh_row.pack(fill="x", padx=6, pady=(3, 1))
        ctk.CTkLabel(sh_row, text="Boost:", font=lbl_font, width=50, anchor="w").pack(side="left")
        self.boost_var = ctk.DoubleVar(value=0.0)
        self.boost_slider = ctk.CTkSlider(sh_row, from_=0.0, to=200.0, number_of_steps=200, variable=self.boost_var, height=14, command=self.on_boost_slider_move)
        self.boost_slider.pack(side="left", fill="x", expand=True, padx=4)
        self.boost_val_lbl = ctk.CTkLabel(sh_row, text="+0%", font=lbl_font, width=42, anchor="e", text_color="#8ab4f8")
        self.boost_val_lbl.pack(side="left", padx=(0, 3))
        self.boost_reset_btn = ctk.CTkButton(sh_row, text="↺", width=24, height=20, fg_color="#3a3a3a", hover_color="#525252", command=self.reset_boost)
        self.boost_reset_btn.pack(side="right")
        CTkToolTip(self.boost_slider, "Sharpening Boost (+0% leftmost = 100% native filter sharpness, +200% = 300%). Hold 'M' to inspect mask!")
        CTkToolTip(self.boost_reset_btn, "Reset Boost to +0%")

        rad_row = ctk.CTkFrame(sr_box, fg_color="transparent")
        rad_row.pack(fill="x", padx=6, pady=(1, 2))
        ctk.CTkLabel(rad_row, text="Radius:", font=lbl_font, width=50, anchor="w").pack(side="left")
        self.rad_var = ctk.DoubleVar(value=1.0)
        self.rad_slider = ctk.CTkSlider(rad_row, from_=0.4, to=2.5, number_of_steps=42, variable=self.rad_var, height=14, command=self.on_radius_slider_move)
        self.rad_slider.pack(side="left", fill="x", expand=True, padx=4)
        self.rad_val_lbl = ctk.CTkLabel(rad_row, text="1.0px", font=lbl_font, width=42, anchor="e", text_color="#8ab4f8")
        self.rad_val_lbl.pack(side="left", padx=(0, 3))
        self.rad_reset_btn = ctk.CTkButton(rad_row, text="↺", width=24, height=20, fg_color="#3a3a3a", hover_color="#525252", command=self.reset_radius)
        self.rad_reset_btn.pack(side="right")
        CTkToolTip(self.rad_slider, "MKS/CAS sampling radius (0.4px = micro-pores, 1.0px = native MKS, 2.5px = wide clarity).")
        CTkToolTip(self.rad_reset_btn, "Reset Radius to 1.0px")

        gbuf_cb_row = ctk.CTkFrame(sr_box, fg_color="transparent")
        gbuf_cb_row.pack(fill="x", padx=6, pady=(0, 4))
        self.depth_guide_var = ctk.BooleanVar(value=True)
        self.depth_guide_cb = ctk.CTkCheckBox(
            gbuf_cb_row, text="Depth Guide", variable=self.depth_guide_var, font=lbl_font,
            checkbox_width=18, checkbox_height=18, command=self.on_downscale_mode_change,
        )
        self.depth_guide_cb.pack(side="left", padx=(2, 10))
        self.depth_guide_tip = CTkToolTip(
            self.depth_guide_cb,
            "Uses log-compressed Z-Depth (second-difference Laplacian) to guard silhouette edges, and feeds Guided Downscale.",
        )

        scale_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        scale_row.pack(fill="x", padx=8, pady=(0, 3))
        s_col1 = ctk.CTkFrame(scale_row, fg_color="transparent")
        s_col1.pack(side="left", padx=(0, 6))
        ctk.CTkLabel(s_col1, text="Scale Mode", font=lbl_font).pack(anchor="w")
        self.scale_mode_var = ctk.StringVar(value="Scale %")
        self.scale_mode_menu = ctk.CTkOptionMenu(
            s_col1, variable=self.scale_mode_var, values=self.scale_mode_options,
            width=142, height=24, dynamic_resizing=False, command=self.on_scale_mode_select,
        )
        self.scale_mode_menu.pack()

        s_col2 = ctk.CTkFrame(scale_row, fg_color="transparent")
        s_col2.pack(side="left", padx=(0, 14))
        ctk.CTkLabel(s_col2, text="Value", font=lbl_font).pack(anchor="w")
        self.scale_val_entry = ctk.CTkEntry(s_col2, width=64, height=24)
        self.scale_val_entry.insert(0, "50")
        self.scale_val_entry.pack()
        self.scale_val_entry.bind("<Return>", lambda _: (self.focus_set(), self.on_downscale_mode_change()))
        self.scale_val_entry.bind("<FocusOut>", lambda _: self.on_downscale_mode_change())

        s_col3 = ctk.CTkFrame(scale_row, fg_color="transparent")
        s_col3.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(s_col3, text="Output Size", font=lbl_font).pack(anchor="w")
        self.scale_out_label = ctk.CTkLabel(s_col3, text="50.0%", font=lbl_font, text_color="#8ab4f8", height=24, anchor="w")
        self.scale_out_label.pack(anchor="w")

        exr_grid = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        exr_grid.pack(fill="x", padx=8, pady=(0, 3))
        c_col1 = ctk.CTkFrame(exr_grid, fg_color="transparent")
        c_col1.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkLabel(c_col1, text="Light EXR Codec", font=lbl_font).pack(anchor="w")
        self.comp_var = ctk.StringVar(value="DWAB (Lossy 256-Line)")
        self.comp_menu = ctk.CTkOptionMenu(
            c_col1, variable=self.comp_var, values=list(self.compression_map.keys()),
            width=155, height=24, dynamic_resizing=False, command=self.on_comp_change,
        )
        self.comp_menu.pack(fill="x")

        c_col2 = ctk.CTkFrame(exr_grid, fg_color="transparent")
        c_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(c_col2, text="Level", font=lbl_font).pack(anchor="w")
        self.comp_level_entry = ctk.CTkEntry(c_col2, width=52, height=24)
        self.comp_level_entry.insert(0, "20")
        self.comp_level_entry.pack()
        self.comp_level_entry.bind("<Return>", lambda _: self.focus_set())

        c_col3 = ctk.CTkFrame(exr_grid, fg_color="transparent")
        c_col3.pack(side="right")
        ctk.CTkLabel(c_col3, text="Bit Depth", font=lbl_font).pack(anchor="w")
        self.exr_depth_var = ctk.StringVar(value="HALF (16-bit)")
        self.exr_depth_menu = ctk.CTkOptionMenu(
            c_col3, variable=self.exr_depth_var, values=["HALF (16-bit)", "FLOAT (32-bit)"],
            width=125, height=24, dynamic_resizing=False,
        )
        self.exr_depth_menu.pack()

        aux_codec_row = ctk.CTkFrame(self.sidebar, fg_color="#222222", corner_radius=6)
        aux_codec_row.pack(fill="x", padx=8, pady=(0, 3))
        aux_top = ctk.CTkFrame(aux_codec_row, fg_color="transparent")
        aux_top.pack(fill="x", padx=6, pady=(3, 1))
        self.sep_aux_comp_var = ctk.BooleanVar(value=True)
        self.sep_aux_cb = ctk.CTkCheckBox(
            aux_top, text="Separate Aux/Mask Codec:", variable=self.sep_aux_comp_var,
            font=lbl_font, checkbox_width=18, checkbox_height=18, command=self._on_toggle_aux_codec,
        )
        self.sep_aux_cb.pack(side="left")

        self.aux_comp_var = ctk.StringVar(value="ZIP (Lossless 16-Line)")
        self.aux_comp_menu = ctk.CTkOptionMenu(
            aux_top, variable=self.aux_comp_var, values=self.aux_compression_options,
            width=165, height=22, dynamic_resizing=False,
        )
        self.aux_comp_menu.pack(side="right")

        self.multilayer_exr_var = ctk.BooleanVar(value=False)
        ml_cb = ctk.CTkCheckBox(
            aux_codec_row, text="Multi-Layer EXR Export (per frame *_multilayer.exr)",
            variable=self.multilayer_exr_var, font=lbl_font, checkbox_width=18, checkbox_height=18,
        )
        ml_cb.pack(anchor="w", padx=6, pady=(1, 4))
        CTkToolTip(
            ml_cb,
            "Exports one consolidated Nuke/Resolve-compatible <frame>_multilayer.exr per frame containing Beauty RGB + "
            "all denoised light passes (.R/G/B) + target-resolution Albedo, Normals, Depth (.Z) and Alpha.",
        )

        self._section_header("3. OIDN & Guide Parameters")
        row_aux = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row_aux.pack(fill="x", padx=8, pady=(0, 2))
        a_col1 = ctk.CTkFrame(row_aux, fg_color="transparent")
        a_col1.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkLabel(a_col1, text="Aux Guide Passes", font=lbl_font).pack(anchor="w")
        self.aux_mode_var = ctk.StringVar(value="Albedo + Normals (Bump)")
        self.aux_mode_menu = ctk.CTkOptionMenu(
            a_col1, variable=self.aux_mode_var, width=170, height=24, dynamic_resizing=False,
            values=["Albedo + Normals (Bump)", "Albedo + Normals (Pre-Bump)", "Albedo Only", "None (Color Only)"],
        )
        self.aux_mode_menu.pack(fill="x")

        a_col2 = ctk.CTkFrame(row_aux, fg_color="transparent")
        a_col2.pack(side="right", fill="x", expand=True)
        ctk.CTkLabel(a_col2, text="Noise De-Correlation", font=lbl_font).pack(anchor="w")
        self.decorr_var = ctk.StringVar(value="Standard 1x (1px Noise)")
        self.decorr_menu = ctk.CTkOptionMenu(
            a_col2, variable=self.decorr_var, width=165, height=24, dynamic_resizing=False,
            values=["Standard 1x (1px Noise)", "2x2 Stride + 1x Polish", "3x3 Stride + 1x Polish"],
        )
        self.decorr_menu.pack(fill="x")

        row_dqp = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row_dqp.pack(fill="x", padx=8, pady=(0, 2))
        d_col1 = ctk.CTkFrame(row_dqp, fg_color="transparent")
        d_col1.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(d_col1, text="Device (-d)", font=lbl_font).pack(anchor="w")
        self.device_var = ctk.StringVar(value="cuda")
        ctk.CTkOptionMenu(d_col1, variable=self.device_var, values=["cuda", "cpu", "default", "sycl", "hip", "metal"], width=105, height=24).pack()

        d_col2 = ctk.CTkFrame(row_dqp, fg_color="transparent")
        d_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(d_col2, text="Quality (-q)", font=lbl_font).pack(anchor="w")
        self.quality_var = ctk.StringVar(value="high")
        ctk.CTkOptionMenu(d_col2, variable=self.quality_var, values=["high", "balanced", "fast"], width=115, height=24).pack()

        d_col3 = ctk.CTkFrame(row_dqp, fg_color="transparent")
        d_col3.pack(side="right")
        ctk.CTkLabel(d_col3, text="Precision (-t)", font=lbl_font).pack(anchor="w")
        self.precision_var = ctk.StringVar(value="half")
        ctk.CTkOptionMenu(d_col3, variable=self.precision_var, values=["half", "float"], width=110, height=24).pack()

        row_hfv = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        row_hfv.pack(fill="x", padx=8, pady=(0, 2))
        h_col1 = ctk.CTkFrame(row_hfv, fg_color="transparent")
        h_col1.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(h_col1, text="HDR Mode", font=lbl_font).pack(anchor="w")
        self.hdr_mode_var = ctk.StringVar(value="HDR Linear")
        ctk.CTkOptionMenu(h_col1, variable=self.hdr_mode_var, values=["HDR Linear", "HDR Clamp Fireflies", "LDR Normalized"], width=150, height=24, dynamic_resizing=False).pack()

        h_col2 = ctk.CTkFrame(row_hfv, fg_color="transparent")
        h_col2.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(h_col2, text="Filter (-f)", font=lbl_font).pack(anchor="w")
        self.filter_var = ctk.StringVar(value="RT")
        ctk.CTkOptionMenu(h_col2, variable=self.filter_var, values=["RT", "RTLightmap"], width=105, height=24).pack()

        h_col3 = ctk.CTkFrame(row_hfv, fg_color="transparent")
        h_col3.pack(side="right")
        ctk.CTkLabel(h_col3, text="Verbose", font=lbl_font).pack(anchor="w")
        self.verbose_var = ctk.StringVar(value="2")
        ctk.CTkOptionMenu(h_col3, variable=self.verbose_var, values=["0", "1", "2", "3", "4"], width=75, height=24).pack()

        cb_grid = ctk.CTkFrame(self.sidebar, fg_color="#222222", corner_radius=6)
        cb_grid.pack(fill="x", padx=8, pady=(2, 3))
        cb_grid.grid_columnconfigure(0, weight=1)
        cb_grid.grid_columnconfigure(1, weight=1)

        self.clean_aux_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(cb_grid, text="CleanAux (--clean_aux)", variable=self.clean_aux_var, font=lbl_font, checkbox_width=18, checkbox_height=18).grid(row=0, column=0, sticky="w", padx=8, pady=(3, 1))
        self.inplace_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(cb_grid, text="Inplace VRAM", variable=self.inplace_var, font=lbl_font, checkbox_width=18, checkbox_height=18).grid(row=0, column=1, sticky="w", padx=8, pady=(3, 1))
        self.downscale_aux_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(cb_grid, text="Scale Aux & Masks", variable=self.downscale_aux_var, font=lbl_font, checkbox_width=18, checkbox_height=18).grid(row=1, column=0, sticky="w", padx=8, pady=1)
        self.async_save_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(cb_grid, text="Async RAM Save", variable=self.async_save_var, font=lbl_font, checkbox_width=18, checkbox_height=18).grid(row=1, column=1, sticky="w", padx=8, pady=1)
        self.skip_empty_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(cb_grid, text="Skip Empty (Black)", variable=self.skip_empty_var, font=lbl_font, checkbox_width=18, checkbox_height=18, command=lambda: self.parse_indigo_passes() if self.input_folder else None).grid(row=2, column=0, sticky="w", padx=8, pady=(1, 3))
        self.skip_existing_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(cb_grid, text="Skip Done (Resume)", variable=self.skip_existing_var, font=lbl_font, checkbox_width=18, checkbox_height=18, command=lambda: self.parse_indigo_passes() if self.input_folder else None).grid(row=2, column=1, sticky="w", padx=8, pady=(1, 3))

        extra_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        extra_frame.pack(fill="x", padx=8, pady=(0, 3))
        self.maxmem_entry = ctk.CTkEntry(extra_frame, placeholder_text="MaxMem MB", width=110, height=24)
        self.maxmem_entry.pack(side="left", padx=(0, 5))
        self.future_args = ctk.CTkEntry(extra_frame, placeholder_text="Custom CLI Args (e.g. OIDN v3)", height=24)
        self.future_args.pack(side="right", fill="x", expand=True)

        prev_row = ctk.CTkFrame(self.sidebar_footer, fg_color="transparent")
        prev_row.pack(fill="x", padx=8, pady=(1, 3))
        self.test_single_btn = ctk.CTkButton(
            prev_row, text="⚡ Preview Current Pass (RAM)", command=self.start_ram_preview_thread,
            fg_color="#1f6aa5", hover_color="#144870", height=32, state="disabled",
        )
        self.test_single_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        CTkToolTip(self.test_single_btn, "Denoises current pass (or ROI crop when active!) into RAM for instant filter testing.")

        self.free_ram_btn = ctk.CTkButton(
            prev_row, text="🗑", width=34, height=32, fg_color="#3a3a3a", hover_color="#8b2626",
            command=lambda: self._purge_all_caches(log_msg=True),
        )
        self.free_ram_btn.pack(side="right")
        CTkToolTip(self.free_ram_btn, "Free all cached RAM preview buffers and release GPU VRAM textures.")

        self.start_btn = ctk.CTkButton(
            self.sidebar_footer, text="Start Full Batch Denoise", command=self.on_start_or_cancel_click,
            fg_color="#2b7a2b", hover_color="#1f5c1f", height=36, state="disabled",
        )
        self.start_btn.pack(fill="x", padx=8, pady=(1, 6))

        # ================= MAIN AREA =================
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.main_frame.grid_rowconfigure(1, weight=5)
        self.main_frame.grid_rowconfigure(3, weight=2)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.top_bar = ctk.CTkFrame(self.main_frame)
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))

        tb_row1 = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        tb_row1.pack(fill="x", padx=6, pady=(5, 2))
        ctk.CTkLabel(tb_row1, text="Frame:").pack(side="left", padx=(2, 3))
        self.frame_var = ctk.StringVar(value="-")
        self.frame_selector = ScrollableDropdown(tb_row1, variable=self.frame_var, values=["-"], command=self.on_frame_change, width=125)
        self.frame_selector.pack(side="left", padx=(0, 6))

        ctk.CTkLabel(tb_row1, text="Channel:").pack(side="left", padx=(2, 3))
        self.channel_var = ctk.StringVar(value="-")
        self.channel_selector = ScrollableDropdown(
            tb_row1, variable=self.channel_var, values=["-"], command=self.on_channel_change,
            width=180, display_formatter=self._format_channel_label,
        )
        self.channel_selector.pack(side="left", padx=(0, 6))

        self.swap_default_btn = ctk.CTkButton(
            tb_row1, text="Hold: Denoised", width=105, height=28, fg_color="#3b5998", hover_color="#2d4373",
            command=self.toggle_default_view_mode,
        )
        self.swap_default_btn.pack(side="left", padx=(0, 6))

        ctk.CTkLabel(tb_row1, text="Zoom:").pack(side="left", padx=(2, 2))
        self.zoom_var = ctk.StringVar(value="100%")
        ctk.CTkOptionMenu(
            tb_row1, variable=self.zoom_var, values=["Fit"] + [f"{z}%" for z in self.ZOOM_STEPS],
            width=74, height=28, dynamic_resizing=False, command=self.on_zoom_change,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkLabel(tb_row1, text="EV:").pack(side="left", padx=(2, 2))
        self.ev_var = ctk.DoubleVar(value=0.0)
        ctk.CTkSlider(tb_row1, from_=-5.0, to=5.0, number_of_steps=100, variable=self.ev_var, width=82, command=self.on_ev_change).pack(side="left", padx=(0, 3))
        self.ev_reset_btn = ctk.CTkButton(tb_row1, text="0.0", width=36, height=24, fg_color="#3a3a3a", command=self.reset_ev)
        self.ev_reset_btn.pack(side="left")

        tb_row2 = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        tb_row2.pack(fill="x", padx=6, pady=(2, 5))

        ctk.CTkLabel(tb_row2, text="Inspector:", font=lbl_font).pack(side="left", padx=(2, 4))
        self.inspector_var = ctk.StringVar(value=INSPECTOR_MODES[0])
        self.inspector_menu = ctk.CTkOptionMenu(
            tb_row2, variable=self.inspector_var, values=INSPECTOR_MODES,
            width=195, height=24, dynamic_resizing=False, command=self.on_inspector_change,
        )
        self.inspector_menu.pack(side="left", padx=(0, 12))
        CTkToolTip(
            self.inspector_menu,
            "False-Color (EV/IRE centered at 18% gray = green), Clipping/Fireflies (>1.0 Amber, >4.0 Red, Spikes Magenta), "
            "or NaN/Inf (Magenta), Negatives (Cyan) & Albedo=0 (Yellow).",
        )

        self.roi_crop_var = ctk.BooleanVar(value=False)
        self.roi_crop_cb = ctk.CTkCheckBox(
            tb_row2, text="✂ ROI Crop Preview (Right-Drag or Shift+Drag)",
            variable=self.roi_crop_var, font=lbl_font, checkbox_width=18, checkbox_height=18,
            command=self.on_roi_toggle,
        )
        self.roi_crop_cb.pack(side="left", padx=(0, 6))
        CTkToolTip(
            self.roi_crop_cb,
            "Restricts GPU shader filtering and ⚡ OIDN Preview to a cropped region for sub-second testing on 4K/8K frames.\n"
            "Draw any rectangle with Right-Click Drag or Shift+Left Drag. Full batch export always writes full-frame.",
        )

        self.roi_reset_btn = ctk.CTkButton(tb_row2, text="Center ROI", width=78, height=22, fg_color="#3a3a3a", hover_color="#505050", font=lbl_font, command=self.reset_roi)
        self.roi_reset_btn.pack(side="left", padx=(0, 8))
        self.roi_info_lbl = ctk.CTkLabel(tb_row2, text="", font=lbl_font, text_color="#ffaa00")
        self.roi_info_lbl.pack(side="left")

        self.canvas_container = ctk.CTkFrame(self.main_frame)
        self.canvas_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)
        self.res_label = ctk.CTkLabel(self.canvas_container, text="", text_color="#8ab4f8", font=lbl_font, anchor="w")
        self.res_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 3))

        self.preview_canvas = tk.Canvas(self.canvas_container, bg="#1a1a1a", highlightthickness=0, cursor="crosshair")
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")

        self.v_scroll = ctk.CTkScrollbar(self.canvas_container, orientation="vertical", command=self._on_vscroll)
        self.h_scroll = ctk.CTkScrollbar(self.canvas_container, orientation="horizontal", command=self._on_hscroll)

        self.preview_canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_left_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.preview_canvas.bind("<Leave>", self._on_canvas_leave)
        self.preview_canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.preview_canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.preview_canvas.bind("<ButtonPress-3>", self._on_right_press)
        self.preview_canvas.bind("<B3-Motion>", self._on_right_drag)
        self.preview_canvas.bind("<ButtonRelease-3>", self._on_right_release)
        self.preview_canvas.bind("<Motion>", self._on_canvas_hover)
        self.preview_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.preview_canvas.bind("<Configure>", self._on_preview_canvas_configure)

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
        if GPU_SHADER_ENGINE.available:
            self.log(f"GPU SHADER DOWNSCALER ACTIVE: {GPU_SHADER_ENGINE.renderer_name} (Persistent VRAM Pool | Hotkeys 1–0, Ctrl+1–7)")
        if self.native_oidn is not None:
            self.log(f"DIRECT RAM/VRAM OIDN ENGINE ACTIVE: OIDN v{self.native_oidn.version_str}")
        self.after(80, self._sync_sidebar_scrollbar)

    # ================= ROI & MOUSE INTERACTIONS =================
    def _canvas_to_image_norm(self, cx: float, cy: float) -> tuple[float, float] | None:
        if self._last_draw_geom is None:
            return None
        draw_x, draw_y, x0, y0, _dw, _dh, scale, orig_w, orig_h = self._last_draw_geom
        ix = ((cx - draw_x) / max(scale, 1e-6)) + x0
        iy = ((cy - draw_y) / max(scale, 1e-6)) + y0
        return float(np.clip(ix / max(orig_w, 1), 0.0, 1.0)), float(np.clip(iy / max(orig_h, 1), 0.0, 1.0))

    def _get_roi_fullres_box(self, full_w: int, full_h: int) -> tuple[int, int, int, int]:
        rn = self.roi_norm
        return clamp_roi_box(rn[0] * full_w, rn[1] * full_h, rn[2] * full_w, rn[3] * full_h, full_w, full_h)

    def _update_roi_badge(self):
        if not hasattr(self, "roi_info_lbl"):
            return
        if not self.roi_crop_var.get():
            self.roi_info_lbl.configure(text="")
            return
        ow, oh = self.last_known_orig_dims
        x0, y0, x1, y1 = self._get_roi_fullres_box(ow, oh)
        self.roi_info_lbl.configure(text=f"ROI {x0},{y0} → {x1 - x0}×{y1 - y0}")

    def on_roi_toggle(self):
        self.ram_roi_denoised.clear()
        self._update_roi_badge()
        if hasattr(self, "channel_selector"):
            self.channel_selector.refresh_label()
        self.reload_current_preview()

    def reset_roi(self):
        self.roi_norm = (0.35, 0.35, 0.65, 0.65)
        self.ram_roi_denoised.clear()
        self._update_roi_badge()
        if not self.roi_crop_var.get():
            self.roi_crop_var.set(True)
        self.reload_current_preview()

    def _begin_roi_draw(self, event):
        pt = self._canvas_to_image_norm(event.x, event.y)
        if pt is None:
            return
        self._roi_drawing = True
        self._roi_drag_start = pt
        self._roi_rubber_norm = (pt[0], pt[1], pt[0], pt[1])

    def _update_roi_draw(self, event):
        if not self._roi_drawing:
            return
        pt = self._canvas_to_image_norm(event.x, event.y)
        if pt is None:
            return
        sx, sy = self._roi_drag_start
        self._roi_rubber_norm = (min(sx, pt[0]), min(sy, pt[1]), max(sx, pt[0]), max(sy, pt[1]))
        self._draw_roi_overlay()

    def _finish_roi_draw(self):
        if not self._roi_drawing:
            return
        self._roi_drawing = False
        rn = self._roi_rubber_norm
        self._roi_rubber_norm = None
        if rn and (rn[2] - rn[0]) > 0.015 and (rn[3] - rn[1]) > 0.015:
            ow, oh = self.last_known_orig_dims
            bx0, by0, bx1, by1 = clamp_roi_box(rn[0] * ow, rn[1] * oh, rn[2] * ow, rn[3] * oh, ow, oh)
            self.roi_norm = (bx0 / max(ow, 1), by0 / max(oh, 1), bx1 / max(ow, 1), by1 / max(oh, 1))
            if not self.roi_crop_var.get():
                self.roi_crop_var.set(True)
            self.ram_roi_denoised.clear()
            self._update_roi_badge()
            if hasattr(self, "channel_selector"):
                self.channel_selector.refresh_label()
            self.reload_current_preview()
        else:
            self._draw_roi_overlay()

    def _on_left_press(self, event):
        self.preview_canvas.focus_set()
        if event.state & 0x0001:  # Shift + Left Drag always draws ROI crop
            self._begin_roi_draw(event)
            return
        self.on_canvas_press_compare(event)

    def _on_left_drag(self, event):
        if self._roi_drawing:
            self._update_roi_draw(event)

    def _on_left_release(self, event):
        if self._roi_drawing:
            self._finish_roi_draw()
            return
        self.on_canvas_release_compare(event)

    def _on_right_press(self, event):
        self.preview_canvas.focus_set()
        if self.roi_crop_var.get():
            self._begin_roi_draw(event)
        else:
            self._on_pan_start(event)

    def _on_right_drag(self, event):
        if self._roi_drawing:
            self._update_roi_draw(event)
        elif not self.roi_crop_var.get():
            self._on_pan_drag(event)

    def _on_right_release(self, _=None):
        if self._roi_drawing:
            self._finish_roi_draw()

    def _on_canvas_leave(self, event=None):
        self.on_canvas_release_compare(event)
        if self.primary_rgb is not None:
            self._update_status_line()

    def _on_canvas_hover(self, event):
        if self._last_draw_geom is None or self.primary_linear is None:
            return
        active_lin = self.alternate_linear if (self.is_holding_compare and self.alternate_linear is not None) else self.primary_linear
        active_diag = self.alternate_diag if (self.is_holding_compare and self.alternate_linear is not None) else self.primary_diag
        draw_x, draw_y, x0, y0, _dw, _dh, scale, ow, oh = self._last_draw_geom
        ix = int(np.floor(((event.x - draw_x) / max(scale, 1e-6)) + x0))
        iy = int(np.floor(((event.y - draw_y) / max(scale, 1e-6)) + y0))
        if 0 <= ix < active_lin.shape[1] and 0 <= iy < active_lin.shape[0]:
            diag_bit = int(active_diag[iy, ix]) if (active_diag is not None and active_diag.shape[:2] == active_lin.shape[:2]) else 0
            probe_txt = probe_pixel_summary(active_lin, ix, iy, self.channel_var.get(), diag_bit=diag_bit)
            self._update_status_line(probe_txt)

    def _get_current_scale(self, orig_w: int, orig_h: int) -> float:
        z_str = self.zoom_var.get()
        if z_str == "Fit":
            cw = max(self.preview_canvas.winfo_width(), 100)
            ch = max(self.preview_canvas.winfo_height(), 100)
            return min(cw / float(max(orig_w, 1)), ch / float(max(orig_h, 1)))
        return float(z_str.replace("%", "")) / 100.0

    def _on_pan_start(self, event):
        self.preview_canvas.focus_set()
        self._drag_start_x, self._drag_start_y = event.x, event.y
        self._drag_pan_x, self._drag_pan_y = self.view_pan_x, self.view_pan_y

    def _on_pan_drag(self, event):
        if self.primary_rgb is None or self.zoom_var.get() == "Fit":
            return
        self.view_pan_x = self._drag_pan_x - (event.x - self._drag_start_x)
        self.view_pan_y = self._drag_pan_y - (event.y - self._drag_start_y)
        self.render_preview_to_canvas()

    def _on_hscroll(self, *args):
        if self.primary_rgb is None:
            return
        oh, ow, _ = self.primary_rgb.shape
        virt_w = max(1.0, ow * self._get_current_scale(ow, oh))
        if args[0] == "moveto":
            self.view_pan_x = float(args[1]) * virt_w
        elif args[0] == "scroll":
            self.view_pan_x += int(args[1]) * 45.0
        self.render_preview_to_canvas()

    def _on_vscroll(self, *args):
        if self.primary_rgb is None:
            return
        oh, ow, _ = self.primary_rgb.shape
        virt_h = max(1.0, oh * self._get_current_scale(ow, oh))
        if args[0] == "moveto":
            self.view_pan_y = float(args[1]) * virt_h
        elif args[0] == "scroll":
            self.view_pan_y += int(args[1]) * 45.0
        self.render_preview_to_canvas()

    def on_mousewheel(self, event):
        if self.primary_rgb is None:
            return
        orig_h, orig_w, _ = self.primary_rgb.shape
        canv_w = max(self.preview_canvas.winfo_width(), 100)
        canv_h = max(self.preview_canvas.winfo_height(), 100)

        old_scale = self._get_current_scale(orig_w, orig_h)
        old_vw, old_vh = orig_w * old_scale, orig_h * old_scale
        old_off_x = max(0.0, (canv_w - old_vw) * 0.5)
        old_off_y = max(0.0, (canv_h - old_vh) * 0.5)

        mx, my = float(event.x), float(event.y)
        img_x = float(np.clip((self.view_pan_x + mx - old_off_x) / max(old_scale, 1e-5), 0.0, float(orig_w)))
        img_y = float(np.clip((self.view_pan_y + my - old_off_y) / max(old_scale, 1e-5), 0.0, float(orig_h)))

        cur_pct = old_scale * 100.0
        if event.delta > 0:
            higher = [z for z in self.ZOOM_STEPS if z > cur_pct + 0.5]
            new_pct = higher[0] if higher else self.ZOOM_STEPS[-1]
        else:
            lower = [z for z in self.ZOOM_STEPS if z < cur_pct - 0.5]
            new_pct = lower[-1] if lower else self.ZOOM_STEPS[0]

        new_zoom_str = f"{new_pct}%"
        if new_zoom_str != self.zoom_var.get():
            self.zoom_var.set(new_zoom_str)
            new_scale = new_pct / 100.0
            new_vw, new_vh = orig_w * new_scale, orig_h * new_scale
            new_off_x = max(0.0, (canv_w - new_vw) * 0.5)
            new_off_y = max(0.0, (canv_h - new_vh) * 0.5)
            self.view_pan_x = img_x * new_scale - (mx - new_off_x)
            self.view_pan_y = img_y * new_scale - (my - new_off_y)
            self.render_preview_to_canvas()

    def _update_status_line(self, probe_txt: str = ""):
        if self.primary_rgb is None:
            return
        active_arr = self.alternate_rgb if (self.is_holding_compare and self.alternate_rgb is not None) else self.primary_rgb
        active_tag = f"COMPARE: {self.alternate_tag}" if (self.is_holding_compare and self.alternate_rgb is not None) else self.primary_tag
        if GPU_SHADER_ENGINE.show_mask_xray:
            active_tag = f"🟢🔴 X-RAY MASK | {active_tag}"
        insp = self.inspector_var.get()
        if insp != INSPECTOR_MODES[0]:
            active_tag = f"[{insp}] {active_tag}"
        orig_h, orig_w, _ = active_arr.shape
        gpu_ms_txt = f" | ⚡{GPU_SHADER_ENGINE.last_gpu_ms:.1f}ms" if GPU_SHADER_ENGINE.available and GPU_SHADER_ENGINE.last_gpu_ms > 0 else ""
        status_str = f"{active_tag} ({orig_w}×{orig_h}{gpu_ms_txt})"
        if probe_txt:
            status_str = f"{probe_txt} | {status_str}"
        if len(status_str) > 165:
            status_str = status_str[:163] + "…"
        self.res_label.configure(text=status_str)

    def _draw_roi_overlay(self):
        self.preview_canvas.delete("roi_overlay")
        if self._last_draw_geom is None:
            return
        rn = self._roi_rubber_norm if self._roi_drawing else (self.roi_norm if self.roi_crop_var.get() else None)
        if rn is None:
            return
        draw_x, draw_y, x0, y0, _dw, _dh, scale, orig_w, orig_h = self._last_draw_geom
        sx0 = draw_x + (rn[0] * orig_w - x0) * scale
        sy0 = draw_y + (rn[1] * orig_h - y0) * scale
        sx1 = draw_x + (rn[2] * orig_w - x0) * scale
        sy1 = draw_y + (rn[3] * orig_h - y0) * scale
        self.preview_canvas.create_rectangle(
            sx0, sy0, sx1, sy1, outline="#ffaa00", width=2, dash=(6, 4), tags="roi_overlay"
        )
        fw, fh = self.last_known_orig_dims
        rx0, ry0, rx1, ry1 = clamp_roi_box(rn[0] * fw, rn[1] * fh, rn[2] * fw, rn[3] * fh, fw, fh)
        self.preview_canvas.create_text(
            sx0 + 4, max(12.0, sy0 - 10.0),
            text=f"ROI {rx1 - rx0}×{ry1 - ry0}",
            fill="#ffaa00", anchor="sw" if sy0 > 20 else "nw", font=("Segoe UI", 9, "bold"), tags="roi_overlay",
        )

    def _on_preview_canvas_configure(self, event):
        dims = (int(event.width), int(event.height))
        if dims == self._last_canvas_dims or dims[0] < 32 or dims[1] < 32:
            return
        self._last_canvas_dims = dims
        if self._resize_debounce_id is not None:
            try:
                self.after_cancel(self._resize_debounce_id)
            except Exception:
                pass
        self._resize_debounce_id = self.after(24, self._flush_canvas_resize)

    def _flush_canvas_resize(self):
        self._resize_debounce_id = None
        self.render_preview_to_canvas()
        self._schedule_sidebar_scrollbar_sync()

    def render_preview_to_canvas(self):
        if self.primary_rgb is None:
            return
        active_arr = self.alternate_rgb if (self.is_holding_compare and self.alternate_rgb is not None) else self.primary_rgb
        orig_h, orig_w, _ = active_arr.shape
        self._update_status_line()

        canv_w = max(self.preview_canvas.winfo_width(), 100)
        canv_h = max(self.preview_canvas.winfo_height(), 100)
        scale = self._get_current_scale(orig_w, orig_h)
        virt_w, virt_h = max(1.0, orig_w * scale), max(1.0, orig_h * scale)

        is_fit = self.zoom_var.get() == "Fit"
        need_h = (not is_fit) and (virt_w > canv_w + 1)
        need_v = (not is_fit) and (virt_h > canv_h + 1)

        if need_v != self._v_scroll_shown:
            self._v_scroll_shown = need_v
            if need_v:
                self.v_scroll.grid(row=0, column=1, sticky="ns")
            else:
                self.v_scroll.grid_remove()
        if need_h != self._h_scroll_shown:
            self._h_scroll_shown = need_h
            if need_h:
                self.h_scroll.grid(row=1, column=0, sticky="ew")
            else:
                self.h_scroll.grid_remove()

        max_pan_x = max(0.0, virt_w - canv_w) if need_h else 0.0
        max_pan_y = max(0.0, virt_h - canv_h) if need_v else 0.0
        self.view_pan_x = float(np.clip(self.view_pan_x, 0.0, max_pan_x))
        self.view_pan_y = float(np.clip(self.view_pan_y, 0.0, max_pan_y))

        if need_h:
            self.h_scroll.set(self.view_pan_x / virt_w, min(1.0, (self.view_pan_x + canv_w) / virt_w))
        if need_v:
            self.v_scroll.set(self.view_pan_y / virt_h, min(1.0, (self.view_pan_y + canv_h) / virt_h))

        x0 = max(0, int(np.floor(self.view_pan_x / scale)))
        y0 = max(0, int(np.floor(self.view_pan_y / scale)))
        x1 = min(orig_w, int(np.ceil((self.view_pan_x + canv_w) / scale)))
        y1 = min(orig_h, int(np.ceil((self.view_pan_y + canv_h) / scale)))
        if x1 <= x0 or y1 <= y0:
            return

        roi = active_arr[y0:y1, x0:x1, :]
        dst_w = max(1, int(round((x1 - x0) * scale)))
        dst_h = max(1, int(round((y1 - y0) * scale)))
        interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
        scaled_roi = cv2.resize(roi, (dst_w, dst_h), interpolation=interp) if (dst_w != (x1 - x0) or dst_h != (y1 - y0)) else roi
        self.tk_img_viewport = ImageTk.PhotoImage(Image.fromarray(scaled_roi))

        draw_x = int(round(x0 * scale - self.view_pan_x)) if need_h else max(0, (canv_w - dst_w) // 2)
        draw_y = int(round(y0 * scale - self.view_pan_y)) if need_v else max(0, (canv_h - dst_h) // 2)
        self._last_draw_geom = (draw_x, draw_y, x0, y0, dst_w, dst_h, scale, orig_w, orig_h)

        if self.canvas_img_id is None:
            self.canvas_img_id = self.preview_canvas.create_image(draw_x, draw_y, anchor="nw", image=self.tk_img_viewport)
        else:
            self.preview_canvas.coords(self.canvas_img_id, draw_x, draw_y)
            self.preview_canvas.itemconfig(self.canvas_img_id, image=self.tk_img_viewport)

        self._draw_roi_overlay()
        if self._spinner_active:
            self._draw_spinner_overlay()

    # ================= BENCHMARK & CONFIG POPUPS =================
    def open_benchmark_popup(self):
        if self.bench_window is not None and self.bench_window.winfo_exists():
            self.bench_window.focus_force()
            return
        self.bench_window = win = ctk.CTkToplevel(self)
        win.title(f"{APP_NAME} — OIDN Hardware Benchmark (CPU & GPU)")
        win.geometry("650x530")
        win.minsize(600, 460)
        win.attributes("-topmost", True)
        win.protocol("WM_DELETE_WINDOW", self._on_bench_close)
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
        ctk.CTkOptionMenu(ctrl_row, variable=self.bench_scope_var, width=255, height=28, values=["Quick Production (6 Tests/Device)", "Quick CleanAux Only (3 Tests/Device)", "Ultra-Fast 1080p (1 Test/Device)", "Full Suite (All 15 Tests/Device)"]).pack(side="left", padx=(0, 8))

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

    def _on_bench_close(self):
        self.bench_cancel_flag = True
        self.bench_running = False
        if self.bench_process and self.bench_process.poll() is None:
            try:
                self.bench_process.terminate()
            except Exception:
                pass
        if self.bench_window is not None and self.bench_window.winfo_exists():
            try:
                self.bench_window.destroy()
            except Exception:
                pass
        self.bench_window = None

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
        if self.bench_running:
            return
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
        total_devs = max(len(devices), 1)
        total_bench_tests = total_devs * max(tests_per_dev, 1)
        for dev_i, dev in enumerate(devices):
            if self.bench_cancel_flag:
                break
            cmd = [str(bench_exe), "-d", dev, "-v", "1"] + (["-r", regex_filter] if regex_filter else [])
            dev_test_idx, dev_upper = 0, dev.upper()
            dev_offset = dev_i * tests_per_dev
            dev_prefix = f"[{dev_i + 1}/{total_devs}] " if total_devs > 1 else ""
            header_msg = f"=== Running OIDN Benchmark on [{dev_upper}] ==="
            self._post(lambda m=header_msg: self._bench_log(m))
            self._post(
                lambda off=dev_offset, tot=total_bench_tests, du=dev_upper, dp=dev_prefix, t=tests_per_dev:
                self._update_bench_progress(off, tot, f"{dp}{du} Test 1/{t}")
            )
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
                                results_by_dev[dev][pending_test_name] = msec_val
                                disp_name = format_benchmark_test_name(pending_test_name)
                                fmt_line = f"  [{dev_test_idx}/{tests_per_dev}] {disp_name:<22} : {int(round(msec_val)):>6d} ms/img  ({1000.0 / max(msec_val, 1e-4):>5.1f} img/s)"
                                self._post(lambda fl=fmt_line: self._bench_replace_running_line(fl))
                                self._post(lambda p=dev_offset + dev_test_idx, tot=total_bench_tests: self._update_bench_progress(p, tot, None))
                                self.log(fmt_line)
                            pending_test_name = None
                            continue
                        m_full = re.search(r"^([\w\.]+)\s+\.\.\.\s+([\d\.]+)\s+msec/image", line)
                        if m_full:
                            dev_test_idx += 1
                            test_name, msec_val = m_full.group(1), float(m_full.group(2))
                            results_by_dev[dev][test_name] = msec_val
                            disp_name = format_benchmark_test_name(test_name)
                            fmt_line = f"  [{dev_test_idx}/{tests_per_dev}] {disp_name:<22} : {int(round(msec_val)):>6d} ms/img  ({1000.0 / max(msec_val, 1e-4):>5.1f} img/s)"
                            self._post(lambda fl=fmt_line: self._bench_log(fl))
                            self._post(
                                lambda p=dev_offset + dev_test_idx, tot=total_bench_tests, du=dev_upper, dp=dev_prefix, c=dev_test_idx, t=tests_per_dev:
                                self._update_bench_progress(p, tot, f"{dp}{du} Test {c}/{t}")
                            )
                            self.log(fmt_line)
                        else:
                            clean_misc = sanitize_benchmark_log_line(line)
                            if clean_misc:
                                self._post(lambda l=clean_misc: self._bench_log(f"  {l}"))
                    m_part = re.search(r"^\s*([\w\.]+)\s+\.\.\.\s*$", buf)
                    if m_part and pending_test_name is None:
                        pending_test_name, buf = m_part.group(1), ""
                        dev_test_idx += 1
                        disp_name = format_benchmark_test_name(pending_test_name)
                        run_line = f"  [{dev_test_idx}/{tests_per_dev}] {disp_name:<22} : running..."
                        self._post(lambda rl=run_line: self._bench_log(rl))
                        self._post(
                            lambda p=dev_offset + max(0, dev_test_idx - 1), tot=total_bench_tests, du=dev_upper, dp=dev_prefix, c=dev_test_idx, t=tests_per_dev:
                            self._update_bench_progress(p, tot, f"{dp}{du} Test {c}/{t}")
                        )
                self.bench_process.wait()
            except Exception as ex:
                self._post(lambda em=f"Benchmark error on {dev}: {ex}": self._bench_log(em))
            finally:
                self.bench_process = None
            self._post(lambda: self._bench_log(""))

        if not self.bench_cancel_flag and len(devices) == 2:
            gpu_dev, cpu_dev = devices[0], devices[1]
            gpu_res, cpu_res = results_by_dev.get(gpu_dev, {}), results_by_dev.get(cpu_dev, {})
            common = [k for k in gpu_res if k in cpu_res]
            if common:
                s_lines = ["=" * 60, f"⚡ SPEEDUP SUMMARY: {gpu_dev.upper()} vs. {cpu_dev.upper()}"]
                for tn in common:
                    g_ms, c_ms = gpu_res[tn], cpu_res[tn]
                    disp_name = format_benchmark_test_name(tn)
                    s_lines.append(f"  • {disp_name:<22}: {gpu_dev.upper()} {c_ms / max(g_ms, 1e-4):.1f}× ({int(round(g_ms))} vs {int(round(c_ms))} ms)")
                s_lines.append("=" * 60)
                sb = "\n".join(s_lines)
                self._post(lambda s=sb: self._bench_log(s))
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

        self._post(_finish_ui)

    def _open_local_license_file(self):
        for cand in (BASE_DIR / "oidn" / "LICENSE.txt", BASE_DIR / "LICENSE", BASE_DIR / "LICENSE.txt"):
            if cand.exists():
                try:
                    os.startfile(str(cand))
                    return
                except Exception:
                    pass
        webbrowser.open("https://www.apache.org/licenses/LICENSE-2.0")

    def _upgrade_openexr_and_restart(self, target_ver: str):
        if getattr(sys, "frozen", False):
            webbrowser.open("https://pypi.org/project/OpenEXR/")
            return
        script_path = BASE_DIR / "polydenoise_studio.py"
        py_exe = Path(sys.executable).resolve()
        bat_path = Path(tempfile.gettempdir()) / "polydenoise_update_openexr.bat"
        bat_content = (
            "@echo off\r\n"
            f"title Updating OpenEXR to v{target_ver}\r\n"
            f"echo Closing {APP_NAME} to release Windows file lock on OpenEXR.pyd...\r\n"
            "timeout /t 2 /nobreak >NUL\r\n"
            f'"{py_exe}" -m pip install --upgrade --only-binary=:all: OpenEXR\r\n'
            "if errorlevel 1 (\r\n    echo Update failed! Press any key...\r\n    pause >NUL\r\n    exit /b 1\r\n)\r\n"
            f"echo OpenEXR updated successfully! Restarting {APP_NAME}...\r\n"
            f'start "" "{py_exe}" "{script_path}"\r\n'
            "exit /b 0\r\n"
        )
        bat_path.write_text(bat_content, encoding="utf-8")
        os.startfile(str(bat_path))
        self._on_closing()

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

        checks = {
            "tool": (APP_VERSION, check_repo_redirect(TOOL_GITHUB_REPO)),
            "oidn": (self.get_oidn_version_str(), check_repo_redirect("RenderKit/oidn")),
            "exr": (getattr(OpenEXR, "__version__", "Unknown"), check_openexr_pypi()),
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
                            slot.configure(text=f"Update: v{latest_v} ⚡", fg_color="#1f6aa5", hover_color="#144870", text_color="#ffffff", hover=True, state="normal", command=lambda v=latest_v: self._upgrade_openexr_and_restart(v))
                            CTkToolTip(slot, f"Click to auto-update OpenEXR to v{latest_v} via pip and restart {APP_NAME}.")
                        else:
                            slot.configure(text=f"Update: v{latest_v} ↗", fg_color="#1f6aa5", hover_color="#144870", text_color="#ffffff", hover=True, state="normal", command=lambda u=rel_url: webbrowser.open(u))
                    else:
                        slot.configure(text="✓ Up to date", fg_color="transparent", text_color="#4caf50", hover=False, state="disabled")
                else:
                    slot.configure(text=latest_v, fg_color="transparent", text_color="#e57373", hover=False, state="disabled")
            self.chk_upd_btn.configure(state="normal", text="Check for Updates")

        self._post(_apply_slots)

    def _on_toggle_aux_codec(self):
        self.aux_comp_menu.configure(state="normal" if self.sep_aux_comp_var.get() else "disabled")

    def clear_log(self):
        self.log_textbox.delete("1.0", "end")

    def _post(self, fn):
        self._ui_q.put(("call", fn))

    def log(self, message):
        self._ui_q.put(("log", message))

    def update_progress(self, current, total, text_prefix=""):
        ratio = current / total if total > 0 else 0.0
        self._ui_q.put(("prog", ratio, f"{text_prefix} ({current} / {total})"))

    def _poll_ui_queue(self):
        logs, prog, calls = [], None, []
        try:
            while True:
                item = self._ui_q.get_nowait()
                if item[0] == "log":
                    logs.append(item[1])
                elif item[0] == "prog":
                    prog = item
                else:
                    calls.append(item[1])
        except queue.Empty:
            pass
        try:
            if logs:
                self.log_textbox.insert("end", "\n".join(logs) + "\n")
                if int(self.log_textbox.index("end-1c").split(".")[0]) > 4000:
                    self.log_textbox.delete("1.0", "1000.0")
                self.log_textbox.see("end")
            if prog is not None:
                self.progress_bar.set(prog[1])
                self.progress_label.configure(text=prog[2])
            for fn in calls:
                try:
                    fn()
                except Exception:
                    pass
        except tk.TclError:
            return
        self.after(25, self._poll_ui_queue)

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
        self._update_roi_badge()

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

    def _get_sharp_multiplier(self) -> float:
        return round(1.0 + float(self.boost_var.get()) / 100.0, 2)

    def _schedule_slider_shader_update(self):
        if self._slider_debounce_id is not None:
            try:
                self.after_cancel(self._slider_debounce_id)
            except Exception:
                pass
        self._slider_debounce_id = self.after(14, self._execute_debounced_slider_update)

    def _execute_debounced_slider_update(self):
        self._slider_debounce_id = None
        self.reload_current_preview()

    def on_boost_slider_move(self, val):
        txt = f"+{int(round(float(val)))}%"
        if txt != self._last_boost_txt:
            self._last_boost_txt = txt
            self.boost_val_lbl.configure(text=txt)
        self._refresh_filter_controls()
        self._schedule_slider_shader_update()

    def reset_boost(self):
        if abs(self.boost_var.get()) > 1e-3:
            self.boost_var.set(0.0)
            self._last_boost_txt = "+0%"
            self.boost_val_lbl.configure(text="+0%")
            self._refresh_filter_controls()
            self._execute_debounced_slider_update()

    def on_radius_slider_move(self, val):
        txt = f"{round(float(val), 1):.1f}px"
        if txt != self._last_rad_txt:
            self._last_rad_txt = txt
            self.rad_val_lbl.configure(text=txt)
        self._schedule_slider_shader_update()

    def reset_radius(self):
        if abs(self.rad_var.get() - 1.0) > 1e-3:
            self.rad_var.set(1.0)
            self._last_rad_txt = "1.0px"
            self.rad_val_lbl.configure(text="1.0px")
            self._execute_debounced_slider_update()

    def _refresh_filter_controls(self):
        mode = self.downscale_var.get()
        boost_gt0 = self._get_sharp_multiplier() > 1.001
        ui_key = (mode, boost_gt0)
        if ui_key == self._last_filter_ui_state:
            return
        self._last_filter_ui_state = ui_key
        spec = get_filter_spec(mode)
        if hasattr(self, "downscale_tip"):
            self.downscale_tip.set_text(spec.description)
        supports_depth = self._filter_supports_depth(mode, self._get_sharp_multiplier())
        if hasattr(self, "depth_guide_cb"):
            self.depth_guide_cb.configure(state="normal" if supports_depth else "disabled")
            if hasattr(self, "depth_guide_tip"):
                if supports_depth:
                    self.depth_guide_tip.set_text(
                        "Uses log-compressed Z-Depth (second-difference Laplacian) to guard geometric silhouette edges without dampening slanted surfaces, "
                        "and feeds the 7th regression channel in Guided Downscale."
                    )
                elif spec.is_luma_only_mks:
                    self.depth_guide_tip.set_text("Disabled for this filter: uses pure luma-contrast or global unmasked sharpening without G-Buffer Depth guidance.")
                else:
                    self.depth_guide_tip.set_text("Inactive at Boost +0% for this kernel: raise Boost above +0% to activate Depth silhouette protection.")

    def on_downscale_mode_change(self, _=None):
        self._refresh_filter_controls()
        self._update_scale_info_label()
        self.reload_current_preview()

    def on_inspector_change(self, _=None):
        if self._ev_debounce_id is not None:
            try:
                self.after_cancel(self._ev_debounce_id)
            except Exception:
                pass
            self._ev_debounce_id = None
        ev = round(float(self.ev_var.get()), 2)
        ch = self.channel_var.get()
        if self.primary_linear is not None:
            self.primary_rgb = self._tonemap_and_dither_fast(
                self.primary_linear, ev, ch, diag_mask=self.primary_diag, albedo_rgb=self.primary_albedo_scaled
            )
        if self.alternate_linear is not None:
            if self.is_holding_compare:
                self.alternate_rgb = self._tonemap_and_dither_fast(
                    self.alternate_linear, ev, ch, diag_mask=self.alternate_diag, albedo_rgb=self.primary_albedo_scaled
                )
                self._alternate_ev_dirty = False
            else:
                self._alternate_ev_dirty = True
        self.render_preview_to_canvas()

    def _execute_ev_update(self):
        self._ev_debounce_id = None
        ev = round(float(self.ev_var.get()), 2)
        ch = self.channel_var.get()
        if self.primary_linear is not None:
            self.primary_rgb = self._tonemap_and_dither_fast(
                self.primary_linear, ev, ch, diag_mask=self.primary_diag, albedo_rgb=self.primary_albedo_scaled
            )
        if self.alternate_linear is not None:
            if self.is_holding_compare:
                self.alternate_rgb = self._tonemap_and_dither_fast(
                    self.alternate_linear, ev, ch, diag_mask=self.alternate_diag, albedo_rgb=self.primary_albedo_scaled
                )
                self._alternate_ev_dirty = False
            else:
                self._alternate_ev_dirty = True
        self.render_preview_to_canvas()

    def toggle_default_view_mode(self):
        self.show_denoised_default = not self.show_denoised_default
        self.swap_default_btn.configure(
            text="Hold: Raw" if self.show_denoised_default else "Hold: Denoised",
            fg_color="#2b7a2b" if self.show_denoised_default else "#3b5998",
        )
        self.reload_current_preview()

    def on_ev_change(self, val):
        ev = round(float(val), 2)
        txt = f"{ev:+.1f}"
        if txt != self._last_ev_txt:
            self._last_ev_txt = txt
            self.ev_reset_btn.configure(text=txt)
        if self._ev_debounce_id is not None:
            try:
                self.after_cancel(self._ev_debounce_id)
            except Exception:
                pass
        self._ev_debounce_id = self.after(12, self._execute_ev_update)

    def reset_ev(self):
        if abs(self.ev_var.get()) > 1e-4:
            self.ev_var.set(0.0)
            self._last_ev_txt = "+0.0"
            self.ev_reset_btn.configure(text="+0.0")
            if self._ev_debounce_id is not None:
                try:
                    self.after_cancel(self._ev_debounce_id)
                except Exception:
                    pass
                self._ev_debounce_id = None
            self._execute_ev_update()

    def reload_current_preview(self):
        frame, ch = self.frame_var.get(), self.channel_var.get()
        if frame in self.frames_data:
            self.load_preview_pair(frame, ch)

    def _on_space_press(self, event):
        if not self._is_typing_in_entry(event):
            self.on_canvas_press_compare()

    def _on_space_release(self, _=None):
        self.on_canvas_release_compare()

    def _ensure_alternate(self):
        if self._alternate_ev_dirty and self.alternate_linear is not None:
            self.alternate_rgb = self._tonemap_and_dither_fast(
                self.alternate_linear, round(float(self.ev_var.get()), 2), self.channel_var.get(),
                diag_mask=self.alternate_diag, albedo_rgb=self.primary_albedo_scaled,
            )
            self._alternate_ev_dirty = False
            return
        if self.alternate_rgb is not None or self._alt_factory is None:
            return
        try:
            res = self._alt_factory()
            lin, tag = res[0], res[1]
            diag = res[2] if len(res) > 2 else None
        except Exception:
            self._alt_factory = None
            return
        if lin is not None:
            self.alternate_linear, self.alternate_tag, self.alternate_diag = lin, tag, diag
            self.alternate_rgb = self._tonemap_and_dither_fast(
                lin, round(float(self.ev_var.get()), 2), self.channel_var.get(),
                diag_mask=diag, albedo_rgb=self.primary_albedo_scaled,
            )
            self._alternate_ev_dirty = False

    def on_canvas_press_compare(self, _=None):
        self._ensure_alternate()
        if self.alternate_rgb is not None:
            self.is_holding_compare = True
            self.render_preview_to_canvas()
        elif self.primary_rgb is not None:
            self.res_label.configure(text=f"No denoised pass yet for '{self.channel_var.get()}' (Click '⚡ Preview')!")

    def on_canvas_release_compare(self, _=None):
        if self.is_holding_compare:
            self.is_holding_compare = False
            self.render_preview_to_canvas()

    def select_binary(self):
        file_path = filedialog.askopenfilename(
            title="Select oidnDenoise.exe or OpenImageDenoise.dll",
            filetypes=[("OIDN Binary / DLL", "*.exe *.dll"), ("All Files", "*.*")],
        )
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
            self._purge_all_caches()
            self.log(f"\n--- Scanned Folder: {self.input_folder.name} (Freed previous RAM/VRAM caches) ---")
            self.parse_indigo_passes()

    def _get_frame_guides_ram(self, frame_id: str):
        with self.cache_lock:
            if frame_id in self.ram_guides_fullres:
                self.ram_guides_fullres.move_to_end(frame_id)
                return self.ram_guides_fullres[frame_id]
        f_dict = self.frames_data.get(frame_id, {})
        norm_p = f_dict.get("normals") or f_dict.get("normals_pre_bump")
        alb_p = f_dict.get("albedo")
        dep_p = f_dict.get("depth")
        n_arr = a_arr = d_arr = None
        if norm_p and norm_p.exists():
            try:
                n_arr = np.nan_to_num(read_exr_native(norm_p)[0], nan=0.0, posinf=1.0, neginf=-1.0)
            except Exception:
                pass
        if alb_p and alb_p.exists():
            try:
                a_arr = np.clip(np.nan_to_num(read_exr_native(alb_p)[0], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            except Exception:
                pass
        if dep_p and dep_p.exists():
            try:
                d_arr = normalize_depth_guide(read_exr_native(dep_p)[0])
            except Exception:
                pass
        self._store_guides(frame_id, n_arr, a_arr, d_arr)
        return n_arr, a_arr, d_arr

    def _store_guides(self, frame_id: str, n_arr, a_arr, d_arr=None):
        with self.cache_lock:
            self.ram_guides_fullres[frame_id] = (n_arr, a_arr, d_arr)
            self.ram_guides_fullres.move_to_end(frame_id)
            while len(self.ram_guides_fullres) > 1:
                self.ram_guides_fullres.popitem(last=False)

    # ================= UNIVERSAL EXPORTER PASS PARSER =================
    def _extract_frame_and_channel(self, stem: str, discovered_prefixes: list[str]):
        return extract_frame_and_channel(stem, discovered_prefixes)

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
        from .exr_pipeline import AUX_ALIASES, KNOWN_STD_LIGHT_PASSES

        all_known = sorted(list(AUX_ALIASES.keys()) + KNOWN_STD_LIGHT_PASSES, key=len, reverse=True)
        for fp in all_exrs:
            st, st_low = fp.stem, fp.stem.lower()
            if st_low.endswith(("_denoised", "_50pct", "_scaled", "_multilayer")) or "material_mask" in st_low or "object_mask" in st_low:
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
        frame_aux_lookup: dict[str, dict[str, Path]] = {}
        raw_light_candidates = []

        for file_path in all_exrs:
            name, stem = file_path.name, file_path.stem
            low_name, low_stem = name.lower(), stem.lower()
            if file_path.stat().st_size < 2048 and "_denoise" in low_name:
                continue
            if low_name.endswith(("_50pct.exr", "_scaled.exr", "_multilayer.exr")):
                continue
            if "material_mask" in low_stem or "object_mask" in low_stem:
                if not low_stem.endswith("_denoised"):
                    frame_id, _ = self._extract_frame_and_channel(stem, sorted_prefixes)
                    self.aux_mask_files.append({"path": file_path, "type": "mask", "frame_id": frame_id})
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
                self.aux_mask_files.append({"path": file_path, "type": pass_name, "frame_id": frame_id})
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
                "target": file_path,
                "albedo": aux_dict.get("albedo"),
                "normal_bump": aux_dict.get("normals"),
                "normal_pre": aux_dict.get("normals_pre_bump"),
                "depth": aux_dict.get("depth"),
                "frame_id": frame_id,
                "pass_name": pass_name,
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

        btn_state = "normal" if (self.render_queue or self.frames_data) else "disabled"
        self.start_btn.configure(state="normal" if self.render_queue else "disabled")
        self.test_single_btn.configure(state=btn_state)

        if raw_light_candidates:
            skip_empty_cfg = bool(self.skip_empty_var.get())
            first_f = sorted(list(self.frames_data.keys()))[0] if self.frames_data else None
            self.start_viewport_spinner(f"Scanning {len(raw_light_candidates)} passes in background...")
            threading.Thread(
                target=self._background_folder_scan_worker,
                args=(current_gen, list(raw_light_candidates), skipped_denoised, skip_empty_cfg, first_f),
                daemon=True,
            ).start()

    def _background_folder_scan_worker(
        self,
        gen_id: int,
        candidates: list,
        paired_den_count: int,
        skip_empty: bool,
        first_frame: str | None,
    ):
        scan_all = len(candidates) <= 60
        discovered_empty = []
        total_c = len(candidates)
        for idx, (frame_id, pass_name, file_path) in enumerate(candidates, start=1):
            if gen_id != self._scan_generation or self.is_running:
                return
            if scan_all or frame_id == first_frame:
                if idx % 3 == 0:
                    self._post(lambda i=idx, t=total_c: self.start_viewport_spinner(f"Scanning passes ({i}/{t})..."))
                if is_exr_black_fast(file_path):
                    discovered_empty.append((frame_id, pass_name))

        def _finish_bg_scan():
            if gen_id != self._scan_generation:
                return
            for f_id, p_name in discovered_empty:
                self.empty_passes.setdefault(f_id, set()).add(p_name)
            if skip_empty and discovered_empty:
                empty_lookup = set(discovered_empty)
                self.render_queue = [it for it in self.render_queue if (it["frame_id"], it["pass_name"]) not in empty_lookup]
            self.channel_selector.refresh_label()
            self.stop_viewport_spinner()
            if not self.is_running:
                self.start_btn.configure(state="normal" if self.render_queue else "disabled")
                self.test_single_btn.configure(state="normal" if self.frames_data else "disabled")
            self.log(
                f"Background scan complete: {len(self.render_queue)} active light passes | "
                f"Flagged {len(discovered_empty)} empty (black) | {len(self.aux_mask_files)} Aux/Mask/ID | Paired {paired_den_count} _denoised."
            )

        self._post(_finish_bg_scan)

    def on_frame_change(self, selected_frame):
        if selected_frame not in self.frames_data:
            return
        self._preview_req_id += 1
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
        self._preview_req_id += 1
        frame = self.frame_var.get()
        if frame in self.frames_data:
            self.load_preview_pair(frame, selected_channel)

    def on_zoom_change(self, _=None):
        self.render_preview_to_canvas()

    def _get_ign_dither(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if key not in self._dither_cache:
            self._dither_cache[key] = compute_ign_dither(h, w)
        return self._dither_cache[key]

    def _tonemap_and_dither_fast(
        self, linear_rgb: np.ndarray, ev_val: float, ch_name: str = "",
        diag_mask: np.ndarray | None = None, albedo_rgb: np.ndarray | None = None,
    ) -> np.ndarray:
        insp_mode = self.inspector_var.get()
        ch_low = ch_name.lower()
        xray_on = GPU_SHADER_ENGINE.show_mask_xray
        is_immutable_cache = (linear_rgb is self.primary_linear) or (linear_rgb is self.alternate_linear)
        if (
            insp_mode == INSPECTOR_MODES[0]
            and not xray_on
            and ch_low != "depth"
            and "normal" not in ch_low
            and GPU_SHADER_ENGINE.available
            and linear_rgb.ndim == 3
            and linear_rgb.shape[2] == 3
        ):
            try:
                return GPU_SHADER_ENGINE.tonemap_gpu(linear_rgb, ev_val, cache_immutable=is_immutable_cache)
            except Exception:
                pass
        fp16_k = None
        if insp_mode == INSPECTOR_MODES[0] and not xray_on and ch_low != "depth" and "normal" not in ch_low and is_immutable_cache:
            if self._fp16_key_ref is not None and self._fp16_key_ref() is linear_rgb:
                fp16_k = self._fp16_key_u16
            else:
                fp16_k = build_fp16_lut_key(linear_rgb)
                try:
                    self._fp16_key_ref = weakref.ref(linear_rgb)
                    self._fp16_key_u16 = fp16_k
                except TypeError:
                    self._fp16_key_ref = None
                    self._fp16_key_u16 = None
        h, w, _ = linear_rgb.shape
        return render_viewport_rgb(
            linear_rgb,
            ev_val,
            ch_name=ch_name,
            inspector_mode=insp_mode,
            ign_dither=self._get_ign_dither(h, w),
            xray_mask_active=xray_on,
            diag_mask=diag_mask,
            albedo_rgb=albedo_rgb,
            fp16_key=fp16_k,
        )

    def _get_roi_background_linear(self, full_src: np.ndarray, frame_id: str, ch_label: str, tw: int, th: int, is_den: bool) -> np.ndarray:
        """Cached dimmed full-frame backdrop when ROI Crop Preview is enabled."""
        bg_key = (frame_id, ch_label, is_den, tw, th, id(full_src))
        if bg_key in self._roi_bg_cache:
            return self._roi_bg_cache[bg_key]
        h, w = full_src.shape[:2]
        bg = full_src if (w, h) == (tw, th) else cv2.resize(full_src, (tw, th), interpolation=cv2.INTER_AREA)
        bg = (bg * 0.45).astype(np.float32)
        self._roi_bg_cache.clear()
        self._roi_bg_cache[bg_key] = bg
        return bg

    def _apply_roi_crop_downscale(
        self, full_src: np.ndarray, frame_id: str, ch_label: str, is_den: bool,
        ds_kernel: str, sharp_str: float, sharp_rad: float,
        norm_g, alb_g, dep_g, sc_mode: str, sc_val: str,
        full_diag: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int]]:
        """Downscale ONLY the selected ROI slice and composite cleanly onto the full canvas grid."""
        oh, ow = full_src.shape[:2]
        tw, th = DownscaleEngine.compute_target_size(ow, oh, sc_mode, sc_val)
        rx0, ry0, rx1, ry1 = self._get_roi_fullres_box(ow, oh)

        dx0 = max(0, int(round(rx0 * tw / float(ow))))
        dy0 = max(0, int(round(ry0 * th / float(oh))))
        dx1 = min(tw, max(dx0 + 2, int(round(rx1 * tw / float(ow)))))
        dy1 = min(th, max(dy0 + 2, int(round(ry1 * th / float(oh)))))
        roi_tw, roi_th = dx1 - dx0, dy1 - dy0

        src_crop = np.ascontiguousarray(full_src[ry0:ry1, rx0:rx1, :])
        n_crop = np.ascontiguousarray(norm_g[ry0:ry1, rx0:rx1, :]) if (norm_g is not None and norm_g.shape[:2] == (oh, ow)) else None
        a_crop = np.ascontiguousarray(alb_g[ry0:ry1, rx0:rx1, :]) if (alb_g is not None and alb_g.shape[:2] == (oh, ow)) else None
        d_crop = np.ascontiguousarray(dep_g[ry0:ry1, rx0:rx1]) if (dep_g is not None and dep_g.shape[:2] == (oh, ow)) else None

        if ch_label in self.blacklist:
            tile = DownscaleEngine.downscale_aux_pass(src_crop, roi_tw, roi_th, ch_label, ds_kernel)
        else:
            tile = DownscaleEngine.downscale(
                src_crop, roi_tw, roi_th, ds_kernel, sharpness=sharp_str, radius=sharp_rad,
                norm_rgb=n_crop, alb_rgb=a_crop, depth_arr=d_crop,
            )

        bg = self._get_roi_background_linear(full_src, frame_id, ch_label, tw, th, is_den).copy()
        bg[dy0:dy1, dx0:dx1, :] = tile
        diag_out = _downsample_diag_mask(full_diag, tw, th) if full_diag is not None else None
        return bg, diag_out, (rx0, ry0, rx1, ry1)

    def _get_scaled_linear_rgb(self, img_path: Path, is_denoised: bool, ch_label: str = "", frame_id: str = ""):
        ds_kernel = self.downscale_var.get()
        sc_mode = self.scale_mode_var.get()
        sc_val = self.scale_val_entry.get().strip()
        sharp_str = self._get_sharp_multiplier()
        sharp_rad = round(float(self.rad_var.get()), 2)
        use_depth = bool(self.depth_guide_var.get()) and self._filter_supports_depth(ds_kernel, sharp_str)
        boost_pct = int(round(float(self.boost_var.get())))
        roi_active = bool(self.roi_crop_var.get())
        f_key = frame_id or self.frame_var.get()

        need_guides = self._downscale_needs_guides(ds_kernel, sharp_str) and (ch_label not in self.blacklist or "roughness" in ch_label.lower())
        need_alb_for_insp = self.inspector_var.get() == INSPECTOR_MODES[3]
        norm_g, alb_g, dep_g = self._get_frame_guides_ram(f_key) if (need_guides or need_alb_for_insp) else (None, None, None)
        depth_arg = dep_g if use_depth else None

        if alb_g is not None and need_alb_for_insp:
            aw, ah = alb_g.shape[1], alb_g.shape[0]
            tw_a, th_a = DownscaleEngine.compute_target_size(aw, ah, sc_mode, sc_val)
            self.primary_albedo_scaled = alb_g if (tw_a, th_a) == (aw, ah) else cv2.resize(alb_g, (tw_a, th_a), interpolation=cv2.INTER_AREA)
        else:
            self.primary_albedo_scaled = None

        # Check ROI denoised cache first when ROI Preview is checked
        if is_denoised and roi_active and (f_key, ch_label) in self.ram_roi_denoised:
            roi_entry = self.ram_roi_denoised[(f_key, ch_label)]
            rx0, ry0, rx1, ry1 = roi_entry["roi_px"]
            den_tile_full = roi_entry["rgb"]
            ow, oh = self.last_known_orig_dims
            cur_box = self._get_roi_fullres_box(ow, oh)
            if cur_box == (rx0, ry0, rx1, ry1):
                tw, th = DownscaleEngine.compute_target_size(ow, oh, sc_mode, sc_val)
                dx0 = max(0, int(round(rx0 * tw / float(ow))))
                dy0 = max(0, int(round(ry0 * th / float(oh))))
                dx1 = min(tw, max(dx0 + 2, int(round(rx1 * tw / float(ow)))))
                dy1 = min(th, max(dy0 + 2, int(round(ry1 * th / float(oh)))))
                n_crop = np.ascontiguousarray(norm_g[ry0:ry1, rx0:rx1, :]) if (norm_g is not None and norm_g.shape[:2] == (oh, ow)) else None
                a_crop = np.ascontiguousarray(alb_g[ry0:ry1, rx0:rx1, :]) if (alb_g is not None and alb_g.shape[:2] == (oh, ow)) else None
                d_crop = np.ascontiguousarray(depth_arg[ry0:ry1, rx0:rx1]) if (depth_arg is not None and depth_arg.shape[:2] == (oh, ow)) else None
                scaled_tile = DownscaleEngine.downscale(
                    den_tile_full, dx1 - dx0, dy1 - dy0, ds_kernel, sharpness=sharp_str, radius=sharp_rad,
                    norm_rgb=n_crop, alb_rgb=a_crop, depth_arr=d_crop,
                )
                raw_cache_k = (f_key, ch_label, False)
                if raw_cache_k in self.ram_raw_fullres:
                    bg = self._get_roi_background_linear(self.ram_raw_fullres[raw_cache_k], f_key, ch_label, tw, th, True).copy()
                else:
                    bg = np.zeros((th, tw, 3), dtype=np.float32)
                bg[dy0:dy1, dx0:dx1, :] = scaled_tile
                tag = f"{ch_label} [✂ ROI Den {rx1-rx0}×{ry1-ry0} ➔ {ds_kernel} +{boost_pct}%]"
                return bg, tag, None

        if is_denoised and (f_key, ch_label) in self.ram_denoised_fullres:
            full_den = self.ram_denoised_fullres[(f_key, ch_label)]
            oh, ow, _ = full_den.shape
            if roi_active:
                bg, dmask, (rx0, ry0, rx1, ry1) = self._apply_roi_crop_downscale(
                    full_den, f_key, ch_label, True, ds_kernel, sharp_str, sharp_rad,
                    norm_g, alb_g, depth_arg, sc_mode, sc_val,
                )
                return bg, f"{ch_label} [✂ ROI ⚡ {ds_kernel} +{boost_pct}%]", dmask
            tw, th = DownscaleEngine.compute_target_size(ow, oh, sc_mode, sc_val)
            scaled_den = DownscaleEngine.downscale(
                full_den, tw, th, ds_kernel, sharpness=sharp_str, radius=sharp_rad,
                norm_rgb=norm_g, alb_rgb=alb_g, depth_arr=depth_arg,
            )
            tag = f"{ch_label} [⚡ {ds_kernel} +{boost_pct}% R={sharp_rad:.1f}]"
            return scaled_den, tag, None

        if img_path is None or not img_path.exists():
            return None, "", None

        raw_ram_key = (f_key, ch_label, is_denoised)
        if raw_ram_key in self.ram_raw_fullres:
            self.ram_raw_fullres.move_to_end(raw_ram_key)
            img_pos = self.ram_raw_fullres[raw_ram_key]
            diag_full = self.ram_diag_fullres.get(raw_ram_key)
        else:
            rgb_float, _, _ = read_exr_native(img_path)
            diag_full = compute_raw_diagnostics_mask(rgb_float)
            if "normal" in ch_label.lower():
                img_pos = np.nan_to_num(rgb_float, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                img_pos = np.maximum(np.nan_to_num(rgb_float, nan=0.0, posinf=65500.0, neginf=0.0), 0.0)
            self.ram_raw_fullres[raw_ram_key] = img_pos
            self.ram_diag_fullres[raw_ram_key] = diag_full
            self.ram_is_empty_cache[(f_key, ch_label)] = float(np.max(np.abs(img_pos))) <= 1e-6
            while len(self.ram_raw_fullres) > 2:
                evict_k, _ = self.ram_raw_fullres.popitem(last=False)
                self.ram_diag_fullres.pop(evict_k, None)

        oh, ow, _ = img_pos.shape
        ch_short = ch_label or img_path.stem

        if is_denoised:
            orig_w, orig_h = self.last_known_orig_dims
            tw, th = DownscaleEngine.compute_target_size(orig_w, orig_h, sc_mode, sc_val)
            if (ow, oh) == (orig_w, orig_h):
                if roi_active:
                    bg, dmask, _ = self._apply_roi_crop_downscale(
                        img_pos, f_key, ch_short, True, ds_kernel, sharp_str, sharp_rad,
                        norm_g, alb_g, depth_arg, sc_mode, sc_val, full_diag=diag_full,
                    )
                    return bg, f"{ch_short} [✂ ROI Den ➔ {ds_kernel}]", dmask
                img_pos = DownscaleEngine.downscale(
                    img_pos, tw, th, ds_kernel, sharpness=sharp_str, radius=sharp_rad,
                    norm_rgb=norm_g, alb_rgb=alb_g, depth_arr=depth_arg,
                )
                return img_pos, f"{ch_short} [Den ➔ {ds_kernel}]", _downsample_diag_mask(diag_full, tw, th)
            return img_pos, f"{ch_short} [Den Disk {ow}×{oh}]", _downsample_diag_mask(diag_full, ow, oh)

        self.last_known_orig_dims = (ow, oh)
        self._post(self._update_scale_info_label)
        empty_cache_k = (f_key, ch_label)
        if empty_cache_k not in self.ram_is_empty_cache:
            self.ram_is_empty_cache[empty_cache_k] = float(np.max(np.abs(img_pos))) <= 1e-6
        is_empty = self.ram_is_empty_cache[empty_cache_k]
        if is_empty and ch_short not in self.blacklist:
            self.empty_passes.setdefault(f_key, set()).add(ch_short)
            self._post(self.channel_selector.refresh_label)

        tw, th = DownscaleEngine.compute_target_size(ow, oh, sc_mode, sc_val)
        if roi_active:
            bg, dmask, (rx0, ry0, rx1, ry1) = self._apply_roi_crop_downscale(
                img_pos, f_key, ch_short, False, ds_kernel, sharp_str, sharp_rad,
                norm_g, alb_g, depth_arg, sc_mode, sc_val, full_diag=diag_full,
            )
            return bg, f"{ch_short} [{'EMPTY ' if is_empty else ''}✂ ROI {rx1-rx0}×{ry1-ry0} ➔ {ds_kernel}]", dmask

        if ch_short in self.blacklist:
            img_pos = DownscaleEngine.downscale_aux_pass(img_pos, tw, th, ch_short, ds_kernel)
        else:
            img_pos = DownscaleEngine.downscale(
                img_pos, tw, th, ds_kernel, sharpness=sharp_str, radius=sharp_rad,
                norm_rgb=norm_g, alb_rgb=alb_g, depth_arr=depth_arg,
            )
        return img_pos, f"{ch_short} [{'EMPTY ' if is_empty else ''}Raw ➔ {ds_kernel}]", _downsample_diag_mask(diag_full, tw, th)

    def _scale_cached_beauty_sum_on_main_thread(self, frame_id: str, ev: float):
        if frame_id not in self.ram_beauty_sum_fullres:
            return
        raw_full_sum, den_full_sum, any_den, light_count = self.ram_beauty_sum_fullres[frame_id]
        ds_kernel = self.downscale_var.get()
        sc_mode = self.scale_mode_var.get()
        sc_val = self.scale_val_entry.get().strip()
        sharp_str = self._get_sharp_multiplier()
        sharp_rad = round(float(self.rad_var.get()), 2)
        use_depth = bool(self.depth_guide_var.get()) and self._filter_supports_depth(ds_kernel, sharp_str)
        roi_active = bool(self.roi_crop_var.get())
        boost_pct = int(round(float(self.boost_var.get())))

        oh, ow, _ = raw_full_sum.shape
        self.last_known_orig_dims = (ow, oh)
        self._update_scale_info_label()
        tw, th = DownscaleEngine.compute_target_size(ow, oh, sc_mode, sc_val)
        need_guides = self._downscale_needs_guides(ds_kernel, sharp_str)
        need_alb = self.inspector_var.get() == INSPECTOR_MODES[3]
        norm_g, alb_g, dep_g = self._get_frame_guides_ram(frame_id) if (need_guides or need_alb) else (None, None, None)
        if alb_g is not None and need_alb:
            self.primary_albedo_scaled = alb_g if (tw, th) == (ow, oh) else cv2.resize(alb_g, (tw, th), interpolation=cv2.INTER_AREA)
        else:
            self.primary_albedo_scaled = None

        if self.show_denoised_default and any_den and den_full_sum is not None:
            prim_src, prim_t, alt_src, alt_t = den_full_sum, "Den", raw_full_sum, "Raw"
        else:
            prim_src, prim_t = raw_full_sum, "Raw"
            alt_src, alt_t = (den_full_sum if (any_den and den_full_sum is not None) else None), "Den"

        dep_arg = dep_g if use_depth else None
        if roi_active:
            self.primary_linear, self.primary_diag, _ = self._apply_roi_crop_downscale(
                prim_src, frame_id, self.BEAUTY_SUM_KEY, prim_t == "Den", ds_kernel,
                sharp_str, sharp_rad, norm_g, alb_g, dep_arg, sc_mode, sc_val,
            )
            self.primary_tag = f"Beauty ({light_count}L) [✂ ROI {prim_t} ➔ {ds_kernel} +{boost_pct}%]"
            if alt_src is not None:
                self._alt_factory = lambda: (
                    self._apply_roi_crop_downscale(
                        alt_src, frame_id, self.BEAUTY_SUM_KEY, alt_t == "Den", ds_kernel,
                        sharp_str, sharp_rad, norm_g, alb_g, dep_arg, sc_mode, sc_val,
                    )[0],
                    f"Beauty ({light_count}L) [✂ ROI {alt_t} ➔ {ds_kernel} +{boost_pct}%]",
                    None,
                )
            else:
                self._alt_factory = None
        else:
            kw = dict(sharpness=sharp_str, radius=sharp_rad, norm_rgb=norm_g, alb_rgb=alb_g, depth_arr=dep_arg)
            self.primary_linear = DownscaleEngine.downscale(prim_src, tw, th, ds_kernel, **kw)
            self.primary_diag = None
            self.primary_tag = f"Beauty ({light_count}L) [{prim_t} ➔ {ds_kernel} +{boost_pct}%]"
            if alt_src is not None:
                self._alt_factory = lambda: (
                    DownscaleEngine.downscale(alt_src, tw, th, ds_kernel, **kw),
                    f"Beauty ({light_count}L) [{alt_t} ➔ {ds_kernel} +{boost_pct}%]",
                    None,
                )
            else:
                self._alt_factory = None

        self.alternate_linear, self.alternate_tag, self.alternate_rgb, self.alternate_diag = None, "", None, None
        self._alternate_ev_dirty = False
        self.primary_rgb = self._tonemap_and_dither_fast(
            self.primary_linear, ev, self.BEAUTY_SUM_KEY, diag_mask=self.primary_diag, albedo_rgb=self.primary_albedo_scaled
        )
        self.render_preview_to_canvas()

    def load_preview_pair(self, frame_id: str, channel_name: str):
        req_id = self._preview_req_id
        ev = round(float(self.ev_var.get()), 2)

        if channel_name == self.BEAUTY_SUM_KEY:
            if frame_id in self.ram_beauty_sum_fullres:
                self.stop_viewport_spinner()
                self._scale_cached_beauty_sum_on_main_thread(frame_id, ev)
                return

            frame_dict = self.frames_data.get(frame_id, {})
            den_dict = self.denoised_data.get(frame_id, {})
            light_chs = self._get_sum_light_channels(frame_id)
            if not light_chs:
                return
            self.start_viewport_spinner(f"Summing {len(light_chs)} light layers (0/{len(light_chs)})...")

            def _bg_linear_sum_worker():
                raw_sum_full = den_sum_full = None
                any_den, total_l = False, len(light_chs)

                def _read_single_light(ch_name):
                    if req_id != self._preview_req_id:
                        return None
                    r_arr = np.maximum(np.nan_to_num(read_exr_native(frame_dict[ch_name])[0], nan=0.0, posinf=65500.0, neginf=0.0), 0.0)
                    rh, rw, _ = r_arr.shape
                    d_arr, is_den_ch = None, False
                    if (frame_id, ch_name) in self.ram_denoised_fullres:
                        d_arr, is_den_ch = self.ram_denoised_fullres[(frame_id, ch_name)], True
                    elif ch_name in den_dict and den_dict[ch_name].exists():
                        d_disk = np.maximum(np.nan_to_num(read_exr_native(den_dict[ch_name])[0], nan=0.0, posinf=65500.0, neginf=0.0), 0.0)
                        if d_disk.shape[:2] != (rh, rw):
                            d_disk = cv2.resize(d_disk, (rw, rh), interpolation=cv2.INTER_CUBIC)
                        d_arr, is_den_ch = d_disk, True
                    return ch_name, r_arr, (d_arr if is_den_ch else r_arr), is_den_ch

                try:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        futures = [pool.submit(_read_single_light, c) for c in light_chs]
                        for idx, fut in enumerate(futures, start=1):
                            if req_id != self._preview_req_id:
                                return
                            res_item = fut.result()
                            if res_item is None or req_id != self._preview_req_id:
                                return
                            _, r_arr, d_arr, is_den_ch = res_item
                            if is_den_ch:
                                any_den = True
                            if raw_sum_full is None:
                                raw_sum_full, den_sum_full = r_arr.copy(), d_arr.copy()
                            else:
                                raw_sum_full += r_arr
                                den_sum_full += d_arr
                            self._post(lambda i=idx, t=total_l: self.start_viewport_spinner(f"Summing light layers ({i}/{t})...") if req_id == self._preview_req_id else None)
                except Exception as ex:
                    self.log(f"Beauty sum error: {ex}")
                    self._post(self.stop_viewport_spinner)
                    return

                if req_id != self._preview_req_id or raw_sum_full is None:
                    return

                def _finish_sum_on_main():
                    if req_id != self._preview_req_id or self.channel_var.get() != self.BEAUTY_SUM_KEY:
                        self.stop_viewport_spinner()
                        return
                    with self.cache_lock:
                        self.ram_beauty_sum_fullres[frame_id] = (raw_sum_full, den_sum_full if any_den else None, any_den, total_l)
                        while len(self.ram_beauty_sum_fullres) > 1:
                            self.ram_beauty_sum_fullres.popitem(last=False)
                    self.stop_viewport_spinner()
                    self._scale_cached_beauty_sum_on_main_thread(frame_id, round(float(self.ev_var.get()), 2))

                self._post(_finish_sum_on_main)

            threading.Thread(target=_bg_linear_sum_worker, daemon=True).start()
            return

        self.stop_viewport_spinner()
        raw_path = self.frames_data.get(frame_id, {}).get(channel_name)
        if not raw_path:
            return
        den_path = self.denoised_data.get(frame_id, {}).get(channel_name)
        has_ram_den = (frame_id, channel_name) in self.ram_denoised_fullres
        has_roi_den = self.roi_crop_var.get() and (frame_id, channel_name) in self.ram_roi_denoised
        has_den = has_ram_den or has_roi_den or (den_path is not None and den_path.exists())

        primary_is_den = self.show_denoised_default and has_den
        first_path = den_path if primary_is_den else raw_path
        second_path = raw_path if primary_is_den else den_path

        try:
            first_lin, first_tag, first_diag = self._get_scaled_linear_rgb(
                first_path, is_denoised=primary_is_den, ch_label=channel_name, frame_id=frame_id
            )
        except Exception as e:
            self.log(f"Warning: Failed to read {channel_name} ({e}).")
            return
        if first_lin is None:
            return

        self.primary_linear, self.primary_tag, self.primary_diag = first_lin, first_tag, first_diag
        self.alternate_linear, self.alternate_tag, self.alternate_diag = None, "", None
        self._alt_factory = None
        self._alternate_ev_dirty = False
        self.primary_rgb = self._tonemap_and_dither_fast(
            self.primary_linear, ev, channel_name, diag_mask=first_diag, albedo_rgb=self.primary_albedo_scaled
        )
        self.alternate_rgb = None
        self.render_preview_to_canvas()

        if has_den:
            self._alt_factory = lambda: self._get_scaled_linear_rgb(
                second_path, is_denoised=not primary_is_den, ch_label=channel_name, frame_id=frame_id
            )

    def run_oidn_cli(self, cmd_args: list[str]):
        oidn_cwd = str(self.oidn_bin.parent) if self.oidn_bin.parent.exists() else None
        self.active_process = subprocess.Popen(cmd_args, cwd=oidn_cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout_str, stderr_str = self.active_process.communicate()
        ret_code = self.active_process.returncode
        self.active_process = None
        return ret_code, stdout_str, stderr_str

    def build_oidn_cmd(self, hdr_pfm, out_pfm, alb_pfm, nrm_pfm, use_ldr, run_cfg: dict | None = None):
        cfg = run_cfg or self._snapshot_run_config()
        return build_oidn_cmd(
            self.oidn_bin, hdr_pfm, out_pfm, alb_pfm, nrm_pfm,
            device=cfg["device"],
            filter_type=cfg["filter"],
            quality=cfg["quality"],
            precision=cfg["precision"],
            verbose=cfg["verbose"],
            use_ldr=use_ldr,
            clean_aux=cfg["clean_aux"],
            inplace=cfg["inplace"],
            maxmem=cfg["maxmem_str"],
            future_args=cfg["future_args"],
        )

    def _snapshot_run_config(self) -> dict:
        """Capture UI variables on the main thread before spawning worker threads."""
        return {
            "initial_channel": self.channel_var.get(),
            "aux_mode": self.aux_mode_var.get(),
            "hdr_mode": self.hdr_mode_var.get(),
            "decorr": self.decorr_var.get(),
            "ds_kernel": self.downscale_var.get(),
            "sc_mode": self.scale_mode_var.get(),
            "sc_val": self.scale_val_entry.get().strip(),
            "sharp_str": self._get_sharp_multiplier(),
            "sharp_rad": round(float(self.rad_var.get()), 2),
            "use_depth": bool(self.depth_guide_var.get()),
            "export_multilayer": bool(self.multilayer_exr_var.get()),
            "verbose": self.verbose_var.get(),
            "device": self.device_var.get(),
            "quality": self.quality_var.get(),
            "precision": self.precision_var.get(),
            "filter": self.filter_var.get(),
            "clean_aux": bool(self.clean_aux_var.get()),
            "inplace": bool(self.inplace_var.get()),
            "skip_empty": bool(self.skip_empty_var.get()),
            "maxmem_str": self.maxmem_entry.get().strip(),
            "future_args": self.future_args.get().strip(),
            "comp": self.comp_var.get(),
            "comp_level": self.comp_level_entry.get().strip(),
            "exr_depth": self.exr_depth_var.get(),
            "sep_aux": bool(self.sep_aux_comp_var.get()),
            "aux_comp": self.aux_comp_var.get(),
            "async_save": bool(self.async_save_var.get()),
            "downscale_aux": bool(self.downscale_aux_var.get()),
            "aux_mask_files": list(self.aux_mask_files),
            "frames_data": {k: dict(v) for k, v in self.frames_data.items()},
            "denoised_data": {k: dict(v) for k, v in self.denoised_data.items()},
        }

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
            self.start_btn.configure(text="Start Full Batch Denoise", fg_color="#2b7a2b", hover_color="#1f5c1f", state="normal" if self.render_queue else "disabled")
            self.test_single_btn.configure(state="normal" if self.frames_data else "disabled")
            self.folder_btn.configure(state="normal")

    def on_start_or_cancel_click(self):
        if self.is_running:
            self.cancel_batch()
        else:
            self.start_batch_thread()

    def start_ram_preview_thread(self):
        if self.is_running:
            return
        if self.native_oidn is None and not self.oidn_bin.exists():
            self.log(f"ERROR: OIDN binary/DLL not found at '{self.oidn_bin}'!")
            return
        frame, ch = self.frame_var.get(), self.channel_var.get()
        if frame not in self.frames_data:
            return
        f_dict = self.frames_data[frame]
        items = []
        if ch == self.BEAUTY_SUM_KEY:
            for c_name in self._get_sum_light_channels(frame):
                raw_p = f_dict.get(c_name)
                if raw_p and raw_p.exists():
                    items.append({
                        "target": raw_p, "albedo": f_dict.get("albedo"), "normal_bump": f_dict.get("normals"),
                        "normal_pre": f_dict.get("normals_pre_bump"), "depth": f_dict.get("depth"),
                        "frame_id": frame, "pass_name": c_name,
                    })
        elif ch in f_dict and ch not in self.blacklist:
            raw_p = f_dict[ch]
            if raw_p.exists():
                items.append({
                    "target": raw_p, "albedo": f_dict.get("albedo"), "normal_bump": f_dict.get("normals"),
                    "normal_pre": f_dict.get("normals_pre_bump"), "depth": f_dict.get("depth"),
                    "frame_id": frame, "pass_name": ch,
                })
        if not items and self.render_queue:
            items = [self.render_queue[0]]
        if not items:
            self.log(f"Note: '{ch}' is an auxiliary/data pass and does not require OIDN denoising.")
            return

        roi_crop = None
        if self.roi_crop_var.get() and ch != self.BEAUTY_SUM_KEY:
            ow, oh = self.last_known_orig_dims
            roi_crop = self._get_roi_fullres_box(ow, oh)

        run_cfg = self._snapshot_run_config()
        self.cancel_event.clear()
        self._set_running_ui_state(True)
        threading.Thread(target=self.process_items, args=(items, True, roi_crop, run_cfg), daemon=True).start()

    def start_batch_thread(self):
        if self.native_oidn is None and not self.oidn_bin.exists():
            self.log(f"ERROR: OIDN binary/DLL not found at '{self.oidn_bin}'!")
            return
        run_cfg = self._snapshot_run_config()
        self.cancel_event.clear()
        self._set_running_ui_state(True)
        threading.Thread(target=self.process_items, args=(list(self.render_queue), False, None, run_cfg), daemon=True).start()

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
        alpha_ch, ds_kernel: str, sc_mode: str, sc_val: str, sharp_str: float, sharp_rad: float,
        norm_g, alb_g, dep_g, comp_attr_name: str, level_str: str, use_half_exr: bool,
        t_denoise_elapsed: float, exr_worker_threads: int, release_gpu: bool = False,
    ):
        try:
            self._save_pass_impl(
                idx, total, out_path, orig_rgb, denoised_rgb, alpha_ch, ds_kernel, sc_mode, sc_val,
                sharp_str, sharp_rad, norm_g, alb_g, dep_g, comp_attr_name, level_str,
                use_half_exr, t_denoise_elapsed, exr_worker_threads,
            )
        finally:
            if release_gpu:
                GPU_SHADER_ENGINE.release_vram_pool()

    def _save_pass_impl(
        self, idx: int, total: int, out_path: Path, orig_rgb: np.ndarray, denoised_rgb: np.ndarray,
        alpha_ch, ds_kernel: str, sc_mode: str, sc_val: str, sharp_str: float, sharp_rad: float,
        norm_g, alb_g, dep_g, comp_attr_name: str, level_str: str, use_half_exr: bool,
        t_denoise_elapsed: float, exr_worker_threads: int,
    ):
        t_save_start = time.perf_counter()
        oh, ow, _ = denoised_rgb.shape
        tw, th = DownscaleEngine.compute_target_size(ow, oh, sc_mode, sc_val)
        out_alpha = alpha_ch
        if (tw, th) != (ow, oh) or abs(sharp_str - 1.0) > 1e-4:
            denoised_rgb = DownscaleEngine.downscale(
                denoised_rgb, tw, th, ds_kernel, sharpness=sharp_str, radius=sharp_rad,
                norm_rgb=norm_g, alb_rgb=alb_g, depth_arr=dep_g,
            )
            if out_alpha is not None and (tw, th) != (ow, oh):
                out_alpha = DownscaleEngine.downscale_aux_pass(np.dstack([out_alpha] * 3), tw, th, "alpha", ds_kernel)[:, :, 0]

        write_exr_static(out_path, denoised_rgb, out_alpha, comp_attr_name, level_str, use_half_exr, worker_threads=exr_worker_threads, is_data_pass=False)
        t_save_elapsed = time.perf_counter() - t_save_start
        red_1px = estimate_noise_fast_roi(orig_rgb, denoised_rgb)
        size_kb = out_path.stat().st_size / 1024.0
        h_f, w_f, _ = denoised_rgb.shape
        self.log(f"[{idx}/{total}] Saved {w_f}×{h_f} ({size_kb:.1f} KB | Denoise: {t_denoise_elapsed:.1f}s | Save: {t_save_elapsed:.1f}s | Noise Reduction: {red_1px:+.1f}%): {out_path.name}")

    def process_items(
        self,
        items_to_process: list[dict],
        is_ram_preview_only: bool,
        roi_crop_px: tuple[int, int, int, int] | None = None,
        run_cfg: dict | None = None,
    ):
        cfg = run_cfg if run_cfg is not None else self._snapshot_run_config()
        initial_channel_before_run = cfg["initial_channel"]
        total = len(items_to_process)
        aux_mode, hdr_mode, decorr_mode = cfg["aux_mode"], cfg["hdr_mode"], cfg["decorr"]
        ds_kernel, sc_mode, sc_val = cfg["ds_kernel"], cfg["sc_mode"], cfg["sc_val"]
        sharp_str = cfg["sharp_str"]
        sharp_rad = cfg["sharp_rad"]
        use_depth_cfg = cfg["use_depth"]
        export_multilayer = (not is_ram_preview_only) and cfg["export_multilayer"]
        verbose_lvl, device_str, quality_str, filter_str = int(cfg["verbose"]), cfg["device"], cfg["quality"], cfg["filter"]
        use_half_oidn, clean_aux, inplace = (cfg["precision"] == "half"), cfg["clean_aux"], cfg["inplace"]
        skip_empty = cfg["skip_empty"]
        maxmem_str, future_args = cfg["maxmem_str"], cfg["future_args"]
        maxmem_mb = int(maxmem_str) if maxmem_str.isdigit() else -1

        comp_attr_name, _ = self.compression_map.get(cfg["comp"], ("DWAB_COMPRESSION", "20"))
        level_str = cfg["comp_level"]
        use_half_exr = cfg["exr_depth"].startswith("HALF")

        if cfg["sep_aux"]:
            aux_comp_attr, aux_comp_lvl = self.compression_map.get(cfg["aux_comp"], ("ZIP_COMPRESSION", "4"))
        else:
            aux_comp_attr, aux_comp_lvl = comp_attr_name, level_str

        use_async_save = (not is_ram_preview_only) and cfg["async_save"] and total > 1
        exr_worker_threads = max(1, CPU_THREADS // 4) if use_async_save else CPU_THREADS
        stride = 2 if decorr_mode.startswith("2x2") else (3 if decorr_mode.startswith("3x3") else 1)
        use_direct_dll = (self.native_oidn is not None) and (not future_args)
        downscale_aux_enabled = cfg["downscale_aux"]
        aux_mask_snapshot = cfg["aux_mask_files"]
        frames_snapshot = cfg["frames_data"]
        denoised_snapshot = cfg["denoised_data"]

        oidn_needs_aux = aux_mode != "None (Color Only)"
        ds_may_scale = (sc_mode != "No Downscale (100%)") or abs(sharp_str - 1.0) > 1e-4
        ds_needs_guides = self._downscale_needs_guides(ds_kernel, sharp_str)
        ds_uses_depth = use_depth_cfg and self._filter_supports_depth(ds_kernel, sharp_str)
        need_any_an_guides = oidn_needs_aux or is_ram_preview_only or (ds_may_scale and ds_needs_guides)
        need_any_depth = is_ram_preview_only or (ds_may_scale and ds_uses_depth)

        t_batch_start = time.perf_counter()
        if is_ram_preview_only and roi_crop_px is not None:
            rx0, ry0, rx1, ry1 = roi_crop_px
            mode_tag = f"ROI PREVIEW ({rx1-rx0}×{ry1-ry0} Crop)"
        elif is_ram_preview_only:
            mode_tag = "RAM PREVIEW (Full-Res Cached)"
        else:
            mode_tag = "BATCH RUN"

        self.log("\n==========================================")
        self.log(f"{mode_tag}: {total} Pass(es) | OIDN: {device_str} ({'FP16' if use_half_oidn else 'FP32'}) | Downscaler: {'GLSL GPU Shader' if GPU_SHADER_ENGINE.available else 'CPU'}")
        self.update_progress(0, total, "Running")

        save_futures = []
        ml_frame_lights: dict[str, dict[str, Path]] = {}
        ml_frame_dims: dict[str, tuple[int, int, Path]] = {}
        ml_frame_aux_scaled: dict[str, dict[str, Path]] = {}
        executor = ThreadPoolExecutor(max_workers=4) if use_async_save else None
        prefetch_executor = ThreadPoolExecutor(max_workers=1) if total > 0 else None
        prefetch_futures = {}

        try:
            if use_direct_dll:
                self.native_oidn.ensure_device(device_str, verbose_lvl)

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                tmp_alb, tmp_nrm = tmp_path / "albedo.pfm", tmp_path / "normal.pfm"
                tmp_alb_s, tmp_nrm_s = tmp_path / "albedo_s.pfm", tmp_path / "normal_s.pfm"
                tmp_hdr, tmp_out = tmp_path / "input.pfm", tmp_path / "output.pfm"
                pf_state = {"frame_id": None, "bundle": None}

                def _slice_if_roi(arr: np.ndarray | None) -> np.ndarray | None:
                    if arr is None or roi_crop_px is None:
                        return arr
                    x0, y0, x1, y1 = roi_crop_px
                    ah, aw = arr.shape[:2]
                    cx0, cy0 = max(0, min(aw - 2, x0)), max(0, min(ah - 2, y0))
                    cx1, cy1 = max(cx0 + 2, min(aw, x1)), max(cy0 + 2, min(ah, y1))
                    return np.ascontiguousarray(arr[cy0:cy1, cx0:cx1])

                def _prefetch_item(q_item: dict):
                    if self.cancel_event.is_set():
                        return None
                    orig_rgb, alpha_ch, _ = read_exr_native(q_item["target"])
                    orig_rgb = np.maximum(np.nan_to_num(orig_rgb, nan=0.0, posinf=65500.0, neginf=0.0), 0.0)
                    orig_rgb_cropped = _slice_if_roi(orig_rgb)
                    if skip_empty and roi_crop_px is None and float(np.max(orig_rgb_cropped)) <= 1e-6:
                        return {"empty": True}

                    fid = q_item["frame_id"]
                    if pf_state["frame_id"] != fid:
                        alb_g = nrm_g = dep_g = p_alb = p_nrm = None
                        if need_any_an_guides and q_item.get("albedo") and q_item["albedo"].exists():
                            alb_raw = read_exr_native(q_item["albedo"])[0]
                            alb_g = np.clip(np.nan_to_num(alb_raw, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                        chosen_nrm = (q_item.get("normal_pre") or q_item.get("normal_bump")) if aux_mode == "Albedo + Normals (Pre-Bump)" else (
                            q_item.get("normal_bump") or q_item.get("normal_pre")
                        )
                        if need_any_an_guides and chosen_nrm and chosen_nrm.exists():
                            nrm_raw = read_exr_native(chosen_nrm)[0]
                            nrm_g = np.nan_to_num(nrm_raw, nan=0.0, posinf=1.0, neginf=-1.0)
                        if need_any_depth and q_item.get("depth") and q_item["depth"].exists():
                            dep_g = normalize_depth_guide(read_exr_native(q_item["depth"])[0])
                        alb_work = _slice_if_roi(alb_g)
                        nrm_work = _slice_if_roi(nrm_g)
                        if stride > 1 and oidn_needs_aux:
                            if alb_work is not None:
                                p_alb = stride_pack(alb_work, stride)[0]
                            if nrm_work is not None:
                                p_nrm = stride_pack(nrm_work, stride)[0]
                        pf_state["frame_id"] = fid
                        pf_state["bundle"] = {
                            "frame_id": fid, "full_alb": alb_g, "full_nrm": nrm_g, "full_depth": dep_g,
                            "alb": alb_work, "nrm": nrm_work, "depth": _slice_if_roi(dep_g),
                            "oidn_has_alb": oidn_needs_aux and (alb_work is not None),
                            "oidn_has_nrm": oidn_needs_aux and (nrm_work is not None),
                            "packed_alb": p_alb, "packed_nrm": p_nrm,
                        }

                    work_rgb = orig_rgb_cropped
                    ldr_scale, use_ldr = 1.0, False
                    if hdr_mode == "HDR Clamp Fireflies":
                        work_rgb = np.minimum(work_rgb, max(float(np.percentile(work_rgb, 99.5)) * 3.0, 1e-4))
                    elif hdr_mode == "LDR Normalized":
                        ldr_scale = max(float(np.percentile(work_rgb, 99.0)), 1e-6)
                        work_rgb, use_ldr = np.clip(work_rgb / ldr_scale, 0.0, 1.0), True

                    packed_hdr = stride_pack(work_rgb, stride)[0] if stride > 1 else None
                    return {
                        "empty": False, "orig_rgb": orig_rgb_cropped, "full_shape": orig_rgb.shape[:2],
                        "alpha": _slice_if_roi(alpha_ch), "work_rgb": work_rgb,
                        "ldr_scale": ldr_scale, "use_ldr": use_ldr,
                        "packed_hdr": packed_hdr, "bundle": pf_state["bundle"],
                    }

                def _submit_prefetch(q_idx: int):
                    if prefetch_executor is not None and 0 <= q_idx < total and q_idx not in prefetch_futures and not self.cancel_event.is_set():
                        prefetch_futures[q_idx] = prefetch_executor.submit(_prefetch_item, items_to_process[q_idx])

                def _queue_lookahead(cur_idx: int):
                    _submit_prefetch(cur_idx + 1)
                    if cur_idx + 2 < total:
                        if items_to_process[cur_idx + 2]["frame_id"] == items_to_process[cur_idx + 1]["frame_id"] or items_to_process[cur_idx + 1]["frame_id"] == items_to_process[cur_idx]["frame_id"]:
                            _submit_prefetch(cur_idx + 2)

                _submit_prefetch(0)
                cached_frame_id = None

                for zero_i, item in enumerate(items_to_process):
                    idx = zero_i + 1
                    if self.cancel_event.is_set():
                        self.log("--- Cancelled by user ---")
                        break
                    if executor is not None and len(save_futures) >= 6:
                        save_futures.pop(0).result()

                    target = item["target"]
                    out_path = target.parent / f"{target.stem}_denoised.exr"
                    self.update_progress(idx - 1, total, f"Pass {idx}/{total}")
                    self._post(lambda i=idx, t=total, n=item["pass_name"]: self.start_viewport_spinner(f"Denoising [{i}/{t}] {n}..."))

                    try:
                        _submit_prefetch(zero_i)
                        _queue_lookahead(zero_i)
                        prep = prefetch_futures.pop(zero_i).result()
                        if prep is None or self.cancel_event.is_set():
                            break
                        if prep["empty"]:
                            self.empty_passes.setdefault(item["frame_id"], set()).add(item["pass_name"])
                            self.log(f"[{idx}/{total}] ⚫ Skipped Empty (100% Black) Pass: {target.name}")
                            self.update_progress(idx, total, "Done" if idx == total else f"Pass {idx}/{total}")
                            continue

                        t_denoise_start = time.perf_counter()
                        orig_rgb, alpha_ch = prep["orig_rgb"], prep["alpha"]
                        work_rgb, ldr_scale, use_ldr = prep["work_rgb"], prep["ldr_scale"], prep["use_ldr"]
                        packed_hdr, bundle = prep["packed_hdr"], prep["bundle"]
                        has_alb, has_nrm = bundle["oidn_has_alb"], bundle["oidn_has_nrm"]
                        alb_rgb_ram, nrm_rgb_ram, dep_ram = bundle["alb"], bundle["nrm"], bundle["depth"]
                        packed_alb_ram, packed_nrm_ram = bundle["packed_alb"], bundle["packed_nrm"]

                        if cached_frame_id != item["frame_id"] or roi_crop_px is not None:
                            self._store_guides(item["frame_id"], bundle["full_nrm"], bundle["full_alb"], bundle["full_depth"])
                            if use_direct_dll:
                                if stride == 1:
                                    self.native_oidn.upload_aux_to_vram(alb_rgb_ram if has_alb else None, nrm_rgb_ram if has_nrm else None, use_half_oidn)
                            else:
                                if has_alb:
                                    write_pfm(alb_rgb_ram, tmp_alb)
                                    if stride > 1 and packed_alb_ram is not None:
                                        write_pfm(packed_alb_ram, tmp_alb_s)
                                if has_nrm:
                                    write_pfm(nrm_rgb_ram, tmp_nrm)
                                    if stride > 1 and packed_nrm_ram is not None:
                                        write_pfm(packed_nrm_ram, tmp_nrm_s)
                            cached_frame_id = item["frame_id"]

                        if use_direct_dll:
                            if stride > 1:
                                self.native_oidn.upload_aux_to_vram(packed_alb_ram if has_alb else None, packed_nrm_ram if has_nrm else None, use_half_oidn)
                                den_packed = self.native_oidn.denoise_in_vram(packed_hdr, has_alb, has_nrm, filter_str, quality_str, use_half_oidn, not use_ldr, clean_aux, inplace, maxmem_mb)
                                work_rgb = stride_unpack(den_packed, work_rgb, stride)
                                self.native_oidn.upload_aux_to_vram(alb_rgb_ram if has_alb else None, nrm_rgb_ram if has_nrm else None, use_half_oidn)
                            denoised_rgb = self.native_oidn.denoise_in_vram(work_rgb, has_alb, has_nrm, filter_str, quality_str, use_half_oidn, not use_ldr, clean_aux, inplace, maxmem_mb)
                            if use_ldr:
                                denoised_rgb = denoised_rgb * ldr_scale
                        else:
                            if stride > 1:
                                write_pfm(packed_hdr, tmp_hdr)
                                if tmp_out.exists():
                                    tmp_out.unlink()
                                rc, _, _ = self.run_oidn_cli(self.build_oidn_cmd(tmp_hdr, tmp_out, tmp_alb_s if has_alb else None, tmp_nrm_s if has_nrm else None, use_ldr, run_cfg=cfg))
                                if rc == 0 and tmp_out.exists():
                                    work_rgb = stride_unpack(read_pfm(tmp_out), work_rgb, stride)
                            if tmp_out.exists():
                                tmp_out.unlink()
                            write_pfm(work_rgb, tmp_hdr)
                            ret_code, stdout_str, stderr_str = self.run_oidn_cli(self.build_oidn_cmd(tmp_hdr, tmp_out, tmp_alb if has_alb else None, tmp_nrm if has_nrm else None, use_ldr, run_cfg=cfg))
                            if self.cancel_event.is_set():
                                break
                            if ret_code != 0 or not tmp_out.exists():
                                self.log(f"ERROR (Exit {ret_code}):\n{stdout_str}\n{stderr_str}")
                                continue
                            denoised_rgb = read_pfm(tmp_out) * (ldr_scale if use_ldr else 1.0)
                        t_denoise_elapsed = time.perf_counter() - t_denoise_start

                        self._invalidate_beauty_sum(item["frame_id"])

                        if is_ram_preview_only:
                            ram_key = (item["frame_id"], item["pass_name"])
                            red_1px = estimate_noise_fast_roi(orig_rgb, denoised_rgb)
                            oh, ow, _ = denoised_rgb.shape
                            if roi_crop_px is not None:
                                self.ram_roi_denoised[ram_key] = {"roi_px": roi_crop_px, "rgb": denoised_rgb}
                                self.log(f"✂ [ROI Preview] Denoised '{item['pass_name']}' ({ow}×{oh}) in {t_denoise_elapsed:.2f}s (Noise Reduction: {red_1px:+.1f}%).")
                            else:
                                self.ram_denoised_fullres[ram_key] = denoised_rgb
                                self.ram_denoised_fullres.move_to_end(ram_key)
                                max_slots = max(16, len(self.frames_data.get(item["frame_id"], {})))
                                while len(self.ram_denoised_fullres) > max_slots:
                                    self.ram_denoised_fullres.popitem(last=False)
                                self.log(f"⚡ [RAM Preview] Denoised '{item['pass_name']}' at 100% ({ow}×{oh}) in {t_denoise_elapsed:.2f}s (Noise Reduction: {red_1px:+.1f}%). Press Hotkeys or hold [M]!")
                        else:
                            oh, ow, _ = denoised_rgb.shape
                            tw, th = DownscaleEngine.compute_target_size(ow, oh, sc_mode, sc_val)
                            will_ds = (tw, th) != (ow, oh) or abs(sharp_str - 1.0) > 1e-4
                            ds_nrm = nrm_rgb_ram if (will_ds and ds_needs_guides) else None
                            ds_alb = alb_rgb_ram if (will_ds and ds_needs_guides) else None
                            ds_dep = dep_ram if (will_ds and ds_uses_depth) else None
                            args_tup = (
                                idx, total, out_path, orig_rgb, denoised_rgb, alpha_ch, ds_kernel, sc_mode, sc_val,
                                sharp_str, sharp_rad, ds_nrm, ds_alb, ds_dep,
                                comp_attr_name, level_str, use_half_exr, t_denoise_elapsed, exr_worker_threads,
                            )
                            if executor is not None:
                                save_futures.append(executor.submit(self._save_pass_worker, *args_tup, release_gpu=True))
                            else:
                                self._save_pass_worker(*args_tup)
                            if export_multilayer:
                                ml_frame_lights.setdefault(item["frame_id"], {})[item["pass_name"]] = out_path
                                ml_frame_dims.setdefault(item["frame_id"], (tw, th, target.parent))

                        self.update_progress(idx, total, "Done" if idx == total else f"Pass {idx}/{total}")
                    except Exception as e:
                        self.log(f"System error on {target.name}: {str(e)}")

                pf_state["bundle"] = None
                for fut in save_futures:
                    if self.cancel_event.is_set():
                        break
                    fut.result()

                if not is_ram_preview_only and not self.cancel_event.is_set() and sc_mode != "No Downscale (100%)" and downscale_aux_enabled:
                    self.log(f"--- Scaling Aux/Mask/ID EXRs (Codec: {aux_comp_attr}) ---")
                    self._post(lambda: self.start_viewport_spinner("Scaling Aux & Mask passes..."))
                    for aux_info in aux_mask_snapshot:
                        if self.cancel_event.is_set():
                            break
                        aux_path, p_type = aux_info["path"], aux_info["type"]
                        fid_aux = aux_info.get("frame_id", "Scene")
                        try:
                            t_aux_start = time.perf_counter()
                            is_id_map = p_type in ("material_id", "object_id")
                            a_rgb, a_alpha, _ = read_exr_native(aux_path, preserve_uint32=is_id_map)
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
                            write_exr_static(out_aux, a_down, a_al_down, aux_comp_attr, aux_comp_lvl, use_half_this_aux, worker_threads=CPU_THREADS, is_data_pass=(p_type != "albedo"))
                            ml_frame_aux_scaled.setdefault(fid_aux, {})[p_type] = out_aux
                            self.log(f" -> Aux [{p_type}] {tw}×{th} ({out_aux.stat().st_size / 1024.0:.1f} KB | {time.perf_counter() - t_aux_start:.1f}s): {out_aux.name}")
                        except Exception as ex:
                            self.log(f"Aux scale warning ({aux_path.name}): {ex}")

                if export_multilayer and not self.cancel_event.is_set() and ml_frame_lights:
                    self.log("--- Exporting Multi-Layer EXR(s) ---")
                    self._post(lambda: self.start_viewport_spinner("Writing Multi-Layer EXR..."))
                    for fid in sorted(ml_frame_lights.keys()):
                        if self.cancel_event.is_set():
                            break
                        tw, th, out_dir = ml_frame_dims[fid]
                        # Merge newly denoised + previously denoised light passes for complete composite
                        all_lights = dict(denoised_snapshot.get(fid, {}))
                        all_lights.update(ml_frame_lights[fid])
                        # Prefer already-scaled aux passes when available, else raw aux passes
                        all_aux = {}
                        for p_k, p_path in frames_snapshot.get(fid, {}).items():
                            if p_k in self.blacklist:
                                all_aux[p_k] = p_path
                        all_aux.update(ml_frame_aux_scaled.get(fid, {}))
                        ml_out = out_dir / f"{fid}_multilayer.exr"
                        layers_written = write_multilayer_exr(
                            ml_out, all_lights, all_aux, tw, th, ds_kernel,
                            comp_attr_name, level_str, use_half_exr, worker_threads=CPU_THREADS,
                        )
                        if layers_written:
                            self.log(f" 📦 Multi-Layer EXR ({len(layers_written)} layers | {ml_out.stat().st_size / 1024.0:.1f} KB): {ml_out.name}")
        finally:
            if prefetch_executor is not None:
                try:
                    prefetch_executor.shutdown(wait=True, cancel_futures=True)
                except TypeError:
                    prefetch_executor.shutdown(wait=True)
            if executor is not None:
                executor.shutdown(wait=True)
            if not is_ram_preview_only:
                GPU_SHADER_ENGINE.release_vram_pool()
                if self.native_oidn is not None:
                    self.native_oidn.release_buffers()

            if not self.cancel_event.is_set():
                self.log(f"--- {'RAM Preview' if is_ram_preview_only else 'Run'} Finished in {time.perf_counter() - t_batch_start:.1f}s ---")

            def _finish_run_ui():
                self._set_running_ui_state(False)
                if is_ram_preview_only:
                    self.show_denoised_default = True
                    self.swap_default_btn.configure(text="Hold: Raw", fg_color="#2b7a2b")
                    self.channel_selector.refresh_label()
                    self.reload_current_preview()
                else:
                    self.parse_indigo_passes(select_channel=initial_channel_before_run)

            self._post(_finish_run_ui)
