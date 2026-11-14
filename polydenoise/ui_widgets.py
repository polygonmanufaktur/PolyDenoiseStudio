# SPDX-License-Identifier: MIT
"""Reusable CustomTkinter UI widgets: vector icons, delayed hover tooltips, searchable dropdowns."""

from __future__ import annotations

import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageDraw


def create_minimal_icon(kind: str, size: int = 18, color: str = "#dcdcdc") -> ctk.CTkImage:
    """Render anti-aliased minimal monochrome UI icons using Pillow."""
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


class CTkToolTip:
    """Delayed hover tooltip supporting dynamic text updates via set_text()."""

    def __init__(self, widget, text: str, delay: int = 320):
        self.widget, self.text, self.delay = widget, text, delay
        self.tip_window = self.after_id = self._label = None
        targets = [widget] + (list(widget.winfo_children()) if hasattr(widget, "winfo_children") else [])
        for target_w in targets:
            target_w.bind("<Enter>", self._schedule, add="+")
            target_w.bind("<Leave>", self._hide, add="+")
            target_w.bind("<ButtonPress>", self._hide, add="+")

    def set_text(self, text: str):
        self.text = text
        if self._label is not None and self.tip_window is not None:
            try:
                self._label.configure(text=text)
            except Exception:
                pass

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
        self._label = tk.Label(
            frame,
            text=self.text,
            justify="left",
            background="#1e1e1e",
            foreground="#e8e8e8",
            font=("Segoe UI", 9),
            wraplength=360,
            padx=8,
            pady=6,
        )
        self._label.pack()

    def _hide(self, _=None):
        self._unschedule()
        self._label = None
        if self.tip_window is not None:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None


class ScrollableDropdown(ctk.CTkFrame):
    """Searchable scrollable dropdown with previous/next step buttons."""

    def __init__(self, master, variable: ctk.StringVar, values: list, command=None, width: int = 185, display_formatter=None):
        super().__init__(master, fg_color="transparent")
        self.variable, self.values, self.command = variable, list(values), command
        self.display_formatter, self.popup, self.btn_width = display_formatter, None, width
        self.prev_btn = ctk.CTkButton(
            self, text="◀", width=24, height=28, fg_color="#3a3a3a", hover_color="#505050", command=lambda: self.step(-1)
        )
        self.prev_btn.pack(side="left", padx=(0, 2))
        self.main_btn = ctk.CTkButton(
            self,
            text=self._format_btn_text(self.variable.get()),
            width=width,
            height=28,
            anchor="w",
            fg_color="#1f538d",
            hover_color="#14375e",
            command=self.toggle_popup,
        )
        self.main_btn.pack(side="left", padx=0)
        self.next_btn = ctk.CTkButton(
            self, text="▶", width=24, height=28, fg_color="#3a3a3a", hover_color="#505050", command=lambda: self.step(1)
        )
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
        listbox = tk.Listbox(
            list_frame,
            bg="#1e1e1e",
            fg="#e8e8e8",
            selectbackground="#1f6aa5",
            selectforeground="#ffffff",
            font=("Consolas", 10),
            highlightthickness=0,
            bd=0,
            activestyle="none",
            yscrollcommand=scrollbar.set,
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=listbox.yview)

        filtered_items: list[str] = []

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
        pop.bind(
            "<FocusOut>",
            lambda _: self.close_popup() if (self.popup and self.popup.winfo_exists() and self.popup.focus_get() is None) else None,
        )
        search_entry.focus_set()
