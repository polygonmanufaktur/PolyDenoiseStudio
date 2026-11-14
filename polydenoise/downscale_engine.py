# SPDX-License-Identifier: MIT
"""GPU GLSL 330 shader downscaler and exact NumPy/OpenCV CPU fallback engine."""

from __future__ import annotations

from collections import OrderedDict
import threading
import time
import weakref

import cv2
import numpy as np

from .filter_registry import FilterSpec, get_filter_spec
from .shaders_glsl import (
    COLOR_SPACE_LIB,
    FRAG_DPID_BODY,
    FRAG_EWA_BODY,
    FRAG_GUIDED_BODY,
    FRAG_H_BODY,
    FRAG_PASS2_BODY,
    FRAG_TONEMAP_BODY,
    FRAG_V_BODY,
    KERNEL_LIB,
    VERT_SRC,
)

try:
    import moderngl
except ImportError:
    moderngl = None

ROBIDOUX_SHARP_B = 6.0 / (13.0 + 7.0 * np.sqrt(2.0))
ROBIDOUX_SHARP_C = (1.0 - ROBIDOUX_SHARP_B) * 0.5

_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 32.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 128.0
_PQ_C3 = 2392.0 / 128.0
_PQ_EXT_SLOPE = 0.10466623306


class GPUShaderDownscaler:
    """Thread-safe OpenGL 3.3 / ModernGL HDR downscaler with persistent per-thread VRAM pools."""

    def __init__(self):
        self._tls = threading.local()
        self.renderer_name = "CPU Fallback"
        self.is_supported = False
        self.last_gpu_ms = 0.0
        self.show_mask_xray = False
        self.init_error = ""
        if moderngl is not None:
            try:
                ctx, _ = self._get_thread_resources()
                if ctx is not None:
                    self.renderer_name = ctx.info.get("GL_RENDERER", "OpenGL GPU")
                    self.is_supported = True
            except Exception:
                self.is_supported = False

    def _get_thread_resources(self):
        if getattr(self._tls, "initialized", False):
            return self._tls.ctx, self._tls.res
        self._tls.initialized = True
        self._tls.vram_pool = None
        self._tls.pool_dims = None
        self._tls.last_src_ref = None
        self._tls.last_norm_ref = None
        self._tls.last_alb_ref = None
        self._tls.last_depth_ref = None
        self._tls.tm_pool = None
        self._tls.tm_dims = None
        self._tls.last_tm_ref = None
        try:
            ctx = moderngl.create_standalone_context()
            hdr_pass1 = "#version 330\n" + KERNEL_LIB
            sources_pass1 = {
                "h": FRAG_H_BODY,
                "v": FRAG_V_BODY,
                "ewa": FRAG_EWA_BODY,
                "dpid": FRAG_DPID_BODY,
                "guided": FRAG_GUIDED_BODY,
            }
            progs = {
                k: ctx.program(vertex_shader=VERT_SRC, fragment_shader=hdr_pass1 + body)
                for k, body in sources_pass1.items()
            }
            progs["p2"] = ctx.program(
                vertex_shader=VERT_SRC, fragment_shader="#version 330\n" + COLOR_SPACE_LIB + FRAG_PASS2_BODY
            )
            progs["tm"] = ctx.program(
                vertex_shader=VERT_SRC, fragment_shader="#version 330\n" + FRAG_TONEMAP_BODY
            )
            quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
            vbo = ctx.buffer(quad.tobytes())
            vaos = {k: ctx.simple_vertex_array(p, vbo, "in_pos") for k, p in progs.items()}
            self._tls.ctx, self._tls.res = ctx, {"p": progs, "v": vaos}
            return self._tls.ctx, self._tls.res
        except Exception as ex:
            self._tls.ctx = self._tls.res = None
            self.init_error = str(ex)
            return None, None

    def release_vram_pool(self):
        for attr in ("vram_pool", "tm_pool"):
            pool = getattr(self._tls, attr, None)
            if pool:
                for obj in pool.values():
                    if obj is not None:
                        try:
                            obj.release()
                        except Exception:
                            pass
        self._tls.vram_pool = None
        self._tls.pool_dims = None
        self._tls.last_src_ref = None
        self._tls.last_norm_ref = None
        self._tls.last_alb_ref = None
        self._tls.last_depth_ref = None
        self._tls.tm_pool = None
        self._tls.tm_dims = None
        self._tls.last_tm_ref = None

    def _ensure_vram_pool(self, ctx, w: int, h: int, target_w: int, target_h: int):
        dims = (w, h, target_w, target_h)
        if getattr(self._tls, "pool_dims", None) == dims and getattr(self._tls, "vram_pool", None) is not None:
            return self._tls.vram_pool
        self.release_vram_pool()
        tex = lambda size, comps: ctx.texture(size, comps, dtype="f4")
        pool = {
            "tex_src": tex((w, h), 3),
            "tex_norm": tex((w, h), 3),
            "tex_alb": tex((w, h), 3),
            "tex_depth": tex((w, h), 1),
            "tex_dummy": ctx.texture((1, 1), 3, np.zeros((1, 1, 3), dtype="f4").tobytes(), dtype="f4"),
            "tex_dummy1": ctx.texture((1, 1), 1, np.zeros((1, 1), dtype="f4").tobytes(), dtype="f4"),
            "tex_h0": tex((target_w, h), 4),
            "tex_dec": tex((target_w, target_h), 4),
            "tex_min": tex((target_w, target_h), 4),
            "tex_max": tex((target_w, target_h), 4),
            "tex_out": tex((target_w, target_h), 4),
        }
        for k in ("tex_dec", "tex_norm", "tex_alb", "tex_depth"):
            pool[k].filter = (moderngl.LINEAR, moderngl.LINEAR)
            pool[k].repeat_x = pool[k].repeat_y = False
        for k in ("tex_src", "tex_h0", "tex_min", "tex_max", "tex_out"):
            pool[k].filter = (moderngl.NEAREST, moderngl.NEAREST)
        pool["tex_src"].repeat_x = pool["tex_src"].repeat_y = False
        pool["fbo_h"] = ctx.framebuffer(color_attachments=[pool["tex_h0"]])
        pool["fbo_v"] = ctx.framebuffer(color_attachments=[pool["tex_dec"], pool["tex_min"], pool["tex_max"]])
        pool["fbo_out"] = ctx.framebuffer(color_attachments=[pool["tex_out"]])
        self._tls.vram_pool = pool
        self._tls.pool_dims = dims
        return pool

    def _ensure_tm_pool(self, ctx, w: int, h: int):
        dims = (w, h)
        if getattr(self._tls, "tm_dims", None) == dims and getattr(self._tls, "tm_pool", None) is not None:
            return self._tls.tm_pool
        old_tm = getattr(self._tls, "tm_pool", None)
        if old_tm:
            for obj in old_tm.values():
                if obj is not None:
                    try:
                        obj.release()
                    except Exception:
                        pass
        tex_in = ctx.texture((w, h), 3, dtype="f4")
        tex_in.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex_in.repeat_x = tex_in.repeat_y = False
        tex_out = ctx.texture((w, h), 4, dtype="f1")
        tex_out.filter = (moderngl.NEAREST, moderngl.NEAREST)
        fbo_tm = ctx.framebuffer(color_attachments=[tex_out])
        pool = {"tex_in": tex_in, "tex_out": tex_out, "fbo_tm": fbo_tm}
        self._tls.tm_pool = pool
        self._tls.tm_dims = dims
        self._tls.last_tm_ref = None
        return pool

    @property
    def available(self) -> bool:
        return self.is_supported

    @staticmethod
    def _u(prog, name: str, value):
        try:
            prog[name].value = value
        except KeyError:
            pass

    @staticmethod
    def _upload(tex, arr: np.ndarray):
        tex.write(np.ascontiguousarray(arr, dtype="f4"))

    @staticmethod
    def _same(ref, arr) -> bool:
        return ref is not None and ref() is arr

    def _decimate(
        self, ctx, res, pool, w: int, h: int, tw: int, th: int,
        spec: FilterSpec, dst_fbo, has_a: int, has_n: int, has_d: int,
    ):
        P, V = res["p"], res["v"]
        src_res, dst_res = (float(w), float(h)), (float(tw), float(th))
        if spec.special == 2:
            prog = P["guided"]
            dst_fbo.use()
            ctx.viewport = (0, 0, tw, th)
            pool["tex_src"].use(location=0)
            (pool["tex_alb"] if has_a else pool["tex_dummy"]).use(location=1)
            (pool["tex_norm"] if has_n else pool["tex_dummy"]).use(location=2)
            (pool["tex_depth"] if has_d else pool["tex_dummy1"]).use(location=3)
            for n, v in (
                ("u_src", 0), ("u_alb", 1), ("u_nrm", 2), ("u_depth", 3),
                ("u_src_res", src_res), ("u_dst_res", dst_res),
                ("u_has_alb", int(has_a)), ("u_has_nrm", int(has_n)), ("u_has_depth", int(has_d)),
                ("u_karis_k", 0.25),
            ):
                self._u(prog, n, v)
            V["guided"].render(moderngl.TRIANGLE_STRIP)
            return

        if spec.special == 1 or spec.ewa_kind >= 0:
            key = "dpid" if spec.special == 1 else "ewa"
            prog = P[key]
            dst_fbo.use()
            ctx.viewport = (0, 0, tw, th)
            pool["tex_src"].use(location=0)
            for n, v in (
                ("u_src", 0), ("u_src_res", src_res), ("u_dst_res", dst_res),
                ("u_use_karis", spec.use_karis), ("u_ewa_kind", max(0, spec.ewa_kind)),
            ):
                self._u(prog, n, v)
            V[key].render(moderngl.TRIANGLE_STRIP)
            return

        ph, pv = P["h"], P["v"]
        pool["fbo_h"].use()
        ctx.viewport = (0, 0, tw, h)
        pool["tex_src"].use(location=0)
        for n, v in (
            ("u_src", 0), ("u_src_res", src_res), ("u_dst_res", dst_res),
            ("u_ktype", spec.ktype), ("u_use_karis", spec.use_karis), ("u_colorspace", spec.colorspace),
        ):
            self._u(ph, n, v)
        V["h"].render(moderngl.TRIANGLE_STRIP)

        dst_fbo.use()
        ctx.viewport = (0, 0, tw, th)
        pool["tex_h0"].use(location=0)
        pool["tex_src"].use(location=1)
        for n, v in (
            ("u_h0", 0), ("u_src", 1), ("u_src_res", src_res), ("u_dst_res", dst_res),
            ("u_ktype", spec.ktype), ("u_use_karis", spec.use_karis), ("u_colorspace", spec.colorspace),
        ):
            self._u(pv, n, v)
        V["v"].render(moderngl.TRIANGLE_STRIP)

    def downscale_gpu(
        self,
        rgb: np.ndarray,
        target_w: int,
        target_h: int,
        mode_label: str,
        sharpness: float = 1.0,
        radius: float = 1.0,
        norm_rgb: np.ndarray | None = None,
        alb_rgb: np.ndarray | None = None,
        depth_arr: np.ndarray | None = None,
    ) -> np.ndarray:
        ctx, res = self._get_thread_resources()
        if ctx is None:
            raise RuntimeError("Thread OpenGL context unavailable")

        t0 = time.perf_counter()
        h, w, _ = rgb.shape
        spec = get_filter_spec(mode_label)
        pool = self._ensure_vram_pool(ctx, w, h, target_w, target_h)

        if not self._same(self._tls.last_src_ref, rgb):
            self._upload(pool["tex_src"], rgb)
            self._tls.last_src_ref = weakref.ref(rgb)

        has_n = has_a = has_d = 0
        if norm_rgb is not None and norm_rgb.shape[:2] == (h, w):
            if not self._same(self._tls.last_norm_ref, norm_rgb):
                self._upload(pool["tex_norm"], norm_rgb)
                self._tls.last_norm_ref = weakref.ref(norm_rgb)
            has_n = 1
        if alb_rgb is not None and alb_rgb.shape[:2] == (h, w):
            if not self._same(self._tls.last_alb_ref, alb_rgb):
                self._upload(pool["tex_alb"], alb_rgb)
                self._tls.last_alb_ref = weakref.ref(alb_rgb)
            has_a = 1
        if depth_arr is not None and depth_arr.shape[:2] == (h, w):
            if not self._same(self._tls.last_depth_ref, depth_arr):
                d_upload = depth_arr[:, :, 0] if depth_arr.ndim == 3 else depth_arr
                self._upload(pool["tex_depth"], d_upload)
                self._tls.last_depth_ref = weakref.ref(depth_arr)
            has_d = 1

        self._decimate(ctx, res, pool, w, h, target_w, target_h, spec, pool["fbo_v"], has_a, has_n, has_d)

        p2 = res["p"]["p2"]
        pool["fbo_out"].use()
        ctx.viewport = (0, 0, target_w, target_h)
        pool["tex_dec"].use(location=0)
        pool["tex_min"].use(location=1)
        pool["tex_max"].use(location=2)
        (pool["tex_norm"] if has_n else pool["tex_dummy"]).use(location=3)
        (pool["tex_alb"] if has_a else pool["tex_dummy"]).use(location=4)
        (pool["tex_depth"] if has_d else pool["tex_dummy1"]).use(location=5)
        for n, v in (
            ("u_dec", 0), ("u_min", 1), ("u_max", 2), ("u_norm", 3), ("u_alb", 4), ("u_depth", 5),
            ("u_dst_res", (float(target_w), float(target_h))),
            ("u_sharp_mode", spec.sharp_mode),
            ("u_ar_mode", spec.ar_mode),
            ("u_is_rev_karis", 1 if (spec.use_karis == 2 and spec.colorspace == 0) else 0),
            ("u_colorspace", spec.colorspace),
            ("u_sharp_str", float(sharpness)),
            ("u_sharp_rad", float(radius)),
            ("u_has_norm", has_n),
            ("u_has_alb", has_a),
            ("u_has_depth", has_d),
            ("u_show_mask", 1 if self.show_mask_xray else 0),
        ):
            self._u(p2, n, v)
        res["v"]["p2"].render(moderngl.TRIANGLE_STRIP)

        out_arr = np.empty((target_h, target_w, 3), dtype=np.float32)
        pool["fbo_out"].read_into(out_arr, components=3, dtype="f4")
        self.last_gpu_ms = (time.perf_counter() - t0) * 1000.0
        return out_arr

    def tonemap_gpu(
        self,
        linear_rgb: np.ndarray,
        ev_val: float = 0.0,
        cache_immutable: bool = False,
    ) -> np.ndarray:
        """Fast GPU ACES Filmic + 2.2 Gamma + IGN Dither returning uint8 RGB (~4 ms @ 2K)."""
        ctx, res = self._get_thread_resources()
        if ctx is None:
            raise RuntimeError("Thread OpenGL context unavailable")
        h, w, c = linear_rgb.shape
        if c != 3:
            raise ValueError("tonemap_gpu expects 3-channel RGB array")
        pool = self._ensure_tm_pool(ctx, w, h)
        if not (cache_immutable and self._same(self._tls.last_tm_ref, linear_rgb)):
            self._upload(pool["tex_in"], linear_rgb)
            if cache_immutable:
                try:
                    self._tls.last_tm_ref = weakref.ref(linear_rgb)
                except TypeError:
                    self._tls.last_tm_ref = None
            else:
                self._tls.last_tm_ref = None
        ptm = res["p"]["tm"]
        pool["fbo_tm"].use()
        ctx.viewport = (0, 0, w, h)
        pool["tex_in"].use(location=0)
        self._u(ptm, "u_src", 0)
        self._u(ptm, "u_src_res", (float(w), float(h)))
        self._u(ptm, "u_gain", float(2.0 ** ev_val))
        res["v"]["tm"].render(moderngl.TRIANGLE_STRIP)
        out_u8 = np.empty((h, w, 3), dtype=np.uint8)
        pool["fbo_tm"].read_into(out_u8, components=3, dtype="f1")
        return out_u8


GPU_SHADER_ENGINE = GPUShaderDownscaler()


# ================= MATH & COLORSPACE HELPERS (CPU FALLBACK PARITY) =================
def _bessel_i0(z: np.ndarray) -> np.ndarray:
    z2_4 = (np.asarray(z, dtype=np.float64) ** 2) * 0.25
    res, term = np.ones_like(z2_4), np.ones_like(z2_4)
    for k in range(1, 13):
        term = term * z2_4 / (k * k)
        res = res + term
    return res


def _bessel_j1(z: np.ndarray) -> np.ndarray:
    half_z = np.asarray(z, dtype=np.float64) * 0.5
    z2_4 = -(half_z ** 2)
    term, res = half_z.copy(), half_z.copy()
    for k in range(1, 20):
        term = term * z2_4 / (k * (k + 1))
        res = res + term
    return res


def _jinc_j1_zero(u: np.ndarray) -> np.ndarray:
    arg = np.abs(np.asarray(u, dtype=np.float64)) * 3.8317059702075123
    return np.where(arg < 1e-8, 1.0, 2.0 * _bessel_j1(arg) / np.maximum(arg, 1e-12))


def _jinc_pi(r: np.ndarray) -> np.ndarray:
    arg = np.abs(np.asarray(r, dtype=np.float64)) * np.pi
    return np.where(arg < 1e-8, 1.0, 2.0 * _bessel_j1(arg) / np.maximum(arg, 1e-12))


def _scbrt(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.cbrt(np.abs(x))


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    c = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    r, g, b = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    l_ = _scbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = _scbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = _scbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack((L, A, B), axis=2).astype(np.float32)


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, A, B = lab[:, :, 0].astype(np.float64), lab[:, :, 1].astype(np.float64), lab[:, :, 2].astype(np.float64)
    l_ = L + 0.3963377774 * A + 0.2158037573 * B
    m_ = L - 0.1055613458 * A - 0.0638541728 * B
    s_ = L - 0.0894841775 * A - 1.2914855480 * B
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return np.maximum(np.stack((r, g, b), axis=2), 0.0).astype(np.float32)


def _pq_oetf_cgi(c_lin: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(c_lin, dtype=np.float64), 0.0) * 0.01
    y_lo = np.minimum(y, 1.0)
    ym = np.power(np.maximum(y_lo, 0.0), _PQ_M1)
    p_lo = np.power((_PQ_C1 + _PQ_C2 * ym) / (1.0 + _PQ_C3 * ym), _PQ_M2)
    p_hi = 1.0 + _PQ_EXT_SLOPE * np.log(np.maximum(y, 1.0))
    return np.where(y <= 1.0, p_lo, p_hi)


def _pq_eotf_cgi(p: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    p_lo = np.minimum(p, 1.0)
    pm = np.power(p_lo, 1.0 / _PQ_M2)
    num = np.maximum(pm - _PQ_C1, 0.0)
    den = np.maximum(_PQ_C2 - _PQ_C3 * pm, 1e-12)
    y_lo = 100.0 * np.power(num / den, 1.0 / _PQ_M1)
    y_hi = 100.0 * np.exp(np.maximum(p - 1.0, 0.0) / _PQ_EXT_SLOPE)
    return np.where(p <= 1.0, y_lo, y_hi)


def rgb_to_ictcp(rgb: np.ndarray) -> np.ndarray:
    c = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    r, g, b = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    lp = _pq_oetf_cgi(0.2958093546 * r + 0.6231013045 * g + 0.0810893409 * b)
    mp = _pq_oetf_cgi(0.1562489762 * r + 0.7273257250 * g + 0.1164252989 * b)
    sp = _pq_oetf_cgi(0.0351356628 * r + 0.1565747272 * g + 0.8082896101 * b)
    I = 0.5 * lp + 0.5 * mp
    Ct = (6610.0 * lp - 13613.0 * mp + 7003.0 * sp) / 4096.0
    Cp = (17933.0 * lp - 17390.0 * mp - 543.0 * sp) / 4096.0
    return np.stack((I, Ct, Cp), axis=2).astype(np.float32)


def ictcp_to_rgb(ic: np.ndarray) -> np.ndarray:
    I, Ct, Cp = ic[:, :, 0].astype(np.float64), ic[:, :, 1].astype(np.float64), ic[:, :, 2].astype(np.float64)
    l = _pq_eotf_cgi(I + 0.0086090370 * Ct + 0.1110296250 * Cp)
    m = _pq_eotf_cgi(I - 0.0086090370 * Ct - 0.1110296250 * Cp)
    s = _pq_eotf_cgi(I + 0.5600313357 * Ct - 0.3206271749 * Cp)
    r = +6.1733627411 * l - 5.3203805675 * m + 0.1470178266 * s
    g = -1.3243115001 * l + 2.5602264209 * m - 0.2359149210 * s
    b = -0.0118169025 * l - 0.2646720350 * m + 1.2764889376 * s
    return np.maximum(np.stack((r, g, b), axis=2), 0.0).astype(np.float32)


def _eval_keys_cubic(ax: np.ndarray, b: float, c: float) -> np.ndarray:
    w = np.zeros_like(ax, dtype=np.float64)
    m1, m2 = ax < 1.0, (ax >= 1.0) & (ax < 2.0)
    x1, x2 = ax[m1], ax[m2]
    w[m1] = ((12.0 - 9.0 * b - 6.0 * c) * (x1 ** 3) + (-18.0 + 12.0 * b + 6.0 * c) * (x1 ** 2) + (6.0 - 2.0 * b)) / 6.0
    w[m2] = ((-b - 6.0 * c) * (x2 ** 3) + (6.0 * b + 30.0 * c) * (x2 ** 2) + (-12.0 * b - 48.0 * c) * x2 + (8.0 * b + 24.0 * c)) / 6.0
    return w


class DownscaleEngine:
    K_MKS2021_SHARP = np.array([-1.0, 6.0, -35.0, 204.0, -35.0, 6.0, -1.0], dtype=np.float32) / 144.0
    _POLYPHASE_CACHE: OrderedDict = OrderedDict()

    @staticmethod
    def compute_target_size(orig_w: int, orig_h: int, scale_mode: str, scale_val_str: str) -> tuple[int, int]:
        if scale_mode == "No Downscale (100%)":
            return orig_w, orig_h
        try:
            val = float(str(scale_val_str).strip().replace(",", "."))
        except ValueError:
            val = 50.0
        if scale_mode == "Scale %":
            factor = float(np.clip(val / 100.0, 0.05, 1.0))
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

    @classmethod
    def compute_roi_target_size(
        cls, full_w: int, full_h: int, roi_w: int, roi_h: int, scale_mode: str, scale_val_str: str
    ) -> tuple[int, int]:
        """Compute proportional target dimensions for an ROI crop matching the global downscale factor."""
        full_tw, full_th = cls.compute_target_size(full_w, full_h, scale_mode, scale_val_str)
        if full_tw == full_w and full_th == full_h:
            return roi_w, roi_h
        sx = full_tw / max(1.0, float(full_w))
        sy = full_th / max(1.0, float(full_h))
        tw = max(2, int(round(roi_w * sx)))
        th = max(2, int(round(roi_h * sy)))
        if tw > 4 and (tw % 2 != 0):
            tw += 1
        if th > 4 and (th % 2 != 0):
            th += 1
        return min(roi_w, tw), min(roi_h, th)

    @staticmethod
    def _eval_continuous_kernel(x: np.ndarray, k_type: str) -> np.ndarray:
        ax = np.abs(x)
        if k_type == "mks":
            w = np.zeros_like(ax)
            m1, m2 = ax < 0.5, (ax >= 0.5) & (ax < 1.5)
            w[m1], w[m2] = 0.75 - (ax[m1] ** 2), 0.5 * ((1.5 - ax[m2]) ** 2)
            return w
        if k_type == "kaiser3":
            ratio = np.clip(ax / 3.0, 0.0, 1.0)
            win = _bessel_i0(6.5 * np.sqrt(np.maximum(0.0, 1.0 - ratio ** 2))) / float(_bessel_i0(np.array([6.5]))[0])
            return np.where(ax < 3.0, np.sinc(x) * win, 0.0)
        if k_type == "ginseng3":
            return np.where(ax < 3.0, np.sinc(x) * _jinc_j1_zero(x / 3.0), 0.0)
        if k_type == "ewa_jinc3":
            return np.where(ax < 3.23831548, _jinc_pi(ax / 0.9812505644) * _jinc_pi(ax / 3.23831548), 0.0)
        if k_type == "ewa_robidoux_sharp":
            return _eval_keys_cubic(ax, ROBIDOUX_SHARP_B, ROBIDOUX_SHARP_C)
        if k_type in ("mitchell", "ewa_mitchell"):
            return _eval_keys_cubic(ax, 1.0 / 3.0, 1.0 / 3.0)
        if k_type == "lanczos3":
            return np.where(ax < 3.0, np.sinc(x) * np.sinc(x / 3.0), 0.0)
        return _eval_keys_cubic(ax, 0.0, 0.5)

    @classmethod
    def _get_polyphase_table(cls, n_in: int, n_out: int, k_type: str):
        key = (n_in, n_out, k_type)
        if key in cls._POLYPHASE_CACHE:
            cls._POLYPHASE_CACHE.move_to_end(key)
            return cls._POLYPHASE_CACHE[key]
        radius_map = {
            "mks": 1.5, "mitchell": 2.0, "ewa_mitchell": 2.0, "ewa_robidoux_sharp": 2.0,
            "catrom": 2.0, "kaiser3": 3.0, "ginseng3": 3.0, "ewa_jinc3": 3.24, "lanczos3": 3.0,
        }
        R = radius_map.get(k_type, 1.5)
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
    def downscale_aux_pass(
        cls,
        rgb: np.ndarray,
        target_w: int,
        target_h: int,
        pass_type: str,
        ds_kernel: str,
    ) -> np.ndarray:
        h, w, _ = rgb.shape
        p_low = pass_type.lower()
        if (target_w, target_h) == (w, h):
            return rgb
        if "material_id" in p_low or "object_id" in p_low or p_low.endswith("_id"):
            if rgb.dtype == np.uint32:
                yi = (np.arange(target_h, dtype=np.float64) * (h / float(target_h))).astype(np.int32)
                xi = (np.arange(target_w, dtype=np.float64) * (w / float(target_w))).astype(np.int32)
                return np.ascontiguousarray(rgb[yi[:, None], xi[None, :], :], dtype=np.uint32)
            res = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            return np.ascontiguousarray(res[:, :, None] if res.ndim == 2 else res, dtype=np.float32)
        if "mask" in p_low or p_low == "alpha" or "roughness" in p_low:
            res = cls._resample_polyphase_2d(np.ascontiguousarray(rgb, dtype=np.float32), target_w, target_h, "mitchell")
            return np.ascontiguousarray(np.clip(res, 0.0, 1.0), dtype=np.float32)
        if "depth" in p_low or "position" in p_low or "normal" in p_low:
            return np.ascontiguousarray(
                cls._resample_polyphase_2d(np.ascontiguousarray(rgb, dtype=np.float32), target_w, target_h, "mitchell"),
                dtype=np.float32,
            )
        return cls.downscale(rgb, target_w, target_h, ds_kernel)

    @classmethod
    def _resize_area_safe(cls, arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        if arr.ndim == 2 or arr.shape[2] <= 4:
            res = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return res[:, :, None] if res.ndim == 2 else res
        parts = [
            cv2.resize(arr[:, :, i : min(i + 4, arr.shape[2])], (target_w, target_h), interpolation=cv2.INTER_AREA)
            for i in range(0, arr.shape[2], 4)
        ]
        parts = [p[:, :, None] if p.ndim == 2 else p for p in parts]
        return np.concatenate(parts, axis=2)

    @classmethod
    def _guided_downscale_cpu(
        cls,
        src: np.ndarray,
        target_w: int,
        target_h: int,
        alb_rgb: np.ndarray | None = None,
        norm_rgb: np.ndarray | None = None,
        depth_arr: np.ndarray | None = None,
    ) -> np.ndarray:
        h, w, c = src.shape
        lum = np.maximum(0.2126 * src[:, :, 0] + 0.7152 * src[:, :, 1] + 0.0722 * src[:, :, 2], 0.0)
        w_k = (1.0 / (1.0 + 0.25 * lum)).astype(np.float32)

        g_list = []
        if alb_rgb is not None and alb_rgb.shape[:2] == (h, w):
            a = np.clip(np.ascontiguousarray(alb_rgb, dtype=np.float32), 0.0, 1.0)
            g_list.extend([a[:, :, 0], a[:, :, 1], a[:, :, 2]])
        else:
            tm = src / (1.0 + lum[:, :, None])
            g_list.extend([tm[:, :, 0], tm[:, :, 1], tm[:, :, 2]])
        if norm_rgb is not None and norm_rgb.shape[:2] == (h, w):
            n = np.ascontiguousarray(norm_rgb, dtype=np.float32) * 0.5
            g_list.extend([n[:, :, 0], n[:, :, 1], n[:, :, 2]])
        if depth_arr is not None and depth_arr.shape[:2] == (h, w):
            d = np.clip(np.ascontiguousarray(depth_arr[:, :, 0] if depth_arr.ndim == 3 else depth_arr, dtype=np.float32), 0.0, 1.0)
            g_list.append(d)

        G = np.stack(g_list, axis=2)
        M = G.shape[2]
        w_ds = np.maximum(cv2.resize(w_k, (target_w, target_h), interpolation=cv2.INTER_AREA), 1e-7)[:, :, None]
        mu_p = cls._resize_area_safe(src * w_k[:, :, None], target_w, target_h) / w_ds
        mu_g = cls._resize_area_safe(G * w_k[:, :, None], target_w, target_h) / w_ds
        g_ds = cls._resize_area_safe(G, target_w, target_h)

        A = np.zeros((target_h, target_w, M, M), dtype=np.float32)
        S_gp = np.zeros((target_h, target_w, M, c), dtype=np.float32)
        w_ds2 = w_ds[:, :, 0]
        for r in range(M):
            gr = G[:, :, r]
            gp_r = cls._resize_area_safe(src * (gr * w_k)[:, :, None], target_w, target_h) / w_ds
            S_gp[:, :, r, :] = gp_r - mu_g[:, :, r : r + 1] * mu_p
            for k in range(r, M):
                gg = cv2.resize(gr * G[:, :, k] * w_k, (target_w, target_h), interpolation=cv2.INTER_AREA) / w_ds2
                cov = gg - mu_g[:, :, r] * mu_g[:, :, k]
                if r == k:
                    cov = cov + 0.0025
                A[:, :, r, k] = cov
                A[:, :, k, r] = cov

        B = np.linalg.solve(A, S_gp)
        q = mu_p + np.einsum("hwmc,hwm->hwc", B, g_ds - mu_g)
        return np.maximum(q, 0.0).astype(np.float32)

    @classmethod
    def downscale(
        cls,
        rgb: np.ndarray,
        target_w: int,
        target_h: int,
        mode_label: str,
        sharpness: float = 1.0,
        radius: float = 1.0,
        norm_rgb: np.ndarray | None = None,
        alb_rgb: np.ndarray | None = None,
        depth_arr: np.ndarray | None = None,
    ) -> np.ndarray:
        h, w, c = rgb.shape
        if (
            target_w >= w
            and target_h >= h
            and abs(sharpness - 1.0) < 1e-4
            and not GPU_SHADER_ENGINE.show_mask_xray
        ):
            return rgb

        if GPU_SHADER_ENGINE.available and c == 3:
            try:
                return GPU_SHADER_ENGINE.downscale_gpu(
                    rgb,
                    target_w,
                    target_h,
                    mode_label,
                    sharpness=sharpness,
                    radius=radius,
                    norm_rgb=norm_rgb,
                    alb_rgb=alb_rgb,
                    depth_arr=depth_arr,
                )
            except Exception:
                pass

        # ---- CPU fallback (complete feature parity including Perceptual, Guided, and EWA) ----
        src = np.ascontiguousarray(np.maximum(rgb, 0.0), dtype=np.float32)
        spec = get_filter_spec(mode_label)

        if spec.colorspace != 0 and c == 3:
            work_src = rgb_to_ictcp(src) if spec.colorspace == 1 else rgb_to_oklab(src)
            base_w = cls._resample_polyphase_2d(work_src, target_w, target_h, spec.cpu_kernel)
            k3 = np.ones((3, 3), dtype=np.uint8)
            area_w = cls._resize_area_safe(work_src, target_w, target_h)
            w_min, w_max = cv2.erode(area_w, k3), cv2.dilate(area_w, k3)
            sh_x = cv2.sepFilter2D(base_w[:, :, 0], cv2.CV_32F, cls.K_MKS2021_SHARP, cls.K_MKS2021_SHARP, borderType=cv2.BORDER_REPLICATE)
            edge_pass = 1.0
            if depth_arr is not None and depth_arr.shape[:2] == (h, w):
                d0 = np.clip(np.ascontiguousarray(depth_arr[:, :, 0] if depth_arr.ndim == 3 else depth_arr, dtype=np.float32), 0.0, 1.0)
                d_ds = cv2.resize(d0, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                d_lap = np.maximum(
                    np.abs(np.roll(d_ds, -1, axis=0) + np.roll(d_ds, 1, axis=0) - 2.0 * d_ds),
                    np.abs(np.roll(d_ds, -1, axis=1) + np.roll(d_ds, 1, axis=1) - 2.0 * d_ds),
                )
                t_d = np.clip((d_lap - 0.015) / 0.065, 0.0, 1.0)
                edge_pass = 1.0 - t_d * t_d * (3.0 - 2.0 * t_d)
            res_w = base_w.copy()
            res_w[:, :, 0] = base_w[:, :, 0] + sharpness * edge_pass * (sh_x - base_w[:, :, 0])
            under = np.maximum(0.0, w_min - res_w)
            res_w = res_w + 0.92 * under
            over = np.maximum(0.0, res_w - w_max)
            res_w = res_w - 0.30 * over
            res = ictcp_to_rgb(res_w) if spec.colorspace == 1 else oklab_to_rgb(res_w)
        else:
            area_ref = cls._resize_area_safe(src, target_w, target_h)
            k3 = np.ones((3, 3), dtype=np.uint8)
            l_min, l_max = cv2.erode(area_ref, k3), cv2.dilate(area_ref, k3)

            if spec.special == 2:
                base_c = cls._guided_downscale_cpu(src, target_w, target_h, alb_rgb=alb_rgb, norm_rgb=norm_rgb, depth_arr=depth_arr)
            elif spec.special == 1:
                lum = 0.2126 * src[:, :, 0] + 0.7152 * src[:, :, 1] + 0.0722 * src[:, :, 2]
                w_k = (1.0 / (1.0 + np.maximum(lum, 0.0)))[:, :, None]
                d = cls._resize_area_safe(np.concatenate([src * w_k, w_k], axis=2), target_w, target_h)
                base_c = d[:, :, :c] / np.maximum(d[:, :, c : c + 1], 1e-7)
            else:
                lum = 0.2126 * src[:, :, 0] + 0.7152 * src[:, :, 1] + 0.0722 * src[:, :, 2]
                w_k = (1.0 / (1.0 + np.maximum(lum, 0.0)))[:, :, None]
                d = cls._resample_polyphase_2d(np.concatenate([src * w_k, w_k], axis=2), target_w, target_h, spec.cpu_kernel)
                base_c = d[:, :, :c] / np.maximum(d[:, :, c : c + 1], 1e-7)

            edge_pass = 1.0
            if depth_arr is not None and depth_arr.shape[:2] == (h, w):
                d0 = np.clip(np.ascontiguousarray(depth_arr[:, :, 0] if depth_arr.ndim == 3 else depth_arr, dtype=np.float32), 0.0, 1.0)
                d_ds = cv2.resize(d0, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                d_lap = np.maximum(
                    np.abs(np.roll(d_ds, -1, axis=0) + np.roll(d_ds, 1, axis=0) - 2.0 * d_ds),
                    np.abs(np.roll(d_ds, -1, axis=1) + np.roll(d_ds, 1, axis=1) - 2.0 * d_ds),
                )
                t_d = np.clip((d_lap - 0.015) / 0.065, 0.0, 1.0)
                edge_pass = (1.0 - t_d * t_d * (3.0 - 2.0 * t_d))[:, :, None]

            if spec.sharp_mode in (2, 3, 4, 5, 6):
                sh = cv2.sepFilter2D(base_c, cv2.CV_32F, cls.K_MKS2021_SHARP, cls.K_MKS2021_SHARP, borderType=cv2.BORDER_REPLICATE)
                mult = 1.0 if spec.is_luma_only_mks else edge_pass
                res = base_c + sharpness * mult * (sh - base_c)
            elif sharpness > 1.001:
                sh = cv2.sepFilter2D(base_c, cv2.CV_32F, cls.K_MKS2021_SHARP, cls.K_MKS2021_SHARP, borderType=cv2.BORDER_REPLICATE)
                res = base_c + (sharpness - 1.0) * edge_pass * (sh - base_c)
            else:
                res = base_c

            under = np.maximum(0.0, l_min - res)
            res = res + 0.92 * under
            over = np.maximum(0.0, res - l_max)
            res = np.maximum(res - 0.35 * over, 0.0)

        return np.ascontiguousarray(np.maximum(res, 0.0), dtype=np.float32)
