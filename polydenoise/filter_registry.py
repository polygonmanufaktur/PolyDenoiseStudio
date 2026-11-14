# SPDX-License-Identifier: MIT
"""Centralized FilterSpec registry driving GPU shaders, CPU fallbacks, hotkeys, and tooltips."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterSpec:
    """Single source of truth for a downscaling filter's parameters and metadata."""

    label: str
    family: str
    hotkey_hint: str
    key_sequence: str | None
    ktype: int
    use_karis: int          # 0: linear, 1: karis_weighted, 2: reversible_karis
    sharp_mode: int         # 0: boost_only, 2: mks2021, 3: edge_bal, 4: edge_strict, 5: gbuf, 6: amd_cas
    ar_mode: int            # 0: unclamped, 1: rev_karis, 2: asym, 3: strict_2x2, 4: span_8pct
    special: int            # 0: standard/ewa, 1: dpid, 2: guided
    colorspace: int         # 0: linear_rgb, 1: ictcp_pq, 2: oklab
    ewa_kind: int           # -1: separable/special, 0: lanczos_sharp, 2: robidoux_sharp, 3: mitchell
    cpu_kernel: str
    always_supports_depth: bool
    always_needs_guides: bool
    is_luma_only_mks: bool
    description: str


FILTER_SPECS: tuple[FilterSpec, ...] = (
    # ---- MKS Family (Strict Halo default first) ----
    FilterSpec(
        label="[1] Edge-MKS 2021 • Strict Halo",
        family="MKS",
        hotkey_hint="1",
        key_sequence="<Key-1>",
        ktype=0, use_karis=2, sharp_mode=4, ar_mode=1, special=0, colorspace=0, ewa_kind=-1,
        cpu_kernel="mks",
        always_supports_depth=True, always_needs_guides=True, is_luma_only_mks=False,
        description=(
            "Edge-MKS 2021 • Strict Halo (MKS Family — Default)\n"
            "• Pass 1: Magic Kernel decimation in Reversible Karis HDR space (c / (1 + Y)) to eliminate bright firefly blowup.\n"
            "• Pass 2: 7-tap MKS 2021 deconvolution modulated by strict perceptual contrast damping (smoothstep 0.05..0.20) "
            "multiplied by geometric G-Buffer protection (Normal min-dot + Depth Laplacian).\n"
            "• Result: Zero ringing halos around high-contrast light edges and silhouette jumps while preserving crisp flat textures."
        ),
    ),
    FilterSpec(
        label="[2] Edge-MKS 2021 • Balanced",
        family="MKS",
        hotkey_hint="2",
        key_sequence="<Key-2>",
        ktype=0, use_karis=2, sharp_mode=3, ar_mode=1, special=0, colorspace=0, ewa_kind=-1,
        cpu_kernel="mks",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=True,
        description=(
            "Edge-MKS 2021 • Balanced (MKS Family)\n"
            "• Pass 1: Reversible Karis Magic Kernel decimation for HDR energy stability.\n"
            "• Pass 2: 7-tap MKS 2021 sharpening damped purely by local perceptual luma contrast (smoothstep 0.14..0.48), "
            "ignoring Normal and Depth G-Buffers.\n"
            "• Result: Clean all-rounder balancing micro-texture bite against moderate highlight halo suppression without external guides."
        ),
    ),
    FilterSpec(
        label="[3] Edge-MKS 2021 • G-Buffer",
        family="MKS",
        hotkey_hint="3",
        key_sequence="<Key-3>",
        ktype=0, use_karis=2, sharp_mode=5, ar_mode=1, special=0, colorspace=0, ewa_kind=-1,
        cpu_kernel="mks",
        always_supports_depth=True, always_needs_guides=True, is_luma_only_mks=False,
        description=(
            "Edge-MKS 2021 • G-Buffer (MKS Family)\n"
            "• Pass 1: Reversible Karis Magic Kernel decimation.\n"
            "• Pass 2: Fully geometry- and surface-guided MKS 2021: boost gain adapts to Albedo texture detail (0.70x–1.30x) "
            "while dampening sharpening across Normal creases, Depth silhouette steps, and extreme HDR contrast.\n"
            "• Result: Maximum fidelity on complex interior and architectural renders with rich surface textures."
        ),
    ),
    FilterSpec(
        label="[4] MKS 2021 + AMD CAS",
        family="MKS",
        hotkey_hint="4",
        key_sequence="<Key-4>",
        ktype=0, use_karis=2, sharp_mode=6, ar_mode=1, special=0, colorspace=0, ewa_kind=-1,
        cpu_kernel="mks",
        always_supports_depth=True, always_needs_guides=True, is_luma_only_mks=False,
        description=(
            "MKS 2021 + AMD CAS (MKS Family)\n"
            "• Pass 1: Reversible Karis Magic Kernel decimation.\n"
            "• Pass 2: Hybrid frequency restoration blending 60% 7-tap MKS 2021 deconvolution with 40% AMD FidelityFX "
            "Contrast Adaptive Sharpening, gated by Normal + Depth geometric edge protection.\n"
            "• Result: Punchy local micro-contrast that dynamically backs off on already high-contrast specular edges."
        ),
    ),
    FilterSpec(
        label="[5] MKS 2021 • Reversible HDR",
        family="MKS",
        hotkey_hint="5",
        key_sequence="<Key-5>",
        ktype=0, use_karis=2, sharp_mode=2, ar_mode=1, special=0, colorspace=0, ewa_kind=-1,
        cpu_kernel="mks",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=True,
        description=(
            "MKS 2021 • Reversible HDR (MKS Family)\n"
            "• Pass 1: Separable Magic Kernel decimation computed in reversible Karis tone-compressed space.\n"
            "• Pass 2: Unmasked 7-tap MKS 2021 deconvolution with asymmetric anti-ringing before HDR expansion (no G-Buffer masking).\n"
            "• Result: Maximum global spectral sharpness across the entire image while staying numerically stable on extreme HDR peaks."
        ),
    ),

    # ---- Perceptual Color-Space Family ----
    FilterSpec(
        label="[6] ICtCp HDR • Perceptual MKS 2021",
        family="Perceptual",
        hotkey_hint="6",
        key_sequence="<Key-6>",
        ktype=0, use_karis=0, sharp_mode=4, ar_mode=2, special=0, colorspace=1, ewa_kind=-1,
        cpu_kernel="mks",
        always_supports_depth=True, always_needs_guides=True, is_luma_only_mks=False,
        description=(
            "ICtCp HDR • Perceptual MKS 2021 (Perceptual Color-Space Family)\n"
            "• Pass 1: Converts Linear Rec.709 HDR via BT.2020 LMS ST-2084 PQ into Dolby ICtCp prior to Magic Kernel decimation.\n"
            "• Pass 2: Applies Normal + Depth-protected MKS 2021 deconvolution strictly to PQ Intensity (I) while leaving Ct/Cp chroma untouched, then converts back.\n"
            "• Result: Zero hue shifts, zero cyan/magenta overshoot fringes on saturated neon/tungsten emitters up to extreme HDR peaks."
        ),
    ),
    FilterSpec(
        label="[7] Oklab • Perceptual MKS 2021",
        family="Perceptual",
        hotkey_hint="7",
        key_sequence="<Key-7>",
        ktype=0, use_karis=0, sharp_mode=4, ar_mode=2, special=0, colorspace=2, ewa_kind=-1,
        cpu_kernel="mks",
        always_supports_depth=True, always_needs_guides=True, is_luma_only_mks=False,
        description=(
            "Oklab • Perceptual MKS 2021 (Perceptual Color-Space Family)\n"
            "• Pass 1: Transforms Linear RGB into Björn Ottosson's perceptual Oklab (L, a, b) via cube-root LMS cone response and decimates with Magic Kernel.\n"
            "• Pass 2: Applies Normal + Depth-protected MKS 2021 deconvolution strictly on perceptual Lightness (L) while preserving opponent (a, b) chromaticity.\n"
            "• Result: Perceptually uniform highlight smoothing and crisp albedo/material edge contrast without color fringing."
        ),
    ),

    # ---- AOV / Guided Family ----
    FilterSpec(
        label="[8] Guided Downscale • He et al.",
        family="AOV / Guided",
        hotkey_hint="8",
        key_sequence="<Key-8>",
        ktype=4, use_karis=1, sharp_mode=0, ar_mode=4, special=2, colorspace=0, ewa_kind=-1,
        cpu_kernel="guided",
        always_supports_depth=True, always_needs_guides=True, is_luma_only_mks=False,
        description=(
            "Guided Downscale • He et al. (AOV / Guided Family)\n"
            "• Pass 1: Solves a weighted 7x7 joint linear regression per target pixel via LDL^T Cholesky factorization against "
            "[Albedo RGB | Normals*0.5 | Log-Depth] (falling back to tonemapped beauty if Albedo is absent).\n"
            "• Pass 2: Optional universal Boost adds Normal/Depth-protected MKS 2021 micro-sharpening bounded by 8% AR span clamping.\n"
            "• Result: Transfers noise-free geometric silhouettes and material boundaries directly into the downscaled beauty with zero ringing."
        ),
    ),
    FilterSpec(
        label="[9] AOV-DPID • Detail-Preserving",
        family="AOV / Guided",
        hotkey_hint="9",
        key_sequence="<Key-9>",
        ktype=4, use_karis=1, sharp_mode=0, ar_mode=0, special=1, colorspace=0, ewa_kind=-1,
        cpu_kernel="dpid",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=False,
        description=(
            "AOV-DPID • Detail-Preserving (AOV / Guided Family)\n"
            "• Pass 1: Tent/Area Karis luma-weighted decimation running on fixed Guide λ = 0 for clean, unbiased energy integration "
            "and strong path-tracing firefly attenuation.\n"
            "• Pass 2: Unclamped base by default; raising Boost above +0% adds MKS 2021 detail protected by Normal + Depth silhouettes.\n"
            "• Result: Smooth, natural optical integration of tiny sub-pixel highlights without high-frequency overshoot."
        ),
    ),

    # ---- EWA 2D Isotropic Family ----
    FilterSpec(
        label="[0] EWA LanczosSharp • 2D Jinc",
        family="EWA 2D",
        hotkey_hint="0",
        key_sequence="<Key-0>",
        ktype=2, use_karis=2, sharp_mode=0, ar_mode=1, special=0, colorspace=0, ewa_kind=0,
        cpu_kernel="ewa_jinc3",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=False,
        description=(
            "EWA LanczosSharp • 2D Jinc (EWA 2D Family)\n"
            "• Pass 1: Non-separable 2D radial Elliptical Weighted Average Jinc-3 windowed by a Jinc lobe (Robidoux blur = 0.981, radius 3.24) "
            "in Reversible Karis HDR space.\n"
            "• Pass 2: Reversible Karis asymmetric anti-ringing, plus optional Normal/Depth-protected Boost above +0%.\n"
            "• Result: Superior isotropic diagonal edges without horizontal/vertical ringing bias of separable 1D Sinc filters."
        ),
    ),
    FilterSpec(
        label="[Ctrl+1] EWA RobidouxSharp • 2D Keys",
        family="EWA 2D",
        hotkey_hint="Ctrl+1",
        key_sequence="<Control-Key-1>",
        ktype=9, use_karis=2, sharp_mode=0, ar_mode=1, special=0, colorspace=0, ewa_kind=2,
        cpu_kernel="ewa_robidoux_sharp",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=False,
        description=(
            "EWA RobidouxSharp • 2D Keys (EWA 2D Family)\n"
            "• Pass 1: Non-separable 2D radial evaluation of ImageMagick's RobidouxSharp Keys cubic spline "
            "(B = 6 / (13 + 7√2) ≈ 0.2620, C = (1 - B)/2 ≈ 0.3690, radius = 2.0) in Reversible Karis HDR space.\n"
            "• Pass 2: Reversible Karis anti-ringing + optional Normal/Depth-protected Boost above +0%.\n"
            "• Result: Tight 2-pixel radial support with maximum crispness and isotropic anti-aliasing—zero cross-hatched stair-stepping."
        ),
    ),
    FilterSpec(
        label="[Ctrl+2] EWA Mitchell • 2D Smooth",
        family="EWA 2D",
        hotkey_hint="Ctrl+2",
        key_sequence="<Control-Key-2>",
        ktype=10, use_karis=1, sharp_mode=0, ar_mode=4, special=0, colorspace=0, ewa_kind=3,
        cpu_kernel="ewa_mitchell",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=False,
        description=(
            "EWA Mitchell • 2D Smooth (EWA 2D Family)\n"
            "• Pass 1: Non-separable 2D radial Mitchell-Netravali cubic spline (B = 1/3, C = 1/3, radius = 2.0) with Karis luma weighting.\n"
            "• Pass 2: 8% local min/max span clamp + optional Normal/Depth-protected Boost above +0%.\n"
            "• Result: Silky isotropic anti-aliasing for organic shapes and animations with minimal diagonal moiré."
        ),
    ),

    # ---- Separable Sinc Family ----
    FilterSpec(
        label="[Ctrl+3] Lanczos-3 • Strict AR",
        family="Sinc",
        hotkey_hint="Ctrl+3",
        key_sequence="<Control-Key-3>",
        ktype=4, use_karis=1, sharp_mode=0, ar_mode=3, special=0, colorspace=0, ewa_kind=-1,
        cpu_kernel="lanczos3",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=False,
        description=(
            "Lanczos-3 • Strict AR (Sinc Family)\n"
            "• Pass 1: Karis luma-weighted 6-lobe separable Sinc-windowed Sinc (a = 3).\n"
            "• Pass 2: Strict 2x2 local min/max box clamping (u_ar_mode = 3) with optional Normal/Depth-protected Boost above +0%.\n"
            "• Result: Crisp Sinc frequency detail while strictly capping negative/positive ringing overshoot to the local source range."
        ),
    ),
    FilterSpec(
        label="[Ctrl+4] Kaiser-Bessel-3 • HDR-AR",
        family="Sinc",
        hotkey_hint="Ctrl+4",
        key_sequence="<Control-Key-4>",
        ktype=1, use_karis=1, sharp_mode=0, ar_mode=2, special=0, colorspace=0, ewa_kind=-1,
        cpu_kernel="kaiser3",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=False,
        description=(
            "Kaiser-Bessel-3 • HDR-AR (Sinc Family)\n"
            "• Pass 1: Separable Sinc windowed by a Kaiser-Bessel I0 bell (beta = 6.5, radius = 3) with Karis luma firefly weighting.\n"
            "• Pass 2: Asymmetric anti-ringing (92% undershoot / 30% overshoot damping) + optional Normal/Depth-protected Boost above +0%.\n"
            "• Result: Near-optimal stopband ripple suppression, cutting aliasing and halo artifacts cleaner than standard Lanczos-3."
        ),
    ),
    FilterSpec(
        label="[Ctrl+5] Ginseng-3 • Soft-AR",
        family="Sinc",
        hotkey_hint="Ctrl+5",
        key_sequence="<Control-Key-5>",
        ktype=3, use_karis=1, sharp_mode=0, ar_mode=2, special=0, colorspace=0, ewa_kind=-1,
        cpu_kernel="ginseng3",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=False,
        description=(
            "Ginseng-3 • Soft-AR (Sinc Family)\n"
            "• Pass 1: Separable Sinc windowed by a first-order Bessel Jinc(J1) bell (radius = 3) with Karis luma weighting.\n"
            "• Pass 2: Asymmetric anti-ringing + optional Normal/Depth-protected Boost above +0%.\n"
            "• Result: Gently tapered side-lobes near Nyquist that preserve foliage, hair, and fine textures without brittle ringing."
        ),
    ),

    # ---- Separable Cubic Family (Mitchell Smooth VFX LAST) ----
    FilterSpec(
        label="[Ctrl+6] Catmull-Rom • Bicubic",
        family="Cubic",
        hotkey_hint="Ctrl+6",
        key_sequence="<Control-Key-6>",
        ktype=6, use_karis=0, sharp_mode=0, ar_mode=0, special=0, colorspace=0, ewa_kind=-1,
        cpu_kernel="catrom",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=False,
        description=(
            "Catmull-Rom • Bicubic (Cubic Family)\n"
            "• Pass 1: Pure linear 4x4 Keys cubic spline (B = 0, C = 0.5) without Karis weighting.\n"
            "• Pass 2: Unclamped native spline response (optional Boost above +0% adds Normal/Depth-protected MKS detail).\n"
            "• Result: Standard compositing/Nuke/Flame crisp bicubic reference with moderate natural edge enhancement."
        ),
    ),
    FilterSpec(
        label="[Ctrl+7] Mitchell • Smooth VFX",
        family="Cubic",
        hotkey_hint="Ctrl+7",
        key_sequence="<Control-Key-7>",
        ktype=7, use_karis=1, sharp_mode=0, ar_mode=4, special=0, colorspace=0, ewa_kind=-1,
        cpu_kernel="mitchell",
        always_supports_depth=False, always_needs_guides=False, is_luma_only_mks=False,
        description=(
            "Mitchell • Smooth VFX (Cubic Family)\n"
            "• Pass 1: Karis luma-weighted Mitchell-Netravali cubic spline (B = 1/3, C = 1/3) balancing blur and ringing.\n"
            "• Pass 2: 8% local min/max span clamp + optional Normal/Depth-protected Boost above +0%.\n"
            "• Result: Film/VFX animation standard that suppresses moiré, temporal shimmer, and hard ringing for clean silky transitions."
        ),
    ),
)

DEFAULT_FILTER_LABEL = FILTER_SPECS[0].label
_SPEC_BY_LABEL: dict[str, FilterSpec] = {spec.label: spec for spec in FILTER_SPECS}


def get_filter_labels() -> list[str]:
    """Return ordered list of display labels for the UI dropdown."""
    return [spec.label for spec in FILTER_SPECS]


def get_filter_spec(mode_label: str) -> FilterSpec:
    """Resolve FilterSpec from exact label or legacy hotkey migration name, falling back to Strict Halo."""
    if mode_label in _SPEC_BY_LABEL:
        return _SPEC_BY_LABEL[mode_label]
    low = (mode_label or "").lower()
    stripped = low.split("]", 1)[1].strip() if "]" in low else low
    for spec in FILTER_SPECS:
        spec_core = spec.label.split("]", 1)[1].strip().lower()
        if stripped and (stripped == spec_core or stripped in spec_core):
            return spec
    return FILTER_SPECS[0]


def describe_filter(mode_label: str) -> str:
    """Return technical tooltip description for the active filter."""
    return get_filter_spec(mode_label).description


def filter_supports_depth(mode_label: str, sharp_str: float) -> bool:
    """Whether Z-Depth silhouette protection or guide regression is active for this filter & boost."""
    spec = get_filter_spec(mode_label)
    if spec.always_supports_depth:
        return True
    if spec.is_luma_only_mks:
        return False
    return sharp_str > 1.001


def downscale_needs_guides(mode_label: str, sharp_str: float) -> bool:
    """Whether downscaling needs Normals/Albedo guides fetched into RAM/VRAM."""
    spec = get_filter_spec(mode_label)
    if spec.always_needs_guides:
        return True
    if spec.is_luma_only_mks:
        return False
    return sharp_str > 1.001
