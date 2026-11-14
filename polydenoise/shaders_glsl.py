# SPDX-License-Identifier: MIT
"""GLSL 330 shader sources for Pass 1 decimation, Pass 2 sharpening/protection, and GPU viewport tonemapping."""

VERT_SRC = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Shared perceptual color-space transforms (Linear Rec.709 <-> BT.2100 PQ ICtCp <-> Oklab).
# Used identically in Pass 1 decimation and Pass 2 sharpening / reconstruction.
COLOR_SPACE_LIB = """
const float PQ_M1 = 2610.0 / 16384.0;
const float PQ_M2 = 2523.0 / 32.0;
const float PQ_C1 = 3424.0 / 4096.0;
const float PQ_C2 = 2413.0 / 128.0;
const float PQ_C3 = 2392.0 / 128.0;
const float PQ_EXT_SLOPE = 0.10466623306;

float safe_pow(float x, float e) {
    return (x <= 1e-12) ? 0.0 : exp(e * log(x));
}

float scbrt(float x) {
    float ax = abs(x);
    return (ax < 1e-12) ? 0.0 : sign(x) * exp(log(ax) / 3.0);
}

// CGI scene linear 1.0 maps to 100 nits (Y = 0.01), preserving ST-2084 accuracy up to 100.0 (10,000 nits)
// with a C1 monotonic log extension above 100.0 so extreme emitters never clip.
float pq_oetf_cgi(float c_lin) {
    float y = max(c_lin, 0.0) * 0.01;
    if (y > 1.0) {
        return 1.0 + PQ_EXT_SLOPE * log(y);
    }
    float ym = safe_pow(y, PQ_M1);
    return safe_pow((PQ_C1 + PQ_C2 * ym) / (1.0 + PQ_C3 * ym), PQ_M2);
}

float pq_eotf_cgi(float p) {
    if (p <= 0.0) return 0.0;
    if (p > 1.0) {
        return 100.0 * exp((p - 1.0) / PQ_EXT_SLOPE);
    }
    float pm = safe_pow(p, 1.0 / PQ_M2);
    float num = max(pm - PQ_C1, 0.0);
    float den = max(PQ_C2 - PQ_C3 * pm, 1e-12);
    return 100.0 * safe_pow(num / den, 1.0 / PQ_M1);
}

vec3 lin_to_ictcp(vec3 c) {
    // Rec.709 -> Rec.2020 -> BT.2100 LMS
    float l = 0.2958093546 * c.r + 0.6231013045 * c.g + 0.0810893409 * c.b;
    float m = 0.1562489762 * c.r + 0.7273257250 * c.g + 0.1164252989 * c.b;
    float s = 0.0351356628 * c.r + 0.1565747272 * c.g + 0.8082896101 * c.b;
    float lp = pq_oetf_cgi(l);
    float mp = pq_oetf_cgi(m);
    float sp = pq_oetf_cgi(s);
    float I  = 0.5 * lp + 0.5 * mp;
    float Ct = (6610.0 * lp - 13613.0 * mp + 7003.0 * sp) / 4096.0;
    float Cp = (17933.0 * lp - 17390.0 * mp - 543.0 * sp) / 4096.0;
    return vec3(I, Ct, Cp);
}

vec3 ictcp_to_lin(vec3 ic) {
    float I = ic.x, Ct = ic.y, Cp = ic.z;
    float lp = I + 0.0086090370 * Ct + 0.1110296250 * Cp;
    float mp = I - 0.0086090370 * Ct - 0.1110296250 * Cp;
    float sp = I + 0.5600313357 * Ct - 0.3206271749 * Cp;
    float l = pq_eotf_cgi(lp);
    float m = pq_eotf_cgi(mp);
    float s = pq_eotf_cgi(sp);
    return vec3(
        +6.1733627411 * l - 5.3203805675 * m + 0.1470178266 * s,
        -1.3243115001 * l + 2.5602264209 * m - 0.2359149210 * s,
        -0.0118169025 * l - 0.2646720350 * m + 1.2764889376 * s
    );
}

vec3 lin_to_oklab(vec3 c) {
    float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    float l_ = scbrt(l), m_ = scbrt(m), s_ = scbrt(s);
    return vec3(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}

vec3 oklab_to_lin(vec3 lab) {
    float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;
    return vec3(
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}

vec3 encode_colorspace(vec3 c, int cs) {
    if (cs == 1) return lin_to_ictcp(c);
    if (cs == 2) return lin_to_oklab(c);
    return c;
}

vec3 decode_colorspace(vec3 w, int cs) {
    if (cs == 1) return ictcp_to_lin(w);
    if (cs == 2) return oklab_to_lin(w);
    return w;
}
"""

KERNEL_LIB = COLOR_SPACE_LIB + """
const float PI = 3.141592653589793;
const float ROBIDOUX_SHARP_B = 0.2620145123990142;
const float ROBIDOUX_SHARP_C = 0.3689927438004929;

float sinc(float x) {
    if (abs(x) < 1e-5) return 1.0;
    float px = PI * x;
    return sin(px) / px;
}
float bessel_i0(float z) {
    float z2 = (z * z) * 0.25;
    float res = 1.0, term = 1.0;
    for (int k = 1; k <= 10; ++k) {
        term *= z2 / float(k * k);
        res += term;
    }
    return res;
}
float bessel_j1(float z) {
    float hz = z * 0.5;
    float z2 = -(hz * hz);
    float term = hz, res = hz;
    for (int k = 1; k <= 14; ++k) {
        term *= z2 / float(k * (k + 1));
        res += term;
    }
    return res;
}
float jinc_pi(float r) {
    float ar = abs(r) * PI;
    return (ar < 1e-5) ? 1.0 : (2.0 * bessel_j1(ar) / ar);
}
float jinc_zero(float u) {
    float ar = abs(u) * 3.83170597;
    return (ar < 1e-5) ? 1.0 : (2.0 * bessel_j1(ar) / ar);
}
float eval_keys_cubic(float ax, float B, float C) {
    if (ax >= 2.0) return 0.0;
    float ax2 = ax * ax;
    float ax3 = ax2 * ax;
    if (ax < 1.0) {
        return ((12.0 - 9.0*B - 6.0*C)*ax3 + (-18.0 + 12.0*B + 6.0*C)*ax2 + (6.0 - 2.0*B)) / 6.0;
    }
    return ((-B - 6.0*C)*ax3 + (6.0*B + 30.0*C)*ax2 + (-12.0*B - 48.0*C)*ax + (8.0*B + 24.0*C)) / 6.0;
}
// 0:mks, 1:kaiser3, 2:ewa_jinc3, 3:ginseng3, 4:lanczos3, 6:catrom, 7:mitchell,
// 9:ewa_robidoux_sharp, 10:ewa_mitchell
float eval_1d(float x, int kt) {
    float ax = abs(x);
    if (kt == 0) {
        if (ax < 0.5) return 0.75 - ax * ax;
        if (ax < 1.5) { float t = 1.5 - ax; return 0.5 * t * t; }
        return 0.0;
    }
    if (kt == 1) {
        if (ax >= 3.0) return 0.0;
        float r = ax / 3.0;
        return sinc(x) * (bessel_i0(6.5 * sqrt(max(0.0, 1.0 - r * r))) / bessel_i0(6.5));
    }
    if (kt == 3) return (ax < 3.0) ? sinc(x) * jinc_zero(x / 3.0) : 0.0;
    if (kt == 4) return (ax < 3.0) ? sinc(x) * sinc(x / 3.0) : 0.0;
    if (kt == 6) return eval_keys_cubic(ax, 0.0, 0.5);
    if (kt == 7 || kt == 10) return eval_keys_cubic(ax, 1.0 / 3.0, 1.0 / 3.0);
    if (kt == 9) return eval_keys_cubic(ax, ROBIDOUX_SHARP_B, ROBIDOUX_SHARP_C);
    return 0.0;
}
float kernel_radius(int kt) {
    if (kt == 0) return 1.5;
    if (kt == 6 || kt == 7 || kt == 9 || kt == 10) return 2.0;
    if (kt == 2) return 3.24;
    return 3.0;
}
float luma_lin(vec3 c) { return max(0.0, dot(c, vec3(0.2126, 0.7152, 0.0722))); }
vec3 fetch_src(sampler2D src, ivec2 co) { return max(texelFetch(src, co, 0).rgb, vec3(0.0)); }
"""

# --- Separable pass A: horizontal (output: target_w x src_h). out0=(sum c*wk, sum wk) ---
FRAG_H_BODY = """
uniform sampler2D u_src;
uniform vec2 u_src_res;
uniform vec2 u_dst_res;
uniform int u_ktype;
uniform int u_use_karis;   // 0:linear, 1:karis_weighted, 2:reversible_karis
uniform int u_colorspace;  // 0:linear_rgb, 1:ictcp_pq, 2:oklab
layout(location = 0) out vec4 out0;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float scale = u_dst_res.x / u_src_res.x;
    float fs = min(scale, 1.0);
    float support = kernel_radius(u_ktype) / fs;
    float cx = (float(p.x) + 0.5) / scale - 0.5;
    int i0 = int(floor(cx - support));
    int i1 = int(ceil(cx + support));
    int lim = int(u_src_res.x) - 1;
    vec3 s = vec3(0.0);
    float sw = 0.0;
    for (int ix = i0; ix <= i1; ++ix) {
        float w = eval_1d((float(ix) - cx) * fs, u_ktype);
        vec3 c = fetch_src(u_src, ivec2(clamp(ix, 0, lim), p.y));
        float lum = luma_lin(c);
        vec3 cw = (u_colorspace != 0) ? encode_colorspace(c, u_colorspace) : ((u_use_karis == 2) ? (c / (1.0 + lum)) : c);
        float wk = (u_use_karis == 1 && u_colorspace == 0) ? (w / (1.0 + lum)) : w;
        s += cw * wk;
        sw += wk;
    }
    out0 = vec4(s, sw);
}
"""

# --- Separable pass B: vertical + local min/max box (from source). Outputs: dec, min, max ---
FRAG_V_BODY = """
uniform sampler2D u_h0;
uniform sampler2D u_src;
uniform vec2 u_src_res;
uniform vec2 u_dst_res;
uniform int u_ktype;
uniform int u_use_karis;
uniform int u_colorspace;
layout(location = 0) out vec4 out_dec;
layout(location = 1) out vec4 out_min;
layout(location = 2) out vec4 out_max;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 scale = u_dst_res / u_src_res;
    vec2 fs = min(scale, vec2(1.0));
    float support = kernel_radius(u_ktype) / fs.y;
    float cy = (float(p.y) + 0.5) / scale.y - 0.5;
    int j0 = int(floor(cy - support));
    int j1 = int(ceil(cy + support));
    int limy = int(u_src_res.y) - 1;
    vec4 S0 = vec4(0.0);
    for (int iy = j0; iy <= j1; ++iy) {
        float wy = eval_1d((float(iy) - cy) * fs.y, u_ktype);
        S0 += wy * texelFetch(u_h0, ivec2(p.x, clamp(iy, 0, limy)), 0);
    }
    out_dec = vec4(S0.rgb / max(S0.a, 1e-7), 1.0);

    vec3 b_min = vec3(1e30), b_max = vec3(-1e30);
    float cx = (float(p.x) + 0.5) / scale.x - 0.5;
    vec2 br = max(vec2(1.1), 1.0 / fs);
    int lx = int(u_src_res.x) - 1;
    for (int iy = int(ceil(cy - br.y)); iy <= int(floor(cy + br.y)); ++iy) {
        for (int ix = int(ceil(cx - br.x)); ix <= int(floor(cx + br.x)); ++ix) {
            vec3 c = fetch_src(u_src, ivec2(clamp(ix, 0, lx), clamp(iy, 0, limy)));
            vec3 cw = (u_colorspace != 0) ? encode_colorspace(c, u_colorspace) : ((u_use_karis == 2) ? (c / (1.0 + luma_lin(c))) : c);
            b_min = min(b_min, cw);
            b_max = max(b_max, cw);
        }
    }
    out_min = vec4(b_min, 1.0);
    out_max = vec4(b_max, 1.0);
}
"""

# --- Non-separable EWA 2D pass (Jinc, RobidouxSharp, Mitchell) ---
FRAG_EWA_BODY = """
uniform sampler2D u_src;
uniform vec2 u_src_res;
uniform vec2 u_dst_res;
uniform int u_use_karis;
uniform int u_ewa_kind;    // 0:ewa_jinc3, 2:ewa_robidoux_sharp, 3:ewa_mitchell
in vec2 v_uv;
layout(location = 0) out vec4 out_dec;
layout(location = 1) out vec4 out_min;
layout(location = 2) out vec4 out_max;

void main() {
    vec2 scale = u_dst_res / u_src_res;
    vec2 f_scale = min(scale, vec2(1.0));
    float max_rad = (u_ewa_kind >= 2) ? 2.0 : 3.24;
    vec2 support = max_rad / f_scale;
    vec2 center_src = v_uv * u_src_res - 0.5;
    ivec2 i_min = ivec2(floor(center_src - support));
    ivec2 i_max = ivec2(ceil(center_src + support));
    ivec2 src_limit = ivec2(u_src_res) - 1;
    vec3 sum_rgb = vec3(0.0);
    float sum_w = 0.0;
    vec3 b_min = vec3(1e30), b_max = vec3(-1e30);
    vec2 box_rad = max(vec2(1.1), 1.0 / f_scale);

    for (int iy = i_min.y; iy <= i_max.y; ++iy) {
        float dy = (float(iy) - center_src.y) * f_scale.y;
        for (int ix = i_min.x; ix <= i_max.x; ++ix) {
            float dx = (float(ix) - center_src.x) * f_scale.x;
            float r = length(vec2(dx, dy));
            float w = 0.0;
            if (u_ewa_kind == 0) {
                w = (r < 3.23831548) ? jinc_pi(r / 0.9812505644) * jinc_pi(r / 3.23831548) : 0.0;
            } else if (u_ewa_kind == 2) {
                w = eval_keys_cubic(r, ROBIDOUX_SHARP_B, ROBIDOUX_SHARP_C);
            } else {
                w = eval_keys_cubic(r, 1.0 / 3.0, 1.0 / 3.0);
            }

            vec3 c = fetch_src(u_src, clamp(ivec2(ix, iy), ivec2(0), src_limit));
            float lum = luma_lin(c);
            vec3 c_work = (u_use_karis == 2) ? (c / (1.0 + lum)) : c;
            if (abs(float(ix) - center_src.x) <= box_rad.x && abs(float(iy) - center_src.y) <= box_rad.y) {
                b_min = min(b_min, c_work);
                b_max = max(b_max, c_work);
            }
            float wk = (u_use_karis == 1) ? (w / (1.0 + lum)) : w;
            sum_rgb += c_work * wk;
            sum_w += wk;
        }
    }
    out_dec = vec4(sum_rgb / max(sum_w, 1e-7), 1.0);
    out_min = vec4(b_min, 1.0);
    out_max = vec4(b_max, 1.0);
}
"""

# --- AOV-DPID (Guide lambda = 0 baseline) ---
FRAG_DPID_BODY = """
uniform sampler2D u_src;
uniform vec2 u_src_res;
uniform vec2 u_dst_res;
uniform int u_use_karis;
in vec2 v_uv;
layout(location = 0) out vec4 out_dec;
layout(location = 1) out vec4 out_min;
layout(location = 2) out vec4 out_max;

void main() {
    vec2 scale = u_dst_res / u_src_res;
    vec2 fs = min(scale, vec2(1.0));
    vec2 R = 1.0 / fs;
    vec2 center = v_uv * u_src_res - 0.5;
    ivec2 i0 = ivec2(floor(center - R));
    ivec2 i1 = ivec2(ceil(center + R));
    ivec2 lim = ivec2(u_src_res) - 1;

    vec3 s = vec3(0.0);
    float sw = 0.0;
    vec3 b_min = vec3(1e30), b_max = vec3(-1e30);
    vec2 br = max(vec2(1.1), R);
    for (int iy = i0.y; iy <= i1.y; ++iy) {
        float ty = max(0.0, 1.0 - abs(float(iy) - center.y) / R.y);
        for (int ix = i0.x; ix <= i1.x; ++ix) {
            float t = ty * max(0.0, 1.0 - abs(float(ix) - center.x) / R.x);
            ivec2 co = clamp(ivec2(ix, iy), ivec2(0), lim);
            vec3 c = fetch_src(u_src, co);
            float lum = luma_lin(c);
            if (abs(float(ix) - center.x) <= br.x && abs(float(iy) - center.y) <= br.y) {
                b_min = min(b_min, c);
                b_max = max(b_max, c);
            }
            if (t <= 0.0) continue;
            float wk = (u_use_karis == 1) ? (t / (1.0 + lum)) : t;
            s += c * wk;
            sw += wk;
        }
    }
    out_dec = vec4(s / max(sw, 1e-9), 1.0);
    out_min = vec4(b_min, 1.0);
    out_max = vec4(b_max, 1.0);
}
"""

# --- Multi-Channel Guided Image Downscaling (He et al.) ---
FRAG_GUIDED_BODY = """
uniform sampler2D u_src;
uniform sampler2D u_alb;
uniform sampler2D u_nrm;
uniform sampler2D u_depth;
uniform vec2 u_src_res;
uniform vec2 u_dst_res;
uniform int u_has_alb;
uniform int u_has_nrm;
uniform int u_has_depth;
uniform float u_karis_k;
in vec2 v_uv;
layout(location = 0) out vec4 out_dec;
layout(location = 1) out vec4 out_min;
layout(location = 2) out vec4 out_max;

void read_guide(ivec2 co, vec3 p_rgb, out float g[7]) {
    if (u_has_alb == 1) {
        vec3 a = clamp(texelFetch(u_alb, co, 0).rgb, 0.0, 1.0);
        g[0] = a.r; g[1] = a.g; g[2] = a.b;
    } else {
        vec3 tm = p_rgb / (1.0 + luma_lin(p_rgb));
        g[0] = tm.r; g[1] = tm.g; g[2] = tm.b;
    }
    if (u_has_nrm == 1) {
        vec3 n = texelFetch(u_nrm, co, 0).rgb * 0.5;
        g[3] = n.x; g[4] = n.y; g[5] = n.z;
    } else {
        g[3] = 0.0; g[4] = 0.0; g[5] = 0.0;
    }
    g[6] = (u_has_depth == 1) ? clamp(texelFetch(u_depth, co, 0).r, 0.0, 1.0) : 0.0;
}

void main() {
    vec2 scale = u_dst_res / u_src_res;
    vec2 fs = min(scale, vec2(1.0));
    vec2 R = max(vec2(1.5), 1.25 / fs);
    vec2 R_ds = max(vec2(1.0), 1.0 / fs);
    vec2 br = max(vec2(1.1), R_ds);
    vec2 center = v_uv * u_src_res - 0.5;
    ivec2 lim = ivec2(u_src_res) - 1;

    float mu_g[7], g_ds[7];
    for (int i = 0; i < 7; ++i) { mu_g[i] = 0.0; g_ds[i] = 0.0; }
    vec3 mu_p = vec3(0.0);
    float S_gg[28];
    for (int i = 0; i < 28; ++i) S_gg[i] = 0.0;
    vec3 S_gp[7];
    for (int i = 0; i < 7; ++i) S_gp[i] = vec3(0.0);
    float sum_w = 0.0, sum_wd = 0.0;
    vec3 b_min = vec3(1e30), b_max = vec3(-1e30);

    if (max(R.x, R.y) <= 5.0) {
        ivec2 i0 = ivec2(floor(center - R));
        ivec2 i1 = ivec2(ceil(center + R));
        for (int iy = i0.y; iy <= i1.y; ++iy) {
            float dy = abs(float(iy) - center.y);
            float wy = max(0.0, 1.0 - dy / R.y);
            if (wy <= 0.0) continue;
            float wdy = max(0.0, 1.0 - dy / R_ds.y);
            for (int ix = i0.x; ix <= i1.x; ++ix) {
                float dx = abs(float(ix) - center.x);
                float wx = max(0.0, 1.0 - dx / R.x);
                float w = wx * wy;
                if (w <= 0.0) continue;
                ivec2 co = clamp(ivec2(ix, iy), ivec2(0), lim);
                vec3 c = fetch_src(u_src, co);
                if (dx <= br.x && dy <= br.y) {
                    b_min = min(b_min, c);
                    b_max = max(b_max, c);
                }
                float wk = w / (1.0 + u_karis_k * luma_lin(c));
                float g[7];
                read_guide(co, c, g);
                mu_p += c * wk;
                sum_w += wk;
                for (int r = 0; r < 7; ++r) { mu_g[r] += g[r] * wk; S_gp[r] += (g[r] * c) * wk; }
                int idx = 0;
                for (int r = 0; r < 7; ++r) for (int k = r; k < 7; ++k) S_gg[idx++] += (g[r] * g[k]) * wk;
                float wd = max(0.0, 1.0 - dx / R_ds.x) * wdy;
                if (wd > 0.0) { for (int r = 0; r < 7; ++r) g_ds[r] += g[r] * wd; sum_wd += wd; }
            }
        }
    } else {
        vec2 step_src = R / 4.0;
        for (int jy = -4; jy <= 4; ++jy) {
            int iy = clamp(int(floor(center.y + float(jy) * step_src.y + 0.5)), 0, lim.y);
            float dy = abs(float(iy) - center.y);
            float wy = max(0.0, 1.0 - dy / (R.y + 0.5));
            float wdy = max(0.0, 1.0 - dy / (R_ds.y + 0.5));
            for (int jx = -4; jx <= 4; ++jx) {
                int ix = clamp(int(floor(center.x + float(jx) * step_src.x + 0.5)), 0, lim.x);
                float dx = abs(float(ix) - center.x);
                float wx = max(0.0, 1.0 - dx / (R.x + 0.5));
                float w = wx * wy;
                if (w <= 0.0) continue;
                ivec2 co = ivec2(ix, iy);
                vec3 c = fetch_src(u_src, co);
                if (dx <= br.x && dy <= br.y) {
                    b_min = min(b_min, c);
                    b_max = max(b_max, c);
                }
                float wk = w / (1.0 + u_karis_k * luma_lin(c));
                float g[7];
                read_guide(co, c, g);
                mu_p += c * wk;
                sum_w += wk;
                for (int r = 0; r < 7; ++r) { mu_g[r] += g[r] * wk; S_gp[r] += (g[r] * c) * wk; }
                int idx = 0;
                for (int r = 0; r < 7; ++r) for (int k = r; k < 7; ++k) S_gg[idx++] += (g[r] * g[k]) * wk;
                float wd = max(0.0, 1.0 - dx / (R_ds.x + 0.5)) * wdy;
                if (wd > 0.0) { for (int r = 0; r < 7; ++r) g_ds[r] += g[r] * wd; sum_wd += wd; }
            }
        }
    }

    if (b_min.x > b_max.x) { b_min = vec3(0.0); b_max = vec3(0.0); }
    float inv_w = 1.0 / max(sum_w, 1e-7);
    mu_p *= inv_w;
    float inv_wd = 1.0 / max(sum_wd, 1e-7);
    for (int r = 0; r < 7; ++r) { mu_g[r] *= inv_w; g_ds[r] *= inv_wd; }

    float A[49];
    vec3 B[7];
    const float EPS = 0.0025;
    int idx = 0;
    for (int r = 0; r < 7; ++r) {
        B[r] = S_gp[r] * inv_w - mu_g[r] * mu_p;
        for (int k = r; k < 7; ++k) {
            float cov = S_gg[idx++] * inv_w - mu_g[r] * mu_g[k];
            if (r == k) cov += EPS;
            A[r * 7 + k] = cov;
            A[k * 7 + r] = cov;
        }
    }
    for (int j = 0; j < 7; ++j) {
        float d = A[j * 7 + j];
        for (int k = 0; k < j; ++k) { float ljk = A[j * 7 + k]; d -= ljk * ljk * A[k * 7 + k]; }
        d = max(d, 1e-6);
        A[j * 7 + j] = d;
        float inv_d = 1.0 / d;
        for (int i = j + 1; i < 7; ++i) {
            float s = A[i * 7 + j];
            for (int k = 0; k < j; ++k) s -= A[i * 7 + k] * A[j * 7 + k] * A[k * 7 + k];
            A[i * 7 + j] = s * inv_d;
        }
    }
    for (int i = 0; i < 7; ++i) {
        vec3 s = B[i];
        for (int k = 0; k < i; ++k) s -= A[i * 7 + k] * B[k];
        B[i] = s;
    }
    for (int i = 6; i >= 0; --i) {
        vec3 s = B[i] / A[i * 7 + i];
        for (int k = i + 1; k < 7; ++k) s -= A[k * 7 + i] * B[k];
        B[i] = s;
    }

    vec3 q = mu_p;
    for (int r = 0; r < 7; ++r) q += B[r] * (g_ds[r] - mu_g[r]);
    out_dec = vec4(max(q, vec3(0.0)), 1.0);
    out_min = vec4(b_min, 1.0);
    out_max = vec4(b_max, 1.0);
}
"""

FRAG_PASS2_BODY = """
uniform sampler2D u_dec;
uniform sampler2D u_min;
uniform sampler2D u_max;
uniform sampler2D u_norm;
uniform sampler2D u_alb;
uniform sampler2D u_depth;
uniform vec2 u_dst_res;
uniform int u_sharp_mode;  // 0:universal_boost, 2:mks2021, 3:edge_mks_bal, 4:edge_mks_strict, 5:edge_mks_gbuf, 6:amd_cas_mks
uniform int u_ar_mode;     // 0:unclamped, 1:reversible_karis, 2:asymmetric, 3:strict_2x2, 4:span_8pct
uniform int u_is_rev_karis;
uniform int u_colorspace;  // 0:linear_rgb, 1:ictcp_pq, 2:oklab
uniform float u_sharp_str; // 1.0 = +0% Boost (native), 3.0 = +200% Boost
uniform float u_sharp_rad; // 0.4 .. 2.5 px radius
uniform int u_has_norm;
uniform int u_has_alb;
uniform int u_has_depth;
uniform int u_show_mask;   // 1 = Hold 'M' X-Ray Mask View
in vec2 v_uv;
out vec4 out_final;

float luma(vec3 c) {
    return max(0.0, dot(c, vec3(0.2126, 0.7152, 0.0722)));
}

float perc_luma(vec3 c) {
    if (u_colorspace != 0) {
        return clamp(c.x, 0.0, 1.0);
    }
    float y = luma(c);
    return (u_is_rev_karis == 1) ? y : (y / (1.0 + y));
}

vec3 sample_dec_rad(vec2 uv, float dx, float dy) {
    vec2 offset = vec2(dx, dy) * u_sharp_rad / u_dst_res;
    return texture(u_dec, clamp(uv + offset, vec2(0.5) / u_dst_res, vec2(1.0) - vec2(0.5) / u_dst_res)).rgb;
}

vec3 eval_mks2021_rad(vec2 uv) {
    const float k[7] = float[7](-1.0/144.0, 6.0/144.0, -35.0/144.0, 204.0/144.0, -35.0/144.0, 6.0/144.0, -1.0/144.0);
    vec3 s = vec3(0.0);
    for (int dy = -3; dy <= 3; ++dy) {
        vec3 row = vec3(0.0);
        for (int dx = -3; dx <= 3; ++dx)
            row += sample_dec_rad(uv, float(dx), float(dy)) * k[dx + 3];
        s += row * k[dy + 3];
    }
    return s;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 dst_lim = ivec2(u_dst_res) - 1;
    vec3 base_c = texelFetch(u_dec, clamp(p, ivec2(0), dst_lim), 0).rgb;
    vec3 l_min = texelFetch(u_min, clamp(p, ivec2(0), dst_lim), 0).rgb;
    vec3 l_max = texelFetch(u_max, clamp(p, ivec2(0), dst_lim), 0).rgb;

    if (u_sharp_rad > 1.05) {
        int reach = int(ceil(u_sharp_rad - 0.5));
        for (int dy = -reach; dy <= reach; ++dy) {
            for (int dx = -reach; dx <= reach; ++dx) {
                ivec2 np = clamp(p + ivec2(dx, dy), ivec2(0), dst_lim);
                l_min = min(l_min, texelFetch(u_min, np, 0).rgb);
                l_max = max(l_max, texelFetch(u_max, np, 0).rgb);
            }
        }
    }

    float k_min = perc_luma(l_min);
    float k_max = perc_luma(l_max);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float py = perc_luma(texelFetch(u_dec, clamp(p + ivec2(dx, dy), ivec2(0), dst_lim), 0).rgb);
            k_min = min(k_min, py);
            k_max = max(k_max, py);
        }
    }
    float edge_contrast = clamp(k_max - k_min, 0.0, 1.0);

    float gbuf_edge_pass = 1.0;
    float alb_detail_boost = 1.0;
    vec2 uv_step = u_sharp_rad / u_dst_res;
    if (u_has_norm == 1) {
        vec3 nC = normalize(texture(u_norm, v_uv).rgb + vec3(1e-6));
        vec3 nN = normalize(texture(u_norm, v_uv + vec2(0.0, -uv_step.y)).rgb + vec3(1e-6));
        vec3 nS = normalize(texture(u_norm, v_uv + vec2(0.0, uv_step.y)).rgb + vec3(1e-6));
        vec3 nW = normalize(texture(u_norm, v_uv + vec2(-uv_step.x, 0.0)).rgb + vec3(1e-6));
        vec3 nE = normalize(texture(u_norm, v_uv + vec2(uv_step.x, 0.0)).rgb + vec3(1e-6));
        float min_dot = min(min(dot(nC, nN), dot(nC, nS)), min(dot(nC, nW), dot(nC, nE)));
        gbuf_edge_pass = smoothstep(0.55, 0.96, min_dot);
    }
    if (u_has_depth == 1) {
        float dC = texture(u_depth, v_uv).r;
        float dN = texture(u_depth, v_uv + vec2(0.0, -uv_step.y)).r;
        float dS = texture(u_depth, v_uv + vec2(0.0, uv_step.y)).r;
        float dW = texture(u_depth, v_uv + vec2(-uv_step.x, 0.0)).r;
        float dE = texture(u_depth, v_uv + vec2(uv_step.x, 0.0)).r;
        float depth_lap = max(abs(dN + dS - 2.0 * dC), abs(dW + dE - 2.0 * dC));
        gbuf_edge_pass *= 1.0 - smoothstep(0.015, 0.08, depth_lap);
    }
    if (u_has_alb == 1) {
        vec3 aC = texture(u_alb, v_uv).rgb;
        vec3 aAvg = 0.25 * (
            texture(u_alb, v_uv + vec2(0.0, -uv_step.y)).rgb +
            texture(u_alb, v_uv + vec2(0.0, uv_step.y)).rgb +
            texture(u_alb, v_uv + vec2(-uv_step.x, 0.0)).rgb +
            texture(u_alb, v_uv + vec2(uv_step.x, 0.0)).rgb
        );
        float alb_diff = length(aC - aAvg);
        alb_detail_boost = mix(0.70, 1.30, smoothstep(0.005, 0.12, alb_diff));
    }

    vec3 mks21_c = eval_mks2021_rad(v_uv);
    vec3 res = base_c;
    float vis_mask = 1.0;

    if (u_colorspace != 0) {
        // Perceptual Luma-Only MKS 2021 Sharpening: sharpens Lightness (.x) and keeps Chroma (.yz) hue-pure
        float edge_stop = (1.0 - smoothstep(0.05, 0.22, edge_contrast)) * gbuf_edge_pass;
        vis_mask = edge_stop;
        float sharp_x = mix(base_c.x, mks21_c.x, u_sharp_str * edge_stop);
        res = vec3(sharp_x, base_c.yz);
    } else if (u_sharp_mode == 2) {
        res = mix(base_c, mks21_c, u_sharp_str);
        vis_mask = clamp(u_sharp_str * 0.5, 0.0, 1.0);
    } else if (u_sharp_mode == 3) {
        float edge_stop = 1.0 - smoothstep(0.14, 0.48, edge_contrast);
        vis_mask = edge_stop;
        res = mix(base_c, mks21_c, u_sharp_str * edge_stop);
    } else if (u_sharp_mode == 4) {
        float edge_stop = (1.0 - smoothstep(0.05, 0.20, edge_contrast)) * gbuf_edge_pass;
        vis_mask = edge_stop;
        res = mix(base_c, mks21_c, u_sharp_str * edge_stop);
    } else if (u_sharp_mode == 5) {
        float luma_guard = 1.0 - smoothstep(0.20, 0.60, edge_contrast);
        float mask = gbuf_edge_pass * luma_guard * alb_detail_boost;
        vis_mask = clamp(mask, 0.0, 1.0);
        res = mix(base_c, mks21_c, clamp(u_sharp_str * mask, 0.0, 2.6));
    } else if (u_sharp_mode == 6) {
        vec3 cN = sample_dec_rad(v_uv, 0.0, -1.0);
        vec3 cS = sample_dec_rad(v_uv, 0.0, 1.0);
        vec3 cW = sample_dec_rad(v_uv, -1.0, 0.0);
        vec3 cE = sample_dec_rad(v_uv, 1.0, 0.0);
        float lN = perc_luma(cN), lS = perc_luma(cS), lW = perc_luma(cW), lE = perc_luma(cE), lC = perc_luma(base_c);
        float mn = min(min(min(lN, lS), min(lW, lE)), lC);
        float mx = max(max(max(lN, lS), max(lW, lE)), lC);
        float cas_amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, 1e-4), 0.0, 1.0));
        vis_mask = cas_amp * gbuf_edge_pass;
        vec3 cross_avg = 0.25 * (cN + cS + cW + cE);
        vec3 cas_detail = (base_c - cross_avg) * cas_amp * 1.15;
        vec3 mks_detail = (mks21_c - base_c) * cas_amp;
        res = base_c + (0.60 * mks_detail + 0.40 * cas_detail) * u_sharp_str * gbuf_edge_pass;
    } else {
        float edge_stop = (1.0 - smoothstep(0.12, 0.42, edge_contrast)) * gbuf_edge_pass;
        vis_mask = edge_stop;
        if (u_sharp_str > 1.001) {
            float boost_amount = u_sharp_str - 1.0;
            res = mix(base_c, mks21_c, boost_amount * edge_stop);
        }
    }

    if (u_show_mask == 1) {
        float bg_y = perc_luma(base_c);
        vec3 heat = mix(vec3(0.95, 0.12, 0.08), vec3(0.10, 0.92, 0.25), clamp(vis_mask, 0.0, 1.0));
        out_final = vec4(mix(vec3(bg_y), heat, 0.72), 1.0);
        return;
    }

    if (u_colorspace != 0) {
        // Asymmetric AR damping inside perceptual space (protects negative chroma/luma overshoot)
        vec3 under = max(vec3(0.0), l_min - res);
        res = res + 0.92 * under;
        vec3 over = max(vec3(0.0), res - l_max);
        res = res - 0.30 * over;
        res = max(decode_colorspace(res, u_colorspace), vec3(0.0));
    } else if (u_ar_mode == 1) {
        vec3 under = max(vec3(0.0), l_min - res);
        res = res + 0.94 * under;
        vec3 over = max(vec3(0.0), res - l_max);
        res = clamp(res - 0.35 * over, vec3(0.0), vec3(0.9999));
        float y_c = min(luma(res), 0.9999);
        res = res / max(1e-5, 1.0 - y_c);
    } else if (u_ar_mode == 2) {
        vec3 under = max(vec3(0.0), l_min - res);
        res = res + 0.92 * under;
        vec3 over = max(vec3(0.0), res - l_max);
        res = max(res - 0.30 * over, vec3(0.0));
    } else if (u_ar_mode == 3) {
        res = clamp(res, l_min, l_max);
    } else if (u_ar_mode == 4) {
        vec3 span = (l_max - l_min) * 0.08;
        res = clamp(res, max(l_min - span, vec3(0.0)), l_max + span);
    }

    out_final = vec4(max(res, vec3(0.0)), 1.0);
}
"""

# --- Fast GPU ACES Filmic + 2.2 Gamma + IGN Triangular Dither (Viewport Preview) ---
FRAG_TONEMAP_BODY = """
uniform sampler2D u_src;
uniform vec2 u_src_res;
uniform float u_gain;
out vec4 out_srgb;

void main() {
    ivec2 lim = ivec2(u_src_res) - 1;
    ivec2 p = clamp(ivec2(gl_FragCoord.xy), ivec2(0), lim);
    vec3 raw = texelFetch(u_src, p, 0).rgb;
    vec3 c = vec3(
        isnan(raw.r) ? 0.0 : (isinf(raw.r) ? (raw.r > 0.0 ? 65500.0 : 0.0) : raw.r),
        isnan(raw.g) ? 0.0 : (isinf(raw.g) ? (raw.g > 0.0 ? 65500.0 : 0.0) : raw.g),
        isnan(raw.b) ? 0.0 : (isinf(raw.b) ? (raw.b > 0.0 ? 65500.0 : 0.0) : raw.b)
    );
    c = max(c, vec3(0.0)) * u_gain;
    vec3 mapped = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), vec3(0.0), vec3(1.0));
    vec3 srgb = pow(mapped, vec3(1.0 / 2.2));
    vec2 xy = vec2(p);
    float ign1 = fract(52.9829189 * fract(0.06711056 * xy.x + 0.00583715 * xy.y));
    float ign2 = fract(52.9829189 * fract(0.06711056 * (xy.x + 17.0) + 0.00583715 * (xy.y + 31.0)));
    float dith = ((ign1 - ign2) * 0.5) / 255.0;
    out_srgb = vec4(clamp(srgb + dith, 0.0, 1.0), 1.0);
}
"""
