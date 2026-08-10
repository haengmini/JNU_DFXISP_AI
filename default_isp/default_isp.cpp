#include "default_isp.hpp"

#include <cstdint>

// =============================================================================
// default_ISP -- standard ISP arm following the AMD Vitis Vision L3
// `isppipeline` stage order and domains. Integer-only; bit-exact mirror of
// tools/gen_default_isp_golden.py. Structure, deviations from the Vitis
// reference, and the constant rationale: default_isp.md (same directory).
// =============================================================================

namespace {

constexpr int RAW12_MAX = 4095;

// --- (1) blackLevelCorrection -------------------------------------------------
// Vitis' blackLevelCorrection subtracts the pedestal AND rescales so the range
// lost to the subtraction is restored (out = (in - blc) * mul). The repo's
// deployed black level is 2 in 8-bit terms = 32 in this 12-bit domain
// (recalibrated 2026-07-20 on real RAW; see dfxisp_accel.md §2).
constexpr int BLC_LEVEL12 = 2 << 4;                 // 32
// mul = round(256 * 4095 / (4095 - 32)) = 258  (Q8)
constexpr int BLC_MUL_Q8 = 258;

// --- (2) gaincontrol (Bayer domain) ------------------------------------------
// Vitis applies the sensor R/B gains per Bayer position BEFORE demosaic; green
// is the reference channel. Same Q8 values the repo already uses for its
// RGB-domain white balance, so the two arms differ in WHERE the gain is
// applied, not in how much (default_isp.md §3).
constexpr int GAIN_R_Q8 = 286;
constexpr int GAIN_B_Q8 = 307;

// --- (4) AWB ------------------------------------------------------------------
// Gray-world per-frame gains, clamped to [0.25x, 4x] in Q8.
constexpr int AWB_GAIN_MIN_Q8 = 64;
constexpr int AWB_GAIN_MAX_Q8 = 1024;

// --- (5) colorcorrectionmatrix ------------------------------------------------
// Real 3x3 Q8 matrix (rows sum to 256, so neutral gray is preserved) instead of
// RM_NORMAL_TONE's identity placeholder. Mild saturation recovery; MUST be
// replaced by a sensor-calibrated matrix before any color-accuracy claim.
constexpr int CCM_Q8[3][3] = {
    {288, -24, -8},
    {-24, 296, -16},
    {-16, -32, 304},
};

// --- (7) gammacorrection ------------------------------------------------------
// gamma 2.0 as floor(sqrt(255*v)); byte-identical to the table in
// dfxisp_accel.cpp so default_ISP and RM_NORMAL_TONE stay comparable on the
// tone axis (each RM owns its own ROM in silicon anyway).
static const uint8_t GAMMA2_LUT[256] = {
      0, 15, 22, 27, 31, 35, 39, 42, 45, 47, 50, 52, 55, 57, 59, 61,
     63, 65, 67, 69, 71, 73, 74, 76, 78, 79, 81, 82, 84, 85, 87, 88,
     90, 91, 93, 94, 95, 97, 98, 99,100,102,103,104,105,107,108,109,
    110,111,112,114,115,116,117,118,119,120,121,122,123,124,125,126,
    127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,141,
    142,143,144,145,146,147,148,148,149,150,151,152,153,153,154,155,
    156,157,158,158,159,160,161,162,162,163,164,165,165,166,167,168,
    168,169,170,171,171,172,173,174,174,175,176,177,177,178,179,179,
    180,181,182,182,183,184,184,185,186,186,187,188,188,189,190,190,
    191,192,192,193,194,194,195,196,196,197,198,198,199,200,200,201,
    201,202,203,203,204,205,205,206,206,207,208,208,209,210,210,211,
    211,212,213,213,214,214,215,216,216,217,217,218,218,219,220,220,
    221,221,222,222,223,224,224,225,225,226,226,227,228,228,229,229,
    230,230,231,231,232,233,233,234,234,235,235,236,236,237,237,238,
    238,239,240,240,241,241,242,242,243,243,244,244,245,245,246,246,
    247,247,248,248,249,249,250,250,251,251,252,252,253,253,254,255,
};

static inline int clamp_i(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static inline uint32_t pack_rgb(uint8_t r, uint8_t g, uint8_t b) {
    return (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
}

// Bayer position -> gaincontrol factor (RGGB: (0,0)=R (0,1)=G (1,0)=G (1,1)=B).
static inline int bayer_gain_q8(int x, int y) {
#pragma HLS INLINE
    const bool even_y = (y & 1) == 0;
    const bool even_x = (x & 1) == 0;
    if (even_y && even_x) return GAIN_R_Q8;   // R site
    if (!even_y && !even_x) return GAIN_B_Q8; // B site
    return 256;                               // G sites (reference channel)
}

// Stages (1)+(2) applied pointwise in the Bayer domain. Done on the fly at each
// window read rather than into a full-frame buffer: the correction is pointwise
// so this is equivalent, and it keeps the module storage-free (default_isp.md §4).
static inline int corrected_bayer(const uint16_t* raw, int width, int height, int x, int y) {
#pragma HLS INLINE
    const int cx = x < 0 ? 0 : (x >= width ? width - 1 : x);
    const int cy = y < 0 ? 0 : (y >= height ? height - 1 : y);
    int v = raw[cy * width + cx];
    // (1) black level: subtract pedestal, then restore range
    v = v > BLC_LEVEL12 ? v - BLC_LEVEL12 : 0;
    v = clamp_i((v * BLC_MUL_Q8) >> 8, 0, RAW12_MAX);
    // (2) gain control (Bayer domain, per position)
    v = clamp_i((v * bayer_gain_q8(cx, cy)) >> 8, 0, RAW12_MAX);
    return v;
}

// (3) demosaicing: RGGB bilinear, 12-bit preserved (no >>4 here). Same
// interpolation class as the rest of the repo so the arms differ in pipeline
// structure, not in demosaic quality.
static void demosaic_rggb12(const uint16_t* raw, int width, int height, int x, int y,
                            int& r12, int& g12, int& b12) {
#pragma HLS INLINE
    int win[3][3];
    for (int wy = 0; wy < 3; ++wy)
        for (int wx = 0; wx < 3; ++wx)
            win[wy][wx] = corrected_bayer(raw, width, height, x + wx - 1, y + wy - 1);

    const bool even_y = (y & 1) == 0;
    const bool even_x = (x & 1) == 0;
    const int c = win[1][1];

    if (even_y && even_x) {          // R site
        r12 = c;
        g12 = (win[1][0] + win[1][2] + win[0][1] + win[2][1]) / 4;
        b12 = (win[0][0] + win[0][2] + win[2][0] + win[2][2]) / 4;
    } else if (even_y && !even_x) {  // G on R row
        g12 = c;
        r12 = (win[1][0] + win[1][2]) / 2;
        b12 = (win[0][1] + win[2][1]) / 2;
    } else if (!even_y && even_x) {  // G on B row
        g12 = c;
        r12 = (win[0][1] + win[2][1]) / 2;
        b12 = (win[1][0] + win[1][2]) / 2;
    } else {                         // B site
        b12 = c;
        g12 = (win[1][0] + win[1][2] + win[0][1] + win[2][1]) / 4;
        r12 = (win[0][0] + win[0][2] + win[2][0] + win[2][2]) / 4;
    }
}

// (4) AWB statistics. Vitis derives them from the previous frame's histogram
// (1-frame lag, double-buffered); we accumulate this frame's Bayer-site means
// instead -- same gray-world statistic, no frame buffer, no lag. Cost: one
// extra read pass over raw_bayer (default_isp.md §4).
// Green is the reference channel, so only the R and B gains are returned --
// a green gain would be 256 (identity) by construction.
static void awb_gains_q8(const uint16_t* raw, int width, int height,
                         int& gain_r_q8, int& gain_b_q8) {
    unsigned long long sum_r = 0, sum_g = 0, sum_b = 0;
    unsigned long long cnt_r = 0, cnt_g = 0, cnt_b = 0;

    for (int y = 0; y < height; ++y) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=1080
        for (int x = 0; x < width; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=1920
            const int v = corrected_bayer(raw, width, height, x, y);
            const bool even_y = (y & 1) == 0;
            const bool even_x = (x & 1) == 0;
            if (even_y && even_x) {
                sum_r += static_cast<unsigned long long>(v);
                ++cnt_r;
            } else if (!even_y && !even_x) {
                sum_b += static_cast<unsigned long long>(v);
                ++cnt_b;
            } else {
                sum_g += static_cast<unsigned long long>(v);
                ++cnt_g;
            }
        }
    }

    const int mean_r = cnt_r ? static_cast<int>(sum_r / cnt_r) : 0;
    const int mean_g = cnt_g ? static_cast<int>(sum_g / cnt_g) : 0;
    const int mean_b = cnt_b ? static_cast<int>(sum_b / cnt_b) : 0;

    gain_r_q8 = (mean_r > 0 && mean_g > 0)
                    ? clamp_i((mean_g * 256) / mean_r, AWB_GAIN_MIN_Q8, AWB_GAIN_MAX_Q8)
                    : 256;
    gain_b_q8 = (mean_b > 0 && mean_g > 0)
                    ? clamp_i((mean_g * 256) / mean_b, AWB_GAIN_MIN_Q8, AWB_GAIN_MAX_Q8)
                    : 256;
}

// (5) colorcorrectionmatrix. Negative coefficients can drive the accumulator
// below zero; it is floored before the shift so C++ and the Python golden agree
// bit-for-bit without relying on arithmetic-shift-of-negative behavior.
static inline int ccm_channel(int row, int r12, int g12, int b12) {
#pragma HLS INLINE
    int acc = CCM_Q8[row][0] * r12 + CCM_Q8[row][1] * g12 + CCM_Q8[row][2] * b12;
    if (acc < 0) acc = 0;
    return clamp_i(acc >> 8, 0, RAW12_MAX);
}

// (6) quantization 12->8 + (7) gamma LUT.
static inline uint8_t quantize_gamma(int v12) {
#pragma HLS INLINE
    return GAMMA2_LUT[clamp_i(v12, 0, RAW12_MAX) >> 4];
}

static void run_default_isp(const uint16_t* raw_bayer, uint32_t* rgb_out,
                            int width, int height, int awb_mode) {
    int awb_r = 256, awb_b = 256;
    if (awb_mode == DEFAULT_ISP_AWB_ON) {
        awb_gains_q8(raw_bayer, width, height, awb_r, awb_b);
    }

    for (int y = 0; y < height; ++y) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=1080
        for (int x = 0; x < width; ++x) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=1920
            int r12 = 0, g12 = 0, b12 = 0;
            demosaic_rggb12(raw_bayer, width, height, x, y, r12, g12, b12);  // (1)(2)(3)

            r12 = clamp_i((r12 * awb_r) >> 8, 0, RAW12_MAX);                 // (4)
            b12 = clamp_i((b12 * awb_b) >> 8, 0, RAW12_MAX);   // g12: reference channel

            const int rc = ccm_channel(0, r12, g12, b12);                    // (5)
            const int gc = ccm_channel(1, r12, g12, b12);
            const int bc = ccm_channel(2, r12, g12, b12);

            rgb_out[y * width + x] = pack_rgb(quantize_gamma(rc),            // (6)(7)
                                              quantize_gamma(gc),
                                              quantize_gamma(bc));
        }
    }
}

}  // namespace

extern "C" void default_isp(
    const uint16_t* raw_bayer, uint32_t* rgb_out, int width, int height, int awb_mode,
    int* out_width, int* out_height) {
#pragma HLS INTERFACE m_axi port=raw_bayer offset=slave bundle=gmem0 depth=2048
#pragma HLS INTERFACE m_axi port=rgb_out offset=slave bundle=gmem1 depth=2048
#pragma HLS INTERFACE s_axilite port=raw_bayer bundle=control
#pragma HLS INTERFACE s_axilite port=rgb_out bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=awb_mode bundle=control
#pragma HLS INTERFACE s_axilite port=out_width bundle=control
#pragma HLS INTERFACE s_axilite port=out_height bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if (!raw_bayer || !rgb_out || width <= 0 || height <= 0) {
        if (out_width) *out_width = 0;
        if (out_height) *out_height = 0;
        return;
    }
    run_default_isp(raw_bayer, rgb_out, width, height, awb_mode);
    if (out_width) *out_width = width;
    if (out_height) *out_height = height;
}

// Port list matches rm_normal_tone_top / rm_low_light_tone_top exactly so this
// is a drop-in RM for the same Reconfigurable Partition slot (SPEC.md §7).
extern "C" void rm_default_isp_top(
    const uint16_t* raw_bayer, uint32_t* rgb_out, int width, int height,
    int* out_width, int* out_height) {
#pragma HLS INTERFACE m_axi port=raw_bayer offset=slave bundle=gmem0 depth=2048
#pragma HLS INTERFACE m_axi port=rgb_out offset=slave bundle=gmem1 depth=2048
#pragma HLS INTERFACE s_axilite port=raw_bayer bundle=control
#pragma HLS INTERFACE s_axilite port=rgb_out bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=out_width bundle=control
#pragma HLS INTERFACE s_axilite port=out_height bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if (!raw_bayer || !rgb_out || width <= 0 || height <= 0) {
        if (out_width) *out_width = 0;
        if (out_height) *out_height = 0;
        return;
    }
    run_default_isp(raw_bayer, rgb_out, width, height, DEFAULT_ISP_AWB_ON);
    if (out_width) *out_width = width;
    if (out_height) *out_height = height;
}
