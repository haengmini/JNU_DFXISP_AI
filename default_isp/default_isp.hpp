#pragma once

#include <cstdint>

// =============================================================================
// File   : isppipeline/hls/include/default_isp.hpp
// Added  : 2026-08-06
// Function: default_ISP -- the reference "standard" ISP arm, restructured to
//           follow the AMD Vitis Vision Library L3 `isppipeline` example
//           (Xilinx/Vitis_Libraries, vision/L3/examples/isppipeline) in stage
//           ORDER and stage DOMAIN, instead of the hand-rolled ordering used by
//           RM_NORMAL_TONE in dfxisp_accel.cpp.
//
// Why this exists (see default_isp.md):
//   RM_NORMAL_TONE corrects in the RGB domain AFTER demosaic and uses fixed
//   white balance with an identity CCM. The Vitis Vision reference corrects in
//   the Bayer domain BEFORE demosaic, applies an adaptive AWB stage after it,
//   and uses a real color correction matrix. That is a structural difference,
//   not a constant difference, so a fair "standard ISP baseline" claim in the
//   paper needs this pipeline rather than RM_NORMAL_TONE.
//
// This module is ADDITIVE: RM_NORMAL_TONE, its golden vectors and the deployed
// BLC/checker decisions are untouched. Whether default_ISP replaces
// RM_NORMAL_TONE as the deployed normal-mode RM is a separate decision
// (STRATEGY.md open question #4).
//
// Stage order (Vitis Vision isppipeline; deviations documented in the .md):
//   RAW Bayer(12-bit)
//     -> (1) blackLevelCorrection   [Bayer domain, subtract + range rescale]
//     -> (2) gaincontrol            [Bayer domain, per-Bayer-position R/B gain]
//     -> (3) demosaicing            [RGGB -> RGB, 12-bit preserved]
//     -> (4) AWB                    [RGB domain, per-frame adaptive, bypassable]
//     -> (5) colorcorrectionmatrix  [RGB domain, real 3x3 Q8 matrix]
//     -> (6) quantization 12->8     [>>4; Vitis' optional dithering omitted]
//     -> (7) gammacorrection        [256-entry LUT, gamma 2.0, shared with the
//                                    rest of the repo for cross-comparability]
//     -> packed RGB888 0x00RRGGBB   [Vitis' rgb2yuyv output CSC omitted --
//                                    downstream DPU/detector consumes RGB]
//
// All arithmetic is integer and bit-exact against tools/gen_default_isp_golden.py.
// =============================================================================

// AWB mode. Mirrors the Vitis Vision example's `mode_reg` bit 0, which selects
// between the adaptive AWB path (fifo_awb) and a bypass (fifo_copy).
enum DefaultIspAwbMode : int {
    DEFAULT_ISP_AWB_OFF = 0,  // bypass: Bayer-domain gaincontrol only
    DEFAULT_ISP_AWB_ON = 1,   // adaptive per-frame gray-world AWB
};

// Development/analysis top: exposes the AWB mode switch.
//   raw_bayer : RAW Bayer RGGB, 12-bit values in uint16_t (W*H)
//   rgb_out   : packed RGB888 0x00RRGGBB (capacity >= W*H); shape is preserved
//   awb_mode  : DefaultIspAwbMode
// out_width/out_height always equal width/height (shape-preserving pipeline).
extern "C" void default_isp(
    const uint16_t* raw_bayer,
    uint32_t* rgb_out,
    int width,
    int height,
    int awb_mode,
    int* out_width,
    int* out_height);

// DFX Reconfigurable Module candidate. Port list is IDENTICAL (type, order,
// count) to rm_normal_tone_top / rm_low_light_tone_top so all three are valid
// implementations of the same Reconfigurable Partition slot -- DFX requires
// identical port signatures across RMs of one RP (SPEC.md §7). AWB is enabled.
extern "C" void rm_default_isp_top(
    const uint16_t* raw_bayer,
    uint32_t* rgb_out,
    int width,
    int height,
    int* out_width,
    int* out_height);
