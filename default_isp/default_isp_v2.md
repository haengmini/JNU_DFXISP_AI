# default_ISP — file overview

`src/default_isp.cpp` and `include/default_isp.hpp` implement default_ISP, a
standard ISP arm restructured to follow the AMD Vitis Vision L3 `isppipeline`
example's stage order and stage domain (instead of the hand-rolled ordering
used by the existing `RM_NORMAL_TONE` in `dfxisp_accel.cpp`).

`default_isp.hpp` declares two entry points:
- `default_isp(...)` — a development/analysis top that exposes the AWB mode
  switch (on/bypass) as a runtime argument.
- `rm_default_isp_top(...)` — a fixed-AWB-on top with the same port signature
  (type, order, count) as the other RM tops, making it a drop-in DFX
  Reconfigurable Module candidate for the same Reconfigurable Partition slot.

`default_isp.cpp` contains the integer-only implementation of both, bit-exact
against the Python golden. The rest of this document covers the pipeline
comparison, intentional deviations, constants, implementation notes, and
verification status.

## 1. Pipeline comparison (Vitis Vision vs. the two arms)

| # | Vitis Vision `isppipeline` | **default_ISP (new)** |
|---|---|---|
| 1 | blackLevelCorrection — **Bayer** | ✅ same (subtract + range restore) |
| 2 | gaincontrol — **Bayer**, per R/B position | ✅ same (Q8 286/307) |
| 3 | demosaicing | ✅ RGGB bilinear | RGGB bilinear |
| 4 | AWB — **RGB, per-frame adaptive** | ✅ gray-world adaptive (bypassable) |
| 5 | colorcorrectionmatrix | ✅ **real 3×3 Q8 matrix** |
| 6 | quantization & dithering | △ `>>4` only (dithering omitted) |
| 7 | gammacorrection (LUT) | ✅ same LUT (γ2.0) |
| 8 | rgb2yuyv (output CSC) | ✗ intentionally omitted → RGB888 |
| — | (gain) | none — exposure gain is specific to the tone RM |

## 2. Three intentional deviations (where we differ from Vitis, with rationale)

1. **Output CSC (rgb2yuyv) omitted** — downstream is a DPU/detector that
   consumes RGB888 as-is (keeps the 32-bit AXI-aligned packing from SPEC.md
   §5.1). Switching to YUYV would require changing the whole evaluation path
   for no benefit.
2. **Dithering omitted** — Vitis' quantization & dithering is an optional
   stage that reduces banding on bit-depth reduction. No effect on detection
   mAP has been confirmed, so only the `>>4` truncation is kept (can be added
   later if needed).
3. **AWB statistics source** — Vitis derives gain from the **previous frame's
   histogram** (1-frame delay, double-buffered). default_ISP uses the
   **current frame's Bayer-site average (gray-world)**: no frame buffer is
   needed and there is no delay, at the cost of reading the raw data a second
   time (§4). The statistic itself is also **simpler** — gray-world instead of
   histogram normalization. Read this as "an adaptive AWB stage at the same
   position with the same role," not "the same algorithm as Vitis."

## 3. Constants

| Stage | Constant | Value | Rationale |
|---|---|---|---|
| (1) BLC | `BLC_LEVEL12` | 32 (= 8-bit 2 << 4) | 2026-07-20 real-RAW recalibration deployed value |
| (1) BLC | `BLC_MUL_Q8` | 258 | `round(256 × 4095/(4095−32))` — restores the range lost by subtraction (Vitis' approach) |
| (2) gain | `GAIN_R_Q8` / `GAIN_B_Q8` | 286 / 307 | same as the existing arm's WB values — the two arms differ in **where** the gain is applied, not its **magnitude** |
| (4) AWB | clamp | Q8 [64, 1024] | 0.25×–4× |
| (5) CCM | 3×3 Q8 | row sum = 256 | preserves neutral gray. **Placeholder until sensor calibration** — do not claim color accuracy |
| (7) gamma | LUT | γ2.0 `isqrt(255v)` | byte-identical to `dfxisp_accel.cpp` (so the tone axis is comparable across arms) |

## 4. Implementation notes

- **BLC/gain are applied on-the-fly, without a buffer**: correction is applied
  each time the demosaic window reads a pixel. Because the correction is
  pointwise, the result is the same and no frame buffer is needed, but the
  same pixel can be recorrected up to 9 times (compute duplication vs. no
  storage trade-off).
- **AWB reads the raw data a second time** (a statistics pass). Combined with
  the checker's full scan, this increases gmem0 reads per frame — factor this
  into the bandwidth budget at implementation time.
- **Accumulator width**: 1920×1080×4095 ≈ 8.5e9 → a 64-bit accumulator is
  used.
- Negative CCM coefficients can make the accumulated value negative, so it is
  **floored to 0 before the shift** — this avoids C++'s implementation-defined
  behavior for right-shifting negative values and keeps the result bit-exact
  with the Python golden.

## 5. Verification

| Gate | Status |
|---|---|
| Python golden ↔ C++ bit-exact (`make default-isp-verify`) | ✅ 528 px, 10 cases (flat/gradient/color-cast/saturated/odd & 1×1) |
| BLC precedes in the Bayer domain (below pedestal → pure black) | ✅ |
| AWB adapts correctly (reduces channel imbalance under color cast) | ✅ |
| CCM preserves neutrality (row sum 256 → flat-input channel deviation ≤ 8) | ✅ |
| No RGB8 overflow on saturated input | ✅ |
| DFX contract (`rm_default_isp_top` 6-argument, output identical to `default_isp(AWB_ON)`) | ✅ |
