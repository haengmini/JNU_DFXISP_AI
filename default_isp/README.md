# Default ISP — 파일 개요

`src/default_isp.cpp`와 `include/default_isp.hpp`는 default_ISP를 구현한다.
default_ISP는 AMD Vitis Vision L3 `isppipeline` 예제의 스테이지 순서와
스테이지 도메인을 따르도록 재구성한 표준 ISP arm이다(`dfxisp_accel.cpp`의
기존 `RM_NORMAL_TONE`이 쓰는 직접 만든 순서를 대신한다).

`default_isp.hpp`는 두 개의 진입점을 선언한다.
- `default_isp(...)` — AWB 모드 스위치(on/bypass)를 런타임 인자로 노출하는
  개발/분석용 top.
- `rm_default_isp_top(...)` — AWB를 켠 상태로 고정한 top. 다른 RM top과 포트
  시그니처(타입, 순서, 개수)가 같아서 동일한 Reconfigurable Partition 슬롯에
  바로 넣을 수 있는 DFX Reconfigurable Module 후보다.

`default_isp.cpp`는 두 top의 정수 전용 구현을 담고 있으며 Python golden과
bit-exact이다. 이 문서의 나머지 부분은 파이프라인 대조, 의도적 편차, 상수,
구현 노트, 검증 상태를 다룬다.

## 1. 파이프라인 대조 (Vitis Vision vs 두 arm)

| # | Vitis Vision `isppipeline` | **default_ISP (신규)** |
|---|---|---|
| 1 | blackLevelCorrection — **Bayer** | ✅ 동일 (감산 + 레인지 복원) |
| 2 | gaincontrol — **Bayer**, R/B 위치별 | ✅ 동일 (Q8 286/307) |
| 3 | demosaicing | ✅ RGGB bilinear | RGGB bilinear |
| 4 | AWB — **RGB, 프레임 적응** | ✅ gray-world 적응(bypass 가능) |
| 5 | colorcorrectionmatrix | ✅ **실제 3×3 Q8 행렬** |
| 6 | quantization & dithering | △ `>>4`만 (디더링 생략) |
| 7 | gammacorrection (LUT) | ✅ 동일 LUT(γ2.0) |
| 8 | rgb2yuyv (출력 CSC) | ✗ 의도적 생략 → RGB888 |
| — | (게인) | 없음 — 노출 게인은 tone RM 고유 |

## 2. 의도적 편차 3건 (Vitis와 다르게 한 것, 근거 포함)

1. **출력 CSC(rgb2yuyv) 생략** — 하류가 DPU/검출기라 RGB888을 그대로
   받는다(SPEC.md §5.1의 AXI 정렬용 32-bit 패킹 유지). YUYV로 바꾸면 평가
   경로 전체를 바꿔야 하고 이득이 없다.
2. **디더링 생략** — Vitis의 quantization & dithering은 비트심도 축소 시
   밴딩을 줄이는 옵션 단계다. 검출 mAP에 영향이 있다고 확인된 바 없어
   `>>4` 절단만 남겼다(향후 필요 시 추가 가능).
3. **AWB 통계원** — Vitis는 **직전 프레임 히스토그램**(1프레임 지연,
   double-buffer)으로 게인을 만든다. default_ISP는 **현재 프레임의 Bayer
   사이트 평균(gray-world)** 을 쓴다: 프레임 버퍼가 필요 없고 지연이 없는
   대신 raw를 한 번 더 읽는다(§4). 통계량 종류도 히스토그램 정규화 대신
   gray-world라 **더 단순하다**. "Vitis와 동일 알고리즘"이 아니라 "동일
   위치·동일 역할의 적응형 AWB 스테이지"로 이해해야 한다.

## 3. 상수

| 스테이지 | 상수 | 값 | 근거 |
|---|---|---|---|
| (1) BLC | `BLC_LEVEL12` | 32 (= 8-bit 2 << 4) | 2026-07-20 실 RAW 재보정 배포값 |
| (1) BLC | `BLC_MUL_Q8` | 258 | `round(256 × 4095/(4095−32))` — 감산으로 잃은 레인지 복원(Vitis 방식) |
| (2) gain | `GAIN_R_Q8` / `GAIN_B_Q8` | 286 / 307 | 기존 arm의 WB 값과 동일 — 두 arm은 게인 **크기**가 아니라 **적용 위치**가 다르다 |
| (4) AWB | 클램프 | Q8 [64, 1024] | 0.25×–4× |
| (5) CCM | 3×3 Q8 | 행 합 = 256 | 중성 회색 보존. **센서 캘리브레이션 전까지는 placeholder** — 색정확도 주장 금지 |
| (7) gamma | LUT | γ2.0 `isqrt(255v)` | `dfxisp_accel.cpp`와 byte-identical(arm 간 tone 축 비교 가능하게) |

## 4. 구현 노트

- **BLC/gain은 버퍼 없이 on-the-fly로 적용한다**: demosaic 윈도우가 픽셀을
  읽을 때마다 보정을 적용한다. 보정이 pointwise라 결과는 동일하고 프레임
  버퍼가 필요 없지만, 같은 픽셀이 최대 9회 재보정될 수 있다(연산 중복 ↔
  저장 없음 트레이드오프).
- **AWB는 raw를 한 번 더 읽는다**(통계 패스). checker의 전수 스캔과 합치면
  프레임당 gmem0 읽기 횟수가 늘어난다 — 구현 시 대역폭 예산에 반영해야 한다.
- **누산기 폭**: 1920×1080×4095 ≈ 8.5e9 → 64-bit 누산기를 사용한다.
- CCM 음수 계수 때문에 누산값이 음수가 될 수 있어 **shift 전에 0으로
  floor**한다. 이렇게 하면 C++의 음수 우측 시프트 구현정의 동작을 피하고
  결과가 Python golden과 bit-exact로 유지된다.

## 5. 검증

| 게이트 | 상태 |
|---|---|
| Python golden ↔ C++ bit-exact (`make default-isp-verify`) | ✅ 528 px, 10 케이스(평탄/그라디언트/색캐스트/포화/홀수·1×1) |
| BLC가 Bayer 도메인에서 선행 (pedestal 이하 → 순흑) | ✅ |
| AWB 적응 동작 (색캐스트에서 채널 불균형 감소) | ✅ |
| CCM 중성 보존 (행 합 256 → 평탄 입력 채널 편차 ≤ 8) | ✅ |
| 포화 입력에서 RGB8 오버플로 없음 | ✅ |
| DFX 계약 (`rm_default_isp_top` 6-인자, `default_isp(AWB_ON)`과 동일 출력) | ✅ |
