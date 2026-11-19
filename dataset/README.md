# RAW 객체 검출 데이터셋

이 디렉터리에는 LOD와 PASCALRAW 테스트 분할, 그리고 소프트웨어 평가와 FPGA/HLS 실험에 사용하는 파생 데이터셋이 들어 있습니다. 원본 데이터와 모든 파생 데이터셋은 로컬 전용 데이터이며 Git에 커밋하지 않습니다. 파생 데이터셋은 아래 명령으로 다시 생성할 수 있습니다.

최종 업데이트: 2026년 8월 13일.

기준이 되는 설계 근거는 `isppipeline/hls/results/hw-dataset-geometry-2026-08-13.md`에 정리되어 있습니다. LOD/PASCALRAW 상호 보완 평가에 대한 추가 배경은 `isppipeline/hls/results/pascalraw-adapter-2026-07-13.md`에서 볼 수 있습니다.

## 데이터셋 다운로드

LOD와 PASCAL 데이터셋 파일은 다음 공유 Google Drive 폴더에서 받을 수 있습니다.

- [Google Drive 데이터셋 폴더](https://drive.google.com/drive/folders/1_TaV0f5hTGxb-TAWWwP04s9ykcJp_Q2G?usp=sharing)

각 데이터셋에는 원 배포처가 정한 라이선스와 이용 약관이 그대로 적용됩니다. 연구 등 비상업적 목적으로 사용하는 경우를 포함하여, 데이터를 사용하기 전에 원 라이선스와 인용 요건을 확인하십시오.

## 데이터셋 요약

| 디렉터리 | 용도 | 프레임 수 | RAW 형식 | 해상도 | 대략적 크기 |
|---|---|---:|---|---|---:|
| `LOD_test/` | 원본 LOD 테스트 분할 | 100 | Sony 카메라 RAW, 14-bit `.ARW` | 5496 x 3672 (가시 영역) | 2.0 GB |
| `PASCAL_test/` | 원본 PASCALRAW 테스트 분할 | 100 | Nikon 카메라 RAW, 12-bit `.NEF` | 6034 x 4012 (가시 영역) | 2.2 GB |
| `LOD_test_eval/` | 기존 소프트웨어 평가 입력 | 100 | Shift-8 uint16 RGGB | 5472 x 3648 | 5.3 GB |
| `LOD_test_hw/` | FPGA/HLS 입력 | 100 | 12-bit uint16 LE RGGB | 1368 x 912 | 240 MB |
| `PASCAL_test_hw/` | FPGA/HLS 입력 | 100 | 12-bit uint16 LE RGGB | 1368 x 912 | 240 MB |

LOD는 Sony RX100 VII로 촬영했으며 전부 야간 또는 저조도 장면으로 구성됩니다. 센서의 black level과 white level은 각각 800과 16380입니다. 저조도 checker의 recall을 측정하는 데 사용합니다.

PASCALRAW는 주간에 Nikon D3200으로 촬영했습니다. 센서의 black level과 white level은 0과 4095입니다. 주간 장면에서 저조도 오검출(false trigger)을 측정하여 LOD를 보완합니다.

두 센서 모두 원점에서 시작하는 RGGB Bayer 패턴을 사용합니다(`raw_pattern=[[0,1],[3,2]]`, `color_desc=RGBG`).

원본 테스트 디렉터리는 다음 프로젝트 클래스 ID를 사용합니다.

```text
0: person
1: car
2: bicycle
```

`*_hw` 라벨은 대신 0부터 시작하는 COCO-80 ID를 사용합니다: `0=person`, `1=bicycle`, `2=car`.

## 디렉터리 구조

```text
dataset/
├── LOD_test/
│   ├── raw/          # 100 original Sony ARW files
│   ├── images/       # 100 JPG files for model input
│   ├── labels/       # 100 YOLO-format TXT labels
│   └── labels.cache  # Local cache generated automatically by YOLO
├── PASCAL_test/
│   ├── raw/          # 100 original Nikon NEF files
│   ├── images/       # 100 JPG files for model input
│   ├── labels/       # 100 YOLO-format TXT labels
│   └── labels.cache  # Local cache generated automatically by YOLO
├── LOD_test_eval/    # Legacy full-resolution shift-8 software-evaluation data
├── LOD_test_hw/
│   ├── raw_bin/      # 100 fixed-size 12-bit FPGA input files
│   ├── labels/       # Crop-adjusted YOLO labels
│   ├── preview/      # Optional visual checks; not part of the data contract
│   └── meta.json     # Per-frame conversion and label-remapping metadata
├── PASCAL_test_hw/   # Same layout and binary contract as LOD_test_hw
├── prepare_lod_yolo.py
└── prepare_pascal_yolo.py
```

`raw`, `images`, `labels` 아래의 파일은 확장자를 뺀 파일명(stem)으로 서로 대응됩니다.

```text
# LOD example
raw/DSC01144.ARW
images/DSC01144.jpg
labels/DSC01144.txt

# PASCAL example
raw/2014_000924.nef
images/2014_000924.jpg
labels/2014_000924.txt
```

## LOD 데이터셋

LOD 테스트 분할의 `raw` 디렉터리에는 Sony 카메라로 촬영한 `.ARW` 파일이 들어 있습니다. 일반 영상 라이브러리로는 이 RAW 파일을 직접 열 수 없을 수 있으므로 `rawpy` 또는 다른 LibRaw 기반 도구를 사용하십시오.

`images` 아래의 JPG 파일은 객체 검출 모델이 읽을 수 있도록 RAW 데이터를 RGB로 변환하여 생성했습니다. 기존 전처리 파이프라인은 원본 종횡비를 유지하면서 가장 긴 변을 1280 픽셀로 조정합니다.

## PASCALRAW 데이터셋

PASCAL 테스트 분할의 `raw` 디렉터리에는 Nikon D3200으로 촬영한 12-bit `.NEF` 파일이 들어 있습니다. `images` 디렉터리에는 모델 입력과 시각화에 쓰는 대응 JPG 파일이 들어 있습니다.

PASCALRAW 전체 배포본은 약 150 GB이며 여러 개의 분할 TAR 파일로 배포됩니다. 이 프로젝트의 테스트 분할에는 선별한 평가용 영상 100장과 대응하는 라벨이 들어 있습니다.

## FPGA/HLS 데이터셋 계약

`LOD_test_hw`와 `PASCAL_test_hw`는 크기와 형식이 동일하므로 같은 파이프라인 너비, 높이, 버퍼 구성을 사용할 수 있습니다. 파생 파일의 모든 stem은 원본 프레임과 일대일로 대응됩니다.

```text
raw_bin/<stem>.bin   1368 x 912, uint16 little-endian, values 0..4095
                     12-bit RGGB with (0,0)=R, row-major
                     exactly 2,495,232 bytes per frame
labels/<stem>.txt    normalized YOLO: class cx cy width height
                     zero-indexed COCO-80 IDs: 0=person, 1=bicycle, 2=car
preview/<stem>.jpg   nearest-neighbor demosaic and gamma for visual checks only
meta.json            crop origin, black/white levels, and box clip/drop counts
```

| 속성 | `LOD_test_hw` | `PASCAL_test_hw` |
|---|---|---|
| 원본 → crop | 5496 x 3672에서 5472 x 3648로, 위치 (12, 12) | 6034 x 4012에서 5472 x 3648로, 위치 (280, 182) |
| 유지되는 화각 | 100% | 약 82% |
| 다운샘플링 | 4x 동일 색상 decimation | 4x 동일 색상 decimation |
| 라벨 박스 | 530개; 29개 clip, 0개 제외 | 162개; 33개 clip, 2개 제외 |
| 저조도 RM 출력 | 684 x 456 | 684 x 456 |

입력 너비를 1368 픽셀로 하면 binning 후 너비가 684가 되어, `dfxisp_accel.cpp`의 `row_r`가 두는 하드 상한 `MAX_BINNED_W=960` 아래로 유지됩니다. 또한 프레임당 크기가 약 40 MB에서 2.44 MB로 줄어듭니다.

동일 색상 decimation은 의도된 선택입니다. 4x4 동일 색상 평균을 쓰면 `RM_LOW_LIGHT`가 수행하는 2x2 binning과 비슷한 연산이 미리 적용되어 노이즈가 줄고, 2026년 7월 20일 배포 캘리브레이션(`BLC=2`, `dark16`, `DARK_RATIO_PCT=62`)이 무효가 됩니다. dark-ratio 보존도를 측정한 결과 최대 절대 차이는 LOD가 0.021 퍼센트포인트, PASCALRAW가 0.067 퍼센트포인트입니다.

가속기는 `RAW12_MAX=4095`인 12-bit 샘플을 입력으로 받습니다. 기존 shift-8 표현을 그대로 입력하지 마십시오. 값이 65280까지 커져 프레임 전체가 포화됩니다. shift-8 형태가 필요하면 `(v >> 4) << 8`로 만드십시오. 이 값은 기존 반올림 결과와 최대 1 code 차이가 납니다.

## 데이터셋 선택

| 작업 | 데이터셋 |
|---|---|
| HLS C 시뮬레이션, FPGA 실행, 타이밍 또는 자원 측정 | `LOD_test_hw` and `PASCAL_test_hw` |
| 동일한 geometry에서 LOD와 PASCALRAW 비교 | `LOD_test_hw` and `PASCAL_test_hw` |
| 2026년 7월 20일 이전에 생성한 결과와의 직접 비교 | `LOD_test_eval` |
| 새 파생 데이터셋 생성 또는 다른 crop/다운샘플링 방식 시험 | `LOD_test` and `PASCAL_test` |

`*_hw`의 절대 mAP를 기존 전체 해상도 결과와 직접 비교하지 마십시오. 해상도가 바뀌었고 PASCALRAW crop으로 화각의 약 18%가 제거되었기 때문입니다. 하드웨어 데이터셋은 실험 arm 간 상대 비교에 사용하십시오. 선정된 검출기는 YOLOv8n이며, 앞서 계획했던 YOLOv8s 교차 검증은 2026년 8월 13일에 철회되었습니다.

## YOLO 라벨 형식

TXT 라벨 파일의 각 줄은 객체 하나를 나타냅니다.

```text
class_id x_center y_center width height
```

좌표와 크기는 영상의 너비와 높이로 정규화되어 있으므로 `0.0`에서 `1.0` 사이의 값을 가집니다.

## RAW 파일 읽기

`rawpy`를 설치합니다.

```bash
pip install rawpy
```

ARW 또는 NEF 파일을 RGB로 변환하는 기본 예제입니다.

```python
from pathlib import Path

import rawpy
from PIL import Image

raw_path = Path("dataset/LOD_test/raw/DSC01144.ARW")

with rawpy.imread(str(raw_path)) as raw:
    rgb = raw.postprocess(
        use_camera_wb=True,
        no_auto_bright=False,
        output_bps=8,
    )

Image.fromarray(rgb).save("converted.jpg", quality=95)
```

같은 코드를 PASCALRAW `.NEF` 파일에도 사용할 수 있습니다.

## 모델 입력 시 유의 사항

- 현재 YOLOv8과 MobileNet SSDLite 학습 파이프라인은 JPG, PNG 같은 RGB 입력을 사용합니다.
- `.ARW` 또는 `.NEF`를 직접 입력하려면 RAW 디코딩을 수행하고 RGB 변환 또는 Bayer 텐서 변환을 하는 데이터 로더가 필요합니다.
- 기존 YOLO TXT 라벨을 그대로 사용하려면 RAW와 JPG 파일의 stem이 같아야 합니다.
- RAW 변환에서 crop 없이 비율을 유지한 크기 조정만 적용하면 YOLO 좌표는 그대로 유효합니다.
- crop을 적용하면 bounding box 좌표를 다시 계산해야 합니다.

## FPGA/HLS 데이터셋 재생성

`isppipeline/hls` 디렉터리에서 builder를 실행합니다.

```bash
cd isppipeline/hls
python3 tools/build_hw_dataset.py --selftest

python3 tools/build_hw_dataset.py \
    --raw-dir ../../dataset/LOD_test/raw \
    --labels-dir ../../dataset/LOD_test/labels \
    --out ../../dataset/LOD_test_hw --preview 3

python3 tools/build_hw_dataset.py \
    --raw-dir ../../dataset/PASCAL_test/raw \
    --labels-dir ../../dataset/PASCAL_test/labels \
    --out ../../dataset/PASCAL_test_hw --preview 3
```

100프레임 변환에는 LOD가 약 40초, PASCALRAW가 약 60초 걸립니다. `rawpy`와 `numpy`가 필요하며, `Pillow`는 preview에만 필요합니다. builder는 decimation 후에도 `(0,0)` quad가 RGGB로 유지되는지 해석적으로 검증하며, Bayer phase가 맞지 않으면 해당 프레임에서 실패합니다.

`--limit N`은 앞의 N개 프레임만 처리합니다. `--min-visible F`는 crop 후 보이는 면적이 원래 면적의 F 비율 미만인 박스를 제외합니다. 기본값은 0.25이며, 검출할 수 없는 ground-truth 조각이 false negative가 되는 것을 막습니다.

## 원본 데이터 정렬 관련 알려진 사항

- LOD 라벨은 5496 x 3672 가시 영역 영상을 기준으로 하지만 기존 `LOD_test_eval/raw_bin` 데이터는 5472 x 3648이라 0.2-0.4%의 오프셋이 생깁니다. 하드웨어 builder는 라벨을 가시 영역 좌표로 해석한 뒤 crop 좌표로 다시 매핑하여 이를 해결합니다.
- PASCALRAW 라벨은 정확히 600 x 400, 3:2 기준이지만 RAW 영상의 종횡비는 1.5040입니다. 그 결과 생기는 0.27%의 비등방성은 `PASCAL_test_hw`에 그대로 남아 있습니다.
- `LOD_test_eval/convert_errors.log`는 비어 있습니다. 기존 변환에서는 오류가 보고되지 않았습니다.

## 캐시 파일

`labels.cache`는 라벨 탐색 속도를 높이기 위해 Ultralytics YOLO가 자동으로 생성합니다. 원본 데이터셋의 일부가 아니며 삭제해도 안전하게 다시 생성됩니다. 데이터셋을 압축하거나 공유할 때는 제외하는 것을 권장합니다.
