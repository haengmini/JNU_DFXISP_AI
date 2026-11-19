# JNU DFXISP AI

LOD와 PASCALRAW 데이터셋으로 RAW 영상 처리 파이프라인을 평가하는 객체 검출 프로젝트입니다. 이 저장소에는 YOLOv8n과 SSDLite-MobileNetV3 모델 가중치, 실험 결과, DFX 연동용 C++ ISP 구현이 들어 있습니다.

## 프로젝트 범위

이 프로젝트는 두 RAW 영상 데이터셋에서 경량 객체 검출 모델의 성능을 비교합니다.

- **LOD / RAW-NOD (Sony):** Sony `.ARW` 형식으로 촬영한 저조도 도로 영상
- **PASCALRAW (Nikon):** 12-bit Nikon `.NEF` 형식으로 촬영한 주간 영상

두 데이터셋은 다음 세 가지 검출 클래스를 사용합니다.

```text
0: person
1: car
2: bicycle
```

검출 모델은 RAW 파일을 변환한 RGB JPEG 영상으로 학습합니다. 대응하는 RAW 파일은 ISP 및 RAW 도메인 평가를 위해 보관합니다.

## 저장소 구조

```text
.
├── .gitignore                      # Git ignore rules
├── default_isp/
│   ├── default_isp.cpp            # Integer-based default ISP implementation
│   ├── default_isp.hpp            # ISP interface and DFX top declaration
│   └── README.md                  # Pipeline design and verification document
├── models/
│   ├── mobilenet/
│   │   └── ssdlite-mobilenetv3.pt # Final SSDLite-MobileNetV3 weights
│   └── yolov8n/
│       └── yolov8n.pt             # Final YOLOv8n weights
└── README.md                       # Project overview and usage information
```

## 데이터셋 다운로드

### LOD / RAW-NOD

LOD는 Sony 카메라로 촬영한 저조도 도로 장면 데이터셋입니다. 원본 영상은 `.ARW` RAW 형식으로 저장되어 있으며, 야간 환경에서 촬영한 사람, 자동차, 자전거가 포함되어 있습니다.

### PASCALRAW

PASCALRAW는 Nikon 카메라로 촬영한 주간 장면 데이터셋입니다. 원본 영상은 12-bit `.NEF` RAW 형식으로 저장되어 있습니다. LOD와 마찬가지로 person, car, bicycle을 객체 검출 클래스로 사용합니다.

RAW 데이터셋과 준비된 테스트 분할은 다음 공유 Google Drive 폴더에서 받을 수 있습니다.

- [LOD 및 PASCALRAW 데이터셋](https://drive.google.com/drive/folders/1_TaV0f5hTGxb-TAWWwP04s9ykcJp_Q2G?usp=sharing)

전체 RAW 데이터셋과 데이터셋 아카이브는 용량 문제로 Git에 저장하지 않습니다.

## 실험 결과

LOD와 PASCALRAW 지표는 각 데이터셋의 테스트 영상 100장으로 측정했습니다. Mixed 지표는 두 데이터셋을 합친 검증 세트로 측정했습니다. MobileNet의 Precision, Recall, F1은 confidence threshold 0.25, IoU threshold 0.50을 사용합니다. YOLO 지표는 기록된 mAP50-95가 가장 높은 epoch의 값입니다. 체크포인트에 기록되지 않은 값은 `—`로 표시합니다.

| 데이터셋 | 검출 모델 | Precision | Recall | F1 | mAP50 | mAP50-95 | 처리량 |
|---|---|---:|---:|---:|---:|---:|---:|
| LOD | SSDLite-MobileNetV3 | 0.6915 | 0.4736 | 0.5622 | 0.5114 | 0.2980 | 60.4 FPS, batch 8 |
| PASCALRAW | SSDLite-MobileNetV3 | 0.9338 | 0.8598 | 0.8952 | 0.9210 | 0.6775 | 79.1 FPS, batch 8 |
| LOD | YOLOv8n | 0.8695 | 0.6381 | — | 0.7161 | 0.4923 | — |
| PASCALRAW | YOLOv8n | 0.9710 | 0.9069 | — | 0.9448 | 0.7713 | — |
| Mixed (LOD + PASCALRAW) | SSDLite-MobileNetV3 | — | — | — | 0.6155 | 0.3929 | — |
| Mixed (LOD + PASCALRAW) | YOLOv8n | 0.9232 | 0.6370 | — | 0.7299 | 0.5250 | — |

MobileNet의 파라미터 수는 약 223만 개이며, 모델 가중치 크기는 약 17.41 MB입니다.

## Default ISP

`default_isp`는 이 프로젝트의 DFX 검출 경로를 위해 설계한 정수 기반 C++ ISP 파이프라인입니다. AMD Vitis Vision L3 ISP 예제의 스테이지 순서와 처리 도메인을 따릅니다.

파이프라인 스테이지:

1. Bayer 도메인 Black Level Correction
2. Bayer 도메인 Red/Blue Gain Control
3. RGGB Bilinear Demosaicing
4. Adaptive Gray-World White Balance
5. Q8 3×3 Color-Correction Matrix
6. 12-bit에서 8-bit로의 Quantization
7. Gamma LUT
8. RGB888 출력

이 모듈은 런타임 AWB 제어가 가능한 개발용 진입점과, 다른 reconfigurable module과 동일한 포트 계약을 공유하는 고정 AWB DFX top을 제공합니다. 구현 상세, Vitis Vision과의 의도적 차이, 상수, 검증 상태는 [default_isp/README.md](default_isp/README.md)를 참고하십시오.
