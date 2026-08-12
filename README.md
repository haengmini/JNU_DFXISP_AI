# JNU DFXISP AI

An object detection project for evaluating RAW image processing pipelines using the LOD and PASCALRAW datasets. The repository contains YOLOv8n and SSDLite-MobileNetV3 model weights, experimental results, and a C++ ISP implementation for DFX integration.

## Project Scope

This project compares the performance of lightweight object detection models on two RAW image datasets.

- **LOD / RAW-NOD (Sony):** Low-light road images captured in Sony `.ARW` format
- **PASCALRAW (Nikon):** Daytime images captured in 12-bit Nikon `.NEF` format

Both datasets use the following three detection classes:

```text
0: person
1: car
2: bicycle
```

The detection models are trained on RGB JPEG images converted from RAW files. The corresponding RAW files are retained for ISP and RAW-domain evaluation.

## Repository Structure

```text
.
├── .gitignore                      # Git ignore rules
├── default_isp/
│   ├── default_isp.cpp            # Integer-based default ISP implementation
│   ├── default_isp.hpp            # ISP interface and DFX top declaration
│   └── README.md                  # Pipeline design and verification document
├── models/
│   ├── mobilenet/
│   │   └── best.pt                # Final SSDLite-MobileNetV3 weights
│   └── yolov8n/
│       └── best.pt                # Final YOLOv8n weights
└── README.md                       # Project overview and usage information
```

## Dataset Download

### LOD / RAW-NOD

LOD is a low-light road-scene dataset captured with a Sony camera. The original images are stored in `.ARW` RAW format and contain people, cars, and bicycles captured in nighttime environments.

### PASCALRAW

PASCALRAW is a daytime-scene dataset captured with a Nikon camera. The original images are stored in 12-bit `.NEF` RAW format. Like LOD, it uses person, car, and bicycle as its object detection classes.

The RAW datasets and prepared test splits are available from the following shared Google Drive folder:

- [LOD and PASCALRAW datasets](https://drive.google.com/drive/folders/1_TaV0f5hTGxb-TAWWwP04s9ykcJp_Q2G?usp=sharing)

The complete RAW datasets and dataset archives are not stored in Git due to their size.

## Experimental Results

The LOD and PASCALRAW metrics were measured on 100 test images from each dataset. The Mixed metrics were measured on a combined validation set containing both datasets. MobileNet Precision, Recall, and F1 use a confidence threshold of 0.25 and an IoU threshold of 0.50. YOLO metrics are taken from the epoch with the highest recorded mAP50-95. Values not recorded in the checkpoints are shown as `—`.

| Dataset | Detection Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Throughput |
|---|---|---:|---:|---:|---:|---:|---:|
| LOD | SSDLite-MobileNetV3 | 0.6915 | 0.4736 | 0.5622 | 0.5114 | 0.2980 | 60.4 FPS, batch 8 |
| PASCALRAW | SSDLite-MobileNetV3 | 0.9338 | 0.8598 | 0.8952 | 0.9210 | 0.6775 | 79.1 FPS, batch 8 |
| LOD | YOLOv8n | 0.8695 | 0.6381 | — | 0.7161 | 0.4923 | — |
| PASCALRAW | YOLOv8n | 0.9710 | 0.9069 | — | 0.9448 | 0.7713 | — |
| Mixed (LOD + PASCALRAW) | SSDLite-MobileNetV3 | — | — | — | 0.6155 | 0.3929 | — |
| Mixed (LOD + PASCALRAW) | YOLOv8n | 0.9232 | 0.6370 | — | 0.7299 | 0.5250 | — |

MobileNet has approximately 2.23 million parameters, and its model weights are approximately 17.41 MB.

## Default ISP

`default_isp` is an integer-based C++ ISP pipeline designed for the project's DFX detection path. It follows the stage order and processing domains of the AMD Vitis Vision L3 ISP example.

Pipeline stages:

1. Bayer-domain Black Level Correction
2. Bayer-domain Red/Blue Gain Control
3. RGGB Bilinear Demosaicing
4. Adaptive Gray-World White Balance
5. Q8 3×3 Color-Correction Matrix
6. 12-bit to 8-bit Quantization
7. Gamma LUT
8. RGB888 Output

This module provides a development entry point with runtime AWB control and a fixed-AWB DFX top that shares the same port contract as the other reconfigurable modules. See [default_isp/README.md](default_isp/README.md) for implementation details, intentional differences from Vitis Vision, constants, and verification status.
