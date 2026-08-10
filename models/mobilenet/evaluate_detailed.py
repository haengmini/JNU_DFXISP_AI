#!/usr/bin/env python3
"""Create detailed detection metrics and YOLO-style plots from an SSDLite checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as F

from train_ssdlite import CLASSES, YoloDetectionDataset, create_model, make_coco_ground_truth


COLORS = ["#ef5350", "#43a047", "#1e88e5"]


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=root / "dataset/LOD_test/images")
    parser.add_argument("--labels", type=Path, default=root / "dataset/LOD_test/labels")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "runs/lod/best.pt")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs/lod")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    return parser.parse_args()


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.empty(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def average_precision(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(points, recall, precision), points))


def class_curve(predictions, ground_truth, class_id, iou_threshold):
    gt = {}
    total_gt = 0
    for image_id, target in ground_truth.items():
        keep = target["labels"] == class_id
        boxes = target["boxes"][keep]
        gt[image_id] = {"boxes": boxes, "matched": np.zeros(len(boxes), dtype=bool)}
        total_gt += len(boxes)
    detections = []
    for image_id, prediction in predictions.items():
        keep = prediction["labels"] == class_id
        for box, score in zip(prediction["boxes"][keep], prediction["scores"][keep]):
            detections.append((float(score), image_id, box))
    detections.sort(reverse=True, key=lambda item: item[0])
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    scores = np.array([item[0] for item in detections])
    for index, (_, image_id, box) in enumerate(detections):
        entry = gt[image_id]
        ious = box_iou(box, entry["boxes"])
        if len(ious):
            match = int(np.argmax(ious))
            if ious[match] >= iou_threshold and not entry["matched"][match]:
                tp[index] = 1
                entry["matched"][match] = True
                continue
        fp[index] = 1
    tp_sum = np.cumsum(tp)
    fp_sum = np.cumsum(fp)
    recall = tp_sum / max(total_gt, 1)
    precision = tp_sum / np.maximum(tp_sum + fp_sum, 1)
    return scores, precision, recall, average_precision(recall, precision), total_gt


def metrics_at_threshold(curves, threshold):
    totals = np.zeros(3)  # tp, fp, gt
    class_rows = []
    for class_id, (scores, precision, recall, ap, total_gt) in curves.items():
        count = int(np.sum(scores >= threshold))
        if count:
            tp = float(recall[count - 1] * total_gt)
            fp = count - tp
        else:
            tp = fp = 0.0
        p = tp / max(tp + fp, 1)
        r = tp / max(total_gt, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        class_rows.append((class_id, p, r, f1, ap, total_gt, int(tp), int(fp), int(total_gt - tp)))
        totals += (tp, fp, total_gt)
    p = totals[0] / max(totals[0] + totals[1], 1)
    r = totals[0] / max(totals[2], 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1, class_rows


def confusion_matrix(predictions, ground_truth, threshold, iou_threshold):
    matrix = np.zeros((4, 4), dtype=int)  # rows=true, columns=pred; last=background
    for image_id, target in ground_truth.items():
        gt_boxes, gt_labels = target["boxes"], target["labels"]
        pred = predictions[image_id]
        keep = pred["scores"] >= threshold
        boxes, labels, scores = pred["boxes"][keep], pred["labels"][keep], pred["scores"][keep]
        order = np.argsort(-scores)
        matched = np.zeros(len(gt_boxes), dtype=bool)
        for index in order:
            ious = box_iou(boxes[index], gt_boxes)
            ious[matched] = -1
            match = int(np.argmax(ious)) if len(ious) else -1
            if match >= 0 and ious[match] >= iou_threshold:
                matrix[gt_labels[match] - 1, labels[index] - 1] += 1
                matched[match] = True
            else:
                matrix[3, labels[index] - 1] += 1
        for label in gt_labels[~matched]:
            matrix[label - 1, 3] += 1
    return matrix


def save_confusion(matrix, output):
    names = CLASSES[1:] + ["background"]
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(4):
        for column in range(4):
            axis.text(column, row, f"{normalized[row,column]:.2f}\n({matrix[row,column]})", ha="center", va="center", color="white" if normalized[row,column] > .5 else "black")
    axis.set_xticks(range(4), names, rotation=25)
    axis.set_yticks(range(4), names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix (normalized by true class)")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_curves(curves, output, kind):
    fig, axis = plt.subplots(figsize=(9, 7))
    if kind == "PR":
        for class_id, (_, precision, recall, ap, _) in curves.items():
            axis.plot(recall, precision, color=COLORS[class_id - 1], label=f"{CLASSES[class_id]} AP={ap:.3f}")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Precision-Recall Curve @ IoU 0.50")
    else:
        thresholds = np.linspace(0, 1, 201)
        for class_id, (scores, precision, recall, _, _) in curves.items():
            values = []
            for threshold in thresholds:
                count = int(np.sum(scores >= threshold))
                p = precision[count - 1] if count else 1.0
                r = recall[count - 1] if count else 0.0
                values.append({"P": p, "R": r, "F1": 2*p*r/max(p+r, 1e-9)}[kind])
            axis.plot(thresholds, values, color=COLORS[class_id - 1], label=CLASSES[class_id])
        axis.set_xlabel("Confidence")
        axis.set_ylabel({"P":"Precision", "R":"Recall", "F1":"F1"}[kind])
        axis.set_title(f"{axis.get_ylabel()}-Confidence Curve")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = YoloDetectionDataset(args.images, args.labels)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = create_model(None, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions, ground_truth, coco_detections = {}, {}, []
    inference_seconds = 0.0
    for start in range(0, len(dataset), args.batch_size):
        indices = list(range(start, min(start + args.batch_size, len(dataset))))
        tensors = []
        for index in indices:
            with Image.open(dataset.images[index]) as source:
                image = source.convert("RGB")
            target = dataset._read_target(index, *image.size)
            ground_truth[index] = {"boxes": target["boxes"].numpy(), "labels": target["labels"].numpy()}
            tensors.append(F.to_tensor(image).to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(tensors)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        for image_id, output in zip(indices, outputs):
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            predictions[image_id] = {"boxes": boxes, "labels": labels, "scores": scores}
            for box, label, score in zip(boxes, labels, scores):
                if score < .001:
                    continue
                x1, y1, x2, y2 = box
                coco_detections.append({"image_id": image_id, "category_id": int(label), "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)], "score": float(score)})

    coco_gt = make_coco_ground_truth(dataset)
    coco_dt = coco_gt.loadRes(coco_detections)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    coco_stats = evaluator.stats.copy()
    per_class_coco = {}
    for class_id in range(1, 4):
        item = COCOeval(coco_gt, coco_dt, "bbox")
        item.params.catIds = [class_id]
        item.evaluate(); item.accumulate(); item.summarize()
        per_class_coco[class_id] = (float(item.stats[0]), float(item.stats[1]))

    curves = {class_id: class_curve(predictions, ground_truth, class_id, args.iou_threshold) for class_id in range(1, 4)}
    precision, recall, f1, class_rows = metrics_at_threshold(curves, args.conf_threshold)
    matrix = confusion_matrix(predictions, ground_truth, args.conf_threshold, args.iou_threshold)
    save_confusion(matrix, args.output / "confusion_matrix.png")
    save_curves(curves, args.output / "PR_curve.png", "PR")
    save_curves(curves, args.output / "P_curve.png", "P")
    save_curves(curves, args.output / "R_curve.png", "R")
    save_curves(curves, args.output / "F1_curve.png", "F1")

    latency_ms = inference_seconds / len(dataset) * 1000
    fps = len(dataset) / inference_seconds
    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "checkpoint": str(args.checkpoint), "images": len(dataset), "confidence": args.conf_threshold,
        "iou": args.iou_threshold, "precision": precision, "recall": recall, "f1": f1,
        "map50_95": float(coco_stats[0]), "map50": float(coco_stats[1]), "map75": float(coco_stats[2]),
        "map_small": float(coco_stats[3]), "map_medium": float(coco_stats[4]), "map_large": float(coco_stats[5]),
        "ar100": float(coco_stats[8]), "ar_small": float(coco_stats[9]), "ar_medium": float(coco_stats[10]), "ar_large": float(coco_stats[11]),
        "latency_ms_per_image_batch": latency_ms, "fps_batch": fps, "parameters": parameters,
        "checkpoint_mb": args.checkpoint.stat().st_size / 1024**2,
    }
    (args.output / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "class_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1", "ap50_custom", "coco_map50_95", "coco_map50", "ground_truth", "tp", "fp", "fn"])
        for class_id, p, r, class_f1, ap, gt, tp, fp, fn in class_rows:
            writer.writerow([CLASSES[class_id], p, r, class_f1, ap, *per_class_coco[class_id], gt, tp, fp, fn])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = CLASSES[1:]
    x = np.arange(3); width = .25
    axes[0,0].bar(x-width, [r[1] for r in class_rows], width, label="Precision")
    axes[0,0].bar(x, [r[2] for r in class_rows], width, label="Recall")
    axes[0,0].bar(x+width, [r[3] for r in class_rows], width, label="F1")
    axes[0,0].set_xticks(x, names); axes[0,0].set_ylim(0,1); axes[0,0].legend(); axes[0,0].set_title(f"Per-class @ conf {args.conf_threshold}, IoU {args.iou_threshold}")
    axes[0,1].bar(x-width/2, [per_class_coco[i][0] for i in range(1,4)], width, label="mAP50-95")
    axes[0,1].bar(x+width/2, [per_class_coco[i][1] for i in range(1,4)], width, label="mAP50")
    axes[0,1].set_xticks(x, names); axes[0,1].set_ylim(0,1); axes[0,1].legend(); axes[0,1].set_title("COCO AP by class")
    size_names=["small","medium","large"]
    axes[1,0].bar(x-width/2, coco_stats[3:6], width, label="AP")
    axes[1,0].bar(x+width/2, coco_stats[9:12], width, label="AR")
    axes[1,0].set_xticks(x,size_names); axes[1,0].set_ylim(0,1); axes[1,0].legend(); axes[1,0].set_title("Metrics by object size")
    axes[1,1].axis("off")
    text = (f"Overall metrics\n\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\n"
            f"mAP50-95: {coco_stats[0]:.4f}\nmAP50: {coco_stats[1]:.4f}\nmAP75: {coco_stats[2]:.4f}\n\n"
            f"Inference: {latency_ms:.2f} ms/image\nThroughput: {fps:.1f} FPS (batch={args.batch_size})\n"
            f"Parameters: {parameters/1e6:.2f} M\nCheckpoint: {summary['checkpoint_mb']:.2f} MB")
    axes[1,1].text(.05,.95,text,va="top",fontsize=14,linespacing=1.45)
    for axis in axes.flat[:3]: axis.grid(axis="y",alpha=.25)
    fig.suptitle("SSDLite MobileNetV3 Detailed Evaluation",fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(args.output / "detailed_results.png",dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
