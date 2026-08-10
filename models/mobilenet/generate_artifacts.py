#!/usr/bin/env python3
"""Generate YOLO-style plots and batch previews for a trained SSDLite model."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

from train_ssdlite import CLASSES, YoloDetectionDataset, create_model


COLORS = {1: "#ff3838", 2: "#00c853", 3: "#2979ff"}


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", type=Path, default=root / "dataset/LOD_train/images")
    parser.add_argument("--train-labels", type=Path, default=root / "dataset/LOD_train/labels")
    parser.add_argument("--val-images", type=Path, default=root / "dataset/LOD_test/images")
    parser.add_argument("--val-labels", type=Path, default=root / "dataset/LOD_test/labels")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "runs/lod/best.pt")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs/lod")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true", help="Skip train batch previews.")
    return parser.parse_args()


def plot_results(csv_path: Path, output_path: Path) -> None:
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"No training rows found in {csv_path}")
    epochs = [int(row["epoch"]) for row in rows]
    loss = [float(row["train_loss"]) for row in rows]
    map_all = [float(row["map50_95"]) for row in rows]
    map_50 = [float(row["map50"]) for row in rows]
    lr = [float(row["lr"]) for row in rows]
    seconds = [float(row["seconds"]) for row in rows]

    def smooth(values, fraction=0.08):
        values = np.asarray(values, dtype=float)
        window = max(3, int(round(len(values) * fraction)))
        if window % 2 == 0:
            window += 1
        padding = window // 2
        padded = np.pad(values, (padding, padding), mode="edge")
        return np.convolve(padded, np.ones(window) / window, mode="valid")

    def raw_and_smooth(axis, values, color, label):
        axis.plot(
            epochs, values, color=color, linewidth=1, alpha=0.38,
            marker="o", markersize=2.8, label=f"{label} results",
        )
        axis.plot(
            epochs, smooth(values), color=color, linewidth=2.5,
            linestyle="--", label=f"{label} smooth",
        )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    raw_and_smooth(axes[0, 0], loss, "#e65100", "loss")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend(fontsize=9)
    raw_and_smooth(axes[0, 1], map_all, "#1565c0", "mAP50-95")
    raw_and_smooth(axes[0, 1], map_50, "#ef6c00", "mAP50")
    axes[0, 1].set_title("Validation Metrics")
    axes[0, 1].set_ylabel("mAP")
    axes[0, 1].legend(fontsize=9, ncol=2)
    raw_and_smooth(axes[1, 0], lr, "#6a1b9a", "learning rate")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_ylabel("lr")
    axes[1, 0].legend(fontsize=9)
    raw_and_smooth(axes[1, 1], seconds, "#00838f", "epoch time")
    axes[1, 1].set_title("Epoch Time")
    axes[1, 1].set_ylabel("seconds")
    axes[1, 1].legend(fontsize=9)
    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    best_index = max(range(len(map_all)), key=map_all.__getitem__)
    fig.suptitle(
        f"SSDLite MobileNetV3 — {len(rows)} epochs | "
        f"best mAP50-95 {map_all[best_index]:.4f} @ epoch {epochs[best_index]} | "
        f"mAP50 {map_50[best_index]:.4f}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def draw_boxes(image: Image.Image, boxes, labels, scores=None) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    for index, (box, label) in enumerate(zip(boxes, labels)):
        label = int(label)
        color = COLORS[label]
        x1, y1, x2, y2 = map(float, box)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, image.width // 500))
        caption = CLASSES[label]
        if scores is not None:
            caption += f" {float(scores[index]):.2f}"
        text_box = draw.textbbox((x1, y1), caption, font=font, stroke_width=1)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), caption, fill="white", font=font, stroke_width=1)
    return image


def save_mosaic(images, title: str, path: Path, columns: int = 4) -> None:
    cell_width, cell_height = 480, 320
    rows = (len(images) + columns - 1) // columns
    header = 42
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height + header), "#202124")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="white", font=ImageFont.load_default(size=20))
    for index, image in enumerate(images):
        preview = image.copy()
        preview.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width + (cell_width - preview.width) // 2
        y = header + (index // columns) * cell_height + (cell_height - preview.height) // 2
        canvas.paste(preview, (x, y))
    canvas.save(path, quality=94)


def labeled_image(dataset, index: int) -> Image.Image:
    path = dataset.images[index]
    with Image.open(path) as source:
        image = source.convert("RGB")
    target = dataset._read_target(index, *image.size)
    return draw_boxes(image, target["boxes"], target["labels"])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_results(args.output / "results.csv", args.output / "results.png")

    val = YoloDetectionDataset(args.val_images, args.val_labels)
    rng = random.Random(args.seed)
    val_indices = list(range(min(len(val), args.batch_size * args.num_batches)))

    if not args.skip_train:
        train = YoloDetectionDataset(args.train_images, args.train_labels)
        train_indices = rng.sample(range(len(train)), args.batch_size * args.num_batches)
        for batch in range(args.num_batches):
            selected = train_indices[batch * args.batch_size : (batch + 1) * args.batch_size]
            save_mosaic(
                [labeled_image(train, index) for index in selected],
                f"Train samples with labels — batch {batch}",
                args.output / f"train_batch{batch}.jpg",
            )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = create_model(None, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for batch in range(args.num_batches):
        selected = val_indices[batch * args.batch_size : (batch + 1) * args.batch_size]
        originals = []
        tensors = []
        labels = []
        for index in selected:
            path = val.images[index]
            with Image.open(path) as source:
                image = source.convert("RGB")
            target = val._read_target(index, *image.size)
            originals.append(image)
            tensors.append(F.to_tensor(image).to(device))
            labels.append(draw_boxes(image.copy(), target["boxes"], target["labels"]))
        with torch.inference_mode():
            predictions = model(tensors)
        predicted = []
        for image, prediction in zip(originals, predictions):
            keep = prediction["scores"].cpu() >= args.score_threshold
            predicted.append(
                draw_boxes(
                    image.copy(),
                    prediction["boxes"].cpu()[keep],
                    prediction["labels"].cpu()[keep],
                    prediction["scores"].cpu()[keep],
                )
            )
        save_mosaic(labels, f"Validation ground truth — batch {batch}", args.output / f"val_batch{batch}_labels.jpg")
        save_mosaic(predicted, f"Validation predictions — batch {batch}", args.output / f"val_batch{batch}_pred.jpg")

    print(f"Artifacts written to {args.output}")


if __name__ == "__main__":
    main()
