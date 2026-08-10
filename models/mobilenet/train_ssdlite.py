#!/usr/bin/env python3
"""Fine-tune Torchvision SSDLite-MobileNetV3 on YOLO-format labels."""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F


CLASSES = ["__background__", "person", "car", "bicycle"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YoloDetectionDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, augment: bool = False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.augment = augment
        self.images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and not path.name.startswith("._")
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise RuntimeError(f"No images found in {image_dir}")

        missing = [
            path.name
            for path in self.images
            if not (label_dir / f"{path.stem}.txt").is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} label files; first missing image: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def _read_target(self, index: int, width: int, height: int) -> dict:
        image_path = self.images[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        boxes = []
        labels = []

        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid label at {label_path}:{line_number}")

            yolo_class = int(parts[0])
            if yolo_class not in (0, 1, 2):
                raise ValueError(
                    f"Unsupported class {yolo_class} at {label_path}:{line_number}"
                )

            x_center, y_center, box_width, box_height = map(float, parts[1:])
            xmin = max(0.0, (x_center - box_width / 2) * width)
            ymin = max(0.0, (y_center - box_height / 2) * height)
            xmax = min(float(width), (x_center + box_width / 2) * width)
            ymax = min(float(height), (y_center + box_height / 2) * height)
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(yolo_class + 1)  # SSD reserves class 0 for background.

        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (
            (box_tensor[:, 2] - box_tensor[:, 0])
            * (box_tensor[:, 3] - box_tensor[:, 1])
        )
        return {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(box_tensor), dtype=torch.int64),
        }

    def __getitem__(self, index: int):
        with Image.open(self.images[index]) as source:
            image = source.convert("RGB")
        width, height = image.size
        target = self._read_target(index, width, height)

        if self.augment and random.random() < 0.5:
            image = F.hflip(image)
            boxes = target["boxes"]
            if len(boxes):
                old_xmin = boxes[:, 0].clone()
                old_xmax = boxes[:, 2].clone()
                boxes[:, 0] = width - old_xmax
                boxes[:, 2] = width - old_xmin

        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(weights_path: Path | None, device: torch.device):
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=len(CLASSES),
    )

    if weights_path is not None:
        pretrained = torch.load(weights_path, map_location="cpu", weights_only=True)
        current = model.state_dict()
        compatible = {
            name: value
            for name, value in pretrained.items()
            if name in current and current[name].shape == value.shape
        }
        result = model.load_state_dict(compatible, strict=False)
        print(
            f"Loaded {len(compatible)}/{len(current)} compatible tensors from "
            f"{weights_path}"
        )
        print(f"New/custom tensors to train: {len(result.missing_keys)}")

    return model.to(device)


def make_coco_ground_truth(dataset: YoloDetectionDataset) -> COCO:
    coco = COCO()
    images = []
    annotations = []
    annotation_id = 1

    for image_id, image_path in enumerate(dataset.images):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        target = dataset._read_target(image_id, width, height)
        for box, label, area in zip(target["boxes"], target["labels"], target["area"]):
            xmin, ymin, xmax, ymax = box.tolist()
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": float(area),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id, "name": CLASSES[class_id]}
            for class_id in range(1, len(CLASSES))
        ],
        "info": {},
        "licenses": [],
    }
    coco.createIndex()
    return coco


@torch.inference_mode()
def evaluate(model, loader, device, coco_gt: COCO, score_threshold: float):
    model.eval()
    detections = []

    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        predictions = model(images)

        for prediction, target in zip(predictions, targets):
            image_id = int(target["image_id"])
            boxes = prediction["boxes"].cpu()
            labels = prediction["labels"].cpu()
            scores = prediction["scores"].cpu()
            keep = scores >= score_threshold

            for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
                xmin, ymin, xmax, ymax = box.tolist()
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                        "score": float(score),
                    }
                )

    if not detections:
        print("No validation detections above the score threshold.")
        return 0.0, 0.0

    coco_predictions = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_predictions, "bbox")
    evaluator.params.imgIds = list(range(len(loader.dataset)))
    evaluator.params.catIds = list(range(1, len(CLASSES)))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[0]), float(evaluator.stats[1])


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, metrics, args):
    checkpoint = {
        "architecture": "ssdlite320_mobilenet_v3_large",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "classes": CLASSES,
        "input_size": 320,
        "args": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", type=Path, default=root / "dataset/PASCAL_train/images")
    parser.add_argument("--train-labels", type=Path, default=root / "dataset/PASCAL_train/labels")
    parser.add_argument("--val-images", type=Path, default=root / "dataset/PASCAL_test/images")
    parser.add_argument("--val-labels", type=Path, default=root / "dataset/PASCAL_test/labels")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).parent / "ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Do not load COCO pretrained weights; initialize the entire detector randomly.",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "runs/pascal")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    train_dataset = YoloDetectionDataset(args.train_images, args.train_labels, augment=True)
    val_dataset = YoloDetectionDataset(args.val_images, args.val_labels, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )

    initial_weights = None if args.resume or args.from_scratch else args.weights
    model = create_model(initial_weights, device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    milestones = sorted({max(1, int(args.epochs * 0.6)), max(1, int(args.epochs * 0.8))})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_map = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint.get("metrics", {}).get("best_map", -1.0))
        print(f"Resuming {args.resume} from epoch {start_epoch + 1}")

    coco_gt = make_coco_ground_truth(val_dataset)
    results_path = args.output / "results.csv"
    if start_epoch == 0:
        with results_path.open("w", newline="") as file:
            csv.writer(file).writerow(["epoch", "train_loss", "map50_95", "map50", "lr", "seconds"])

    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Train images: {len(train_dataset)}; validation images: {len(val_dataset)}")
    print(f"Output: {args.output}")

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        running_loss = 0.0

        for step, (images, targets) in enumerate(train_loader, 1):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with amp_context:
                losses = model(images, targets)
                total_loss = sum(losses.values())

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss: {losses}")
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(total_loss.detach())

            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f}",
                    flush=True,
                )

        train_loss = running_loss / len(train_loader)
        map_50_95, map_50 = evaluate(
            model, val_loader, device, coco_gt, args.score_threshold
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - started

        metrics = {
            "train_loss": train_loss,
            "map50_95": map_50_95,
            "map50": map_50,
            "best_map": max(best_map, map_50_95),
        }
        save_checkpoint(
            args.output / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            metrics,
            args,
        )
        if map_50_95 > best_map:
            best_map = map_50_95
            metrics["best_map"] = best_map
            save_checkpoint(
                args.output / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                metrics,
                args,
            )

        with results_path.open("a", newline="") as file:
            csv.writer(file).writerow(
                [epoch + 1, train_loss, map_50_95, map_50, current_lr, elapsed]
            )
        print(
            f"Epoch {epoch + 1} complete: loss={train_loss:.4f}, "
            f"mAP50-95={map_50_95:.4f}, mAP50={map_50:.4f}, "
            f"time={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
