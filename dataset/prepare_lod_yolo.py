#!/usr/bin/env python3
"""Prepare labeled Sony RAW-NOD images for YOLO and SSDLite training."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rawpy
from PIL import Image


CLASSES = ["person", "car", "bicycle"]
TEST_SIZE = 100
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    dataset_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=dataset_root / "LOD/Sony")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=dataset_root / "LOD/annotations/Sony",
    )
    parser.add_argument("--train-root", type=Path, default=dataset_root / "LOD_train")
    parser.add_argument("--test-root", type=Path, default=dataset_root / "LOD_test")
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--max-size",
        type=int,
        default=1280,
        help="Maximum JPEG width/height; 0 preserves the full RAW resolution.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def load_annotations(annotation_dir: Path):
    """Merge the official train/val/test COCO annotations without duplicates."""
    paths = [
        annotation_dir / "raw_new_Sony_RX100m7_train.json",
        annotation_dir / "raw_new_Sony_RX100m7_val.json",
        annotation_dir / "raw_str_labeled_new_Sony_RX100m7_test.json",
    ]
    images = {}
    boxes = defaultdict(list)
    categories = {}

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        categories.update({item["id"]: item["name"] for item in data["categories"]})
        for image in data["images"]:
            images[str(image["id"])] = image
        for annotation in data["annotations"]:
            boxes[str(annotation["image_id"])].append(annotation)

    class_to_id = {name: index for index, name in enumerate(CLASSES)}
    category_to_id = {category: class_to_id[name] for category, name in categories.items()}
    return images, boxes, category_to_id


def convert_raw(source: Path, destination: Path, max_size: int, quality: int) -> None:
    if destination.is_file():
        return
    with rawpy.imread(str(source)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
        )
    image = Image.fromarray(rgb)
    if max_size > 0:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    temporary = destination.with_suffix(".tmp.jpg")
    image.save(temporary, "JPEG", quality=quality, optimize=True)
    temporary.replace(destination)


def yolo_labels(image, annotations, category_to_id) -> list[str]:
    width = float(image["width"])
    height = float(image["height"])
    labels = []
    for annotation in annotations:
        x, y, box_width, box_height = map(float, annotation["bbox"])
        x1 = min(width, max(0.0, x))
        y1 = min(height, max(0.0, y))
        x2 = min(width, max(0.0, x + box_width))
        y2 = min(height, max(0.0, y + box_height))
        if x2 <= x1 or y2 <= y1:
            continue
        labels.append(
            f"{category_to_id[annotation['category_id']]} "
            f"{((x1 + x2) / 2) / width:.6f} {((y1 + y2) / 2) / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return labels


def main() -> None:
    args = parse_args()
    images, annotations, category_to_id = load_annotations(args.annotations)
    available = [
        image for image in images.values() if (args.source / image["file_name"]).is_file()
    ]
    missing = sorted(set(images) - {str(image["id"]) for image in available})
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} labeled RAW files; first: {missing[0]}")
    if len(available) <= args.test_size:
        raise ValueError("test-size must be smaller than the number of labeled images")

    available.sort(key=lambda item: item["file_name"])
    random.Random(args.seed).shuffle(available)
    splits = {
        args.test_root: available[: args.test_size],
        args.train_root: available[args.test_size :],
    }

    jobs = []
    for output_root, selected in splits.items():
        image_dir = output_root / "images"
        label_dir = output_root / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in selected:
            jobs.append((image, image_dir, label_dir))

    def prepare_one(job) -> None:
        image, image_dir, label_dir = job
        stem = Path(image["file_name"]).stem
        convert_raw(
            args.source / image["file_name"],
            image_dir / f"{stem}.jpg",
            args.max_size,
            args.jpeg_quality,
        )
        labels = yolo_labels(image, annotations[str(image["id"])], category_to_id)
        (label_dir / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")

    total = len(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(prepare_one, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed == 1 or completed % 100 == 0 or completed == total:
                print(f"Prepared {completed}/{total}", flush=True)

    yaml = f"""train: {args.train_root.resolve() / 'images'}
val: {args.test_root.resolve() / 'images'}
test: {args.test_root.resolve() / 'images'}

names:
  0: person
  1: car
  2: bicycle
"""
    (args.train_root / "data.yaml").write_text(yaml, encoding="utf-8")
    raw_count = sum(1 for path in args.source.glob("*.ARW") if not path.name.startswith("._"))
    print(f"LOD_train: {len(splits[args.train_root])}")
    print(f"LOD_test:  {len(splits[args.test_root])}")
    print(f"Unlabeled RAW files excluded: {raw_count - total}")
    print(f"data.yaml: {args.train_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
