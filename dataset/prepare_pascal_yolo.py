from pathlib import Path
import os
import random
import shutil
import xml.etree.ElementTree as ET


SOURCE = Path(__file__).resolve().parent / "PASCALRAW"
TRAIN_ROOT = Path(__file__).resolve().parent / "PASCAL_train"
TEST_ROOT = Path(__file__).resolve().parent / "PASCAL_test"
CLASSES = ["person", "car", "bicycle"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
TEST_SIZE = 100
RANDOM_SEED = 42


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def convert_annotation(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))
    labels = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name", "").strip()
        if class_name not in CLASS_TO_ID:
            continue

        box = obj.find("bndbox")
        xmin = max(0.0, float(box.findtext("xmin")))
        ymin = max(0.0, float(box.findtext("ymin")))
        xmax = min(width, float(box.findtext("xmax")))
        ymax = min(height, float(box.findtext("ymax")))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        labels.append(
            f"{CLASS_TO_ID[class_name]} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    return labels


def main() -> None:
    images = sorted(
        path
        for path in (SOURCE / "jpg").glob("*.jpg")
        if not path.name.startswith("._")
    )
    random.Random(RANDOM_SEED).shuffle(images)

    splits = {
        TEST_ROOT: images[:TEST_SIZE],
        TRAIN_ROOT: images[TEST_SIZE:],
    }

    for dataset_root, selected_images in splits.items():
        image_dir = dataset_root / "images"
        label_dir = dataset_root / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in selected_images:
            xml_path = SOURCE / "annotations" / f"{image_path.stem}.xml"
            labels = convert_annotation(xml_path)
            link_or_copy(image_path, image_dir / image_path.name)
            (label_dir / f"{image_path.stem}.txt").write_text(
                "\n".join(labels), encoding="utf-8"
            )

    yaml = f"""train: {TRAIN_ROOT / 'images'}
val: {TEST_ROOT / 'images'}
test: {TEST_ROOT / 'images'}

names:
  0: person
  1: car
  2: bicycle
"""
    (TRAIN_ROOT / "data.yaml").write_text(yaml, encoding="utf-8")

    print(f"PASCAL_train: {len(splits[TRAIN_ROOT])}")
    print(f"PASCAL_test:  {len(splits[TEST_ROOT])}")
    print(f"data.yaml:    {TRAIN_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
