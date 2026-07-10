#!/usr/bin/env python3
"""
Auto-prelabel images with YOLO-seg predictions in YOLO polygon format.

Typical workflow:
1) Save selected frames to datasets/agrobot_seg_v2/images
2) Run this script to create draft labels in datasets/agrobot_seg_v2/labels
3) Open annotate.py and fix polygons manually
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    imgs = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not imgs:
        raise SystemExit(f"No images found in: {images_dir}")
    return imgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto prelabel segmentation masks using YOLO model")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("datasets/agrobot_seg_v2/images"),
        help="Directory with input images (default: datasets/agrobot_seg_v2/images)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("datasets/agrobot_seg_v2/labels"),
        help="Directory to write YOLO-seg labels (default: datasets/agrobot_seg_v2/labels)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolo26m-seg.pt"),
        help="YOLO-seg weights path (default: yolo26m-seg.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files (default: skip existing labels)",
    )
    parser.add_argument(
        "--empty-labels",
        action="store_true",
        help="Write empty .txt file when object is not detected",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Output class id for all polygons (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.conf <= 0 or args.conf > 1:
        raise SystemExit("--conf should be in range (0, 1]")
    if args.imgsz < 32:
        raise SystemExit("--imgsz should be >= 32")
    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from exc

    images_dir = args.images.resolve()
    labels_dir = args.labels.resolve()
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(images_dir)

    model = YOLO(str(args.model.resolve()))

    created = 0
    skipped = 0
    with_masks = 0
    no_detections = 0

    for i, image_path in enumerate(images, start=1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{i}/{len(images)}] unreadable image, skip: {image_path.name}")
            continue

        result = model.predict(image, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        lines: list[str] = []
        if result.masks is not None and result.masks.xyn:
            for poly in result.masks.xyn:
                if len(poly) < 3:
                    continue
                coords: list[str] = []
                for x, y in poly:
                    xn = min(1.0, max(0.0, float(x)))
                    yn = min(1.0, max(0.0, float(y)))
                    coords.append(f"{xn:.6f}")
                    coords.append(f"{yn:.6f}")
                lines.append(f"{args.class_id} " + " ".join(coords))

        if lines:
            label_path.write_text("\n".join(lines) + "\n")
            created += 1
            with_masks += 1
            print(f"[{i}/{len(images)}] masks={len(lines)} -> {label_path.name}")
        else:
            no_detections += 1
            if args.empty_labels:
                label_path.write_text("")
                created += 1
                print(f"[{i}/{len(images)}] no detections -> empty label")
            else:
                if label_path.exists():
                    label_path.unlink()
                print(f"[{i}/{len(images)}] no detections -> no label")

    print()
    print("Done.")
    print(f"Images total        : {len(images)}")
    print(f"Created label files : {created}")
    print(f"With masks          : {with_masks}")
    print(f"No detections       : {no_detections}")
    print(f"Skipped existing    : {skipped}")
    print(f"Labels dir          : {labels_dir}")


if __name__ == "__main__":
    main()
