#!/usr/bin/env python3
"""
Bounding-box annotation tool for YOLO datasets.

Controls:
  1-9        — select class by number
  LMB drag   — draw bounding box
  Z / Ctrl+Z — undo last box on current image
  S          — save & next image
  A / Left   — previous image
  D / Right  — next image (without saving)
  Q / Esc    — quit

Usage:
  python3 annotate.py --classes duck robot obstacle \
                      --input  datasets/raw_data \
                      --output datasets/annotated
"""

import argparse
import copy
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
WINDOW = "Annotator"
DEFAULT_CLASSES = ["duckie", "red_line", "white_line", "yellow_line", "obstacle"]
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (128, 0, 255),
]


def color_for(class_id: int) -> tuple:
    return COLORS[class_id % len(COLORS)]


class Annotator:
    def __init__(self, classes: list[str], input_dir: str, output_dir: str):
        self.classes = classes
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths = sorted(
            p
            for p in self.input_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            sys.exit(f"No images found in {self.input_dir}")

        self.idx = 0
        self.current_class = 0
        self.boxes: list[tuple[int, int, int, int, int]] = []  # (cls, x1, y1, x2, y2)
        self.drawing = False
        self.ix = 0
        self.iy = 0
        self.temp_box: tuple | None = None

        self._skip_to_first_unannotated()

    def _skip_to_first_unannotated(self):
        for i, p in enumerate(self.image_paths):
            label_path = self.labels_dir / (p.stem + ".txt")
            if not label_path.exists():
                self.idx = i
                return

    def _label_path(self, img_path: Path) -> Path:
        return self.labels_dir / (img_path.stem + ".txt")

    def _load_existing(self, img_path: Path, img_w: int, img_h: int):
        """Load previously saved YOLO labels back into pixel boxes."""
        self.boxes.clear()
        lp = self._label_path(img_path)
        if not lp.exists():
            return
        for line in lp.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            self.boxes.append((cls, x1, y1, x2, y2))

    def _save(self, img_path: Path, img: np.ndarray):
        h, w = img.shape[:2]
        dst_img = self.images_dir / img_path.name
        if not dst_img.exists():
            cv2.imwrite(str(dst_img), img)

        lines = []
        for cls, x1, y1, x2, y2 in self.boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        self._label_path(img_path).write_text("\n".join(lines) + "\n" if lines else "")

    def _save_dataset_yaml(self):
        yaml_path = self.output_dir / "dataset.yaml"
        lines = [
            f"path: {self.output_dir.resolve()}",
            "train: images",
            "val: images",
            "",
            f"nc: {len(self.classes)}",
            f"names: {self.classes}",
            "",
        ]
        yaml_path.write_text("\n".join(lines))

    def _draw(self, img: np.ndarray) -> np.ndarray:
        vis = img.copy()
        h, w = vis.shape[:2]

        for cls, x1, y1, x2, y2 in self.boxes:
            c = color_for(cls)
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            label = self.classes[cls] if cls < len(self.classes) else str(cls)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        if self.temp_box:
            tx1, ty1, tx2, ty2 = self.temp_box
            c = color_for(self.current_class)
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), c, 1)

        bar_h = 36
        cv2.rectangle(vis, (0, h - bar_h), (w, h), (40, 40, 40), -1)

        status = f"[{self.idx + 1}/{len(self.image_paths)}]  "
        for i, name in enumerate(self.classes):
            marker = ">> " if i == self.current_class else "   "
            status += f"{marker}{i + 1}:{name}  "
        status += f" | boxes: {len(self.boxes)}"

        cv2.putText(vis, status, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        return vis

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.temp_box = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp_box = (self.ix, self.iy, x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.temp_box = None
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                self.boxes.append((self.current_class, x1, y1, x2, y2))

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._mouse_cb)

        while True:
            img_path = self.image_paths[self.idx]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Cannot read {img_path}, skipping")
                self.idx = min(self.idx + 1, len(self.image_paths) - 1)
                continue

            h_img, w_img = img.shape[:2]
            self._load_existing(img_path, w_img, h_img)

            cv2.setWindowTitle(WINDOW, f"Annotator — {img_path.name}")

            while True:
                vis = self._draw(img)
                cv2.imshow(WINDOW, vis)
                key = cv2.waitKey(30) & 0xFF

                if key == 27 or key == ord("q"):
                    self._save_dataset_yaml()
                    cv2.destroyAllWindows()
                    print("Done. Dataset saved to", self.output_dir)
                    return

                if key == 26 or key == ord("z"):  # Ctrl+Z or Z
                    if self.boxes:
                        removed = self.boxes.pop()
                        cls_name = self.classes[removed[0]] if removed[0] < len(self.classes) else str(removed[0])
                        print(f"  undo: removed {cls_name} box")

                for i in range(min(9, len(self.classes))):
                    if key == ord(str(i + 1)):
                        self.current_class = i
                        print(f"  class → {self.classes[i]}")

                if key == ord("s"):
                    self._save(img_path, img)
                    print(f"  saved {img_path.name}: {len(self.boxes)} boxes")
                    self.idx = min(self.idx + 1, len(self.image_paths) - 1)
                    break

                if key == ord("d") or key == 83:  # D or Right arrow
                    self.idx = min(self.idx + 1, len(self.image_paths) - 1)
                    break

                if key == ord("a") or key == 81:  # A or Left arrow
                    self.idx = max(self.idx - 1, 0)
                    break

                try:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        self._save_dataset_yaml()
                        cv2.destroyAllWindows()
                        return
                except cv2.error:
                    self._save_dataset_yaml()
                    return

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO bounding-box annotation tool")
    parser.add_argument(
        "--classes", "-c",
        nargs="+",
        default=DEFAULT_CLASSES,
        help=f"List of class names (default: {DEFAULT_CLASSES})",
    )
    parser.add_argument(
        "--input", "-i",
        default="datasets/raw_data",
        help="Directory with source images (default: datasets/raw_data)",
    )
    parser.add_argument(
        "--output", "-o",
        default="datasets/annotated",
        help="Output directory for the YOLO dataset (default: datasets/annotated)",
    )
    args = parser.parse_args()

    out = Path(args.output).resolve()
    print(f"Classes : {args.classes}")
    print(f"Input   : {Path(args.input).resolve()}")
    print(f"Output  : {out}")
    print(f"  images → {out / 'images'}")
    print(f"  labels → {out / 'labels'}")
    print(f"  config → {out / 'dataset.yaml'}")
    print()
    print("Controls:")
    print("  1-9        select class")
    print("  LMB drag   draw box")
    print("  Z/Ctrl+Z   undo last box")
    print("  S          save & next")
    print("  A/D        prev/next image")
    print("  Q/Esc      quit")
    print()

    annotator = Annotator(args.classes, args.input, args.output)
    annotator.run()


if __name__ == "__main__":
    main()
