#!/usr/bin/env python3
"""
Polygon segmentation annotator for YOLO-seg.

Single class workflow for rail masks:
  - class name: rail (configurable)
  - each image can contain multiple polygons of the same class

Controls:
  LMB           add polygon point
  C / Enter     close current polygon (min 3 points)
  X             clear current in-progress polygon
  Z / Ctrl+Z    undo last finished polygon
  S             save & next image
  A / Left      previous image
  D / Right     next image (without saving)
  Q / Esc       quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
WINDOW = "Seg Annotator"
MASK_COLOR = (40, 220, 80)
POINT_COLOR = (0, 240, 255)
EDGE_COLOR = (30, 180, 255)


class SegAnnotator:
    def __init__(self, class_name: str, input_dir: str, output_dir: str) -> None:
        self.class_name = class_name
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths = sorted(
            p for p in self.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            sys.exit(f"No images found in {self.input_dir}")

        self.idx = 0
        self.polygons: list[list[tuple[int, int]]] = []
        self.current_poly: list[tuple[int, int]] = []
        self._skip_to_first_unannotated()

    def _skip_to_first_unannotated(self) -> None:
        for i, p in enumerate(self.image_paths):
            if not self._label_path(p).exists():
                self.idx = i
                return

    def _label_path(self, img_path: Path) -> Path:
        return self.labels_dir / f"{img_path.stem}.txt"

    def _load_existing(self, img_path: Path, img_w: int, img_h: int) -> None:
        self.polygons.clear()
        self.current_poly.clear()
        label_path = self._label_path(img_path)
        if not label_path.exists():
            return

        text = label_path.read_text().strip()
        if not text:
            return

        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = parts[1:]
            if len(coords) % 2 != 0:
                continue

            poly: list[tuple[int, int]] = []
            for i in range(0, len(coords), 2):
                x = int(float(coords[i]) * img_w)
                y = int(float(coords[i + 1]) * img_h)
                poly.append((x, y))
            if len(poly) >= 3:
                self.polygons.append(poly)

    def _save(self, img_path: Path, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        dst_img = self.images_dir / img_path.name
        if not dst_img.exists():
            cv2.imwrite(str(dst_img), img)

        lines: list[str] = []
        for poly in self.polygons:
            if len(poly) < 3:
                continue
            pairs: list[str] = []
            for x, y in poly:
                xn = min(1.0, max(0.0, x / w))
                yn = min(1.0, max(0.0, y / h))
                pairs.append(f"{xn:.6f}")
                pairs.append(f"{yn:.6f}")
            lines.append("0 " + " ".join(pairs))

        self._label_path(img_path).write_text("\n".join(lines) + ("\n" if lines else ""))

    def _save_dataset_yaml(self) -> None:
        yaml_path = self.output_dir / "dataset.yaml"
        lines = [
            f"path: {self.output_dir.resolve()}",
            "train: images",
            "val: images",
            "",
            "nc: 1",
            f"names: ['{self.class_name}']",
            "",
        ]
        yaml_path.write_text("\n".join(lines))

    @staticmethod
    def _poly_np(poly: list[tuple[int, int]]) -> np.ndarray:
        return np.array(poly, dtype=np.int32).reshape(-1, 1, 2)

    def _draw(self, img: np.ndarray) -> np.ndarray:
        vis = img.copy()
        h, w = vis.shape[:2]

        if self.polygons:
            overlay = vis.copy()
            for poly in self.polygons:
                cv2.fillPoly(overlay, [self._poly_np(poly)], MASK_COLOR)
            vis = cv2.addWeighted(overlay, 0.30, vis, 0.70, 0)

        for i, poly in enumerate(self.polygons, start=1):
            cv2.polylines(vis, [self._poly_np(poly)], isClosed=True, color=EDGE_COLOR, thickness=2)
            anchor = poly[0]
            cv2.putText(
                vis,
                f"{self.class_name}#{i}",
                (anchor[0] + 4, anchor[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        if self.current_poly:
            for pt in self.current_poly:
                cv2.circle(vis, pt, 4, POINT_COLOR, -1)
            for i in range(1, len(self.current_poly)):
                cv2.line(vis, self.current_poly[i - 1], self.current_poly[i], POINT_COLOR, 2)

        bar_h = 74
        cv2.rectangle(vis, (0, h - bar_h), (w, h), (30, 30, 30), -1)
        line1 = (
            f"[{self.idx + 1}/{len(self.image_paths)}] class={self.class_name} "
            f"| masks={len(self.polygons)} | current_points={len(self.current_poly)}"
        )
        line2 = "LMB add point | C/Enter close | X clear current | Z undo | S save | A/D prev/next | Q exit"
        cv2.putText(vis, line1, (8, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        cv2.putText(vis, line2, (8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        return vis

    def _mouse_cb(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_poly.append((x, y))

    def _close_current_polygon(self) -> bool:
        if len(self.current_poly) < 3:
            print("  need at least 3 points to close polygon")
            return False
        self.polygons.append(self.current_poly.copy())
        self.current_poly.clear()
        return True

    def run(self) -> None:
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
            cv2.setWindowTitle(WINDOW, f"Seg Annotator — {img_path.name}")

            while True:
                cv2.imshow(WINDOW, self._draw(img))
                key = cv2.waitKey(30) & 0xFF

                if key in (27, ord("q")):
                    self._save_dataset_yaml()
                    cv2.destroyAllWindows()
                    print("Done. Segmentation dataset saved to", self.output_dir)
                    return

                if key in (ord("c"), 13):
                    if self._close_current_polygon():
                        print(f"  polygon closed, total masks: {len(self.polygons)}")

                if key == ord("x"):
                    if self.current_poly:
                        self.current_poly.clear()
                        print("  current polygon cleared")

                if key in (26, ord("z")):
                    if self.current_poly:
                        self.current_poly.pop()
                        print("  removed last point from current polygon")
                    elif self.polygons:
                        self.polygons.pop()
                        print(f"  removed last saved mask, left: {len(self.polygons)}")

                if key == ord("s"):
                    if self.current_poly:
                        print("  current polygon not closed, auto-closing before save")
                        self._close_current_polygon()
                    self._save(img_path, img)
                    print(f"  saved {img_path.name}: {len(self.polygons)} mask(s)")
                    self.idx = min(self.idx + 1, len(self.image_paths) - 1)
                    break

                if key in (ord("d"), 83):
                    self.idx = min(self.idx + 1, len(self.image_paths) - 1)
                    break

                if key in (ord("a"), 81):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO segmentation polygon annotator")
    parser.add_argument(
        "--class-name",
        default="rail",
        help="Single class name for masks (default: rail)",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="saved_frames/raw",
        help="Directory with source images (default: saved_frames/raw)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="datasets/agrobot_seg_v1",
        help="Output directory for YOLO-seg dataset (default: datasets/agrobot_seg_v1)",
    )
    args = parser.parse_args()

    out = Path(args.output).resolve()
    print(f"Class   : {args.class_name}")
    print(f"Input   : {Path(args.input).resolve()}")
    print(f"Output  : {out}")
    print(f"  images → {out / 'images'}")
    print(f"  labels → {out / 'labels'}")
    print(f"  config → {out / 'dataset.yaml'}")
    print()
    print("Controls:")
    print("  LMB         add point")
    print("  C / Enter   close current polygon")
    print("  X           clear current polygon")
    print("  Z / Ctrl+Z  undo point / last polygon")
    print("  S           save & next image")
    print("  A / D       previous / next image")
    print("  Q / Esc     quit")
    print()

    annotator = SegAnnotator(args.class_name, args.input, args.output)
    annotator.run()


if __name__ == "__main__":
    main()
