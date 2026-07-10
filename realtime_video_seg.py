#!/usr/bin/env python3
"""
Realtime segmentation player for videos.

Runs YOLO-seg on every frame and shows live overlay.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
WINDOW_NAME = "Realtime Segmentation"


def collect_videos(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() in VIDEO_EXTENSIONS:
            return [source]
        raise SystemExit(f"Unsupported video file: {source}")

    if not source.exists():
        raise SystemExit(f"Source path not found: {source}")

    videos = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"No videos found in: {source}")
    return videos


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-3:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, max(0, frame_count)


def draw_overlay(
    frame,
    video_name: str,
    video_index: int,
    total_videos: int,
    frame_index: int,
    frame_count: int,
    infer_ms: float,
    avg_ms: float,
    paused: bool,
    conf: float,
    imgsz: int,
):
    vis = frame.copy()
    h, w = vis.shape[:2]
    bar_h = 92
    cv2.rectangle(vis, (0, h - bar_h), (w, h), (20, 20, 20), -1)

    avg_fps = 1000.0 / avg_ms if avg_ms > 1e-6 else 0.0
    frame_total = frame_count if frame_count > 0 else -1
    line1 = (
        f"[{video_index + 1}/{total_videos}] {video_name} | "
        f"frame {frame_index + 1}/{frame_total} | "
        f"infer={infer_ms:.1f}ms avg={avg_ms:.1f}ms ({avg_fps:.1f} FPS)"
    )
    line2 = f"model conf={conf:.2f} imgsz={imgsz} | state={'PAUSE' if paused else 'PLAY'}"
    line3 = "Space play/pause | A/D prev/next frame | N/P next/prev video | Q/Esc quit"
    cv2.putText(vis, line1, (8, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1)
    cv2.putText(vis, line2, (8, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
    cv2.putText(vis, line3, (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 205, 205), 1)
    return vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime segmentation on videos")
    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        default=Path("RealSence_data"),
        help="Video file or directory with videos (default: RealSence_data)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=Path("ros_ws/weights/runs/segment/ros_ws/weights/runs/agrobot_seg_v2_fresh_v1/weights/best.pt"),
        help="YOLO-seg model path",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--device",
        default="0",
        help="Inference device passed to Ultralytics, e.g. '0', 'cpu' (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    model_path = args.model.resolve()

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if args.conf <= 0 or args.conf > 1:
        raise SystemExit("--conf should be in (0, 1]")
    if args.imgsz < 32:
        raise SystemExit("--imgsz should be >= 32")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required: pip install ultralytics") from exc

    videos = collect_videos(source)
    model = YOLO(str(model_path))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    video_idx = 0
    cap, src_fps, frame_count = open_video(videos[video_idx])
    frame_idx = 0
    paused = False
    avg_ms = 0.0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if video_idx < len(videos) - 1:
                    cap.release()
                    video_idx += 1
                    cap, src_fps, frame_count = open_video(videos[video_idx])
                    frame_idx = 0
                    continue
                break
        else:
            ok, frame = cap.read()
            if not ok:
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        t0 = time.perf_counter()
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0
        if avg_ms <= 1e-6:
            avg_ms = infer_ms
        else:
            avg_ms = 0.90 * avg_ms + 0.10 * infer_ms

        shown = result.plot()
        shown = draw_overlay(
            frame=shown,
            video_name=videos[video_idx].name,
            video_index=video_idx,
            total_videos=len(videos),
            frame_index=frame_idx,
            frame_count=frame_count,
            infer_ms=infer_ms,
            avg_ms=avg_ms,
            paused=paused,
            conf=args.conf,
            imgsz=args.imgsz,
        )
        cv2.imshow(WINDOW_NAME, shown)

        frame_delay_ms = max(1, int(1000.0 / max(1e-3, src_fps) - infer_ms))
        wait_ms = 30 if paused else frame_delay_ms
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
            continue

        if key in (ord("d"), 83):
            paused = True
            frame_idx = min(frame_idx + 1, max(0, frame_count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            continue

        if key in (ord("a"), 81):
            paused = True
            frame_idx = max(0, frame_idx - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            continue

        if key == ord("n"):
            if video_idx < len(videos) - 1:
                cap.release()
                video_idx += 1
                cap, src_fps, frame_count = open_video(videos[video_idx])
                frame_idx = 0
                paused = False
            continue

        if key == ord("p"):
            if video_idx > 0:
                cap.release()
                video_idx -= 1
                cap, src_fps, frame_count = open_video(videos[video_idx])
                frame_idx = 0
                paused = False
            continue

        if not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
