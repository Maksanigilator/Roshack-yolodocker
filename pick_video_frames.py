#!/usr/bin/env python3
"""
Interactive frame picker for dataset creation.

Use this tool to browse videos and save selected frames into an image dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
WINDOW_NAME = "Frame Picker"


def collect_videos(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return [path]
        raise SystemExit(f"Not a supported video file: {path}")

    if not path.exists():
        raise SystemExit(f"Input path not found: {path}")

    videos = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"No videos found in: {path}")
    return videos


def draw_overlay(
    frame,
    video_path: Path,
    video_index: int,
    video_count: int,
    frame_index: int,
    frame_count: int,
    saved_count: int,
    paused: bool,
):
    vis = frame.copy()
    h, w = vis.shape[:2]
    bar_height = 84
    cv2.rectangle(vis, (0, h - bar_height), (w, h), (20, 20, 20), -1)

    frame_total = frame_count if frame_count > 0 else -1
    line1 = (
        f"[{video_index + 1}/{video_count}] {video_path.name} | "
        f"frame {frame_index + 1}/{frame_total} | saved={saved_count} | "
        f"{'PAUSE' if paused else 'PLAY'}"
    )
    line2 = (
        "Space play/pause | A/D prev/next frame | J/L -/+30 | N/P next/prev video | "
        "S save frame | Q/Esc quit"
    )
    cv2.putText(vis, line1, (8, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)
    cv2.putText(vis, line2, (8, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)
    return vis


def build_frame_name(video_path: Path, frame_index: int) -> str:
    return f"{video_path.stem}_f{frame_index:06d}.png"


def open_capture(video_path: Path) -> tuple[cv2.VideoCapture, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, max(0, count)


def seek_frame(cap: cv2.VideoCapture, frame_index: int) -> tuple[bool, object]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return ok, frame


def ensure_output_writable(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".write_probe_tmp"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"Cannot write to output directory: {output_dir}\n"
            "Fix permissions or choose another --output path.\n"
            f"Details: {exc}"
        ) from exc


def run(input_path: Path, output_dir: Path, step: int) -> None:
    videos = collect_videos(input_path)
    ensure_output_writable(output_dir)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    video_idx = 0
    frame_idx = 0
    saved_count = 0
    paused = True

    cap, frame_count = open_capture(videos[video_idx])
    ok, frame = seek_frame(cap, frame_idx)
    if not ok:
        raise SystemExit(f"Failed to read first frame from {videos[video_idx]}")

    while True:
        shown = draw_overlay(
            frame,
            videos[video_idx],
            video_idx,
            len(videos),
            frame_idx,
            frame_count,
            saved_count,
            paused,
        )
        cv2.imshow(WINDOW_NAME, shown)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord("q")):
            break

        if key == ord(" "):
            paused = not paused

        if key in (ord("s"),):
            out_name = build_frame_name(videos[video_idx], frame_idx)
            out_path = output_dir / out_name
            if out_path.exists():
                out_path = output_dir / f"{videos[video_idx].stem}_f{frame_idx:06d}_{saved_count:04d}.png"
            ok = cv2.imwrite(str(out_path), frame)
            if ok:
                saved_count += 1
                print(f"Saved: {out_path}")
            else:
                print(f"Failed to save: {out_path}")

        if key in (ord("d"), 83):
            paused = True
            next_idx = frame_idx + 1
            if frame_count > 0:
                next_idx = min(next_idx, frame_count - 1)
            ok, next_frame = seek_frame(cap, next_idx)
            if ok:
                frame_idx = next_idx
                frame = next_frame

        if key in (ord("a"), 81):
            paused = True
            prev_idx = max(0, frame_idx - 1)
            ok, prev_frame = seek_frame(cap, prev_idx)
            if ok:
                frame_idx = prev_idx
                frame = prev_frame

        if key == ord("l"):
            paused = True
            target = frame_idx + step
            if frame_count > 0:
                target = min(target, frame_count - 1)
            ok, jumped = seek_frame(cap, target)
            if ok:
                frame_idx = target
                frame = jumped

        if key == ord("j"):
            paused = True
            target = max(0, frame_idx - step)
            ok, jumped = seek_frame(cap, target)
            if ok:
                frame_idx = target
                frame = jumped

        if key == ord("n"):
            if video_idx < len(videos) - 1:
                cap.release()
                video_idx += 1
                frame_idx = 0
                paused = True
                cap, frame_count = open_capture(videos[video_idx])
                ok, frame = seek_frame(cap, frame_idx)
                if not ok:
                    raise SystemExit(f"Failed to read first frame from {videos[video_idx]}")

        if key == ord("p"):
            if video_idx > 0:
                cap.release()
                video_idx -= 1
                frame_idx = 0
                paused = True
                cap, frame_count = open_capture(videos[video_idx])
                ok, frame = seek_frame(cap, frame_idx)
                if not ok:
                    raise SystemExit(f"Failed to read first frame from {videos[video_idx]}")

        if not paused and key == 255:
            if frame_count > 0 and frame_idx >= frame_count - 1:
                paused = True
                continue
            ok, next_frame = cap.read()
            if ok:
                frame_idx += 1
                frame = next_frame
            else:
                paused = True

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved frames: {saved_count}")
    print(f"Output dir: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive video frame picker")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("RealSence_data"),
        help="Video file or directory with videos (default: RealSence_data)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("datasets/agrobot_seg_v2/images"),
        help="Directory to save selected frames (default: datasets/agrobot_seg_v2/images)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=30,
        help="Jump size for J/L keys in frames (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step < 1:
        raise SystemExit("--step should be >= 1")
    run(args.input.resolve(), args.output.resolve(), args.step)


if __name__ == "__main__":
    main()
