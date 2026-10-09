#!/usr/bin/env python3
"""Convert RealSense Viewer ROS1 bag color stream to MP4."""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

try:
    from rosbags.rosbag1 import Reader
except ImportError as exc:
    print("Install dependency: pip install rosbags opencv-python", file=sys.stderr)
    raise SystemExit(1) from exc

DEFAULT_TOPIC = "/device_0/sensor_1/Color_0/image/data"


def parse_ros1_image(raw: bytes) -> tuple[int, int, str, np.ndarray]:
    off = 0
    off += 4  # seq
    off += 8  # stamp
    (flen,) = struct.unpack_from("<I", raw, off)
    off += 4 + flen
    height, width = struct.unpack_from("<II", raw, off)
    off += 8
    (elen,) = struct.unpack_from("<I", raw, off)
    off += 4
    encoding = raw[off : off + elen].decode()
    off += elen + 1  # is_bigendian
    off += 4  # step
    (dlen,) = struct.unpack_from("<I", raw, off)
    off += 4
    data = raw[off : off + dlen]
    img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, -1)
    if encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == "bgr8":
        pass
    elif encoding == "rgba8":
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif encoding == "mono8":
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")
    return height, width, encoding, img


def list_topics(bag_path: Path) -> None:
    with Reader(bag_path) as reader:
        print(f"duration: {reader.duration / 1e9:.3f}s")
        print(f"messages: {reader.message_count}")
        print("topics with image-like types:")
        for conn in reader.connections:
            if "Image" in conn.msgtype or conn.msgcount > 10:
                print(f"  {conn.topic}  ({conn.msgtype})  count={conn.msgcount}")


def _ffmpeg_exe() -> str | None:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def write_video_ffmpeg(
    frames: Iterator[np.ndarray],
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
) -> int:
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found. Install ffmpeg or: pip install imageio-ffmpeg")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.4f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    assert proc.stdin is not None
    for frame in frames:
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        count += 1
    proc.stdin.close()
    stderr = proc.stderr.read().decode() if proc.stderr else ""
    code = proc.wait()
    if code != 0:
        raise SystemExit(f"ffmpeg failed ({code}):\n{stderr[-2000:]}")
    return count


def write_video_opencv(
    frames: Iterator[np.ndarray],
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    codec: str,
) -> int:
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Failed to open video writer for {output_path}")

    count = 0
    for frame in frames:
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        count += 1
    writer.release()
    return count


def bag_to_video(
    bag_path: Path,
    output_path: Path,
    topic: str,
    fps: float | None,
    encoder: str,
    codec: str,
    crf: int,
    png_dir: Path | None,
) -> None:
    frames: list[np.ndarray] = []
    timestamps: list[int] = []

    with Reader(bag_path) as reader:
        for conn, ts, raw in reader.messages():
            if conn.topic != topic:
                continue
            frames.append(parse_ros1_image(raw)[3])
            timestamps.append(ts)

    if not frames:
        raise SystemExit(f"No frames found on topic: {topic}")

    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(str(png_dir / f"frame_{i:06d}.png"), frame)
        print(f"Wrote {len(frames)} PNG frames -> {png_dir.resolve()}")
        return

    if fps is None:
        if len(timestamps) > 1:
            dt = (timestamps[-1] - timestamps[0]) / 1e9
            fps = (len(frames) - 1) / dt if dt > 0 else 30.0
        else:
            fps = 30.0

    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if encoder == "ffmpeg":
        count = write_video_ffmpeg(iter(frames), output_path, width, height, fps, crf)
    else:
        count = write_video_opencv(iter(frames), output_path, width, height, fps, codec)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Wrote {count} frames @ {fps:.2f} fps, {width}x{height}, "
        f"encoder={encoder}, size={size_mb:.1f} MB -> {output_path.resolve()}"
    )


def parse_args() -> argparse.Namespace:
    default_encoder = "ffmpeg" if _ffmpeg_exe() else "opencv"
    parser = argparse.ArgumentParser(description="Convert RealSense ROS1 bag to MP4")
    parser.add_argument("bag", type=Path, help="Input .bag file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: <bag_stem>.mp4)",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Image topic (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument("--fps", type=float, default=None, help="Output FPS (default: from bag timestamps)")
    parser.add_argument(
        "--encoder",
        choices=("ffmpeg", "opencv"),
        default=default_encoder,
        help=f"Video encoder backend (default: {default_encoder})",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=15,
        help="H.264 quality for ffmpeg encoder, lower=better (default: 15, visually lossless ~10)",
    )
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc when --encoder opencv (default: mp4v)")
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="Export lossless PNG frames instead of video",
    )
    parser.add_argument("--list-topics", action="store_true", help="List bag topics and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bag.exists():
        raise SystemExit(f"Bag not found: {args.bag}")

    if args.list_topics:
        list_topics(args.bag)
        return

    output = args.output or args.bag.with_suffix(".mp4")
    bag_to_video(
        args.bag,
        output,
        args.topic,
        args.fps,
        args.encoder,
        args.codec,
        args.crf,
        args.png_dir,
    )


if __name__ == "__main__":
    main()
