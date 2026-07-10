#!/usr/bin/env python3
"""Record a RealSense color camera directly to MP4 without Docker."""

from __future__ import annotations

import argparse
import datetime as dt
import http.server
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Protocol

import cv2
import numpy as np


WINDOW_NAME = "RealSense MP4 Recorder"


class PreviewState:
    def __init__(self) -> None:
        self.frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.running = True

    def update(self, frame: np.ndarray) -> None:
        with self.lock:
            self.frame = frame.copy()

    def jpeg(self, quality: int) -> bytes | None:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return encoded.tobytes() if ok else None


class PreviewHandler(http.server.BaseHTTPRequestHandler):
    server: "PreviewServer"

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"""<!doctype html>
<html>
<head>
  <title>RealSense Preview</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    .wrap { display: grid; place-items: center; min-height: 100vh; gap: 12px; }
    img { max-width: 100vw; max-height: 94vh; object-fit: contain; }
    .hint { opacity: .75; font-size: 14px; }
  </style>
</head>
<body>
  <div class="wrap">
    <img src="/stream.mjpg" />
    <div class="hint">Stop recording with Ctrl+C in the terminal.</div>
  </div>
</body>
</html>"""
            )
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        state = self.server.preview_state
        while state.running:
            jpeg = state.jpeg(self.server.jpeg_quality)
            if jpeg is None:
                time.sleep(0.05)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(1.0 / max(1.0, self.server.preview_fps))

    def log_message(self, _format: str, *_args: object) -> None:
        return


class PreviewServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        state: PreviewState,
        preview_fps: float,
        jpeg_quality: int,
    ) -> None:
        super().__init__(server_address, PreviewHandler)
        self.preview_state = state
        self.preview_fps = preview_fps
        self.jpeg_quality = jpeg_quality


def ffmpeg_exe() -> str | None:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


class FfmpegMp4Writer:
    def __init__(
        self,
        output: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        video_quality: str,
    ) -> None:
        ffmpeg = ffmpeg_exe()
        if ffmpeg is None:
            raise SystemExit("ffmpeg not found. Install system ffmpeg or: pip install imageio-ffmpeg")

        output.parent.mkdir(parents=True, exist_ok=True)

        if video_quality == "standard":
            codec_args = ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"]
        elif video_quality == "hq":
            # Keep full chroma resolution without blocking capture too much.
            # "slow" looked better on paper, but it drops capture FPS in live recording.
            codec_args = ["-c:v", "libx264", "-crf", str(crf), "-preset", "ultrafast", "-pix_fmt", "yuv444p"]
        elif video_quality == "lossless":
            # Preserves RGB values best, but produces large files and less universally compatible MP4s.
            codec_args = ["-c:v", "libx264rgb", "-crf", "0", "-preset", "fast", "-pix_fmt", "bgr24"]
        else:
            raise ValueError(f"Unknown video quality: {video_quality}")

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
            *codec_args,
            "-movflags",
            "+faststart",
            str(output),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.output = output

    def write(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        print("Finalizing MP4, please wait...")
        while True:
            try:
                code = self.proc.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
            except KeyboardInterrupt:
                print("Still finalizing MP4; ignoring Ctrl+C until file is closed...")
        # ffmpeg exits with 255 when Python receives Ctrl+C while ffmpeg is still flushing.
        # The MP4 is finalized correctly in this case, so treat it as a normal stop.
        if code not in (0, 255):
            raise SystemExit(f"ffmpeg failed with exit code {code}")


class AsyncVideoWriter:
    def __init__(self, writer: FfmpegMp4Writer, max_queue: int) -> None:
        self.writer = writer
        self.frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max_queue)
        self.error: BaseException | None = None
        self.written = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                frame = self.frames.get()
                if frame is None:
                    break
                self.writer.write(frame)
                self.written += 1
        except BaseException as exc:
            self.error = exc

    def write(self, frame: np.ndarray) -> None:
        if self.error is not None:
            raise self.error
        self.frames.put(frame.copy())

    def close(self) -> int:
        self.frames.put(None)
        while self.thread.is_alive():
            try:
                self.thread.join(timeout=1)
            except KeyboardInterrupt:
                print("Still draining video frames; ignoring Ctrl+C until writer stops...")
        if self.error is not None:
            raise self.error
        self.writer.close()
        return self.written


class FrameSource(Protocol):
    width: int
    height: int
    fps: float

    def start(self) -> None:
        ...

    def read(self) -> np.ndarray | None:
        ...

    def stop(self) -> None:
        ...


class OpenCvCameraSource:
    """Read RealSense color via Linux V4L2, e.g. /dev/video0."""

    def __init__(self, device: int | str, width: int, height: int, fps: int, fourcc: str) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = float(fps)
        self.fourcc = fourcc
        self.cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise SystemExit(
                f"Cannot open camera {self.device}. Try --list-devices, choose another --device, "
                "or close apps using the camera."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        if self.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or self.fps)
        self.width = actual_w or self.width
        self.height = actual_h or self.height
        self.fps = actual_fps or self.fps
        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
        print(f"Camera: {self.device} via OpenCV/V4L2")
        print(f"Requested input format: {self.fourcc}")
        print(f"Actual stream: {self.width}x{self.height}@{self.fps:.2f}, format={fourcc!r}")

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class RealSenseSdkSource:
    """Read RealSense color via librealsense SDK on the host."""

    def __init__(self, width: int, height: int, fps: int, stream_format: str) -> None:
        self.width = width
        self.height = height
        self.fps = float(fps)
        self.stream_format = stream_format
        self.pipeline = None
        self.align = None
        self.rs = None

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise SystemExit(
                "pyrealsense2 is not installed. Install it on host:\n"
                "  python3 -m pip install pyrealsense2\n"
                "Or use the fallback V4L2 backend:\n"
                "  python3 record_realsense_mp4.py --backend opencv --device /dev/video12"
            ) from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        rs_format = {
            "rgb8": rs.format.rgb8,
            "bgr8": rs.format.bgr8,
        }[self.stream_format]
        config.enable_stream(rs.stream.color, self.width, self.height, rs_format, int(self.fps))
        profile = self.pipeline.start(config)
        sensor = profile.get_device().first_color_sensor()
        for option in (rs.option.enable_auto_exposure, rs.option.enable_auto_white_balance):
            try:
                if sensor.supports(option):
                    sensor.set_option(option, 1.0)
            except RuntimeError:
                pass
        print("Camera: RealSense SDK")
        print(f"Color sensor: {sensor.get_info(rs.camera_info.name)}")

    def read(self) -> np.ndarray | None:
        if self.pipeline is None:
            return None
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        frame = np.asanyarray(color_frame.get_data())
        if self.stream_format == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


def list_devices(max_devices: int) -> None:
    print("Available OpenCV/V4L2 cameras:")
    found = False
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        found = True
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  --device {idx}: {width}x{height}@{fps:.2f}")
        cap.release()
    if not found:
        print("  none found")

    by_id = Path("/dev/v4l/by-id")
    if by_id.exists():
        print("\nStable device paths:")
        for path in sorted(by_id.iterdir()):
            target = path.resolve()
            print(f"  {path} -> {target}")


def resolve_realsense_device() -> str:
    by_id = Path("/dev/v4l/by-id")
    if by_id.exists():
        candidates: list[Path] = []
        for path in sorted(by_id.iterdir()):
            name = path.name.lower()
            if "realsense" in name and "video-index" in name:
                candidates.append(path)

        # D435 color is usually video-index0 and exposes YUYV. Depth/IR nodes expose
        # Z16/GREY/UYVY and can look grayscale or incorrectly decoded as color.
        for path in candidates:
            if "video-index0" not in path.name:
                continue
            if _device_supports_format(str(path.resolve()), "YUYV"):
                return str(path)

        # RealSense exposes data and metadata nodes. Prefer readable YUYV color nodes.
        for path in candidates:
            dev = str(path)
            if not _device_supports_format(str(path.resolve()), "YUYV"):
                continue
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            if ok and frame is not None:
                return dev

    for idx in range(20):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        ok = cap.isOpened()
        cap.release()
        if not ok:
            continue
        dev = f"/dev/video{idx}"
        try:
            props = subprocess.check_output(
                ["udevadm", "info", "--query=property", f"--name={dev}"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        if "RealSense" in props or "realsense" in props.lower():
            if _device_supports_format(dev, "YUYV"):
                return dev

    raise SystemExit("Could not find a RealSense V4L color device. Check connection or use --device /dev/video12.")


def _device_supports_format(device: str, fourcc: str) -> bool:
    try:
        output = subprocess.check_output(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return f"'{fourcc}'" in output


def opencv_has_gui() -> bool:
    info = cv2.getBuildInformation()
    for line in info.splitlines():
        if line.strip().startswith("GUI:"):
            return "NONE" not in line
    return False


def default_output_path() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("saved_frames") / "videos" / f"realsense_{stamp}.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record RealSense color stream directly to MP4")
    parser.add_argument("-o", "--output", type=Path, default=default_output_path(), help="Output MP4 path")
    parser.add_argument(
        "--backend",
        choices=("opencv", "realsense"),
        default="realsense",
        help="Capture backend: realsense uses pyrealsense2 on host (default); opencv uses /dev/video* fallback",
    )
    parser.add_argument(
        "--device",
        default="realsense",
        help="OpenCV camera index/path, or 'realsense' for auto-detect when --backend opencv (default: realsense)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Color stream width")
    parser.add_argument("--height", type=int, default=720, help="Color stream height")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS")
    parser.add_argument("--duration", type=float, default=0.0, help="Record duration in seconds; 0 means until Ctrl+C")
    parser.add_argument("--crf", type=int, default=10, help="H.264 quality: lower is better (default: 10)")
    parser.add_argument(
        "--video-quality",
        choices=("standard", "hq", "lossless"),
        default="hq",
        help="Encoding quality: standard=yuv420p, hq=yuv444p, lossless=libx264rgb (default: hq)",
    )
    parser.add_argument(
        "--rs-format",
        choices=("rgb8", "bgr8"),
        default="rgb8",
        help="RealSense SDK color stream format (default: rgb8, converted to BGR for encoding)",
    )
    parser.add_argument("--preview", action="store_true", help="Show live preview while recording")
    parser.add_argument(
        "--preview-backend",
        choices=("opencv", "web"),
        default="opencv",
        help="Preview UI backend: opencv window or web page (default: opencv)",
    )
    parser.add_argument("--preview-host", default="127.0.0.1", help="Preview web server host (default: 127.0.0.1)")
    parser.add_argument("--preview-port", type=int, default=8090, help="Preview web server port (default: 8090)")
    parser.add_argument("--preview-fps", type=float, default=15.0, help="Preview stream FPS (default: 15)")
    parser.add_argument("--preview-quality", type=int, default=80, help="Preview JPEG quality 1-100 (default: 80)")
    parser.add_argument("--warmup-frames", type=int, default=15, help="Drop initial frames while auto exposure settles")
    parser.add_argument("--queue-size", type=int, default=300, help="Frame buffer size for async video writer")
    parser.add_argument(
        "--input-fourcc",
        default="YUYV",
        help="Requested V4L2 input format for --backend opencv (default: YUYV for RealSense color)",
    )
    parser.add_argument("--list-devices", action="store_true", help="List available OpenCV cameras and exit")
    parser.add_argument("--scan-devices", type=int, default=20, help="Max camera index for --list-devices")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_devices(args.scan_devices)
        return

    source: FrameSource
    if args.backend == "opencv":
        if args.device.lower() == "realsense":
            device = resolve_realsense_device()
            print(f"Auto-selected RealSense device: {device} -> {os.path.realpath(device)}")
        else:
            device: int | str = int(args.device) if args.device.isdigit() else args.device
        source = OpenCvCameraSource(device, args.width, args.height, args.fps, args.input_fourcc)
    else:
        source = RealSenseSdkSource(args.width, args.height, args.fps, args.rs_format)

    print(f"Starting RealSense recording backend={args.backend}")
    print(f"Output: {args.output.resolve()}")
    print(f"Quality: {args.video_quality}, CRF {args.crf}")

    source.start()
    frame_count = 0
    start_t = time.monotonic()
    capture_elapsed = 0.0

    preview_state: PreviewState | None = None
    preview_server: PreviewServer | None = None
    preview_thread: threading.Thread | None = None
    if args.preview:
        if args.preview_backend == "opencv":
            if not opencv_has_gui():
                raise SystemExit(
                    "Your current cv2 build has no GUI support (GUI: NONE), so OpenCV preview cannot open.\n"
                    "Fix it on the host with:\n"
                    "  python3 -m pip uninstall -y opencv-python-headless opencv-python\n"
                    "  python3 -m pip install opencv-python\n\n"
                    "Or use web preview for now:\n"
                    "  python3 record_realsense_mp4.py --preview --preview-backend web --device 0"
                )
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            print("Preview: OpenCV window (press Q/Esc in the window or Ctrl+C in terminal to stop)")
        else:
            preview_state = PreviewState()
            preview_server = PreviewServer(
                (args.preview_host, args.preview_port),
                preview_state,
                args.preview_fps,
                args.preview_quality,
            )
            preview_thread = threading.Thread(target=preview_server.serve_forever, daemon=True)
            preview_thread.start()
            print(f"Preview: http://{args.preview_host}:{args.preview_port}/")

    try:
        print(f"Warming up {max(0, args.warmup_frames)} frame(s) before recording...")
        for _ in range(max(0, args.warmup_frames)):
            source.read()

        writer = AsyncVideoWriter(
            FfmpegMp4Writer(
                args.output,
                source.width,
                source.height,
                source.fps,
                args.crf,
                args.video_quality,
            ),
            args.queue_size,
        )
        start_t = time.monotonic()
        print("Recording... press Ctrl+C to stop.")
        while True:
            if args.duration > 0 and time.monotonic() - start_t >= args.duration:
                break

            frame = source.read()
            if frame is None:
                continue

            if frame.shape[1] != source.width or frame.shape[0] != source.height:
                frame = cv2.resize(frame, (source.width, source.height))
            writer.write(frame)
            frame_count += 1

            if args.preview and args.preview_backend == "opencv":
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            elif preview_state is not None:
                preview_state.update(frame)
    except KeyboardInterrupt:
        print("\nStopping by Ctrl+C...")
    finally:
        capture_elapsed = max(1.0e-6, time.monotonic() - start_t)
        source.stop()
        if "writer" in locals():
            written_count = writer.close()
            if written_count != frame_count:
                print(f"Warning: captured {frame_count} frames, wrote {written_count} frames")
        if preview_state is not None:
            preview_state.running = False
        if preview_server is not None:
            preview_server.shutdown()
            preview_server.server_close()
        if preview_thread is not None:
            preview_thread.join(timeout=1.0)
        if args.preview and args.preview_backend == "opencv" and opencv_has_gui():
            cv2.destroyAllWindows()

    elapsed = max(1.0e-6, time.monotonic() - start_t)
    size_mb = args.output.stat().st_size / (1024 * 1024) if args.output.exists() else 0.0
    print(
        f"Wrote {frame_count} frames; capture {capture_elapsed:.2f}s "
        f"({frame_count / capture_elapsed:.2f} fps), total {elapsed:.2f}s, {size_mb:.1f} MB"
    )


if __name__ == "__main__":
    main()
