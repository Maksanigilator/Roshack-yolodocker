#!/usr/bin/env python3
"""
Realtime RealSense + rail segmentation + centerline + PID visualization.

Features in one window:
  - Live RealSense RGB stream
  - Segmentation overlay from YOLO-seg
  - Left and right mask boundaries, computed centerline
  - Adjustable robot reference line (offset + tilt) via trackbars
  - Live errors and PID outputs
  - Save frame snapshots and persist calibration/PID settings
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    import pyrealsense2 as rs
except ImportError as exc:
    print(f"Failed to import pyrealsense2: {exc}", file=sys.stderr)
    print("Install pyrealsense2 and required system libs (for example libusb-1.0-0).", file=sys.stderr)
    raise SystemExit(1) from exc


WINDOW_NAME = "Rail Seg + PID"
BUTTON_TOP_LEFT = (20, 20)
BUTTON_SIZE = (220, 62)
SETTINGS_BUTTON_TOP_LEFT = (260, 20)
SETTINGS_BUTTON_SIZE = (260, 62)


@dataclass(frozen=True)
class SequenceModeParams:
    """Параметры режима sequence, все в одном месте."""

    yaw_align_tol_deg: float = 1.0  # Допуск по углу (градусы) для завершения фазы выравнивания по yaw.
    yaw_align_hold_s: float = 0.8  # Сколько секунд угол должен оставаться в допуске перед переходом дальше.
    yaw_lateral_ref_m: float = 0.2  # Масштаб для учета бокового смещения в yaw-выравнивании (чем меньше, тем сильнее влияние center_error).
    yaw_lateral_weight: float = 0.25  # Вес боковой добавки в yaw-контуре на фазе ALIGN_YAW (0..1).
    yaw_deadband_deg: float = 1.0  # Мертвая зона по yaw-ошибке: внутри нее cmd_yaw принудительно 0.
    lateral_align_tol_m: float = 0.035  # Допуск по боковой ошибке (метры) для завершения фазы бокового выравнивания.
    lateral_align_hold_s: float = 0.8  # Сколько секунд боковая ошибка должна быть в допуске.
    forward_speed: float = 0.5  # Постоянная скорость движения вперед в фазе FORWARD (м/с).
    forward_duration_s: float = 8.0  # Длительность фазы FORWARD (сек).
    backward_speed: float = 0.1  # Модуль скорости движения назад в фазе BACKWARD (м/с).
    backward_duration_s: float = 2.5  # Длительность фазы BACKWARD (сек).


@dataclass
class SequenceRuntime:
    phase: str = "ALIGN_YAW"
    phase_started_t: float = 0.0
    yaw_stable_since: float | None = None
    lateral_stable_since: float | None = None


# Меняй значения здесь, чтобы подстроить поведение sequence-режима.
SEQUENCE_MODE_PARAMS = SequenceModeParams()


@dataclass
class DetectionState:
    ok: bool = False
    center_error_m: float = 0.0
    heading_error_rad: float = 0.0
    center_x_target: float = 0.0
    center_x_lookahead: float = 0.0
    center_x_target_viz: float = 0.0
    center_x_lookahead_viz: float = 0.0
    robot_x_target: float = 0.0
    robot_x_lookahead: float = 0.0
    confidence: float = 0.0
    kalman_enabled: bool = True


@dataclass
class PidTerms:
    p: float
    i: float
    d: float
    output: float


class PidController:
    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = 1e9, integral_limit: float = 1e9) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = abs(output_limit)
        self.integral_limit = abs(integral_limit)
        self.integral = 0.0
        self.previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def update(self, error: float, dt: float) -> PidTerms:
        dt = max(dt, 1.0e-3)
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral + error * dt))
        derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
        self.previous_error = error
        p = self.kp * error
        i = self.ki * self.integral
        d = self.kd * derivative
        output = max(-self.output_limit, min(self.output_limit, p + i + d))
        return PidTerms(p=p, i=i, d=d, output=output)


class Kalman1D:
    """Simple constant-velocity Kalman filter for scalar measurements."""

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 4.0) -> None:
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.measurementMatrix = np.array([[1.0, 0.0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        self.kf.processNoiseCov = np.array(
            [[process_noise, 0.0], [0.0, process_noise]],
            dtype=np.float32,
        )
        self.kf.measurementNoiseCov = np.array([[measurement_noise]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32)
        self.initialized = False

    def reset(self) -> None:
        self.initialized = False

    def update(self, measurement: float) -> float:
        if not self.initialized:
            self.kf.statePost = np.array([[measurement], [0.0]], dtype=np.float32)
            self.initialized = True
            return measurement
        self.kf.predict()
        corrected = self.kf.correct(np.array([[measurement]], dtype=np.float32))
        return float(corrected[0, 0])


class UiButtonState:
    def __init__(self) -> None:
        self.save_frame_clicked = False
        self.save_settings_clicked = False

    def consume_save_frame(self) -> bool:
        if self.save_frame_clicked:
            self.save_frame_clicked = False
            return True
        return False

    def consume_save_settings(self) -> bool:
        if self.save_settings_clicked:
            self.save_settings_clicked = False
            return True
        return False


class CmdNavPublisher(Node):
    def __init__(self, topic: str, publish_enabled: bool) -> None:
        super().__init__("rail_seg_pid_viewer")
        self.topic = topic
        self.publish_enabled = publish_enabled
        self.pub = self.create_publisher(Twist, topic, 10)

    def publish_cmd(self, x: float, y: float, yaw: float) -> None:
        msg = Twist()
        msg.linear.x = float(x)
        msg.linear.y = float(y)
        msg.angular.z = float(yaw)
        if self.publish_enabled:
            self.pub.publish(msg)

    def publish_zero(self) -> None:
        self.publish_cmd(0.0, 0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime rail segmentation and PID tuning")
    parser.add_argument("--width", type=int, default=1280, help="Color stream width")
    parser.add_argument("--height", type=int, default=720, help="Color stream height")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.125, help="YOLO confidence threshold")
    parser.add_argument("--rail-class", type=int, default=0, help="Class id for rail mask")
    parser.add_argument("--approach-speed", type=float, default=0.12, help="Displayed forward speed command")
    parser.add_argument("--meters-per-pixel", type=float, default=0.0035, help="Pixel-to-meter scale for lateral error")
    parser.add_argument("--cmd-topic", default="/cmd_nav", help="ROS2 Twist topic for control output")
    parser.add_argument("--no-publish", action="store_true", help="Disable publishing Twist commands on startup")
    parser.add_argument("--no-kalman", action="store_true", help="Disable Kalman filter for mask-derived centerline points")
    parser.add_argument("--kalman-q", type=float, default=0.0001, help="Kalman process noise (Q)")
    parser.add_argument("--kalman-r", type=float, default=80.0, help="Kalman measurement noise (R)")
    parser.add_argument(
        "--control-mode",
        choices=("pid", "sequence"),
        default="pid",
        help="Control mode: pid (current behavior) or sequence (yaw->lateral->forward->backward cycle)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ros_ws/weights/runs/rail_seg_v1/best.pt"),
        help="Path to YOLO-seg model weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saved_frames") / "raw",
        help="Directory to save snapshots",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("saved_frames") / "rail_pid_settings.json",
        help="Path to calibration/PID settings json",
    )
    return parser.parse_args()


def in_button(x: int, y: int, top_left: tuple[int, int], size: tuple[int, int]) -> bool:
    x0, y0 = top_left
    w, h = size
    return x0 <= x <= x0 + w and y0 <= y <= y0 + h


def in_save_frame_button(x: int, y: int) -> bool:
    x0, y0 = BUTTON_TOP_LEFT
    w, h = BUTTON_SIZE
    return in_button(x, y, (x0, y0), (w, h))


def in_save_settings_button(x: int, y: int) -> bool:
    x0, y0 = SETTINGS_BUTTON_TOP_LEFT
    w, h = SETTINGS_BUTTON_SIZE
    return in_button(x, y, (x0, y0), (w, h))


def mouse_cb(event: int, x: int, y: int, _flags: int, state: UiButtonState) -> None:
    if event == cv2.EVENT_LBUTTONUP:
        if in_save_frame_button(x, y):
            state.save_frame_clicked = True
        elif in_save_settings_button(x, y):
            state.save_settings_clicked = True


def save_frame(frame: np.ndarray, output_dir: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_path = output_dir / f"realsense_{stamp}.png"
    cv2.imwrite(str(out_path), frame)
    return out_path


def build_rail_mask(result, shape_hw: tuple[int, int], rail_class: int) -> tuple[np.ndarray, float]:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    confidence = 0.0
    if result.masks is None:
        return mask, confidence

    polygons = result.masks.xy
    class_ids = []
    confs = []
    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
        confs = result.boxes.conf.detach().cpu().numpy().tolist() if result.boxes.conf is not None else []
    else:
        class_ids = [rail_class for _ in polygons]
        confs = [1.0 for _ in polygons]

    for poly, cls_id, conf in zip(polygons, class_ids, confs):
        if cls_id != rail_class:
            continue
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) < 3:
            continue
        cv2.fillPoly(mask, [pts], 255)
        confidence = max(confidence, float(conf))
    return mask, confidence


def robot_x_for_y(y: int, h: int, x_center: float, tilt_deg: float) -> float:
    dy = (h - 1) - y
    return x_center + math.tan(math.radians(tilt_deg)) * dy


def compute_centerline_state(mask: np.ndarray, x_center: float, tilt_deg: float, target_y: int, lookahead_y: int, mpp: float) -> tuple[DetectionState, list[tuple[int, int, int]]]:
    h, w = mask.shape[:2]
    rows: list[tuple[int, int, int]] = []
    for y in range(0, h, 2):
        xs = np.where(mask[y] > 0)[0]
        if xs.size < 2:
            continue
        left_x = int(xs.min())
        right_x = int(xs.max())
        rows.append((y, left_x, right_x))

    state = DetectionState(ok=False)
    if len(rows) < 8:
        return state, rows

    y_vals = np.array([r[0] for r in rows], dtype=np.float32)
    center_vals = np.array([(r[1] + r[2]) * 0.5 for r in rows], dtype=np.float32)

    if target_y < y_vals.min() or target_y > y_vals.max() or lookahead_y < y_vals.min() or lookahead_y > y_vals.max():
        return state, rows

    center_target = float(np.interp(target_y, y_vals, center_vals))
    center_lookahead = float(np.interp(lookahead_y, y_vals, center_vals))

    robot_target = robot_x_for_y(target_y, h, x_center, tilt_deg)
    robot_lookahead = robot_x_for_y(lookahead_y, h, x_center, tilt_deg)
    lateral_error_px = center_target - robot_target
    lateral_error_m = lateral_error_px * mpp
    # Heading error relative to robot reference line orientation.
    heading_num = (center_target - center_lookahead) - (robot_target - robot_lookahead)
    heading_error_rad = math.atan2(heading_num, max(1.0, target_y - lookahead_y))

    state.ok = True
    state.center_error_m = lateral_error_m
    state.heading_error_rad = heading_error_rad
    state.center_x_target = center_target
    state.center_x_lookahead = center_lookahead
    state.robot_x_target = robot_target
    state.robot_x_lookahead = robot_lookahead
    return state, rows


def draw_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    rows: list[tuple[int, int, int]],
    state: DetectionState,
    conf: float,
    target_y: int,
    lookahead_y: int,
    robot_x: float,
    robot_tilt_deg: float,
    saved_count: int,
    output_dir: Path,
    lat_terms: PidTerms,
    head_terms: PidTerms,
    cmd_x: float,
    cmd_y: float,
    cmd_yaw: float,
    cmd_topic: str,
    publish_enabled: bool,
    kalman_enabled: bool,
    x_gate_ready: bool,
    x_gate_wait_s: float,
    cmd_x_ref: float,
    yaw_invert_enabled: bool,
    control_mode: str,
    sequence_phase: str,
    sequence_elapsed_s: float,
) -> np.ndarray:
    view = frame.copy()
    if np.count_nonzero(mask) > 0:
        color_mask = np.zeros_like(view)
        color_mask[:, :, 1] = mask
        view = cv2.addWeighted(view, 1.0, color_mask, 0.35, 0)

    for y, left_x, right_x in rows[::8]:
        center_x = (left_x + right_x) // 2
        cv2.circle(view, (left_x, y), 1, (40, 180, 255), -1)
        cv2.circle(view, (right_x, y), 1, (255, 200, 20), -1)
        cv2.circle(view, (center_x, y), 1, (220, 60, 240), -1)

    h, _ = view.shape[:2]
    x_bottom = int(robot_x_for_y(h - 1, h, robot_x, robot_tilt_deg))
    x_top = int(robot_x_for_y(0, h, robot_x, robot_tilt_deg))
    cv2.line(view, (x_bottom, h - 1), (x_top, 0), (255, 255, 255), 2)

    cv2.line(view, (0, target_y), (view.shape[1] - 1, target_y), (100, 255, 100), 1)
    cv2.line(view, (0, lookahead_y), (view.shape[1] - 1, lookahead_y), (100, 200, 255), 1)

    if state.ok:
        cv2.circle(view, (int(state.center_x_target_viz), target_y), 6, (0, 255, 0), -1)
        cv2.circle(view, (int(state.center_x_lookahead_viz), lookahead_y), 6, (0, 180, 255), -1)
        cv2.circle(view, (int(state.robot_x_target), target_y), 6, (255, 255, 255), 2)
        cv2.line(
            view,
            (int(state.robot_x_target), target_y),
            (int(state.center_x_target_viz), target_y),
            (0, 255, 255),
            2,
        )

    # Control command indicators: lateral shift arrow and yaw arrow.
    cmd_panel_x = view.shape[1] - 220
    cmd_panel_y = 18
    cmd_panel_w = 200
    cmd_panel_h = 110
    cv2.rectangle(view, (cmd_panel_x, cmd_panel_y), (cmd_panel_x + cmd_panel_w, cmd_panel_y + cmd_panel_h), (35, 35, 35), -1)
    cv2.rectangle(view, (cmd_panel_x, cmd_panel_y), (cmd_panel_x + cmd_panel_w, cmd_panel_y + cmd_panel_h), (220, 220, 220), 1)
    cv2.putText(view, "CONTROL", (cmd_panel_x + 52, cmd_panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)

    max_cmd_y = 0.7
    max_cmd_yaw = 1.2
    shift_norm = max(-1.0, min(1.0, cmd_y / max_cmd_y))
    yaw_norm = max(-1.0, min(1.0, cmd_yaw / max_cmd_yaw))

    # Lateral shift indicator.
    shift_cx = cmd_panel_x + 100
    shift_cy = cmd_panel_y + 48
    shift_len = int(70 * abs(shift_norm))
    if shift_len > 1:
        shift_end = (shift_cx + int(shift_len * (1 if shift_norm >= 0 else -1)), shift_cy)
        cv2.arrowedLine(view, (shift_cx, shift_cy), shift_end, (80, 255, 255), 3, tipLength=0.25)
    cv2.line(view, (shift_cx - 72, shift_cy), (shift_cx + 72, shift_cy), (90, 90, 90), 1)
    cv2.putText(view, "shift", (cmd_panel_x + 12, shift_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    # Yaw indicator.
    yaw_cx = cmd_panel_x + 100
    yaw_cy = cmd_panel_y + 86
    yaw_len = int(22 * abs(yaw_norm))
    if yaw_len > 1:
        yaw_end = (yaw_cx + int(yaw_len * (1 if yaw_norm >= 0 else -1)), yaw_cy)
        cv2.arrowedLine(view, (yaw_cx, yaw_cy), yaw_end, (120, 220, 120), 3, tipLength=0.35)
    cv2.line(view, (yaw_cx - 24, yaw_cy), (yaw_cx + 24, yaw_cy), (90, 90, 90), 1)
    yaw_dir = "CCW" if cmd_yaw >= 0 else "CW"
    cv2.putText(view, f"yaw {yaw_dir}", (cmd_panel_x + 12, yaw_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    x0, y0 = BUTTON_TOP_LEFT
    bw, bh = BUTTON_SIZE
    cv2.rectangle(view, (x0, y0), (x0 + bw, y0 + bh), (45, 170, 75), -1)
    cv2.rectangle(view, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), 2)
    cv2.putText(view, "SAVE [S / SPACE]", (x0 + 12, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)
    sx0, sy0 = SETTINGS_BUTTON_TOP_LEFT
    sbw, sbh = SETTINGS_BUTTON_SIZE
    cv2.rectangle(view, (sx0, sy0), (sx0 + sbw, sy0 + sbh), (35, 120, 220), -1)
    cv2.rectangle(view, (sx0, sy0), (sx0 + sbw, sy0 + sbh), (255, 255, 255), 2)
    cv2.putText(view, "SAVE SETTINGS [K]", (sx0 + 14, sy0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)

    info_x, info_y = 15, y0 + bh + 28
    mode_detail = (
        f"x_gate={'READY' if x_gate_ready else 'WAIT'} wait={x_gate_wait_s:.2f}s (|y|,|yaw| < 0.1 for 2.0s)"
        if control_mode == "pid"
        else "sequence: ALIGN_YAW -> ALIGN_LATERAL -> FORWARD -> BACKWARD"
    )
    lines = [
        f"saved={saved_count}  conf={conf:.2f}  detect={'YES' if state.ok else 'NO'}",
        f"mode={control_mode}  phase={sequence_phase}  phase_t={sequence_elapsed_s:.2f}s",
        f"center_err={state.center_error_m:+.4f} m  heading_err={math.degrees(state.heading_error_rad):+.2f} deg",
        f"cmd_x={cmd_x:+.3f} (ref={cmd_x_ref:+.3f})  cmd_y={cmd_y:+.4f}  cmd_yaw={cmd_yaw:+.4f}",
        mode_detail,
        f"PID_lat P={lat_terms.p:+.4f} I={lat_terms.i:+.4f} D={lat_terms.d:+.4f}",
        f"PID_yaw P={head_terms.p:+.4f} I={head_terms.i:+.4f} D={head_terms.d:+.4f}",
        f"topic={cmd_topic} publish={'ON' if publish_enabled else 'OFF'} kalman={'ON' if kalman_enabled else 'OFF'} yaw_inv={'ON' if yaw_invert_enabled else 'OFF'}",
        "P publish | O kalman | Y yaw invert | K save | R reset | Q quit",
        f"output={output_dir}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(view, txt, (info_x, info_y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1)
    return view


def trackbar_get(name: str) -> int:
    return cv2.getTrackbarPos(name, WINDOW_NAME)


def load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def create_trackbars(frame_w: int, frame_h: int, loaded: dict) -> None:
    cv2.createTrackbar("robot_x_pct", WINDOW_NAME, int(loaded.get("robot_x_pct", 50)), 100, lambda _v: None)
    cv2.createTrackbar("robot_tilt_deg+45", WINDOW_NAME, int(loaded.get("robot_tilt_off", 45)), 90, lambda _v: None)
    cv2.createTrackbar("target_y_pct", WINDOW_NAME, int(loaded.get("target_y_pct", 80)), 95, lambda _v: None)
    cv2.createTrackbar("lookahead_y_pct", WINDOW_NAME, int(loaded.get("lookahead_y_pct", 45)), 95, lambda _v: None)
    cv2.createTrackbar("kp_lat_x1000", WINDOW_NAME, int(loaded.get("kp_lat_x1000", 900)), 3000, lambda _v: None)
    cv2.createTrackbar("ki_lat_x1000", WINDOW_NAME, int(loaded.get("ki_lat_x1000", 0)), 1200, lambda _v: None)
    cv2.createTrackbar("kd_lat_x1000", WINDOW_NAME, int(loaded.get("kd_lat_x1000", 90)), 3000, lambda _v: None)
    cv2.createTrackbar("kp_yaw_x1000", WINDOW_NAME, int(loaded.get("kp_yaw_x1000", 1400)), 4000, lambda _v: None)
    cv2.createTrackbar("ki_yaw_x1000", WINDOW_NAME, int(loaded.get("ki_yaw_x1000", 0)), 1200, lambda _v: None)
    cv2.createTrackbar("kd_yaw_x1000", WINDOW_NAME, int(loaded.get("kd_yaw_x1000", 100)), 4000, lambda _v: None)
    cv2.createTrackbar("cmd_x_x1000", WINDOW_NAME, int(loaded.get("cmd_x_x1000", 120)), 500, lambda _v: None)
    cv2.createTrackbar("mpp_um", WINDOW_NAME, int(loaded.get("mpp_um", 3500)), 20000, lambda _v: None)

    cv2.setTrackbarPos("robot_x_pct", WINDOW_NAME, min(100, max(0, trackbar_get("robot_x_pct"))))
    cv2.setTrackbarPos("target_y_pct", WINDOW_NAME, min(95, max(5, trackbar_get("target_y_pct"))))
    cv2.setTrackbarPos("lookahead_y_pct", WINDOW_NAME, min(95, max(5, trackbar_get("lookahead_y_pct"))))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    model = YOLO(str(args.model))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    print(f"Starting RealSense color stream {args.width}x{args.height}@{args.fps}")
    profile = pipeline.start(config)
    sensor = profile.get_device().first_color_sensor()
    print(f"Color sensor: {sensor.get_info(rs.camera_info.name)}")
    print(f"Model: {args.model.resolve()}")
    print(f"Snapshots: {args.output_dir.resolve()}")
    print(f"Settings: {args.settings.resolve()}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    loaded = load_settings(args.settings)
    create_trackbars(args.width, args.height, loaded)

    button_state = UiButtonState()
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb, button_state)
    saved_count = 0

    lat_pid = PidController(0.0, 0.0, 0.0, output_limit=0.7, integral_limit=0.6)
    yaw_pid = PidController(0.0, 0.0, 0.0, output_limit=1.2, integral_limit=1.0)
    kalman_target_x = Kalman1D(process_noise=args.kalman_q, measurement_noise=args.kalman_r)
    kalman_lookahead_x = Kalman1D(process_noise=args.kalman_q, measurement_noise=args.kalman_r)
    kalman_enabled = not args.no_kalman
    yaw_invert_enabled = bool(loaded.get("yaw_invert_enabled", True))
    x_gate_hold_s = 2.0
    x_gate_threshold = 0.1
    align_stable_since: float | None = None
    last_t = time.monotonic()
    seq = SequenceRuntime(phase_started_t=last_t)
    rclpy.init(args=None)
    ros_node = CmdNavPublisher(topic=args.cmd_topic, publish_enabled=not args.no_publish)
    ros_node.get_logger().info(f"Publishing Twist to {args.cmd_topic}, enabled={ros_node.publish_enabled}")

    try:
        while True:
            now = time.monotonic()
            loop_dt = now - last_t
            last_t = now

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            h, w = frame.shape[:2]

            robot_x_pct = trackbar_get("robot_x_pct")
            robot_tilt_deg = trackbar_get("robot_tilt_deg+45") - 45
            target_y_pct = max(5, trackbar_get("target_y_pct"))
            lookahead_y_pct = max(5, min(target_y_pct - 2, trackbar_get("lookahead_y_pct")))
            kp_lat = trackbar_get("kp_lat_x1000") / 1000.0
            ki_lat = trackbar_get("ki_lat_x1000") / 1000.0
            kd_lat = trackbar_get("kd_lat_x1000") / 1000.0
            kp_yaw = trackbar_get("kp_yaw_x1000") / 1000.0
            ki_yaw = trackbar_get("ki_yaw_x1000") / 1000.0
            kd_yaw = trackbar_get("kd_yaw_x1000") / 1000.0
            approach_speed = trackbar_get("cmd_x_x1000") / 1000.0
            mpp = max(1.0e-6, trackbar_get("mpp_um") / 1_000_000.0)

            lat_pid.kp, lat_pid.ki, lat_pid.kd = kp_lat, ki_lat, kd_lat
            yaw_pid.kp, yaw_pid.ki, yaw_pid.kd = kp_yaw, ki_yaw, kd_yaw

            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            mask, conf = build_rail_mask(result, (h, w), args.rail_class)

            target_y = int((target_y_pct / 100.0) * (h - 1))
            lookahead_y = int((lookahead_y_pct / 100.0) * (h - 1))
            robot_x = (robot_x_pct / 100.0) * (w - 1)

            state, rows = compute_centerline_state(
                mask=mask,
                x_center=robot_x,
                tilt_deg=float(robot_tilt_deg),
                target_y=target_y,
                lookahead_y=lookahead_y,
                mpp=mpp,
            )
            state.confidence = conf
            state.kalman_enabled = kalman_enabled
            state.center_x_target_viz = state.center_x_target
            state.center_x_lookahead_viz = state.center_x_lookahead

            if state.ok and kalman_enabled:
                # Kalman is for visualization stability only. Control uses raw mask geometry.
                state.center_x_target_viz = kalman_target_x.update(state.center_x_target)
                state.center_x_lookahead_viz = kalman_lookahead_x.update(state.center_x_lookahead)

            lat_terms = PidTerms(0.0, 0.0, 0.0, 0.0)
            head_terms = PidTerms(0.0, 0.0, 0.0, 0.0)
            cmd_x = 0.0
            cmd_y = 0.0
            cmd_yaw = 0.0
            x_gate_ready = False
            wait_left = x_gate_hold_s

            if not state.ok:
                lat_pid.reset()
                yaw_pid.reset()
                kalman_target_x.reset()
                kalman_lookahead_x.reset()
                align_stable_since = None
                seq.phase = "ALIGN_YAW"
                seq.phase_started_t = now
                seq.yaw_stable_since = None
                seq.lateral_stable_since = None
            else:
                if args.control_mode == "pid":
                    # Принятая конвенция: center_error_m > 0 означает, что целевая линия правее робота.
                    # Поэтому cmd_y должен иметь тот же знак, чтобы ехать в сторону линии.
                    lat_terms = lat_pid.update(state.center_error_m, loop_dt)
                    head_terms = yaw_pid.update(state.heading_error_rad, loop_dt)
                    cmd_y = lat_terms.output
                    cmd_yaw = head_terms.output
                    if yaw_invert_enabled:
                        cmd_yaw = -cmd_yaw
                    stable = abs(cmd_y) < x_gate_threshold and abs(cmd_yaw) < x_gate_threshold
                    if stable:
                        if align_stable_since is None:
                            align_stable_since = now
                    else:
                        align_stable_since = None
                    ready = align_stable_since is not None and (now - align_stable_since) >= x_gate_hold_s
                    cmd_x = approach_speed if ready else 0.0
                    if align_stable_since is not None:
                        elapsed = now - align_stable_since
                        wait_left = max(0.0, x_gate_hold_s - elapsed)
                        x_gate_ready = elapsed >= x_gate_hold_s
                else:
                    params = SEQUENCE_MODE_PARAMS
                    lat_terms = lat_pid.update(state.center_error_m, loop_dt)
                    cmd_y = lat_terms.output

                    if seq.phase == "ALIGN_YAW":
                        # Боковую добавку используем только для стартового выравнивания yaw,
                        # чтобы не "залипал" знак поворота при смещении робота от целевой линии.
                        lateral_yaw_term = math.atan2(state.center_error_m, max(1e-6, params.yaw_lateral_ref_m))
                        yaw_error_for_control = state.heading_error_rad + params.yaw_lateral_weight * lateral_yaw_term
                        head_terms = yaw_pid.update(yaw_error_for_control, loop_dt)
                        cmd_yaw = -head_terms.output if yaw_invert_enabled else head_terms.output
                        if abs(math.degrees(yaw_error_for_control)) < params.yaw_deadband_deg:
                            cmd_yaw = 0.0
                        cmd_x = 0.0
                        cmd_y = 0.0
                        yaw_ok = abs(math.degrees(state.heading_error_rad)) <= params.yaw_align_tol_deg
                        if yaw_ok:
                            if seq.yaw_stable_since is None:
                                seq.yaw_stable_since = now
                            elif (now - seq.yaw_stable_since) >= params.yaw_align_hold_s:
                                seq.phase = "ALIGN_LATERAL"
                                seq.phase_started_t = now
                                seq.lateral_stable_since = None
                                lat_pid.reset()
                        else:
                            seq.yaw_stable_since = None

                    elif seq.phase == "ALIGN_LATERAL":
                        # На боковом выравнивании yaw-контур идет только от heading_error (без боковой добавки).
                        head_terms = yaw_pid.update(state.heading_error_rad, loop_dt)
                        cmd_yaw = -head_terms.output if yaw_invert_enabled else head_terms.output
                        if abs(math.degrees(state.heading_error_rad)) < params.yaw_deadband_deg:
                            cmd_yaw = 0.0
                        cmd_x = 0.0
                        lateral_ok = abs(state.center_error_m) <= params.lateral_align_tol_m
                        if lateral_ok:
                            if seq.lateral_stable_since is None:
                                seq.lateral_stable_since = now
                            elif (now - seq.lateral_stable_since) >= params.lateral_align_hold_s:
                                seq.phase = "FORWARD"
                                seq.phase_started_t = now
                        else:
                            seq.lateral_stable_since = None

                    elif seq.phase == "FORWARD":
                        # На движении вперед yaw также стабилизируем только по heading_error.
                        head_terms = yaw_pid.update(state.heading_error_rad, loop_dt)
                        cmd_yaw = -head_terms.output if yaw_invert_enabled else head_terms.output
                        if abs(math.degrees(state.heading_error_rad)) < params.yaw_deadband_deg:
                            cmd_yaw = 0.0
                        cmd_x = params.forward_speed
                        if (now - seq.phase_started_t) >= params.forward_duration_s:
                            seq.phase = "BACKWARD"
                            seq.phase_started_t = now
                            lat_pid.reset()
                            yaw_pid.reset()

                    elif seq.phase == "BACKWARD":
                        cmd_x = -params.backward_speed
                        cmd_y = 0.0
                        cmd_yaw = 0.0
                        lat_terms = PidTerms(0.0, 0.0, 0.0, 0.0)
                        head_terms = PidTerms(0.0, 0.0, 0.0, 0.0)
                        if (now - seq.phase_started_t) >= params.backward_duration_s:
                            seq.phase = "ALIGN_YAW"
                            seq.phase_started_t = now
                            seq.yaw_stable_since = None
                            seq.lateral_stable_since = None
                            lat_pid.reset()
                            yaw_pid.reset()

            ros_node.publish_cmd(cmd_x, cmd_y, cmd_yaw)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            view = draw_overlay(
                frame=frame,
                mask=mask,
                rows=rows,
                state=state,
                conf=conf,
                target_y=target_y,
                lookahead_y=lookahead_y,
                robot_x=robot_x,
                robot_tilt_deg=float(robot_tilt_deg),
                saved_count=saved_count,
                output_dir=args.output_dir,
                lat_terms=lat_terms,
                head_terms=head_terms,
                cmd_x=cmd_x,
                cmd_y=cmd_y,
                cmd_yaw=cmd_yaw,
                cmd_topic=args.cmd_topic,
                publish_enabled=ros_node.publish_enabled,
                kalman_enabled=kalman_enabled,
                x_gate_ready=x_gate_ready,
                x_gate_wait_s=wait_left,
                cmd_x_ref=approach_speed,
                yaw_invert_enabled=yaw_invert_enabled,
                control_mode=args.control_mode,
                sequence_phase=seq.phase if state.ok else "NO_DETECTION",
                sequence_elapsed_s=now - seq.phase_started_t,
            )
            cv2.imshow(WINDOW_NAME, view)

            key = cv2.waitKey(1) & 0xFF
            should_save = button_state.consume_save_frame() or key in (ord("s"), ord("S"), 32)
            if should_save:
                out_path = save_frame(frame, args.output_dir)
                saved_count += 1
                print(f"[{saved_count}] saved: {out_path}")

            if key in (ord("r"), ord("R")):
                lat_pid.reset()
                yaw_pid.reset()
                kalman_target_x.reset()
                kalman_lookahead_x.reset()
                align_stable_since = None
                seq.phase = "ALIGN_YAW"
                seq.phase_started_t = now
                seq.yaw_stable_since = None
                seq.lateral_stable_since = None
                ros_node.publish_zero()
                print("PID and Kalman filters reset.")

            if key in (ord("p"), ord("P")):
                ros_node.publish_enabled = not ros_node.publish_enabled
                if not ros_node.publish_enabled:
                    ros_node.publish_zero()
                ros_node.get_logger().info(f"Publish {'enabled' if ros_node.publish_enabled else 'disabled'}")

            if key in (ord("o"), ord("O")):
                kalman_enabled = not kalman_enabled
                kalman_target_x.reset()
                kalman_lookahead_x.reset()
                print(f"Kalman {'enabled' if kalman_enabled else 'disabled'}")

            if key in (ord("y"), ord("Y")):
                yaw_invert_enabled = not yaw_invert_enabled
                print(f"Yaw invert {'enabled' if yaw_invert_enabled else 'disabled'}")

            should_save_settings = button_state.consume_save_settings() or key in (ord("k"), ord("K"))
            if should_save_settings:
                settings = {
                    "robot_x_pct": robot_x_pct,
                    "robot_tilt_off": robot_tilt_deg + 45,
                    "target_y_pct": target_y_pct,
                    "lookahead_y_pct": lookahead_y_pct,
                    "kp_lat_x1000": int(kp_lat * 1000),
                    "ki_lat_x1000": int(ki_lat * 1000),
                    "kd_lat_x1000": int(kd_lat * 1000),
                    "kp_yaw_x1000": int(kp_yaw * 1000),
                    "ki_yaw_x1000": int(ki_yaw * 1000),
                    "kd_yaw_x1000": int(kd_yaw * 1000),
                    "cmd_x_x1000": int(approach_speed * 1000),
                    "mpp_um": int(mpp * 1_000_000),
                    "model": str(args.model),
                    "cmd_topic": args.cmd_topic,
                    "publish_enabled": ros_node.publish_enabled,
                    "kalman_enabled": kalman_enabled,
                    "yaw_invert_enabled": yaw_invert_enabled,
                    "kalman_q": args.kalman_q,
                    "kalman_r": args.kalman_r,
                    "control_mode": args.control_mode,
                    "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
                save_settings(args.settings, settings)
                print(f"Saved settings -> {args.settings}")

            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        ros_node.publish_zero()
        ros_node.destroy_node()
        rclpy.shutdown()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped realtime segmentation viewer.")


if __name__ == "__main__":
    main()
