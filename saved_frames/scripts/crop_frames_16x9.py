#!/usr/bin/env python3
"""
Интерактивный выбор области с соотношением 16:9 (как 1280×720), затем обрезка и ресайз до 1280×720.

Зависимости: pip install pillow matplotlib

Примеры (из корня репозитория, кадры лежат в saved_frames рядом со скриптом):
  # Подобрать прямоугольник на первом кадре из папки, вывести координаты в консоль:
  python saved_frames/scripts/crop_frames_16x9.py --input saved_frames --interactive

  # Указать эталонный файл:
  python saved_frames/scripts/crop_frames_16x9.py --input saved_frames --ref frame_0001.png --interactive

  # Сохранить конфиг и обработать все:
  python saved_frames/scripts/crop_frames_16x9.py --input saved_frames --interactive --save-config saved_frames/crop_16x9.json
  python saved_frames/scripts/crop_frames_16x9.py --input saved_frames --load-config saved_frames/crop_16x9.json --output saved_frames/out

Нормализованные координаты (0..1) относительно ширины/высоты исходника: left, top, width, height.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:
    print("Нужен пакет Pillow: pip install pillow", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
except ImportError:
    print("Нужны numpy и matplotlib: pip install numpy matplotlib", file=sys.stderr)
    sys.exit(1)

ASPECT = 1280 / 720  # 16:9
OUT_W, OUT_H = 1280, 720

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    out = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def crop_16_9_from_selection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int] | None:
    """По произвольному прямоугольнику выбора — вписанный 16:9 по центру, затем ужатие под размер кадра."""
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return None
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if w / h >= ASPECT:
        cw, ch = w, w / ASPECT
    else:
        ch, cw = h, h * ASPECT
    if cw > img_w:
        cw = float(img_w)
        ch = cw / ASPECT
    if ch > img_h:
        ch = float(img_h)
        cw = ch * ASPECT
    left = cx - cw / 2
    top = cy - ch / 2
    left = max(0.0, min(left, img_w - cw))
    top = max(0.0, min(top, img_h - ch))
    return int(round(left)), int(round(top)), int(round(cw)), int(round(ch))


def to_normalized(
    left: int, top: int, cw: int, ch: int, img_w: int, img_h: int
) -> dict[str, float]:
    return {
        "left": left / img_w,
        "top": top / img_h,
        "width": cw / img_w,
        "height": ch / img_h,
        "ref_width": float(img_w),
        "ref_height": float(img_h),
    }


def from_normalized(norm: dict[str, Any], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    left = int(round(norm["left"] * img_w))
    top = int(round(norm["top"] * img_h))
    cw = min(int(round(norm["width"] * img_w)), img_w)
    ch = int(round(cw / ASPECT))
    if ch > img_h:
        ch = img_h
        cw = int(round(ch * ASPECT))
    left = max(0, min(left, img_w - cw))
    top = max(0, min(top, img_h - ch))
    return left, top, cw, ch


def run_interactive(ref_path: Path, save_config: Path | None) -> dict[str, Any]:
    # Без панели навигации: иначе Pan/Zoom перехватывают ЛКМ и выделение «не работает».
    plt.rcParams["toolbar"] = "None"

    img = Image.open(ref_path).convert("RGB")
    arr = np.asarray(img)
    img_w, img_h = img.size

    HANDLE_PX = 11
    MIN_SIDE = 16.0

    state: dict[str, Any] = {
        "norm": None,
        "pixel": None,
        "done": False,
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(arr, aspect="equal", interpolation="nearest")
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_visible(False)

    # Стартовый прямоугольник 16:9 по центру кадра
    cw = min(img_w * 0.65, img_h * ASPECT * 0.65)
    ch = cw / ASPECT
    if ch > img_h * 0.65:
        ch = img_h * 0.65
        cw = ch * ASPECT
    left = (img_w - cw) / 2
    top = (img_h - ch) / 2
    rect = [left, top, cw, ch]

    ui_patches: list[patches.Patch] = []
    drag_mode: list[str | None] = [None]
    move_off: list[float] = [0.0, 0.0]
    rb_start: list[float | None] = [None, None]
    handle_data_size: list[float | None] = [None]

    def event_to_xy(event) -> tuple[float, float] | None:
        if event.xdata is not None and event.ydata is not None:
            return float(event.xdata), float(event.ydata)
        try:
            inv = ax.transData.inverted()
            x, y = inv.transform((event.x, event.y))
        except Exception:
            return None
        x = float(np.clip(x, 0, img_w))
        y = float(np.clip(y, 0, img_h))
        return x, y

    def handle_size_data() -> float:
        if handle_data_size[0] is None:
            p0 = ax.transData.inverted().transform((0.0, 0.0))
            p1 = ax.transData.inverted().transform((float(HANDLE_PX), 0.0))
            handle_data_size[0] = max(abs(p1[0] - p0[0]), 1e-6)
        return handle_data_size[0]

    def corners_data() -> list[tuple[float, float]]:
        l, t, w, h = rect
        return [
            (l, t),
            (l + w, t),
            (l + w, t + h),
            (l, t + h),
        ]

    def hit_handle(ex: float, ey: float) -> int | None:
        """Индекс угла 0..3 (TL,TR,BR,BL) или None; ex,ey — пиксели окна matplotlib."""
        best: int | None = None
        best_d = HANDLE_PX + 1.0
        for i, (cx, cy) in enumerate(corners_data()):
            sx, sy = ax.transData.transform((cx, cy))
            d = float(np.hypot(sx - ex, sy - ey))
            if d < best_d and d <= HANDLE_PX:
                best_d = d
                best = i
        return best

    def inside_rect(mx: float, my: float) -> bool:
        l, t, w, h = rect
        return l <= mx <= l + w and t <= my <= t + h

    def fit_rect(l: float, t: float, w: float) -> tuple[float, float, float, float]:
        """Вписать 16:9 с левым верхним углом (l,t) и шириной w."""
        w = max(MIN_SIDE, min(w, img_w - l))
        h = w / ASPECT
        if t + h > img_h:
            h = img_h - t
            w = h * ASPECT
        w = max(MIN_SIDE, w)
        h = w / ASPECT
        if l + w > img_w:
            w = img_w - l
            h = w / ASPECT
        l = float(np.clip(l, 0, img_w - w))
        t = float(np.clip(t, 0, img_h - h))
        return l, t, w, h

    def resize_corner(which: int, mx: float, my: float) -> None:
        l, t, w, h = rect
        mx = float(np.clip(mx, 0, img_w))
        my = float(np.clip(my, 0, img_h))
        if which == 2:
            rect[0], rect[1], rect[2], rect[3] = fit_rect(l, t, max(MIN_SIDE, mx - l))
        elif which == 0:
            brx, bry = l + w, t + h
            nl = float(np.clip(mx, 0, max(0.0, brx - MIN_SIDE)))
            nw = brx - nl
            nt = bry - nw / ASPECT
            rect[0], rect[1], rect[2], rect[3] = fit_rect(nl, nt, nw)
        elif which == 1:
            bly = t + h
            nw = max(MIN_SIDE, mx - l)
            nt = bly - nw / ASPECT
            rect[0], rect[1], rect[2], rect[3] = fit_rect(l, nt, nw)
        else:
            trx = l + w
            nl = float(np.clip(mx, 0, max(0.0, trx - MIN_SIDE)))
            nw = trx - nl
            rect[0], rect[1], rect[2], rect[3] = fit_rect(nl, t, nw)

    def sync_state() -> None:
        l, t, w, h = rect
        state["pixel"] = {
            "left": int(round(l)),
            "top": int(round(t)),
            "width": int(round(w)),
            "height": int(round(h)),
        }
        state["norm"] = to_normalized(
            state["pixel"]["left"],
            state["pixel"]["top"],
            state["pixel"]["width"],
            state["pixel"]["height"],
            img_w,
            img_h,
        )

    def update_title() -> None:
        l, t, w, h = rect
        ax.set_title(
            "Прямоугольник 16:9 — тяните за углы (квадратики) или за область внутри, чтобы сдвинуть.\n"
            "Пустое место: протащите ЛКМ, чтобы задать новую область (как раньше).\n"
            "Enter — подтвердить | Esc — выход\n"
            f"Сейчас: {int(round(w))}×{int(round(h))} px (будет 1280×720 после ресайза)"
        )

    def redraw() -> None:
        for p in ui_patches:
            p.remove()
        ui_patches.clear()
        l, t, w, h = rect
        hs = handle_size_data()
        half = hs / 2
        main = patches.Rectangle(
            (l, t),
            w,
            h,
            linewidth=2.5,
            edgecolor="lime",
            facecolor=(0.2, 1.0, 0.2, 0.18),
            linestyle="-",
            zorder=5,
        )
        ax.add_patch(main)
        ui_patches.append(main)
        for cx, cy in corners_data():
            hdl = patches.Rectangle(
                (cx - half, cy - half),
                hs,
                hs,
                linewidth=1.5,
                edgecolor="white",
                facecolor="lime",
                zorder=6,
            )
            ax.add_patch(hdl)
            ui_patches.append(hdl)
        sync_state()
        update_title()
        fig.canvas.draw_idle()

    def on_press(event):
        if event.button != 1:
            return
        if event.inaxes is not ax:
            return
        xy = event_to_xy(event)
        if xy is None:
            return
        mx, my = xy
        hi = hit_handle(event.x, event.y)
        if hi is not None:
            drag_mode[0] = f"corner_{hi}"
            return
        if inside_rect(mx, my):
            drag_mode[0] = "move"
            move_off[0] = mx - rect[0]
            move_off[1] = my - rect[1]
            return
        rb_start[0], rb_start[1] = mx, my
        drag_mode[0] = "rubber"

    def on_motion(event):
        mode = drag_mode[0]
        if mode is None:
            return
        xy = event_to_xy(event)
        if xy is None:
            return
        mx, my = xy
        if mode == "move":
            nl = mx - move_off[0]
            nt = my - move_off[1]
            l, t, w, h = rect
            nl = float(np.clip(nl, 0, img_w - w))
            nt = float(np.clip(nt, 0, img_h - h))
            rect[0], rect[1] = nl, nt
            redraw()
        elif mode is not None and mode.startswith("corner_"):
            which = int(mode.split("_")[1])
            resize_corner(which, mx, my)
            redraw()
        elif mode == "rubber" and rb_start[0] is not None:
            x0, y0 = rb_start[0], rb_start[1]
            r = crop_16_9_from_selection(x0, y0, mx, my, img_w, img_h)
            if r is not None:
                rect[0], rect[1], rect[2], rect[3] = r
                redraw()

    def on_release(event):
        if event.button != 1:
            return
        drag_mode[0] = None
        rb_start[0] = rb_start[1] = None

    def on_key(event):
        if event.key in ("enter", "return"):
            state["done"] = True
            plt.close(fig)
        elif event.key == "escape":
            state["norm"] = None
            state["pixel"] = None
            state["done"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    fig.canvas.draw()
    redraw()
    plt.show()

    if state.get("norm") is None:
        print("Область не выбрана (или отменено Esc).", file=sys.stderr)
        sys.exit(1)

    norm = state["norm"]
    pix = state["pixel"]
    assert pix is not None

    print("\n--- Эталон:", ref_path)
    print("Размер исходника (px):", img_w, "×", img_h)
    print(
        "Обрезка (px): left=%d, top=%d, width=%d, height=%d"
        % (pix["left"], pix["top"], pix["width"], pix["height"])
    )
    print(
        "Нормализованно (0..1): left=%.6f, top=%.6f, width=%.6f, height=%.6f"
        % (norm["left"], norm["top"], norm["width"], norm["height"])
    )

    out = {
        "aspect": "16:9",
        "output_size": [OUT_W, OUT_H],
        "reference_image": str(ref_path.resolve()),
        "normalized_crop": {k: norm[k] for k in ("left", "top", "width", "height")},
    }
    if save_config:
        save_config.parent.mkdir(parents=True, exist_ok=True)
        with open(save_config, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("Конфиг записан:", save_config.resolve())

    return out


def process_folder(
    input_dir: Path,
    output_dir: Path,
    norm: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(input_dir)
    if not paths:
        print("Нет изображений в", input_dir, file=sys.stderr)
        sys.exit(1)

    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            left, top, cw, ch = from_normalized(norm, w, h)
            cropped = im.crop((left, top, left + cw, top + ch))
            resized = cropped.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
            out_path = output_dir / p.name
            resized.save(out_path, quality=95)
        print(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Обрезка 16:9 и ресайз до 1280×720")
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Папка с кадрами (saved_frames)",
    )
    ap.add_argument(
        "--ref",
        type=Path,
        default=None,
        help="Эталонное изображение для интерактива (по умолчанию — первый файл в папке)",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Открыть окно и выбрать прямоугольник",
    )
    ap.add_argument(
        "--save-config",
        type=Path,
        default=None,
        help="Куда сохранить JSON с нормализованной обрезкой",
    )
    ap.add_argument(
        "--load-config",
        type=Path,
        default=None,
        help="Загрузить JSON (без интерактива)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Папка для результатов (при обработке по конфигу)",
    )
    args = ap.parse_args()

    if not args.input.is_dir():
        print("Нет такой папки:", args.input, file=sys.stderr)
        sys.exit(1)

    if args.interactive:
        ref = args.ref
        if ref is None:
            imgs = list_images(args.input)
            if not imgs:
                print("В папке нет изображений:", args.input, file=sys.stderr)
                sys.exit(1)
            ref = imgs[0]
        elif not ref.is_file():
            print("Нет файла:", ref, file=sys.stderr)
            sys.exit(1)
        cfg = run_interactive(ref, args.save_config)
        if args.output:
            process_folder(args.input, args.output, cfg["normalized_crop"])
        return

    if args.load_config:
        with open(args.load_config, encoding="utf-8") as f:
            cfg = json.load(f)
        norm = cfg["normalized_crop"]
        out = args.output
        if out is None:
            print("Укажите --output для сохранения кадров", file=sys.stderr)
            sys.exit(1)
        process_folder(args.input, out, norm)
        return

    ap.print_help()
    print(
        "\nУкажите --interactive для выбора области или --load-config для пакетной обработки.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
