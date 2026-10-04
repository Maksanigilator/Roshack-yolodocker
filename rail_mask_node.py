#!/usr/bin/env python3
"""Узел-сегментатор: кадр на входе, маска рельсов на выходе.

Запускается в контейнере `yolo:latest`, где есть ultralytics и видеокарта.
Камеру НЕ открывает: устройство занято узлом realsense2_camera в
навигационном контейнере, и второй клиент librealsense просто не получит
его — проверено. Поэтому здесь только подписка на готовый топик.

Обмен идёт через общий ROS_DOMAIN_ID. Оба контейнера на host-сети и
host-ipc, поэтому DDS несёт кадры через разделяемую память, а не через
петлю: цветной кадр 848x480 это 1.2 МБ, и на 30 Гц разница существенная.

ЧТО ЗДЕСЬ НАРОЧНО НЕ ДЕЛАЕТСЯ

Ни ПИД, ни центральной линии, ни «метров на пиксель». Всё это есть в
capture_realsense.py и в detector.py, но там геометрия считается в
пикселях, и метры получаются умножением на одну постоянную. Масштаб
пикселя зависит от дальности, поэтому такая оценка верна лишь на одной
строке кадра. Здесь маска отдаётся как есть, а в метрику её переводит
dropoff_detector.py по глубине и измеренной опорной плоскости.

Сама сегментация — перенос метода `_segment_mask` из
`greenhouse_pipe_rail_nav/detector.py` (проект
~/Desktop/Agrobot/greenhouse_pipe_rail_autodock): те же полигоны, та же
морфология, тот же порог площади. Скопировано, а не импортировано,
потому что тот пакет лежит в другом репозитории и в этот контейнер не
смонтирован, а зависимость ради одного метода не стоит сшивания двух
проектов.

ПРО МЕТКУ ВРЕМЕНИ. Маска публикуется с header исходного кадра — и
меткой, и frame_id. Это не формальность: при частоте ниже кадровой маска
относится к конкретному снимку, и потребитель обязан сопоставлять её с
глубиной по времени, а не брать последнюю пришедшую. Пока робот стоит,
разницы нет; на ходу маска уезжает относительно глубины.

Запускается службой rail_mask из docker-compose.yml:

    docker compose up rail_mask

Эта служба отдельная от pipe_rail_autodock: ей нужен UID хоста ради
разделяемой памяти Fast DDS, а под ним каталог /root недоступен, поэтому
проект монтируется в /ws. Подробности — в комментариях к службе.
"""
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

import cv2
import torch
from ultralytics import YOLO


class RailMask(Node):
    def __init__(self):
        super().__init__('rail_mask')
        p = self.declare_parameter
        p('image_topic', '/camera/color/image_raw')
        p('mask_topic', '/rails/mask')
        # Путь внутри контейнера: служба rail_mask монтирует корень
        # проекта в /ws, см. docker-compose.yml.
        p('model_path',
          '/ws/ros_ws/weights/runs/agrobot_seg_v2_fresh_v2/weights/best.pt')
        # Пустая строка = пусть ultralytics выбирает сам. В контейнере с
        # gpus:all это будет видеокарта.
        p('device', '')
        # 640 — родной размер обучения модели. Замерено на RTX 4080 на
        # двенадцати кадрах датасета: 7.6 мс против 15.6 мс при 960, и
        # рельсы найдены в 12 кадрах из 12 при обоих размерах. То есть
        # 960 вдвое дороже и ничего не добавляет.
        p('input_size', 640)
        p('conf_threshold', 0.35)
        p('iou_threshold', 0.45)
        p('max_det', 6)
        # У модели один класс, rail = 0.
        p('rail_class', 0)
        # Морфология и порог площади — значения из detector.py, подобраны
        # на этом же стенде.
        p('morph_close', 9)
        p('morph_open', 5)
        p('min_mask_area_px', 1200)
        # Верхняя частота обработки. Ограничение здесь НЕ ради модели:
        # 9.4 мс на кадр это больше ста герц, вчетверо выше частоты камеры.
        # Значение выбрано по ПОТРЕБИТЕЛЮ: маска нужна карте, а карту
        # строит RTAB-Map с частотой Rtabmap/DetectionRate = 2 Гц (см.
        # robot_nav.launch.py в Nav_Test). Всё, что приходит чаще, он
        # отбрасывает не обрабатывая, так что считать чаще — значит жечь
        # процессор впустую. Замерено: на 2 Гц узел занимает 4% одного
        # ядра, на 10 Гц занимал бы около 20%.
        #
        # Если частоту карты поднимут, это значение надо поднять следом.
        p('max_rate', 2.0)
        # Публиковать ли пустую маску, когда рельсов не нашлось. По
        # умолчанию да: молчание потребитель истолкует как «маска просто
        # ещё не пришла» и продолжит держать старую, а это хуже честного
        # «сейчас не вижу».
        p('publish_empty', True)
        # Топик с замерами: время модели, задержка приёма, своя загрузка
        # процессора и занятая видеопамять. Нужен стенду frame_pub.py из
        # Nav_Test/yolo_transport_test — он сводит это с круговой задержкой
        # и показывает, что в пути дорого, а что дёшево. Время модели
        # больше неоткуда взять: снаружи видно только сумму.
        p('stats_topic', '/rails/mask_stats')

        self.rail_class = int(self.get_parameter('rail_class').value)
        self.min_area = int(self.get_parameter('min_mask_area_px').value)
        self.k_close = self._kernel(self.get_parameter('morph_close').value)
        self.k_open = self._kernel(self.get_parameter('morph_open').value)
        rate = float(self.get_parameter('max_rate').value)
        self.min_period_ns = int(1e9 / rate) if rate > 0 else 0
        self.last_ns = 0
        self.seen = self.found = 0

        path = self.get_parameter('model_path').value
        self.get_logger().info(f'загружаю модель {path}')
        self.model = YOLO(path)
        dev = self.get_parameter('device').value
        self.device = dev if dev else None

        self.pub = self.create_publisher(
            Image, self.get_parameter('mask_topic').value, 5)
        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self.on_image,
            QoSPresetProfiles.SENSOR_DATA.value)
        self.stats = self.create_publisher(
            Float32MultiArray, self.get_parameter('stats_topic').value, 5)
        # Для расчёта собственной загрузки: такты процессора из /proc и
        # момент, на который они сняты. Проценты считаются от ОДНОГО ядра,
        # то есть 100% здесь — это полностью занятое ядро, а не вся машина.
        self.tick_hz = os.sysconf('SC_CLK_TCK')
        self.cpu_t0, self.cpu_ticks0 = time.monotonic(), self._ticks()
        self.infer_ms = []
        self.create_timer(5.0, self.report)
        self.get_logger().info(
            f"жду кадры из {self.get_parameter('image_topic').value}, "
            f"маску шлю в {self.get_parameter('mask_topic').value}, "
            f'не чаще {rate:.0f} Гц')

    @staticmethod
    def _kernel(size):
        """Ядро морфологии. Размер обязан быть нечётным, иначе центр ядра
        не определён и операция сдвигает картинку на полпикселя."""
        size = max(1, int(size) | 1)
        return (None if size <= 1 else
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))

    def on_image(self, m):
        self.seen += 1
        now = self.get_clock().now().nanoseconds
        if self.min_period_ns and now - self.last_ns < self.min_period_ns:
            return
        self.last_ns = now

        bgr = self.to_bgr(m)
        if bgr is None:
            self.get_logger().warn(f'кодировка {m.encoding} не поддержана',
                                   throttle_duration_sec=10.0)
            return

        # Задержка приёма: от метки кадра до момента, когда он сюда дошёл.
        # Часы у контейнеров общие — это один хост, — поэтому величина
        # осмысленная и меряет именно передачу «туда».
        stamp_ns = (int(m.header.stamp.sec) * 1_000_000_000
                    + int(m.header.stamp.nanosec))
        inbound_ms = (now - stamp_ns) * 1e-6

        t0 = time.perf_counter()
        mask = self.segment(bgr)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self.infer_ms.append(infer_ms)
        self.publish_stats(infer_ms, inbound_ms)

        if mask is None:
            if bool(self.get_parameter('publish_empty').value):
                mask = np.zeros(bgr.shape[:2], np.uint8)
            else:
                return
        else:
            self.found += 1
        self.publish(mask, m.header)

    @staticmethod
    def to_bgr(m):
        """Image -> BGR без cv_bridge.

        В образе стоит ros-jazzy-ros-base, а cv_bridge туда не входит, и
        тянуть ради перестановки каналов целый пакет незачем. Шаг строки
        учитываем честно: он не обязан равняться ширине на три.
        """
        if m.encoding not in ('rgb8', 'bgr8'):
            return None
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
        a = a[:, :m.width * 3].reshape(m.height, m.width, 3)
        return a[:, :, ::-1] if m.encoding == 'rgb8' else a

    def segment(self, bgr):
        try:
            res = self.model.predict(
                source=bgr,
                imgsz=int(self.get_parameter('input_size').value),
                conf=float(self.get_parameter('conf_threshold').value),
                iou=float(self.get_parameter('iou_threshold').value),
                max_det=int(self.get_parameter('max_det').value),
                device=self.device,
                verbose=False,
                # Маска в полном разрешении кадра, а не в сетке модели.
                # Это обязательно: глубина выровнена по цвету попиксельно,
                # и маска должна ложиться на неё без пересчёта.
                retina_masks=True,
            )
        except Exception as e:
            self.get_logger().error(f'сегментация упала: {e}',
                                    throttle_duration_sec=10.0)
            return None
        if not res:
            return None
        r = res[0]
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            return None

        h, w = bgr.shape[:2]
        out = np.zeros((h, w), np.uint8)
        cls = r.boxes.cls.detach().cpu().numpy().astype(np.int32)
        polys = r.masks.xy
        hit = False
        for i in range(min(len(cls), len(polys))):
            if int(cls[i]) != self.rail_class:
                continue
            poly = np.asarray(polys[i], np.float32)
            if poly.ndim != 2 or poly.shape[0] < 3:
                continue
            cv2.fillPoly(out, [np.round(poly).astype(np.int32)], 255)
            hit = True
        if not hit:
            return None

        if self.k_close is not None:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, self.k_close)
        if self.k_open is not None:
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, self.k_open)
        # Мелкие пятна — это не рельс, а срабатывание на блик или кромку.
        return out if np.count_nonzero(out) >= self.min_area else None

    def _ticks(self):
        """Потраченные процессом такты процессора, пользовательские плюс
        системные. Поля 14 и 15 в /proc/self/stat, нумерация с единицы."""
        try:
            f = open('/proc/self/stat').read().rsplit(')', 1)[1].split()
            return int(f[11]) + int(f[12])
        except Exception:
            return 0

    def cpu_percent(self):
        now, ticks = time.monotonic(), self._ticks()
        dt = now - self.cpu_t0
        if dt <= 0:
            return 0.0
        pct = (ticks - self.cpu_ticks0) / self.tick_hz / dt * 100.0
        self.cpu_t0, self.cpu_ticks0 = now, ticks
        return pct

    def publish_stats(self, infer_ms, inbound_ms):
        gpu_mb = (torch.cuda.memory_reserved() / 1e6
                  if torch.cuda.is_available() else 0.0)
        msg = Float32MultiArray()
        msg.data = [float(infer_ms), float(inbound_ms),
                    float(self.cpu_percent()), float(gpu_mb)]
        self.stats.publish(msg)

    def publish(self, mask, header):
        m = Image()
        # header исходного кадра целиком: метка нужна для сопоставления с
        # глубиной, frame_id — чтобы маска знала, из какой камеры она.
        m.header = header
        m.height, m.width = mask.shape
        m.encoding = 'mono8'
        m.is_bigendian = 0
        m.step = m.width
        m.data = np.ascontiguousarray(mask).tobytes()
        self.pub.publish(m)

    def report(self):
        if not self.seen:
            self.get_logger().warn('кадров нет: проверь топик и домен')
            return
        t = ''
        if self.infer_ms:
            a = np.array(self.infer_ms)
            t = (f', модель {np.median(a):.1f} мс '
                 f'(худшая {a.max():.1f}), видеопамять '
                 f'{torch.cuda.memory_reserved() / 1e6:.0f} МБ'
                 if torch.cuda.is_available() else
                 f', модель {np.median(a):.1f} мс (худшая {a.max():.1f})')
        self.get_logger().info(
            f'кадров {self.seen}, обработано {self.found} с рельсами{t}')
        self.seen = self.found = 0
        self.infer_ms.clear()


def main():
    rclpy.init()
    n = RailMask()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()


if __name__ == '__main__':
    main()
