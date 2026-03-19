# Roshack-yolodocker

Docker-окружение для запуска YOLO26 с поддержкой GPU (NVIDIA) и визуальным выводом через X11.

## Быстрый старт

Полная инструкция по установке NVIDIA Container Toolkit и запуску — в [setup.md](setup.md).

```bash
docker build -f docker/Dockerfile -t yolo:latest .

xhost +local:docker

docker run --gpus all -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  yolo:latest zsh
```

## Структура файлов внутри контейнера

```
/
├── ultralytics/                    # Исходный код ultralytics (WORKDIR базового образа)
│   ├── ultralytics/                # Python-пакет ultralytics
│   ├── yolo26n.pt                  # Предзагруженная модель YOLO26n (базовый образ)
│   ├── pyproject.toml
│   └── ...
│
├── root/                           # Домашняя директория root (WORKDIR контейнера)
│   └── .config/
│       └── Ultralytics/
│           ├── Arial.ttf           # Шрифты для визуализации
│           └── Arial.Unicode.ttf
│
├── tmp/
│   └── entrypoint.sh              # Точка входа контейнера
│
├── opt/conda/                     # Conda-окружение (из базового образа pytorch)
│   ├── bin/
│   │   ├── python3
│   │   ├── pip
│   │   └── yolo                   # CLI ultralytics
│   └── lib/python3.*/site-packages/
│       ├── ultralytics/           # Установленный пакет ultralytics
│       ├── torch/                 # PyTorch с CUDA
│       ├── cv2/                   # OpenCV
│       └── ...
│
├── usr/
│   ├── bin/
│   │   ├── zsh                    # Оболочка zsh (добавлена в Dockerfile)
│   │   ├── git
│   │   ├── wget
│   │   └── curl
│   └── lib/
│       ├── libGL.so               # OpenGL (для визуализации)
│       ├── libglib-2.0.so
│       ├── libSM.so               # X11 Session Management
│       ├── libXext.so
│       └── libXrender.so
│
└── root/                          
    ├── .zshrc                     # Конфигурация zsh (тема bira, плагины)
    └── .oh-my-zsh/                # Oh My Zsh + плагины
        └── custom/plugins/
            ├── zsh-autosuggestions/
            └── zsh-completions/
```

### Ключевые пути

| Путь | Описание |
|------|----------|
| `/ultralytics/yolo26n.pt` | Предзагруженная модель YOLO26n |
| `/root/` | Рабочая директория при запуске |
| `/tmp/entrypoint.sh` | Entrypoint-скрипт |
| `/opt/conda/bin/yolo` | CLI для запуска YOLO |
| `/root/.config/Ultralytics/` | Конфиг и шрифты ultralytics |

## Сборка с другой моделью

По умолчанию скачивается `yolo26n.pt` (nano). Можно указать другую модель через build-arg:

```bash
docker build -f docker/Dockerfile -t yolo:latest --build-arg YOLO_MODEL=yolo26s.pt .
docker build -f docker/Dockerfile -t yolo:latest --build-arg YOLO_MODEL=yolo26m.pt .
docker build -f docker/Dockerfile -t yolo:latest --build-arg YOLO_MODEL=yolo26x.pt .
```

Доступные модели YOLO26: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x` (а также варианты `-seg`, `-cls`, `-pose`, `-obb`).
