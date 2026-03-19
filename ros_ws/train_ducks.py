import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Fine-tune YOLO26 on rubber ducks")
parser.add_argument("-m", "--model", default="weights/yolo26m.pt", help="base model weights")
parser.add_argument("-d", "--data", default="/root/ros_ws/datasets/rubber-ducks-yolo/dataset.yaml")
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-b", "--batch", type=int, default=16)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--name", default="ducks", help="run name")
args = parser.parse_args()

model = YOLO(args.model)
model.train(
    data=args.data,
    epochs=args.epochs,
    batch=args.batch,
    imgsz=args.imgsz,
    name=args.name,
    project="/root/ros_ws/weights/runs",
    patience=10,
    save=True,
    plots=True,
)
