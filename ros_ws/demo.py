import argparse
import os
import cv2
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

parser = argparse.ArgumentParser(description="YOLO26 detection demo")
parser.add_argument("source", nargs="?", default=os.path.join(DATA_DIR, "image.png"), help="image/video path")
parser.add_argument("-m", "--model", default=os.path.join(WEIGHTS_DIR, "yolo26n.pt"), help="model weights path")
parser.add_argument("-c", "--conf", type=float, default=0.25, help="confidence threshold (0-1)")
args = parser.parse_args()

model = YOLO(args.model)
results = model.predict(source=args.source, conf=args.conf)

for result in results:
    img = result.plot()
    cv2.imshow("YOLO26 Detection", img)

print("Press any key in the image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
