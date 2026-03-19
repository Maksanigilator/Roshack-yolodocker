



curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list



sudo apt-get update



sudo apt-get install -y nvidia-container-toolkit \
  nvidia-container-toolkit-base libnvidia-container-tools \
  libnvidia-container1


sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker




docker build -f docker/Dockerfile -t yolo:latest .

xhost +local:docker

docker run --gpus all -it --shm-size=8g \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v $(pwd)/datasets:/root/ros_ws/datasets \
  -v $(pwd)/ros_ws/data:/root/ros_ws/data \
  -v $(pwd)/ros_ws/weights:/root/ros_ws/weights \
  -v $(pwd)/ros_ws/demo.py:/root/ros_ws/demo.py \
  -v $(pwd)/ros_ws/train_ducks.py:/root/ros_ws/train_ducks.py \
  yolo:latest zsh






python3 train_ducks.py --epochs 100 --batch 8 --model weights/yolo26n.pt

python3 demo.py data/image.png -m weights/yolo26m.pt -c 0.7

python3 demo.py -m weights/runs/ducks_m/weights/best.pt datasets/yolo-rubber-ducks/data/image_100.jpg -c 0.1