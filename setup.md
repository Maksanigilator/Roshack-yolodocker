



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

docker run --gpus all -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v $(pwd)/datasets:/root/ros_ws/datasets \
  -v $(pwd)/ros_ws/data:/root/ros_ws/data \
  -v $(pwd)/ros_ws/weights:/root/ros_ws/weights \
  -v $(pwd)/ros_ws/demo.py:/root/ros_ws/demo.py \
  yolo:latest zsh

