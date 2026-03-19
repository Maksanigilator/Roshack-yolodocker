ARG BASE_IMAGE=ultralytics/ultralytics:latest
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    zsh \
    libgtk2.0-dev \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && chsh -s "$(which zsh)" root 2>/dev/null || true

COPY requirements.txt /tmp/requirements.txt
RUN pip install --force-reinstall -r /tmp/requirements.txt

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t bira \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions


RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    ros-dev-tools \
    ros-jazzy-ros-base \
    ros-jazzy-joy \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'eval "$(register-python-argcomplete3 ros2)"' >> ~/.zshrc && \
    echo 'eval "$(register-python-argcomplete3 colcon)"' >> ~/.zshrc

RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo '[ -f /root/ros_ws/install/setup.bash ] && source /root/ros_ws/install/setup.bash' >> ~/.bashrc

RUN echo "source /opt/ros/jazzy/setup.zsh" >> ~/.zshrc && \
    echo '[ -f /root/ros_ws/install/setup.zsh ] && source /root/ros_ws/install/setup.zsh' >> ~/.zshrc

WORKDIR /root/ros_ws




ARG YOLO_MODEL=yolo26n.pt
RUN mkdir -p /opt/yolo_weights \
    && python3 -c "from ultralytics import YOLO; YOLO('${YOLO_MODEL}')" \
    && mv ${YOLO_MODEL} /opt/yolo_weights/

ENV QT_QPA_PLATFORM=xcb
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/cv2/qt/fonts \
    && ln -s /root/.config/Ultralytics/Arial.ttf /usr/local/lib/python3.12/dist-packages/cv2/qt/fonts/Arial.ttf

COPY docker/entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh

ENTRYPOINT ["/tmp/entrypoint.sh"]
CMD ["zsh"]
