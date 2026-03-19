#!/usr/bin/env zsh

source /opt/ros/jazzy/setup.zsh
[ -f /root/ros_ws/install/setup.zsh ] && source /root/ros_ws/install/setup.zsh

mkdir -p /root/ros_ws/weights
cp -n /opt/yolo_weights/*.pt /root/ros_ws/weights/ 2>/dev/null

exec "$@"
