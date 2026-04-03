# Kurat Phase 4 Bring-Up

## Expected Order
1. Source ROS 2 Humble.
2. Start the RealSense ROS wrapper.
3. Start Ollama and confirm the Qwen model is available.
4. Launch the Kurat ROS node.
5. Confirm topics and status output.
6. Publish a test query.

## Example Commands

### 1. Source ROS 2 Humble
```bash
source /opt/ros/humble/setup.bash
```

### 2. Start RealSense
```bash
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true
```

### 3. Start Ollama and pull the language model
```bash
ollama serve
ollama pull qwen2.5:3b-instruct
ollama pull moondream
```

### 4. Launch Kurat
If the package is in a workspace, build and source it first:
```bash
cd ~/your_ros2_ws
colcon build --packages-select kurat
source install/setup.bash
```

Then launch Kurat:
```bash
ros2 launch kurat kurat.launch.py \
  color_topic:=/camera/color/image_raw \
  depth_topic:=/camera/depth/image_rect_raw \
  enable_depth:=true \
  stale_frame_threshold:=1.0 \
  log_level:=INFO \
  ollama_host:=http://127.0.0.1:11434
```

### 5. Confirm topics
```bash
ros2 topic list | grep kurat
ros2 topic echo /kurat/status
ros2 topic echo /kurat/reply
```

### 6. Send a test query
```bash
ros2 topic pub --once /kurat/query std_msgs/msg/String "{data: 'What do you see?'}"
```

## Notes
- If a vision query arrives before a fresh frame is available, Kurat should reply with a clear no-frame or stale-frame message instead of crashing.
- `/kurat/status` publishes lightweight JSON status with the last query, pipeline used, and latest frame age when available.
