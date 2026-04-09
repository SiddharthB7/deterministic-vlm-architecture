# Deterministic VLM Architecture

This repository contains the core KURAT three-brain runtime:

- language intent routing and answer generation through Ollama-hosted LLMs
- visual semantic understanding through Moondream
- open-vocabulary object detection through YOLO-World
- orchestration logic with a shared power-aware execution manager for Jetson-friendly scheduling

This repo is intentionally code-focused. It does not include reports, result dumps, media assets, or presentation files.

## What is included

- `kurat_core/`: core runtime, orchestrator, model services, power-aware scheduler
- `kurat_io/`: frame providers
- `kurat_ros/`: ROS 2 node and launch integration
- `scripts/power_aware_demo.py`: simple scheduler demo
- `scripts/evaluate_power_aware_runtime.py`: runtime evaluation workflow
- `scripts/testing/architecture_test.py`: local image-based smoke test
- `setup.py`, `setup.cfg`, `package.xml`: packaging and ROS metadata

## What is not included

- thesis/report documents
- prior experiment folders
- result artifacts
- local sample images
- pre-downloaded model weights

## Python setup

Create and activate an environment, then install the Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Jetson or ROS hosts, install ROS 2 system dependencies separately. `rclpy`, `sensor_msgs`, `std_msgs`, and `cv_bridge` are expected from the ROS installation rather than from `pip`.

## Ollama setup

Start Ollama and pull the required models:

```bash
ollama serve
ollama pull qwen2.5:3b-instruct
ollama pull moondream
```

## YOLO-World weights

The default configured path is:

```text
assets/models/yolov8s-world.pt
```

If that file is missing, the runtime falls back to the bare model name `yolov8s-world.pt` so Ultralytics can resolve/download it on first use when supported by your environment. You can also place the weights manually under `assets/models/`.

## Local smoke test

Run the image-based architecture test with your own image:

```bash
python scripts/testing/architecture_test.py --image path/to/image.jpg --query "Describe the scene in front of you."
```

No sample image is bundled in this repository, so `--image` is required.

## Runtime evaluation

The evaluation script compares baseline vs power-aware behavior:

```bash
python scripts/evaluate_power_aware_runtime.py
```

If no sample image is available under `assets/sample_images/raw/`, the evaluator falls back to mocked scheduler validation. To exercise the real image path, place one of the following in `assets/sample_images/raw/`:

- `test_image.png`
- `test2.jpg`
- `captured_image.jpg`
- `captured_20260128_215752.jpg`

## ROS 2 launch

After building the package in a ROS 2 workspace:

```bash
ros2 launch kurat kurat.launch.py \
  color_topic:=/camera/color/image_raw \
  depth_topic:=/camera/depth/image_rect_raw \
  enable_depth:=true \
  stale_frame_threshold:=1.0 \
  ollama_host:=http://127.0.0.1:11434 \
  power_aware_mode:=true \
  heavy_task_max_concurrency:=1 \
  allow_concurrent_heavy_inference:=false \
  inference_cooldown_ms:=300 \
  prompt_min_interval_ms:=500 \
  post_model_switch_delay_ms:=500 \
  enable_telemetry:=true \
  prefer_smaller_models:=true
```

## Notes

- The runtime expects Ollama to be reachable.
- Moondream may sometimes return free-form text instead of strict JSON; the code handles this with a conservative fallback.
- The power-aware manager is designed to serialize heavy tasks and reduce overlapping inference load on Jetson-class hardware.
