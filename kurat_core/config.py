from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelConfig:
    # Jetson-friendly default language brain for intent routing and answer phrasing.
    language_model_name: str = "qwen2.5:3b-instruct"
    moondream_model_name: str = "moondream"
    yolo_model_path: str = "yolov8s-world.pt"
    ollama_generate_url: str = "http://127.0.0.1:11434/api/generate"
    ollama_host: str = "http://127.0.0.1:11434"
    intent_timeout_s: int = 120
    chat_timeout_s: int = 180
    yolo_semantic_timeout_s: int = 20
    yolo_device: str = "cpu"
    moondream_device: str = ""

    @property
    def mistral_model_name(self) -> str:
        return self.language_model_name


@dataclass(slots=True)
class RuntimeConfig:
    debug: bool = False
    log_level: str = "INFO"
    debug_save_images: bool = False
    debug_image_dir: str = "debug_frames"
    keep_temp_frames: bool = False
    enable_moondream_fallback_on_find: bool = True
    max_frame_age_s: float = 1.0
    skip_stale_frames: bool = True
    moondream_frame_max_dim: int = 768
    yolo_frame_max_dim: int = 960
    yolo_imgsz: int = 1280
    yolo_conf: float = 0.20
    yolo_iou: float = 0.5
    yolo_max_det: int = 200


@dataclass(slots=True)
class TopicConfig:
    ros_color_topic: str = "/camera/color/image_raw"
    ros_depth_topic: str = "/camera/depth/image_rect_raw"
    text_query_topic: str = "/kurat/query"
    text_reply_topic: str = "/kurat/reply"
    status_topic: str = "/kurat/status"
    queue_size: int = 10
    store_depth: bool = False


@dataclass(slots=True)
class AppConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    topics: TopicConfig = field(default_factory=TopicConfig)
