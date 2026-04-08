from __future__ import annotations

import json
import logging

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from kurat_core.config import AppConfig
from kurat_core.intent_router import MistralIntentRouter
from kurat_core.mistral_chat import MistralChat
from kurat_core.moondream_service import MoondreamService
from kurat_core.orchestrator import KuratOrchestrator
from kurat_core.power_manager import PowerAwareExecutionManager, PowerAwareSettings
from kurat_core.yolo_world_service import YoloWorldService
from kurat_io.frame_sources.ros_realsense_provider import ROSRealSenseFrameProvider


class KuratOrchestratorNode(Node):
    def __init__(self, config: AppConfig | None = None):
        super().__init__("kurat_orchestrator")
        self.config = config or AppConfig()
        self._load_parameters_into_config()
        self.history = ""
        self._configure_logging()
        language_model_name = self.config.models.select_language_model(
            prefer_small=self.config.runtime.prefer_smaller_models,
        )
        # Share one execution manager across all heavy inference components to prevent power spikes from overlap.
        self.execution_manager = PowerAwareExecutionManager(
            PowerAwareSettings(
                power_aware_mode=self.config.runtime.power_aware_mode,
                heavy_task_max_concurrency=self.config.runtime.heavy_task_max_concurrency,
                allow_concurrent_heavy_inference=self.config.runtime.allow_concurrent_heavy_inference,
                inference_cooldown_ms=self.config.runtime.inference_cooldown_ms,
                post_model_switch_delay_ms=self.config.runtime.post_model_switch_delay_ms,
                prompt_min_interval_ms=self.config.runtime.prompt_min_interval_ms,
                heavy_task_acquire_timeout_s=self.config.runtime.heavy_task_acquire_timeout_s,
                enable_telemetry=self.config.runtime.enable_telemetry,
            )
        )

        self.frame_provider = ROSRealSenseFrameProvider(
            node=self,
            color_topic=self.config.topics.ros_color_topic,
            depth_topic=self.config.topics.ros_depth_topic,
            queue_size=self.config.topics.queue_size,
            store_depth=self.config.topics.store_depth,
            max_frame_age_s=self.config.runtime.max_frame_age_s,
            skip_stale_frames=self.config.runtime.skip_stale_frames,
        )
        self.intent_router = MistralIntentRouter(
            model=language_model_name,
            ollama_url=self.config.models.ollama_generate_url,
            timeout_s=self.config.models.intent_timeout_s,
            execution_manager=self.execution_manager,
        )
        self.chat = MistralChat(
            model=language_model_name,
            ollama_url=self.config.models.ollama_generate_url,
            timeout_s=self.config.models.chat_timeout_s,
            execution_manager=self.execution_manager,
        )
        self.yolo = YoloWorldService(
            model_path=self.config.models.yolo_model_path,
            device=self.config.models.yolo_device,
            imgsz=self.config.runtime.yolo_imgsz,
            conf=self.config.runtime.yolo_conf,
            iou=self.config.runtime.yolo_iou,
            max_det=self.config.runtime.yolo_max_det,
            debug_save_images=self.config.runtime.debug_save_images,
            debug_image_dir=self.config.runtime.debug_image_dir,
            execution_manager=self.execution_manager,
        )
        self.moondream = MoondreamService(
            model=self.config.models.moondream_model_name,
            host=self.config.models.ollama_host,
            execution_manager=self.execution_manager,
        )
        self.orchestrator = KuratOrchestrator(
            frame_provider=self.frame_provider,
            intent_router=self.intent_router,
            chat=self.chat,
            yolo=self.yolo,
            moon=self.moondream,
            enable_moondream_fallback_on_find=self.config.runtime.enable_moondream_fallback_on_find,
            max_frame_age_s=self.config.runtime.max_frame_age_s,
            skip_stale_frames=self.config.runtime.skip_stale_frames,
            moondream_frame_max_dim=self.config.runtime.moondream_frame_max_dim,
            yolo_frame_max_dim=self.config.runtime.yolo_frame_max_dim,
            execution_manager=self.execution_manager,
        )

        self.query_subscription = self.create_subscription(
            String,
            self.config.topics.text_query_topic,
            self._on_query,
            self.config.topics.queue_size,
        )
        self.reply_publisher = self.create_publisher(
            String,
            self.config.topics.text_reply_topic,
            self.config.topics.queue_size,
        )
        self.status_publisher = self.create_publisher(
            String,
            self.config.topics.status_topic,
            self.config.topics.queue_size,
        )

        self.get_logger().info(
            "Models: language=%s moondream=%s yolo=%s",
            language_model_name,
            self.config.models.moondream_model_name,
            self.config.models.yolo_model_path,
        )
        self.get_logger().info(
            "Ollama host=%s YOLO device=%s log_level=%s power_aware=%s prefer_small=%s",
            self.config.models.ollama_host,
            self.config.models.yolo_device,
            self.config.runtime.log_level,
            self.config.runtime.power_aware_mode,
            self.config.runtime.prefer_smaller_models,
        )
        self.get_logger().info(
            "Topics: color=%s depth=%s query=%s reply=%s status=%s",
            self.config.topics.ros_color_topic,
            self.config.topics.ros_depth_topic,
            self.config.topics.text_query_topic,
            self.config.topics.text_reply_topic,
            self.config.topics.status_topic,
        )
        self.get_logger().info(
            "Runtime: store_depth=%s skip_stale_frames=%s max_frame_age_s=%.2f cooldown_ms=%s prompt_interval_ms=%s",
            self.config.topics.store_depth,
            self.config.runtime.skip_stale_frames,
            self.config.runtime.max_frame_age_s,
            self.config.runtime.inference_cooldown_ms,
            self.config.runtime.prompt_min_interval_ms,
        )

    def _on_query(self, msg: String) -> None:
        user_text = (msg.data or "").strip()
        if not user_text:
            return

        self.get_logger().info("Received query: %s", user_text)
        try:
            result = self.orchestrator.handle_text(user_text, history=self.history)
            reply_text = result.reply_text
            self.history += f"\nUser: {user_text}\nAssistant: {reply_text}\n"
            self.history = self.history[-4000:]
            self._publish_status(user_text, result)
        except Exception as exc:
            self.get_logger().warning("Failed to handle query: %s", exc)
            reply_text = f"Error: {exc}"
            self._publish_status_error(user_text, str(exc))

        reply_msg = String()
        reply_msg.data = reply_text
        self.reply_publisher.publish(reply_msg)
        self.get_logger().info("Published reply")

    def _publish_status(self, user_text: str, result) -> None:
        meta = None
        pipeline = "chat"
        if result.vision_result is not None:
            payload = result.vision_result.payload or {}
            meta = payload.get("frame_meta")
            pipeline = self._classify_pipeline(payload)

        status_payload = {
            "last_query": user_text,
            "pipeline": pipeline,
            "vision_used": result.vision_used,
            "frame_timestamp": None if meta is None else str(meta.get("timestamp")),
            "frame_age_s": None if meta is None else meta.get("age_s"),
        }
        msg = String()
        msg.data = json.dumps(status_payload)
        self.status_publisher.publish(msg)
        self.get_logger().debug("Status published: %s", msg.data)

    def _publish_status_error(self, user_text: str, error_text: str) -> None:
        status_payload = {
            "last_query": user_text,
            "pipeline": "error",
            "error": error_text,
        }
        msg = String()
        msg.data = json.dumps(status_payload)
        self.status_publisher.publish(msg)

    def _classify_pipeline(self, payload) -> str:
        method = payload.get("method", "")
        if method == "yolo_world_find":
            return "yolo+moondream" if payload.get("analysis") else "yolo_only"
        if method in {"moondream_scene", "moondream_attribute", "fallback_moondream"}:
            return "moondream"
        if method == "vision_unavailable":
            return "no_frame"
        return method or "unknown"

    def _load_parameters_into_config(self) -> None:
        self.declare_parameter("color_topic", self.config.topics.ros_color_topic)
        self.declare_parameter("depth_topic", self.config.topics.ros_depth_topic)
        self.declare_parameter("query_topic", self.config.topics.text_query_topic)
        self.declare_parameter("reply_topic", self.config.topics.text_reply_topic)
        self.declare_parameter("status_topic", self.config.topics.status_topic)
        self.declare_parameter("enable_depth", self.config.topics.store_depth)
        self.declare_parameter("log_level", self.config.runtime.log_level)
        self.declare_parameter("stale_frame_threshold", self.config.runtime.max_frame_age_s)
        self.declare_parameter("ollama_host", self.config.models.ollama_host)
        self.declare_parameter("power_aware_mode", self.config.runtime.power_aware_mode)
        self.declare_parameter("heavy_task_max_concurrency", self.config.runtime.heavy_task_max_concurrency)
        self.declare_parameter("allow_concurrent_heavy_inference", self.config.runtime.allow_concurrent_heavy_inference)
        self.declare_parameter("inference_cooldown_ms", self.config.runtime.inference_cooldown_ms)
        self.declare_parameter("prompt_min_interval_ms", self.config.runtime.prompt_min_interval_ms)
        self.declare_parameter("post_model_switch_delay_ms", self.config.runtime.post_model_switch_delay_ms)
        self.declare_parameter("enable_telemetry", self.config.runtime.enable_telemetry)
        self.declare_parameter("prefer_smaller_models", self.config.runtime.prefer_smaller_models)

        self.config.topics.ros_color_topic = str(self.get_parameter("color_topic").value)
        self.config.topics.ros_depth_topic = str(self.get_parameter("depth_topic").value)
        self.config.topics.text_query_topic = str(self.get_parameter("query_topic").value)
        self.config.topics.text_reply_topic = str(self.get_parameter("reply_topic").value)
        self.config.topics.status_topic = str(self.get_parameter("status_topic").value)
        self.config.topics.store_depth = bool(self.get_parameter("enable_depth").value)
        self.config.runtime.log_level = str(self.get_parameter("log_level").value).upper()
        self.config.runtime.max_frame_age_s = float(self.get_parameter("stale_frame_threshold").value)
        self.config.models.ollama_host = str(self.get_parameter("ollama_host").value)
        self.config.models.ollama_generate_url = self.config.models.ollama_host.rstrip("/") + "/api/generate"
        self.config.runtime.power_aware_mode = bool(self.get_parameter("power_aware_mode").value)
        self.config.runtime.heavy_task_max_concurrency = int(self.get_parameter("heavy_task_max_concurrency").value)
        self.config.runtime.allow_concurrent_heavy_inference = bool(self.get_parameter("allow_concurrent_heavy_inference").value)
        self.config.runtime.inference_cooldown_ms = int(self.get_parameter("inference_cooldown_ms").value)
        self.config.runtime.prompt_min_interval_ms = int(self.get_parameter("prompt_min_interval_ms").value)
        self.config.runtime.post_model_switch_delay_ms = int(self.get_parameter("post_model_switch_delay_ms").value)
        self.config.runtime.enable_telemetry = bool(self.get_parameter("enable_telemetry").value)
        self.config.runtime.prefer_smaller_models = bool(self.get_parameter("prefer_smaller_models").value)

    def _configure_logging(self) -> None:
        level_name = (self.config.runtime.log_level or "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(level=level)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KuratOrchestratorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
