from __future__ import annotations

import argparse
import logging

from kurat_core.config import AppConfig
from kurat_core.intent_router import MistralIntentRouter
from kurat_core.mistral_chat import MistralChat
from kurat_core.moondream_service import MoondreamService
from kurat_core.orchestrator import KuratOrchestrator
from kurat_core.power_manager import PowerAwareExecutionManager, PowerAwareSettings
from kurat_core.yolo_world_service import YoloWorldService
from kurat_io.frame_sources.image_file_provider import ImageFileFrameProvider


def build_orchestrator(image_path: str) -> KuratOrchestrator:
    cfg = AppConfig()
    language_model_name = cfg.models.select_language_model(prefer_small=cfg.runtime.prefer_smaller_models)
    # Use one shared power-aware manager so the laptop smoke test exercises the same serialized heavy-task path.
    execution_manager = PowerAwareExecutionManager(
        PowerAwareSettings(
            power_aware_mode=cfg.runtime.power_aware_mode,
            heavy_task_max_concurrency=cfg.runtime.heavy_task_max_concurrency,
            allow_concurrent_heavy_inference=cfg.runtime.allow_concurrent_heavy_inference,
            inference_cooldown_ms=cfg.runtime.inference_cooldown_ms,
            post_model_switch_delay_ms=cfg.runtime.post_model_switch_delay_ms,
            prompt_min_interval_ms=cfg.runtime.prompt_min_interval_ms,
            heavy_task_acquire_timeout_s=cfg.runtime.heavy_task_acquire_timeout_s,
            enable_telemetry=cfg.runtime.enable_telemetry,
        )
    )

    frame_provider = ImageFileFrameProvider(image_path)
    router = MistralIntentRouter(
        model=language_model_name,
        ollama_url=cfg.models.ollama_generate_url,
        timeout_s=cfg.models.intent_timeout_s,
        execution_manager=execution_manager,
    )
    chat = MistralChat(
        model=language_model_name,
        ollama_url=cfg.models.ollama_generate_url,
        timeout_s=cfg.models.chat_timeout_s,
        execution_manager=execution_manager,
    )
    yolo = YoloWorldService(
        model_path=cfg.models.yolo_model_path,
        device=cfg.models.yolo_device,
        imgsz=cfg.runtime.yolo_imgsz,
        conf=cfg.runtime.yolo_conf,
        iou=cfg.runtime.yolo_iou,
        max_det=cfg.runtime.yolo_max_det,
        debug_save_images=cfg.runtime.debug_save_images,
        debug_image_dir=cfg.runtime.debug_image_dir,
        execution_manager=execution_manager,
    )
    moon = MoondreamService(
        model=cfg.models.moondream_model_name,
        host=cfg.models.ollama_host,
        execution_manager=execution_manager,
    )

    return KuratOrchestrator(
        frame_provider=frame_provider,
        intent_router=router,
        chat=chat,
        yolo=yolo,
        moon=moon,
        enable_moondream_fallback_on_find=cfg.runtime.enable_moondream_fallback_on_find,
        max_frame_age_s=cfg.runtime.max_frame_age_s,
        skip_stale_frames=cfg.runtime.skip_stale_frames,
        moondream_frame_max_dim=cfg.runtime.moondream_frame_max_dim,
        yolo_frame_max_dim=cfg.runtime.yolo_frame_max_dim,
        execution_manager=execution_manager,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick laptop smoke test for the new Kurat architecture.")
    parser.add_argument(
        "--image",
        default="",
        help="Path to an image file used as the current frame.",
    )
    parser.add_argument(
        "--query",
        default="What do you see?",
        help="Text query to send through the orchestrator.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if not args.image:
        parser.error("Provide --image PATH. No sample image is bundled in this repository.")

    orchestrator = build_orchestrator(args.image)
    result = orchestrator.handle_text(args.query)

    print("\nReply:")
    print(result.reply_text)

    print("\nStructured result:")
    print(result)


if __name__ == "__main__":
    main()
