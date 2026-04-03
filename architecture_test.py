from __future__ import annotations

import argparse
import logging

from kurat_core.config import AppConfig
from kurat_core.intent_router import MistralIntentRouter
from kurat_core.mistral_chat import MistralChat
from kurat_core.moondream_service import MoondreamService
from kurat_core.orchestrator import KuratOrchestrator
from kurat_core.yolo_world_service import YoloWorldService
from kurat_io.frame_sources.image_file_provider import ImageFileFrameProvider


def build_orchestrator(image_path: str) -> KuratOrchestrator:
    cfg = AppConfig()

    frame_provider = ImageFileFrameProvider(image_path)
    router = MistralIntentRouter(
        model=cfg.models.language_model_name,
        ollama_url=cfg.models.ollama_generate_url,
        timeout_s=cfg.models.intent_timeout_s,
    )
    chat = MistralChat(
        model=cfg.models.language_model_name,
        ollama_url=cfg.models.ollama_generate_url,
        timeout_s=cfg.models.chat_timeout_s,
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
    )
    moon = MoondreamService(
        model=cfg.models.moondream_model_name,
        host=cfg.models.ollama_host,
    )

    return KuratOrchestrator(
        frame_provider=frame_provider,
        intent_router=router,
        chat=chat,
        yolo=yolo,
        moon=moon,
        enable_moondream_fallback_on_find=cfg.runtime.enable_moondream_fallback_on_find,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick laptop smoke test for the new Kurat architecture.")
    parser.add_argument(
        "--image",
        default="test_image.png",
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

    orchestrator = build_orchestrator(args.image)
    result = orchestrator.handle_text(args.query)

    print("\nReply:")
    print(result.reply_text)

    print("\nStructured result:")
    print(result)


if __name__ == "__main__":
    main()
