# orchestrator.py
"""
Deterministic Orchestrator (Brain 3)
-----------------------------------
Connects:
- Mistral Intent Router (mode + targets)
- YOLO-World (localization)
- Moondream (semantic vision)
- Mistral Chat (final phrasing)
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from PIL import Image

from camera import LaptopCameraFrameProvider
from mistral_chat import MistralChat
from mistral_intent import MistralIntentRouter
from moondream_utils import MoondreamService
from stt import WhisperSTTService
from tts import WindowsTTSService
from yolo_world_utils import YoloWorldService


class FrameProvider(Protocol):
    def get_frame(self) -> Image.Image: ...


@dataclass
class OrchestratorConfig:
    keep_temp_frames: bool = False
    enable_moondream_fallback_on_find: bool = True


class Orchestrator:
    def __init__(
        self,
        frame_provider: FrameProvider,
        intent_router: MistralIntentRouter,
        chat: MistralChat,
        yolo: YoloWorldService,
        moon: MoondreamService,
        config: Optional[OrchestratorConfig] = None,
    ):
        self.camera = frame_provider
        self.router = intent_router
        self.chat = chat
        self.yolo = yolo
        self.moon = moon
        self.cfg = config or OrchestratorConfig()

    def handle_user_text(self, user_text: str, history: str = "") -> Dict[str, Any]:
        intent = self.router.classify(user_text)
        mode = intent.get("mode", "chat")

        if mode == "chat":
            reply = self.chat.normal_chat(user_text, history=history)
            return {"intent": intent, "vision_used": False, "vision_result": None, "reply": reply}

        frame = self.camera.get_frame()
        img_path = self._save_temp_jpg(frame)

        try:
            vision_result = self._run_vision(mode, img_path, intent, user_text)
            reply = self.chat.answer_with_vision(user_text, vision_result=vision_result, history=history)
            return {"intent": intent, "vision_used": True, "vision_result": vision_result, "reply": reply}
        finally:
            if not self.cfg.keep_temp_frames:
                self._safe_remove(img_path)

    def _run_vision(self, mode: str, img_path: str, intent: Dict[str, Any], user_text: str) -> Dict[str, Any]:
        if mode == "vision_scene":
            return self._do_scene(img_path, user_text)
        if mode == "vision_find":
            return self._do_find(img_path, intent.get("targets", []), user_text)
        if mode == "vision_attribute":
            target = self._pick_primary_target(intent.get("targets", []), fallback="object")
            return self._do_attribute(img_path, target, user_text)

        return {
            "method": "vision_fallback",
            "analysis": self.moon.analyze(img_path, user_text),
        }

    def _do_scene(self, img_path: str, user_text: str) -> Dict[str, Any]:
        analysis = self.moon.analyze(img_path, user_text)
        return {
            "method": "moondream_scene",
            "analysis": analysis,
        }

    def _do_find(self, img_path: str, targets: List[str], user_text: str) -> Dict[str, Any]:
        clean_targets = [target for target in targets if target]
        request_type = self._find_request_type(user_text)
        yolo_res = self.yolo.detect(img_path, clean_targets, save_annotated=True)

        if yolo_res["count"] > 0:
            return {
                "method": "yolo_world_find",
                "request_type": request_type,
                "targets": clean_targets,
                "yolo": yolo_res,
                "analysis": self.moon.analyze(img_path, user_text, self._pick_primary_target(clean_targets)),
            }

        if not self.cfg.enable_moondream_fallback_on_find:
            return {
                "method": "yolo_world_find",
                "request_type": request_type,
                "targets": clean_targets,
                "yolo": yolo_res,
                "note": "no detections",
            }

        primary_target = self._pick_primary_target(clean_targets)
        result = {
            "method": "fallback_moondream",
            "request_type": request_type,
            "targets": clean_targets,
            "yolo": yolo_res,
        }

        if request_type == "coordinates":
            result["note"] = "coordinates require a YOLO detection box"
            return result

        return {
            **result,
            "analysis": self.moon.analyze(img_path, user_text, primary_target),
        }

    def _do_attribute(self, img_path: str, target: str, user_text: str) -> Dict[str, Any]:
        return {
            "method": "moondream_attribute",
            "target": target,
            "analysis": self.moon.analyze(img_path, user_text, target),
        }

    def _pick_primary_target(self, targets: List[str], fallback: str = "") -> str:
        for target in targets or []:
            target = (target or "").strip()
            if target:
                return target
        return fallback

    def _find_request_type(self, user_text: str) -> str:
        text = (user_text or "").lower()
        if any(term in text for term in ("coordinate", "coordinates", "bbox", "bounding box", "box", "location", "position")):
            return "coordinates"
        return "presence"

    def _save_temp_jpg(self, pil_img: Image.Image) -> str:
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        pil_img.save(path, "JPEG", quality=95)
        return path

    def _safe_remove(self, path: str):
        try:
            os.remove(path)
        except Exception:
            pass


class ImageFileFrameProvider:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def get_frame(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")


def choose_input_mode(preselected_mode: str = "") -> str:
    mode = (preselected_mode or "").strip().lower()
    if mode in ("type", "talk"):
        return mode

    print("\nChoose how you want to talk to Kurat:")
    print("1. Type")
    print("2. Talk")

    while True:
        choice = input("\nSelect 1 or 2: ").strip().lower()
        if choice in ("1", "type", "typing", "t"):
            return "type"
        if choice in ("2", "talk", "talking", "voice", "v"):
            return "talk"
        print("Please choose 1 for typing or 2 for talking.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kurat multimodal assistant")
    parser.add_argument("--mode", choices=("type", "talk"), default="", help="Startup mode")
    parser.add_argument("--voice", action="store_true", help="Use microphone input and speak replies")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--stt-model", default="base", help="Whisper model size")
    parser.add_argument("--tts-voice", default="", help="Optional Windows TTS voice name")
    parser.add_argument("--tts-rate", type=int, default=0, help="Windows TTS speech rate (-10 to 10)")
    args = parser.parse_args()

    camera = LaptopCameraFrameProvider(camera_index=args.camera_index)

    router = MistralIntentRouter(model="mistral")
    chat = MistralChat(model="mistral")

    yolo = YoloWorldService(model_path="yolov8s-world.pt", device="cpu")
    moon = MoondreamService(model="moondream")

    orch = Orchestrator(camera, router, chat, yolo, moon)

    history = ""
    selected_mode = "talk" if args.voice else choose_input_mode(args.mode)
    try:
        if selected_mode == "talk":
            stt = WhisperSTTService(model_name=args.stt_model)
            tts = WindowsTTSService(voice_name=args.tts_voice, rate=args.tts_rate)

            while True:
                print("\nTalk mode ready. Hold S to record your speech. Say 'exit' or 'quit' to stop.")
                audio = stt.record_while_key("s")
                user = stt.transcribe(audio).strip()

                if not user:
                    print("\nWhisper heard: (no speech detected)")
                    continue

                print(f"\nWhisper heard: {user}")
                print(f"You: {user}")
                if user.lower().strip() in ("exit", "quit", "stop"):
                    break

                out = orch.handle_user_text(user, history=history)
                print("\nKurat:", out["reply"])
                tts.speak(out["reply"])

                history += f"\nUser: {user}\nAssistant: {out['reply']}\n"
                history = history[-4000:]
        else:
            print("\nType mode ready. Type your message and press ENTER. Type 'exit' or 'quit' to stop.")
            while True:
                user = input("\nYou: ").strip()
                if user.lower() in ("exit", "quit"):
                    break

                out = orch.handle_user_text(user, history=history)
                print("\nKurat:", out["reply"])

                history += f"\nUser: {user}\nAssistant: {out['reply']}\n"
                history = history[-4000:]
    finally:
        if hasattr(camera, "release"):
            camera.release()
