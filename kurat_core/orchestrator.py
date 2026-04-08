from __future__ import annotations

import cv2
import logging
import threading
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from .interfaces import ChatService, IntentRouter, LatestFrameProvider
from .moondream_service import MoondreamService
from .power_manager import PowerAwareExecutionManager
from .types import IntentResult, OrchestratorResult, VisionResult
from .yolo_world_service import YoloWorldService


LOGGER = logging.getLogger(__name__)


class KuratOrchestrator:
    def __init__(
        self,
        frame_provider: Optional[LatestFrameProvider],
        intent_router: IntentRouter,
        chat: ChatService,
        yolo: YoloWorldService,
        moon: MoondreamService,
        enable_moondream_fallback_on_find: bool = True,
        max_frame_age_s: float = 1.0,
        skip_stale_frames: bool = True,
        moondream_frame_max_dim: int = 768,
        yolo_frame_max_dim: int = 960,
        execution_manager: PowerAwareExecutionManager | None = None,
    ):
        self.frame_provider = frame_provider
        self.router = intent_router
        self.chat = chat
        self.yolo = yolo
        self.moon = moon
        self.enable_moondream_fallback_on_find = enable_moondream_fallback_on_find
        self.max_frame_age_s = max_frame_age_s
        self.skip_stale_frames = skip_stale_frames
        self.moondream_frame_max_dim = moondream_frame_max_dim
        self.yolo_frame_max_dim = yolo_frame_max_dim
        # This lightweight request lock prevents overlapping prompt handling across threads.
        self._request_lock = threading.Lock()
        self.execution_manager = execution_manager

    def handle_text(self, user_text: str, history: str = "") -> OrchestratorResult:
        if self.execution_manager is not None:
            self.execution_manager.before_prompt("handle_text")

        with self._request_lock:
            intent = self._safe_classify(user_text)
            LOGGER.debug(
                "Intent classified: mode=%s targets=%s confidence=%.3f",
                intent.mode,
                intent.targets,
                intent.confidence,
            )

            if intent.mode == "chat":
                reply = self._safe_normal_chat(user_text, history=history)
                return OrchestratorResult(
                    user_text=user_text,
                    intent=intent,
                    vision_used=False,
                    vision_result=None,
                    reply_text=reply,
                )

            frame, frame_meta, frame_error = self._get_latest_frame_for_vision()
            if frame is None:
                LOGGER.warning("Vision query could not use a frame: %s", frame_error)
                payload = {
                    "method": "vision_unavailable",
                    "reason": frame_error,
                    "frame_meta": frame_meta,
                }
                return OrchestratorResult(
                    user_text=user_text,
                    intent=intent,
                    vision_used=False,
                    vision_result=VisionResult(method=payload["method"], payload=payload),
                    reply_text=self._build_no_frame_reply(frame_error),
                )

            vision_payload = self._run_vision(intent, user_text, frame)
            if frame_meta is not None:
                vision_payload["frame_meta"] = frame_meta
            vision_result = VisionResult(method=vision_payload["method"], payload=vision_payload)
            reply = self._safe_answer_with_vision(user_text, vision_payload, history=history)

            return OrchestratorResult(
                user_text=user_text,
                intent=intent,
                vision_used=True,
                vision_result=vision_result,
                reply_text=reply,
            )

    def handle_text_dict(self, user_text: str, history: str = "") -> Dict[str, Any]:
        result = self.handle_text(user_text, history=history)
        payload = asdict(result)
        payload["reply"] = payload.pop("reply_text")
        return payload

    def _run_vision(self, intent: IntentResult, user_text: str, frame) -> Dict[str, Any]:
        if intent.mode == "vision_scene":
            return self._do_scene(frame, user_text)
        if intent.mode == "vision_find":
            return self._do_find(frame, intent.targets, user_text)
        if intent.mode == "vision_attribute":
            target = self._pick_primary_target(intent.targets, fallback="object")
            return self._do_attribute(frame, target, user_text)

        analysis = self._safe_moondream_analyze(self._prepare_frame_for_moondream(frame), user_text)
        return {"method": "vision_fallback", "analysis": analysis}

    def _do_scene(self, frame, user_text: str) -> Dict[str, Any]:
        analysis = self._safe_moondream_analyze(self._prepare_frame_for_moondream(frame), user_text)
        return {"method": "moondream_scene", "analysis": analysis}

    def _do_find(self, frame, targets, user_text: str) -> Dict[str, Any]:
        clean_targets = [target for target in targets if target]
        request_type = self._find_request_type(user_text)
        yolo_res = self._safe_yolo_detect(self._prepare_frame_for_yolo(frame), clean_targets)

        if yolo_res is None:
            LOGGER.warning("YOLO failed during find request; attempting semantic fallback")
            return self._build_find_fallback_without_yolo(frame, clean_targets, user_text, request_type)

        if yolo_res["count"] > 0:
            result = {
                "method": "yolo_world_find",
                "request_type": request_type,
                "targets": clean_targets,
                "yolo": yolo_res,
            }
            if self._find_requires_semantic_explanation(request_type, user_text):
                analysis = self._safe_moondream_analyze(
                    self._prepare_frame_for_moondream(frame),
                    user_text,
                    self._pick_primary_target(clean_targets),
                )
                if analysis is not None:
                    self._patch_detector_confirmed_presence(
                        analysis,
                        self._pick_primary_target(clean_targets),
                    )
                    result["analysis"] = analysis
            return result

        if not self.enable_moondream_fallback_on_find:
            LOGGER.info("YOLO found no detections and semantic fallback is disabled")
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

        LOGGER.info("YOLO found no detections; using Moondream semantic fallback")
        analysis = self._safe_moondream_analyze(
            self._prepare_frame_for_moondream(frame),
            user_text,
            primary_target,
        )
        if analysis is None:
            result["note"] = "semantic fallback unavailable"
            return result
        return {**result, "analysis": analysis}

    def _do_attribute(self, frame, target: str, user_text: str) -> Dict[str, Any]:
        analysis = self._safe_moondream_analyze(self._prepare_frame_for_moondream(frame), user_text, target)
        return {
            "method": "moondream_attribute",
            "target": target,
            "analysis": analysis,
        }

    def _pick_primary_target(self, targets, fallback: str = "") -> str:
        for target in targets or []:
            target = (target or "").strip()
            if target:
                return target
        return fallback

    def _find_request_type(self, user_text: str) -> str:
        text = (user_text or "").lower()
        location_terms = (
            "where is",
            "where's",
            "wheres",
            "locate",
            "show me where",
            "position",
            "coordinates",
            "coordinate",
            "bounding box",
            "bbox",
            "location",
            "box",
        )
        if any(term in text for term in location_terms):
            return "coordinates"
        return "presence"

    def _find_requires_semantic_explanation(self, request_type: str, user_text: str) -> bool:
        if request_type in {"coordinates", "presence"}:
            text = (user_text or "").lower()
            semantic_terms = (
                "describe",
                "explain",
                "what is it doing",
                "what are they doing",
                "tell me about",
                "what is on",
                "what's on",
                "whats on",
                "read",
            )
            return any(term in text for term in semantic_terms)
        return True

    def _safe_classify(self, user_text: str) -> IntentResult:
        try:
            return self.router.classify(user_text)
        except Exception as exc:
            LOGGER.warning("Intent classification failed, using fallback chat intent: %s", exc)
            return IntentResult(mode="chat", targets=[], confidence=0.0)

    def _safe_normal_chat(self, user_text: str, history: str = "") -> str:
        try:
            return self.chat.normal_chat(user_text, history=history)
        except Exception as exc:
            LOGGER.warning("Chat generation failed, using plain fallback reply: %s", exc)
            return "I could not format a full response just now, but I am still running and can try again."

    def _safe_yolo_detect(self, frame, targets) -> Optional[Dict[str, Any]]:
        try:
            return self.yolo.detect(frame, targets)
        except Exception as exc:
            LOGGER.warning("YOLO failed, falling back if possible: %s", exc)
            return None

    def _safe_moondream_analyze(self, frame, user_text: str, target: str = "") -> Optional[Dict[str, Any]]:
        try:
            return self.moon.analyze(frame, user_text, target or None)
        except Exception as exc:
            LOGGER.warning("Moondream failed, continuing without semantic analysis: %s", exc)
            return None

    def _safe_answer_with_vision(self, user_text: str, vision_payload: Dict[str, Any], history: str = "") -> str:
        try:
            return self.chat.answer_with_vision(user_text, vision_result=vision_payload, history=history)
        except Exception as exc:
            LOGGER.warning("Vision answer formatting failed, using simple fallback text: %s", exc)
            return self._simple_vision_reply(user_text, vision_payload)

    def _build_find_fallback_without_yolo(self, frame, targets, user_text: str, request_type: str) -> Dict[str, Any]:
        result = {
            "method": "fallback_moondream",
            "request_type": request_type,
            "targets": targets,
            "yolo": {
                "requested_targets": targets,
                "query_vocab": [],
                "count": 0,
                "detections": [],
                "annotated_image": None,
                "stage": "error",
                "primary_pass": {"query_vocab": [], "count": 0},
                "fallback_pass": None,
            },
            "note": "yolo unavailable",
        }
        if not self.enable_moondream_fallback_on_find or request_type == "coordinates":
            if request_type == "coordinates":
                result["note"] = "coordinates require a YOLO detection box"
            return result

        analysis = self._safe_moondream_analyze(
            self._prepare_frame_for_moondream(frame),
            user_text,
            self._pick_primary_target(targets),
        )
        if analysis is not None:
            LOGGER.info("Using Moondream semantic fallback because YOLO was unavailable")
            result["analysis"] = analysis
        else:
            result["note"] = "yolo unavailable and semantic fallback failed"
        return result

    def _patch_detector_confirmed_presence(self, analysis: Dict[str, Any], target: str) -> None:
        if not analysis:
            return
        presence = analysis.get("presence")
        if not isinstance(presence, dict):
            presence = {}
            analysis["presence"] = presence
        presence["target"] = target
        presence["present"] = True
        presence["source"] = "yolo_confirmed"
        analysis["target"] = target

    def _simple_vision_reply(self, user_text: str, vision_payload: Dict[str, Any]) -> str:
        method = vision_payload.get("method", "")
        if method == "vision_unavailable":
            return self._build_no_frame_reply(vision_payload.get("reason"))

        if method == "yolo_world_find":
            yolo = vision_payload.get("yolo") or {}
            detections = yolo.get("detections") or []
            targets = vision_payload.get("targets") or []
            target_text = self._pick_primary_target(targets, fallback="object")
            if detections:
                top = detections[0]
                return (
                    f"I found {target_text}. "
                    f"Top box: {top.get('box')} with confidence {float(top.get('confidence', 0.0)):.2f}."
                )
            return f"I could not localize {target_text} in the current frame."

        analysis = vision_payload.get("analysis") or {}
        qa = analysis.get("qa") if isinstance(analysis, dict) else None
        if isinstance(qa, dict) and qa.get("answer"):
            return str(qa["answer"])

        scene = analysis.get("scene") if isinstance(analysis, dict) else None
        if isinstance(scene, dict) and scene.get("description"):
            return str(scene["description"])

        return "I processed the request, but I could not format a detailed reply."

    def _get_latest_frame_for_vision(self) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
        if self.frame_provider is None:
            return None, None, "no frame provider is configured"

        try:
            if hasattr(self.frame_provider, "get_latest_rgb_frame_with_meta"):
                meta = self.frame_provider.get_latest_rgb_frame_with_meta(
                    max_age_s=self.max_frame_age_s,
                    reject_stale=self.skip_stale_frames,
                )
                if meta is None:
                    age = self._get_frame_age_if_available()
                    if age is None:
                        return None, None, "no RGB frame has arrived yet"
                    return None, {"age_s": age, "is_stale": True}, "latest frame is stale"
                return meta.get("frame"), meta, None

            frame = self.frame_provider.get_latest_rgb_frame()
            if frame is None:
                return None, None, "no RGB frame has arrived yet"
            return frame, None, None
        except Exception as exc:
            LOGGER.warning("Failed to fetch latest frame: %s", exc)
            return None, None, str(exc)

    def _get_frame_age_if_available(self) -> Optional[float]:
        if hasattr(self.frame_provider, "get_latest_frame_age_seconds"):
            try:
                return self.frame_provider.get_latest_frame_age_seconds()
            except Exception:
                return None
        return None

    def _build_no_frame_reply(self, reason: Optional[str]) -> str:
        if reason == "latest frame is stale":
            return "I need a fresher camera frame before I can answer that vision request."
        if reason == "no RGB frame has arrived yet":
            return "I do not have a camera frame yet, so I cannot answer that vision request right now."
        if reason == "no frame provider is configured":
            return "Vision is not configured in this runtime, so I cannot answer that visual request."
        return "I could not access a usable camera frame for that vision request."

    def _prepare_frame_for_yolo(self, frame):
        return self._resize_frame_if_needed(frame, self.yolo_frame_max_dim)

    def _prepare_frame_for_moondream(self, frame):
        return self._resize_frame_if_needed(frame, self.moondream_frame_max_dim)

    def _resize_frame_if_needed(self, frame, max_dim: int):
        if frame is None or not max_dim or max_dim <= 0:
            return frame
        height, width = frame.shape[:2]
        largest_dim = max(height, width)
        if largest_dim <= max_dim:
            return frame
        scale = float(max_dim) / float(largest_dim)
        resized = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        LOGGER.debug(
            "Resized frame from %sx%s to %sx%s",
            width,
            height,
            resized.shape[1],
            resized.shape[0],
        )
        return resized
