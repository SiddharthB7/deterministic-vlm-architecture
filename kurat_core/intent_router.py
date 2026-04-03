from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

import requests

from .types import IntentResult


LOGGER = logging.getLogger(__name__)

ATTRIBUTE_WORDS = {
    "color", "colour", "wearing", "holding", "size", "shape", "brand",
    "label", "text", "word", "words", "number", "numbers", "screen",
    "name", "material", "pattern", "condition", "expression", "pose",
    "position", "type", "kind", "style",
}

CHAT_PATTERNS = (
    "what is your name",
    "what's your name",
    "whats your name",
    "who are you",
    "hello",
    "hi",
    "hey",
    "how are you",
    "thank you",
    "thanks",
)

ASSISTANT_IDENTITY_PATTERNS = (
    "what is your name",
    "what's your name",
    "whats your name",
    "who are you",
    "tell me your name",
    "say your name",
    "i want your name",
    "i wanted your name",
    "i just want your name",
    "i just wanted your name",
)

SCENE_PATTERNS = (
    "what do you see",
    "what's in front of you",
    "whats in front of you",
    "describe the scene",
    "describe this",
    "describe what you see",
    "what is happening",
    "what's happening",
    "whats happening",
    "tell me about this image",
)

FIND_PREFIXES = (
    "where is", "where's", "wheres",
    "find", "locate", "look for",
    "do you see", "can you see", "is there", "are there",
)

VISION_HINT_WORDS = (
    "this", "that", "here", "there", "see", "look", "camera", "image",
    "photo", "picture", "screen", "scene", "room", "frame", "showing",
)


class MistralIntentRouter:
    def __init__(
        self,
        model: str = "mistral",
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
        timeout_s: int = 120,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout_s = timeout_s

    def classify(self, user_text: str) -> IntentResult:
        user_text = (user_text or "").strip()
        if not user_text:
            return IntentResult(mode="chat", targets=[], confidence=0.0)

        hint = self._vision_hint(user_text)
        heuristic = self._heuristic_classify(user_text, hint)
        prompt = self._build_prompt(user_text, hint)

        try:
            raw = self._ollama_generate(prompt)
            data = self._safe_json(raw)
        except Exception as exc:
            LOGGER.warning("Intent routing fell back to heuristics: %s", exc)
            data = {}

        normalized = self._normalize(data, user_text, hint)
        return self._merge_with_heuristic(normalized, heuristic)

    def _build_prompt(self, user_text: str, hint: bool) -> str:
        return f"""You are an intent router for a robot assistant.
Return ONLY valid JSON. No extra text. No markdown.

Decide if the user's request needs vision.

Modes:
- chat: no vision needed.
- vision_scene: user wants a general description, summary, reading, or explanation of what is visible.
- vision_find: user wants to locate, confirm, count, or check presence of an object.
- vision_attribute: user asks about an object's attributes, text, relationship, action, or any specific detail in the image.

JSON schema:
{{
  "mode": "chat|vision_scene|vision_find|vision_attribute",
  "targets": ["object1"],
  "confidence": 0.0-1.0
}}

Rules:
- chat: targets must be []
- vision_scene: targets should usually be []
- vision_find: targets should be one or more short object nouns
- vision_attribute: include the main object if obvious, otherwise []
- If the user refers to the current image/camera/scene, prefer a vision mode
- If uncertain, choose chat only when vision is not actually needed

Examples:
User: "Where is my phone?" -> {{ "mode":"vision_find","targets":["phone"],"confidence":0.9 }}
User: "What color is my shirt?" -> {{ "mode":"vision_attribute","targets":["shirt"],"confidence":0.9 }}
User: "Read the label on this bottle." -> {{ "mode":"vision_attribute","targets":["bottle"],"confidence":0.9 }}
User: "What do you see?" -> {{ "mode":"vision_scene","targets":[],"confidence":0.9 }}

User text: {user_text}
vision_hint: {str(hint).lower()}

JSON:"""

    def _ollama_generate(self, prompt: str) -> str:
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return (response.json().get("response") or "").strip()

    def _safe_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return {}

    def _normalize(self, data: Dict[str, Any], user_text: str, hint: bool) -> IntentResult:
        mode = (data.get("mode") or "chat").strip()
        if mode not in ("chat", "vision_scene", "vision_find", "vision_attribute"):
            mode = "chat"

        targets = data.get("targets") or []
        if not isinstance(targets, list):
            targets = []

        clean_targets: List[str] = []
        for target in targets:
            normalized = self._clean_target(str(target))
            if normalized:
                clean_targets.append(normalized)

        if mode == "chat":
            clean_targets = []
        elif mode == "vision_scene":
            clean_targets = clean_targets[:2]
        elif mode == "vision_find":
            clean_targets = clean_targets[:5]
        elif mode == "vision_attribute":
            clean_targets = clean_targets[:1]

        confidence = data.get("confidence", 0.6)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))

        if hint and mode == "chat" and confidence < 0.7:
            mode = "vision_scene"
            confidence = 0.7

        return IntentResult(mode=mode, targets=clean_targets, confidence=confidence)

    def _merge_with_heuristic(self, model_result: IntentResult, heuristic: IntentResult) -> IntentResult:
        if heuristic.confidence > model_result.confidence:
            return heuristic

        if heuristic.mode != "chat" and model_result.mode == "chat":
            return heuristic

        if not model_result.targets and heuristic.targets:
            model_result.targets = heuristic.targets

        return model_result

    def _heuristic_classify(self, user_text: str, hint: bool) -> IntentResult:
        text = user_text.lower().strip()
        cleaned = re.sub(r"\s+", " ", text)

        if self._is_assistant_identity_question(cleaned):
            return IntentResult(mode="chat", targets=[], confidence=0.995)

        if cleaned in CHAT_PATTERNS or cleaned.startswith(("hello ", "hi ", "hey ")):
            return IntentResult(mode="chat", targets=[], confidence=0.99)

        if any(pattern in cleaned for pattern in SCENE_PATTERNS):
            return IntentResult(mode="vision_scene", targets=[], confidence=0.95)

        if self._looks_like_attribute_question(cleaned):
            target = self._extract_target(cleaned, default="object")
            return IntentResult(
                mode="vision_attribute",
                targets=[target] if target else [],
                confidence=0.9,
            )

        if any(cleaned.startswith(prefix) for prefix in FIND_PREFIXES):
            target = self._extract_target(cleaned)
            return IntentResult(mode="vision_find", targets=[target] if target else [], confidence=0.88)

        if hint and self._contains_visual_question(cleaned):
            target = self._extract_target(cleaned)
            mode = "vision_attribute" if target else "vision_scene"
            return IntentResult(mode=mode, targets=[target] if target else [], confidence=0.8)

        return IntentResult(mode="chat", targets=[], confidence=0.4)

    def _looks_like_attribute_question(self, text: str) -> bool:
        if self._is_assistant_identity_question(text):
            return False

        if any(word in text for word in ATTRIBUTE_WORDS):
            return True
        return any(
            phrase in text
            for phrase in (
                "what is on",
                "what's on",
                "whats on",
                "what does",
                "what do the",
                "who is",
                "which",
                "read",
                "describe the",
                "tell me about the",
            )
        ) and self._vision_hint(text)

    def _contains_visual_question(self, text: str) -> bool:
        return "?" in text or any(word in text for word in ("what", "which", "who", "where", "is there", "are there"))

    def _extract_target(self, text: str, default: str = "") -> str:
        patterns = [
            r"(?:where is|where's|wheres|find|locate|look for)\s+(?:my|the|a|an|this|that)?\s*([a-z0-9][a-z0-9\s-]{0,40})",
            r"(?:coordinates of|coordinate of|location of|position of)\s+(?:my|the|a|an|this|that)?\s*([a-z0-9][a-z0-9\s-]{0,40})",
            r"(?:is there|are there|do you see|can you see)\s+(?:a|an|the|any|my|this|that)?\s*([a-z0-9][a-z0-9\s-]{0,40})",
            r"(?:what color is|what colour is|what is the color of|what is the colour of)\s+(?:my|the|a|an|this|that)?\s*([a-z0-9][a-z0-9\s-]{0,40})",
            r"(?:read|describe|tell me about|what is on|what's on|whats on)\s+(?:the|a|an|my|this|that)?\s*([a-z0-9][a-z0-9\s-]{0,40})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                target = self._clean_target(match.group(1))
                if target:
                    return target

        if default:
            return self._clean_target(default)
        return ""

    def _is_assistant_identity_question(self, text: str) -> bool:
        text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        text = re.sub(r"\s+", " ", text).strip()
        return any(pattern in text for pattern in ASSISTANT_IDENTITY_PATTERNS)

    def _clean_target(self, target: str) -> str:
        target = re.sub(r"[^a-z0-9\s-]", " ", (target or "").lower())
        target = re.sub(r"\b(in|on|at|near|with|from|to|of|for|please)\b.*$", "", target).strip()
        target = re.sub(r"^(the|a|an|my|your|this|that|these|those)\s+", "", target)
        target = re.sub(r"\b(coordinates|coordinate|location|position)\b", "", target)
        target = re.sub(r"\s+", " ", target).strip(" -")
        return target[:50]

    def _vision_hint(self, user_text: str) -> bool:
        text = user_text.lower()
        return any(word in text for word in VISION_HINT_WORDS)
