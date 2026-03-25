"""
Moondream Semantic Vision Service
--------------------------------
Purpose:
- Semantic understanding ONLY
- No localization
- No control flow
- No orchestration logic

Used by:
- Orchestrator (Brain 3)

Provides:
- list_objects()
- is_present()
- describe_scene()
- answer_question()
- analyze()
"""

from typing import Dict, List, Optional
from pathlib import Path
import base64
import io
import os
import re

import ollama
from PIL import Image


class MoondreamService:
    def __init__(
        self,
        model: str = "moondream",
        host: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.temperature = temperature
        self.client = ollama.Client(host=self.host)

        self._verify_model()

    def list_objects(self, image_path: str) -> Dict[str, object]:
        """
        List all visible physical objects (semantic, not spatial).
        """
        prompt = (
            "Identify the visible physical objects in this image. "
            "Return only a comma-separated list of simple object names. "
            "Prefer broad everyday categories. "
            "Do not include counts, colors, explanations, or full sentences."
        )

        raw = self._query(image_path, prompt).lower()
        objects = self._parse_object_list(raw)

        return {
            "objects": objects,
            "count": len(objects),
            "raw": raw,
        }

    def is_present(self, image_path: str, target: str) -> Dict[str, object]:
        """
        Binary presence check (YES / NO).
        """
        target = self._normalize_target(target)
        prompt = (
            "Answer strictly with 'yes' or 'no'.\n"
            f"Question: Is there a visible {target!r} in the image?"
        )

        raw = self._query(image_path, prompt).lower()
        present = self._parse_yes_no(raw)

        return {
            "target": target,
            "present": present,
            "raw": raw,
        }

    def describe_scene(self, image_path: str) -> Dict[str, str]:
        """
        Short natural language scene description.
        """
        prompt = (
            "Describe the scene in 1-2 short sentences. "
            "Mention the most important visible objects, people, actions, text, and setting if present. "
            "Do not speculate about unclear details."
        )

        description = self._query(image_path, prompt)
        return {"description": description}

    def answer_question(self, image_path: str, question: str) -> Dict[str, str]:
        """
        Answer a user-specified visual question.
        """
        prompt = (
            f"{question}\n"
            "Answer concisely using only what is visible in the image. "
            "If the image is unclear, say so instead of guessing."
        )

        answer = self._query(image_path, prompt)
        return {
            "question": question,
            "answer": answer,
        }

    def analyze(self, image_path: str, question: str, target: Optional[str] = None) -> Dict[str, object]:
        """
        General-purpose semantic analysis for arbitrary visual questions.
        """
        result: Dict[str, object] = {
            "question": question,
            "scene": self.describe_scene(image_path),
            "objects": self.list_objects(image_path),
            "qa": self.answer_question(image_path, question),
        }

        normalized_target = self._normalize_target(target or "")
        if normalized_target:
            result["target"] = normalized_target
            result["presence"] = self.is_present(image_path, normalized_target)

        return result

    def _verify_model(self):
        try:
            models = self.client.list()
            model_list = getattr(models, "models", None) or models.get("models", [])
            names = [
                (getattr(m, "model", None) or m.get("model", "")).lower()
                for m in model_list
            ]
            if not any("moondream" in name for name in names):
                raise RuntimeError("Moondream model not found. Run: ollama pull moondream")
        except Exception as exc:
            raise RuntimeError(f"Ollama unavailable: {exc}")

    def _query(self, image_path: str, prompt: str) -> str:
        b64 = self._encode_image(image_path)

        resp = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }
            ],
            options={"temperature": self.temperature},
        )

        if isinstance(resp, dict):
            return (resp.get("message", {}).get("content", "")).strip()
        if hasattr(resp, "message"):
            return (resp.message.content or "").strip()
        return ""

    def _encode_image(self, image_path: str) -> str:
        if not Path(image_path).exists():
            raise FileNotFoundError(image_path)

        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _parse_object_list(self, raw: str) -> List[str]:
        normalized = raw.replace("\n", ",")
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        cleaned: List[str] = []

        for part in parts:
            item = re.sub(r"[^a-z0-9\s]", "", part)
            item = re.sub(r"^(the|a|an)\s+", "", item)
            item = re.sub(r"\b(and|or)\b", "", item).strip()
            if item.endswith("s") and not item.endswith("ss"):
                item = item[:-1]
            item = item.strip()
            if item in {"none", "nothing", "no object", "no objects", "unknown"}:
                continue
            if 2 <= len(item) <= 30:
                cleaned.append(item)

        seen = set()
        result = []
        for item in cleaned:
            if item not in seen:
                seen.add(item)
                result.append(item)

        return result[:40]

    def _parse_yes_no(self, raw: str) -> bool:
        raw = raw.strip().lower()

        if raw.startswith("yes") or raw.startswith("y"):
            return True
        if raw.startswith("no") or raw.startswith("n"):
            return False

        if "not" in raw or "isn't" in raw or "isnt" in raw:
            return False
        if "present" in raw or "visible" in raw or "there is" in raw:
            return True

        return False

    def _normalize_target(self, target: str) -> str:
        target = re.sub(r"[^a-z0-9\s]", " ", (target or "").lower())
        target = re.sub(r"\s+", " ", target).strip()
        target = re.sub(r"^(the|a|an|my|your|this|that)\s+", "", target)
        return target[:50]
