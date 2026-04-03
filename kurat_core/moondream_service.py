from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import Dict, List, Optional, Union

import numpy as np
import ollama
from PIL import Image


ImageInput = Union[Image.Image, np.ndarray]
LOGGER = logging.getLogger(__name__)


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

    def list_objects(self, image: ImageInput) -> Dict[str, object]:
        prompt = (
            "Identify the visible physical objects in this image. "
            "Return only a comma-separated list of simple object names. "
            "Prefer broad everyday categories. "
            "Do not include counts, colors, explanations, or full sentences."
        )
        raw = self._query(image, prompt).lower()
        objects = self._parse_object_list(raw)
        return {"objects": objects, "count": len(objects), "raw": raw}

    def is_present(self, image: ImageInput, target: str) -> Dict[str, object]:
        target = self._normalize_target(target)
        prompt = (
            "Answer strictly with 'yes' or 'no'.\n"
            f"Question: Is there a visible {target!r} in the image?"
        )
        raw = self._query(image, prompt).lower()
        present = self._parse_yes_no(raw)
        return {"target": target, "present": present, "raw": raw}

    def describe_scene(self, image: ImageInput) -> Dict[str, str]:
        prompt = (
            "Describe the scene in 1-2 short sentences. "
            "Mention the most important visible objects, people, actions, text, and setting if present. "
            "Do not speculate about unclear details."
        )
        description = self._query(image, prompt)
        return {"description": description}

    def answer_question(self, image: ImageInput, question: str) -> Dict[str, str]:
        prompt = (
            f"{question}\n"
            "Answer concisely using only what is visible in the image. "
            "If the image is unclear, say so instead of guessing."
        )
        answer = self._query(image, prompt)
        return {"question": question, "answer": answer}

    def analyze(self, image: ImageInput, question: str, target: Optional[str] = None) -> Dict[str, object]:
        normalized_target = self._normalize_target(target or "")
        prompt = self._build_analyze_prompt(question, normalized_target)
        raw = self._query(image, prompt)
        parsed = self._safe_json(raw)

        scene_description = self._extract_scene_description(parsed)
        objects_raw = parsed.get("objects") or []
        answer_text = self._extract_answer_text(parsed)

        if isinstance(objects_raw, list):
            object_items = [self._normalize_target(str(item)) for item in objects_raw]
            objects = [item for item in object_items if item]
        else:
            objects = self._parse_object_list(str(objects_raw))

        if not scene_description and raw:
            LOGGER.debug("Moondream parse missing scene description; using raw-text fallback")
            scene_description = raw

        if not answer_text and raw:
            LOGGER.debug("Moondream parse missing QA answer; using raw-text fallback")
            answer_text = raw

        if not parsed:
            LOGGER.warning("Moondream JSON parsing failed; using raw-text fallback where possible")
        elif not scene_description and not answer_text and not objects and raw:
            LOGGER.warning("Moondream structured fields were mostly empty; using raw-text fallback")

        result: Dict[str, object] = {
            "question": question,
            "scene": {"description": scene_description},
            "objects": {
                "objects": objects,
                "count": len(objects),
                "raw": raw,
            },
            "qa": {
                "question": question,
                "answer": answer_text,
            },
            "raw": raw,
        }

        if normalized_target:
            presence_raw = (parsed.get("presence") or {}).get("present")
            present = self._coerce_presence_value(presence_raw, raw)
            result["target"] = normalized_target
            result["presence"] = {
                "target": normalized_target,
                "present": present,
                "raw": raw,
            }

        return result

    def _verify_model(self) -> None:
        try:
            models = self.client.list()
            model_list = getattr(models, "models", None) or models.get("models", [])
            names = [
                (getattr(model, "model", None) or model.get("model", "")).lower()
                for model in model_list
            ]
            if not any("moondream" in name for name in names):
                raise RuntimeError("Moondream model not found. Run: ollama pull moondream")
        except Exception as exc:
            raise RuntimeError(f"Ollama unavailable: {exc}") from exc

    def _query(self, image: ImageInput, prompt: str) -> str:
        b64 = self._encode_image(image)
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [b64]}],
            options={"temperature": self.temperature},
        )

        if isinstance(resp, dict):
            return (resp.get("message", {}).get("content", "")).strip()
        if hasattr(resp, "message"):
            return (resp.message.content or "").strip()
        return ""

    def _build_analyze_prompt(self, question: str, target: str) -> str:
        schema_lines = [
            '{',
            '  "scene": {',
            '    "description": "1-2 short sentences"',
            '  },',
            '  "objects": ["object1", "object2"],',
            '  "qa": {',
            '    "answer": "short answer to the user question"',
            '  }',
        ]
        if target:
            schema_lines.extend(
                [
                    ',  "presence": {',
                    f'    "target": "{target}",',
                    '    "present": true',
                    '  }',
                ]
            )
        schema_lines.append('}')
        schema = "\n".join(schema_lines)
        return f"""Analyze this image and return ONLY valid JSON.
Do not include markdown fences or extra text.

Use this schema:
{schema}

Rules:
- Keep the scene description concise and grounded in the image.
- `objects` should be a short list of visible physical objects using simple nouns.
- `qa.answer` must answer the user question using only visible evidence.
- If something is unclear, say so briefly instead of guessing.
{"- `presence.present` must be true or false based only on whether the target is visibly present." if target else ""}

User question: {question}
"""

    def _extract_scene_description(self, parsed: Dict[str, object]) -> str:
        scene = parsed.get("scene") or {}
        if isinstance(scene, dict):
            return str(scene.get("description") or "").strip()
        return ""

    def _extract_answer_text(self, parsed: Dict[str, object]) -> str:
        qa = parsed.get("qa") or {}
        if isinstance(qa, dict):
            return str(qa.get("answer") or "").strip()
        return ""

    def _coerce_presence_value(self, presence_raw: object, raw_text: str) -> bool:
        if isinstance(presence_raw, bool):
            return presence_raw
        if presence_raw is None:
            return self._parse_yes_no(raw_text)
        return self._parse_yes_no(str(presence_raw))

    def _encode_image(self, image: ImageInput) -> str:
        pil_image = self._to_pil(image)
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _to_pil(self, image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")
            if image.ndim == 3 and image.shape[2] == 3:
                return Image.fromarray(image.astype(np.uint8), mode="RGB")
            raise ValueError("Unsupported numpy image shape for Moondream input")

        raise TypeError("Unsupported image type for Moondream input")

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
        result: List[str] = []
        for item in cleaned:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result[:40]

    def _parse_yes_no(self, raw: str) -> bool:
        raw = raw.strip().lower()
        if raw in {"true", "1"}:
            return True
        if raw in {"false", "0"}:
            return False
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

    def _safe_json(self, text: str) -> Dict[str, object]:
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

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            try:
                return json.loads(fenced_match.group(1))
            except Exception:
                pass

        return {}
