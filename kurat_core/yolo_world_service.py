from __future__ import annotations

from contextlib import contextmanager
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


FALLBACK_VOCAB = [
    "bottle", "cup", "mug", "glass", "plate", "bowl",
    "jar", "vase", "pot", "pan", "kettle", "container", "basket", "bin",
    "table", "chair", "sofa", "couch", "bed", "desk",
    "stool", "bench", "shelf", "cabinet", "drawer", "dresser", "wardrobe",
    "laptop", "phone", "keyboard", "mouse", "monitor",
    "book", "remote", "tv", "lamp", "plant", "fan", "clock", "picture", "frame",
    "door", "window", "curtain", "fridge", "refrigerator", "microwave",
    "sink", "faucet", "stove", "oven", "toaster", "person", "bag", "box",
]

TARGET_ALIASES = {
    "wooden chair": ["chair"],
    "office chair": ["chair"],
    "dining chair": ["chair"],
    "armchair": ["chair"],
    "stool": ["chair"],
    "table top": ["table", "desk"],
    "side table": ["table"],
    "coffee table": ["table"],
    "dining table": ["table"],
    "study table": ["table", "desk"],
    "tv": ["television", "monitor"],
    "television": ["tv", "monitor"],
    "screen": ["monitor", "tv"],
    "cell phone": ["phone"],
    "mobile phone": ["phone"],
    "smartphone": ["phone"],
    "air conditioner": ["ac"],
    "ac": ["air conditioner"],
    "couch": ["sofa"],
    "sofa": ["couch"],
}

STOPWORDS = {
    "the", "a", "an", "my", "your", "this", "that", "these", "those",
    "left", "right", "front", "back", "near", "nearby", "middle",
    "center", "centre", "top", "bottom", "corner", "coordinates",
    "coordinate", "position", "location",
}


class YoloWorldService:
    def __init__(
        self,
        model_path: str = "yolov8s-world.pt",
        device: str = "cpu",
        imgsz: int = 1280,
        conf: float = 0.20,
        iou: float = 0.5,
        max_det: int = 200,
        debug_save_images: bool = False,
        debug_image_dir: str = "debug_frames",
    ):
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.debug_save_images = debug_save_images
        self.debug_image_dir = Path(debug_image_dir)

        with self._trusted_torch_load():
            self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            pass

        self._last_vocab: Optional[Tuple[str, ...]] = None

    @contextmanager
    def _trusted_torch_load(self):
        original_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = patched_torch_load
        try:
            yield
        finally:
            torch.load = original_torch_load

    def detect(self, image_rgb: np.ndarray, targets: List[str], save_annotated: Optional[bool] = None) -> Dict[str, object]:
        if image_rgb is None:
            raise ValueError("image_rgb must not be None")

        requested_targets = self._normalize_targets(targets)
        primary_vocab, primary_map = self._build_primary_vocab(requested_targets)
        primary = self._run_detection_pass(image_rgb, primary_vocab, primary_map, "primary", save_annotated=False)

        final = primary
        fallback = None

        if requested_targets and primary["count"] == 0:
            fallback_vocab, fallback_map = self._build_fallback_vocab(requested_targets)
            fallback = self._run_detection_pass(image_rgb, fallback_vocab, fallback_map, "fallback", save_annotated=False)
            if fallback["count"] > 0:
                final = fallback

        should_save = self.debug_save_images if save_annotated is None else save_annotated
        annotated_path = None
        if should_save:
            annotated_path = self._save_annotated_image(image_rgb, final["detections"], suffix=f"{final['stage']}_yoloworld")

        return {
            "requested_targets": requested_targets,
            "query_vocab": primary_vocab,
            "count": final["count"],
            "detections": final["detections"],
            "annotated_image": annotated_path,
            "stage": final["stage"],
            "primary_pass": {
                "query_vocab": primary["query_vocab"],
                "count": primary["count"],
            },
            "fallback_pass": None if fallback is None else {
                "query_vocab": fallback["query_vocab"],
                "count": fallback["count"],
            },
        }

    def diagnose_target(self, image_rgb: np.ndarray, target: str) -> Dict[str, object]:
        if image_rgb is None:
            raise ValueError("image_rgb must not be None")

        target = self._clean_phrase(target)
        if not target:
            raise ValueError("Target must not be empty")

        exact_vocab = [target]
        exact_map = {target: target}
        exact = self._run_detection_pass(image_rgb, exact_vocab, exact_map, "exact", save_annotated=False)

        normalized_vocab, normalized_map = self._build_primary_vocab([target])
        normalized = self._run_detection_pass(image_rgb, normalized_vocab, normalized_map, "normalized", save_annotated=False)

        fallback_vocab, fallback_map = self._build_fallback_vocab([target])
        fallback = self._run_detection_pass(image_rgb, fallback_vocab, fallback_map, "fallback", save_annotated=False)

        return {
            "target": target,
            "exact": {
                "query_vocab": exact["query_vocab"],
                "count": exact["count"],
                "detections": exact["detections"],
            },
            "normalized": {
                "query_vocab": normalized["query_vocab"],
                "count": normalized["count"],
                "detections": normalized["detections"],
            },
            "fallback": {
                "query_vocab": fallback["query_vocab"],
                "count": fallback["count"],
                "detections": fallback["detections"],
            },
        }

    def _run_detection_pass(
        self,
        image_rgb: np.ndarray,
        vocab: List[str],
        canonical_map: Dict[str, str],
        stage: str,
        save_annotated: bool,
    ) -> Dict[str, object]:
        self._set_vocabulary(vocab)
        results = self.model.predict(
            source=self._to_bgr_uint8(image_rgb),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
            save=False,
        )[0]

        detections = self._extract_detections(results, canonical_map)
        annotated_path = None
        if save_annotated:
            annotated_path = self._save_annotated_image(image_rgb, detections, suffix=stage)

        return {
            "stage": stage,
            "query_vocab": list(vocab),
            "detections": detections,
            "count": len(detections),
            "annotated_image": annotated_path,
        }

    def _extract_detections(self, results, canonical_map: Dict[str, str]) -> List[Dict[str, object]]:
        detections: List[Dict[str, object]] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        xyxy = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        scores = results.boxes.conf.cpu().numpy()
        names = results.names

        for box, cls, score in zip(xyxy, classes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            raw_label = names[cls] if cls < len(names) else "object"
            raw_clean = self._clean_phrase(raw_label)
            canonical_label = canonical_map.get(raw_clean, raw_clean)
            detections.append(
                {
                    "label": canonical_label,
                    "raw_label": raw_label,
                    "confidence": float(score),
                    "box": [x1, y1, x2, y2],
                }
            )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def _save_annotated_image(self, image_rgb: np.ndarray, detections: List[Dict[str, object]], suffix: str) -> Optional[str]:
        self.debug_image_dir.mkdir(parents=True, exist_ok=True)
        image_bgr = self._to_bgr_uint8(image_rgb).copy()

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            label = detection["label"]
            score = detection["confidence"]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                image_bgr,
                f"{label} {score:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        output_path = self.debug_image_dir / f"annotated_{suffix}.jpg"
        cv2.imwrite(str(output_path), image_bgr)
        return str(output_path)

    def _to_bgr_uint8(self, image_rgb: np.ndarray) -> np.ndarray:
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape HxWx3")
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def _set_vocabulary(self, targets: List[str]) -> None:
        vocab = tuple(targets) if targets else tuple(FALLBACK_VOCAB)
        if vocab != self._last_vocab:
            self.model.set_classes(list(vocab))
            self._last_vocab = vocab

    def _build_primary_vocab(self, requested_targets: List[str]) -> Tuple[List[str], Dict[str, str]]:
        if not requested_targets:
            vocab = list(FALLBACK_VOCAB)
            return vocab, {self._clean_phrase(label): self._clean_phrase(label) for label in vocab}

        vocab: List[str] = []
        canonical_map: Dict[str, str] = {}
        for target in requested_targets:
            for candidate in self._expand_target(target):
                cleaned = self._clean_phrase(candidate)
                if cleaned and cleaned not in vocab:
                    vocab.append(cleaned)
                canonical_map[cleaned] = target
        return vocab, canonical_map

    def _build_fallback_vocab(self, requested_targets: List[str]) -> Tuple[List[str], Dict[str, str]]:
        vocab: List[str] = []
        canonical_map: Dict[str, str] = {}
        matched_fallbacks = self._match_fallback_synonyms(requested_targets)

        for label in matched_fallbacks + FALLBACK_VOCAB:
            cleaned = self._clean_phrase(label)
            if cleaned and cleaned not in vocab:
                vocab.append(cleaned)

        for label in vocab:
            canonical_map[label] = self._map_fallback_label(label, requested_targets)

        return vocab, canonical_map

    def _match_fallback_synonyms(self, requested_targets: List[str]) -> List[str]:
        matches: List[str] = []
        for target in requested_targets:
            target_tokens = set(self._tokenize(target))
            alias_tokens = set()
            for alias in self._expand_target(target):
                alias_tokens.update(self._tokenize(alias))

            for fallback_label in FALLBACK_VOCAB:
                fallback_tokens = set(self._tokenize(fallback_label))
                if fallback_label == target:
                    matches.append(fallback_label)
                    continue
                if target_tokens and fallback_tokens.intersection(target_tokens):
                    matches.append(fallback_label)
                    continue
                if alias_tokens and fallback_tokens.intersection(alias_tokens):
                    matches.append(fallback_label)

        ordered: List[str] = []
        seen = set()
        for label in matches:
            cleaned = self._clean_phrase(label)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                ordered.append(cleaned)
        return ordered

    def _map_fallback_label(self, fallback_label: str, requested_targets: List[str]) -> str:
        fallback_tokens = set(self._tokenize(fallback_label))
        best_target = fallback_label
        best_score = 0

        for target in requested_targets:
            target_tokens = set(self._tokenize(target))
            alias_tokens = set()
            for alias in self._expand_target(target):
                alias_tokens.update(self._tokenize(alias))

            score = len(fallback_tokens.intersection(target_tokens | alias_tokens))
            if fallback_label == target:
                score += 3
            if score > best_score:
                best_score = score
                best_target = target

        return best_target

    def _expand_target(self, target: str) -> List[str]:
        target = self._clean_phrase(target)
        expanded = [target]
        expanded.extend(TARGET_ALIASES.get(target, []))

        tokens = [token for token in target.split() if token not in STOPWORDS]
        if len(tokens) > 1:
            expanded.append(" ".join(tokens))
            expanded.append(tokens[-1])
        elif tokens:
            expanded.append(tokens[0])

        result: List[str] = []
        seen = set()
        for item in expanded:
            cleaned = self._clean_phrase(item)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result

    def _normalize_targets(self, targets: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for target in targets:
            cleaned = self._clean_phrase(target)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
        return normalized

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in self._clean_phrase(text).split() if token and token not in STOPWORDS]

    def _clean_phrase(self, text: str) -> str:
        text = re.sub(r"[^a-z0-9\s-]", " ", (text or "").lower())
        text = re.sub(r"\s+", " ", text).strip(" -")
        return text[:50]
