from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image


class ImageFileFrameProvider:
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self._last_timestamp = None

    def get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        if not self.image_path.exists():
            raise FileNotFoundError(str(self.image_path))

        image_bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image file: {self.image_path}")
        self._last_timestamp = self._read_timestamp()
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def get_latest_rgb_frame_with_meta(
        self,
        max_age_s: Optional[float] = None,
        reject_stale: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        frame = self.get_latest_rgb_frame()
        if frame is None:
            return None

        timestamp = self._last_timestamp or self._read_timestamp()
        age_s = self._compute_age_seconds(timestamp)
        # A static image file is a deterministic test source, not a live camera stream.
        # We report age metadata for observability but do not reject it on freshness grounds.
        is_stale = False

        return {
            "frame": frame,
            "timestamp": timestamp,
            "age_s": age_s,
            "is_stale": is_stale,
            "source_type": "image_file",
            "freshness_policy": "file_mtime_best_effort",
            "max_age_s_requested": max_age_s,
            "reject_stale_requested": reject_stale,
        }

    def get_latest_rgb_pil(self) -> Optional[Image.Image]:
        frame = self.get_latest_rgb_frame()
        if frame is None:
            return None
        return Image.fromarray(frame, mode="RGB")

    def get_latest_timestamp(self):
        return self._last_timestamp or self._read_timestamp()

    def get_latest_frame_age_seconds(self) -> Optional[float]:
        return self._compute_age_seconds(self.get_latest_timestamp())

    def _read_timestamp(self):
        if not self.image_path.exists():
            return None
        try:
            return datetime.fromtimestamp(self.image_path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return None

    def _compute_age_seconds(self, timestamp: Any) -> Optional[float]:
        if timestamp is None:
            return None
        if isinstance(timestamp, datetime):
            try:
                now = datetime.now(timezone.utc)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                return max(0.0, (now - timestamp.astimezone(timezone.utc)).total_seconds())
            except Exception:
                return None
        return None
