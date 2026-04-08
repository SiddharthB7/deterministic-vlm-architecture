from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import numpy as np
from PIL import Image

from .types import IntentResult


class LatestFrameProvider(Protocol):
    def get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        ...

    def get_latest_rgb_frame_with_meta(
        self,
        max_age_s: Optional[float] = None,
        reject_stale: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        ...

    def get_latest_rgb_pil(self) -> Optional[Image.Image]:
        ...

    def get_latest_timestamp(self) -> Any:
        ...

    def get_latest_frame_age_seconds(self) -> Optional[float]:
        ...


class IntentRouter(Protocol):
    def classify(self, user_text: str) -> IntentResult:
        ...


class ChatService(Protocol):
    def normal_chat(self, user_text: str, history: str = "") -> str:
        ...

    def answer_with_vision(
        self,
        user_text: str,
        vision_result: Dict[str, Any],
        history: str = "",
    ) -> str:
        ...
