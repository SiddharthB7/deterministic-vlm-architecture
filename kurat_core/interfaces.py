from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import numpy as np
from PIL import Image

from .types import IntentResult


class LatestFrameProvider(Protocol):
    def get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        ...

    def get_latest_rgb_frame_with_meta(self) -> Optional[Dict[str, Any]]:
        ...

    def get_latest_rgb_pil(self) -> Optional[Image.Image]:
        ...

    def get_latest_timestamp(self) -> Any:
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
