from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class ImageFileFrameProvider:
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)

    def get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        if not self.image_path.exists():
            raise FileNotFoundError(str(self.image_path))

        image_bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image file: {self.image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def get_latest_rgb_pil(self) -> Optional[Image.Image]:
        frame = self.get_latest_rgb_frame()
        if frame is None:
            return None
        return Image.fromarray(frame, mode="RGB")

    def get_latest_timestamp(self):
        return None
