from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Dict, Optional

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage


LOGGER = logging.getLogger(__name__)


class ROSRealSenseFrameProvider:
    def __init__(
        self,
        node: Node,
        color_topic: str,
        depth_topic: Optional[str] = None,
        queue_size: int = 10,
        store_depth: bool = False,
        max_frame_age_s: float = 1.0,
        skip_stale_frames: bool = True,
    ):
        self.node = node
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.store_depth = store_depth and bool(depth_topic)
        self.max_frame_age_s = max_frame_age_s
        self.skip_stale_frames = skip_stale_frames
        self._bridge = CvBridge()
        self._lock = Lock()
        self._latest_rgb_frame: Optional[np.ndarray] = None
        self._latest_depth_frame: Optional[np.ndarray] = None
        self._latest_stamp = None

        self._color_subscription = self.node.create_subscription(
            RosImage,
            self.color_topic,
            self._on_color_image,
            queue_size,
        )

        self._depth_subscription = None
        if self.store_depth and self.depth_topic:
            self._depth_subscription = self.node.create_subscription(
                RosImage,
                self.depth_topic,
                self._on_depth_image,
                queue_size,
            )

    def get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        meta = self.get_latest_rgb_frame_with_meta()
        if meta is None:
            return None
        return meta["frame"]

    def get_latest_rgb_frame_with_meta(
        self,
        max_age_s: Optional[float] = None,
        reject_stale: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._latest_rgb_frame is None:
                return None
            frame = self._latest_rgb_frame.copy()
            stamp = self._latest_stamp

        age_s = self._compute_age_seconds(stamp)
        threshold = self.max_frame_age_s if max_age_s is None else max_age_s
        should_reject = self.skip_stale_frames if reject_stale is None else reject_stale
        is_stale = age_s is not None and threshold > 0.0 and age_s > threshold

        if is_stale and should_reject:
            LOGGER.warning(
                "Rejecting stale RGB frame from %s (age=%.3fs threshold=%.3fs)",
                self.color_topic,
                age_s,
                threshold,
            )
            return None

        return {
            "frame": frame,
            "timestamp": stamp,
            "age_s": age_s,
            "is_stale": is_stale,
        }

    def get_latest_rgb_pil(self) -> Optional[Image.Image]:
        meta = self.get_latest_rgb_frame_with_meta()
        if meta is None:
            return None
        return Image.fromarray(meta["frame"], mode="RGB")

    def get_latest_depth_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_depth_frame is None:
                return None
            return self._latest_depth_frame.copy()

    def get_latest_timestamp(self):
        with self._lock:
            return self._latest_stamp

    def get_latest_frame_age_seconds(self) -> Optional[float]:
        with self._lock:
            stamp = self._latest_stamp
        return self._compute_age_seconds(stamp)

    def _on_color_image(self, msg: RosImage) -> None:
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except (CvBridgeError, RuntimeError, ValueError) as exc:
            LOGGER.warning("Failed to convert color frame from %s: %s", self.color_topic, exc)
            return

        with self._lock:
            self._latest_rgb_frame = rgb_image
            self._latest_stamp = msg.header.stamp

    def _on_depth_image(self, msg: RosImage) -> None:
        try:
            depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except (CvBridgeError, RuntimeError, ValueError) as exc:
            LOGGER.warning("Failed to convert depth frame from %s: %s", self.depth_topic, exc)
            return

        with self._lock:
            self._latest_depth_frame = np.asarray(depth_image)

    def _compute_age_seconds(self, stamp) -> Optional[float]:
        if stamp is None:
            return None

        try:
            now_ns = self.node.get_clock().now().nanoseconds
            stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            return max(0.0, (now_ns - stamp_ns) / 1_000_000_000.0)
        except Exception:
            return None
