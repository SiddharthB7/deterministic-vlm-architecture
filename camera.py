import cv2
from PIL import Image


class LaptopCameraFrameProvider:
    """
    Simple laptop webcam provider using OpenCV.
    Captures a single frame on demand.
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open laptop camera")

        # Optional: set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self) -> Image.Image:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        # OpenCV → RGB → PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def release(self):
        if self.cap:
            self.cap.release()
