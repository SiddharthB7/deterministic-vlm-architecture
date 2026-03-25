from __future__ import annotations

from typing import Optional
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

try:
    import keyboard
except Exception:
    keyboard = None


class WhisperSTTService:
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16000,
        default_duration_s: int = 5,
        language: str = "en",
        min_record_seconds: float = 0.35,
        silence_rms_threshold: float = 0.003,
    ):
        self.sample_rate = sample_rate
        self.default_duration_s = default_duration_s
        self.language = language
        self.min_record_seconds = min_record_seconds
        self.silence_rms_threshold = silence_rms_threshold
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def record_audio(self, seconds: Optional[int] = None) -> np.ndarray:
        seconds = seconds or self.default_duration_s
        print("Speak now...")
        audio = sd.rec(
            int(seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()
        return audio.flatten()

    def record_while_enter(self) -> np.ndarray:
        """
        Record while ENTER is held. Falls back to fixed-duration recording
        if the optional keyboard module is unavailable.
        """
        if keyboard is None:
            return self.record_audio()

        return self.record_while_key("s")

    def record_while_key(self, key: str = "s") -> np.ndarray:
        """
        Record while a keyboard key is held.
        Falls back to fixed-duration recording if the optional keyboard
        module is unavailable.
        """
        if keyboard is None:
            return self.record_audio()

        print(f"Hold {key.upper()} to talk (release {key.upper()} to stop)...")
        while keyboard.is_pressed(key):
            time.sleep(0.02)
        keyboard.wait(key)
        print(f"Recording now... release {key.upper()} to stop.")

        frames = []

        def callback(indata, frames_count, time_info, status):
            if status:
                print(status)
            frames.append(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while keyboard.is_pressed(key):
                time.sleep(0.01)

        if not frames:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames, axis=0).flatten()
        if self._duration_seconds(audio) < self.min_record_seconds:
            return np.array([], dtype=np.float32)
        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        if self._is_too_quiet(audio):
            return ""
        segments, _ = self.model.transcribe(audio, language=self.language, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text.strip()

    def _duration_seconds(self, audio: np.ndarray) -> float:
        return float(audio.size) / float(self.sample_rate)

    def _is_too_quiet(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return rms < self.silence_rms_threshold


if __name__ == "__main__":
    stt = WhisperSTTService()
    while True:
        input("\nPress ENTER to record 5 seconds (Ctrl+C to exit)")
        audio = stt.record_audio()
        text = stt.transcribe(audio)
        print(text if text else "(no speech detected)")
