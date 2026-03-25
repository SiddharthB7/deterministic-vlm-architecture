from __future__ import annotations

import os
import subprocess


class WindowsTTSService:
    """
    Minimal Windows text-to-speech using System.Speech via PowerShell.
    """

    def __init__(self, voice_name: str = "", rate: int = 0, volume: int = 100):
        self.voice_name = voice_name
        self.rate = max(-10, min(10, rate))
        self.volume = max(0, min(100, volume))

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        env = os.environ.copy()
        env["KURAT_TTS_TEXT"] = text
        env["KURAT_TTS_VOICE"] = self.voice_name
        env["KURAT_TTS_RATE"] = str(self.rate)
        env["KURAT_TTS_VOLUME"] = str(self.volume)

        script = r"""
Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.Rate = [int]$env:KURAT_TTS_RATE
$voice.Volume = [int]$env:KURAT_TTS_VOLUME
if ($env:KURAT_TTS_VOICE) {
    try { $voice.SelectVoice($env:KURAT_TTS_VOICE) } catch {}
}
$voice.Speak($env:KURAT_TTS_TEXT)
"""
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            env=env,
        )
