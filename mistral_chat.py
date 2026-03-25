"""
Mistral Conversation Brain (Language only)
------------------------------------------
- normal_chat(): responds without vision
- answer_with_vision(): turns tool output into a human answer

This file does NOT run tools.
"""

from __future__ import annotations

from typing import Any, Dict

import requests


class MistralChat:
    def __init__(
        self,
        model: str = "mistral",
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
        timeout_s: int = 180,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout_s = timeout_s

    def normal_chat(self, user_text: str, history: str = "") -> str:
        prompt = f"""You are Kurat, a helpful robot assistant.
Reply naturally, clearly, and briefly.

Conversation:
{history}

User: {user_text}
Assistant:"""
        return self._gen(prompt, temperature=0.3)

    def answer_with_vision(self, user_text: str, vision_result: Dict[str, Any], history: str = "") -> str:
        prompt = f"""You are Kurat, a helpful robot assistant.
Answer using the vision result below.

Rules:
- Be natural, clear, and moderately detailed.
- For scene questions, prefer 2-4 informative sentences instead of a very short reply.
- Summarize the main visible objects first, then mention useful secondary details if they are supported.
- Use confident wording only for details supported by the vision result.
- If something is unclear or missing, say that clearly instead of guessing.
- If the user asked to find something and the result is uncertain, mention that it may not be clearly visible and suggest a better angle.
- For coordinates, location, or position requests, use ONLY YOLO detections and reported boxes. Never infer coordinates from scene descriptions.
- If YOLO found no box for the requested object, say that coordinates cannot be provided from this frame.
- Do not mention extra objects unless they appear explicitly in the vision result.
- If semantic analysis mentions an object but YOLO could not localize it, say it may be visible semantically but not reliably localized.

Conversation:
{history}

User: {user_text}

Vision result:
{vision_result}

Assistant:"""
        return self._gen(prompt, temperature=0.2)

    def _gen(self, prompt: str, temperature: float) -> str:
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return (response.json().get("response") or "").strip()
