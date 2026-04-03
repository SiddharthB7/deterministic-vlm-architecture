from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class IntentResult:
    mode: str
    targets: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class VisionResult:
    method: str
    payload: Dict[str, Any]


@dataclass(slots=True)
class OrchestratorResult:
    user_text: str
    intent: IntentResult
    vision_used: bool
    vision_result: Optional[VisionResult]
    reply_text: str
