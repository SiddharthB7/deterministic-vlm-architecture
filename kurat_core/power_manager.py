from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(slots=True)
class PowerAwareSettings:
    power_aware_mode: bool = True
    heavy_task_max_concurrency: int = 1
    allow_concurrent_heavy_inference: bool = False
    inference_cooldown_ms: int = 300
    post_model_switch_delay_ms: int = 500
    prompt_min_interval_ms: int = 500
    heavy_task_acquire_timeout_s: float = 30.0
    enable_telemetry: bool = True


class PowerAwareExecutionManager:
    def __init__(self, settings: PowerAwareSettings):
        self.settings = settings
        # This condition serializes heavy inference and provides backpressure without redesigning the call flow.
        self._condition = threading.Condition()
        self._active_heavy_tasks = 0
        self._last_prompt_started_at = 0.0
        self._last_heavy_finished_at = 0.0
        self._last_model_name = ""
        self._tegrastats_available = None

    def before_prompt(self, prompt_name: str) -> None:
        if not self.settings.power_aware_mode:
            return

        # This prompt-level throttle reduces burst current spikes from back-to-back user requests.
        min_interval_s = max(0.0, self.settings.prompt_min_interval_ms / 1000.0)
        with self._condition:
            while True:
                now = time.monotonic()
                remaining = min_interval_s - (now - self._last_prompt_started_at)
                if remaining <= 0.0:
                    self._last_prompt_started_at = now
                    return
                LOGGER.info("Prompt throttle active for %s; waiting %.3fs", prompt_name, remaining)
                self._condition.wait(timeout=remaining)

    def run_heavy_task(
        self,
        task_name: str,
        model_name: str,
        func: Callable[[], T],
        timeout_s: Optional[float] = None,
    ) -> T:
        effective_timeout = self.settings.heavy_task_acquire_timeout_s if timeout_s is None else timeout_s
        telemetry_before = None
        started_at = time.monotonic()

        self._acquire_heavy_slot(task_name=task_name, model_name=model_name, timeout_s=effective_timeout)
        try:
            telemetry_before = self._sample_telemetry()
            LOGGER.info(
                "Heavy task start task=%s model=%s telemetry=%s",
                task_name,
                model_name,
                telemetry_before or "unavailable",
            )
            result = func()
            elapsed = time.monotonic() - started_at
            telemetry_after = self._sample_telemetry()
            LOGGER.info(
                "Heavy task complete task=%s model=%s elapsed_s=%.3f cooldown_ms=%s telemetry=%s",
                task_name,
                model_name,
                elapsed,
                self.settings.inference_cooldown_ms,
                telemetry_after or "unavailable",
            )
            return result
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            telemetry_after = self._sample_telemetry()
            LOGGER.warning(
                "Heavy task aborted task=%s model=%s elapsed_s=%.3f error=%s telemetry=%s",
                task_name,
                model_name,
                elapsed,
                exc,
                telemetry_after or "unavailable",
            )
            raise
        finally:
            self._release_heavy_slot(model_name=model_name)

    def _acquire_heavy_slot(self, task_name: str, model_name: str, timeout_s: float) -> None:
        if not self.settings.power_aware_mode:
            return

        deadline = None if timeout_s is None or timeout_s <= 0 else time.monotonic() + timeout_s
        max_concurrency = self._effective_max_concurrency()

        with self._condition:
            while True:
                now = time.monotonic()
                cooldown_remaining = self._cooldown_remaining(now)

                if self._active_heavy_tasks < max_concurrency and cooldown_remaining <= 0.0:
                    break

                wait_reason = "another heavy inference is running" if self._active_heavy_tasks >= max_concurrency else "cooldown active"
                wait_for = max(0.05, cooldown_remaining) if cooldown_remaining > 0.0 else 0.05
                if deadline is not None:
                    remaining_budget = deadline - now
                    if remaining_budget <= 0.0:
                        raise RuntimeError(f"Power-aware scheduler rejected {task_name}: {wait_reason}")
                    wait_for = min(wait_for, remaining_budget)
                LOGGER.info("Heavy task deferred task=%s model=%s reason=%s wait_s=%.3f", task_name, model_name, wait_reason, wait_for)
                self._condition.wait(timeout=wait_for)

            switch_delay_s = self._model_switch_delay_remaining(model_name)
            if switch_delay_s > 0.0:
                if deadline is not None and (deadline - time.monotonic()) <= 0.0:
                    raise RuntimeError(f"Power-aware scheduler rejected {task_name}: model switch delay exceeded budget")
                LOGGER.info(
                    "Applying model switch delay task=%s from=%s to=%s wait_s=%.3f",
                    task_name,
                    self._last_model_name or "none",
                    model_name,
                    switch_delay_s,
                )
                self._condition.wait(timeout=switch_delay_s)

            self._active_heavy_tasks += 1
            LOGGER.debug(
                "Heavy task lock acquired task=%s model=%s active=%s",
                task_name,
                model_name,
                self._active_heavy_tasks,
            )

    def _release_heavy_slot(self, model_name: str) -> None:
        if not self.settings.power_aware_mode:
            return

        with self._condition:
            self._active_heavy_tasks = max(0, self._active_heavy_tasks - 1)
            self._last_heavy_finished_at = time.monotonic()
            if model_name:
                self._last_model_name = model_name
            LOGGER.debug(
                "Heavy task lock released model=%s active=%s",
                model_name,
                self._active_heavy_tasks,
            )
            self._condition.notify_all()

    def _effective_max_concurrency(self) -> int:
        if not self.settings.allow_concurrent_heavy_inference:
            return 1
        return max(1, self.settings.heavy_task_max_concurrency)

    def _cooldown_remaining(self, now: float) -> float:
        cooldown_s = max(0.0, self.settings.inference_cooldown_ms / 1000.0)
        if cooldown_s <= 0.0 or self._last_heavy_finished_at <= 0.0:
            return 0.0
        return max(0.0, cooldown_s - (now - self._last_heavy_finished_at))

    def _model_switch_delay_remaining(self, model_name: str) -> float:
        if not model_name or not self._last_model_name or model_name == self._last_model_name:
            return 0.0
        return max(0.0, self.settings.post_model_switch_delay_ms / 1000.0)

    def _sample_telemetry(self) -> Optional[str]:
        if not self.settings.enable_telemetry:
            return None

        if self._tegrastats_available is None:
            self._tegrastats_available = shutil.which("tegrastats") is not None
            LOGGER.debug("Telemetry snapshot availability: tegrastats=%s", self._tegrastats_available)

        if not self._tegrastats_available:
            return None

        try:
            completed = subprocess.run(
                ["tegrastats", "--interval", "1000", "--count", "1"],
                capture_output=True,
                text=True,
                timeout=2.5,
                check=False,
            )
            output = (completed.stdout or completed.stderr or "").strip()
            return output.splitlines()[-1] if output else None
        except Exception:
            return None
