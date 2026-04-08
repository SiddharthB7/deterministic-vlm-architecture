from __future__ import annotations

import logging
import threading
import time

from kurat_core.config import AppConfig
from kurat_core.power_manager import PowerAwareExecutionManager, PowerAwareSettings


def main() -> None:
    cfg = AppConfig()
    logging.basicConfig(level=logging.INFO)

    manager = PowerAwareExecutionManager(
        PowerAwareSettings(
            power_aware_mode=cfg.runtime.power_aware_mode,
            heavy_task_max_concurrency=cfg.runtime.heavy_task_max_concurrency,
            allow_concurrent_heavy_inference=cfg.runtime.allow_concurrent_heavy_inference,
            inference_cooldown_ms=cfg.runtime.inference_cooldown_ms,
            post_model_switch_delay_ms=cfg.runtime.post_model_switch_delay_ms,
            prompt_min_interval_ms=cfg.runtime.prompt_min_interval_ms,
            heavy_task_acquire_timeout_s=cfg.runtime.heavy_task_acquire_timeout_s,
            enable_telemetry=False,
        )
    )

    preferred_model = cfg.models.select_language_model(prefer_small=cfg.runtime.prefer_smaller_models)
    print(f"Preferred language model: {preferred_model}")
    print("Starting two simulated heavy tasks. They should serialize and respect cooldown.")

    def task(name: str, delay_s: float) -> None:
        manager.before_prompt(name)

        # This dummy task is intentionally lightweight but flows through the same power-aware scheduler.
        def _work() -> str:
            time.sleep(delay_s)
            return name

        result = manager.run_heavy_task(
            task_name=name,
            model_name=preferred_model,
            func=_work,
            timeout_s=5.0,
        )
        print(f"Completed: {result}")

    t1 = threading.Thread(target=task, args=("task_one", 0.4))
    t2 = threading.Thread(target=task, args=("task_two", 0.4))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print("Power-aware demo finished.")


if __name__ == "__main__":
    main()
