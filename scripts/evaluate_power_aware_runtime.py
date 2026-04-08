from __future__ import annotations

import csv
import json
import importlib.util
import logging
import os
import platform
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_module(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_POWER_MANAGER_MODULE = _load_module("kurat_power_manager_runtime_eval", "kurat_core/power_manager.py")
PowerAwareExecutionManager = _POWER_MANAGER_MODULE.PowerAwareExecutionManager
PowerAwareSettings = _POWER_MANAGER_MODULE.PowerAwareSettings


RESULTS_DIR = PROJECT_ROOT / "results"
CSV_PATH = RESULTS_DIR / "runtime_eval_results.csv"
JSON_PATH = RESULTS_DIR / "runtime_eval_raw.json"
SUMMARY_PATH = RESULTS_DIR / "runtime_eval_summary.md"

REPRESENTATIVE_PROMPTS = [
    "Describe the scene in front of you.",
    "What objects are visible?",
    "Find the main object in the image.",
    "Is there a person in view?",
    "Locate any bottle, cup, or chair if present.",
    "What is the most important thing in this image?",
]

SEQUENTIAL_REQUEST_COUNT = 3
BURST_REQUEST_COUNT = 3
BURST_GAP_SEC = 0.01
REQUEST_TIMEOUT_SEC = 180.0


@dataclass
class RequestRecord:
    request_id: str
    scenario: str
    mode: str
    prompt: str
    start_time: str
    end_time: str
    duration_sec: float
    success: bool
    timeout: bool
    model_used: str
    execution_label: str
    notes: str


class InstrumentedExecutionManager(PowerAwareExecutionManager):
    def __init__(self, settings: PowerAwareSettings):
        super().__init__(settings)
        self._metrics_lock = threading.Lock()
        self.prompt_wait_events = 0
        self.heavy_wait_events = 0
        self.cooldown_wait_events = 0
        self.active_conflict_events = 0
        self.overlap_attempts = 0
        self.total_heavy_tasks = 0
        self.current_runtime_active = 0
        self.max_runtime_active = 0
        self.rejections = 0

    def before_prompt(self, prompt_name: str) -> None:
        if self.settings.power_aware_mode:
            min_interval_s = max(0.0, self.settings.prompt_min_interval_ms / 1000.0)
            with self._condition:
                now = time.monotonic()
                remaining = min_interval_s - (now - self._last_prompt_started_at)
                if remaining > 0.0:
                    with self._metrics_lock:
                        self.prompt_wait_events += 1
        return super().before_prompt(prompt_name)

    def run_heavy_task(
        self,
        task_name: str,
        model_name: str,
        func: Callable[[], Any],
        timeout_s: Optional[float] = None,
    ) -> Any:
        with self._metrics_lock:
            if self.current_runtime_active > 0:
                self.overlap_attempts += 1
            self.current_runtime_active += 1
            self.total_heavy_tasks += 1
            self.max_runtime_active = max(self.max_runtime_active, self.current_runtime_active)
        try:
            return super().run_heavy_task(task_name=task_name, model_name=model_name, func=func, timeout_s=timeout_s)
        finally:
            with self._metrics_lock:
                self.current_runtime_active = max(0, self.current_runtime_active - 1)

    def _acquire_heavy_slot(self, task_name: str, model_name: str, timeout_s: float) -> None:
        if not self.settings.power_aware_mode:
            return

        deadline = None if timeout_s is None or timeout_s <= 0 else time.monotonic() + timeout_s
        max_concurrency = self._effective_max_concurrency()

        with self._condition:
            while True:
                now = time.monotonic()
                cooldown_remaining = self._cooldown_remaining(now)
                conflict_active = self._active_heavy_tasks >= max_concurrency

                if not conflict_active and cooldown_remaining <= 0.0:
                    break

                with self._metrics_lock:
                    self.heavy_wait_events += 1
                    if conflict_active:
                        self.active_conflict_events += 1
                    if cooldown_remaining > 0.0:
                        self.cooldown_wait_events += 1

                wait_reason = "another heavy inference is running" if conflict_active else "cooldown active"
                wait_for = max(0.05, cooldown_remaining) if cooldown_remaining > 0.0 else 0.05
                if deadline is not None:
                    remaining_budget = deadline - now
                    if remaining_budget <= 0.0:
                        with self._metrics_lock:
                            self.rejections += 1
                        raise RuntimeError(f"Power-aware scheduler rejected {task_name}: {wait_reason}")
                    wait_for = min(wait_for, remaining_budget)
                LOGGER.info(
                    "Heavy task deferred task=%s model=%s reason=%s wait_s=%.3f",
                    task_name,
                    model_name,
                    wait_reason,
                    wait_for,
                )
                self._condition.wait(timeout=wait_for)

            switch_delay_s = self._model_switch_delay_remaining(model_name)
            if switch_delay_s > 0.0:
                with self._metrics_lock:
                    self.heavy_wait_events += 1
                if deadline is not None and (deadline - time.monotonic()) <= 0.0:
                    with self._metrics_lock:
                        self.rejections += 1
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

    def snapshot(self) -> Dict[str, int]:
        with self._metrics_lock:
            return {
                "prompt_wait_events": self.prompt_wait_events,
                "heavy_wait_events": self.heavy_wait_events,
                "cooldown_wait_events": self.cooldown_wait_events,
                "active_conflict_events": self.active_conflict_events,
                "overlap_attempts": self.overlap_attempts,
                "total_heavy_tasks": self.total_heavy_tasks,
                "max_runtime_active": self.max_runtime_active,
                "rejections": self.rejections,
            }


class BaseRuntimeHarness:
    def __init__(self, mode_label: str, manager: InstrumentedExecutionManager):
        self.mode_label = mode_label
        self.manager = manager
        self.model_used = "unknown"
        self.execution_label = "unknown"
        self.environment_note = ""
        self.sample_frame_note = ""

    def handle_prompt(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class MockRuntimeHarness(BaseRuntimeHarness):
    def __init__(self, mode_label: str, manager: InstrumentedExecutionManager):
        super().__init__(mode_label, manager)
        self.model_used = "mocked-heavy-tasks"
        self.execution_label = "mocked scheduler validation"
        self.environment_note = "Mock execution path using the real PowerAwareExecutionManager with deterministic sleep-based heavy tasks."
        self.sample_frame_note = "No real frame required in mock mode."

    def handle_prompt(self, prompt: str) -> Dict[str, Any]:
        self.manager.before_prompt("handle_text")
        routed_mode = self._route_prompt(prompt)
        heavy_plan = self._build_heavy_plan(routed_mode)

        for task_name, model_name, delay_s, timeout_s in heavy_plan:
            self.manager.run_heavy_task(
                task_name=task_name,
                model_name=model_name,
                func=lambda delay_s=delay_s, task_name=task_name: self._simulate_work(task_name, delay_s),
                timeout_s=timeout_s,
            )

        return {
            "reply_text": f"Mocked pipeline completed for {routed_mode}.",
            "routed_mode": routed_mode,
            "task_count": len(heavy_plan),
            "tasks": [task_name for task_name, _, _, _ in heavy_plan],
        }

    def _simulate_work(self, task_name: str, delay_s: float) -> str:
        time.sleep(delay_s)
        return task_name

    def _route_prompt(self, prompt: str) -> str:
        text = prompt.lower()
        if "find" in text or "is there" in text:
            return "vision_find"
        if "most important object" in text:
            return "vision_attribute"
        if "describe" in text or "what do you see" in text:
            return "vision_scene"
        return "chat"

    def _build_heavy_plan(self, routed_mode: str) -> List[tuple[str, str, float, float]]:
        plan: List[tuple[str, str, float, float]] = [
            ("intent_routing", "qwen2.5:3b-instruct", 0.08, 5.0),
        ]
        if routed_mode == "vision_find":
            plan.extend(
                [
                    ("yolo_detect_primary", "assets/models/yolov8s-world.pt", 0.22, 5.0),
                    ("semantic_vision", "moondream", 0.16, 5.0),
                    ("language_answer", "qwen2.5:3b-instruct", 0.11, 5.0),
                ]
            )
        elif routed_mode == "vision_scene":
            plan.extend(
                [
                    ("semantic_vision", "moondream", 0.24, 5.0),
                    ("language_answer", "qwen2.5:3b-instruct", 0.12, 5.0),
                ]
            )
        elif routed_mode == "vision_attribute":
            plan.extend(
                [
                    ("semantic_vision", "moondream", 0.18, 5.0),
                    ("language_answer", "qwen2.5:3b-instruct", 0.10, 5.0),
                ]
            )
        else:
            plan.append(("language_answer", "qwen2.5:3b-instruct", 0.10, 5.0))
        return plan


class RealRuntimeHarness(BaseRuntimeHarness):
    def __init__(self, mode_label: str, manager: InstrumentedExecutionManager, sample_image: Path):
        super().__init__(mode_label, manager)
        self.sample_image = sample_image
        self.orchestrator = self._build_orchestrator()
        self.execution_label = "real local models"
        self.model_used = "qwen2.5:3b-instruct + moondream + assets/models/yolov8s-world.pt"
        self.environment_note = "Real local execution using the current architecture around a static sample frame."
        self.sample_frame_note = f"Static sample frame: {self.sample_image.name}"

    def _build_orchestrator(self):
        from kurat_core.intent_router import MistralIntentRouter
        from kurat_core.mistral_chat import MistralChat
        from kurat_core.moondream_service import MoondreamService
        from kurat_core.orchestrator import KuratOrchestrator
        from kurat_core.yolo_world_service import YoloWorldService
        from kurat_io.frame_sources.image_file_provider import ImageFileFrameProvider

        config_module = _load_module("kurat_config_runtime_eval", "kurat_core/config.py")
        cfg = config_module.AppConfig()
        cfg.runtime.enable_telemetry = False
        cfg.runtime.heavy_task_acquire_timeout_s = 5.0
        if self.mode_label == "baseline":
            cfg.runtime.power_aware_mode = False
            cfg.runtime.heavy_task_max_concurrency = 8
            cfg.runtime.allow_concurrent_heavy_inference = True
            cfg.runtime.inference_cooldown_ms = 0
            cfg.runtime.prompt_min_interval_ms = 0
            cfg.runtime.post_model_switch_delay_ms = 0
        else:
            cfg.runtime.power_aware_mode = True
            cfg.runtime.heavy_task_max_concurrency = 1
            cfg.runtime.allow_concurrent_heavy_inference = False
            cfg.runtime.inference_cooldown_ms = 300
            cfg.runtime.prompt_min_interval_ms = 500
            cfg.runtime.post_model_switch_delay_ms = 500

        language_model_name = cfg.models.select_language_model(prefer_small=cfg.runtime.prefer_smaller_models)
        frame_provider = ImageFileFrameProvider(str(self.sample_image))
        router = MistralIntentRouter(
            model=language_model_name,
            ollama_url=cfg.models.ollama_generate_url,
            timeout_s=cfg.models.intent_timeout_s,
            execution_manager=self.manager,
        )
        chat = MistralChat(
            model=language_model_name,
            ollama_url=cfg.models.ollama_generate_url,
            timeout_s=cfg.models.chat_timeout_s,
            execution_manager=self.manager,
        )
        yolo = YoloWorldService(
            model_path=cfg.models.yolo_model_path,
            device=cfg.models.yolo_device,
            imgsz=cfg.runtime.yolo_imgsz,
            conf=cfg.runtime.yolo_conf,
            iou=cfg.runtime.yolo_iou,
            max_det=cfg.runtime.yolo_max_det,
            debug_save_images=False,
            debug_image_dir=cfg.runtime.debug_image_dir,
            execution_manager=self.manager,
        )
        moon = MoondreamService(
            model=cfg.models.moondream_model_name,
            host=cfg.models.ollama_host,
            execution_manager=self.manager,
        )
        return KuratOrchestrator(
            frame_provider=frame_provider,
            intent_router=router,
            chat=chat,
            yolo=yolo,
            moon=moon,
            enable_moondream_fallback_on_find=cfg.runtime.enable_moondream_fallback_on_find,
            max_frame_age_s=cfg.runtime.max_frame_age_s,
            skip_stale_frames=cfg.runtime.skip_stale_frames,
            moondream_frame_max_dim=cfg.runtime.moondream_frame_max_dim,
            yolo_frame_max_dim=cfg.runtime.yolo_frame_max_dim,
            execution_manager=self.manager,
        )

    def handle_prompt(self, prompt: str) -> Dict[str, Any]:
        result = self.orchestrator.handle_text(prompt)
        pipeline = result.vision_result.method if result.vision_result is not None else "chat"
        return {
            "reply_text": result.reply_text,
            "routed_mode": result.intent.mode,
            "task_count": None,
            "tasks": [],
            "pipeline": pipeline,
        }


def build_manager(mode_label: str) -> InstrumentedExecutionManager:
    if mode_label == "baseline":
        settings = PowerAwareSettings(
            power_aware_mode=False,
            heavy_task_max_concurrency=8,
            allow_concurrent_heavy_inference=True,
            inference_cooldown_ms=0,
            post_model_switch_delay_ms=0,
            prompt_min_interval_ms=0,
            heavy_task_acquire_timeout_s=5.0,
            enable_telemetry=False,
        )
    else:
        settings = PowerAwareSettings(
            power_aware_mode=True,
            heavy_task_max_concurrency=1,
            allow_concurrent_heavy_inference=False,
            inference_cooldown_ms=300,
            post_model_switch_delay_ms=500,
            prompt_min_interval_ms=500,
            heavy_task_acquire_timeout_s=5.0,
            enable_telemetry=False,
        )
    return InstrumentedExecutionManager(settings)


def find_sample_image() -> Optional[Path]:
    for candidate in [
        PROJECT_ROOT / "assets" / "sample_images" / "raw" / "test_image.png",
        PROJECT_ROOT / "assets" / "sample_images" / "raw" / "test2.jpg",
        PROJECT_ROOT / "assets" / "sample_images" / "raw" / "captured_image.jpg",
        PROJECT_ROOT / "assets" / "sample_images" / "raw" / "captured_20260128_215752.jpg",
    ]:
        if candidate.exists():
            return candidate
    return None


def try_build_real_harness(mode_label: str, manager: InstrumentedExecutionManager) -> tuple[BaseRuntimeHarness, Optional[str]]:
    sample_image = find_sample_image()
    if sample_image is None:
        return MockRuntimeHarness(mode_label, manager), "No sample image found; used mocked scheduler validation."

    try:
        harness = RealRuntimeHarness(mode_label, manager, sample_image)
        return harness, None
    except Exception as exc:
        return MockRuntimeHarness(mode_label, manager), f"Real runtime unavailable ({type(exc).__name__}: {exc}); used mocked scheduler validation."


def generate_prompts(count: int) -> List[str]:
    prompts: List[str] = []
    for idx in range(count):
        prompts.append(REPRESENTATIVE_PROMPTS[idx % len(REPRESENTATIVE_PROMPTS)])
    return prompts


def build_request_record(
    harness: BaseRuntimeHarness,
    scenario: str,
    request_index: int,
    prompt: str,
) -> RequestRecord:
    started_wall = datetime.now().astimezone()
    started_perf = time.perf_counter()
    success = True
    timeout = False
    notes = ""

    try:
        payload = harness.handle_prompt(prompt)
        routed_mode = payload.get("routed_mode", "unknown")
        pipeline = payload.get("pipeline", "")
        tasks = payload.get("tasks") or []
        notes = f"routed_mode={routed_mode}"
        if pipeline:
            notes += f"; pipeline={pipeline}"
        if tasks:
            notes += f"; tasks={','.join(tasks)}"
    except TimeoutError as exc:
        success = False
        timeout = True
        notes = f"timeout={exc}"
    except Exception as exc:
        success = False
        timeout = "timeout" in str(exc).lower()
        notes = f"{type(exc).__name__}: {exc}"

    ended_wall = datetime.now().astimezone()
    duration = time.perf_counter() - started_perf
    record = RequestRecord(
        request_id=f"{scenario}_{harness.mode_label}_{request_index:02d}",
        scenario=scenario,
        mode=harness.mode_label,
        prompt=prompt,
        start_time=started_wall.isoformat(timespec="milliseconds"),
        end_time=ended_wall.isoformat(timespec="milliseconds"),
        duration_sec=round(duration, 4),
        success=success,
        timeout=timeout,
        model_used=harness.model_used,
        execution_label=harness.execution_label,
        notes=notes,
    )
    return record


def execute_request_with_timeout(
    harness: BaseRuntimeHarness,
    scenario: str,
    request_index: int,
    prompt: str,
    timeout_s: float,
) -> RequestRecord:
    outcome: Dict[str, RequestRecord] = {}
    outcome_lock = threading.Lock()

    def _worker() -> None:
        record = build_request_record(harness, scenario, request_index, prompt)
        with outcome_lock:
            outcome["record"] = record

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)

    with outcome_lock:
        finished_record = outcome.get("record")
    if finished_record is not None:
        return finished_record

    now = datetime.now().astimezone().isoformat(timespec="milliseconds")
    return RequestRecord(
        request_id=f"{scenario}_{harness.mode_label}_{request_index:02d}",
        scenario=scenario,
        mode=harness.mode_label,
        prompt=prompt,
        start_time=now,
        end_time=now,
        duration_sec=round(timeout_s, 4),
        success=False,
        timeout=True,
        model_used=harness.model_used,
        execution_label=harness.execution_label,
        notes=f"request timed out after {timeout_s:.1f}s; worker did not complete cleanly",
    )


def run_scenario(harness: BaseRuntimeHarness, scenario: str, prompts: List[str]) -> Dict[str, Any]:
    records: List[RequestRecord] = []
    metrics_before = harness.manager.snapshot()

    if scenario == "sequential":
        for idx, prompt in enumerate(prompts, start=1):
            records.append(
                execute_request_with_timeout(
                    harness=harness,
                    scenario=scenario,
                    request_index=idx,
                    prompt=prompt,
                    timeout_s=REQUEST_TIMEOUT_SEC,
                )
            )
    elif scenario == "burst":
        threads: List[threading.Thread] = []
        results: Dict[int, RequestRecord] = {}
        results_lock = threading.Lock()

        def _run_burst_request(request_index: int, burst_prompt: str) -> None:
            record = execute_request_with_timeout(
                harness=harness,
                scenario=scenario,
                request_index=request_index,
                prompt=burst_prompt,
                timeout_s=REQUEST_TIMEOUT_SEC,
            )
            with results_lock:
                results[request_index] = record

        for idx, prompt in enumerate(prompts, start=1):
            thread = threading.Thread(
                target=_run_burst_request,
                args=(idx, prompt),
                daemon=True,
            )
            threads.append(thread)
            thread.start()
            time.sleep(BURST_GAP_SEC)
        for thread in threads:
            thread.join(timeout=REQUEST_TIMEOUT_SEC + 5.0)
        for idx, prompt in enumerate(prompts, start=1):
            with results_lock:
                record = results.get(idx)
            if record is None:
                now = datetime.now().astimezone().isoformat(timespec="milliseconds")
                record = RequestRecord(
                    request_id=f"{scenario}_{harness.mode_label}_{idx:02d}",
                    scenario=scenario,
                    mode=harness.mode_label,
                    prompt=prompt,
                    start_time=now,
                    end_time=now,
                    duration_sec=round(REQUEST_TIMEOUT_SEC + 5.0, 4),
                    success=False,
                    timeout=True,
                    model_used=harness.model_used,
                    execution_label=harness.execution_label,
                    notes="scenario watchdog triggered; burst worker did not publish a result",
                )
            records.append(record)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    metrics_after = harness.manager.snapshot()
    records.sort(key=lambda item: item.request_id)
    scenario_metrics = compute_metrics(records)
    scenario_metrics["scheduler"] = diff_snapshots(metrics_before, metrics_after)
    scenario_metrics["scheduler"]["max_runtime_active_observed"] = metrics_after["max_runtime_active"]
    scenario_metrics["scheduler"]["notes"] = build_scheduler_notes(harness.mode_label, scenario_metrics)
    return {
        "records": [asdict(record) for record in records],
        "metrics": scenario_metrics,
    }


def compute_metrics(records: List[RequestRecord]) -> Dict[str, Any]:
    durations = [record.duration_sec for record in records]
    completed = sum(1 for record in records if record.success)
    failed = len(records) - completed
    timeout_count = sum(1 for record in records if record.timeout)
    if durations:
        average_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)
        max_duration = max(durations)
        min_duration = min(durations)
    else:
        average_duration = median_duration = max_duration = min_duration = 0.0

    return {
        "total_requests": len(records),
        "completed_requests": completed,
        "failed_requests": failed,
        "average_duration_sec": round(average_duration, 4),
        "median_duration_sec": round(median_duration, 4),
        "max_duration_sec": round(max_duration, 4),
        "min_duration_sec": round(min_duration, 4),
        "timeout_count": timeout_count,
    }


def diff_snapshots(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    diff: Dict[str, int] = {}
    for key, after_value in after.items():
        before_value = before.get(key, 0)
        diff[key] = after_value - before_value
    return diff


def build_scheduler_notes(mode_label: str, metrics: Dict[str, Any]) -> str:
    scheduler = metrics["scheduler"]
    if mode_label == "power_aware":
        return (
            f"Serialized mode observed with {scheduler['heavy_wait_events']} heavy-task wait events, "
            f"{scheduler['prompt_wait_events']} prompt-throttle events, and "
            f"{scheduler['cooldown_wait_events']} cooldown waits."
        )
    return (
        f"Baseline approximation with power-aware gating disabled. "
        f"Observed {scheduler['overlap_attempts']} overlap attempts and "
        f"max concurrent heavy tasks {scheduler['max_runtime_active_observed']}."
    )


def aggregate_mode_metrics(mode_runs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for mode_label, mode_payload in mode_runs.items():
        all_records = [
            RequestRecord(**record)
            for scenario_payload in mode_payload["scenarios"].values()
            for record in scenario_payload["records"]
        ]
        aggregate_metrics = compute_metrics(all_records)
        overlap_events = sum(
            scenario_payload["metrics"]["scheduler"]["overlap_attempts"]
            for scenario_payload in mode_payload["scenarios"].values()
        )
        conflict_events = sum(
            scenario_payload["metrics"]["scheduler"]["active_conflict_events"]
            for scenario_payload in mode_payload["scenarios"].values()
        )
        wait_events = sum(
            scenario_payload["metrics"]["scheduler"]["heavy_wait_events"]
            for scenario_payload in mode_payload["scenarios"].values()
        )
        prompt_wait_events = sum(
            scenario_payload["metrics"]["scheduler"]["prompt_wait_events"]
            for scenario_payload in mode_payload["scenarios"].values()
        )
        aggregate_metrics["overlap_conflict_events"] = overlap_events if mode_label == "baseline" else conflict_events
        aggregate_metrics["scheduler_wait_events"] = wait_events
        aggregate_metrics["prompt_wait_events"] = prompt_wait_events
        aggregate_metrics["stability_note"] = derive_stability_note(mode_label, aggregate_metrics)
        aggregated[mode_label] = aggregate_metrics
    return aggregated


def derive_stability_note(mode_label: str, metrics: Dict[str, Any]) -> str:
    if metrics["failed_requests"] > 0 or metrics["timeout_count"] > 0:
        return "Observed failures or timeouts during the run."
    if mode_label == "power_aware":
        if metrics["scheduler_wait_events"] > 0 or metrics["prompt_wait_events"] > 0:
            return "Scheduler visibly serialized or throttled requests while completing the workload."
        return "Completed cleanly, but little queueing was observable in this environment."
    if metrics["overlap_conflict_events"] > 0:
        return "Concurrent heavy-task overlap attempts were observed with gating disabled."
    return "Completed cleanly with limited observable contention."


def write_csv(mode_runs: Dict[str, Dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for mode_payload in mode_runs.values():
        for scenario_payload in mode_payload["scenarios"].values():
            rows.extend(scenario_payload["records"])

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "request_id",
                "scenario",
                "mode",
                "prompt",
                "start_time",
                "end_time",
                "duration_sec",
                "success",
                "timeout",
                "model_used",
                "execution_label",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_comparison_table(aggregated: Dict[str, Dict[str, Any]]) -> str:
    baseline = aggregated["baseline"]
    power_aware = aggregated["power_aware"]
    rows = [
        ("Total Requests", baseline["total_requests"], power_aware["total_requests"]),
        ("Completed", baseline["completed_requests"], power_aware["completed_requests"]),
        ("Failed", baseline["failed_requests"], power_aware["failed_requests"]),
        ("Avg Duration (s)", baseline["average_duration_sec"], power_aware["average_duration_sec"]),
        ("Median Duration (s)", baseline["median_duration_sec"], power_aware["median_duration_sec"]),
        ("Max Duration (s)", baseline["max_duration_sec"], power_aware["max_duration_sec"]),
        ("Timeout Count", baseline["timeout_count"], power_aware["timeout_count"]),
        ("Overlap/Conflict Events", baseline["overlap_conflict_events"], power_aware["overlap_conflict_events"]),
        ("Stability Observation", baseline["stability_note"], power_aware["stability_note"]),
    ]
    lines = [
        "| Metric | Baseline | Power-aware |",
        "|--------|----------|-------------|",
    ]
    for label, baseline_value, power_value in rows:
        lines.append(f"| {label} | {baseline_value} | {power_value} |")
    return "\n".join(lines)


def build_observations(aggregated: Dict[str, Dict[str, Any]], mode_runs: Dict[str, Dict[str, Any]]) -> str:
    baseline = aggregated["baseline"]
    power_aware = aggregated["power_aware"]
    lines = [
        f"- Baseline completed {baseline['completed_requests']} of {baseline['total_requests']} requests with {baseline['overlap_conflict_events']} observed overlap/conflict events.",
        f"- Power-aware completed {power_aware['completed_requests']} of {power_aware['total_requests']} requests with {power_aware['prompt_wait_events']} prompt-throttle events and {power_aware['scheduler_wait_events']} heavy-task wait events.",
        f"- Average duration changed from {baseline['average_duration_sec']} s in baseline mode to {power_aware['average_duration_sec']} s in power-aware mode.",
        f"- Maximum duration changed from {baseline['max_duration_sec']} s in baseline mode to {power_aware['max_duration_sec']} s in power-aware mode.",
        f"- Burst-mode power-aware scheduler note: {mode_runs['power_aware']['scenarios']['burst']['metrics']['scheduler']['notes']}",
    ]
    if "mocked" in mode_runs["baseline"]["execution_label"] or "mocked" in mode_runs["power_aware"]["execution_label"]:
        lines.append("- At least one run used mocked heavy-task calls because the full local model stack was unavailable in this environment.")
    return "\n".join(lines)


def detect_full_three_brain_execution(mode_runs: Dict[str, Dict[str, Any]]) -> bool:
    saw_scene = False
    saw_detection = False
    for mode_payload in mode_runs.values():
        for scenario_payload in mode_payload["scenarios"].values():
            for record in scenario_payload["records"]:
                notes = str(record.get("notes", ""))
                if "pipeline=moondream_scene" in notes or "routed_mode=vision_scene" in notes:
                    saw_scene = True
                if "pipeline=yolo_world_find" in notes or "pipeline=fallback_moondream" in notes or "routed_mode=vision_find" in notes:
                    saw_detection = True
    return saw_scene and saw_detection


def build_jetson_stress_observations(aggregated: Dict[str, Dict[str, Any]], mode_runs: Dict[str, Dict[str, Any]]) -> str:
    power_aware = aggregated["power_aware"]
    lines = []
    if power_aware["failed_requests"] == 0 and power_aware["timeout_count"] == 0:
        lines.append("- completed successfully")
    else:
        lines.append("- runtime failures or timeouts were observed")
    if power_aware["average_duration_sec"] > aggregated["baseline"]["average_duration_sec"] * 2:
        lines.append("- slowed significantly under the power-aware schedule")
    else:
        lines.append("- slowdown was limited in this workload")
    if power_aware["scheduler_wait_events"] > 0 or power_aware["prompt_wait_events"] > 0:
        lines.append("- scheduler intervened heavily")
    else:
        lines.append("- little scheduler intervention was observable")
    if power_aware["failed_requests"] == 0:
        lines.append("- no crash observed during this run")
    else:
        lines.append("- instability or crash symptoms were observed")
    if "mocked" in mode_runs["power_aware"]["execution_label"]:
        lines.append("- observations reflect scheduler behavior only, not full Jetson inference stress")
    return "\n".join(lines)


def write_summary(
    run_started_at: datetime,
    execution_mode_label: str,
    methodology_note: str,
    aggregated: Dict[str, Dict[str, Any]],
    mode_runs: Dict[str, Dict[str, Any]],
) -> None:
    comparison_table = build_comparison_table(aggregated)
    baseline_exec = mode_runs["baseline"]["execution_label"]
    power_exec = mode_runs["power_aware"]["execution_label"]
    observations = build_observations(aggregated, mode_runs)
    full_three_brain = detect_full_three_brain_execution(mode_runs)
    real_models_used = "yes" if all("real local models" in mode_runs[key]["execution_label"] for key in ["baseline", "power_aware"]) else "no"
    real_frame_input = "yes" if "Static sample frame:" in mode_runs["baseline"]["sample_frame_note"] else "no"
    degraded_notes = []
    for key in ["baseline", "power_aware"]:
        if "mocked" in mode_runs[key]["execution_label"]:
            degraded_notes.append(f"- {key}: {mode_runs[key]['execution_label']}")
    degraded_section = "\n".join(degraded_notes) if degraded_notes else "- none"
    jetson_stress = build_jetson_stress_observations(aggregated, mode_runs)
    summary = f"""# KURAT Runtime Evaluation Summary

## Date/time of run
{run_started_at.astimezone().isoformat(timespec="seconds")}

## Environment used
- OS: {platform.platform()}
- Python: {platform.python_version()}
- Working directory: `{PROJECT_ROOT}`
- Evaluation focus: software/runtime scheduling behavior only
- Sample frame note: {mode_runs['baseline']['sample_frame_note']}

## Execution mode
- Baseline run used: {baseline_exec}
- Power-aware run used: {power_exec}
- Overall label: {execution_mode_label}

## Real models used
{real_models_used}

## Real frame input used
{real_frame_input}

## Full Three-Brain architecture executed
{"yes" if full_three_brain else "no"}

## Methodology
{methodology_note}

{comparison_table}

## Key observations
{observations}

## Runtime errors or degraded modes
{degraded_section}

## Jetson stress observations
{jetson_stress}

## Scenario notes
- Baseline sequential scheduler note: {mode_runs['baseline']['scenarios']['sequential']['metrics']['scheduler']['notes']}
- Baseline burst scheduler note: {mode_runs['baseline']['scenarios']['burst']['metrics']['scheduler']['notes']}
- Power-aware sequential scheduler note: {mode_runs['power_aware']['scenarios']['sequential']['metrics']['scheduler']['notes']}
- Power-aware burst scheduler note: {mode_runs['power_aware']['scenarios']['burst']['metrics']['scheduler']['notes']}

## Honest limitations
This evaluation validates runtime control behavior and scheduling stability. It does not represent hardware-level Jetson power, current draw, thermal behavior, or any physical telemetry unless such telemetry is separately collected outside this workflow.
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        handle.write(summary)


def print_terminal_summary(aggregated: Dict[str, Dict[str, Any]], overall_label: str) -> None:
    baseline = aggregated["baseline"]
    power_aware = aggregated["power_aware"]
    print("KURAT runtime evaluation complete")
    print(f"Execution label: {overall_label}")
    print(
        "Baseline: "
        f"completed={baseline['completed_requests']}/{baseline['total_requests']} "
        f"avg={baseline['average_duration_sec']}s "
        f"max={baseline['max_duration_sec']}s "
        f"overlap={baseline['overlap_conflict_events']}"
    )
    print(
        "Power-aware: "
        f"completed={power_aware['completed_requests']}/{power_aware['total_requests']} "
        f"avg={power_aware['average_duration_sec']}s "
        f"max={power_aware['max_duration_sec']}s "
        f"conflicts={power_aware['overlap_conflict_events']} "
        f"prompt_waits={power_aware['prompt_wait_events']}"
    )
    print("Saved CSV: results/runtime_eval_results.csv")
    print("Saved Markdown summary: results/runtime_eval_summary.md")
    print("Saved raw JSON: results/runtime_eval_raw.json")


def run_evaluation() -> Dict[str, Any]:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    run_started_at = datetime.now().astimezone()
    prompt_sets = {
        "sequential": generate_prompts(SEQUENTIAL_REQUEST_COUNT),
        "burst": generate_prompts(BURST_REQUEST_COUNT),
    }

    mode_runs: Dict[str, Dict[str, Any]] = {}
    fallback_reasons: List[str] = []
    execution_labels: List[str] = []

    for mode_label in ["baseline", "power_aware"]:
        manager = build_manager(mode_label)
        harness, fallback_reason = try_build_real_harness(mode_label, manager)
        if fallback_reason:
            fallback_reasons.append(f"{mode_label}: {fallback_reason}")

        mode_payload = {
            "mode": mode_label,
            "execution_label": harness.execution_label,
            "environment_note": harness.environment_note,
            "sample_frame_note": harness.sample_frame_note,
            "manager_settings": asdict(manager.settings),
            "scenarios": {},
        }
        execution_labels.append(harness.execution_label)

        for scenario_name, prompts in prompt_sets.items():
            mode_payload["scenarios"][scenario_name] = run_scenario(harness, scenario_name, prompts)

        mode_runs[mode_label] = mode_payload

    aggregated = aggregate_mode_metrics(mode_runs)
    overall_label = "real local models"
    if any("mocked" in label for label in execution_labels):
        overall_label = "mocked scheduler validation"
    elif any("real" not in label for label in execution_labels):
        overall_label = "lightweight local execution"

    methodology_note = (
        "Two workloads were executed for both a baseline approximation and the current power-aware mode. "
        f"The sequential workload issued {SEQUENTIAL_REQUEST_COUNT} requests one after another. "
        f"The burst workload launched {BURST_REQUEST_COUNT} requests with a minimal gap to expose queueing and overlap behavior. "
        "The baseline approximation disables power-aware gating in a local wrapper around the existing scheduler. "
        "If the full model stack cannot be constructed, the evaluation falls back to deterministic mocked heavy-task calls that preserve scheduler timing and orchestration structure."
    )
    if fallback_reasons:
        methodology_note += " Fallback notes: " + " ".join(fallback_reasons)

    output_payload = {
        "run_started_at": run_started_at.isoformat(timespec="seconds"),
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "project_root": str(PROJECT_ROOT),
            "cwd": os.getcwd(),
        },
        "overall_execution_label": overall_label,
        "methodology_note": methodology_note,
        "modes": mode_runs,
        "aggregated": aggregated,
    }

    write_csv(mode_runs)
    write_json(output_payload)
    write_summary(
        run_started_at=run_started_at,
        execution_mode_label=overall_label,
        methodology_note=methodology_note,
        aggregated=aggregated,
        mode_runs=mode_runs,
    )
    print_terminal_summary(aggregated, overall_label)
    return output_payload


if __name__ == "__main__":
    run_evaluation()
