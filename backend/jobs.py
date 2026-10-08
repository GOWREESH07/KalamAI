"""
jobs.py — Background pipeline runner with per-step status, logs, timings.
"""
from __future__ import annotations
import logging
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from .pipeline.aoi import load_config, validate_aoi
from .pipeline.acquire import run_acquire
from .pipeline.preprocess import run_preprocess
from .pipeline.features import run_features
from .pipeline.weak_labels import generate_weak_labels
from .pipeline.segment import run_segment
from .pipeline.classify import run_classify
from .pipeline.export import run_export

logger = logging.getLogger(__name__)

STEPS = ["acquire", "preprocess", "features", "labels", "segment", "classify", "export"]


class StepStatus:
    def __init__(self, name: str):
        self.name = name
        self.status: str = "pending"  # pending / running / done / failed
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.logs: list[str] = []
        self.error: str | None = None

    @property
    def duration(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
            "log_tail": self.logs[-20:],
            "error": self.error,
        }


class PipelineJob:
    def __init__(self, cfg: dict, cache_dir: Path, outputs_dir: Path):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.outputs_dir = outputs_dir
        self.steps: dict[str, StepStatus] = {s: StepStatus(s) for s in STEPS}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.results: dict = {}

    def _log(self, step_name: str, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        logger.info("%s: %s", step_name, msg)
        with self._lock:
            self.steps[step_name].logs.append(entry)

    def _run_steps(self, from_step: str = "acquire"):
        start_idx = STEPS.index(from_step)

        # Invalidate cached artifacts from from_step onward
        step_outputs = {
            "acquire": [self.cache_dir / "s2_raw.nc", self.cache_dir / "s1_raw.nc"],
            "preprocess": [self.cache_dir / "s2_monthly.nc", self.cache_dir / "s1_monthly.nc"],
            "features": [self.cache_dir / "features.nc"],
            "labels": [self.cache_dir / "labels.csv"],
            "segment": [self.outputs_dir / "segments.gpkg"],
            "classify": [
                self.outputs_dir / "classified.tif",
                self.outputs_dir / "metrics.json",
                self.outputs_dir / "area_stats.csv",
                self.outputs_dir / "parcels.gpkg",
                self.outputs_dir / "parcels.geojson",
            ],
            "export": [
                self.outputs_dir / "thematic_map.png",
                self.outputs_dir / "thematic_map.pdf",
                self.outputs_dir / "classified_rgb.png",
            ],
        }
        for s in STEPS[start_idx:]:
            for f in step_outputs.get(s, []):
                if f.exists():
                    try:
                        f.unlink()
                    except Exception as e:
                        logger.warning("Could not unlink %s: %s", f, e)

        for step_name in STEPS[start_idx:]:
            with self._lock:
                self.steps[step_name].status = "running"
                self.steps[step_name].started_at = time.time()

            cb = lambda msg, sn=step_name: self._log(sn, msg)

            try:
                if step_name == "acquire":
                    result = run_acquire(self.cfg, self.cache_dir, cb)
                elif step_name == "preprocess":
                    result = run_preprocess(self.cfg, self.cache_dir, cb)
                elif step_name == "features":
                    result = run_features(self.cfg, self.cache_dir, cb)
                elif step_name == "labels":
                    manual_lbl_path = self.cache_dir / "manual_labels.geojson"
                    result = generate_weak_labels(
                        self.cfg, self.cache_dir,
                        manual_lbl_path if manual_lbl_path.exists() else None,
                        cb
                    )
                elif step_name == "segment":
                    result = run_segment(self.cfg, self.cache_dir, self.outputs_dir, cb)
                elif step_name == "classify":
                    result = run_classify(self.cfg, self.cache_dir, self.outputs_dir, cb)
                elif step_name == "export":
                    result = run_export(self.cfg, self.cache_dir, self.outputs_dir, cb)
                else:
                    result = {}

                self.results[step_name] = result
                with self._lock:
                    self.steps[step_name].status = "done"
                    self.steps[step_name].finished_at = time.time()
                self._log(step_name, f"Step done in {self.steps[step_name].duration:.1f}s")
                time.sleep(0.4)

            except Exception as e:
                tb = traceback.format_exc()
                with self._lock:
                    self.steps[step_name].status = "failed"
                    self.steps[step_name].finished_at = time.time()
                    self.steps[step_name].error = str(e)
                    self.steps[step_name].logs.append(f"ERROR: {e}")
                    self.steps[step_name].logs.append(tb)
                logger.error("Step %s failed: %s", step_name, tb)
                # Stop pipeline on failure
                break

    def run(self, from_step: str = "acquire"):
        if self._thread and self._thread.is_alive():
            return {"status": "already_running"}
        # Reset statuses for steps that will run
        start_idx = STEPS.index(from_step)
        for s in STEPS[start_idx:]:
            self.steps[s] = StepStatus(s)
        self._thread = threading.Thread(
            target=self._run_steps, args=(from_step,), daemon=True
        )
        self._thread.start()
        return {"status": "started", "from_step": from_step}

    def get_status(self) -> list[dict]:
        with self._lock:
            return [s.to_dict() for s in self.steps.values()]

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
