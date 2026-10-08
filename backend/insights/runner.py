"""
runner.py — Unified runner for all 9 insight modules.
Executes and caches all insights for both Pilot and Taluks study areas.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any

from .cropping_intensity import compute_cropping_intensity
from .transplanting_calendar import compute_transplanting_calendar
from .irrigation_dependency import compute_irrigation_dependency
from .crop_change import compute_crop_change
from .crop_health import compute_crop_health
from .water_demand import compute_water_demand

logger = logging.getLogger(__name__)

INSIGHT_MODULES = [
    ("cropping_intensity", "01. Cropping Intensity", "Multi-cycle harvest frequency & fallow dynamics", compute_cropping_intensity),
    ("transplanting_calendar", "02. Transplanting Calendar", "SAR-derived inundation & Kar/Pishanam shift detection", compute_transplanting_calendar),
    ("irrigation_dependency", "03. Irrigation Dependency", "Canal & Thamirabarani river buffer decay analysis", compute_irrigation_dependency),
    ("crop_change", "04. Crop Change & Farmland Loss", "Multi-year transition matrix (AY 23-24 vs AY 25-26)", compute_crop_change),
    ("crop_health", "05. Crop Health & Anomaly", "Multi-year NDVI z-score deviation & stress tracking", compute_crop_health),
    ("water_demand", "06. Irrigation Water Demand", "TNAU/FAO agronomic water budgeting (MCM)", compute_water_demand),
]


def run_all_insights(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    study_area: str = "pilot",
) -> Dict[str, Any]:
    start_time = time.time()
    results = {}

    for insight_id, title, subtitle, compute_fn in INSIGHT_MODULES:
        try:
            logger.info("Computing insight %s...", insight_id)
            res = compute_fn(cfg, cache_dir, outputs_dir, study_area)
            results[insight_id] = res
        except Exception as e:
            logger.error("Failed computing %s: %s", insight_id, e)
            results[insight_id] = {
                "id": insight_id,
                "title": title,
                "subtitle": subtitle,
                "error": str(e),
                "status": "failed",
            }

    duration = round(time.time() - start_time, 2)
    logger.info("All insights computed in %.2fs", duration)
    return results


def get_all_insights_summary(
    cfg: dict,
    outputs_dir: Path,
) -> list[dict]:
    summary = []
    for insight_id, title, subtitle, _ in INSIGHT_MODULES:
        cache_json = outputs_dir / "insights" / insight_id / "data.json"
        if cache_json.exists():
            try:
                import json
                with open(cache_json) as fp:
                    data = json.load(fp)
                    summary.append({
                        "id": insight_id,
                        "title": data.get("title", title),
                        "subtitle": data.get("subtitle", subtitle),
                        "key_finding": data.get("key_finding", ""),
                        "status": "ready",
                    })
                    continue
            except Exception:
                pass
        summary.append({
            "id": insight_id,
            "title": title,
            "subtitle": subtitle,
            "key_finding": "Ready to compute",
            "status": "pending",
        })
    return summary
