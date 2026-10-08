"""
cropping_intensity.py — Insight 1: Cropping intensity analysis across Agricultural Years.
Detects crop cycles on 12-month smoothed NDVI series using scipy.signal.find_peaks.
Classes: Fallow (0), Single (1), Double (2), Triple (3), Perennial (4).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from scipy.signal import find_peaks
import xarray as xr
from PIL import Image

logger = logging.getLogger(__name__)

INTENSITY_CLASSES = {
    0: {"name": "Fallow / Uncultivated", "color": "#CBA670"},
    1: {"name": "Single Crop", "color": "#A8DADC"},
    2: {"name": "Double Crop", "color": "#457B9D"},
    3: {"name": "Triple Crop", "color": "#1D3557"},
    4: {"name": "Perennial (Banana/Tree)", "color": "#2E6B3F"},
}


def compute_cropping_intensity(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    study_area: str = "pilot",
) -> Dict[str, Any]:
    out_dir = outputs_dir / "insights" / "cropping_intensity"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = out_dir / "data.json"

    # Read config parameters
    ci_cfg = cfg.get("insights", {}).get("cropping_intensity", {})
    min_h = ci_cfg.get("peak_min_height", 0.50)
    prom = ci_cfg.get("peak_prominence", 0.20)
    dist = ci_cfg.get("peak_min_distance_months", 3)

    # Load 40m features or 16m features
    feat_path = cache_dir / "features_40m.nc"
    if not feat_path.exists():
        feat_path = cache_dir / "features.nc"
    if not feat_path.exists():
        feat_path = Path("backend/cache/features.nc")

    ds = xr.open_dataset(feat_path)
    ndvi_all = ds["ndvi"].values  # (M, Y, X)
    ny, nx = ndvi_all.shape[1], ndvi_all.shape[2]

    # Load classified map to mask non-cropland
    cls_path = outputs_dir / "final_class.tif"
    if not cls_path.exists():
        cls_path = outputs_dir / "classified.tif"
    if not cls_path.exists():
        cls_path = Path("backend/outputs/classified.tif")

    with rasterio.open(cls_path) as src:
        cls_arr = src.read(1)
        transform = src.transform
        crs = src.crs

    # Only analyze cropland pixels (Paddy=1, Banana=2, OtherVeg=3, Fallow=6)
    # Exclude Builtup=4, Water=5
    cropland_mask = (cls_arr == 1) | (cls_arr == 2) | (cls_arr == 3) | (cls_arr == 6)

    # Analyze latest completed Agricultural Year (months 24:36 if 40m, or 0:12)
    if ndvi_all.shape[0] >= 36:
        ndvi_ay = ndvi_all[24:36]  # AY 2025-26
        ay_label = "AY 2025-26"
    else:
        ndvi_ay = ndvi_all[:12]
        ay_label = "AY 2025-26 (12-month baseline)"

    intensity_map = np.zeros((ny, nx), dtype=np.uint8)

    # Smooth 3-month rolling mean
    w = np.array([0.25, 0.50, 0.25])
    
    # Iterate over valid cropland pixels
    for r in range(ny):
        for c in range(nx):
            if not cropland_mask[r, c]:
                continue

            # Check if Banana class
            if cls_arr[r, c] == 2:
                intensity_map[r, c] = 4  # Perennial
                continue

            ts = ndvi_ay[:, r, c]
            if np.all(np.isnan(ts)) or np.nanmax(ts) < 0.30:
                intensity_map[r, c] = 0  # Fallow
                continue

            # Fill NaNs with interp
            valid_idx = np.where(~np.isnan(ts))[0]
            if len(valid_idx) >= 4:
                ts_clean = np.interp(np.arange(12), valid_idx, ts[valid_idx])
            else:
                ts_clean = np.nan_to_num(ts, nan=0.2)

            # 3-month smoothing
            ts_smooth = np.convolve(ts_clean, w, mode="same")

            peaks, _ = find_peaks(ts_smooth, height=min_h, prominence=prom, distance=dist)
            n_peaks = len(peaks)

            if n_peaks == 0:
                # If high NDVI throughout year with low std -> perennial
                if np.mean(ts_smooth > 0.50) >= 0.75 and np.std(ts_smooth) < 0.12:
                    intensity_map[r, c] = 4
                else:
                    intensity_map[r, c] = 0
            elif n_peaks == 1:
                intensity_map[r, c] = 1
            elif n_peaks == 2:
                intensity_map[r, c] = 2
            else:
                intensity_map[r, c] = 3

    # Statistics breakdown
    total_cropland_px = int(np.sum(cropland_mask))
    if total_cropland_px == 0:
        total_cropland_px = 1

    stats_by_intensity = {}
    chart_data = []

    for k, info in INTENSITY_CLASSES.items():
        px_cnt = int(np.sum(intensity_map == k))
        area_ha = round(px_cnt * 100 / 1e4, 1)
        pct = round((px_cnt / total_cropland_px) * 100.0, 1)
        stats_by_intensity[info["name"]] = {"area_ha": area_ha, "pct": pct, "color": info["color"]}
        chart_data.append({
            "category": info["name"],
            "area_ha": area_ha,
            "pct": pct,
            "color": info["color"],
        })

    # Taluk breakdown
    double_pct = stats_by_intensity["Double Crop"]["pct"]
    single_pct = stats_by_intensity["Single Crop"]["pct"]
    perennial_pct = stats_by_intensity["Perennial (Banana/Tree)"]["pct"]

    # Key finding statement
    if study_area == "taluks":
        key_finding = (
            f"{double_pct:.1f}% of cultivated plains across Ambasamudram & Cheranmahadevi is double-cropped "
            f"(Kar + Pishanam dual-season paddy), with {perennial_pct:.1f}% under perennial banana/orchard canopy."
        )
    else:
        key_finding = (
            f"{double_pct:.1f}% of pilot cropland is double-cropped (Kar & Pishanam paddy cycles), "
            f"supported by perennial banana ({perennial_pct:.1f}%) and single-crop systems ({single_pct:.1f}%)."
        )

    # Save GeoTIFF
    tif_path = out_dir / "cropping_intensity.tif"
    with rasterio.open(
        tif_path, "w",
        driver="GTiff", height=ny, width=nx,
        count=1, dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(intensity_map, 1)

    # Save colored PNG
    png_path = out_dir / "cropping_intensity.png"
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    for k, info in INTENSITY_CLASSES.items():
        h = info["color"].lstrip("#")
        c_rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
        rgb[intensity_map == k] = c_rgb

    Image.fromarray(rgb).save(png_path)

    result = {
        "id": "cropping_intensity",
        "title": "Cropping Intensity",
        "subtitle": f"Multi-cycle harvest frequency ({ay_label})",
        "key_finding": key_finding,
        "ay_label": ay_label,
        "total_cropland_ha": round(total_cropland_px * 100 / 1e4, 1),
        "double_cropped_pct": double_pct,
        "single_cropped_pct": single_pct,
        "perennial_pct": perennial_pct,
        "stats": stats_by_intensity,
        "chart_data": chart_data,
        "legend": [
            {"id": k, "label": info["name"], "color": info["color"]}
            for k, info in INTENSITY_CLASSES.items()
        ],
        "method": (
            "Per-pixel phenological peak detection across 12 monthly NDVI composites (Jun–May). "
            "NDVI is smoothed using a 3-month triangular filter (weights [0.25, 0.50, 0.25]). "
            f"Crop cycles are resolved using scipy.signal.find_peaks with peak height ≥ {min_h}, "
            f"prominence ≥ {prom}, and inter-peak distance ≥ {dist} months. "
            "Non-cropland (urban, open water, upland forest) is strictly excluded."
        ),
        "limitations": (
            "Inter-cropping with short-duration pulses (under 60 days) between major seasons may not reach "
            "the 10-day satellite composite threshold and is not counted as a separate cycle."
        ),
        "raster_url": "/api/insights/cropping_intensity/raster",
        "png_url": "/api/insights/cropping_intensity/png",
    }

    with open(cache_json, "w") as fp:
        json.dump(result, fp, indent=2)

    return result
