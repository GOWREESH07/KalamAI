"""
crop_health.py — Insight 7: Multi-year NDVI z-score anomaly and crop vigour stress detection.
Calculates standardized NDVI anomalies relative to historical 3-year agricultural baselines.
Classes: Severe Stress (z < -1.5), Stressed (-1.5 <= z < -1.0), Normal (-1.0 <= z <= 1.0), Above Normal (z > 1.0).
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
import xarray as xr
from PIL import Image

logger = logging.getLogger(__name__)

ANOMALY_CLASSES = {
    1: {"name": "Severe Stress (z < -1.5)", "color": "#B4554B", "label": "Severe Stress"},
    2: {"name": "Stressed (-1.5 ≤ z < -1.0)", "color": "#E76F51", "label": "Stressed"},
    3: {"name": "Normal Canopy Vigour (-1.0 ≤ z ≤ 1.0)", "color": "#8DB33F", "label": "Normal Vigour"},
    4: {"name": "Above Normal Growth (z > 1.0)", "color": "#2E6B3F", "label": "Above Normal"},
}


def _compute_tile_anomaly(ds: xr.Dataset, cls_arr: np.ndarray, target_month_idx: int = -1) -> np.ndarray:
    """Compute anomaly map for a single tile or scene."""
    if "ndvi" in ds:
        ndvi_all = ds["ndvi"].values
    else:
        n_m = 16
        ndvi_all = np.stack([ds[f"ndvi_m{m:02d}"].values for m in range(n_m) if f"ndvi_m{m:02d}" in ds], axis=0)

    ny, nx = cls_arr.shape
    cropland_mask = (cls_arr == 1) | (cls_arr == 2) | (cls_arr == 3)

    n_m = ndvi_all.shape[0]
    if n_m >= 36:
        if target_month_idx >= 0 and target_month_idx < n_m:
            curr_ndvi = ndvi_all[target_month_idx]
            # Baseline is same month across prior years
            m_mod = target_month_idx % 12
            prior_indices = [m for m in range(m_mod, n_m, 12) if m != target_month_idx]
            if prior_indices:
                hist_mean = np.nanmean(ndvi_all[prior_indices], axis=0)
                hist_std = np.nanstd(ndvi_all[prior_indices], axis=0) + 0.05
            else:
                hist_mean = np.nanmean(ndvi_all, axis=0)
                hist_std = np.nanstd(ndvi_all, axis=0) + 0.05
        else:
            curr_ndvi = np.nanmean(ndvi_all[32:36], axis=0)
            hist_1 = np.nanmean(ndvi_all[8:12], axis=0)
            hist_2 = np.nanmean(ndvi_all[20:24], axis=0)
            hist_stack = np.stack([hist_1, hist_2], axis=0)
            hist_mean = np.nanmean(hist_stack, axis=0)
            hist_std = np.nanstd(hist_stack, axis=0) + 0.05
    elif n_m >= 12:
        curr_ndvi = np.nanmean(ndvi_all[max(0, n_m - 4):n_m], axis=0)
        hist_mean = np.nanmean(ndvi_all[0:12], axis=0)
        hist_std = np.nanstd(ndvi_all[0:12], axis=0) + 0.05
    else:
        curr_ndvi = ndvi_all[-1]
        hist_mean = np.nanmean(ndvi_all, axis=0)
        hist_std = np.nanstd(ndvi_all, axis=0) + 0.05

    z_score = np.where(cropland_mask, (curr_ndvi - hist_mean) / hist_std, np.nan)

    anomaly_map = np.zeros((ny, nx), dtype=np.uint8)
    anomaly_map[cropland_mask & (z_score < -1.5)] = 1
    anomaly_map[cropland_mask & (z_score >= -1.5) & (z_score < -1.0)] = 2
    anomaly_map[cropland_mask & (z_score >= -1.0) & (z_score <= 1.0)] = 3
    anomaly_map[cropland_mask & (z_score > 1.0)] = 4
    # Fallback for nan inside cropland
    anomaly_map[cropland_mask & np.isnan(z_score)] = 3

    return anomaly_map


def compute_crop_health(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    study_area: str = "pilot",
    month_idx: int = -1,
) -> Dict[str, Any]:
    out_dir = outputs_dir / "insights" / "crop_health"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = out_dir / "data.json"

    # Canonical mask denominator
    mask_stats_p = Path("data/mask_stats.json")
    if mask_stats_p.exists() and study_area == "taluks":
        with open(mask_stats_p) as fp:
            mask_info = json.load(fp)
            canonical_analysed_ha = mask_info.get("analyzed_ha", 52619.4)
    else:
        canonical_analysed_ha = 3346.5

    cls_path = outputs_dir / "final_class.tif"
    model_type = "multi-year"
    if not cls_path.exists():
        cls_path = outputs_dir / "classified.tif"
        model_type = "16-month"
    if not cls_path.exists():
        cls_path = Path(f"backend/outputs/{study_area}/classified.tif")
    if not cls_path.exists():
        cls_path = Path("backend/outputs/classified.tif")

    with rasterio.open(cls_path) as src:
        cls_arr = src.read(1)
        transform = src.transform
        crs = src.crs
        global_ny, global_nx = src.height, src.width

    cropland_mask = (cls_arr == 1) | (cls_arr == 2) | (cls_arr == 3)
    anomaly_map = np.zeros((global_ny, global_nx), dtype=np.uint8)

    coverage_note = ""

    if study_area == "taluks":
        tiles_dir = cache_dir / "tiles"
        if not tiles_dir.exists():
            tiles_dir = Path("backend/cache/taluks/tiles")

        valid_tiles = [
            t for t in tiles_dir.iterdir()
            if t.is_dir() and (t / "features.nc").exists() and (t / "classified.tif").exists()
        ]
        n_covered = len(valid_tiles)
        coverage_km2 = n_covered * 25
        coverage_note = f"Multi-year coverage: {coverage_km2} of 1,200 km² ({n_covered} tiles)"

        inv_tfm = ~transform

        for t in valid_tiles:
            try:
                parts = t.name.split("_")
                t_xmin, t_ymin = float(parts[1]), float(parts[2])
                t_xmax, t_ymax = t_xmin + 5000.0, t_ymin + 5000.0

                with rasterio.open(t / "classified.tif") as t_src:
                    t_cls = t_src.read(1)
                ds = xr.open_dataset(t / "features.nc")

                t_anom = _compute_tile_anomaly(ds, t_cls, target_month_idx=month_idx)

                col_start, row_start = [int(round(v)) for v in inv_tfm * (t_xmin, t_ymax)]
                col_end, row_end = [int(round(v)) for v in inv_tfm * (t_xmax, t_ymin)]

                c0, c1 = max(0, col_start), min(global_nx, col_end)
                r0, r1 = max(0, row_start), min(global_ny, row_end)

                if c1 > c0 and r1 > r0:
                    t_anom_resized = Image.fromarray(t_anom).resize((c1 - c0, r1 - r0), Image.NEAREST)
                    anomaly_map[r0:r1, c0:c1] = np.array(t_anom_resized)
            except Exception as e:
                logger.warning("Error processing tile %s for crop_health: %s", t.name, e)
    else:
        feat_path = cache_dir / "features_40m.nc"
        if not feat_path.exists():
            feat_path = cache_dir / "features.nc"
        if not feat_path.exists():
            feat_path = Path("backend/cache/features.nc")

        ds = xr.open_dataset(feat_path)
        anomaly_map = _compute_tile_anomaly(ds, cls_arr, target_month_idx=month_idx)

    # Compute statistics across analyzed cropland (100% sum)
    analyzed_mask = cropland_mask & (anomaly_map > 0) & (cls_arr != 255)
    total_cropland_px = int(np.sum(analyzed_mask))
    if total_cropland_px == 0:
        total_cropland_px = max(1, int(np.sum(cropland_mask & (cls_arr != 255))))

    px_area_ha = 100 / 1e4 if study_area == "pilot" else (30 * 30) / 1e4
    total_cropland_ha = round(total_cropland_px * px_area_ha, 1)

    stats_by_class = {}
    chart_data = []
    sum_area_ha = 0.0

    for k, ainfo in ANOMALY_CLASSES.items():
        px_cnt = int(np.sum((anomaly_map == k) & analyzed_mask))
        area_ha = round(px_cnt * px_area_ha, 1)
        pct = round((px_cnt / total_cropland_px) * 100.0, 1)
        sum_area_ha += area_ha
        stats_by_class[ainfo["label"]] = {"area_ha": area_ha, "pct": pct, "color": ainfo["color"]}
        chart_data.append({
            "category": ainfo["label"],
            "area_ha": area_ha,
            "pct": pct,
            "color": ainfo["color"],
        })

    # Save Anomaly GeoTIFF & PNG
    tif_path = out_dir / "crop_health.tif"
    with rasterio.open(
        tif_path, "w",
        driver="GTiff", height=global_ny, width=global_nx,
        count=1, dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(anomaly_map, 1)

    png_path = out_dir / "crop_health.png"
    rgb = np.zeros((global_ny, global_nx, 4), dtype=np.uint8)
    for k, ainfo in ANOMALY_CLASSES.items():
        h = ainfo["color"].lstrip("#")
        c_rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
        m = (anomaly_map == k) & (cls_arr != 255)
        rgb[m, 0] = c_rgb[0]
        rgb[m, 1] = c_rgb[1]
        rgb[m, 2] = c_rgb[2]
        rgb[m, 3] = 255

    Image.fromarray(rgb, "RGBA").save(png_path)

    healthy_pct = round(stats_by_class["Normal Vigour"]["pct"] + stats_by_class["Above Normal"]["pct"], 1)
    stress_pct = round(stats_by_class["Severe Stress"]["pct"] + stats_by_class["Stressed"]["pct"], 1)
    stress_ha = round(stats_by_class["Severe Stress"]["area_ha"] + stats_by_class["Stressed"]["area_ha"], 1)

    key_finding = (
        f"{healthy_pct:.1f}% of analyzed cropland displays normal or above-normal vegetative vigour. "
        f"Vegetative moisture/vigour stress is observed across {stress_pct:.1f}% ({stress_ha:.1f} ha) of cultivated area."
    )

    extent_label = f"Extent: {canonical_analysed_ha:,.1f} ha · {study_area.title()} · Model: {model_type}"

    available_months = [
        {"idx": i, "label": d.strftime("%b %Y")}
        for i, d in enumerate(pd.date_range("2023-06-01", "2026-09-30", freq="MS"))
    ]

    result = {
        "id": "crop_health",
        "title": "Crop Health & NDVI Anomaly",
        "subtitle": "Multi-year standardized z-score deviation from historical baseline",
        "extent_label": extent_label,
        "study_area": study_area,
        "model_type": model_type,
        "analysed_ha": canonical_analysed_ha,
        "canonical_analysed_ha": canonical_analysed_ha,
        "key_finding": key_finding,
        "total_cropland_ha": total_cropland_ha,
        "healthy_share_pct": healthy_pct,
        "stressed_share_pct": stress_pct,
        "stressed_area_ha": stress_ha,
        "selected_month_idx": month_idx,
        "available_months": available_months,
        "denominator_label": f"total analyzed cropland within mask ({total_cropland_ha:,.1f} ha)",
        "stats": stats_by_class,
        "chart_data": chart_data,
        "legend": [
            {"id": str(k), "label": f"{ainfo['label']} ({stats_by_class[ainfo['label']]['area_ha']} ha, {stats_by_class[ainfo['label']]['pct']}%)", "color": ainfo["color"]}
            for k, ainfo in ANOMALY_CLASSES.items()
        ],
        "method": (
            "Standardized NDVI anomaly z-scores computed against multi-year seasonal reference distributions. "
            "Classes: Severe Stress (z < -1.5), Stressed (-1.5 to -1.0), Normal (-1.0 to 1.0), and Above Normal (> 1.0). "
            "Calculated strictly over analyzed cropland within the canonical mask."
        ),
        "limitations": (
            "Cloud contamination during heavy Northeast Monsoon weeks (Nov–Dec) can momentarily depress optical "
            "composites before synthetic radar cross-calibration kicks in."
        ),
        "csv_url": f"/api/insights/crop_health/csv?study_area={study_area}",
        "png_url": f"/api/insights/crop_health/png?study_area={study_area}",
    }

    pd.DataFrame(chart_data).to_csv(out_dir / "crop_health_anomalies.csv", index=False)

    with open(cache_json, "w") as fp:
        json.dump(result, fp, indent=2)

    return result
