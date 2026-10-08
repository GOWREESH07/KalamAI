"""
water_demand.py — Insight 8: Indicative gross irrigation requirement based on TNAU / FAO benchmarks.
Paddy: 1,200 mm per crop cycle (12,000 m³/ha)
Banana: 2,000 mm per year (20,000 m³/ha)
Other Crops: 500 mm per cycle (5,000 m³/ha)
Calculates seasonal and annual water demand in Million Cubic Meters (MCM).
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
from PIL import Image

logger = logging.getLogger(__name__)


def compute_water_demand(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    study_area: str = "pilot",
) -> Dict[str, Any]:
    out_dir = outputs_dir / "insights" / "water_demand"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = out_dir / "data.json"

    wd_cfg = cfg.get("insights", {}).get("water_demand", {})
    paddy_mm = wd_cfg.get("paddy_water_mm_per_season", 1200)
    banana_mm = wd_cfg.get("banana_water_mm_per_year", 2000)
    other_mm = wd_cfg.get("other_crops_mm_per_season", 500)

    # 1. Load classified raster to get spatial water demand grid
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
        ny, nx = cls_arr.shape
        transform = src.transform
        crs = src.crs

    paddy_px = int(np.sum(cls_arr == 1))
    banana_px = int(np.sum(cls_arr == 2))
    other_px = int(np.sum(cls_arr == 3))

    paddy_ha = round(paddy_px * 100 / 1e4, 1)
    banana_ha = round(banana_px * 100 / 1e4, 1)
    other_ha = round(other_px * 100 / 1e4, 1)
    total_cropland_ha = round(paddy_ha + banana_ha + other_ha, 1)

    # Kar paddy (~65% of paddy area) + Pishanam paddy (100% of paddy area)
    kar_paddy_ha = round(paddy_ha * 0.65, 1)
    pishanam_paddy_ha = paddy_ha

    kar_water_mcm = round((kar_paddy_ha * (paddy_mm * 10)) / 1e6, 2)
    pishanam_water_mcm = round((pishanam_paddy_ha * (paddy_mm * 10)) / 1e6, 2)
    banana_water_mcm = round((banana_ha * (banana_mm * 10)) / 1e6, 2)
    other_water_mcm = round((other_ha * (other_mm * 10)) / 1e6, 2)

    total_water_mcm = round(kar_water_mcm + pishanam_water_mcm + banana_water_mcm + other_water_mcm, 2)
    if total_water_mcm == 0:
        total_water_mcm = 0.01

    chart_data = [
        {
            "category": "Paddy (Kar Season)",
            "water_mcm": kar_water_mcm,
            "area_ha": kar_paddy_ha,
            "pct": round((kar_water_mcm / total_water_mcm) * 100.0, 1),
            "color": "#8DB33F",
            "rate_desc": f"{paddy_mm:,} mm ({paddy_mm*10:,} m³/ha)",
        },
        {
            "category": "Paddy (Pishanam Season)",
            "water_mcm": pishanam_water_mcm,
            "area_ha": pishanam_paddy_ha,
            "pct": round((pishanam_water_mcm / total_water_mcm) * 100.0, 1),
            "color": "#457B9D",
            "rate_desc": f"{paddy_mm:,} mm ({paddy_mm*10:,} m³/ha)",
        },
        {
            "category": "Banana (Annual Perennial)",
            "water_mcm": banana_water_mcm,
            "area_ha": banana_ha,
            "pct": round((banana_water_mcm / total_water_mcm) * 100.0, 1),
            "color": "#2E6B3F",
            "rate_desc": f"{banana_mm:,} mm ({banana_mm*10:,} m³/ha/yr)",
        },
        {
            "category": "Other Crops / Pulses",
            "water_mcm": other_water_mcm,
            "area_ha": other_ha,
            "pct": round((other_water_mcm / total_water_mcm) * 100.0, 1),
            "color": "#BFCB8F",
            "rate_desc": f"{other_mm:,} mm ({other_mm*10:,} m³/ha)",
        },
    ]

    paddy_total_mcm = round(kar_water_mcm + pishanam_water_mcm, 2)
    paddy_water_share = round((paddy_total_mcm / total_water_mcm) * 100.0, 1)
    banana_water_share = round((banana_water_mcm / total_water_mcm) * 100.0, 1)

    # Spatial Water Demand Density Map (m³/100m² pixel)
    # Paddy (dual) = 24,000 m³/ha -> 240 m³/pixel
    # Banana = 20,000 m³/ha -> 200 m³/pixel
    # Other = 5,000 m³/ha -> 50 m³/pixel
    demand_raster = np.zeros((ny, nx), dtype=np.float32)
    demand_raster[cls_arr == 1] = 240.0
    demand_raster[cls_arr == 2] = 200.0
    demand_raster[cls_arr == 3] = 50.0

    # Save GeoTIFF
    tif_path = out_dir / "water_demand.tif"
    with rasterio.open(
        tif_path, "w",
        driver="GTiff", height=ny, width=nx,
        count=1, dtype=np.float32,
        crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(demand_raster, 1)

    # Render PNG
    png_path = out_dir / "water_demand.png"
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    rgb[cls_arr == 1] = [141, 179, 63]
    rgb[cls_arr == 2] = [46, 107, 63]
    rgb[cls_arr == 3] = [191, 203, 143]
    rgb[cls_arr == 5] = [58, 124, 165]
    Image.fromarray(rgb).save(png_path)

    key_finding = (
        f"Total indicative gross irrigation water requirement is {total_water_mcm:.1f} MCM/year. "
        f"Dual-season paddy accounts for {paddy_water_share:.1f}% ({paddy_total_mcm:.1f} MCM) of total demand, "
        f"while perennial banana accounts for {banana_water_share:.1f}% ({banana_water_mcm:.1f} MCM)."
    )

    pd.DataFrame(chart_data).to_csv(out_dir / "water_demand_summary.csv", index=False)

    extent_label = f"Extent: {total_cropland_ha:,.1f} ha · {study_area.title()} · Model: {model_type}"

    result = {
        "id": "water_demand",
        "title": "Indicative Gross Irrigation Requirement",
        "subtitle": "Agronomic volumetric water budgeting & canal allocation estimates",
        "extent_label": extent_label,
        "study_area": study_area,
        "model_type": model_type,
        "key_finding": key_finding,
        "total_water_mcm": total_water_mcm,
        "paddy_water_mcm": paddy_total_mcm,
        "banana_water_mcm": banana_water_mcm,
        "paddy_water_share_pct": paddy_water_share,
        "banana_water_share_pct": banana_water_share,
        "total_cropland_ha": total_cropland_ha,
        "denominator_label": f"total gross irrigation requirement ({total_water_mcm:.1f} MCM)",
        "chart_data": chart_data,
        "legend": [
            {"id": d["category"], "label": f"{d['category']} ({d['pct']}%)", "color": d["color"]}
            for d in chart_data
        ],
        "method": (
            "area × standard seasonal water requirement (config values, to be verified against TNAU/FAO-56); "
            "rainfall offset not applied. Wetland paddy: 1,200 mm/season; perennial banana: 2,000 mm/year; other crops: 500 mm."
        ),
        "limitations": (
            "Rainfall offset (effective precipitation during Northeast Monsoon) and groundwater conjunctive use "
            "are not subtracted in this gross baseline estimate."
        ),
        "csv_url": f"/api/insights/water_demand/csv?study_area={study_area}",
        "png_url": f"/api/insights/water_demand/png?study_area={study_area}",
    }

    with open(cache_json, "w") as fp:
        json.dump(result, fp, indent=2)

    return result
