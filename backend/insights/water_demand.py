"""
water_demand.py — Insight 8: Indicative irrigation water requirements based on TNAU / FAO benchmarks.
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
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from PIL import Image

logger = logging.getLogger(__name__)

WATER_BENCHMARKS = {
    "Paddy (Kar Season)": {"mm": 1200, "m3_per_ha": 12000, "color": "#8DB33F"},
    "Paddy (Pishanam Season)": {"mm": 1200, "m3_per_ha": 12000, "color": "#457B9D"},
    "Banana (Annual Perennial)": {"mm": 2000, "m3_per_ha": 20000, "color": "#2E6B3F"},
    "Other Veg / Pulses": {"mm": 500, "m3_per_ha": 5000, "color": "#BFCB8F"},
}


def compute_water_demand(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    study_area: str = "pilot",
) -> Dict[str, Any]:
    out_dir = outputs_dir / "insights" / "water_demand"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = out_dir / "data.json"

    # Load stats or classified raster to get acreage
    stats_csv = outputs_dir / "area_stats.csv"
    if not stats_csv.exists():
        stats_csv = Path("backend/outputs/area_stats.csv")

    paddy_ha = 0.0
    banana_ha = 0.0
    other_ha = 0.0

    if stats_csv.exists():
        df_stats = pd.read_csv(stats_csv)
        for _, row in df_stats.iterrows():
            c_name = str(row["class"]).lower()
            if "paddy" in c_name:
                paddy_ha += float(row["area_ha"])
            elif "banana" in c_name:
                banana_ha += float(row["area_ha"])
            elif "veg" in c_name:
                other_ha += float(row["area_ha"])

    if paddy_ha == 0.0:
        # Read from classified raster
        cls_tif = outputs_dir / "final_class.tif"
        if not cls_tif.exists():
            cls_tif = outputs_dir / "classified.tif"
        if not cls_tif.exists():
            cls_tif = Path("backend/outputs/classified.tif")

        with rasterio.open(cls_tif) as src:
            arr = src.read(1)
            paddy_ha = round(float(np.sum(arr == 1)) * 100 / 1e4, 1)
            banana_ha = round(float(np.sum(arr == 2)) * 100 / 1e4, 1)
            other_ha = round(float(np.sum(arr == 3)) * 100 / 1e4, 1)

    # Double cropping assumption: ~65% of paddy is double cropped (Kar + Pishanam)
    kar_paddy_ha = round(paddy_ha * 0.70, 1)
    pishanam_paddy_ha = paddy_ha

    # Compute Million Cubic Meters (MCM) -> 1 MCM = 1,000,000 m³
    kar_water_mcm = round((kar_paddy_ha * 12000) / 1e6, 2)
    pishanam_water_mcm = round((pishanam_paddy_ha * 12000) / 1e6, 2)
    banana_water_mcm = round((banana_ha * 20000) / 1e6, 2)
    other_water_mcm = round((other_ha * 5000) / 1e6, 2)

    total_water_mcm = round(kar_water_mcm + pishanam_water_mcm + banana_water_mcm + other_water_mcm, 2)

    chart_data = [
        {"component": "Paddy (Kar)", "water_mcm": kar_water_mcm, "area_ha": kar_paddy_ha, "color": "#8DB33F"},
        {"component": "Paddy (Pishanam)", "water_mcm": pishanam_water_mcm, "area_ha": pishanam_paddy_ha, "color": "#457B9D"},
        {"component": "Banana (Annual)", "water_mcm": banana_water_mcm, "area_ha": banana_ha, "color": "#2E6B3F"},
        {"component": "Other Crops", "water_mcm": other_water_mcm, "area_ha": other_ha, "color": "#BFCB8F"},
    ]

    paddy_total_mcm = round(kar_water_mcm + pishanam_water_mcm, 2)
    paddy_water_share = round((paddy_total_mcm / max(0.1, total_water_mcm)) * 100.0, 1)
    banana_water_share = round((banana_water_mcm / max(0.1, total_water_mcm)) * 100.0, 1)

    key_finding = (
        f"Total estimated crop water requirement is {total_water_mcm:.1f} MCM/year. "
        f"Dual-season paddy accounts for {paddy_water_share:.1f}% ({paddy_total_mcm:.1f} MCM) of agricultural water demand, "
        f"while high-intensity perennial banana accounts for {banana_water_share:.1f}% ({banana_water_mcm:.1f} MCM)."
    )

    pd.DataFrame(chart_data).to_csv(out_dir / "water_demand_summary.csv", index=False)

    result = {
        "id": "water_demand",
        "title": "Irrigation Water Demand",
        "subtitle": "Agronomic water budgeting & canal allocation estimates",
        "key_finding": key_finding,
        "total_water_mcm": total_water_mcm,
        "paddy_water_mcm": paddy_total_mcm,
        "banana_water_mcm": banana_water_mcm,
        "paddy_water_share_pct": paddy_water_share,
        "banana_water_share_pct": banana_water_share,
        "chart_data": chart_data,
        "legend": [
            {"id": d["component"], "label": d["component"], "color": d["color"]}
            for d in chart_data
        ],
        "method": (
            "Agronomic crop evapotranspiration (ETc) and percolation budgeting using Tamil Nadu Agricultural University "
            "(TNAU) and FAO-56 irrigation standards. Wetland paddy requires ~1,200 mm/season including land preparation soaking; "
            "perennial banana requires ~2,000 mm/year under tropical wetland regimes."
        ),
        "limitations": (
            "Effective monsoon rainfall offset (typically 400–600 mm during NE Monsoon) reduces net canal irrigation "
            "draft requirements during peak November–December flows."
        ),
        "csv_url": "/api/insights/water_demand/csv",
    }

    with open(cache_json, "w") as fp:
        json.dump(result, fp, indent=2)

    return result
