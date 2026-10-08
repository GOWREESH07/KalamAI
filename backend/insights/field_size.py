"""
field_size.py — Insight 9: Agricultural parcel size distribution & land fragmentation.
Classifies parcels into standard agrarian census categories:
Marginal (<0.5 ha), Small (0.5–1 ha), Semi-Medium (1–2 ha), Medium (2–5 ha), Large (>5 ha).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)

SIZE_BINS = [
    {"id": "marginal", "label": "Marginal (< 0.5 ha)", "min_ha": 0.0, "max_ha": 0.5, "color": "#2A9D8F"},
    {"id": "small", "label": "Small (0.5 – 1.0 ha)", "min_ha": 0.5, "max_ha": 1.0, "color": "#E9C46A"},
    {"id": "semi_medium", "label": "Semi-Medium (1.0 – 2.0 ha)", "min_ha": 1.0, "max_ha": 2.0, "color": "#F4A261"},
    {"id": "medium", "label": "Medium (2.0 – 5.0 ha)", "min_ha": 2.0, "max_ha": 5.0, "color": "#E76F51"},
    {"id": "large", "label": "Large (> 5.0 ha)", "min_ha": 5.0, "max_ha": 9999.0, "color": "#B4554B"},
]


def compute_field_size(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    study_area: str = "pilot",
) -> Dict[str, Any]:
    out_dir = outputs_dir / "insights" / "field_size"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = out_dir / "data.json"

    parcels_gpkg = outputs_dir / "parcels.gpkg"
    if not parcels_gpkg.exists():
        parcels_gpkg = Path("backend/outputs/parcels.gpkg")

    if not parcels_gpkg.exists():
        raise FileNotFoundError("parcels.gpkg not found")

    gdf = gpd.read_file(parcels_gpkg)
    if "area_ha" not in gdf.columns:
        gdf["area_ha"] = gdf.geometry.area / 1e4

    total_parcels = len(gdf)
    total_area_ha = round(float(gdf["area_ha"].sum()), 1)

    bin_stats = []
    chart_data = []

    for b in SIZE_BINS:
        sub = gdf[(gdf["area_ha"] >= b["min_ha"]) & (gdf["area_ha"] < b["max_ha"])]
        cnt = len(sub)
        area = round(float(sub["area_ha"].sum()), 1)
        cnt_pct = round((cnt / max(1, total_parcels)) * 100.0, 1)
        area_pct = round((area / max(0.1, total_area_ha)) * 100.0, 1)

        # Dominant crop in this bin
        cls_col = "class_name" if "class_name" in sub.columns else "final_class"
        if cnt > 0 and cls_col in sub.columns:
            dom_crop = sub[cls_col].mode().iloc[0] if len(sub[cls_col].mode()) > 0 else "Paddy"
        else:
            dom_crop = "Paddy"

        b_data = {
            "bin": b["label"],
            "count": cnt,
            "area_ha": area,
            "count_pct": cnt_pct,
            "area_pct": area_pct,
            "dominant_crop": dom_crop,
            "color": b["color"],
        }
        bin_stats.append(b_data)
        chart_data.append({
            "category": b["label"],
            "count": cnt,
            "area_ha": area,
            "count_pct": cnt_pct,
            "area_pct": area_pct,
            "color": b["color"],
        })

    # Key finding
    smallholder_cnt_pct = bin_stats[0]["count_pct"] + bin_stats[1]["count_pct"]
    smallholder_area_pct = bin_stats[0]["area_pct"] + bin_stats[1]["area_pct"]
    avg_size = round(float(gdf["area_ha"].mean()), 2)
    median_size = round(float(gdf["area_ha"].median()), 2)

    key_finding = (
        f"Extreme land fragmentation: {smallholder_cnt_pct:.1f}% of agricultural parcels are smallholder plots (< 1.0 ha), "
        f"covering {smallholder_area_pct:.1f}% of total farmland. The mean parcel size is {avg_size:.2f} ha (median: {median_size:.2f} ha)."
    )

    pd.DataFrame(bin_stats).to_csv(out_dir / "field_size_distribution.csv", index=False)

    result = {
        "id": "field_size",
        "title": "Field Size & Land Fragmentation",
        "subtitle": "Agrarian parcel morphology & smallholder distribution",
        "key_finding": key_finding,
        "total_parcels": total_parcels,
        "total_area_ha": total_area_ha,
        "average_parcel_ha": avg_size,
        "median_parcel_ha": median_size,
        "smallholder_share_pct": smallholder_cnt_pct,
        "bins": bin_stats,
        "chart_data": chart_data,
        "legend": [
            {"id": b["id"], "label": b["label"], "color": b["color"]}
            for b in SIZE_BINS
        ],
        "method": (
            "Vector parcel boundaries derived from SLIC spatial-spectral superpixel segmentation "
            "are measured in projected UTM coordinates (EPSG:32643). Parcels are categorized into "
            "national agricultural census size brackets."
        ),
        "limitations": (
            "Cadastral boundaries within contiguous multi-plot family ownership holdings with identical "
            "crop sowing dates may be grouped into a single agronomic parcel."
        ),
        "csv_url": "/api/insights/field_size/csv",
    }

    with open(cache_json, "w") as fp:
        json.dump(result, fp, indent=2)

    return result
