"""
field_size.py — Insight 9: Segment size distribution (image-derived).
Classifies image-derived parcel segments into standard agricultural size brackets:
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
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from PIL import Image

logger = logging.getLogger(__name__)

SIZE_BINS = [
    {"id": "marginal", "label": "Marginal (< 0.5 ha)", "min_ha": 0.0, "max_ha": 0.5, "val": 1, "color": "#2A9D8F"},
    {"id": "small", "label": "Small (0.5 – 1.0 ha)", "min_ha": 0.5, "max_ha": 1.0, "val": 2, "color": "#E9C46A"},
    {"id": "semi_medium", "label": "Semi-Medium (1.0 – 2.0 ha)", "min_ha": 1.0, "max_ha": 2.0, "val": 3, "color": "#F4A261"},
    {"id": "medium", "label": "Medium (2.0 – 5.0 ha)", "min_ha": 2.0, "max_ha": 5.0, "val": 4, "color": "#E76F51"},
    {"id": "large", "label": "Large (> 5.0 ha)", "min_ha": 5.0, "max_ha": 9999.0, "val": 5, "color": "#B4554B"},
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
    model_type = "multi-year"
    if not parcels_gpkg.exists():
        parcels_gpkg = Path(f"backend/outputs/{study_area}/parcels.gpkg")
    if not parcels_gpkg.exists():
        parcels_gpkg = Path("backend/outputs/parcels.gpkg")
        model_type = "16-month"

    if not parcels_gpkg.exists():
        raise FileNotFoundError(f"parcels.gpkg not found for study area: {study_area}")

    gdf = gpd.read_file(parcels_gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=32643)

    if "area_ha" not in gdf.columns:
        gdf["area_ha"] = gdf.geometry.area / 1e4

    total_parcels = len(gdf)
    total_area_ha = round(float(gdf["area_ha"].sum()), 1)

    bin_stats = []
    chart_data = []
    sum_area_ha = 0.0

    # Assign size bin id to each parcel for rasterization
    bin_assignments = np.zeros(len(gdf), dtype=np.uint8)

    for idx, b in enumerate(SIZE_BINS, start=1):
        mask = (gdf["area_ha"] >= b["min_ha"]) & (gdf["area_ha"] < b["max_ha"])
        bin_assignments[mask] = b["val"]
        sub = gdf[mask]
        cnt = len(sub)
        area = round(float(sub["area_ha"].sum()), 1)
        cnt_pct = round((cnt / max(1, total_parcels)) * 100.0, 1)
        area_pct = round((area / max(0.1, total_area_ha)) * 100.0, 1)
        sum_area_ha += area

        cls_col = "class_name" if "class_name" in sub.columns else "final_class"
        if cnt > 0 and cls_col in sub.columns:
            modes = sub[cls_col].mode()
            dom_crop = str(modes.iloc[0]) if len(modes) > 0 else "Paddy"
        else:
            dom_crop = "Paddy"

        b_data = {
            "id": b["id"],
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
            "area_ha": area,
            "count": cnt,
            "pct": area_pct,
            "count_pct": cnt_pct,
            "area_pct": area_pct,
            "color": b["color"],
        })

    gdf["size_bin_val"] = bin_assignments

    # Rasterize size categories
    bounds = gdf.total_bounds
    res = 10.0
    nx = int((bounds[2] - bounds[0]) / res)
    ny = int((bounds[3] - bounds[1]) / res)
    nx = max(50, min(nx, 2000))
    ny = max(50, min(ny, 2000))
    tfm = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], nx, ny)

    shapes_b = [(geom, val) for geom, val in zip(gdf.geometry, gdf["size_bin_val"])]
    size_raster = rasterize(shapes_b, out_shape=(ny, nx), transform=tfm, fill=0, dtype=np.uint8)

    # Render PNG
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    for b in SIZE_BINS:
        h = b["color"].lstrip("#")
        c_rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
        rgb[size_raster == b["val"]] = c_rgb

    png_path = out_dir / "field_size.png"
    Image.fromarray(rgb).save(png_path)

    # Save GeoTIFF
    tif_path = out_dir / "field_size.tif"
    with rasterio.open(
        tif_path, "w",
        driver="GTiff", height=ny, width=nx,
        count=1, dtype=np.uint8,
        crs=CRS.from_epsg(32643), transform=tfm, compress="deflate",
    ) as dst:
        dst.write(size_raster, 1)

    avg_size = round(float(gdf["area_ha"].mean()), 2)
    median_size = round(float(gdf["area_ha"].median()), 2)
    small_cnt_pct = round(bin_stats[0]["count_pct"] + bin_stats[1]["count_pct"], 1)
    small_area_pct = round(bin_stats[0]["area_pct"] + bin_stats[1]["area_pct"], 1)

    # Key finding (strictly objective numbers without exaggeration)
    key_finding = (
        f"{small_cnt_pct:.1f}% of image-derived field segments ({small_area_pct:.1f}% of agricultural area) "
        f"are below 1.0 ha. The mean segment size is {avg_size:.2f} ha with a median of {median_size:.2f} ha."
    )

    pd.DataFrame(bin_stats).to_csv(out_dir / "field_size_distribution.csv", index=False)

    extent_label = f"Extent: {total_area_ha:,.1f} ha · {study_area.title()} · Model: {model_type}"

    result = {
        "id": "field_size",
        "title": "Segment Size Distribution (Image-Derived)",
        "subtitle": "Agrarian parcel morphology & smallholder distribution",
        "extent_label": extent_label,
        "study_area": study_area,
        "model_type": model_type,
        "key_finding": key_finding,
        "total_parcels": total_parcels,
        "total_area_ha": total_area_ha,
        "average_parcel_ha": avg_size,
        "median_parcel_ha": median_size,
        "smallholder_count_pct": small_cnt_pct,
        "smallholder_area_pct": small_area_pct,
        "denominator_label": f"total agricultural area ({total_area_ha:,.1f} ha)",
        "bins": bin_stats,
        "chart_data": chart_data,
        "legend": [
            {"id": b["id"], "label": f"{b.get('label', b.get('bin'))} ({b['area_pct']}% area, {b['count_pct']}% count)", "color": b["color"]}
            for b in bin_stats
        ],
        "method": (
            "Vector parcel boundaries derived from SLIC spatial-spectral superpixel segmentation "
            "are measured in projected UTM coordinates (EPSG:32643). Parcels are categorized into "
            "national agricultural census holding size brackets."
        ),
        "limitations": (
            "Segment sizes depend on the SLIC segmentation scale and are not cadastral holdings."
        ),
        "csv_url": f"/api/insights/field_size/csv?study_area={study_area}",
        "png_url": f"/api/insights/field_size/png?study_area={study_area}",
    }

    with open(cache_json, "w") as fp:
        json.dump(result, fp, indent=2)

    return result
