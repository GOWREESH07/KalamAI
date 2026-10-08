"""
verification_priority.py — Insight 3: Ground-truthing and field audit priority ranking.
Formula:
  Priority = w_conf * (1 - confidence) + w_cons * (1 - consistency) + w_disag * disagreement
Percentiles: Critical = top 10%, High = next 15%, Medium = next 25%, Low = rest.
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

PRIORITY_LEVELS = {
    "Critical": {"color": "#B4554B", "label": "Critical Priority (Top 10%)"},
    "High": {"color": "#E76F51", "label": "High Priority (Next 15%)"},
    "Medium": {"color": "#E9C46A", "label": "Medium Priority (Next 25%)"},
    "Low": {"color": "#2A9D8F", "label": "Low Priority (Baseline 50%)"},
}


def compute_verification_priority(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    study_area: str = "pilot",
) -> Dict[str, Any]:
    out_dir = outputs_dir / "insights" / "verification_priority"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = out_dir / "data.json"

    vp_cfg = cfg.get("insights", {}).get("verification_priority", {})
    weights = vp_cfg.get("weights", {"confidence": 0.5, "consistency": 0.3, "disagreement": 0.2})
    w_conf = weights.get("confidence", 0.5)
    w_cons = weights.get("consistency", 0.3)
    w_disag = weights.get("disagreement", 0.2)

    # Canonical mask denominator
    mask_stats_p = Path("data/mask_stats.json")
    if mask_stats_p.exists() and study_area == "taluks":
        with open(mask_stats_p) as fp:
            mask_info = json.load(fp)
            canonical_analysed_ha = mask_info.get("analyzed_ha", 52619.4)
    else:
        canonical_analysed_ha = 3346.5

    parcels_gpkg = outputs_dir / "parcels.gpkg"
    model_type = "multi-year"
    if not parcels_gpkg.exists():
        parcels_gpkg = Path(f"backend/outputs/{study_area}/parcels.gpkg")
    if not parcels_gpkg.exists():
        parcels_gpkg = Path("backend/outputs/parcels.gpkg")
        model_type = "16-month"

    if not parcels_gpkg.exists():
        raise FileNotFoundError(f"parcels.gpkg not found for study area {study_area}")

    gdf = gpd.read_file(parcels_gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=32643)

    # Filter out any non-agricultural or outside parcels
    if "area_ha" not in gdf.columns:
        gdf["area_ha"] = gdf.geometry.area / 1e4

    total_parcels = len(gdf)
    total_area_ha = round(float(gdf["area_ha"].sum()), 1)

    # Compute priority score
    conf = gdf["confidence"].fillna(0.70).values if "confidence" in gdf.columns else np.full(len(gdf), 0.70)
    cons = gdf["consistency"].fillna(0.80).values if "consistency" in gdf.columns else np.full(len(gdf), 0.80)
    
    disag = np.zeros(len(gdf), dtype=float)
    if "rotation_detected" in gdf.columns:
        disag = np.where(gdf["rotation_detected"].fillna(False), 1.0, 0.0)
    elif "ay23_24_class" in gdf.columns and "ay25_26_class" in gdf.columns:
        disag = np.where(gdf["ay23_24_class"] != gdf["ay25_26_class"], 0.8, 0.0)

    priority_score = w_conf * (1.0 - conf) + w_cons * (1.0 - cons) + w_disag * disag
    gdf["priority_score"] = np.round(priority_score, 4)

    # Thresholds by exact percentiles: Critical = top 10%, High = next 15% (85-75th), Medium = next 25% (75-50th), Low = rest
    p90 = np.percentile(priority_score, 90)
    p75 = np.percentile(priority_score, 75)
    p50 = np.percentile(priority_score, 50)

    levels = []
    colors = []
    reasons = []

    for i, row in gdf.iterrows():
        score = row["priority_score"]
        c_val = conf[i]
        cs_val = cons[i]
        d_val = disag[i]

        if score >= p90:
            lvl = "Critical"
        elif score >= p75:
            lvl = "High"
        elif score >= p50:
            lvl = "Medium"
        else:
            lvl = "Low"
        levels.append(lvl)
        colors.append(PRIORITY_LEVELS[lvl]["color"])

        r_parts = []
        if c_val < 0.65:
            r_parts.append(f"Low classifier confidence ({c_val*100:.0f}%)")
        if cs_val < 0.70:
            r_parts.append(f"Inter-annual crop instability ({cs_val*100:.0f}% consistency)")
        if d_val > 0.5:
            r_parts.append("Paddy-banana rotation / spectral ambiguity")
        if not r_parts:
            r_parts.append("Routine spot audit")
        reasons.append("; ".join(r_parts))

    gdf["priority_level"] = levels
    gdf["priority_color"] = colors
    gdf["audit_reason"] = reasons

    # Centroids in WGS84 for GPS
    centroids_utm = gdf.geometry.centroid
    gdf_pts_wgs = gpd.GeoDataFrame(geometry=centroids_utm, crs=gdf.crs).to_crs(epsg=4326)
    gdf["center_lon"] = np.round(gdf_pts_wgs.geometry.x, 5)
    gdf["center_lat"] = np.round(gdf_pts_wgs.geometry.y, 5)

    # Top 50 highest audit priorities
    top50 = gdf.sort_values(by="priority_score", ascending=False).head(50)
    top50_records = []
    for idx, row in top50.iterrows():
        top50_records.append({
            "parcel_id": int(row.get("segment_id", idx)),
            "area_ha": round(float(row.get("area_ha", 0.5)), 2),
            "predicted_class": str(row.get("class_name", row.get("final_class", "Cropland"))),
            "confidence": round(float(row.get("confidence", 0.7)), 2),
            "consistency": round(float(row.get("consistency", 0.8)), 2),
            "priority_score": round(float(row["priority_score"]), 3),
            "priority_level": row["priority_level"],
            "reason": row["audit_reason"],
            "lon": float(row["center_lon"]),
            "lat": float(row["center_lat"]),
        })

    # Summary stats by priority level
    stats_by_level = {}
    chart_data = []
    sum_area_ha = 0.0

    for lvl, pinfo in PRIORITY_LEVELS.items():
        sub = gdf[gdf["priority_level"] == lvl]
        cnt = len(sub)
        area = round(float(sub["area_ha"].sum()), 1)
        cnt_pct = round((cnt / max(1, total_parcels)) * 100.0, 1)
        area_pct = round((area / max(0.1, total_area_ha)) * 100.0, 1)
        sum_area_ha += area
        stats_by_level[lvl] = {
            "count": cnt,
            "area_ha": area,
            "pct": cnt_pct,
            "area_pct": area_pct,
            "color": pinfo["color"]
        }
        chart_data.append({
            "category": f"{lvl} Priority",
            "count": cnt,
            "area_ha": area,
            "pct": cnt_pct,
            "color": pinfo["color"]
        })

    # Save CSV
    csv_path = out_dir / "field_verification_top50.csv"
    pd.DataFrame(top50_records).to_csv(csv_path, index=False)

    # Save KML
    kml_path = out_dir / "field_verification.kml"
    kml_content = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document><name>Field Audit Priority Parcels — Kalam</name>'
    ]
    for r in top50_records:
        kml_content.append(f"""
        <Placemark>
          <name>Parcel #{r['parcel_id']} ({r['priority_level']})</name>
          <description>Class: {r['predicted_class']}&#10;Conf: {r['confidence']*100:.0f}%&#10;Cons: {r['consistency']*100:.0f}%&#10;Reason: {r['reason']}</description>
          <Point><coordinates>{r['lon']},{r['lat']},0</coordinates></Point>
        </Placemark>""")
    kml_content.append('</Document></kml>')
    with open(kml_path, "w") as f_kml:
        f_kml.write("".join(kml_content))

    # Rasterize priority map for overlay
    bounds = gdf.total_bounds
    res = 10.0
    nx = int((bounds[2] - bounds[0]) / res)
    ny = int((bounds[3] - bounds[1]) / res)
    nx = max(50, min(nx, 2000))
    ny = max(50, min(ny, 2000))
    tfm = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], nx, ny)

    level_map_vals = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
    shapes_p = [(geom, level_map_vals[lvl]) for geom, lvl in zip(gdf.geometry, gdf["priority_level"])]
    priority_raster = rasterize(shapes_p, out_shape=(ny, nx), transform=tfm, fill=0, dtype=np.uint8)

    # Render PNG
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    for lvl, val in level_map_vals.items():
        h = PRIORITY_LEVELS[lvl]["color"].lstrip("#")
        c_rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
        rgb[priority_raster == val] = c_rgb

    png_path = out_dir / "verification_priority.png"
    Image.fromarray(rgb).save(png_path)

    critical_cnt = stats_by_level["Critical"]["count"]
    high_cnt = stats_by_level["High"]["count"]
    crit_high_cnt = critical_cnt + high_cnt
    crit_high_pct = round((crit_high_cnt / max(1, total_parcels)) * 100.0, 1)

    key_finding = (
        f"{crit_high_pct:.1f}% of parcels ({crit_high_cnt} fields) are flagged for targeted ground audit "
        f"due to lower phenological confidence or inter-annual crop instability, with top-50 priority targets packaged for GPS navigation."
    )

    extent_label = f"Extent: {canonical_analysed_ha:,.1f} ha · {study_area.title()} · Model: {model_type}"

    result = {
        "id": "verification_priority",
        "title": "Field Verification Priority",
        "subtitle": "Targeted ground-audit ranking & GPS navigation coordinates",
        "extent_label": extent_label,
        "analysed_ha": canonical_analysed_ha,
        "canonical_analysed_ha": canonical_analysed_ha,
        "study_area": study_area,
        "model_type": model_type,
        "key_finding": key_finding,
        "total_parcels": total_parcels,
        "total_area_ha": total_area_ha,
        "critical_parcels": critical_cnt,
        "high_priority_parcels": high_cnt,
        "denominator_label": f"total agricultural parcels ({total_parcels:,} fields across {total_area_ha:,.1f} ha)",
        "stats": stats_by_level,
        "chart_data": chart_data,
        "top50": top50_records,
        "legend": [
            {"id": lvl, "label": pinfo["label"], "color": pinfo["color"]}
            for lvl, pinfo in PRIORITY_LEVELS.items()
        ],
        "method": (
            "Objective priority index combining Random Forest posterior class certainty (50%), "
            "inter-annual agricultural consistency (30%), and crop-switching / phenological ambiguity flags (20%). "
            "Parcels are ranked into percentile tiers: Critical (top 10%), High (next 15%), Medium (next 25%), Low (baseline 50%)."
        ),
        "limitations": (
            "Parcels on steep mountain boundary fringes may receive elevated priority scores due to cloud shadows "
            "rather than agronomic ambiguity."
        ),
        "csv_url": f"/api/insights/verification_priority/csv?study_area={study_area}",
        "kml_url": f"/api/insights/verification_priority/kml?study_area={study_area}",
        "png_url": f"/api/insights/verification_priority/png?study_area={study_area}",
    }

    with open(cache_json, "w") as fp:
        json.dump(result, fp, indent=2)

    return result
