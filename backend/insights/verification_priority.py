"""
verification_priority.py — Insight 3: Ground-truthing and field audit priority ranking.
Formula:
  Priority = w_conf * (1 - confidence) + w_cons * (1 - consistency) + w_disag * disagreement
Exports Top-50 list, CSV, KMZ, and priority choropleth map.
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
    "Critical": {"color": "#B4554B", "min_score": 0.55},
    "High": {"color": "#E76F51", "min_score": 0.40},
    "Medium": {"color": "#E9C46A", "min_score": 0.25},
    "Low": {"color": "#2A9D8F", "min_score": 0.0},
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

    parcels_gpkg = outputs_dir / "parcels.gpkg"
    if not parcels_gpkg.exists():
        parcels_gpkg = Path("backend/outputs/parcels.gpkg")

    if not parcels_gpkg.exists():
        raise FileNotFoundError("parcels.gpkg not found")

    gdf = gpd.read_file(parcels_gpkg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=32643)

    # Compute priority components
    # 1. Confidence (0..1)
    conf = gdf["confidence"].fillna(0.70).values if "confidence" in gdf.columns else np.full(len(gdf), 0.70)
    # 2. Consistency (0..1)
    cons = gdf["consistency"].fillna(0.80).values if "consistency" in gdf.columns else np.full(len(gdf), 0.80)
    # 3. Disagreement / Rotation flag
    disag = np.zeros(len(gdf), dtype=float)
    if "rotation_detected" in gdf.columns:
        disag = np.where(gdf["rotation_detected"].fillna(False), 1.0, 0.0)
    elif "ay23_24_class" in gdf.columns and "ay25_26_class" in gdf.columns:
        disag = np.where(gdf["ay23_24_class"] != gdf["ay25_26_class"], 0.8, 0.0)

    # Score
    priority_score = w_conf * (1.0 - conf) + w_cons * (1.0 - cons) + w_disag * disag
    gdf["priority_score"] = np.round(priority_score, 4)

    # Assign Priority Level
    levels = []
    colors = []
    reasons = []
    for i, row in gdf.iterrows():
        score = row["priority_score"]
        c_val = conf[i]
        cs_val = cons[i]
        d_val = disag[i]

        if score >= 0.50:
            lvl = "Critical"
        elif score >= 0.35:
            lvl = "High"
        elif score >= 0.20:
            lvl = "Medium"
        else:
            lvl = "Low"
        levels.append(lvl)
        colors.append(PRIORITY_LEVELS[lvl]["color"])

        # Reason generator
        r_parts = []
        if c_val < 0.65:
            r_parts.append(f"Low classifier confidence ({c_val*100:.0f}%)")
        if cs_val < 0.70:
            r_parts.append(f"Inter-annual crop instability ({cs_val*100:.0f}% consistency)")
        if d_val > 0.5:
            r_parts.append("Paddy-banana rotation / boundary mixture")
        if not r_parts:
            r_parts.append("Routine spot verification")
        reasons.append("; ".join(r_parts))

    gdf["priority_level"] = levels
    gdf["priority_color"] = colors
    gdf["audit_reason"] = reasons

    # Centroids in WGS84 for mobile GPS
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    centroids = gdf_wgs84.geometry.centroid
    gdf["center_lon"] = np.round(centroids.x, 5)
    gdf["center_lat"] = np.round(centroids.y, 5)

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
    total_parcels = len(gdf)
    for lvl, pinfo in PRIORITY_LEVELS.items():
        cnt = int(np.sum(gdf["priority_level"] == lvl))
        area = round(float(gdf[gdf["priority_level"] == lvl]["area_ha"].sum()), 1)
        pct = round((cnt / max(1, total_parcels)) * 100.0, 1)
        stats_by_level[lvl] = {"count": cnt, "area_ha": area, "pct": pct, "color": pinfo["color"]}
        chart_data.append({"category": lvl, "count": cnt, "area_ha": area, "pct": pct, "color": pinfo["color"]})

    # Save CSV
    csv_path = out_dir / "field_verification_top50.csv"
    pd.DataFrame(top50_records).to_csv(csv_path, index=False)

    # Save KMZ / KML for mobile GPS ground audit
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

    shapes_p = [(geom, score) for geom, score in zip(gdf.geometry, gdf["priority_score"])]
    priority_raster = rasterize(shapes_p, out_shape=(ny, nx), transform=tfm, fill=0.0, dtype=np.float32)

    # Render PNG
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    for r in range(ny):
        for c in range(nx):
            val = priority_raster[r, c]
            if val == 0:
                continue
            if val >= 0.50:
                rgb[r, c] = [180, 85, 75]
            elif val >= 0.35:
                rgb[r, c] = [231, 111, 81]
            elif val >= 0.20:
                rgb[r, c] = [233, 196, 106]
            else:
                rgb[r, c] = [42, 157, 143]

    png_path = out_dir / "verification_priority.png"
    Image.fromarray(rgb).save(png_path)

    critical_cnt = stats_by_level["Critical"]["count"]
    high_cnt = stats_by_level["High"]["count"]
    crit_high_pct = stats_by_level["Critical"]["pct"] + stats_by_level["High"]["pct"]

    key_finding = (
        f"{crit_high_pct:.1f}% of parcels ({critical_cnt + high_cnt} fields) are flagged for targeted ground audit "
        f"due to lower phenological confidence (<65%) or multi-year crop switching, enabling efficient sample-based field verification."
    )

    result = {
        "id": "verification_priority",
        "title": "Field Verification Priority",
        "subtitle": "Targeted ground-audit ranking & GPS navigation coordinates",
        "key_finding": key_finding,
        "total_parcels": total_parcels,
        "critical_parcels": critical_cnt,
        "high_priority_parcels": high_cnt,
        "stats": stats_by_level,
        "chart_data": chart_data,
        "top50": top50_records,
        "legend": [
            {"id": lvl, "label": f"{lvl} Priority", "color": pinfo["color"]}
            for lvl, pinfo in PRIORITY_LEVELS.items()
        ],
        "method": (
            "Multi-criteria objective index combining Random Forest posterior class certainty (50%), "
            "inter-annual agricultural consistency (30%), and crop-switching / phenological ambiguity flags (20%). "
            "Parcels are sorted and packaged with geographic coordinates for mobile GPS field verification."
        ),
        "limitations": (
            "Parcels in cloud-prone mountain boundary fringes may receive elevated priority scores due to higher "
            "optical composite variance rather than genuine ground-cover ambiguity."
        ),
        "csv_url": "/api/insights/verification_priority/csv",
        "kml_url": "/api/insights/verification_priority/kml",
        "png_url": "/api/insights/verification_priority/png",
    }

    with open(cache_json, "w") as fp:
        json.dump(result, fp, indent=2)

    return result
