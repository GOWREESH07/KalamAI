"""
cropping_intensity.py — Insight 1: Cropping intensity analysis across Agricultural Years.
Detects crop cycles on padded NDVI series with moving-max gap-fill, height 0.40, and SAR flood dip verification.
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


def _moving_max_gapfill(ts: np.ndarray, window: int = 2) -> np.ndarray:
    """Gap-fill cloud-depressed NDVI using local temporal moving max."""
    out = ts.copy()
    n = len(ts)
    for i in range(n):
        w = ts[max(0, i - window):min(n, i + window + 1)]
        valid = w[~np.isnan(w)]
        if len(valid) > 0:
            out[i] = max(ts[i] if not np.isnan(ts[i]) else -1.0, float(np.max(valid)))
    return out


def _compute_tile_intensity(
    ds: xr.Dataset,
    cls_arr: np.ndarray,
    min_h: float = 0.40,
    prom: float = 0.12,
    dist: int = 3,
    pad_m: int = 2,
) -> tuple[np.ndarray, list[int]]:
    """Compute cropping intensity for a single tile or scene."""
    if "ndvi" in ds:
        ndvi_all = ds["ndvi"].values
    else:
        n_m = 16
        ndvi_all = np.stack([ds[f"ndvi_m{m:02d}"].values for m in range(n_m) if f"ndvi_m{m:02d}" in ds], axis=0)

    if "vh" in ds:
        vh_all = ds["vh"].values
    elif "vh_m00" in ds:
        vh_all = np.stack([ds[f"vh_m{m:02d}"].values for m in range(16) if f"vh_m{m:02d}" in ds], axis=0)
    else:
        vh_all = np.full_like(ndvi_all, -15.0)

    ny, nx = cls_arr.shape
    cropland_mask = (cls_arr == 1) | (cls_arr == 2) | (cls_arr == 3) | (cls_arr == 6)

    n_total_months = ndvi_all.shape[0]
    if n_total_months >= 36:
        s_idx = max(0, 24 - pad_m)
        e_idx = min(n_total_months, 36 + pad_m)
        ndvi_pad = ndvi_all[s_idx:e_idx]
        vh_pad = vh_all[s_idx:e_idx]
    else:
        ndvi_pad = ndvi_all
        vh_pad = vh_all

    intensity_map = np.zeros((ny, nx), dtype=np.uint8)
    paddy_peak_counts = []

    w = np.array([0.25, 0.50, 0.25])
    len_ts = ndvi_pad.shape[0]

    for r in range(ny):
        for c in range(nx):
            if not cropland_mask[r, c]:
                continue

            # Perennial rule: Banana (2)
            if cls_arr[r, c] == 2:
                intensity_map[r, c] = 4
                continue

            ts = ndvi_pad[:, r, c]
            max_ndvi = np.nanmax(ts)

            # Fallow rule: max NDVI < 0.35
            if np.all(np.isnan(ts)) or max_ndvi < 0.35:
                intensity_map[r, c] = 0
                continue

            # 2-month moving max gap-fill for cloud depression
            ts_gf = _moving_max_gapfill(ts, window=2)

            valid_idx = np.where(~np.isnan(ts_gf))[0]
            if len(valid_idx) >= 4:
                ts_clean = np.interp(np.arange(len_ts), valid_idx, ts_gf[valid_idx])
            else:
                ts_clean = np.nan_to_num(ts_gf, nan=0.25)

            ts_smooth = np.convolve(ts_clean, w, mode="same")
            peaks, _ = find_peaks(ts_smooth, height=min_h, prominence=prom, distance=dist)
            n_peaks = len(peaks)

            # SAR Flood Dip verification
            vh_ts = vh_pad[:, r, c]
            for m in range(len(vh_ts) - 2):
                if vh_ts[m] < -18.0 and (ts_clean[m + 2] - ts_clean[m] > 0.20):
                    if n_peaks < 2:
                        n_peaks = max(n_peaks, 2)
                        break

            if cls_arr[r, c] == 1:
                paddy_peak_counts.append(n_peaks)

            if n_peaks == 0:
                if np.mean(ts_smooth > 0.50) >= 0.70 and np.std(ts_smooth) < 0.14:
                    intensity_map[r, c] = 4
                else:
                    intensity_map[r, c] = 1
            elif n_peaks == 1:
                intensity_map[r, c] = 1
            elif n_peaks == 2:
                intensity_map[r, c] = 2
            else:
                intensity_map[r, c] = 3

    return intensity_map, paddy_peak_counts


def compute_cropping_intensity(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    study_area: str = "pilot",
) -> Dict[str, Any]:
    out_dir = outputs_dir / "insights" / "cropping_intensity"
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

    cropland_mask = (cls_arr == 1) | (cls_arr == 2) | (cls_arr == 3) | (cls_arr == 6)
    intensity_map = np.zeros((global_ny, global_nx), dtype=np.uint8)
    all_paddy_peaks = []
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

                t_int, t_peaks = _compute_tile_intensity(ds, t_cls, min_h=0.40, prom=0.12, dist=3, pad_m=2)
                all_paddy_peaks.extend(t_peaks)

                col_start, row_start = [int(round(v)) for v in inv_tfm * (t_xmin, t_ymax)]
                col_end, row_end = [int(round(v)) for v in inv_tfm * (t_xmax, t_ymin)]

                c0, c1 = max(0, col_start), min(global_nx, col_end)
                r0, r1 = max(0, row_start), min(global_ny, row_end)

                if c1 > c0 and r1 > r0:
                    t_int_resized = Image.fromarray(t_int).resize((c1 - c0, r1 - r0), Image.NEAREST)
                    intensity_map[r0:r1, c0:c1] = np.array(t_int_resized)
            except Exception as e:
                logger.warning("Error processing tile %s for cropping_intensity: %s", t.name, e)
    else:
        feat_path = cache_dir / "features_40m.nc"
        if not feat_path.exists():
            feat_path = cache_dir / "features.nc"
        if not feat_path.exists():
            feat_path = Path("backend/cache/features.nc")

        ds = xr.open_dataset(feat_path)
        intensity_map, all_paddy_peaks = _compute_tile_intensity(ds, cls_arr, min_h=0.40, prom=0.12, dist=3, pad_m=2)

    # Sanity check logging for paddy pixels
    if all_paddy_peaks:
        pad_arr = np.array(all_paddy_peaks)
        pad_double_share = np.mean(pad_arr >= 2) * 100.0
        pad_0 = np.mean(pad_arr == 0) * 100.0
        pad_1 = np.mean(pad_arr == 1) * 100.0
        pad_2 = np.mean(pad_arr == 2) * 100.0
        pad_3 = np.mean(pad_arr >= 3) * 100.0
        logger.info(
            "Paddy peak distribution (with 2-mo max gapfill + 0.40h + SAR flood dip): "
            "0-peak: %.1f%%, 1-peak: %.1f%%, 2-peak: %.1f%%, 3-peak: %.1f%%. "
            "Double-crop paddy share: %.1f%% (Target ≥ 50%%: %s)",
            pad_0, pad_1, pad_2, pad_3, pad_double_share, "PASS" if pad_double_share >= 50 else "WARN"
        )

    # Calculate statistics across analyzed cropland (excluding built-up & water)
    analyzed_cropland_mask = cropland_mask & (cls_arr != 4) & (cls_arr != 5) & (cls_arr != 255)
    total_cropland_px = int(np.sum(analyzed_cropland_mask))
    if total_cropland_px == 0:
        total_cropland_px = 1

    px_area_ha = 100 / 1e4 if study_area == "pilot" else (30 * 30) / 1e4
    total_cropland_ha = round(total_cropland_px * px_area_ha, 1)

    stats_by_intensity = {}
    chart_data = []
    sum_area_ha = 0.0

    for k, info in INTENSITY_CLASSES.items():
        px_cnt = int(np.sum((intensity_map == k) & analyzed_cropland_mask))
        area_ha = round(px_cnt * px_area_ha, 1)
        pct = round((px_cnt / total_cropland_px) * 100.0, 1)
        sum_area_ha += area_ha
        stats_by_intensity[info["name"]] = {
            "area_ha": area_ha,
            "pct": pct,
            "color": info["color"],
        }
        chart_data.append({
            "category": info["name"],
            "area_ha": area_ha,
            "pct": pct,
            "color": info["color"],
        })

    # Strict Area Sum Assertion
    diff_pct = abs(sum_area_ha - total_cropland_ha) / max(0.1, total_cropland_ha) * 100.0
    area_check_pass = diff_pct <= 0.5

    double_pct = stats_by_intensity["Double Crop"]["pct"]
    single_pct = stats_by_intensity["Single Crop"]["pct"]
    perennial_pct = stats_by_intensity["Perennial (Banana/Tree)"]["pct"]

    ay_label = "AY 2025-26" if model_type == "multi-year" else "AY 2025-26 (16-month baseline)"

    if not area_check_pass:
        key_finding = "Finding withheld — area check failed."
    else:
        if study_area == "taluks":
            key_finding = (
                f"{double_pct:.1f}% of cropland ({stats_by_intensity['Double Crop']['area_ha']:.1f} ha) in Ambasamudram & Cheranmahadevi is double-cropped "
                f"with dual Kar and Pishanam cycles, alongside {perennial_pct:.1f}% under perennial banana canopy."
            )
        else:
            key_finding = (
                f"{double_pct:.1f}% of cropland ({stats_by_intensity['Double Crop']['area_ha']:.1f} ha) is double-cropped with Kar and Pishanam paddy cycles, "
                f"with {perennial_pct:.1f}% under perennial banana."
            )

    # Save GeoTIFF
    tif_path = out_dir / "cropping_intensity.tif"
    with rasterio.open(
        tif_path, "w",
        driver="GTiff", height=global_ny, width=global_nx,
        count=1, dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(intensity_map, 1)

    # Save colored PNG
    png_path = out_dir / "cropping_intensity.png"
    rgb = np.zeros((global_ny, global_nx, 4), dtype=np.uint8)
    for k, info in INTENSITY_CLASSES.items():
        h = info["color"].lstrip("#")
        c_rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
        m = (intensity_map == k) & (cls_arr != 255)
        rgb[m, 0] = c_rgb[0]
        rgb[m, 1] = c_rgb[1]
        rgb[m, 2] = c_rgb[2]
        rgb[m, 3] = 255

    Image.fromarray(rgb, "RGBA").save(png_path)

    extent_label = f"Extent: {canonical_analysed_ha:,.1f} ha · {study_area.title()} · Model: {model_type}"

    result = {
        "id": "cropping_intensity",
        "title": "Cropping Intensity",
        "subtitle": f"Multi-cycle harvest frequency ({ay_label})",
        "extent_label": extent_label,
        "study_area": study_area,
        "model_type": model_type,
        "analysed_ha": canonical_analysed_ha,
        "canonical_analysed_ha": canonical_analysed_ha,
        "total_cropland_ha": total_cropland_ha,
        "sum_area_ha": round(sum_area_ha, 1),
        "area_check_pass": area_check_pass,
        "key_finding": key_finding,
        "ay_label": ay_label,
        "double_cropped_pct": double_pct,
        "single_cropped_pct": single_pct,
        "perennial_pct": perennial_pct,
        "denominator_label": f"total analyzed cropland within mask ({total_cropland_ha:,.1f} ha)",
        "stats": stats_by_intensity,
        "chart_data": chart_data,
        "legend": [
            {"id": str(k), "label": f"{info['name']} ({stats_by_intensity[info['name']]['area_ha']} ha, {stats_by_intensity[info['name']]['pct']}%)", "color": info["color"]}
            for k, info in INTENSITY_CLASSES.items()
        ],
        "method": (
            "Per-pixel phenological peak detection across monthly NDVI composites padded with 2 boundary months. "
            "Applies 2-month moving maximum gap filling for monsoon cloud suppression and integrates SAR specular "
            "inundation dip confirmation. Minimum peak prominence threshold set to 0.12 with height threshold 0.40."
        ),
        "limitations": (
            "Short-duration relay pulses (under 60 days) between major seasons may not reach "
            "the 10-day satellite composite prominence threshold."
        ),
        "png_url": f"/api/insights/cropping_intensity/png?study_area={study_area}",
        "csv_url": f"/api/insights/cropping_intensity/csv?study_area={study_area}",
    }

    pd.DataFrame(chart_data).to_csv(out_dir / "cropping_intensity_breakdown.csv", index=False)

    with open(cache_json, "w") as fp:
        json.dump(result, fp, indent=2)

    return result
