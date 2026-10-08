"""
weak_labels.py — Bootstrapped phenology-based seed pixel generation.
All thresholds from config.yaml. Manual GeoJSON labels take precedence.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from shapely.geometry import shape

logger = logging.getLogger(__name__)

CLASS_IDS = {
    "paddy": 1,
    "banana": 2,
    "other_veg": 3,
    "builtup": 4,
    "water": 5,
    "fallow": 6,
}


def _confidence_score(mask: np.ndarray, feat: dict, class_name: str) -> np.ndarray:
    """Compute a confidence score for pixels in mask. Higher = more confident."""
    scores = np.zeros(mask.shape, dtype=np.float32)
    if class_name == "paddy":
        # More negative VH min = more flooding signal
        scores = np.where(mask, -feat["vh_min"], 0.0)
    elif class_name == "banana":
        scores = np.where(mask, feat["ndvi_months_high"], 0.0)
    elif class_name == "water":
        scores = np.where(mask, feat["ndwi_max_months"], 0.0)
    elif class_name == "builtup":
        scores = np.where(mask, -feat["ndvi_max"], 0.0)
    elif class_name == "fallow":
        scores = np.where(mask, -feat["ndvi_max"], 0.0)
    elif class_name == "other_veg":
        scores = np.where(mask, feat["ndvi_mean"], 0.0)
    return scores


def generate_weak_labels(
    cfg: dict,
    cache_dir: Path,
    manual_labels_path: Path | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Generate seed pixel labels. Returns path to labels CSV."""
    labels_path = cache_dir / "labels.csv"
    if labels_path.exists() and manual_labels_path is None:
        logger.info("Labels cache hit")
        if progress_cb:
            progress_cb("Labels cache hit — skipping")
        return {"labels": str(labels_path)}

    if progress_cb:
        progress_cb("Loading features for label generation...")
    feat_ds = xr.open_dataset(cache_dir / "features.nc")
    wl = cfg["weak_labels"]
    max_px = wl.get("max_pixels_per_class", 2000)

    # Pull 2D arrays
    # Pull 2D and 3D arrays
    ndvi_max = feat_ds["ndvi_max"].values
    ndvi_min = feat_ds["ndvi_min"].values
    ndvi_mean = feat_ds["ndvi_mean"].values
    ndvi_std = feat_ds["ndvi_std"].values
    ndvi_amplitude = feat_ds["ndvi_amplitude"].values
    vh_min = feat_ds["vh_min"].values
    vh_mean = feat_ds["vh_mean"].values
    vh_min_month = feat_ds["vh_min_month"].values
    ndwi_monthly = feat_ds["ndwi"].values  # (M, Y, X)
    lswi_monthly = feat_ds["lswi"].values  # (M, Y, X)
    ndvi_monthly = feat_ds["ndvi"].values  # (M, Y, X)
    vh_monthly = feat_ds["vh"].values      # (M, Y, X)

    # Cloud masking to ignore interpolated/cloud-corrupted months
    s2_raw_path = cache_dir / "s2_raw.nc"
    if s2_raw_path.exists():
        ds_raw = xr.open_dataset(s2_raw_path)
        scl = ds_raw["SCL"].values
        cloud_mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    else:
        cloud_mask = np.zeros_like(ndvi_monthly, dtype=bool)

    ndvi_clean = np.where(cloud_mask, np.nan, ndvi_monthly)
    ndwi_clean = np.where(cloud_mask, np.nan, ndwi_monthly)
    lswi_clean = np.where(cloud_mask, np.nan, lswi_monthly)

    ny, nx = ndvi_max.shape
    n_months = ndvi_monthly.shape[0]

    if progress_cb:
        progress_cb("Applying tuned phenology rules...")

    # --- Paddy Rule (Tightened) ---
    # Paddy requires:
    # 1. NDVI peak >= 0.6 in at least one season (clean months)
    # 2. NDVI trough <= 0.3 in the months before that peak
    # 3. A flood signal: (VH min < -19 dB) OR (LSWI > 0 in the trough month). Use LSWI when VH is missing.
    paddy_cfg = wl["paddy"]
    vh_min_threshold = paddy_cfg.get("vh_min_db", -19.0)
    ndvi_peak_thr = paddy_cfg.get("ndvi_peak_threshold", 0.60)
    ndvi_trough_thr = paddy_cfg.get("ndvi_trough_max", 0.30)

    paddy_mask = np.zeros((ny, nx), dtype=bool)

    for r in range(ny):
        for c in range(nx):
            ts_ndvi = ndvi_clean[:, r, c]
            ts_lswi = lswi_clean[:, r, c]
            ts_vh = vh_monthly[:, r, c]

            # Check if there is an NDVI peak >= 0.6
            # Find peak month(s)
            valid_peaks = [m for m in range(n_months) if ts_ndvi[m] >= ndvi_peak_thr]
            if not valid_peaks:
                continue

            # For any valid peak, check if there was a preceding trough <= 0.3
            is_paddy_pixel = False
            for p_m in valid_peaks:
                if p_m == 0:
                    continue  # need months before peak
                # Months before peak: 1 to 4 months prior
                prior_months = list(range(max(0, p_m - 4), p_m))
                if not prior_months:
                    continue
                troughs = [m for m in prior_months if ts_ndvi[m] <= ndvi_trough_thr]
                if not troughs:
                    continue

                # For the trough month(s), check flood signal:
                # (VH min < -19 dB) OR (LSWI > 0 in trough month)
                for t_m in troughs:
                    flood_signal = False
                    # Check VH min in window around trough or overall
                    vh_val = ts_vh[t_m] if not np.isnan(ts_vh[t_m]) else (vh_min[r, c] if not np.isnan(vh_min[r, c]) else None)
                    if vh_val is not None and vh_val < vh_min_threshold:
                        flood_signal = True
                    elif ts_lswi[t_m] > 0.0 or (t_m > 0 and ts_lswi[t_m - 1] > 0.0):
                        # LSWI > 0 in the trough month (standing water)
                        flood_signal = True

                    if flood_signal:
                        is_paddy_pixel = True
                        break
                if is_paddy_pixel:
                    break

            if is_paddy_pixel:
                paddy_mask[r, c] = True

    # --- Banana Rule ---
    # NDVI > 0.5 in >= 7 of the 12 months of best rolling 12-month window;
    # NDVI std < 0.15; NO flooding signal (VH min > -18 dB). Remove VH mean condition.
    banana_cfg = wl["banana"]
    win_len = banana_cfg.get("rolling_months", 12)
    months_req = banana_cfg.get("ndvi_months_above", 7)
    ndvi_thr = banana_cfg.get("ndvi_threshold", 0.50)
    std_max = banana_cfg.get("ndvi_std_max", 0.15)
    vh_no_flood = banana_cfg.get("vh_min_threshold_db", -18.0)

    n_windows = max(1, n_months - win_len + 1)
    max_months_above = np.zeros((ny, nx), dtype=int)
    for w in range(n_windows):
        win_ndvi = ndvi_clean[w:w + win_len]
        counts = np.nansum(win_ndvi > ndvi_thr, axis=0)
        max_months_above = np.maximum(max_months_above, counts)

    clean_std = np.nanstd(ndvi_clean, axis=0)
    banana_mask = (max_months_above >= months_req) & (clean_std < std_max) & (vh_min > vh_no_flood) & (~paddy_mask)

    # --- Water ---
    water_cfg = wl["water"]
    ndwi_max_months = np.sum(ndwi_clean > water_cfg["ndwi_threshold"], axis=0).astype(float)
    water_mask = (ndwi_max_months >= water_cfg["ndwi_months_above"]) & (~paddy_mask) & (~banana_mask)

    # --- Built-up ---
    bu_cfg = wl["builtup"]
    builtup_mask = (ndvi_max < bu_cfg["ndvi_max_threshold"]) & (ndwi_max_months == 0) & (~water_mask) & (~paddy_mask)

    # --- Fallow ---
    fal_cfg = wl["fallow"]
    fallow_mask = (
        (ndvi_max < fal_cfg["ndvi_max_threshold"])
        & (~builtup_mask)
        & (~water_mask)
        & (~paddy_mask)
        & (~banana_mask)
    )

    # --- Other vegetation ---
    ov_cfg = wl["other_veg"]
    assigned = paddy_mask | banana_mask | water_mask | builtup_mask | fallow_mask
    other_veg_mask = (~assigned) & (ndvi_mean > ov_cfg["ndvi_mean_threshold"])

    masks = {
        "paddy": paddy_mask,
        "banana": banana_mask,
        "water": water_mask,
        "builtup": builtup_mask,
        "fallow": fallow_mask,
        "other_veg": other_veg_mask,
    }

    # Build feature dict for confidence scoring
    feat = {
        "vh_min": vh_min,
        "ndvi_months_high": np.sum(ndvi_monthly > 0.5, axis=0),
        "ndwi_max_months": ndwi_max_months,
        "ndvi_max": ndvi_max,
        "ndvi_mean": ndvi_mean,
    }

    # --- Sample pixels ---
    rows_list = []
    y_coords = feat_ds.y.values
    x_coords = feat_ds.x.values

    for class_name, mask in masks.items():
        class_id = CLASS_IDS[class_name]
        px_indices = np.argwhere(mask)
        if len(px_indices) == 0:
            logger.warning("No seed pixels found for class: %s", class_name)
            if progress_cb:
                progress_cb(f"  Warning: no seeds for {class_name}")
            continue

        scores = _confidence_score(mask, feat, class_name)
        px_scores = scores[px_indices[:, 0], px_indices[:, 1]]
        # Sort by descending confidence
        order = np.argsort(-px_scores)
        px_indices = px_indices[order]
        n_sample = min(len(px_indices), max_px)
        sampled = px_indices[:n_sample]

        for row_col in sampled:
            r, c = row_col
            rows_list.append({
                "class": class_name,
                "class_id": class_id,
                "row": int(r),
                "col": int(c),
                "y": float(y_coords[r]),
                "x": float(x_coords[c]),
                "source": "phenology",
            })
        if progress_cb:
            progress_cb(f"  {class_name}: {n_sample} seed pixels")

    df_labels = pd.DataFrame(rows_list)

    # --- Manual labels (take precedence) ---
    if manual_labels_path and Path(manual_labels_path).exists():
        if progress_cb:
            progress_cb("Merging manual labels (these take precedence)...")
        with open(manual_labels_path) as f:
            geojson = json.load(f)

        xmin, ymax = float(x_coords.min()), float(y_coords.max())
        xmax, ymin = float(x_coords.max()), float(y_coords.min())
        transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)

        manual_rows = []
        for feature in geojson.get("features", []):
            props = feature.get("properties", {})
            cls = props.get("class", "other_veg")
            split = props.get("split", "train") # "train" or "val"
            class_id = CLASS_IDS.get(cls, 3)
            geom = shape(feature["geometry"])
            msk = ~geometry_mask([feature["geometry"]], out_shape=(ny, nx), transform=transform, invert=False)
            px = np.argwhere(msk)
            for r, c in px:
                manual_rows.append({
                    "class": cls,
                    "class_id": class_id,
                    "row": int(r),
                    "col": int(c),
                    "y": float(y_coords[r]),
                    "x": float(x_coords[c]),
                    "source": "manual",
                    "split": split,
                    "weight": 3.0,
                })

        df_manual = pd.DataFrame(manual_rows)
        # Default seeds to train and weight 1.0
        df_labels["split"] = "train"
        df_labels["weight"] = 1.0

        # Manual takes precedence: remove overlapping phenology pixels
        if len(df_manual) > 0:
            manual_rc = set(zip(df_manual["row"], df_manual["col"]))
            df_labels = df_labels[
                ~df_labels.apply(lambda r: (r["row"], r["col"]) in manual_rc, axis=1)
            ]
            df_labels = pd.concat([df_labels, df_manual], ignore_index=True)
    else:
        df_labels["split"] = "train"
        df_labels["weight"] = 1.0

    if progress_cb:
        progress_cb(f"Total labels: {len(df_labels)} pixels across {df_labels['class'].nunique()} classes")

    df_labels.to_csv(labels_path, index=False)
    logger.info("Labels saved to %s", labels_path)
    return {"labels": str(labels_path)}
