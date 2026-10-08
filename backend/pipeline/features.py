"""
features.py — Compute spectral indices and temporal feature stack.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(np.abs(b) < 1e-6, np.nan, a / b)
    return result


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED)"""
    return _safe_div(nir - red, nir + red)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (GREEN - NIR) / (GREEN + NIR)"""
    return _safe_div(green - nir, green + nir)


def compute_lswi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """LSWI = (NIR - SWIR) / (NIR + SWIR)"""
    return _safe_div(nir - swir, nir + swir)


def run_features(
    cfg: dict,
    cache_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Compute feature stack. Returns path to features NetCDF."""
    feat_path = cache_dir / "features.nc"
    if feat_path.exists():
        logger.info("Features cache hit")
        if progress_cb:
            progress_cb("Features cache hit — skipping")
        return {"features": str(feat_path)}

    if progress_cb:
        progress_cb("Loading monthly composites...")
    s2 = xr.open_dataset(cache_dir / "s2_monthly.nc")
    s1 = xr.open_dataset(cache_dir / "s1_monthly.nc")

    # Align S1 to S2 months if different
    # Use S2 months as reference
    s2_months = s2.month.values
    n_months = len(s2_months)
    ny, nx = s2["B04"].shape[1], s2["B04"].shape[2]

    if progress_cb:
        progress_cb(f"Computing indices for {n_months} months ({ny}×{nx} px)...")

    # --- Spectral indices per month ---
    nir = s2["B08"].values  # (M, Y, X)
    red = s2["B04"].values
    green = s2["B03"].values
    swir = s2["B11"].values

    ndvi = compute_ndvi(nir, red)   # (M, Y, X)
    ndwi = compute_ndwi(green, nir)
    lswi = compute_lswi(nir, swir)

    # Align VH to S2 grid (simple nearest-month reindex)
    # If S1 has different months, interpolate to S2 months
    s1_months = s1.month.values
    if list(s1_months) == list(s2_months):
        vh = s1["vh"].values if "vh" in s1.data_vars else s1[list(s1.data_vars)[0]].values
    else:
        # Reindex S1 to S2 month coords by nearest
        s1_reindexed = s1.reindex(month=s2_months, method="nearest", tolerance="1") 
        band_name = "vh" if "vh" in s1_reindexed.data_vars else list(s1_reindexed.data_vars)[0]
        vh = s1_reindexed[band_name].values

    if progress_cb:
        progress_cb("Computing temporal statistics...")

    # --- Temporal stats ---
    ndvi_max = np.nanmax(ndvi, axis=0)
    ndvi_min = np.nanmin(ndvi, axis=0)
    ndvi_mean = np.nanmean(ndvi, axis=0)
    ndvi_std = np.nanstd(ndvi, axis=0)
    ndvi_amplitude = ndvi_max - ndvi_min

    ndvi_high_thr = cfg["features"].get("ndvi_high_threshold", 0.6)
    ndvi_months_high = np.sum(ndvi > ndvi_high_thr, axis=0).astype(float)

    # VH stats (already in dB from preprocess)
    vh_min = np.nanmin(vh, axis=0)
    vh_mean = np.nanmean(vh, axis=0)
    vh_min_month = np.argmin(np.where(np.isnan(vh), np.inf, vh), axis=0).astype(float)

    if progress_cb:
        progress_cb("Stacking features into NetCDF...")

    # Build dataset
    coords = {"month": s2_months, "y": s2.y.values, "x": s2.x.values}

    feat_ds = xr.Dataset(
        {
            # Monthly time series
            "ndvi": (["month", "y", "x"], ndvi.astype(np.float32)),
            "ndwi": (["month", "y", "x"], ndwi.astype(np.float32)),
            "lswi": (["month", "y", "x"], lswi.astype(np.float32)),
            "vh":   (["month", "y", "x"], vh.astype(np.float32)),
            # Temporal stats (2D)
            "ndvi_max":          (["y", "x"], ndvi_max.astype(np.float32)),
            "ndvi_min":          (["y", "x"], ndvi_min.astype(np.float32)),
            "ndvi_mean":         (["y", "x"], ndvi_mean.astype(np.float32)),
            "ndvi_std":          (["y", "x"], ndvi_std.astype(np.float32)),
            "ndvi_amplitude":    (["y", "x"], ndvi_amplitude.astype(np.float32)),
            "ndvi_months_high":  (["y", "x"], ndvi_months_high.astype(np.float32)),
            "vh_min":            (["y", "x"], vh_min.astype(np.float32)),
            "vh_mean":           (["y", "x"], vh_mean.astype(np.float32)),
            "vh_min_month":      (["y", "x"], vh_min_month.astype(np.float32)),
            # RGB for segmentation (mean across time)
            "red_mean":   (["y", "x"], np.nanmean(red, axis=0).astype(np.float32)),
            "green_mean": (["y", "x"], np.nanmean(green, axis=0).astype(np.float32)),
            "blue_mean":  (["y", "x"], np.nanmean(s2["B02"].values, axis=0).astype(np.float32)),
        },
        coords=coords,
    )
    feat_ds.attrs["crs"] = f"EPSG:{cfg['aoi']['epsg']}"
    feat_ds.to_netcdf(feat_path)
    logger.info("Features saved to %s", feat_path)
    if progress_cb:
        progress_cb("Features complete.")
    return {"features": str(feat_path)}
