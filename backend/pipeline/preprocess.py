"""
preprocess.py — Monthly composites for S2 and S1.
- S2: mask SCL, monthly median, linear interpolate empty months
- S1: monthly median, convert to dB, 3×3 median filter
- Put everything on common 10m grid in EPSG:32643
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)


def _monthly_median_s2(ds: xr.Dataset, scl_mask_classes: list[int]) -> xr.Dataset:
    """Mask clouds with SCL, then compute monthly median composites."""
    scl = ds["SCL"]
    mask = xr.ones_like(scl, dtype=bool)
    for cls in scl_mask_classes:
        mask = mask & (scl != cls)

    bands = [b for b in ds.data_vars if b != "SCL"]
    monthly_arrays = {}
    times = pd.DatetimeIndex(ds.time.values)
    months = times.to_period("M").unique()

    for band in bands:
        arr = ds[band].where(mask).astype(float)
        monthly_slices = []
        for m in months:
            idx = times.to_period("M") == m
            if idx.sum() == 0:
                monthly_slices.append(xr.full_like(arr.isel(time=0), np.nan))
            else:
                monthly_slices.append(arr.isel(time=idx).median(dim="time"))
        monthly_arrays[band] = xr.concat(monthly_slices, dim="month")

    month_coords = [str(m) for m in months]
    out = xr.Dataset(
        {k: (["month", "y", "x"], v.values) for k, v in monthly_arrays.items()},
        coords={"month": month_coords, "y": ds.y.values, "x": ds.x.values},
    )
    return out


def _interpolate_empty_months(ds: xr.Dataset) -> tuple[xr.Dataset, list[str]]:
    """Linearly interpolate pixels that are NaN for a whole month."""
    flagged = []
    interpolated = {}
    for var in ds.data_vars:
        arr = ds[var].values  # (month, y, x)
        # Check which months are fully NaN
        nan_months = []
        for i in range(arr.shape[0]):
            frac_nan = np.isnan(arr[i]).mean()
            if frac_nan > 0.8:
                nan_months.append(i)
                flagged.append(ds.month.values[i])
        # Interpolate along month axis pixel-wise using numpy interp
        if nan_months:
            valid_idx = [i for i in range(arr.shape[0]) if i not in nan_months]
            if len(valid_idx) >= 2:
                flat = arr.reshape(arr.shape[0], -1)
                for px in range(flat.shape[1]):
                    flat[:, px] = np.interp(
                        np.arange(arr.shape[0]),
                        valid_idx,
                        flat[valid_idx, px],
                        left=flat[valid_idx[0], px],
                        right=flat[valid_idx[-1], px],
                    )
                arr = flat.reshape(arr.shape)
        interpolated[var] = (["month", "y", "x"], arr)
    out = xr.Dataset(interpolated, coords=ds.coords)
    return out, list(set(flagged))


def _monthly_median_s1(ds: xr.Dataset) -> xr.Dataset:
    """Monthly median composite for S1."""
    times = pd.DatetimeIndex(ds.time.values)
    months = times.to_period("M").unique()
    monthly_arrays = {}
    bands = [b for b in ds.data_vars if b in ["vv", "vh"]]
    for band in bands:
        arr = ds[band].astype(float)
        monthly_slices = []
        for m in months:
            idx = times.to_period("M") == m
            if idx.sum() == 0:
                monthly_slices.append(xr.full_like(arr.isel(time=0), np.nan))
            else:
                monthly_slices.append(arr.isel(time=idx).median(dim="time"))
        monthly_arrays[band] = xr.concat(monthly_slices, dim="month")

    month_coords = [str(m) for m in months]
    return xr.Dataset(
        {k: (["month", "y", "x"], v.values) for k, v in monthly_arrays.items()},
        coords={"month": month_coords, "y": ds.y.values, "x": ds.x.values},
    )


def _to_db(arr: np.ndarray) -> np.ndarray:
    """Convert linear SAR backscatter to dB. Clip negatives."""
    arr = np.where(arr <= 0, np.nan, arr)
    return 10.0 * np.log10(arr)


def _median_filter_s1(arr: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply median filter to each monthly slice."""
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        slice_2d = arr[i]
        valid = ~np.isnan(slice_2d)
        tmp = np.where(valid, slice_2d, 0.0)
        filtered = median_filter(tmp, size=size)
        out[i] = np.where(valid, filtered, np.nan)
    return out


def _interpolate_s1_gaps(arr: np.ndarray, max_gap: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate gaps in S1 along the time/month axis of at most max_gap month.
    Does not interpolate longer gaps.
    Returns (interpolated_arr, is_gap_filled_mask).
    arr shape: (n_months, ny, nx)
    """
    n_months, ny, nx = arr.shape
    out = arr.copy()
    gap_filled = np.zeros_like(arr, dtype=bool)

    # Process pixel-wise across time
    for r in range(ny):
        for c in range(nx):
            ts = arr[:, r, c]
            nan_mask = np.isnan(ts)
            if not np.any(nan_mask):
                continue
            
            # Find contiguous NaN blocks
            i = 0
            while i < n_months:
                if nan_mask[i]:
                    start = i
                    while i < n_months and nan_mask[i]:
                        i += 1
                    gap_len = i - start
                    # Only interpolate if gap_len <= max_gap and bounded by valid values
                    if gap_len <= max_gap and start > 0 and i < n_months:
                        v_prev = ts[start - 1]
                        v_next = ts[i]
                        if not np.isnan(v_prev) and not np.isnan(v_next):
                            step = (v_next - v_prev) / (gap_len + 1)
                            for k in range(gap_len):
                                out[start + k, r, c] = v_prev + step * (k + 1)
                                gap_filled[start + k, r, c] = True
                else:
                    i += 1

    return out, gap_filled


def run_preprocess(
    cfg: dict,
    cache_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Run preprocessing. Returns paths to preprocessed NetCDFs."""
    s2_out = cache_dir / "s2_monthly.nc"
    s1_out = cache_dir / "s1_monthly.nc"

    if s2_out.exists() and s1_out.exists():
        logger.info("Preprocess cache hit")
        if progress_cb:
            progress_cb("Preprocess cache hit — skipping")
        return {"s2_monthly": str(s2_out), "s1_monthly": str(s1_out)}

    # --- S2 ---
    s2_raw_path = cache_dir / "s2_raw.nc"
    if not s2_out.exists():
        if progress_cb:
            progress_cb("Loading S2 raw data...")
        ds_s2 = xr.open_dataset(s2_raw_path)

        if progress_cb:
            progress_cb("Masking clouds and computing S2 monthly medians...")
        scl_classes = cfg["sentinel2"]["scl_mask_classes"]
        s2_monthly = _monthly_median_s2(ds_s2, scl_classes)

        if progress_cb:
            progress_cb("Interpolating empty S2 months...")
        s2_monthly, flagged = _interpolate_empty_months(s2_monthly)
        if flagged:
            logger.warning("Interpolated months: %s", flagged)
            if progress_cb:
                progress_cb(f"Flagged/interpolated months: {flagged}")

        # Scale reflectances: S2 L2A is 0–10000
        for var in s2_monthly.data_vars:
            s2_monthly[var] = s2_monthly[var] / 10000.0

        if progress_cb:
            progress_cb("Saving S2 monthly composites...")
        s2_monthly.to_netcdf(s2_out)
        logger.info("S2 monthly saved to %s", s2_out)

    # --- S1 ---
    s1_raw_path = cache_dir / "s1_raw.nc"
    if not s1_out.exists():
        if progress_cb:
            progress_cb("Loading S1 raw data...")
        ds_s1 = xr.open_dataset(s1_raw_path)

        if progress_cb:
            progress_cb("Computing S1 monthly medians...")
        s1_monthly = _monthly_median_s1(ds_s1)

        if progress_cb:
            progress_cb("Converting S1 to dB, filling <=1 mo gaps, and applying median filter...")
        fsize = cfg["preprocessing"].get("s1_filter_size", 3)
        for band in list(s1_monthly.data_vars):
            arr = s1_monthly[band].values
            arr_db = _to_db(arr)
            # Fill gaps of at most 1 month
            arr_filled, gap_mask = _interpolate_s1_gaps(arr_db, max_gap=1)
            n_filled = int(np.sum(gap_mask))
            if n_filled > 0:
                logger.info("S1 %s: filled %d pixel-months with 1-month gap interpolation", band, n_filled)
            arr_filtered = _median_filter_s1(arr_filled, size=fsize)
            s1_monthly[band] = (["month", "y", "x"], arr_filtered)
            if band == "vh":
                s1_monthly["vh_gap_filled"] = (["month", "y", "x"], gap_mask)

        if progress_cb:
            progress_cb("Saving S1 monthly composites...")
        s1_monthly.to_netcdf(s1_out)
        logger.info("S1 monthly saved to %s", s1_out)

    if progress_cb:
        progress_cb("Preprocessing complete.")
    return {"s2_monthly": str(s2_out), "s1_monthly": str(s1_out)}
