"""
acquire.py — Download Sentinel-2 L2A and Sentinel-1 RTC data.
Uses pystac-client + planetary-computer + odc-stac.
Caches to NetCDF. Retries with exponential backoff.
"""
from __future__ import annotations
import logging
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .aoi import load_config, get_aoi_bbox

logger = logging.getLogger(__name__)


def _retry(fn: Callable, retries: int = 4, base_delay: float = 2.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait:.0f}s...")
            time.sleep(wait)


def _get_months(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS"))


def acquire_sentinel2(
    cfg: dict,
    cache_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """Download S2 monthly stack. Returns path to NetCDF."""
    cache_path = cache_dir / "s2_raw.nc"
    if cache_path.exists():
        logger.info("S2 cache hit: %s", cache_path)
        if progress_cb:
            progress_cb("S2 cache hit — skipping download")
        return cache_path

    try:
        import pystac_client
        import planetary_computer
        import odc.stac
    except ImportError as e:
        raise ImportError(f"Missing package: {e}. Run: pip install pystac-client planetary-computer odc-stac") from e

    bbox = get_aoi_bbox(cfg)  # lon_min, lat_min, lon_max, lat_max
    s2cfg = cfg["sentinel2"]
    bands = s2cfg["bands"]
    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    if progress_cb:
        progress_cb(f"Searching S2 items {start} → {end}...")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    def _search():
        return catalog.search(
            collections=[s2cfg["collection"]],
            bbox=bbox,
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": s2cfg["cloud_cover_max"]}},
        ).item_collection()

    items = _retry(_search)
    logger.info("Found %d raw S2 items", len(items))

    # Optimization: Filter to 1-2 least cloudy scenes per month to drastically reduce download volume
    import pandas as pd
    monthly_items = {}
    for it in items:
        dt_str = it.datetime.strftime("%Y-%m") if hasattr(it.datetime, "strftime") else str(it.datetime)[:7]
        cc = it.properties.get("eo:cloud_cover", 100)
        if dt_str not in monthly_items or cc < monthly_items[dt_str][0]:
            monthly_items[dt_str] = (cc, it)

    selected_items = [v[1] for v in sorted(monthly_items.values(), key=lambda x: str(x[1].datetime))]
    logger.info("Filtered to %d monthly best S2 scenes", len(selected_items))

    if progress_cb:
        progress_cb(f"Filtered to {len(selected_items)} monthly best scenes (out of {len(items)}). Loading...")

    if len(selected_items) == 0:
        raise RuntimeError("No Sentinel-2 items found for AOI/time range.")

    def _load():
        return odc.stac.load(
            selected_items,
            bands=bands,
            bbox=bbox,
            resolution=s2cfg["resolution_m"],
            crs=f"EPSG:{cfg['aoi']['epsg']}",
            chunks={"time": 1, "x": 512, "y": 512},
            dtype="uint16",
        )

    ds = _retry(_load)
    ds = ds.compute()

    if progress_cb:
        progress_cb(f"S2 loaded: {ds.dims}. Saving to cache...")

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Convert to int32 to avoid uint16 issues in netCDF4
    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.int32)
    ds.to_netcdf(cache_path)
    logger.info("S2 saved to %s", cache_path)
    if progress_cb:
        progress_cb(f"S2 acquisition complete ({len(items)} scenes)")
    return cache_path


def acquire_sentinel1(
    cfg: dict,
    cache_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """Download S1 RTC (or GRD fallback) monthly stack.
    Includes both ascending and descending orbits and all platforms (S1A + S1C).
    Logs number of S1 scenes per month.
    """
    cache_path = cache_dir / "s1_raw.nc"
    if cache_path.exists():
        logger.info("S1 cache hit: %s", cache_path)
        if progress_cb:
            progress_cb("S1 cache hit — skipping download")
        return cache_path

    try:
        import pystac_client
        import planetary_computer
        import odc.stac
    except ImportError as e:
        raise ImportError(f"Missing package: {e}") from e

    bbox = get_aoi_bbox(cfg)
    s1cfg = cfg["sentinel1"]
    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    if progress_cb:
        progress_cb(f"Searching S1 items {start} → {end} (ascending & descending, S1A/S1C)...")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    collections_to_try = [s1cfg.get("collection", "sentinel-1-rtc"), "sentinel-1-grd"]
    items = []
    active_collection = s1cfg.get("collection", "sentinel-1-rtc")

    for coll in collections_to_try:
        def _search():
            return catalog.search(
                collections=[coll],
                bbox=bbox,
                datetime=f"{start}/{end}",
            ).item_collection()

        try:
            res = _retry(_search)
            if len(res) > 0:
                items = res
                active_collection = coll
                break
        except Exception as e:
            logger.warning("Search failed for collection %s: %e", coll, e)

    logger.info("Found %d raw S1 items from %s", len(items), active_collection)

    # Log the number of S1 scenes per month
    month_counts: dict[str, int] = {}
    monthly_s1: dict[str, list] = {}
    for it in items:
        dt_str = it.datetime.strftime("%Y-%m") if hasattr(it.datetime, "strftime") else str(it.datetime)[:7]
        month_counts[dt_str] = month_counts.get(dt_str, 0) + 1
        monthly_s1.setdefault(dt_str, []).append(it)

    logger.info("Sentinel-1 scenes per month: %s", month_counts)
    if progress_cb:
        msg = f"S1 ({active_collection}): {len(items)} scenes across {len(month_counts)} months"
        progress_cb(msg)
        for m_str in sorted(month_counts.keys()):
            logger.info("  %s: %d S1 scenes", m_str, month_counts[m_str])

    # Select 1 representative scene per month (or composite)
    selected_s1 = [scenes[0] for scenes in monthly_s1.values()]
    logger.info("Filtered to %d monthly S1 scenes", len(selected_s1))

    if len(selected_s1) == 0:
        raise RuntimeError("No Sentinel-1 items found for AOI/time range.")

    bands = s1cfg["bands"]
    def _load():
        return odc.stac.load(
            selected_s1,
            bands=bands,
            bbox=bbox,
            resolution=s1cfg["resolution_m"],
            crs=f"EPSG:{cfg['aoi']['epsg']}",
            chunks={"time": 1, "x": 512, "y": 512},
            dtype="float32",
        )

    ds = _retry(_load)
    ds = ds.compute()

    if progress_cb:
        progress_cb(f"S1 loaded: {ds.dims}. Saving to cache...")

    cache_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(cache_path)
    logger.info("S1 saved to %s", cache_path)
    if progress_cb:
        progress_cb(f"S1 acquisition complete ({len(items)} scenes)")
    return cache_path


def run_acquire(cfg: dict, cache_dir: Path, progress_cb=None) -> dict:
    """Run both S2 and S1 acquisition. Returns paths dict."""
    s2_path = acquire_sentinel2(cfg, cache_dir, progress_cb)
    s1_path = acquire_sentinel1(cfg, cache_dir, progress_cb)
    return {"s2_raw": str(s2_path), "s1_raw": str(s1_path)}
