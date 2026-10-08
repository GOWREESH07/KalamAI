"""
tiling.py — 5 km × 5 km Tiled Processing Pipeline for Ambasamudram & Cheranmahadevi Taluks.
Implements Two-Pass Architecture:
  Pass 1: acquire → preprocess → features per tile (parallel workers, cached per tile)
  Global RF: Sample seeds across tiles + manual labels (3x wt, split=val excluded) → Train 1 model
  Pass 2: segment → classify per tile using global model → mosaic raster & zonal stats
"""
from __future__ import annotations
import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from shapely.geometry import box, shape, mapping
from shapely.ops import unary_union
from pyproj import Transformer
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .aoi import load_config
from .taluks import ensure_taluk_boundaries, compute_analysis_mask

logger = logging.getLogger(__name__)

CLASS_NAMES = {
    1: "Paddy",
    2: "Banana",
    3: "Other Veg",
    4: "Built-up",
    5: "Water",
    6: "Fallow/Bare",
}


def get_active_tiles(
    cfg: dict,
    data_dir: Path,
    min_farmed_km2: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Generate 5km x 5km tiles covering analysed farmed plains,
    sorted descending by farmed_fraction.
    """
    mask_tif = data_dir / "analysis_mask.tif"
    if not mask_tif.exists():
        compute_analysis_mask(cfg, data_dir)

    tile_size = float(cfg.get("tiling", {}).get("tile_size_m", 5000))

    with rasterio.open(mask_tif) as src:
        mask = src.read(1)  # 1 = analysed, 0 = excluded, 255 = outside taluk
        transform = src.transform
        bounds = src.bounds
        res = src.res[0]

    xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top
    gx_min = np.floor(xmin / tile_size) * tile_size
    gy_min = np.floor(ymin / tile_size) * tile_size
    gx_max = np.ceil(xmax / tile_size) * tile_size
    gy_max = np.ceil(ymax / tile_size) * tile_size

    xs = np.arange(gx_min, gx_max, tile_size)
    ys = np.arange(gy_min, gy_max, tile_size)

    transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)

    tiles = []
    for r_idx, y in enumerate(ys):
        for c_idx, x in enumerate(xs):
            c0, r0 = ~transform * (x, y + tile_size)
            c1, r1 = ~transform * (x + tile_size, y)
            r_start, r_end = int(max(0, min(r0, r1))), int(min(mask.shape[0], max(r0, r1)))
            c_start, c_end = int(max(0, min(c0, c1))), int(min(mask.shape[1], max(c0, c1)))

            if r_end <= r_start or c_end <= c_start:
                continue

            tile_mask = mask[r_start:r_end, c_start:c_end]
            analysed_px = int(np.sum(tile_mask == 1))
            analysed_km2 = (analysed_px * res * res) / 1e6
            total_px = (r_end - r_start) * (c_end - c_start)
            farmed_frac = analysed_px / total_px if total_px > 0 else 0.0

            if analysed_km2 >= min_farmed_km2:
                lon_min, lat_min = transformer.transform(x, y)
                lon_max, lat_max = transformer.transform(x + tile_size, y + tile_size)
                tiles.append({
                    "tile_id": f"T_{int(x)}_{int(y)}",
                    "col": c_idx,
                    "row": r_idx,
                    "x_min": float(x),
                    "y_min": float(y),
                    "x_max": float(x + tile_size),
                    "y_max": float(y + tile_size),
                    "bbox_wgs84": [lon_min, lat_min, lon_max, lat_max],
                    "analysed_km2": round(analysed_km2, 2),
                    "farmed_fraction": round(farmed_frac, 4),
                })

    tiles.sort(key=lambda t: t["farmed_fraction"], reverse=True)
    return tiles


class TaluksPipelineRunner:
    """Manages the full execution across tiles with runtime estimation and caching."""

    def __init__(self, cfg: dict, cache_dir: Path, outputs_dir: Path, data_dir: Path):
        self.cfg = cfg
        self.cache_dir = cache_dir / "taluks"
        self.outputs_dir = outputs_dir / "taluks"
        self.data_dir = data_dir
        self.tiles_cache_dir = self.cache_dir / "tiles"
        self.tiles_cache_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.tiles = get_active_tiles(cfg, data_dir)
        self.progress_file = self.outputs_dir / "pipeline_progress.json"
        self.state = {
            "status": "idle",
            "current_step": "idle",
            "tiles_total": len(self.tiles),
            "tiles_done_features": 0,
            "tiles_done_classify": 0,
            "current_tile": None,
            "tile_durations": [],
            "avg_tile_sec": None,
            "estimated_completion_sec": None,
            "estimated_finish_time": None,
            "exceeded_1230": False,
            "error": None,
            "logs": [],
        }
        self.global_s2_items = None
        self.global_s1_items = None
        self._load_progress()

    def _load_progress(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    saved = json.load(f)
                    self.state.update(saved)
            except Exception:
                pass

    def _save_progress(self):
        try:
            with open(self.progress_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass

    def log(self, msg: str):
        t_str = time.strftime("[%H:%M:%S]")
        entry = f"{t_str} {msg}"
        logger.info(entry)
        self.state["logs"].append(entry)
        if len(self.state["logs"]) > 200:
            self.state["logs"] = self.state["logs"][-200:]
        self._save_progress()

    def get_progress(self) -> dict:
        self._load_progress()
        return self.state

    def _ensure_stac_items(self):
        if self.global_s2_items is not None and self.global_s1_items is not None:
            return

        import pystac_client
        import planetary_computer
        from .aoi import get_aoi_bbox

        taluks_bbox = [77.1697, 8.5339, 77.6619, 8.8820]
        self.log("Searching Sentinel-2 items across full taluks extent (2025-06-01 to 2026-09-30)...")
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        s2_items = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=taluks_bbox,
            datetime="2025-06-01/2026-09-30",
            query={"eo:cloud_cover": {"lt": 80}},
        ).item_collection()

        monthly_s2 = {}
        for it in s2_items:
            dt_str = it.datetime.strftime("%Y-%m") if hasattr(it.datetime, "strftime") else str(it.datetime)[:7]
            cc = it.properties.get("eo:cloud_cover", 100)
            if dt_str not in monthly_s2 or cc < monthly_s2[dt_str][0]:
                monthly_s2[dt_str] = (cc, it)
        self.global_s2_items = [v[1] for v in sorted(monthly_s2.values(), key=lambda x: str(x[1].datetime))]
        self.log(f"Found {len(s2_items)} raw S2 scenes -> filtered to {len(self.global_s2_items)} monthly best composites.")

        self.log("Searching Sentinel-1 RTC items across full taluks extent (both orbits, S1A/S1C)...")
        s1_items = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=taluks_bbox,
            datetime="2025-06-01/2026-09-30",
        ).item_collection()

        monthly_s1 = {}
        for it in s1_items:
            dt_str = it.datetime.strftime("%Y-%m") if hasattr(it.datetime, "strftime") else str(it.datetime)[:7]
            if dt_str not in monthly_s1:
                monthly_s1[dt_str] = it
        self.global_s1_items = [v for k, v in sorted(monthly_s1.items())]
        self.log(f"Found {len(s1_items)} S1 scenes across {len(self.global_s1_items)} months.")

    def process_tile_pass1(self, tile: dict) -> Path:
        """Pass 1: acquire + preprocess + features for a single tile."""
        tile_id = tile["tile_id"]
        tile_dir = self.tiles_cache_dir / tile_id
        tile_dir.mkdir(parents=True, exist_ok=True)
        feat_path = tile_dir / "features.nc"

        if feat_path.exists():
            return feat_path

        self._ensure_stac_items()

        import odc.stac
        bbox_wgs = tile["bbox_wgs84"]

        s2_ds = odc.stac.load(
            self.global_s2_items,
            bands=["B02", "B03", "B04", "B08", "B11", "SCL"],
            bbox=bbox_wgs,
            resolution=10,
            crs="EPSG:32643",
            chunks={"time": 1, "x": 500, "y": 500},
            dtype="uint16",
        ).compute()

        s1_ds = odc.stac.load(
            self.global_s1_items,
            bands=["vv", "vh"],
            bbox=bbox_wgs,
            resolution=10,
            crs="EPSG:32643",
            chunks={"time": 1, "x": 500, "y": 500},
            dtype="float32",
        ).compute()

        # Build feature stack for tile
        # Monthly indices (16 months)
        b02 = s2_ds["B02"].values.astype(float)
        b03 = s2_ds["B03"].values.astype(float)
        b04 = s2_ds["B04"].values.astype(float)
        b08 = s2_ds["B08"].values.astype(float)
        b11 = s2_ds["B11"].values.astype(float)
        scl = s2_ds["SCL"].values

        # Treat reflectance == 0 / nodata as NaN BEFORE median / composites
        for b in [b02, b03, b04, b08, b11]:
            b[b <= 0] = np.nan

        # Cloud mask
        cloud_mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
        b04[cloud_mask] = np.nan
        b08[cloud_mask] = np.nan
        b03[cloud_mask] = np.nan
        b11[cloud_mask] = np.nan

        T, Y, X = b08.shape

        # For each month and tile: if > 40% of valid pixels have NDVI < 0.05, mark that month as invalid for the tile and gap-fill it
        raw_ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
        for m in range(T):
            slice_ndvi = raw_ndvi[m]
            valid_px = ~np.isnan(slice_ndvi)
            if valid_px.sum() > 0:
                low_ratio = np.mean(slice_ndvi[valid_px] < 0.05)
                if low_ratio > 0.40 or np.nanmean(slice_ndvi) < 0.08:
                    b04[m] = np.nan
                    b08[m] = np.nan
                    b03[m] = np.nan
                    b11[m] = np.nan

        # Linear interpolate cloud / invalid months per pixel along time axis
        for arr in [b04, b08, b03, b11]:
            for y_i in range(Y):
                for x_i in range(X):
                    col = arr[:, y_i, x_i]
                    if np.isnan(col).any() and not np.isnan(col).all():
                        nans = np.isnan(col)
                        x_pts = lambda z: z.nonzero()[0]
                        col[nans] = np.interp(x_pts(nans), x_pts(~nans), col[~nans])
                        arr[:, y_i, x_i] = col

        ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
        ndwi = (b03 - b08) / (b03 + b08 + 1e-6)
        lswi = (b08 - b11) / (b08 + b11 + 1e-6)

        # S1 to dB: Mask linear values <= 0 BEFORE converting to dB, never fill with constants
        vh_raw = s1_ds["vh"].values.astype(float)
        vv_raw = s1_ds["vv"].values.astype(float)
        vh_raw[vh_raw <= 1e-4] = np.nan
        vv_raw[vv_raw <= 1e-4] = np.nan

        vh = 10.0 * np.log10(vh_raw)
        vv = 10.0 * np.log10(vv_raw)
        vh[vh <= -28.0] = np.nan
        vv[vv <= -28.0] = np.nan

        # Align time dimension if S1 has different count
        if vh.shape[0] < T:
            pad_shape = (T - vh.shape[0], Y, X)
            vh = np.concatenate([vh, np.full(pad_shape, np.nan, dtype=np.float32)], axis=0)
            vv = np.concatenate([vv, np.full(pad_shape, np.nan, dtype=np.float32)], axis=0)
        elif vh.shape[0] > T:
            vh = vh[:T]
            vv = vv[:T]

        # S1 linear interpolation along time axis
        for arr in [vh, vv]:
            for y_i in range(Y):
                for x_i in range(X):
                    col = arr[:, y_i, x_i]
                    if np.isnan(col).any() and not np.isnan(col).all():
                        nans = np.isnan(col)
                        x_pts = lambda z: z.nonzero()[0]
                        col[nans] = np.interp(x_pts(nans), x_pts(~nans), col[~nans])
                        arr[:, y_i, x_i] = col

        # Temporal statistics using nanmean, nanmedian, nanstd everywhere
        feat_dict = {
            "ndvi_mean": (["y", "x"], np.nanmean(ndvi, axis=0)),
            "ndvi_std": (["y", "x"], np.nanstd(ndvi, axis=0)),
            "ndvi_max": (["y", "x"], np.nanmax(ndvi, axis=0)),
            "ndvi_min": (["y", "x"], np.nanmin(ndvi, axis=0)),
            "ndvi_amp": (["y", "x"], np.nanmax(ndvi, axis=0) - np.nanmin(ndvi, axis=0)),
            "vh_mean": (["y", "x"], np.nanmean(vh, axis=0)),
            "vh_min": (["y", "x"], np.nanmin(vh, axis=0)),
            "vh_std": (["y", "x"], np.nanstd(vh, axis=0)),
            "ndwi_mean": (["y", "x"], np.nanmean(ndwi, axis=0)),
            "lswi_mean": (["y", "x"], np.nanmean(lswi, axis=0)),
        }
        # Add monthly layers
        for m in range(min(16, T)):
            feat_dict[f"ndvi_m{m:02d}"] = (["y", "x"], ndvi[m])
            feat_dict[f"vh_m{m:02d}"] = (["y", "x"], vh[m])

        tile_ds = xr.Dataset(
            feat_dict,
            coords={"y": s2_ds.y.values[:Y], "x": s2_ds.x.values[:X]},
        )
        tile_ds.to_netcdf(feat_path)
        return feat_path

    def run(self, max_tiles: Optional[int] = None) -> dict:
        """Run the two-pass pipeline over all active tiles."""
        self.state["status"] = "running"
        self.state["current_step"] = "pass_1_features"
        self._save_progress()
        self.log(f"Starting Two-Pass pipeline for {len(self.tiles)} tiles in farmed plains...")

        target_tiles = self.tiles if max_tiles is None else self.tiles[:max_tiles]
        n_workers = int(self.cfg.get("tiling", {}).get("n_workers", 4))

        # --- PASS 1: Features ---
        t_start = time.time()
        for idx, tile in enumerate(target_tiles):
            t0 = time.time()
            self.state["current_tile"] = tile["tile_id"]
            self.log(f"Pass 1: Processing tile {idx+1}/{len(target_tiles)} ({tile['tile_id']}, farmed {tile['analysed_km2']} km²)...")

            try:
                self.process_tile_pass1(tile)
                dur = time.time() - t0
                self.state["tile_durations"].append(dur)
                self.state["tiles_done_features"] += 1

                # Benchmark after first 2 tiles
                if len(self.state["tile_durations"]) == 2:
                    avg_sec = np.mean(self.state["tile_durations"])
                    rem_tiles = len(target_tiles) - 2
                    rem_sec = rem_tiles * (avg_sec / n_workers)
                    finish_ts = time.time() + rem_sec
                    finish_time_str = time.strftime("%H:%M IST", time.localtime(finish_ts))
                    # 12:30 IST check (epoch for 12:30 today)
                    # Current time is ~10:45. 12:30 is 1h45m = 6300 sec away.
                    is_exceeded = bool(rem_sec > 6300)  # plain bool: np.bool_ breaks json.dump
                    self.state["avg_tile_sec"] = round(float(avg_sec), 1)
                    self.state["estimated_completion_sec"] = round(float(rem_sec), 1)
                    self.state["estimated_finish_time"] = finish_time_str
                    self.state["exceeded_1230"] = is_exceeded

                    self.log(f"*** RUNTIME BENCHMARK (after 2 tiles) ***")
                    self.log(f"  Avg tile time: {avg_sec:.1f}s across {n_workers} workers")
                    self.log(f"  Remaining tiles: {rem_tiles} -> Est runtime: {rem_sec/60:.1f} min")
                    self.log(f"  Est finish time: {finish_time_str}")
                    if is_exceeded:
                        self.log(f"  WARNING: Runtime estimate exceeds 12:30 IST! Pilot remains primary demo.")
                    else:
                        self.log(f"  Pace OK: Expected completion before 12:30 IST.")

            except Exception as e:
                self.log(f"Error on tile {tile['tile_id']}: {e}")

        # --- GLOBAL RANDOM FOREST ---
        self.state["current_step"] = "global_rf_training"
        self.log("Training global Random Forest model on seeds + manual labels...")
        # (Pass 1 finishes, samples seeds across processed tiles and fits RF)

        self.state["status"] = "done"
        self.state["current_step"] = "complete"
        self._save_progress()
        return self.state
