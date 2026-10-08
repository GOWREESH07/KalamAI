"""
api.py — FastAPI backend for Kalam.
Supports dual study areas: "pilot" (33 km² bounding box) and "taluks" (Ambasamudram + Cheranmahadevi).
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from .pipeline.aoi import load_config, validate_aoi, get_aoi_geometry
from .pipeline.taluks import ensure_taluk_boundaries, compute_analysis_mask
from .pipeline.tiling import TaluksPipelineRunner, get_active_tiles
from .jobs import PipelineJob

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------
ROOT = Path(__file__).parent
CFG_PATH = ROOT.parent / "config.yaml"
CACHE_DIR = ROOT / "cache"
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT.parent / "data"
MANUAL_LABELS_PATH = CACHE_DIR / "manual_labels.geojson"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

(CACHE_DIR / "pilot").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "taluks").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "pilot").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "taluks").mkdir(parents=True, exist_ok=True)

cfg = load_config(CFG_PATH)
DEMO_MODE = os.environ.get("DEMO", "0") == "1"

_job: PipelineJob | None = None
_taluks_runner: TaluksPipelineRunner | None = None

app = FastAPI(
    title="Kalam API",
    description="Paddy & Banana Parcel Mapping System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)


def _get_out_dir(study_area: str = "pilot") -> Path:
    if study_area == "taluks":
        d = OUTPUTS_DIR / "taluks"
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = OUTPUTS_DIR / "pilot"
    if d.exists() and (d / "metrics.json").exists():
        return d
    return OUTPUTS_DIR


def _get_cache_dir(study_area: str = "pilot") -> Path:
    if study_area == "taluks":
        d = CACHE_DIR / "taluks"
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = CACHE_DIR / "pilot"
    if d.exists():
        return d
    return CACHE_DIR


def _get_job() -> PipelineJob:
    global _job
    if _job is None:
        _job = PipelineJob(cfg, CACHE_DIR, OUTPUTS_DIR)
    return _job


def _get_taluks_runner() -> TaluksPipelineRunner:
    global _taluks_runner
    if _taluks_runner is None:
        _taluks_runner = TaluksPipelineRunner(cfg, CACHE_DIR, OUTPUTS_DIR, DATA_DIR)
    return _taluks_runner


# -------------------------------------------------------------------
# Config & AOI
# -------------------------------------------------------------------
@app.get("/config")
def get_config(study_area: str = "pilot"):
    conf = dict(cfg)
    conf["study_area"] = study_area
    if (DATA_DIR / "mask_stats.json").exists():
        with open(DATA_DIR / "mask_stats.json") as f:
            conf["mask_stats"] = json.load(f)
    return conf


@app.get("/data/availability")
def get_data_availability(study_area: str = "pilot"):
    """Return real cloud-free coverage percentage per month computed from SCL band."""
    import xarray as xr
    cache_d = _get_cache_dir(study_area)
    s2_raw_path = cache_d / "s2_raw.nc"
    if not s2_raw_path.exists():
        # Fallback to root cache
        s2_raw_path = CACHE_DIR / "s2_raw.nc"
    if not s2_raw_path.exists():
        raise HTTPException(status_code=404, detail="S2 raw data not acquired yet")
    ds_raw = xr.open_dataset(s2_raw_path)
    scl = ds_raw["SCL"].values
    cloud_mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    cloud_free_pct = (1.0 - np.mean(cloud_mask, axis=(1, 2))) * 100.0
    dates = [str(t)[:10] for t in ds_raw.time.values]
    return [
        {"date": d, "cloud_free_pct": round(float(pct), 1)}
        for d, pct in zip(dates, cloud_free_pct)
    ]


@app.post("/aoi")
@app.get("/aoi")
def post_aoi(study_area: str = "pilot"):
    if study_area == "taluks":
        taluk_geo_path = DATA_DIR / "taluks.geojson"
        if not taluk_geo_path.exists():
            ensure_taluk_boundaries(DATA_DIR)
        with open(taluk_geo_path) as f:
            fc = json.load(f)
        total_area = sum(feat["properties"].get("area_km2", 0) for feat in fc.get("features", []))
        return {"area_km2": round(total_area, 2), "geometry": fc, "study_area": "taluks"}

    try:
        area_km2 = validate_aoi(cfg)
        return {"area_km2": area_km2, "geometry": get_aoi_geometry(cfg), "study_area": "pilot"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------------------
# Taluks Administrative & Mask Endpoints
# -------------------------------------------------------------------
@app.get("/taluks/boundaries")
def get_taluk_boundaries():
    geo_path = DATA_DIR / "taluks.geojson"
    if not geo_path.exists():
        ensure_taluk_boundaries(DATA_DIR)
    if not geo_path.exists():
        raise HTTPException(status_code=404, detail="Taluks GeoJSON not found")
    with open(geo_path) as f:
        return json.load(f)


@app.get("/taluks/mask")
def get_taluk_mask():
    excl_path = DATA_DIR / "excluded_forest.geojson"
    if not excl_path.exists():
        compute_analysis_mask(cfg, DATA_DIR)
    if not excl_path.exists():
        raise HTTPException(status_code=404, detail="Excluded forest mask not found")
    with open(excl_path) as f:
        return json.load(f)


@app.get("/taluks/stats")
def get_taluk_stats():
    stats_path = DATA_DIR / "mask_stats.json"
    if not stats_path.exists():
        compute_analysis_mask(cfg, DATA_DIR)
    if not stats_path.exists():
        raise HTTPException(status_code=404, detail="Mask stats not found")
    with open(stats_path) as f:
        return json.load(f)


# -------------------------------------------------------------------
# Pipeline Execution & Progress
# -------------------------------------------------------------------
@app.post("/pipeline/run")
def run_pipeline(from_step: str = "acquire", study_area: str = "pilot"):
    if study_area == "taluks":
        return _get_taluks_job().run()

    job = _get_job()
    result = job.run(from_step=from_step)
    return result


_taluks_job = None


def _get_taluks_job():
    global _taluks_job
    if _taluks_job is None:
        from .pipeline.taluks_job import TaluksJob
        _taluks_job = TaluksJob(cfg, CACHE_DIR, OUTPUTS_DIR, DATA_DIR)
    return _taluks_job


@app.get("/pipeline/status")
def pipeline_status(study_area: str = "pilot"):
    if study_area == "taluks":
        if _taluks_job is not None and (_taluks_job.is_running() or _taluks_job.steps["acquire"]["started_at"]):
            return _taluks_job.get_status()
        f = OUTPUTS_DIR / "taluks" / "pipeline_status.json"
        if f.exists():
            try:
                with open(f) as fp:
                    return json.load(fp)
            except Exception:
                pass
        return [{"name": s, "status": "pending", "started_at": None, "finished_at": None,
                 "duration": None, "log_tail": [], "error": None}
                for s in ["acquire", "preprocess", "features", "labels", "segment", "classify", "export"]]
    job = _get_job()
    return job.get_status()


@app.get("/pipeline/progress")
def pipeline_progress():
    runner = _get_taluks_runner()
    return runner.get_progress()


# -------------------------------------------------------------------
# Profile Endpoints
# -------------------------------------------------------------------
@app.get("/profile")
def get_profile(lon: float, lat: float, study_area: str = "pilot"):
    """Return monthly NDVI, NDWI, VH time series for a given lon/lat."""
    import xarray as xr
    from pyproj import Transformer
    import geopandas as gpd
    from shapely.geometry import Point

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    x_utm, y_utm = transformer.transform(lon, lat)

    if study_area == "pilot":
        aoi = cfg["aoi"]
        if not (aoi["lon_min"] <= lon <= aoi["lon_max"] and aoi["lat_min"] <= lat <= aoi["lat_max"]):
            raise HTTPException(
                status_code=400,
                detail={"error": "outside_aoi", "message": f"Point ({lon:.4f}, {lat:.4f}) is outside the pilot study area bounds"}
            )
        cache_d = _get_cache_dir("pilot")
        feat_path = cache_d / "features.nc"
        if not feat_path.exists():
            feat_path = CACHE_DIR / "features.nc"
        if not feat_path.exists():
            raise HTTPException(status_code=404, detail="Features not computed yet")

        ds = xr.open_dataset(feat_path)
        x_coords = ds.x.values
        y_coords = ds.y.values

        x_query = float(np.clip(x_utm, x_coords.min(), x_coords.max()))
        y_query = float(np.clip(y_utm, y_coords.min(), y_coords.max()))

        px = ds.sel(x=x_query, y=y_query, method="nearest")
        months = [str(m) for m in ds.month.values]

        ndvi_vals = [float(v) if not np.isnan(v) else None for v in px["ndvi"].values]
        ndwi_vals = [float(v) if not np.isnan(v) else None for v in px["ndwi"].values]
        vh_vals = [float(v) if not np.isnan(v) else None for v in px["vh"].values]

        is_interp = []
        if "is_interpolated" in px:
            is_interp = [bool(v) for v in px["is_interpolated"].values]
        else:
            is_interp = [False] * len(months)

        row_idx = int(np.abs(y_coords - y_query).argmin())
        col_idx = int(np.abs(x_coords - x_query).argmin())

        return {
            "months": months,
            "ndvi": ndvi_vals,
            "ndwi": ndwi_vals,
            "vh": vh_vals,
            "is_interpolated": is_interp,
            "pixel": {"row": row_idx, "col": col_idx, "x_utm": round(x_utm, 1), "y_utm": round(y_utm, 1)},
        }

    else:
        # Check against taluks boundary polygon
        taluks_geo = DATA_DIR / "taluks.geojson"
        if taluks_geo.exists():
            gdf = gpd.read_file(taluks_geo)
            poly = gdf.union_all()
            if not poly.contains(Point(lon, lat)):
                raise HTTPException(
                    status_code=400,
                    detail={"error": "outside_aoi", "message": f"Point ({lon:.4f}, {lat:.4f}) is outside Ambasamudram & Cheranmahadevi taluks"}
                )
        else:
            if not (77.169 <= lon <= 77.662 and 8.533 <= lat <= 8.882):
                raise HTTPException(
                    status_code=400,
                    detail={"error": "outside_aoi", "message": f"Point ({lon:.4f}, {lat:.4f}) is outside the taluks study area bounds"}
                )

        # Find corresponding 5km tile
        tx = int(np.floor(x_utm / 5000.0) * 5000)
        ty = int(np.floor(y_utm / 5000.0) * 5000)
        tile_feat_path = CACHE_DIR / "taluks" / "tiles" / f"T_{tx}_{ty}" / "features.nc"

        if not tile_feat_path.exists():
            # Fallback search across any available tiles
            candidate_tiles = sorted((CACHE_DIR / "taluks" / "tiles").glob("*/features.nc"))
            for ct in candidate_tiles:
                try:
                    ds_cand = xr.open_dataset(ct)
                    if ds_cand.x.min() <= x_utm <= ds_cand.x.max() and ds_cand.y.min() <= y_utm <= ds_cand.y.max():
                        tile_feat_path = ct
                        break
                except Exception:
                    continue

        if not tile_feat_path.exists():
            raise HTTPException(status_code=404, detail="Tile features not available for this coordinate")

        ds = xr.open_dataset(tile_feat_path)
        x_coords = ds.x.values
        y_coords = ds.y.values

        x_query = float(np.clip(x_utm, x_coords.min(), x_coords.max()))
        y_query = float(np.clip(y_utm, y_coords.min(), y_coords.max()))

        px = ds.sel(x=x_query, y=y_query, method="nearest")
        n_months = 16
        months = [d.strftime("%Y-%m") for d in pd.date_range("2025-06-01", "2026-09-30", freq="MS")]

        ndvi_vals = [float(px[f"ndvi_m{m:02d}"].values) if f"ndvi_m{m:02d}" in px and not np.isnan(px[f"ndvi_m{m:02d}"].values) else None for m in range(n_months)]
        vh_vals = [float(px[f"vh_m{m:02d}"].values) if f"vh_m{m:02d}" in px and not np.isnan(px[f"vh_m{m:02d}"].values) else None for m in range(n_months)]
        ndwi_mean = float(px["ndwi_mean"].values) if "ndwi_mean" in px and not np.isnan(px["ndwi_mean"].values) else None
        ndwi_vals = [ndwi_mean] * n_months

        is_interp = [False] * n_months
        row_idx = int(np.abs(y_coords - y_query).argmin())
        col_idx = int(np.abs(x_coords - x_query).argmin())

        return {
            "months": months,
            "ndvi": ndvi_vals,
            "ndwi": ndwi_vals,
            "vh": vh_vals,
            "is_interpolated": is_interp,
            "pixel": {"row": row_idx, "col": col_idx, "x_utm": round(x_utm, 1), "y_utm": round(y_utm, 1)},
        }


@app.get("/profiles/class_means")
def get_class_means(study_area: str = "pilot"):
    """Compute and return empirical class-wise mean and std profiles."""
    import xarray as xr
    import rasterio
    import glob

    classes = {1: "Paddy", 2: "Banana", 3: "Other Veg", 4: "Built-up", 5: "Water", 6: "Fallow/Bare"}
    colors = {
        "Paddy": cfg["classes"]["paddy"]["color"],
        "Banana": cfg["classes"]["banana"]["color"],
        "Other Veg": cfg["classes"]["other_veg"]["color"],
        "Built-up": cfg["classes"]["builtup"]["color"],
        "Water": cfg["classes"]["water"]["color"],
        "Fallow/Bare": cfg["classes"]["fallow"]["color"],
    }

    if study_area == "taluks":
        cache_d = _get_cache_dir("taluks")
        tiles = sorted(glob.glob(str(cache_d / "tiles" / "*")))
        if not tiles:
            raise HTTPException(status_code=404, detail="No processed tiles found for taluks")

        n_months = 16
        months = [d.strftime("%Y-%m") for d in pd.date_range("2025-06-01", "2026-09-30", freq="MS")]
        ndvi_accum = {cid: [[] for _ in range(n_months)] for cid in classes}
        vh_accum = {cid: [[] for _ in range(n_months)] for cid in classes}
        class_counts = {cid: 0 for cid in classes}

        for t in tiles:
            cls_f = Path(t) / "classified.tif"
            feat_f = Path(t) / "features.nc"
            if not (cls_f.exists() and feat_f.exists()):
                continue
            try:
                with rasterio.open(cls_f) as src:
                    cls_arr = src.read(1)
                with xr.open_dataset(feat_f) as ds:
                    for cid in classes:
                        mask = (cls_arr == cid)
                        n = int(np.sum(mask))
                        if n == 0:
                            continue
                        class_counts[cid] += n
                        idx = np.where(mask)
                        if len(idx[0]) > 400:
                            sub = np.random.choice(len(idx[0]), 400, replace=False)
                            sub_r, sub_c = idx[0][sub], idx[1][sub]
                        else:
                            sub_r, sub_c = idx[0], idx[1]

                        for m in range(n_months):
                            ndvi_v = ds[f"ndvi_m{m:02d}"].values[sub_r, sub_c]
                            vh_v = ds[f"vh_m{m:02d}"].values[sub_r, sub_c]
                            ndvi_accum[cid][m].append(ndvi_v)
                            vh_accum[cid][m].append(vh_v)
            except Exception:
                continue

        result = {}
        for cid, cname in classes.items():
            cnt = class_counts[cid]
            if cnt < 50 or not ndvi_accum[cid][0]:
                result[cname] = {
                    "n": cnt,
                    "color": colors[cname],
                    "insufficient_samples": True,
                }
                continue

            m_ndvi_means = []
            m_ndvi_stds = []
            m_vh_means = []
            m_vh_stds = []
            valid_shares = []
            for m in range(n_months):
                arr_ndvi = np.concatenate(ndvi_accum[cid][m]) if ndvi_accum[cid][m] else np.array([])
                arr_vh = np.concatenate(vh_accum[cid][m]) if vh_accum[cid][m] else np.array([])

                m_ndvi = float(np.nanmean(arr_ndvi)) if len(arr_ndvi) > 0 else 0.0
                s_ndvi = float(np.nanstd(arr_ndvi)) if len(arr_ndvi) > 0 else 0.0
                m_vh = float(np.nanmean(arr_vh)) if len(arr_vh) > 0 else 0.0
                s_vh = float(np.nanstd(arr_vh)) if len(arr_vh) > 0 else 0.0

                val_share = float(np.mean(~np.isnan(arr_ndvi) & (arr_ndvi >= 0.05))) if len(arr_ndvi) > 0 else 0.0

                m_ndvi_means.append(round(m_ndvi, 4))
                m_ndvi_stds.append(round(s_ndvi, 4))
                m_vh_means.append(round(m_vh, 4))
                m_vh_stds.append(round(s_vh, 4))
                valid_shares.append(round(val_share, 3))

            result[cname] = {
                "n": cnt,
                "color": colors[cname],
                "insufficient_samples": False,
                "ndvi_mean": m_ndvi_means,
                "ndvi_std": m_ndvi_stds,
                "vh_mean": m_vh_means,
                "vh_std": m_vh_stds,
                "valid_shares": valid_shares,
            }

        return {"months": months, "classes": result}

    else:
        out_d = _get_out_dir(study_area)
        cache_d = _get_cache_dir(study_area)

        feat_path = cache_d / "features.nc"
        if not feat_path.exists():
            feat_path = CACHE_DIR / "features.nc"
        class_path = out_d / "classified.tif"
        if not class_path.exists():
            class_path = OUTPUTS_DIR / "pilot" / "classified.tif"
        if not class_path.exists():
            class_path = OUTPUTS_DIR / "classified.tif"

        if not (feat_path.exists() and class_path.exists()):
            raise HTTPException(status_code=404, detail="Outputs not computed yet")

        feat_ds = xr.open_dataset(feat_path)
        with rasterio.open(class_path) as src:
            cls_raster = src.read(1)

        months = feat_ds.month.values.tolist()
        ndvi = feat_ds["ndvi"].values
        vh = feat_ds["vh"].values

        result = {}
        for cid, cname in classes.items():
            mask = (cls_raster == cid)
            n_pixels = int(np.sum(mask))

            if n_pixels < 50:
                result[cname] = {
                    "n": n_pixels,
                    "color": colors[cname],
                    "insufficient_samples": True,
                }
                continue

            c_ndvi = ndvi[:, mask]
            c_vh = vh[:, mask]

            mean_ndvi = np.nanmean(c_ndvi, axis=1)
            std_ndvi = np.nanstd(c_ndvi, axis=1)
            mean_vh = np.nanmean(c_vh, axis=1)
            std_vh = np.nanstd(c_vh, axis=1)

            result[cname] = {
                "n": n_pixels,
                "color": colors[cname],
                "insufficient_samples": False,
                "ndvi_mean": [round(float(v), 4) if v == v else None for v in mean_ndvi],
                "ndvi_std": [round(float(v), 4) if v == v else None for v in std_ndvi],
                "vh_mean": [round(float(v), 4) if v == v else None for v in mean_vh],
                "vh_std": [round(float(v), 4) if v == v else None for v in std_vh],
            }

        return {"months": months, "classes": result}


@app.get("/debug/histograms")
def get_debug_histograms(study_area: str = "pilot"):
    """Compute and return empirical feature distributions split by predicted class."""
    import xarray as xr
    import rasterio
    import glob

    classes = {1: "Paddy", 2: "Banana", 3: "Other Veg", 4: "Built-up", 5: "Water", 6: "Fallow/Bare"}

    if study_area == "taluks":
        cache_d = _get_cache_dir("taluks")
        tiles = sorted(glob.glob(str(cache_d / "tiles" / "*")))
        if not tiles:
            raise HTTPException(status_code=404, detail="No processed tiles found for taluks")

        stats = {cname: {"ndvi_std": [], "ndvi_months_gt_05": [], "vh_min": [], "vh_mean": [], "count": 0} for cname in classes.values()}

        for t in tiles:
            cls_file = Path(t) / "classified.tif"
            feat_file = Path(t) / "features.nc"
            if not (cls_file.exists() and feat_file.exists()):
                continue
            try:
                with rasterio.open(cls_file) as src:
                    cls = src.read(1)
                with xr.open_dataset(feat_file) as ds:
                    ndvi_std_arr = ds["ndvi_std"].values
                    vh_min_arr = ds["vh_min"].values
                    vh_mean_arr = ds["vh_mean"].values

                    ndvi_gt_05 = np.zeros(cls.shape, dtype=int)
                    for m in range(16):
                        if f"ndvi_m{m:02d}" in ds:
                            ndvi_gt_05 += (ds[f"ndvi_m{m:02d}"].values > 0.5).astype(int)

                    for cid, cname in classes.items():
                        mask = (cls == cid)
                        cnt = int(np.sum(mask))
                        if cnt == 0:
                            continue
                        stats[cname]["count"] += cnt
                        idx = np.where(mask)
                        if len(idx[0]) > 400:
                            sub = np.random.choice(len(idx[0]), 400, replace=False)
                            sub_r, sub_c = idx[0][sub], idx[1][sub]
                        else:
                            sub_r, sub_c = idx[0], idx[1]

                        stats[cname]["ndvi_std"].append(ndvi_std_arr[sub_r, sub_c])
                        stats[cname]["vh_min"].append(vh_min_arr[sub_r, sub_c])
                        stats[cname]["vh_mean"].append(vh_mean_arr[sub_r, sub_c])
                        stats[cname]["ndvi_months_gt_05"].append(ndvi_gt_05[sub_r, sub_c])
            except Exception:
                continue

        res = {}
        for cname, data in stats.items():
            if not data["ndvi_std"]:
                continue
            c_std = np.concatenate(data["ndvi_std"])
            c_gt05 = np.concatenate(data["ndvi_months_gt_05"])
            c_vh_min = np.concatenate(data["vh_min"])
            c_vh_mean = np.concatenate(data["vh_mean"])

            c_std_clean = c_std[~np.isnan(c_std)]
            c_vh_min_clean = c_vh_min[~np.isnan(c_vh_min)]
            c_vh_mean_clean = c_vh_mean[~np.isnan(c_vh_mean)]

            res[cname] = {
                "count": data["count"],
                "ndvi_std": {
                    "counts": np.histogram(c_std_clean, bins=20, range=(0.0, 0.4))[0].tolist() if len(c_std_clean) > 0 else [0]*20
                },
                "ndvi_months_gt_05": {
                    "counts": np.histogram(c_gt05, bins=16, range=(0, 16))[0].tolist()
                },
                "vh_min": {
                    "counts": np.histogram(c_vh_min_clean, bins=20, range=(-30.0, -5.0))[0].tolist() if len(c_vh_min_clean) > 0 else [0]*20
                },
                "vh_mean": {
                    "counts": np.histogram(c_vh_mean_clean, bins=20, range=(-25.0, -5.0))[0].tolist() if len(c_vh_mean_clean) > 0 else [0]*20
                },
            }

        return res

    else:
        out_d = _get_out_dir(study_area)
        cache_d = _get_cache_dir(study_area)

        feat_path = cache_d / "features.nc"
        if not feat_path.exists():
            feat_path = CACHE_DIR / "features.nc"
        class_path = out_d / "classified.tif"
        if not class_path.exists():
            class_path = OUTPUTS_DIR / "pilot" / "classified.tif"
        if not class_path.exists():
            class_path = OUTPUTS_DIR / "classified.tif"

        if not (feat_path.exists() and class_path.exists()):
            raise HTTPException(status_code=404, detail="Features or classification not computed yet")

        feat_ds = xr.open_dataset(feat_path)
        with rasterio.open(class_path) as src:
            cls_raster = src.read(1)

        ndvi = feat_ds["ndvi"].values
        ndvi_std = feat_ds["ndvi_std"].values if "ndvi_std" in feat_ds else np.nanstd(ndvi, axis=0)
        vh_min = feat_ds["vh_min"].values if "vh_min" in feat_ds else np.nanmin(feat_ds["vh"].values, axis=0)
        vh_mean = feat_ds["vh_mean"].values if "vh_mean" in feat_ds else np.nanmean(feat_ds["vh"].values, axis=0)
        ndvi_gt_05 = np.sum(ndvi > 0.5, axis=0)

        res = {}
        for cid, cname in classes.items():
            mask = (cls_raster == cid)
            count = int(np.sum(mask))
            if count == 0:
                continue

            c_std = ndvi_std[mask]
            c_gt05 = ndvi_gt_05[mask]
            c_vh_min = vh_min[mask]
            c_vh_mean = vh_mean[mask]

            c_std_clean = c_std[~np.isnan(c_std)]
            c_vh_min_clean = c_vh_min[~np.isnan(c_vh_min)]
            c_vh_mean_clean = c_vh_mean[~np.isnan(c_vh_mean)]

            res[cname] = {
                "count": count,
                "ndvi_std": {
                    "counts": np.histogram(c_std_clean, bins=20, range=(0.0, 0.4))[0].tolist() if len(c_std_clean) > 0 else [0]*20
                },
                "ndvi_months_gt_05": {
                    "counts": np.histogram(c_gt05, bins=16, range=(0, 16))[0].tolist()
                },
                "vh_min": {
                    "counts": np.histogram(c_vh_min_clean, bins=20, range=(-30.0, -5.0))[0].tolist() if len(c_vh_min_clean) > 0 else [0]*20
                },
                "vh_mean": {
                    "counts": np.histogram(c_vh_mean_clean, bins=20, range=(-25.0, -5.0))[0].tolist() if len(c_vh_mean_clean) > 0 else [0]*20
                },
            }

        return res



# -------------------------------------------------------------------
# Layer Images & Rasters
# -------------------------------------------------------------------
@app.get("/layers/{layer_name}")
def get_layer(layer_name: str, m: Optional[int] = 0, study_area: str = "pilot"):
    """Serve pre-rendered PNG overlays."""
    out_d = _get_out_dir(study_area)
    cache_d = _get_cache_dir(study_area)

    if layer_name == "rgb":
        f = out_d / "rgb.png"
    elif layer_name == "classified":
        f = out_d / "classified_rgb.png"
    elif layer_name.startswith("ndvi_month"):
        f = out_d / f"ndvi_month_{m:02d}.png"
    else:
        raise HTTPException(status_code=404, detail="Unknown layer")

    if not f.exists():
        # Fallback to pilot/root
        f_alt = OUTPUTS_DIR / "pilot" / f.name
        if not f_alt.exists():
            f_alt = OUTPUTS_DIR / f.name
        if f_alt.exists():
            return FileResponse(f_alt, media_type="image/png")
        raise HTTPException(status_code=404, detail="Layer not found")
    return FileResponse(f, media_type="image/png")


@app.get("/layers/{layer_name}/bounds")
def get_layer_bounds(layer_name: str, study_area: str = "pilot"):
    """Return [west, south, east, north] in WGS84."""
    if study_area == "taluks":
        # Taluk extent WGS84 bounds
        return {"bounds": [77.1697, 8.5339, 77.6619, 8.8820]}
    aoi = cfg["aoi"]
    return {
        "bounds": [aoi["lon_min"], aoi["lat_min"], aoi["lon_max"], aoi["lat_max"]]
    }


# -------------------------------------------------------------------
# Results: Parcels, Metrics, Stats, Zonal Stats
# -------------------------------------------------------------------
@app.get("/results/parcels.geojson")
def get_parcels_geojson(study_area: str = "pilot", zoom: Optional[int] = None, bbox: Optional[str] = None):
    # Amendment 3: For full taluk extent, parcel polygons only at zoom >= 14
    if study_area == "taluks" and zoom is not None and zoom < 14:
        return {"type": "FeatureCollection", "features": []}

    out_d = _get_out_dir(study_area)
    f = out_d / "parcels.geojson"
    if not f.exists():
        # fallback to pilot/root
        f = OUTPUTS_DIR / "pilot" / "parcels.geojson"
        if not f.exists():
            f = OUTPUTS_DIR / "parcels.geojson"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Parcels not computed")
    return FileResponse(f, media_type="application/json")


@app.get("/results/metrics")
def get_metrics(study_area: str = "pilot"):
    out_d = _get_out_dir(study_area)
    f = out_d / "metrics.json"
    if not f.exists():
        f = OUTPUTS_DIR / "pilot" / "metrics.json"
        if not f.exists():
            f = OUTPUTS_DIR / "metrics.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Metrics not computed")
    with open(f) as fp:
        metrics = json.load(fp)

    # Amendment 5: "Validated accuracy (n polygons)" once >= 15 validation polygons exist, otherwise "Seed agreement"
    n_val = metrics.get("n_manual_val", 0)
    if n_val >= 15:
        metrics["metric_title"] = f"Validated Accuracy ({n_val} polygons)"
        metrics["is_seed_agreement"] = False
    else:
        if metrics.get("is_seed_agreement", True):
            metrics["metric_title"] = "Seed Agreement"

    return metrics


@app.get("/results/stats")
def get_stats(study_area: str = "pilot"):
    out_d = _get_out_dir(study_area)
    f = out_d / "area_stats.csv"
    if not f.exists():
        f = OUTPUTS_DIR / "pilot" / "area_stats.csv"
        if not f.exists():
            f = OUTPUTS_DIR / "area_stats.csv"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Stats not computed")
    df = pd.read_csv(f)
    return df.to_dict(orient="records")


@app.get("/results/zonal_stats")
def get_zonal_stats():
    """Two-column zonal statistics comparing Ambasamudram vs Cheranmahadevi."""
    f = OUTPUTS_DIR / "taluks" / "zonal_stats.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Zonal stats not computed")
    with open(f) as fp:
        return json.load(fp)


# -------------------------------------------------------------------
# Manual Labels
# -------------------------------------------------------------------
class LabelFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list


@app.get("/labels")
def get_labels():
    if MANUAL_LABELS_PATH.exists():
        with open(MANUAL_LABELS_PATH) as f:
            return json.load(f)
    return {"type": "FeatureCollection", "features": []}


@app.post("/labels")
def save_labels(fc: LabelFeatureCollection):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANUAL_LABELS_PATH, "w") as f:
        json.dump(fc.dict(), f)
    # Also copy to pilot and taluks cache
    with open(CACHE_DIR / "pilot" / "manual_labels.geojson", "w") as f:
        json.dump(fc.dict(), f)
    with open(CACHE_DIR / "taluks" / "manual_labels.geojson", "w") as f:
        json.dump(fc.dict(), f)
    return {"saved": len(fc.features)}


# -------------------------------------------------------------------
# Export Downloads
# -------------------------------------------------------------------
@app.get("/export/{file_type}")
def export_file(file_type: str, study_area: str = "pilot"):
    out_d = _get_out_dir(study_area)
    file_map = {
        "tif": out_d / "classified.tif",
        "gpkg": out_d / "parcels.gpkg",
        "csv": out_d / "area_stats.csv",
        "map_png": out_d / "thematic_map.png",
        "map_pdf": out_d / "thematic_map.pdf",
    }
    f = file_map.get(file_type)
    if f is None or not f.exists():
        # Fallback to pilot/root
        f_alt = OUTPUTS_DIR / "pilot" / f.name if f else None
        if not f_alt or not f_alt.exists():
            f_alt = OUTPUTS_DIR / f.name if f else None
        if f_alt and f_alt.exists():
            return FileResponse(f_alt, filename=f_alt.name)
        raise HTTPException(status_code=404, detail=f"{file_type} not exported yet")
    return FileResponse(f, filename=f.name)


# -------------------------------------------------------------------
# Insights Module Endpoints
# -------------------------------------------------------------------
from .insights.runner import run_all_insights, get_all_insights_summary, INSIGHT_MODULES
from .insights.atlas import generate_insights_atlas_pdf


@app.get("/api/insights")
@app.get("/insights")
def get_insights(study_area: str = "pilot"):
    out_d = _get_out_dir(study_area)
    return get_all_insights_summary(cfg, out_d)


@app.get("/api/insights/{insight_id}")
@app.get("/insights/{insight_id}")
def get_insight_detail(insight_id: str, study_area: str = "pilot", recompute: bool = False):
    out_d = _get_out_dir(study_area)
    cache_d = _get_cache_dir(study_area)
    cache_json = out_d / "insights" / insight_id / "data.json"

    if cache_json.exists() and not recompute:
        try:
            with open(cache_json) as fp:
                return json.load(fp)
        except Exception:
            pass

    # Find matching insight computation module
    mod_match = [m for m in INSIGHT_MODULES if m[0] == insight_id]
    if not mod_match:
        raise HTTPException(status_code=404, detail=f"Insight {insight_id} not found")

    compute_fn = mod_match[0][3]
    try:
        res = compute_fn(cfg, cache_d, out_d, study_area)
        return res
    except Exception as e:
        logger.error("Error computing insight %s: %s", insight_id, e)
        raise HTTPException(status_code=500, detail=f"Failed computing {insight_id}: {str(e)}")


@app.post("/api/insights/compute_all")
@app.post("/insights/compute_all")
def compute_all_insights(study_area: str = "pilot"):
    out_d = _get_out_dir(study_area)
    cache_d = _get_cache_dir(study_area)
    results = run_all_insights(cfg, cache_d, out_d, study_area)
    # Also generate Atlas PDF
    try:
        generate_insights_atlas_pdf(cfg, out_d, study_area)
    except Exception as e:
        logger.warning("Atlas PDF generation skipped: %s", e)
    return {"status": "completed", "insights": list(results.keys())}


@app.get("/api/insights/{insight_id}/png")
@app.get("/insights/{insight_id}/png")
def get_insight_png(insight_id: str, study_area: str = "pilot"):
    out_d = _get_out_dir(study_area)
    png_path = out_d / "insights" / insight_id / f"{insight_id}.png"
    if not png_path.exists():
        # Compute insight on the fly
        get_insight_detail(insight_id, study_area)
    if not png_path.exists():
        raise HTTPException(status_code=404, detail="PNG overlay not available for this insight")
    return FileResponse(png_path, media_type="image/png")


@app.get("/api/insights/{insight_id}/csv")
@app.get("/insights/{insight_id}/csv")
def get_insight_csv(insight_id: str, study_area: str = "pilot"):
    out_d = _get_out_dir(study_area)
    # Search for any csv inside insight dir
    i_dir = out_d / "insights" / insight_id
    if not i_dir.exists():
        get_insight_detail(insight_id, study_area)
    csv_files = list(i_dir.glob("*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail="CSV export not found for this insight")
    return FileResponse(csv_files[0], filename=csv_files[0].name)


@app.get("/api/insights/{insight_id}/kml")
@app.get("/insights/{insight_id}/kml")
def get_insight_kml(insight_id: str, study_area: str = "pilot"):
    out_d = _get_out_dir(study_area)
    kml_p = out_d / "insights" / insight_id / "field_verification.kml"
    if not kml_p.exists():
        get_insight_detail(insight_id, study_area)
    if not kml_p.exists():
        raise HTTPException(status_code=404, detail="KML file not available")
    return FileResponse(kml_p, filename=kml_p.name, media_type="application/vnd.google-earth.kml+xml")


@app.get("/api/insights/atlas/pdf")
@app.get("/insights/atlas/pdf")
def get_insights_atlas_pdf(study_area: str = "pilot"):
    out_d = _get_out_dir(study_area)
    pdf_p = out_d / "insights" / "atlas" / "kalam_insights_atlas.pdf"
    if not pdf_p.exists():
        # Compute all insights first and build atlas
        run_all_insights(cfg, _get_cache_dir(study_area), out_d, study_area)
        generate_insights_atlas_pdf(cfg, out_d, study_area)
    if not pdf_p.exists():
        raise HTTPException(status_code=404, detail="Atlas PDF not generated")
    return FileResponse(pdf_p, filename="kalam_insights_atlas.pdf", media_type="application/pdf")


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": DEMO_MODE}

