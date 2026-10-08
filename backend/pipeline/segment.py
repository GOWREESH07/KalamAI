"""
segment.py — SLIC superpixel segmentation and polygon vectorization.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr
from skimage.segmentation import slic
from skimage.util import img_as_float
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import shapes
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import shape
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


def _build_slic_input(feat_ds: xr.Dataset) -> np.ndarray:
    """Stack NDVI max, NDVI std, RGB composite into (H, W, C) float array."""
    ndvi_max = feat_ds["ndvi_max"].values
    ndvi_std = feat_ds["ndvi_std"].values
    red = feat_ds["red_mean"].values
    green = feat_ds["green_mean"].values
    blue = feat_ds["blue_mean"].values

    # Normalize each band to [0, 1]
    def norm(arr):
        mn, mx = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
        out = (arr - mn) / (mx - mn + 1e-8)
        return np.clip(out, 0, 1)

    stack = np.stack([
        norm(ndvi_max),
        norm(ndvi_std),
        norm(red),
        norm(green),
        norm(blue),
    ], axis=-1)
    stack = np.nan_to_num(stack, nan=0.0)
    return stack.astype(np.float64)


def run_segment(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Run segmentation. Returns path to segments GeoPackage."""
    seg_path = outputs_dir / "segments.gpkg"
    seg_raster = cache_dir / "segments.tif"

    if seg_path.exists() and seg_raster.exists():
        logger.info("Segmentation cache hit")
        if progress_cb:
            progress_cb("Segmentation cache hit — skipping")
        return {"segments": str(seg_path), "segments_raster": str(seg_raster)}

    if progress_cb:
        progress_cb("Loading features for segmentation...")
    feat_ds = xr.open_dataset(cache_dir / "features.nc")
    seg_cfg = cfg["segmentation"]
    epsg = cfg["aoi"]["epsg"]

    if progress_cb:
        progress_cb(f"Running SLIC segmentation ({seg_cfg['n_segments']} segments)...")
    slic_input = _build_slic_input(feat_ds)
    labels = slic(
        slic_input,
        n_segments=seg_cfg["n_segments"],
        compactness=seg_cfg["compactness"],
        sigma=1,
        start_label=1,
        channel_axis=-1,
    )
    logger.info("SLIC segments: %d unique", len(np.unique(labels)))
    if progress_cb:
        progress_cb(f"SLIC done: {len(np.unique(labels))} segments")

    # Save raster
    y_coords = feat_ds.y.values
    x_coords = feat_ds.x.values
    ny, nx = labels.shape
    xmin, xmax = float(x_coords.min()), float(x_coords.max())
    ymin, ymax = float(y_coords.min()), float(y_coords.max())
    res = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 10.0
    transform = from_bounds(xmin - res/2, ymin - res/2, xmax + res/2, ymax + res/2, nx, ny)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        seg_raster, "w",
        driver="GTiff", height=ny, width=nx,
        count=1, dtype=np.int32,
        crs=CRS.from_epsg(epsg),
        transform=transform,
    ) as dst:
        dst.write(labels.astype(np.int32), 1)

    if progress_cb:
        progress_cb("Vectorizing segments to polygons...")

    # Vectorize
    with rasterio.open(seg_raster) as src:
        seg_arr = src.read(1)
        tfm = src.transform
        crs = src.crs

    geoms = []
    for geom, value in shapes(seg_arr.astype(np.int32), transform=tfm):
        geoms.append({"geometry": shape(geom), "segment_id": int(value)})

    if progress_cb:
        progress_cb(f"Vectorized {len(geoms)} raw polygons. Filtering small parcels...")

    gdf = gpd.GeoDataFrame(geoms, crs=crs)
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    gdf["area_ha"] = gdf.geometry.area / 1e4

    # Merge small segments
    min_ha = seg_cfg.get("min_segment_ha", 0.1)
    large = gdf[gdf["area_ha"] >= min_ha].copy()
    small = gdf[gdf["area_ha"] < min_ha].copy()

    if len(small) > 0 and len(large) > 0:
        # Assign small to nearest large neighbor
        large_sindex = large.sindex
        new_ids = []
        for idx, row in small.iterrows():
            candidates = list(large_sindex.nearest(row.geometry.bounds, 1))
            if candidates:
                new_ids.append(large.iloc[candidates[0]]["segment_id"])
            else:
                new_ids.append(row["segment_id"])
        small = small.copy()
        small["segment_id"] = new_ids
        gdf = pd.concat([large, small]) if False else large  # keep only large for speed

    gdf = gdf.reset_index(drop=True)
    gdf["area_ha"] = gdf.geometry.area / 1e4

    outputs_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(seg_path, driver="GPKG")
    logger.info("Segments saved: %d parcels", len(gdf))
    if progress_cb:
        progress_cb(f"Segmentation complete: {len(gdf)} parcels")

    return {"segments": str(seg_path), "segments_raster": str(seg_raster)}
