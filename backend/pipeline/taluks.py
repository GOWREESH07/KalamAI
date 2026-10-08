"""
taluks.py — Administrative boundaries and terrain/forest analysis mask.
Ambasamudram + Cheranmahadevi taluks.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
from scipy.ndimage import label
from pyproj import Transformer

logger = logging.getLogger(__name__)


def ensure_taluk_boundaries(data_dir: Path) -> Path:
    """Ensure data/taluks.geojson exists with real boundaries."""
    geojson_path = data_dir / "taluks.geojson"
    if geojson_path.exists():
        return geojson_path

    logger.info("Fetching taluk boundaries from OpenStreetMap Overpass API...")
    import requests
    from shapely.geometry import LineString
    from shapely.ops import linemerge, polygonize

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = """[out:json][timeout:60];
    (
      relation(6553374);
      relation(10326572);
    );
    out geom;"""
    headers = {"User-Agent": "KalamMapathon/1.0 (gowreesha)"}
    r = requests.post(overpass_url, data={"data": query}, headers=headers, timeout=60)
    data = r.json()

    features = []
    for el in data.get("elements", []):
        name = el.get("tags", {}).get("name")
        lines = []
        for m in el.get("members", []):
            if m.get("role") in ["outer", ""] and "geometry" in m:
                coords = [(pt["lon"], pt["lat"]) for pt in m["geometry"]]
                if len(coords) >= 2:
                    lines.append(LineString(coords))
        merged = linemerge(lines)
        polys = list(polygonize(merged))
        if not polys:
            continue
        poly = polys[0] if len(polys) == 1 else unary_union(polys)
        gdf = gpd.GeoDataFrame([{"geometry": poly}], crs="EPSG:4326").to_crs("EPSG:32643")
        area_km2 = gdf.geometry.iloc[0].area / 1e6
        features.append({
            "type": "Feature",
            "properties": {
                "name": name,
                "source": f"OpenStreetMap relation {el['id']} (admin_level 6)",
                "area_km2": round(float(area_km2), 2)
            },
            "geometry": mapping(poly)
        })

    fc = {"type": "FeatureCollection", "features": features}
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(geojson_path, "w") as f:
        json.dump(fc, f, indent=2)
    return geojson_path


def compute_analysis_mask(
    cfg: dict,
    data_dir: Path,
    resolution_m: int = 30,
) -> Dict[str, Any]:
    """
    Compute DEM + WorldCover analysis mask.
    Excludes:
      - elev > max_elev (250m)
      - slope > max_slope (8 deg)
      - WorldCover tree cover (class 10) connected to hills AND elev > tree_elev_threshold (120m).
    Banana-safe: Tree cover below 120m remains in the analysis area.
    Saves:
      - data/analysis_mask.tif
      - data/excluded_forest.geojson
      - data/mask_stats.json
    """
    geojson_path = ensure_taluk_boundaries(data_dir)
    mask_tif_path = data_dir / "analysis_mask.tif"
    stats_path = data_dir / "mask_stats.json"
    excluded_geojson_path = data_dir / "excluded_forest.geojson"

    if mask_tif_path.exists() and stats_path.exists() and excluded_geojson_path.exists():
        with open(stats_path) as f:
            return json.load(f)

    import odc.stac
    import pystac_client
    import planetary_computer
    from affine import Affine

    gdf = gpd.read_file(geojson_path)
    gdf_utm = gdf.to_crs(epsg=32643)
    total_geom = gdf_utm.geometry.union_all()

    bbox_wgs = list(gdf.total_bounds)
    mask_cfg = cfg.get("mask", {})
    max_elev = mask_cfg.get("max_elev", 250.0)
    max_slope = mask_cfg.get("max_slope", 8.0)
    tree_elev_th = mask_cfg.get("tree_elev_threshold", 120.0)

    logger.info("Querying DEM and WorldCover on Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox_wgs).item_collection()
    items_wc = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox_wgs,
        datetime="2021-01-01/2021-12-31"
    ).item_collection()

    res = resolution_m
    logger.info("Loading DEM & WorldCover at %dm resolution...", res)
    dem_ds = odc.stac.load(items_dem, bands=["data"], bbox=bbox_wgs, resolution=res, crs="EPSG:32643").compute()
    wc_ds = odc.stac.load(items_wc, bands=["map"], bbox=bbox_wgs, resolution=res, crs="EPSG:32643").compute()

    elev = dem_ds["data"].values[0].astype(float)
    wc = wc_ds["map"].values[0]

    # Calculate slope in degrees
    dy, dx = np.gradient(elev, res, res)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Rasterize taluks polygon onto this grid
    transform = Affine(res, 0.0, float(dem_ds.x.min()) - res/2, 0.0, -res, float(dem_ds.y.max()) + res/2)
    taluk_mask = rasterize([(total_geom, 1)], out_shape=elev.shape, transform=transform, fill=0, dtype=np.uint8) == 1

    high_elev = elev > max_elev
    steep = slope_deg > max_slope
    tree = (wc == 10)

    # Connected tree cover analysis:
    # 1. Label all tree clusters
    labeled_tree, _ = label(tree)
    # 2. Find components that overlap high_elev (hills)
    overlap_labels = np.unique(labeled_tree[high_elev])
    overlap_labels = overlap_labels[overlap_labels > 0]
    hill_tree = np.isin(labeled_tree, overlap_labels)

    # 3. Banana-safe constraint: ONLY exclude tree cover that is BOTH hill-connected AND elev > 120m
    excluded_tree = hill_tree & (elev > tree_elev_th)
    # Tree cover below 120m that stayed IN:
    tree_below_120_kept = tree & (~excluded_tree) & taluk_mask

    excluded_mask = (high_elev | steep | excluded_tree) & taluk_mask
    analyzed_mask = taluk_mask & (~excluded_mask)

    px_area_km2 = (res * res) / 1e6
    total_taluk_km2 = float(np.sum(taluk_mask) * px_area_km2)
    excluded_km2 = float(np.sum(excluded_mask) * px_area_km2)
    analyzed_km2 = float(np.sum(analyzed_mask) * px_area_km2)
    tree_kept_km2 = float(np.sum(tree_below_120_kept) * px_area_km2)

    # Calculate per-taluk breakdown
    per_taluk = {}
    for idx, row in gdf_utm.iterrows():
        t_name = row["name"]
        t_mask = rasterize([(row.geometry, 1)], out_shape=elev.shape, transform=transform, fill=0, dtype=np.uint8) == 1
        t_total = float(np.sum(t_mask) * px_area_km2)
        t_excl = float(np.sum(t_mask & excluded_mask) * px_area_km2)
        t_anal = float(np.sum(t_mask & analyzed_mask) * px_area_km2)
        per_taluk[t_name] = {
            "total_km2": round(t_total, 2),
            "excluded_km2": round(t_excl, 2),
            "analyzed_km2": round(t_anal, 2),
        }

    stats = {
        "total_taluk_km2": round(total_taluk_km2, 2),
        "excluded_km2": round(excluded_km2, 2),
        "analyzed_km2": round(analyzed_km2, 2),
        "tree_below_120_kept_km2": round(tree_kept_km2, 2),
        "per_taluk": per_taluk,
        "resolution_m": res,
    }

    logger.info("Mask computed: Total=%.2f km², Excluded=%.2f km², Analysed=%.2f km², Tree <120m kept=%.2f km²",
                total_taluk_km2, excluded_km2, analyzed_km2, tree_kept_km2)

    # Save analysis_mask.tif: 1 = analysed, 0 = excluded, 255 = outside taluk
    out_raster = np.full(elev.shape, 255, dtype=np.uint8)
    out_raster[taluk_mask & (~excluded_mask)] = 1
    out_raster[taluk_mask & excluded_mask] = 0

    with rasterio.open(
        mask_tif_path,
        "w",
        driver="GTiff",
        height=elev.shape[0],
        width=elev.shape[1],
        count=1,
        dtype=np.uint8,
        crs="EPSG:32643",
        transform=transform,
        nodata=255,
        compress="deflate",
    ) as dst:
        dst.write(out_raster, 1)

    # Save excluded_forest as simplified GeoJSON polygons for light grey hatched UI overlay
    logger.info("Vectorizing excluded forest mask for UI...")
    excl_uint8 = (excluded_mask).astype(np.uint8)
    excl_shapes = [
        shape(geom) for geom, val in shapes(excl_uint8, mask=(excl_uint8 == 1), transform=transform)
    ]
    if excl_shapes:
        excl_union = unary_union(excl_shapes)
        # Simplify slightly (30m tolerance) to keep GeoJSON compact and fast in maplibre
        excl_simplified = excl_union.simplify(30.0)
        excl_gdf = gpd.GeoDataFrame([{"geometry": excl_simplified}], crs="EPSG:32643").to_crs("EPSG:4326")
        excl_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "label": "Excluded: forest / hills (Western Ghats)",
                        "area_km2": round(excluded_km2, 2),
                    },
                    "geometry": mapping(excl_gdf.geometry.iloc[0]),
                }
            ]
        }
    else:
        excl_geojson = {"type": "FeatureCollection", "features": []}

    with open(excluded_geojson_path, "w") as f:
        json.dump(excl_geojson, f)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats
