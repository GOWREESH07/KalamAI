"""
aoi.py — AOI validation and geometry helpers.
Computes area in EPSG:32643 and asserts >= 20 km².
"""
from __future__ import annotations
import yaml
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform
import json
import math


def load_config(config_path: str | Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_aoi_bbox(cfg: dict) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max)."""
    a = cfg["aoi"]
    return a["lon_min"], a["lat_min"], a["lon_max"], a["lat_max"]


def get_aoi_geometry(cfg: dict) -> dict:
    """Return AOI as GeoJSON geometry (WGS84)."""
    lon_min, lat_min, lon_max, lat_max = get_aoi_bbox(cfg)
    return mapping(box(lon_min, lat_min, lon_max, lat_max))


def compute_area_km2(cfg: dict) -> float:
    """Compute AOI area in km² using projected CRS EPSG:32643."""
    lon_min, lat_min, lon_max, lat_max = get_aoi_bbox(cfg)
    geom_wgs84 = box(lon_min, lat_min, lon_max, lat_max)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{cfg['aoi']['epsg']}", always_xy=True)
    geom_proj = transform(transformer.transform, geom_wgs84)
    area_m2 = geom_proj.area
    return area_m2 / 1e6


def validate_aoi(cfg: dict) -> float:
    """Assert area >= min_area_km2. Returns area_km2."""
    area = compute_area_km2(cfg)
    min_area = cfg["aoi"].get("min_area_km2", 20.0)
    if area < min_area:
        raise ValueError(f"AOI area {area:.2f} km² is below the required {min_area} km²")
    return area


def get_utm_bbox(cfg: dict) -> tuple[float, float, float, float]:
    """Return AOI bbox in EPSG:32643 (xmin, ymin, xmax, ymax)."""
    lon_min, lat_min, lon_max, lat_max = get_aoi_bbox(cfg)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{cfg['aoi']['epsg']}", always_xy=True)
    xmin, ymin = transformer.transform(lon_min, lat_min)
    xmax, ymax = transformer.transform(lon_max, lat_max)
    return xmin, ymin, xmax, ymax


if __name__ == "__main__":
    cfg = load_config()
    area = validate_aoi(cfg)
    print(f"AOI area: {area:.2f} km² ✓")
    print(f"Bbox WGS84: {get_aoi_bbox(cfg)}")
    print(f"Bbox UTM: {get_utm_bbox(cfg)}")
