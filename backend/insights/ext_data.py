"""
ext_data.py — External data acquisition for the Groundwater & Recharge and Geo-Economic modules.

Every layer is cached in data/ext/ on the canonical analysis-mask grid (EPSG:32643) and recorded in
data/ext/manifest.json with its source URL, retrieval date and status (ok / fallback / failed).
Nothing is faked: if a source fails, the manifest records the failure and the consuming module
applies the documented fallback and states it in Limitations.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

logger = logging.getLogger(__name__)

EXT = Path("data/ext")
MASK = Path("data/analysis_mask.tif")
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ------------------------------------------------------------------ helpers
def grid():
    with rasterio.open(MASK) as s:
        return s.read(1), s.transform, s.crs, s.shape, s.bounds


def bbox_wgs84(pad_deg: float = 0.0):
    from rasterio.warp import transform_bounds
    _, _, crs, _, b = grid()
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *b)
    return (w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg)


def manifest_load() -> dict:
    p = EXT / "manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def manifest_set(key: str, url: str, status: str, note: str = "", **extra):
    EXT.mkdir(parents=True, exist_ok=True)
    m = manifest_load()
    m[key] = {"source_url": url, "retrieved": datetime.now().isoformat(timespec="seconds"),
              "status": status, "note": note, **extra}
    (EXT / "manifest.json").write_text(json.dumps(m, indent=2))


def save_grid_tif(path: Path, arrs, names=None, dtype="float32", nodata=np.nan):
    _, tfm, crs, (ny, nx), _ = grid()
    if isinstance(arrs, np.ndarray) and arrs.ndim == 2:
        arrs = [arrs]
    with rasterio.open(path, "w", driver="GTiff", height=ny, width=nx, count=len(arrs),
                       dtype=dtype, crs=crs, transform=tfm, compress="deflate", nodata=nodata) as d:
        for i, a in enumerate(arrs, 1):
            d.write(a.astype(dtype), i)
        if names:
            d.update_tags(**{f"band_{i}": n for i, n in enumerate(names, 1)})


def read_grid_tif(path: Path):
    with rasterio.open(path) as s:
        return s.read().astype("float32"), s.tags()


def _geobox():
    from odc.geo.geobox import GeoBox
    _, tfm, crs, shape, _ = grid()
    return GeoBox(shape, tfm, str(crs))


def _pc_catalog():
    import pystac_client
    import planetary_computer
    return pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)


def _retry(fn, n=3):
    for i in range(n):
        try:
            return fn()
        except Exception as e:  # noqa
            if i == n - 1:
                raise
            logger.warning("retry %d: %s", i + 1, e)
            time.sleep(3 * (i + 1))


# ------------------------------------------------------------------ DEM
def fetch_dem(force=False):
    out = EXT / "dem_glo30.tif"
    if out.exists() and not force:
        return out
    import odc.stac
    cat = _pc_catalog()
    items = cat.search(collections=["cop-dem-glo-30"], bbox=bbox_wgs84(0.02)).item_collection()
    ds = _retry(lambda: odc.stac.load(items, bands=["data"], geobox=_geobox(), resampling="bilinear").compute())
    dem = ds["data"].max("time").values.astype("float32") if "time" in ds.dims else ds["data"].values
    save_grid_tif(out, dem, ["elevation_m"])
    manifest_set("dem", "https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30", "ok",
                 f"{len(items)} tiles, bilinear to mask grid")
    return out


# ------------------------------------------------------------------ Sentinel-2
def _s2_load(start, end, cloud=40, bands=("B04", "B08", "B11", "SCL")):
    import odc.stac
    cat = _pc_catalog()
    items = cat.search(collections=["sentinel-2-l2a"], bbox=bbox_wgs84(), datetime=f"{start}/{end}",
                       query={"eo:cloud_cover": {"lt": cloud}}).item_collection()
    if len(items) == 0:
        return None, 0
    ds = _retry(lambda: odc.stac.load(items, bands=list(bands), geobox=_geobox(), groupby="solar_day",
                                      chunks={"x": 1024, "y": 1024}, resampling="average").compute())
    return ds, len(items)


def _s2_ndvi(ds):
    scl = ds["SCL"]
    good = ~scl.isin([0, 1, 3, 8, 9, 10, 11])
    red = ds["B04"].where(good).astype("float32")
    nir = ds["B08"].where(good).astype("float32")
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi, good


def fetch_s2_dry(force=False):
    """Dry-season (Mar–May 2026) monthly median NDVI + median B11/B08 ratio."""
    out = EXT / "s2_dry_2026.tif"
    if out.exists() and not force:
        return out
    ds, n = _s2_load("2026-03-01", "2026-05-31", cloud=40)
    if ds is None:
        manifest_set("s2_dry", PC_STAC, "failed", "no scenes")
        return None
    ndvi, good = _s2_ndvi(ds)
    months = ds.time.dt.month
    layers, names = [], []
    for m in (3, 4, 5):
        sel = ndvi.where(months == m, drop=True)
        layers.append(sel.median("time").values if sel.sizes.get("time", 0) else np.full(ndvi.shape[1:], np.nan))
        names.append(f"ndvi_2026_{m:02d}")
    b11 = ds["B11"].where(good).astype("float32")
    b08 = ds["B08"].where(good).astype("float32")
    layers.append((b11 / (b08 + 1e-6)).median("time").values)
    names.append("b11_b08_ratio")
    save_grid_tif(out, layers, names)
    manifest_set("s2_dry", "https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a", "ok",
                 f"{n} scenes Mar–May 2026, SCL masked, monthly medians", dates=sorted({str(t)[:10] for t in ds.time.values}))
    return out


def fetch_s2_peaks(force=False):
    """Seasonal peak NDVI (90th pct of Oct–Feb scenes) per agricultural year, for stability (CV)."""
    out = EXT / "s2_peak_ndvi_ay.tif"
    if out.exists() and not force:
        return out
    layers, names, notes = [], [], []
    for ay, (s, e) in {"AY2023-24": ("2023-10-01", "2024-02-29"),
                       "AY2024-25": ("2024-10-01", "2025-02-28"),
                       "AY2025-26": ("2025-10-01", "2026-02-28")}.items():
        try:
            ds, n = _s2_load(s, e, cloud=30, bands=("B04", "B08", "SCL"))
            if ds is None:
                notes.append(f"{ay}: no scenes")
                continue
            ndvi, _ = _s2_ndvi(ds)
            layers.append(ndvi.quantile(0.9, dim="time", skipna=True).values.astype("float32"))
            names.append(ay)
            notes.append(f"{ay}: {n} scenes")
            del ds, ndvi
        except Exception as ex:
            notes.append(f"{ay}: failed {ex}")
    if not layers:
        manifest_set("s2_peaks", PC_STAC, "failed", "; ".join(notes))
        return None
    save_grid_tif(out, layers, names)
    manifest_set("s2_peaks", "https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a", "ok", "; ".join(notes))
    return out


# ------------------------------------------------------------------ Sentinel-2 water occurrence
def fetch_s2_water_months(force=False):
    """Number of months (Jun 2025 – May 2026) with monthly median NDWI > 0.2, per pixel."""
    out = EXT / "s2_water_months.tif"
    if out.exists() and not force:
        return out
    import pandas as pd
    _, _, _, (ny, nx), _ = grid()
    count = np.zeros((ny, nx), "float32")
    valid = np.zeros((ny, nx), "float32")
    notes = []
    for ms in pd.date_range("2025-06-01", "2026-05-01", freq="MS"):
        me = (ms + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        try:
            ds, n = _s2_load(ms.strftime("%Y-%m-%d"), me, cloud=60, bands=("B03", "B08", "SCL"))
            if ds is None:
                notes.append(f"{ms:%Y-%m}:0")
                continue
            good = ~ds["SCL"].isin([0, 1, 3, 8, 9, 10, 11])
            g = ds["B03"].where(good).astype("float32")
            nr = ds["B08"].where(good).astype("float32")
            ndwi = ((g - nr) / (g + nr + 1e-6)).median("time").values
            count += np.nan_to_num(ndwi > 0.2).astype("float32")
            valid += (~np.isnan(ndwi)).astype("float32")
            notes.append(f"{ms:%Y-%m}:{n}")
            del ds
        except Exception as ex:
            notes.append(f"{ms:%Y-%m}:fail({str(ex)[:40]})")
    save_grid_tif(out, [count, valid], ["water_months", "valid_months"])
    manifest_set("s2_water_months", "https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a", "ok",
                 "monthly median NDWI > 0.2 count; scenes per month " + ", ".join(notes))
    return out


# ------------------------------------------------------------------ Sentinel-1 Dec 2023 flood
def fetch_s1_flood(force=False):
    out = EXT / "s1_flood_2023.tif"
    if out.exists() and not force:
        return out
    import odc.stac
    cat = _pc_catalog()
    res = {}
    for tag, (s, e) in {"pre": ("2023-11-01", "2023-11-30"), "post": ("2023-12-17", "2023-12-31")}.items():
        items = cat.search(collections=["sentinel-1-rtc"], bbox=bbox_wgs84(), datetime=f"{s}/{e}").item_collection()
        dates = sorted({it.datetime.strftime("%Y-%m-%d") for it in items})
        if len(items) == 0:
            res[tag] = (None, dates)
            continue
        ds = _retry(lambda: odc.stac.load(items, bands=["vv", "vh"], geobox=_geobox(), groupby="solar_day",
                                          chunks={"x": 1024, "y": 1024}, resampling="average").compute())
        vv = ds["vv"].where(ds["vv"] > 0).median("time").values
        vh = ds["vh"].where(ds["vh"] > 0).median("time").values
        res[tag] = ((10 * np.log10(vv)).astype("float32"), (10 * np.log10(vh)).astype("float32"), dates)
    pre, post = res.get("pre"), res.get("post")
    if pre is None or pre[0] is None or post is None or post[0] is None:
        manifest_set("s1_flood", "https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc", "failed",
                     "missing pre or post scenes", pre_dates=pre[-1] if pre else [], post_dates=post[-1] if post else [])
        return None
    save_grid_tif(out, [pre[0], pre[1], post[0], post[1]], ["pre_vv_db", "pre_vh_db", "post_vv_db", "post_vh_db"])
    manifest_set("s1_flood", "https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc", "ok",
                 "median dB composites", pre_dates=pre[2], post_dates=post[2])
    return out


# ------------------------------------------------------------------ SoilGrids
def fetch_soilgrids(force=False):
    out = EXT / "soilgrids_0_30cm.tif"
    if out.exists() and not force:
        return out
    import requests
    w, s, e, n = bbox_wgs84(0.01)
    _, tfm, crs, (ny, nx), _ = grid()
    depths = [("0-5cm", 5), ("5-15cm", 10), ("15-30cm", 15)]
    result, urls = [], []
    for prop in ("sand", "clay", "silt"):
        acc = np.zeros((ny, nx), "float32")
        wsum = 0
        for d, wt in depths:
            url = (f"https://maps.isric.org/mapserv?map=/map/{prop}.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
                   f"&COVERAGEID={prop}_{d}_mean&FORMAT=image/tiff&SUBSET=long({w},{e})&SUBSET=lat({s},{n})"
                   f"&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
                   f"&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326")
            urls.append(url)
            r = _retry(lambda: requests.get(url, timeout=120))
            if r.status_code != 200 or not r.content[:4] in (b"II*\x00", b"MM\x00*"):
                raise RuntimeError(f"SoilGrids WCS failed for {prop} {d}: HTTP {r.status_code}")
            tmp = EXT / f"_sg_{prop}_{d}.tif"
            tmp.write_bytes(r.content)
            with rasterio.open(tmp) as src:
                dst = np.full((ny, nx), np.nan, "float32")
                reproject(src.read(1).astype("float32"), dst, src_transform=src.transform, src_crs=src.crs,
                          src_nodata=src.nodata if src.nodata is not None else 0, dst_transform=tfm, dst_crs=crs,
                          dst_nodata=np.nan, resampling=Resampling.bilinear)
            acc += np.nan_to_num(dst) * wt
            wsum += wt
        result.append(acc / wsum / 10.0)  # g/kg -> %
    save_grid_tif(out, result, ["sand_pct", "clay_pct", "silt_pct"])
    manifest_set("soilgrids", "https://maps.isric.org (ISRIC SoilGrids 2.0 WCS, 250 m)", "ok",
                 "depth-weighted mean 0–30 cm, bilinear to grid", example_url=urls[0])
    return out


# ------------------------------------------------------------------ CHIRPS
def fetch_chirps(force=False, years=range(2015, 2026)):
    out = EXT / "chirps_annual_mean.tif"
    if out.exists() and not force:
        return out
    from rasterio.windows import from_bounds as win_from_bounds
    w, s, e, n = bbox_wgs84(0.15)
    _, tfm, crs, (ny, nx), _ = grid()
    stack, used = [], []
    for y in years:
        url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_annual/tifs/chirps-v2.0.{y}.tif"
        try:
            with rasterio.Env(GDAL_HTTP_TIMEOUT=60, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open("/vsicurl/" + url) as src:
                    win = win_from_bounds(w, s, e, n, src.transform)
                    arr = src.read(1, window=win).astype("float32")
                    wt = src.window_transform(win)
                    arr[arr < 0] = np.nan
                    dst = np.full((ny, nx), np.nan, "float32")
                    reproject(arr, dst, src_transform=wt, src_crs=src.crs, src_nodata=np.nan, dst_transform=tfm,
                              dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
                    stack.append(dst)
                    used.append(y)
        except Exception as ex:
            logger.warning("CHIRPS %s failed: %s", y, ex)
    if not stack:
        manifest_set("chirps", "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_annual/tifs/", "failed",
                     "all years failed; IMD fallback required")
        return None
    mean = np.nanmean(np.stack(stack), axis=0)
    save_grid_tif(out, mean, ["annual_rain_mm"])
    manifest_set("chirps", "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_annual/tifs/", "ok",
                 f"annual mean of {used[0]}–{used[-1]} ({len(used)} yrs), bilinear", years=used)
    return out


# ------------------------------------------------------------------ OSM
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]


def _overpass(query: str):
    import requests
    last = None
    for ep in OVERPASS:
        try:
            r = requests.post(ep, data={"data": query}, headers={"User-Agent": "Kalam/1.0"}, timeout=180)
            if r.status_code == 200:
                return r.json(), ep
            last = f"HTTP {r.status_code}"
        except Exception as ex:
            last = str(ex)
    raise RuntimeError(f"Overpass failed: {last}")


def fetch_osm(force=False):
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    roads_p, places_p, rail_p, power_p = (EXT / "osm_roads.geojson", EXT / "osm_places.geojson",
                                          EXT / "osm_rail.geojson", EXT / "osm_power.geojson")
    if roads_p.exists() and places_p.exists() and not force:
        return roads_p, places_p
    w, s, e, n = bbox_wgs84(0.05)
    bb = f"{s},{w},{n},{e}"
    q = f"""[out:json][timeout:170];
    (way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|track|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|service"]({bb}););
    out geom;"""
    data, ep = _overpass(q)
    rows = []
    for el in data.get("elements", []):
        g = el.get("geometry")
        if el["type"] == "way" and g and len(g) >= 2:
            rows.append({"highway": el.get("tags", {}).get("highway", ""),
                         "geometry": LineString([(p["lon"], p["lat"]) for p in g])})
    gpd.GeoDataFrame(rows, crs="EPSG:4326").to_file(roads_p, driver="GeoJSON")
    manifest_set("osm_roads", ep, "ok", f"{len(rows)} highway ways (out geom)", bbox=bb)

    # rail + power lines
    q2 = f"""[out:json][timeout:120];(way["railway"~"rail|narrow_gauge"]({bb});way["power"~"line|minor_line"]({bb}););out geom;"""
    try:
        d2, ep2 = _overpass(q2)
        rail, power = [], []
        for el in d2.get("elements", []):
            g = el.get("geometry")
            if not g or len(g) < 2:
                continue
            ls = LineString([(p["lon"], p["lat"]) for p in g])
            t = el.get("tags", {})
            (rail if "railway" in t else power).append({"kind": t.get("railway") or t.get("power"), "geometry": ls})
        gpd.GeoDataFrame(rail or [{"kind": None, "geometry": None}], crs="EPSG:4326").dropna().to_file(rail_p, driver="GeoJSON") if rail else None
        if power:
            gpd.GeoDataFrame(power, crs="EPSG:4326").to_file(power_p, driver="GeoJSON")
        manifest_set("osm_rail_power", ep2, "ok", f"{len(rail)} rail ways, {len(power)} power lines")
    except Exception as ex:
        manifest_set("osm_rail_power", OVERPASS[0], "failed", str(ex))

    # towns + markets within 20 km (~0.18 deg) of taluks, incl. Tirunelveli city
    w2, s2, e2, n2 = bbox_wgs84(0.20)
    bb2 = f"{s2},{w2},{n2},{e2}"
    q3 = f"""[out:json][timeout:120];(
      node["place"~"city|town"]({bb2}); way["place"~"city|town"]({bb2});
      node["amenity"="marketplace"]({bb2}); way["amenity"="marketplace"]({bb2}););out center;"""
    d3, ep3 = _overpass(q3)
    pl = []
    for el in d3.get("elements", []):
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None:
            continue
        t = el.get("tags", {})
        pl.append({"name": t.get("name", ""), "kind": t.get("place") or t.get("amenity"), "geometry": Point(lon, lat)})
    gpd.GeoDataFrame(pl, crs="EPSG:4326").to_file(places_p, driver="GeoJSON")
    manifest_set("osm_places", ep3, "ok", f"{len(pl)} towns/cities/marketplaces within ~20 km", bbox=bb2)
    return roads_p, places_p


def fetch_all():
    EXT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("dem", fetch_dem), ("osm", fetch_osm), ("soilgrids", fetch_soilgrids),
                     ("chirps", fetch_chirps), ("s1_flood", fetch_s1_flood), ("s2_dry", fetch_s2_dry),
                     ("s2_peaks", fetch_s2_peaks)]:
        t0 = time.time()
        try:
            r = fn()
            print(f"[{name}] -> {r} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as ex:
            print(f"[{name}] FAILED: {ex}", flush=True)
            if name not in manifest_load() or manifest_load()[name].get("status") != "ok":
                manifest_set(name, "", "failed", str(ex)[:300])


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            print(a, globals()[f"fetch_{a}"](), flush=True)
    else:
        fetch_all()
