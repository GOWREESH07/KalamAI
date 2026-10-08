"""
export.py — Print-ready thematic map and file exports.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
from matplotlib.ticker import MultipleLocator
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xarray as xr

logger = logging.getLogger(__name__)

CLASS_COLORS = {
    1: "#8DB33F",  # Paddy
    2: "#2E6B3F",  # Banana
    3: "#BFCB8F",  # Other veg
    4: "#B4554B",  # Built-up
    5: "#3A7CA5",  # Water
    6: "#CBA670",  # Fallow/Bare
}
CLASS_NAMES = {
    1: "Paddy", 2: "Banana", 3: "Other Veg",
    4: "Built-up", 5: "Water", 6: "Fallow/Bare"
}


def _hex_to_rgb(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def render_rgb_png(cfg: dict, cache_dir: Path, outputs_dir: Path) -> Path:
    """Render a true-color RGB PNG from S2 monthly median."""
    out_path = outputs_dir / "rgb.png"
    if out_path.exists():
        return out_path
    s2 = xr.open_dataset(cache_dir / "s2_monthly.nc")
    # Use median across all months
    red = np.nanmedian(s2["B04"].values, axis=0)
    green = np.nanmedian(s2["B03"].values, axis=0)
    blue = np.nanmedian(s2["B02"].values, axis=0)
    # Normalize to 0-255
    def norm(arr):
        p2, p98 = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
        return np.clip((arr - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([norm(red), norm(green), norm(blue)], axis=-1)
    from PIL import Image
    img = Image.fromarray(rgb)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def render_ndvi_png(cfg: dict, cache_dir: Path, outputs_dir: Path, month_idx: int = 0) -> Path:
    """Render NDVI for a given month index as PNG."""
    out_path = outputs_dir / f"ndvi_month_{month_idx:02d}.png"
    if out_path.exists():
        return out_path
    feat_ds = xr.open_dataset(cache_dir / "features.nc")
    ndvi = feat_ds["ndvi"].values[month_idx]
    ndvi_norm = np.clip((ndvi + 0.2) / 1.2, 0, 1)
    cmap = plt.get_cmap("RdYlGn")
    rgba = cmap(ndvi_norm)
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    from PIL import Image
    img = Image.fromarray(rgba_uint8)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def render_classified_png(cfg: dict, outputs_dir: Path) -> Path:
    """Render classified map as coloured PNG."""
    out_path = outputs_dir / "classified_rgb.png"
    if out_path.exists():
        return out_path
    classified_tif = outputs_dir / "classified.tif"
    if not classified_tif.exists():
        raise FileNotFoundError("classified.tif not found")
    with rasterio.open(classified_tif) as src:
        classified = src.read(1)
    ny, nx = classified.shape
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    for cls_id, hex_color in CLASS_COLORS.items():
        r, g, b = _hex_to_rgb(hex_color)
        mask = classified == cls_id
        rgb[mask] = [int(r * 255), int(g * 255), int(b * 255)]
    from PIL import Image
    img = Image.fromarray(rgb)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def export_map(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Export thematic map as PNG and PDF."""
    map_png = outputs_dir / "thematic_map.png"
    map_pdf = outputs_dir / "thematic_map.pdf"

    if map_png.exists() and map_pdf.exists():
        if progress_cb:
            progress_cb("Map export cache hit")
        return {"map_png": str(map_png), "map_pdf": str(map_pdf)}

    if progress_cb:
        progress_cb("Rendering thematic map...")

    classified_tif = outputs_dir / "classified.tif"
    if not classified_tif.exists():
        raise FileNotFoundError("classified.tif not found")

    with rasterio.open(classified_tif) as src:
        classified = src.read(1)
        bounds = src.bounds
        crs = src.crs

    ny, nx = classified.shape
    rgb_img = np.zeros((ny, nx, 3), dtype=np.uint8)
    for cls_id, hex_color in CLASS_COLORS.items():
        r, g, b = _hex_to_rgb(hex_color)
        mask = classified == cls_id
        rgb_img[mask] = [int(r * 255), int(g * 255), int(b * 255)]

    export_cfg = cfg.get("export", {})
    dpi = export_cfg.get("dpi", 150)

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#F5F4F0")
    ax.imshow(
        rgb_img,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
    )
    ax.set_facecolor("#F5F4F0")

    # Grid
    ax.grid(True, color="white", linewidth=0.3, alpha=0.5, linestyle="--")
    ax.tick_params(labelsize=8)

    # Labels
    ax.set_xlabel("Easting (m, EPSG:32643)", fontsize=9)
    ax.set_ylabel("Northing (m, EPSG:32643)", fontsize=9)
    ax.set_title(
        export_cfg.get("map_title", "Land Cover Classification"),
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.text(
        0.5, 1.02,
        export_cfg.get("map_subtitle", ""),
        transform=ax.transAxes, ha="center", fontsize=10, color="#6B6862",
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=hex_color, label=CLASS_NAMES[cls_id])
        for cls_id, hex_color in CLASS_COLORS.items()
    ]
    legend = ax.legend(
        handles=legend_patches,
        loc="lower right",
        title="Land Cover",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.9,
        edgecolor="#E2DED5",
    )

    # North arrow (approximate)
    ax.annotate(
        "", xy=(0.97, 0.12), xytext=(0.97, 0.05),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#1C1C1A", lw=1.5),
    )
    ax.text(0.97, 0.13, "N", transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")

    # Scale bar (approximate: 2 km)
    x_range = bounds.right - bounds.left
    scale_len = 2000  # 2 km in meters
    x0 = bounds.left + x_range * 0.05
    y0 = bounds.bottom + (bounds.top - bounds.bottom) * 0.04
    ax.plot([x0, x0 + scale_len], [y0, y0], color="#1C1C1A", lw=2)
    ax.text(x0 + scale_len / 2, y0 + (bounds.top - bounds.bottom) * 0.01,
            "2 km", ha="center", va="bottom", fontsize=8)

    # Credits
    credits = export_cfg.get("credits", "Contains modified Copernicus Sentinel data 2025–26")
    fig.text(0.5, 0.01, credits, ha="center", fontsize=7, color="#6B6862", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(map_png, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(map_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    if progress_cb:
        progress_cb("Map exported.")
    return {"map_png": str(map_png), "map_pdf": str(map_pdf)}


def run_export(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Run all exports."""
    results = {}
    try:
        results["rgb_png"] = str(render_rgb_png(cfg, cache_dir, outputs_dir))
    except Exception as e:
        logger.warning("RGB export failed: %s", e)
    try:
        results["ndvi_png"] = str(render_ndvi_png(cfg, cache_dir, outputs_dir, 0))
    except Exception as e:
        logger.warning("NDVI export failed: %s", e)
    try:
        results["classified_png"] = str(render_classified_png(cfg, outputs_dir))
    except Exception as e:
        logger.warning("Classified PNG export failed: %s", e)
    map_result = export_map(cfg, cache_dir, outputs_dir, progress_cb)
    results.update(map_result)
    return results
