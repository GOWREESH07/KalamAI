"""
classify.py — Random Forest classification + parcel voting + metrics.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, classification_report
)
from sklearn.model_selection import GroupKFold
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ndvi_max", "ndvi_min", "ndvi_mean", "ndvi_std", "ndvi_amplitude",
    "ndvi_months_high", "vh_min", "vh_mean", "vh_min_month",
]
# Add monthly ndvi/vh features
MONTHLY_FEATURES = ["ndvi", "vh"]  # will be expanded to ndvi_m00, ndvi_m01...

CLASS_NAMES = {
    1: "Paddy", 2: "Banana", 3: "Other Veg",
    4: "Built-up", 5: "Water", 6: "Fallow/Bare"
}


def _build_feature_matrix(feat_ds: xr.Dataset, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Build feature matrix for given pixel row/col indices."""
    static_feats = []
    for col in FEATURE_COLS:
        arr = feat_ds[col].values
        static_feats.append(arr[rows, cols])

    # Monthly NDVI
    ndvi_ts = feat_ds["ndvi"].values  # (M, Y, X)
    for m in range(ndvi_ts.shape[0]):
        static_feats.append(ndvi_ts[m, rows, cols])

    # Monthly VH
    vh_ts = feat_ds["vh"].values
    for m in range(vh_ts.shape[0]):
        static_feats.append(vh_ts[m, rows, cols])

    X = np.column_stack(static_feats)
    return X


def _build_full_feature_matrix(feat_ds: xr.Dataset) -> np.ndarray:
    """Build feature matrix for ALL pixels."""
    ny = feat_ds.dims.get("y", feat_ds["ndvi_max"].shape[0])
    nx = feat_ds.dims.get("x", feat_ds["ndvi_max"].shape[1])
    rows, cols = np.mgrid[0:ny, 0:nx]
    rows = rows.ravel()
    cols = cols.ravel()
    return _build_feature_matrix(feat_ds, rows, cols), rows, cols, ny, nx


def run_classify(
    cfg: dict,
    cache_dir: Path,
    outputs_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Run classification. Returns paths and metrics dict."""
    classified_tif = outputs_dir / "classified.tif"
    metrics_path = outputs_dir / "metrics.json"
    parcels_gpkg = outputs_dir / "parcels.gpkg"
    parcels_geojson = outputs_dir / "parcels.geojson"
    stats_csv = outputs_dir / "area_stats.csv"

    if classified_tif.exists() and metrics_path.exists():
        logger.info("Classification cache hit")
        if progress_cb:
            progress_cb("Classification cache hit — skipping")
        with open(metrics_path) as f:
            metrics = json.load(f)
        return {
            "classified": str(classified_tif),
            "metrics": metrics,
            "parcels": str(parcels_gpkg),
            "parcels_geojson": str(parcels_geojson),
            "stats": str(stats_csv),
        }

    if progress_cb:
        progress_cb("Loading features and labels...")
    feat_ds = xr.open_dataset(cache_dir / "features.nc")
    df_labels = pd.read_csv(cache_dir / "labels.csv")

    clf_cfg = cfg["classification"]
    epsg = cfg["aoi"]["epsg"]

    if progress_cb:
        progress_cb(f"Building training features ({len(df_labels)} label pixels)...")

    rows = df_labels["row"].values.astype(int)
    cols = df_labels["col"].values.astype(int)
    y_labels = df_labels["class_id"].values.astype(int)

    X_all = _build_feature_matrix(feat_ds, rows, cols)
    # Replace NaN with median
    col_medians = np.nanmedian(X_all, axis=0)
    for j in range(X_all.shape[1]):
        mask = np.isnan(X_all[:, j])
        X_all[mask, j] = col_medians[j]

    # Check for manual validation split
    weights = df_labels["weight"].values if "weight" in df_labels.columns else np.ones(len(df_labels))
    splits = df_labels["split"].values if "split" in df_labels.columns else np.array(["train"] * len(df_labels))
    sources = df_labels["source"].values if "source" in df_labels.columns else np.array(["phenology"] * len(df_labels))

    has_manual_val = np.any((splits == "val") & (sources == "manual"))

    if has_manual_val:
        train_mask = (splits == "train")
        val_mask = (splits == "val") & (sources == "manual")
        metric_title = "Field Accuracy (Manual Ground Truth)"
        metric_explanation = "Evaluated exclusively on held-out manual validation polygons separate from training seeds."
        is_seed_agreement = False
    else:
        # Spatial block split
        ny_grid = feat_ds.dims.get("y", feat_ds["ndvi_max"].shape[0])
        nx_grid = feat_ds.dims.get("x", feat_ds["ndvi_max"].shape[1])
        block_size_px = max(1, int(clf_cfg["block_size_m"] / 10))
        block_ids = (rows // block_size_px) * (nx_grid // block_size_px + 1) + (cols // block_size_px)

        unique_blocks = np.unique(block_ids)
        np.random.seed(clf_cfg["random_state"])
        np.random.shuffle(unique_blocks)
        n_test = max(1, int(len(unique_blocks) * clf_cfg["test_split"]))
        test_blocks = set(unique_blocks[:n_test])
        train_mask = np.array([bid not in test_blocks for bid in block_ids])
        val_mask = ~train_mask
        metric_title = "Agreement with Seed Labels"
        metric_explanation = "No manual validation polygons provided yet; score reflects consistency against phenological rule seeds."
        is_seed_agreement = True

    X_train = X_all[train_mask]
    y_train = y_labels[train_mask]
    w_train = weights[train_mask]

    X_test = X_all[val_mask]
    y_test = y_labels[val_mask]

    if progress_cb:
        progress_cb(f"Training RF ({clf_cfg['n_estimators']} trees, {X_train.shape[0]} train px)...")

    rf = RandomForestClassifier(
        n_estimators=clf_cfg["n_estimators"],
        n_jobs=-1,
        random_state=clf_cfg["random_state"],
        class_weight="balanced",
    )
    rf.fit(X_train, y_train, sample_weight=w_train)

    if progress_cb:
        progress_cb(f"Evaluating validation set ({metric_title})...")

    y_pred = rf.predict(X_test)
    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=list(CLASS_NAMES.keys()))
    f1 = f1_score(y_test, y_pred, labels=list(CLASS_NAMES.keys()), average=None, zero_division=0)
    feature_importance = rf.feature_importances_.tolist()
    # Seed vs Manual counts per class
    counts_per_class = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        n_seed = int(np.sum((y_labels == cls_id) & (sources == "phenology")))
        n_manual = int(np.sum((y_labels == cls_id) & (sources == "manual")))
        counts_per_class[cls_name] = {
            "seed": n_seed,
            "manual": n_manual,
            "total": n_seed + n_manual,
        }
    logger.info("Training label breakdown (seed vs manual): %s", counts_per_class)
    if progress_cb:
        for cname, ccounts in counts_per_class.items():
            if ccounts["manual"] > 0:
                progress_cb(f"  {cname}: {ccounts['manual']} manual (3x wt) + {ccounts['seed']} seeds")

    metrics = {
        "overall_accuracy": float(oa),
        "kappa": float(kappa),
        "metric_title": metric_title,
        "metric_explanation": metric_explanation,
        "is_seed_agreement": is_seed_agreement,
        "n_manual_val": int(np.sum(val_mask)) if has_manual_val else 0,
        "counts_per_class": counts_per_class,
        "confusion_matrix": cm.tolist(),
        "f1_per_class": {CLASS_NAMES[cls]: float(f) for cls, f in zip(CLASS_NAMES.keys(), f1)},
        "feature_importance": feature_importance,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "class_names": CLASS_NAMES,
    }

    logger.info("OA=%.3f Kappa=%.3f", oa, kappa)
    if progress_cb:
        progress_cb(f"OA={oa:.3f} Kappa={kappa:.3f}")

    # --- Predict full scene ---
    if progress_cb:
        progress_cb("Predicting all pixels...")
    X_full, all_rows, all_cols, ny, nx = _build_full_feature_matrix(feat_ds)
    # NaN fill
    for j in range(X_full.shape[1]):
        mask_nan = np.isnan(X_full[:, j])
        X_full[mask_nan, j] = col_medians[j]

    pred_flat = rf.predict(X_full)
    prob_flat = rf.predict_proba(X_full)
    conf_flat = prob_flat.max(axis=1)
    classified = pred_flat.reshape(ny, nx).astype(np.uint8)
    confidence = conf_flat.reshape(ny, nx).astype(np.float32)

    # --- Post-classification refinement for Paddy ---
    # Reassign any "paddy" pixel with max NDVI < 0.5 to Fallow/Bare or Other vegetation,
    # whichever its features are closest to.
    ndvi_max_2d = feat_ds["ndvi_max"].values
    paddy_low_ndvi = (classified == 1) & (ndvi_max_2d < 0.50)
    n_reassigned = int(np.sum(paddy_low_ndvi))

    if n_reassigned > 0:
        logger.info("Reassigning %d Paddy pixels with max NDVI < 0.50 to Fallow/Bare or Other Veg", n_reassigned)
        # Compare distances to Fallow (class 6) vs Other Veg (class 3) centroids
        fallow_mask_tr = (y_labels == 6)
        ov_mask_tr = (y_labels == 3)
        if np.any(fallow_mask_tr) and np.any(ov_mask_tr):
            centroid_fal = np.median(X_all[fallow_mask_tr], axis=0)
            centroid_ov = np.median(X_all[ov_mask_tr], axis=0)
            # Standardize distances using feature scales
            feat_scale = np.nanstd(X_all, axis=0)
            feat_scale = np.where(feat_scale < 1e-6, 1.0, feat_scale)

            # Get indices of pixels to reassign
            reassign_idx = np.where(paddy_low_ndvi.ravel())[0]
            X_reassign = X_full[reassign_idx]
            dist_fal = np.linalg.norm((X_reassign - centroid_fal) / feat_scale, axis=1)
            dist_ov = np.linalg.norm((X_reassign - centroid_ov) / feat_scale, axis=1)

            new_classes = np.where(dist_fal < dist_ov, 6, 3).astype(np.uint8)
            classified_flat = classified.ravel()
            classified_flat[reassign_idx] = new_classes
            classified = classified_flat.reshape(ny, nx)
        else:
            # Fallback heuristic based on NDVI mean
            ndvi_mean_2d = feat_ds["ndvi_mean"].values
            classified = np.where(paddy_low_ndvi, np.where(ndvi_mean_2d > 0.35, 3, 6), classified).astype(np.uint8)

    # --- Save classified raster ---
    y_coords = feat_ds.y.values
    x_coords = feat_ds.x.values
    res = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 10.0
    xmin, xmax = float(x_coords.min()), float(x_coords.max())
    ymin, ymax = float(y_coords.min()), float(y_coords.max())
    transform = from_bounds(xmin - res/2, ymin - res/2, xmax + res/2, ymax + res/2, nx, ny)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        classified_tif, "w",
        driver="GTiff", height=ny, width=nx,
        count=2, dtype=np.float32,
        crs=CRS.from_epsg(epsg),
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(classified.astype(np.float32), 1)
        dst.write(confidence, 2)
        dst.update_tags(band_1="class_id", band_2="confidence")

    if progress_cb:
        progress_cb("Assigning classes to parcels...")

    # --- Parcel voting ---
    seg_path = outputs_dir / "segments.gpkg"
    if seg_path.exists():
        parcels = gpd.read_file(seg_path)
        with rasterio.open(classified_tif) as src:
            tfm = src.transform

        parcel_classes = []
        parcel_confs = []
        for idx, row in parcels.iterrows():
            from rasterio.features import geometry_mask
            msk = ~geometry_mask(
                [row.geometry.__geo_interface__],
                out_shape=(ny, nx),
                transform=transform,
                invert=False,
            )
            px_classes = classified[msk]
            if len(px_classes) == 0:
                parcel_classes.append(0)
                parcel_confs.append(0.0)
                continue
            counts = np.bincount(px_classes.astype(int), minlength=7)
            maj_class = int(counts[1:].argmax()) + 1
            maj_share = counts[maj_class] / max(1, counts[1:].sum())
            parcel_classes.append(maj_class)
            parcel_confs.append(float(maj_share))

        parcels["class_id"] = parcel_classes
        parcels["class_name"] = parcels["class_id"].map(CLASS_NAMES)
        parcels["confidence"] = parcel_confs
        parcels["area_ha"] = parcels.geometry.area / 1e4
        parcels.to_file(parcels_gpkg, driver="GPKG")
        parcels_wgs = parcels.to_crs("EPSG:4326")
        parcels_wgs.to_file(parcels_geojson, driver="GeoJSON")
    else:
        logger.warning("Segments file not found; skipping parcel voting")

    # --- Area stats ---
    class_areas = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        n_px = int(np.sum(classified == cls_id))
        area_ha = n_px * 100 / 1e4  # 10m px = 100 m²
        class_areas[cls_name] = area_ha

    df_stats = pd.DataFrame(
        [{"class": k, "area_ha": v, "area_km2": v / 100} for k, v in class_areas.items()]
    )
    df_stats.to_csv(stats_csv, index=False)
    metrics["area_stats"] = class_areas

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if progress_cb:
        progress_cb("Classification complete.")

    return {
        "classified": str(classified_tif),
        "metrics": metrics,
        "parcels": str(parcels_gpkg),
        "parcels_geojson": str(parcels_geojson),
        "stats": str(stats_csv),
    }
