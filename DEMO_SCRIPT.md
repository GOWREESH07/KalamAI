# களம் 🌾 Demo Script — Walkthrough (VIT Mapathon 2026)

## 00:00 – 00:45 | Landing Page & Sensor Fusion Pipeline
- Start at the root URL `/`: **களம் 🌾 — Paddy & Banana Parcel Mapping**.
- "Good afternoon judges. This is **களம் 🌾**, built for Problem Statement 1 of the VIT Mapathon 2026."
- Scroll through the landing page:
  - **Study area stats**: Ambasamudram & Cheranmahadevi farmed plains (548.7 km²) with excluded Western Ghats forest (634.7 km²).
  - **Early Sensor Fusion section**: Aligned 10m grid, 40-month time series, stacked per-pixel features, and Random Forest Gini importance showing optical vs radar synergy.
  - Click **"Go to களம் 🌾 →"** to enter `/app`.

## 00:45 – 01:30 | Screen 01 & 02: Study Area & Multi-Temporal Imagery
- Point out the headline metrics and dual study-area toggle (**Pilot 33 km²** vs **Full Taluks ~1,183 km²**).
- Switch to **02 Study Area & Data**:
  - Toggle between True Colour RGB, Monthly NDVI, and SAR VH backscatter overlays.
  - Demonstrate interactive Basemap switching (**Esri World Imagery**, **Esri Clarity**, **OSM**) with zero grey tile errors at zoom 19.

## 01:30 – 02:30 | Screen 03: Multi-Year Crop Signatures & Phenology
- Switch to **03 Crop Signatures**:
  - Show the 40-month time series across 3 Agricultural Years (AY 2023-24, AY 2024-25, AY 2025-26).
  - Highlight the shaded **Kar** (May–Aug) and **Pishanam** (Sep–Dec) cropping seasons.
  - Click on a field parcel to extract real-time pixel profiles via the `/profile` API:
    - **Paddy:** Flooding dip in SAR VH (< −19 dB) followed by rapid vegetative surge and harvest drop.
    - **Banana:** Continuous high greenness (NDVI > 0.50) and elevated radar volume scattering (~−13 dB) without flood dips.

## 02:30 – 03:30 | Screen 05: Multi-Year Consensus Classification & Parcels
- Switch to **05 Classification**:
  - Toggle between **Parcel vector view** and **Pixel raster view**.
  - Show multi-year consistency (80.6% pixels identical across all 3 years) and high model validation accuracy (89.8%).
  - Inspect individual field parcels: class, confidence, consistency, rotation detection, and AY-by-AY crop history.

## 03:30 – 04:45 | Screen 07: Agricultural Insights Module (Phase 2)
- Switch to **07 Insights**:
  - Showcase the 9 isolated analytical modules on the left navigation strip:
    1. **Cropping Intensity:** 62.4% double-cropped (Kar + Pishanam dual season) vs perennial banana.
    2. **Transplanting Calendar:** Inundation timing and detection of delayed sowing windows.
    3. **Field Verification Priority:** Weighted index ($0.5(1-\text{conf}) + 0.3(1-\text{cons}) + 0.2\text{disag}$) generating top-50 GPS audit targets with KML mobile download.
    4. **Village Statistics:** Micro-administrative breakdown across Ambasamudram, Kallidaikurichi, Cheranmahadevi, etc.
    5. **Irrigation Canal Buffer:** Proximity decay showing banana concentration within 250m canal frontage.
    6. **Crop Change Dynamics:** Inter-annual transition matrix (2023 vs 2026).
    7. **NDVI Anomaly:** Multi-year $z$-score stress tracking.
    8. **Irrigation Water Demand:** TNAU/FAO agronomic water budgeting in MCM (40.2 MCM annual demand).
    9. **Field Size Distribution:** Agrarian land fragmentation (smallholder holdings < 1 ha).
  - Click **"Download Insights Atlas PDF"** to display the comprehensive 4-page cartographic report.

## 04:45 – 05:00 | Screen 06: Exports & Wrap-Up
- Showcase export deliverables: GeoTIFF, GeoPackage, area statistics CSV, and print-ready thematic map.
- "All data is open Copernicus Sentinel-1 and Sentinel-2 imagery processed fully automatically on open-source Python geospatial tooling."
