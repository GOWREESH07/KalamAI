# களம் 🌾 — Paddy & Banana Parcel Mapping System
**VIT Mapathon 2026 · SustainAI 2026 · Problem Statement 1**  
*Differentiate land parcels and identify/differentiate paddy and banana cultivation in Tirunelveli District (Ambasamudram & Cheranmahadevi Taluks).*

---

## 1. Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### Launching the System
```bash
# 1. Kill any existing processes on ports 8000 & 5173
kill -9 $(lsof -t -i:8000 -i:5173) 2>/dev/null || true

# 2. Launch backend & frontend concurrently
cd "Your Directory"
./run.sh
```
- **Landing Page & Web App:** `http://localhost:5173`
- **Interactive Backend API:** `http://localhost:8000/docs`

---

## 2. Key Architecture & Features

### A. Landing Page & Sensor Fusion (`/` and `/app`)
- **Landing Page (`/`):** Live study area stats, early sensor fusion architecture, interactive map preview, and 8-stage methodology strip.
- **Sensor Fusion:** Stacked 10m feature vector combining Sentinel-2 optical indices (NDVI, NDWI, LSWI, EVI) with Sentinel-1 SAR C-band radar backscatter (VV, VH, VH/VV) and seasonal phenological statistics.

### B. Multi-Year Time-Series Modeling (40 Months: Jun 2023 – Sep 2026)
- **Standardized Agricultural Years:** AY 2023-24, AY 2024-25, and AY 2025-26 (each 12 monthly composites, Jun–May).
- **Pooled Random Forest Classifier:** 300-tree ensemble trained on multi-year pooled samples, predicting each year separately to compute:
  - `final_class` (majority vote with probability tie-breaking)
  - `consistency` (% of years agreeing with consensus)
  - `final_confidence` ($\text{mean probability} \times \text{consistency}$)
  - `rotation_detected` (flagging paddy-banana crop alternation across years)

### C. Screen 07: Agricultural Insights & Decisions Module
Nine modular, cached analytical engines with dedicated interactive panels and cartographic exports:
1. **Cropping Intensity:** Single, Double (Kar + Pishanam), Triple, and Perennial cycle classification via `scipy.signal.find_peaks`.
2. **Transplanting Calendar:** Pinpointing SAR inundation dips ($\le -19\text{ dB}$) and detecting delayed planting shifts.
3. **Field Verification Priority:** Multi-criteria audit index ($0.5(1-\text{conf}) + 0.3(1-\text{cons}) + 0.2\text{disag}$) exporting top-50 targets to CSV and mobile GPS KML.
4. **Village-Level Statistics:** Revenue settlement breakdown of crop acreage and dominant farming systems.
5. **Irrigation Dependency:** Proximity decay analysis across concentric Euclidean distance buffers ($<250\text{m}$ to $>2\text{km}$) from the Thamirabarani canal network.
6. **Crop Change Dynamics:** Inter-annual transition matrix between AY 2023-24 and AY 2025-26.
7. **Crop Health Anomaly:** Standardized NDVI $z$-score deviation relative to 3-year historical baseline.
8. **Irrigation Water Demand:** TNAU / FAO agronomic water budgeting in Million Cubic Meters ($\text{MCM}$) for canal release management.
9. **Field Size Distribution:** Agricultural parcel morphology and smallholder land fragmentation analysis.

### D. Multi-Page Insights Atlas PDF
- **Endpoint:** `GET /api/insights/atlas/pdf`
- **Output:** 4-page high-resolution PDF with executive cover, phenology dynamics, canal infrastructure, and agronomic water budgeting.

---

## 3. The Remote Sensing Science: Paddy vs. Banana

| Parameter | Paddy (*Oryza sativa*) | Banana (*Musa acuminata*) |
| :--- | :--- | :--- |
| **Radar Backscatter (SAR VH)** | Pronounced dip to $< -19\text{ dB}$ during flooding / land prep | Stable volume scattering consistently $> -15\text{ dB}$ |
| **Optical Greenness (NDVI)** | Rapid surge ($> 0.60$) followed by harvest plunge | Persistent high greenness (NDVI $> 0.50$ for $\ge 7$ months) |
| **Temporal Variability** | High amplitude ($> 0.35$), 2 peaks (Kar & Pishanam) | Invariant flat trajectory (std dev $< 0.12$) |
| **Water Demand (TNAU/FAO)** | $1{,}200\text{ mm}$ / season ($12{,}000\text{ m}^3/\text{ha}$) | $2{,}000\text{ mm}$ / year ($20{,}000\text{ m}^3/\text{ha}$) |
| **Canal Proximity** | Distributed across middle & tail-end reaches | Concentrated within $<250\text{--}500\text{m}$ canal frontage |

---

## 4. Output Deliverables
- **Classified GeoTIFF:** `backend/outputs/classified.tif` & `final_class.tif` (10m UTM EPSG:32643)
- **Vector Parcels:** `backend/outputs/parcels.gpkg` & `parcels.geojson`
- **Insights Atlas PDF:** `backend/outputs/insights/atlas/kalam_insights_atlas.pdf`
- **Field Audit Targets:** `backend/outputs/insights/verification_priority/field_verification_top50.csv` & `.kml`
- **Cartographic Thematic Map:** `backend/outputs/thematic_map.png` & `.pdf`
