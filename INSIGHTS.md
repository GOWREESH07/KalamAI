# களம் 🌾 — Agricultural Insights & Decisions Module (Screen 07)

## Overview
The **Insights Module** in **களம் 🌾** derives policy-actionable intelligence directly from the multi-year (40-month, Jun 2023 – Sep 2026) Sentinel-1 SAR and Sentinel-2 optical time series. Rather than just reporting single-season classifications, it addresses real-world agronomic, irrigation, ground-truth audit, and land dynamics questions across **Ambasamudram** and **Cheranmahadevi** taluks.

Every metric, figure, and chart is computed from genuine pipeline outputs with parameters defined in `config.yaml` under `insights:`.

---

## The 9 Insight Modules

### 1. Cropping Intensity
- **Goal:** Resolve harvest frequency across Agricultural Years (AY 2023-24, AY 2024-25, AY 2025-26).
- **Methodology:** 3-month triangular moving average smoothing on 12-month monthly NDVI series followed by `scipy.signal.find_peaks(height >= 0.50, prominence >= 0.20, distance >= 3 months)`.
- **Classes:** Fallow (0 cycles), Single Crop (1 cycle), Double Crop (2 cycles: Kar & Pishanam), Triple Crop (3 cycles), and Perennial (perennial banana / orchards).
- **Output Products:** `cropping_intensity.tif`, `cropping_intensity.png`, JSON statistics.

### 2. Transplanting Calendar & Flood Timing
- **Goal:** Pinpoint exact month of field inundation and crop transplantation from SAR backscatter dynamics.
- **Methodology:** Detects month of minimum SAR VH backscatter (open water/mud specular reflection $\le -19\text{ dB}$) followed by steep optical green-up. Identifies delayed sowing when transplanting shifts past historical onset windows (Kar: May–Aug, Pishanam: Sep–Dec).
- **Output Products:** `transplanting_calendar.tif`, `transplanting_calendar.png`, seasonal histogram charts.

### 3. Field Verification Priority & Audit Target List
- **Goal:** Maximize ground audit efficiency for field enumerators and revenue inspectors.
- **Formulation:**
  $$\text{Priority} = 0.5 \times (1 - \text{Confidence}) + 0.3 \times (1 - \text{Consistency}) + 0.2 \times \text{Disagreement}$$
- **Features:** Categorizes parcels into **Critical** ($>90$th percentile), **High**, **Medium**, and **Low** priority.
- **Deliverables:** Top-50 inspection table, `field_verification_top50.csv`, and mobile-ready `field_verification.kml` with WGS84 GPS coordinates for navigation in Google Earth / OSM Tracker.

### 4. Village-Level Administrative Statistics
- **Goal:** Provide micro-administrative crop allocation for panchayat-level credit and crop insurance.
- **Methodology:** Spatial catchment aggregation of field parcels across historical revenue settlements (Ambasamudram, Kallidaikurichi, Cheranmahadevi, Pathamadai, Veeravanallur, Mukkudal, Melaseval, etc.).
- **Deliverables:** Village-wise breakdown of Paddy ha, Banana ha, Other ha, crop share %, and dominant classification (`village_statistics.csv`).

### 5. Irrigation Dependency & Canal Buffers
- **Goal:** Analyze crop spatial decay as distance increases from surface water channels.
- **Methodology:** Euclidean Distance Transformation (EDT) at 10m resolution from Thamirabarani river channels, distributary canals (Kannadian, Kodagan), and irrigation tanks (*ery* / *kulams*).
- **Buffer Bins:** $0\text{--}250\text{ m}$, $250\text{--}500\text{ m}$, $500\text{--}1000\text{ m}$, $1000\text{--}2000\text{ m}$, $>2000\text{ m}$.
- **Insight:** Highlights high-water perennial banana concentration in the immediate $0\text{--}500\text{ m}$ canal command corridor.

### 6. Multi-Year Crop Change & Farmland Loss
- **Goal:** Quantify inter-annual crop rotation and agricultural land conversion.
- **Methodology:** Cross-tabulation of independent Random Forest predictions for high-confidence pixels ($\ge 0.60$) between AY 2023-24 and AY 2025-26.
- **Transitions:** Stable Paddy, Stable Banana, Paddy $\rightarrow$ Banana diversification, Banana $\rightarrow$ Paddy rotation, and Cropland $\rightarrow$ Fallow/Built-up loss.

### 7. Crop Health & NDVI Anomaly
- **Goal:** Track vegetative vigour and localized moisture/delayed growth stress.
- **Methodology:** Standardized NDVI $z$-score relative to 3-year historical baseline for corresponding calendar months:
  $$z = \frac{\text{NDVI}_{\text{current}} - \mu_{\text{historical}}}{\sigma_{\text{historical}} + \epsilon}$$
- **Categories:** Severe Stress ($z < -1.5$), Mild Stress ($-1.5 \le z < -0.5$), Normal Vigour ($-0.5 \le z \le 0.5$), Above Normal ($z > 0.5$).

### 8. Indicative Irrigation Water Demand
- **Goal:** Estimate volumetric irrigation requirements in Million Cubic Meters ($\text{MCM}$) to inform PWD canal water releases.
- **Agronomic Benchmarks (TNAU / FAO-56):**
  - Lowland flooded wetland paddy: $1{,}200\text{ mm}$ ($12{,}000\text{ m}^3/\text{ha}$ per crop cycle)
  - Commercial perennial banana: $2{,}000\text{ mm}$ ($20{,}000\text{ m}^3/\text{ha}$ per year)
  - Other seasonal crops/pulses: $500\text{ mm}$ ($5{,}000\text{ m}^3/\text{ha}$)
- **Deliverables:** Seasonal water requirement breakdown for Kar season, Pishanam season, and annual perennial banana command areas (`water_demand_summary.csv`).

### 9. Field Size Distribution & Agrarian Fragmentation
- **Goal:** Characterize parcel morphology and smallholder land structure.
- **Census Categories:** Marginal ($<0.5\text{ ha}$), Small ($0.5\text{--}1.0\text{ ha}$), Semi-Medium ($1.0\text{--}2.0\text{ ha}$), Medium ($2.0\text{--}5.0\text{ ha}$), Large ($>5.0\text{ ha}$).
- **Key Metrics:** Smallholder plot share, mean and median parcel hectares, and dominant crop per holding size bracket (`field_size_distribution.csv`).

---

## Multi-Page Agricultural Insights Atlas PDF
- **Endpoint:** `GET /api/insights/atlas/pdf`
- **Output:** `backend/outputs/insights/atlas/kalam_insights_atlas.pdf`
- **Structure:**
  - **Page 1:** Executive cover, key metric callout tiles, and synthesis table of all 9 insight findings.
  - **Page 2:** Phenology & Verification Prioritization (Cropping Intensity, Transplanting Calendar, Audit Map).
  - **Page 3:** Water Infrastructure & Multi-Year Dynamics (Village stats, Canal buffers, Crop Change).
  - **Page 4:** Agronomic Resources & Morphology (NDVI Anomaly, Water Demand Budget, Field Size Distribution).

---

## API Endpoints Reference
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/insights` | `GET` | Summary list of all 9 insights with titles, status, and key findings |
| `/api/insights/{insight_id}` | `GET` | Complete computed analytics, stats, chart data, methodology, and limitations |
| `/api/insights/compute_all` | `POST` | Recomputes all 9 insight modules and generates the Atlas PDF |
| `/api/insights/{insight_id}/png` | `GET` | Serves the cartographic PNG map overlay |
| `/api/insights/{insight_id}/csv` | `GET` | Downloads tabular statistical CSV data |
| `/api/insights/verification_priority/kml` | `GET` | Downloads Google Earth KML / KMZ with GPS coordinates |
| `/api/insights/atlas/pdf` | `GET` | Downloads the publication-ready 4-page Insights Atlas PDF |
