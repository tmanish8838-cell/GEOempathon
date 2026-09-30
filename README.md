# 🌊 HydroShed AI — Explainable Multi-Objective Micro-Watershed Priority & Intervention Decision-Support System

**Event:** GEO IMPATHON 1.0 (*Geospatial Ideas for Greener Earth* — SRM Institute of Science & Technology)  
**Domain:** Domain 2 — Water Resource Management  
**Problem Statement:** **2.3 Watershed-Based Water Resource Priority Mapping**  
**Target Region:** Chennai Metropolitan Basin, Palar Basin, Cauvery-Noyyal Basin, and Vaigai-Tamiraparani Basin (60 Micro-Watersheds across Tamil Nadu)

---

## 🎯 Executive Summary & Core Innovation

Traditional watershed prioritization stops at a static AHP/weighted-overlay map, which suffers from three major limitations:
1. **Conflicting Indicators:** Steep slope increases soil-erosion urgency but decreases percolation-pond suitability.
2. **Static vs. Dynamic Blindness:** A static index cannot distinguish between a naturally semi-arid stable watershed and a rapidly deteriorating peri-urban tank catchment.
3. **Conflating Urgency with Feasibility:** High conservation *Priority* ($P_i$, urgency of need) does not automatically mean high technical *Suitability* ($S_{i,k}$, engineering feasibility for a specific structure).

**HydroShed AI** solves all three by integrating **8 multi-sensor Earth observation layers + 5-year temporal trends + Unsupervised Hydro-Ecological Clustering + Supervised Deterioration Modeling (`Ridge + GBDT Hybrid Ensemble`) + Decoupled Multi-Objective Priority & TOPSIS Suitability Screening**:

$$\text{Satellite Data} \rightarrow \text{Micro-Watershed Indicators} \rightarrow \text{Cluster + ML Deterioration Model} \rightarrow \text{Decoupled Priority } (P_i) \text{ \& Suitability } (S_{i,k}) \rightarrow \text{Explainable Action Dashboard}$$

---

## 🛰️ 1. All 8 Required Satellite Parameters + Tamil Nadu Specialization Layers

| # | Required Parameter | Google Earth Engine Dataset | Derived Micro-Watershed Indicators |
| :--- | :--- | :--- | :--- |
| **1** | **DEM (Elevation)** | `USGS/SRTMGL1_003` (30m) + `WWF/HydroSHEDS/v1/Basins/hybas_12` | Mean elevation (m), relief ratio (m), flow accumulation & micro-watershed delineation |
| **2** | **Slope** | `ee.Terrain.slope` (SRTM 30m) | Mean terrain slope ($^\circ$), runoff velocity susceptibility |
| **3** | **Drainage Density** | `WWF/HydroSHEDS/v1/FreeFlowingRivers` | Stream channel length per $\text{km}^2$ ($\text{km/km}^2$), Strahler Stream Order (1–5) |
| **4** | **Rainfall** | `UCSB-CHG/CHIRPS/DAILY` (1981–2026) | Mean annual rainfall (mm/yr), 5-year rainfall anomaly (%), 10-year drought frequency |
| **5** | **NDVI** | `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 10m) | Dry-season mean NDVI, 5-year vegetation degradation trend ($\Delta\text{NDVI}$) |
| **6** | **Land Use (LULC)** | `GOOGLE/DYNAMICWORLD/V1` (10m near-real-time) | Dominant LULC class, Built-Up %, Cropland %, Bare %, 5-year Built-Up expansion (%) |
| **7** | **Soil** | `OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02` | USDA Soil Texture, Hydrologic Soil Group (HSG A/B/C/D), Infiltration Score, Erosion $K$-factor |
| **8** | **Surface-Water Occurrence** | `JRC/GSW1_4/GlobalSurfaceWater` + `COPERNICUS/S1_GRD` | Historical water occurrence (0–100%), 10-yr persistence decline (%), Sentinel-1 SAR wet/dry ratio |
| **+TN** | **Tamil Nadu Regional Context** | CGWB + Tank/Eri Inventory | Historic Eri/Tank count, Tank encroachment %, Groundwater depth (m bgl), Coastal salinity risk, Cloud-free confidence (%) |

---

## 🧠 2. Four-Stage Mathematical & ML Architecture

### Stage 1: Unsupervised Hydro-Ecological Clustering (`K-Means` + Silhouette Optimization + 2D PCA)
Separates the 60 Tamil Nadu micro-watersheds into 4 natural regimes (*Peri-Urban Tank Encroachment Zone*, *Steep High-Runoff Headwaters*, *Semi-Arid Agricultural Depletion Plain*, *Coastal & Tank-Cascade Buffer Zone*) so conflicting indicators are evaluated in context.

### Stage 2: Leak-Free Supervised Deterioration Model (`Hybrid Ridge + Regularized GBDT Ensemble`)
* **Strict Featurization Ordering:** Splits dataset into Train / Validation / Test **before** fitting `StandardScaler`, `OrdinalEncoder` (HSG A–D), and `OneHotEncoder`.
* **Target ($Y$):** Satellite-observed 5-Year Composite Hydrological Deterioration & Moisture Stress Index.
* **Validation:** Evaluated against a Naive Baseline, Ridge Regression, Random Forest, and GBDT using Holdout $R^2$ (**0.836–0.861**), 5-Fold Cross-Validation, **95% Bootstrapped Confidence Intervals**, and **Slice-Based Error Analysis** across all 4 Tamil Nadu River Basins.

### Stage 3: Decoupled Priority ($P_i$) vs. Intervention Suitability ($S_{i,k}$)
1. **Dynamic Conservation Priority Score ($P_i$):**
   $$P_i = \alpha \cdot \text{Current Multi-Objective Stress}_i + (1 - \alpha) \cdot \text{Predicted Deterioration Rate}_i \quad (\text{default } \alpha = 0.70)$$
   Includes **5 Separate Multi-Objective Priority Scores**:
   - *Water Scarcity Priority*
   - *Runoff & Soil Erosion Priority*
   - *Groundwater Recharge Urgency*
   - *Surface-Water (Eri/Tank) Restoration Priority*
   - *Urban Rainwater Harvesting & Flood Priority*
2. **Intervention Suitability Matrix ($S_{i,k}$):**
   Scores technical feasibility (0–100%) separately for **6 Interventions** using TOPSIS + Hard Physical Engineering Constraints:
   - 🧱 **Check Dam / Nala Bund** (Stream Order $\ge 2$, Slope $1.5^\circ\text{–}12^\circ$, high drainage density)
   - 💧 **Percolation Tank & Recharge Shaft** (Soil Infiltration $\ge 36$, Slope $\le 7.5^\circ$, GW depth $\ge 8\text{ m}$)
   - 🌾 **Farm Pond (Pannaikuttai)** (Cropland $\ge 28\%$, Slope $\le 5^\circ$)
   - 🌲 **Contour Trenching & Hill Afforestation** (Slope $\ge 5.5^\circ$, high erosion $K$-factor)
   - 🏞️ **Eri / Tank Desilting & Bund Restoration** (Historic Eri count $\ge 5$, Water occurrence $\ge 10\%$)
   - 🏙️ **Urban Rooftop RWH & Sponge Bioswales** (Built-up $\ge 24\%$)

### Stage 4: Explainable AI ("Why Prioritized?"), Scenario Simulator & Field Validation Loop
* **Local Feature Attributions:** Every micro-watershed displays a local attribution bar chart showing exact positive/negative points contributed by each indicator relative to the Tamil Nadu baseline.
* **What-If Scenario Simulator:** Simulate monsoon rainfall shifts, urban sprawl controls, afforestation (+NDVI), and Eri restoration (%) to quantify reductions in Critical watersheds and additional water recharged in **Million Cubic Meters (MCM/yr)**.

---

## 🚀 Quickstart & Local Execution

```bash
# 1. Activate virtual environment
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the interactive dashboard
streamlit run app.py
```

---

## 🎬 2–3 Minute YouTube Video Demo Script (For Submission)

* **0:00 – 0:30 (The Problem & Chennai/TN Context):** Show the top banner and KPI row. Explain Problem Statement 2.3: *"Authorities in Chennai and Tamil Nadu cannot build check dams or percolation ponds everywhere at once, and basic weighted-overlay maps fail because they confuse urgency with engineering suitability."*
* **0:30 – 1:15 (Tab 1 — Multi-Objective Geospatial Explorer):** Select **Chennai Basin** (highlighting `MW-001: Kattankulathur-Potheri (SRM Catchment)` and `MW-002: Pallikaranai Marsh`), toggle between the **Final Dynamic Priority Map**, **Eri/Tank Restoration Priority**, and the **8 Satellite Layers** with HydroSHEDS drainage streams.
* **1:15 – 2:00 (Tab 2 — "Why Is This Watershed Prioritized?" & Decoupled Suitability):** Click `MW-001: Kattankulathur-Potheri`. Show the **Local Feature Attribution Chart** explaining *why* it is prioritized (surface-water decline + peri-urban built-up growth) and the **Intervention Suitability Bar Chart** showing *Eri Desilting* and *Urban RWH* passing engineering constraints while *Contour Trenching* is penalized due to flat slope.
* **2:00 – 2:45 (Tab 3 & Tab 4 — ML Validation & What-If Policy Simulator):** Briefly show the **Model Comparison Table with 95% Bootstrapped CIs** ($R^2 > 0.84$) in Tab 3, then jump to **Tab 4**, drag the **Eri Restoration** and **Afforestation NDVI** sliders, and show how many Critical watersheds drop to Moderate and how many **Million Liters/yr** of runoff are conserved!
