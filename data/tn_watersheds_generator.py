"""
Hydrologically grounded Micro-Watershed Dataset & GeoJSON Polygon Generator
for Chennai Metropolitan & Tamil Nadu River Basins (Problem Statement 2.3).

Covers all 8 mandatory satellite parameters + 5-year temporal trends +
Tamil Nadu regional specialization layers (Eri/Tank density, CGWB groundwater stage,
Sentinel-1 SAR validation, and cloud-free observation confidence).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# 60 Real Sub-Basin / Micro-Watershed Locations across Chennai & Tamil Nadu
# Format: (id, name, basin, district, lat, lon, base_elev_m, base_slope_deg, archetype_hint)
TN_WATERSHED_CATALOG: List[Tuple[str, str, str, str, float, float, float, float, str]] = [
    # --- 1. CHENNAI METROPOLITAN BASIN (Adyar, Cooum, Kosasthalaiyar, Kovalam) ---
    ("MW-001", "Kattankulathur-Potheri (SRM Catchment)", "Chennai Basin", "Chengalpattu", 12.823, 80.044, 48.0, 2.8, "peri_urban_tank"),
    ("MW-002", "Pallikaranai Marsh South", "Chennai Basin", "Chennai", 12.935, 80.215, 7.5, 0.9, "urban_wetland"),
    ("MW-003", "Velachery-Adambakkam Urban Sub-basin", "Chennai Basin", "Chennai", 12.979, 80.218, 9.0, 0.8, "dense_urban"),
    ("MW-004", "Chembarambakkam Upper Catchment", "Chennai Basin", "Kancheepuram", 13.012, 80.025, 38.0, 2.4, "peri_urban_tank"),
    ("MW-005", "Chembarambakkam Surplus-Adyar Mid", "Chennai Basin", "Chennai", 13.005, 80.115, 22.0, 1.6, "peri_urban_tank"),
    ("MW-006", "Tambaram-Mudichur Floodplain", "Chennai Basin", "Chengalpattu", 12.922, 80.095, 29.0, 2.1, "peri_urban_tank"),
    ("MW-007", "Guduvancheri-Nandivaram Eri Zone", "Chennai Basin", "Chengalpattu", 12.845, 80.062, 42.0, 2.6, "peri_urban_tank"),
    ("MW-008", "Porur-Manapakkam Sub-basin", "Chennai Basin", "Chennai", 13.032, 80.158, 16.0, 1.3, "dense_urban"),
    ("MW-009", "Korattur-Ambattur Industrial Tank Zone", "Chennai Basin", "Chennai", 13.108, 80.162, 18.0, 1.2, "dense_urban"),
    ("MW-010", "Red Hills (Puzhal) Reservoir Catchment", "Chennai Basin", "Tiruvallur", 13.168, 80.175, 26.0, 1.8, "peri_urban_tank"),
    ("MW-011", "Sholavaram-Kosasthalaiyar Mid", "Chennai Basin", "Tiruvallur", 13.232, 80.155, 34.0, 2.2, "agri_plain"),
    ("MW-012", "Poondi Reservoir Upper Catchment", "Chennai Basin", "Tiruvallur", 13.210, 79.865, 68.0, 3.9, "agri_plain"),
    ("MW-013", "Manali-Ennore Creek Estuarine", "Chennai Basin", "Chennai", 13.205, 80.285, 5.0, 0.6, "coastal_estuarine"),
    ("MW-014", "Kelambakkam-Kovalam Backwater Basin", "Chennai Basin", "Chengalpattu", 12.788, 80.222, 8.0, 1.1, "coastal_estuarine"),
    ("MW-015", "Sriperumbudur Industrial-Tank Sub-basin", "Chennai Basin", "Kancheepuram", 12.968, 79.948, 56.0, 2.9, "peri_urban_tank"),
    ("MW-016", "Cooum Mid-Paruthipattu Catchment", "Chennai Basin", "Tiruvallur", 13.075, 80.105, 24.0, 1.5, "dense_urban"),
    ("MW-017", "Adyar Estuary-Guindy Urban Canopy", "Chennai Basin", "Chennai", 13.010, 80.235, 11.0, 1.1, "dense_urban"),
    ("MW-018", "Thiruporur-OMR IT Corridor Catchment", "Chennai Basin", "Chengalpattu", 12.725, 80.185, 24.0, 1.9, "peri_urban_tank"),

    # --- 2. PALAR RIVER BASIN (Chengalpattu, Kancheepuram, Vellore, Ranipet, Tirupattur) ---
    ("MW-019", "Madurantakam Eri Catchment", "Palar Basin", "Chengalpattu", 12.512, 79.885, 46.0, 2.4, "agri_plain"),
    ("MW-020", "Chengalpattu-Kolavai Lake Sub-basin", "Palar Basin", "Chengalpattu", 12.695, 79.982, 52.0, 3.1, "peri_urban_tank"),
    ("MW-021", "Uthiramerur Historic Cascade Tank Zone", "Palar Basin", "Kancheepuram", 12.615, 79.755, 64.0, 2.5, "agri_plain"),
    ("MW-022", "Walajabad-Palar Confluence", "Palar Basin", "Kancheepuram", 12.785, 79.822, 69.0, 2.3, "agri_plain"),
    ("MW-023", "Kancheepuram-Vegavathi Sub-basin", "Palar Basin", "Kancheepuram", 12.838, 79.705, 84.0, 2.7, "peri_urban_tank"),
    ("MW-024", "Kaveripakkam Big Tank Catchment", "Palar Basin", "Ranipet", 12.905, 79.465, 132.0, 3.4, "agri_plain"),
    ("MW-025", "Arcot-Ranipet Palar Mid-Valley", "Palar Basin", "Ranipet", 12.918, 79.332, 165.0, 3.8, "peri_urban_tank"),
    ("MW-026", "Vellore-Ponnaiyar Tributary Sub-basin", "Palar Basin", "Vellore", 12.935, 79.138, 225.0, 6.8, "hill_catchment"),
    ("MW-027", "Ambur-Vaniyambadi Upper Palar", "Palar Basin", "Tirupattur", 12.782, 78.715, 330.0, 9.4, "hill_catchment"),
    ("MW-028", "Yelagiri-Jolarpettai Steep Headwaters", "Palar Basin", "Tirupattur", 12.585, 78.635, 640.0, 18.5, "hill_catchment"),
    ("MW-029", "Javadi Hills Northern Escarpment", "Palar Basin", "Vellore", 12.665, 78.925, 580.0, 16.8, "hill_catchment"),
    ("MW-030", "Cheyyar-Vandavasi Dry Tank Plain", "Palar Basin", "Tiruvannamalai", 12.525, 79.605, 112.0, 2.6, "agri_plain"),
    ("MW-031", "Gudiyatham-Koundinya Sub-basin", "Palar Basin", "Vellore", 12.948, 78.875, 275.0, 8.2, "hill_catchment"),
    ("MW-032", "Kalpakkam-Palar Mouth Estuarine", "Palar Basin", "Chengalpattu", 12.505, 80.145, 6.5, 0.8, "coastal_estuarine"),

    # --- 3. CAUVERY & NOYYAL-BHAVANI BASIN (Coimbatore, Erode, Trichy, Thanjavur, Nagapattinam) ---
    ("MW-033", "Siruvani-Boluvampatti Western Ghats", "Cauvery Basin", "Coimbatore", 10.955, 76.725, 685.0, 19.4, "hill_catchment"),
    ("MW-034", "Noyyal Urban Coimbatore (Singanallur)", "Cauvery Basin", "Coimbatore", 10.995, 77.015, 412.0, 3.2, "dense_urban"),
    ("MW-035", "Sulur-Palladam Semi-Arid Noyyal", "Cauvery Basin", "Tiruppur", 11.015, 77.245, 345.0, 2.9, "agri_plain"),
    ("MW-036", "Tiruppur-Orathupalayam Noyyal Mid", "Cauvery Basin", "Tiruppur", 11.105, 77.415, 290.0, 2.7, "peri_urban_tank"),
    ("MW-037", "Bhavani Sagar Foothill Catchment", "Cauvery Basin", "Erode", 11.475, 77.115, 360.0, 11.2, "hill_catchment"),
    ("MW-038", "Gobichettipalayam-Kodiveri Command", "Cauvery Basin", "Erode", 11.452, 77.435, 215.0, 2.8, "agri_plain"),
    ("MW-039", "Perundurai-Chennimalai Dry Upland", "Cauvery Basin", "Erode", 11.225, 77.585, 265.0, 3.5, "agri_plain"),
    ("MW-040", "Karur-Amaravathi Confluence", "Cauvery Basin", "Karur", 10.958, 78.085, 138.0, 2.2, "agri_plain"),
    ("MW-041", "Srirangam-Kollidam Head Regulator", "Cauvery Basin", "Tiruchirappalli", 10.855, 78.695, 76.0, 1.4, "agri_plain"),
    ("MW-042", "Kallanai (Grand Anicut) Apex Delta", "Cauvery Basin", "Thanjavur", 10.832, 78.822, 64.0, 1.2, "agri_plain"),
    ("MW-043", "Orathanadu-Vennar Mid Delta", "Cauvery Basin", "Thanjavur", 10.625, 79.245, 34.0, 0.9, "agri_plain"),
    ("MW-044", "Mannargudi-Pamaniyar Intensive Paddy", "Cauvery Basin", "Thiruvarur", 10.665, 79.452, 22.0, 0.8, "agri_plain"),
    ("MW-045", "Mayiladuthurai-Kollidam Lower", "Cauvery Basin", "Mayiladuthurai", 11.105, 79.655, 14.0, 0.7, "coastal_estuarine"),
    ("MW-046", "Vedaranyam-Point Calimere Tail-End", "Cauvery Basin", "Nagapattinam", 10.385, 79.825, 4.5, 0.5, "coastal_estuarine"),

    # --- 4. VAIGAI & TAMIRAPARANI BASIN (Madurai, Theni, Dindigul, Ramnad, Tirunelveli, Thoothukudi) ---
    ("MW-047", "Megamalai-Varushanad Steep Headwaters", "Vaigai-Tamiraparani Basin", "Theni", 9.725, 77.385, 790.0, 21.6, "hill_catchment"),
    ("MW-048", "Cumbum Valley-Suruli Sub-basin", "Vaigai-Tamiraparani Basin", "Theni", 9.815, 77.312, 430.0, 7.8, "agri_plain"),
    ("MW-049", "Vaigai Reservoir-Andipatti Catchment", "Vaigai-Tamiraparani Basin", "Theni", 10.045, 77.585, 310.0, 6.4, "hill_catchment"),
    ("MW-050", "Palani-Kodaikanal Southern Escarpment", "Vaigai-Tamiraparani Basin", "Dindigul", 10.315, 77.525, 840.0, 22.8, "hill_catchment"),
    ("MW-051", "Madurai Urban Vaigai-Vandiyur Tank", "Vaigai-Tamiraparani Basin", "Madurai", 9.925, 78.145, 134.0, 1.7, "dense_urban"),
    ("MW-052", "Thirumangalam-Gundar Dry Sub-basin", "Vaigai-Tamiraparani Basin", "Madurai", 9.815, 77.985, 142.0, 2.3, "agri_plain"),
    ("MW-053", "Sivaganga-Manamadurai Kanmai Cascade", "Vaigai-Tamiraparani Basin", "Sivaganga", 9.765, 78.485, 82.0, 1.6, "agri_plain"),
    ("MW-054", "Paramakudi Arid Vaigai Lower", "Vaigai-Tamiraparani Basin", "Ramanathapuram", 9.545, 78.585, 44.0, 1.2, "agri_plain"),
    ("MW-055", "Ramanathapuram Periya Kanmai Coastal", "Vaigai-Tamiraparani Basin", "Ramanathapuram", 9.365, 78.835, 9.5, 0.7, "coastal_estuarine"),
    ("MW-056", "Agasthiyar-Papanasam Tamiraparani Upper", "Vaigai-Tamiraparani Basin", "Tirunelveli", 8.685, 77.345, 620.0, 18.2, "hill_catchment"),
    ("MW-057", "Ambasamudram-Gadana Foothill Valley", "Vaigai-Tamiraparani Basin", "Tirunelveli", 8.715, 77.455, 115.0, 4.6, "agri_plain"),
    ("MW-058", "Tirunelveli-Palayamkottai Channel Zone", "Vaigai-Tamiraparani Basin", "Tirunelveli", 8.728, 77.715, 48.0, 1.5, "peri_urban_tank"),
    ("MW-059", "Kovilpatti Rain-Fed Black Cotton Plain", "Vaigai-Tamiraparani Basin", "Thoothukudi", 9.175, 77.865, 96.0, 2.1, "agri_plain"),
    ("MW-060", "Srivaikuntam-Punnaiyakayal Estuarine", "Vaigai-Tamiraparani Basin", "Thoothukudi", 8.632, 78.025, 7.0, 0.6, "coastal_estuarine"),
]


def _generate_watershed_polygon(
    lat: float, lon: float, seed_idx: int, area_km2: float
) -> List[List[float]]:
    """
    Generates a natural irregular 10-vertex micro-watershed catchment polygon
    in [lon, lat] GeoJSON coordinate order around (lat, lon).
    """
    rng = np.random.default_rng(seed=1000 + seed_idx * 17)
    base_radius_deg = math.sqrt(area_km2 / math.pi) / 110.5
    n_vertices = 10
    angles = np.linspace(0, 2 * math.pi, n_vertices, endpoint=False)
    coords: List[List[float]] = []
    elongation = rng.uniform(0.82, 1.22)
    orientation = rng.uniform(0, math.pi)

    for angle in angles:
        jitter_r = base_radius_deg * rng.uniform(0.78, 1.24)
        rot_angle = angle - orientation
        dx = jitter_r * elongation * math.cos(rot_angle)
        dy = (jitter_r / elongation) * math.sin(rot_angle)
        # Rotate back
        dlon = dx * math.cos(orientation) - dy * math.sin(orientation)
        dlat = dx * math.sin(orientation) + dy * math.cos(orientation)
        coords.append([round(lon + dlon, 5), round(lat + dlat, 5)])

    coords.append(coords[0])  # Close ring
    return coords


def _generate_stream_linestring(
    lat: float, lon: float, seed_idx: int, area_km2: float
) -> List[List[float]]:
    """Generates a realistic main drainage stream channel inside the micro-watershed."""
    rng = np.random.default_rng(seed=5000 + seed_idx * 31)
    reach_deg = (math.sqrt(area_km2 / math.pi) / 110.5) * 0.75
    angle = rng.uniform(0, math.pi)
    points: List[List[float]] = []
    for t in np.linspace(-1.0, 1.0, 5):
        meander = rng.uniform(-0.18, 0.18) * reach_deg
        dlon = t * reach_deg * math.cos(angle) - meander * math.sin(angle)
        dlat = t * reach_deg * math.sin(angle) + meander * math.cos(angle)
        points.append([round(lon + dlon, 5), round(lat + dlat, 5)])
    return points


def build_tamil_nadu_watershed_dataset() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Constructs the 60-micro-watershed dataset covering all 8 mandatory satellite parameters,
    5-year temporal trends, Tamil Nadu local hydrological layers, and GeoJSON geometries.
    """
    records: List[Dict[str, Any]] = []
    polygon_features: List[Dict[str, Any]] = []
    stream_features: List[Dict[str, Any]] = []

    for idx, (
        mw_id,
        name,
        basin,
        district,
        lat,
        lon,
        elev_m,
        slope_deg,
        archetype_hint,
    ) in enumerate(TN_WATERSHED_CATALOG):
        rng = np.random.default_rng(seed=2026 + idx * 29)
        area_km2 = round(float(rng.uniform(28.0, 68.0)), 1)

        # 1. DEM & Slope (USGS/SRTMGL1_003)
        dem_mean_m = round(elev_m + float(rng.uniform(-4.0, 4.0)), 1)
        dem_relief_m = round(max(8.0, slope_deg * 14.5 + float(rng.uniform(5.0, 25.0))), 1)
        slope_mean_deg = round(max(0.4, slope_deg + float(rng.uniform(-0.3, 0.4))), 2)

        # 2. Drainage Density & Stream Order (HydroSHEDS + SRTM Flow Accumulation)
        if archetype_hint == "hill_catchment":
            drainage_density = round(float(rng.uniform(2.35, 3.65)), 2)
            stream_order = int(rng.choice([3, 4, 5], p=[0.35, 0.45, 0.20]))
        elif archetype_hint in ("peri_urban_tank", "agri_plain"):
            drainage_density = round(float(rng.uniform(1.45, 2.55)), 2)
            stream_order = int(rng.choice([2, 3, 4], p=[0.30, 0.50, 0.20]))
        elif archetype_hint == "dense_urban":
            drainage_density = round(float(rng.uniform(1.85, 2.95)), 2)
            stream_order = int(rng.choice([2, 3, 4], p=[0.40, 0.45, 0.15]))
        else:  # coastal_estuarine / urban_wetland
            drainage_density = round(float(rng.uniform(1.30, 2.20)), 2)
            stream_order = int(rng.choice([3, 4, 5], p=[0.30, 0.50, 0.20]))

        # 3. CHIRPS Rainfall (mm/yr, 5-Yr Anomaly %, Drought Frequency)
        if basin == "Chennai Basin":
            rainfall_annual_mm = round(float(rng.uniform(1180, 1420)), 0)
            rainfall_anomaly_pct = round(float(rng.uniform(-16.5, 8.5)), 1)
        elif basin == "Palar Basin":
            rainfall_annual_mm = round(float(rng.uniform(860, 1140)), 0)
            rainfall_anomaly_pct = round(float(rng.uniform(-24.0, -3.5)), 1)
        elif basin == "Cauvery Basin":
            rainfall_annual_mm = round(float(rng.uniform(720, 1180)), 0)
            rainfall_anomaly_pct = round(float(rng.uniform(-22.5, 4.0)), 1)
        else:  # Vaigai-Tamiraparani
            rainfall_annual_mm = round(float(rng.uniform(680, 1120)), 0)
            rainfall_anomaly_pct = round(float(rng.uniform(-26.0, 3.0)), 1)

        if archetype_hint == "hill_catchment":
            rainfall_annual_mm = round(rainfall_annual_mm * 1.28, 0)

        drought_freq_10yr = int(
            np.clip(round(3.2 - rainfall_anomaly_pct * 0.11 + rng.uniform(-0.6, 0.8)), 1, 7)
        )

        # 4. Sentinel-2 NDVI (Dry-season Mean & 5-Yr Trend)
        if archetype_hint == "hill_catchment":
            ndvi_mean = round(float(rng.uniform(0.34, 0.64)), 3)
            ndvi_5yr_trend = round(float(rng.uniform(-0.085, 0.015)), 3)
        elif archetype_hint == "dense_urban":
            ndvi_mean = round(float(rng.uniform(0.14, 0.26)), 3)
            ndvi_5yr_trend = round(float(rng.uniform(-0.095, -0.020)), 3)
        elif archetype_hint == "peri_urban_tank":
            ndvi_mean = round(float(rng.uniform(0.21, 0.36)), 3)
            ndvi_5yr_trend = round(float(rng.uniform(-0.110, -0.025)), 3)
        elif archetype_hint == "agri_plain":
            ndvi_mean = round(float(rng.uniform(0.25, 0.48)), 3)
            ndvi_5yr_trend = round(float(rng.uniform(-0.080, 0.020)), 3)
        else:  # coastal_estuarine / urban_wetland
            ndvi_mean = round(float(rng.uniform(0.22, 0.42)), 3)
            ndvi_5yr_trend = round(float(rng.uniform(-0.075, 0.010)), 3)

        # 5. Dynamic World 10m Land Use / Land Cover (LULC) & 5-Yr Built-up Change
        if archetype_hint == "dense_urban":
            builtup_pct = round(float(rng.uniform(54.0, 82.0)), 1)
            cropland_pct = round(float(rng.uniform(2.0, 12.0)), 1)
            builtup_5yr_growth_pct = round(float(rng.uniform(12.0, 26.5)), 1)
            dominant_lulc = "Built-Up Urban"
        elif archetype_hint == "peri_urban_tank":
            builtup_pct = round(float(rng.uniform(28.0, 52.0)), 1)
            cropland_pct = round(float(rng.uniform(18.0, 42.0)), 1)
            builtup_5yr_growth_pct = round(float(rng.uniform(15.5, 34.0)), 1)
            dominant_lulc = "Peri-Urban Mixed / Tank"
        elif archetype_hint == "agri_plain":
            builtup_pct = round(float(rng.uniform(7.0, 22.0)), 1)
            cropland_pct = round(float(rng.uniform(48.0, 76.0)), 1)
            builtup_5yr_growth_pct = round(float(rng.uniform(3.5, 11.5)), 1)
            dominant_lulc = "Cropland (Paddy/Dryland)"
        elif archetype_hint == "hill_catchment":
            builtup_pct = round(float(rng.uniform(3.0, 12.0)), 1)
            cropland_pct = round(float(rng.uniform(12.0, 32.0)), 1)
            builtup_5yr_growth_pct = round(float(rng.uniform(2.0, 8.5)), 1)
            dominant_lulc = "Forest / Scrubland"
        else:  # coastal / wetland
            builtup_pct = round(float(rng.uniform(18.0, 44.0)), 1)
            cropland_pct = round(float(rng.uniform(15.0, 38.0)), 1)
            builtup_5yr_growth_pct = round(float(rng.uniform(8.0, 22.0)), 1)
            dominant_lulc = "Wetland / Coastal Buffer"

        bare_degraded_pct = round(
            float(np.clip(100.0 - builtup_pct - cropland_pct - ndvi_mean * 55.0, 4.0, 36.0)), 1
        )

        # 6. OpenLandMap Soil Texture & Hydrologic Soil Group (HSG A/B/C/D)
        if archetype_hint == "coastal_estuarine":
            soil_texture = "Sandy Clay Loam"
            hydrologic_soil_group = str(rng.choice(["B", "C"], p=[0.55, 0.45]))
        elif archetype_hint == "hill_catchment":
            soil_texture = "Gravelly Red Loam"
            hydrologic_soil_group = str(rng.choice(["B", "C"], p=[0.45, 0.55]))
        elif archetype_hint == "dense_urban":
            soil_texture = "Compacted Urban Clay"
            hydrologic_soil_group = "D"
        elif "Black" in name or district in ("Thoothukudi", "Madurai", "Ramanathapuram"):
            soil_texture = "Vertisol / Clay Loam"
            hydrologic_soil_group = str(rng.choice(["C", "D"], p=[0.60, 0.40]))
        else:
            soil_texture = str(rng.choice(["Sandy Loam", "Red Sandy Clay", "Alluvial Loam"]))
            hydrologic_soil_group = str(rng.choice(["A", "B", "C"], p=[0.25, 0.45, 0.30]))

        infiltration_Map = {"A": 85.0, "B": 68.0, "C": 42.0, "D": 22.0}
        soil_infiltration_score = round(
            infiltration_Map[hydrologic_soil_group] + float(rng.uniform(-6.0, 6.0)), 1
        )
        soil_erosion_k_factor = round(
            0.18 + (100.0 - soil_infiltration_score) * 0.0032 + slope_mean_deg * 0.008, 3
        )

        # 7. JRC Global Surface Water Occurrence (0-100%) & 10-Yr Persistence Decline (%)
        if archetype_hint in ("peri_urban_tank", "urban_wetland"):
            water_occurrence_pct = round(float(rng.uniform(22.0, 54.0)), 1)
            water_10yr_decline_pct = round(float(rng.uniform(18.0, 38.5)), 1)
            historic_eri_count = int(rng.integers(8, 26))
            tank_encroachment_pct = round(float(rng.uniform(19.0, 44.0)), 1)
        elif archetype_hint == "dense_urban":
            water_occurrence_pct = round(float(rng.uniform(8.0, 24.0)), 1)
            water_10yr_decline_pct = round(float(rng.uniform(24.0, 46.0)), 1)
            historic_eri_count = int(rng.integers(4, 15))
            tank_encroachment_pct = round(float(rng.uniform(32.0, 58.0)), 1)
        elif archetype_hint == "agri_plain":
            water_occurrence_pct = round(float(rng.uniform(12.0, 36.0)), 1)
            water_10yr_decline_pct = round(float(rng.uniform(11.0, 29.0)), 1)
            historic_eri_count = int(rng.integers(10, 32))
            tank_encroachment_pct = round(float(rng.uniform(8.0, 24.0)), 1)
        elif archetype_hint == "hill_catchment":
            water_occurrence_pct = round(float(rng.uniform(6.0, 19.0)), 1)
            water_10yr_decline_pct = round(float(rng.uniform(6.0, 18.0)), 1)
            historic_eri_count = int(rng.integers(1, 6))
            tank_encroachment_pct = round(float(rng.uniform(2.0, 9.0)), 1)
        else:  # coastal_estuarine
            water_occurrence_pct = round(float(rng.uniform(28.0, 62.0)), 1)
            water_10yr_decline_pct = round(float(rng.uniform(14.0, 31.0)), 1)
            historic_eri_count = int(rng.integers(6, 18))
            tank_encroachment_pct = round(float(rng.uniform(14.0, 34.0)), 1)

        # 8. Tamil Nadu Regional Validation & Groundwater Context (CGWB + Sentinel-1 SAR)
        gw_depth_mbgl = round(
            float(
                np.clip(
                    6.5
                    + (100.0 - soil_infiltration_score) * 0.09
                    - rainfall_anomaly_pct * 0.25
                    + cropland_pct * 0.08
                    + builtup_pct * 0.07
                    + rng.uniform(-2.5, 3.5),
                    3.2,
                    28.5,
                )
            ),
            1,
        )

        if gw_depth_mbgl > 18.0:
            cgwb_stage = "Over-Exploited (>100%)"
        elif gw_depth_mbgl > 13.5:
            cgwb_stage = "Critical (90-100%)"
        elif gw_depth_mbgl > 9.5:
            cgwb_stage = "Semi-Critical (70-90%)"
        else:
            cgwb_stage = "Safe (<70%)"

        coastal_salinity_risk = (
            round(float(rng.uniform(58.0, 88.0)), 1)
            if archetype_hint == "coastal_estuarine"
            else round(float(rng.uniform(5.0, 24.0)), 1)
        )

        # Sentinel-1 SAR Wet-to-Dry Inundation Ratio & Cloud-Free Observation Confidence
        s1_sar_wet_dry_ratio = round(
            float(np.clip(1.35 + (water_10yr_decline_pct / 35.0) + rng.uniform(-0.15, 0.25), 1.1, 3.4)),
            2,
        )
        data_confidence_pct = round(
            float(
                np.clip(
                    93.0 - (slope_mean_deg * 0.35) + rng.uniform(-4.5, 3.5),
                    76.0,
                    98.5,
                )
            ),
            1,
        )

        # Self-Supervised Satellite Ground-Truth Proxy Target:
        # 5-Year Composite Hydrological Deterioration & Moisture Stress Index (0 - 100)
        # Driven by physical relationships + realistic sensor noise
        true_deterioration_score = (
            0.26 * (water_10yr_decline_pct / 48.0 * 100.0)
            + 0.19 * (builtup_5yr_growth_pct / 35.0 * 100.0)
            + 0.16 * (max(0.0, -rainfall_anomaly_pct) / 26.0 * 100.0)
            + 0.15 * (max(0.0, -ndvi_5yr_trend) / 0.11 * 100.0)
            + 0.12 * ((100.0 - soil_infiltration_score))
            + 0.12 * (min(slope_mean_deg, 22.0) / 22.0 * 60.0 + drainage_density / 3.7 * 40.0)
            + float(rng.normal(0.0, 2.4))
        )
        observed_deterioration_index = round(float(np.clip(true_deterioration_score, 12.0, 96.0)), 2)

        poly_coords = _generate_watershed_polygon(lat, lon, idx, area_km2)
        stream_coords = _generate_stream_linestring(lat, lon, idx, area_km2)

        rec = {
            "watershed_id": mw_id,
            "name": name,
            "basin": basin,
            "district": district,
            "lat": lat,
            "lon": lon,
            "area_km2": area_km2,
            "archetype_hint": archetype_hint,
            # 8 Required Satellite Parameters
            "dem_elevation_m": dem_mean_m,
            "dem_relief_m": dem_relief_m,
            "slope_deg": slope_mean_deg,
            "drainage_density_km_km2": drainage_density,
            "stream_order": stream_order,
            "rainfall_annual_mm": rainfall_annual_mm,
            "rainfall_anomaly_pct": rainfall_anomaly_pct,
            "drought_freq_10yr": drought_freq_10yr,
            "ndvi_mean": ndvi_mean,
            "ndvi_5yr_trend": ndvi_5yr_trend,
            "dominant_lulc": dominant_lulc,
            "builtup_pct": builtup_pct,
            "cropland_pct": cropland_pct,
            "bare_degraded_pct": bare_degraded_pct,
            "builtup_5yr_growth_pct": builtup_5yr_growth_pct,
            "soil_texture": soil_texture,
            "hydrologic_soil_group": hydrologic_soil_group,
            "soil_infiltration_score": soil_infiltration_score,
            "soil_erosion_k_factor": soil_erosion_k_factor,
            "water_occurrence_pct": water_occurrence_pct,
            "water_10yr_decline_pct": water_10yr_decline_pct,
            # Tamil Nadu Regional & Validation Layers
            "historic_eri_count": historic_eri_count,
            "tank_encroachment_pct": tank_encroachment_pct,
            "gw_depth_mbgl": gw_depth_mbgl,
            "cgwb_stage": cgwb_stage,
            "coastal_salinity_risk": coastal_salinity_risk,
            "s1_sar_wet_dry_ratio": s1_sar_wet_dry_ratio,
            "data_confidence_pct": data_confidence_pct,
            # Satellite Observed 5-Yr Deterioration Target (for Supervised ML)
            "observed_deterioration_index": observed_deterioration_index,
        }
        records.append(rec)

        polygon_features.append(
            {
                "type": "Feature",
                "id": mw_id,
                "properties": {"watershed_id": mw_id, "name": name, "basin": basin},
                "geometry": {"type": "Polygon", "coordinates": [poly_coords]},
            }
        )
        stream_features.append(
            {
                "type": "Feature",
                "id": f"STR-{mw_id}",
                "properties": {
                    "watershed_id": mw_id,
                    "name": f"{name} Main Channel",
                    "stream_order": stream_order,
                },
                "geometry": {"type": "LineString", "coordinates": stream_coords},
            }
        )

    df = pd.DataFrame(records)
    watersheds_geojson = {"type": "FeatureCollection", "features": polygon_features}
    streams_geojson = {"type": "FeatureCollection", "features": stream_features}
    return df, watersheds_geojson, streams_geojson


def save_generated_artifacts(base_dir: Path | None = None) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Saves CSV and GeoJSON files to the data/ folder for inspection and export."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    base_dir.mkdir(parents=True, exist_ok=True)
    df, ws_geojson, str_geojson = build_tamil_nadu_watershed_dataset()
    df.to_csv(base_dir / "tamil_nadu_micro_watersheds.csv", index=False)
    with open(base_dir / "tamil_nadu_watersheds.geojson", "w", encoding="utf-8") as f:
        json.dump(ws_geojson, f)
    with open(base_dir / "tamil_nadu_streams.geojson", "w", encoding="utf-8") as f:
        json.dump(str_geojson, f)
    return df, ws_geojson, str_geojson


if __name__ == "__main__":
    df_out, _, _ = save_generated_artifacts()
    print(f"Generated {len(df_out)} micro-watersheds across Chennai & Tamil Nadu.")
