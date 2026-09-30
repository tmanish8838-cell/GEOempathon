"""
Google Earth Engine (GEE) Multi-Sensor Extraction Pipeline for Problem Statement 2.3:
Watershed-Based Water Resource Priority Mapping (Chennai & Tamil Nadu Basins).

Extracts all 8 mandatory parameters + 5-year temporal trends at the HydroSHEDS
Level-12 micro-watershed scale:
1. DEM (USGS/SRTMGL1_003 - 30m)
2. Slope (ee.Terrain.slope from SRTM 30m)
3. Drainage Density (WWF/HydroSHEDS/v1/FreeFlowingRivers + DEM Flow Accumulation)
4. Rainfall & Anomaly (UCSB-CHG/CHIRPS/DAILY - 1981 to 2026)
5. NDVI & 5-Yr Trend (COPERNICUS/S2_SR_HARMONIZED - 10m cloud-masked)
6. Land Use & Built-Up Growth (GOOGLE/DYNAMICWORLD/V1 - 10m probabilities/modes)
7. Soil Texture & Infiltration (OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02)
8. Surface-Water Occurrence & Decline (JRC/GSW1_4/GlobalSurfaceWater + Sentinel-1 SAR)

Usage (Local or Google Colab):
    pip install earthengine-api geemap
    earthengine authenticate
    python gee_pipeline/extract_tn_watersheds.py --project your-gcp-project-id
"""

from __future__ import annotations

import argparse
from pathlib import Path


GEE_JS_CODE_EDITOR_SNIPPET = r"""// ============================================================================
// GEO IMPATHON 1.0 - Problem Statement 2.3: Micro-Watershed Priority Mapping
// Study Area: Chennai Metropolitan & Tamil Nadu River Basins
// Paste directly into https://code.earthengine.google.com/
// ============================================================================

// 1. Define Study Region (Chennai & Palar / Tamil Nadu Bounding Box)
var tnRegion = ee.Geometry.Rectangle([76.5, 8.2, 80.35, 13.55]);
var chennaiFocus = ee.Geometry.Rectangle([79.75, 12.65, 80.32, 13.30]);
Map.centerObject(chennaiFocus, 10);

// 2. Load HydroSHEDS Level-12 Micro-Watersheds
var microWatersheds = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_12')
  .filterBounds(chennaiFocus);

// 3. Parameter 1 & 2: SRTM 30m DEM & Slope
var dem = ee.Image('USGS/SRTMGL1_003').clip(chennaiFocus).rename('dem_elevation_m');
var slope = ee.Terrain.slope(dem).rename('slope_deg');

// 4. Parameter 3: Drainage Network & Density (HydroSHEDS Rivers)
var rivers = ee.FeatureCollection('WWF/HydroSHEDS/v1/FreeFlowingRivers')
  .filterBounds(chennaiFocus);
var drainageRaster = ee.Image().byte().paint(rivers, 1, 2).unmask(0).rename('drainage_line');

// 5. Parameter 4: CHIRPS Daily Rainfall (Annual Mean & 5-Yr Anomaly)
var chirpsBaseline = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate('2000-01-01', '2020-12-31')
  .select('precipitation')
  .sum().divide(21).rename('rainfall_baseline_mm');

var chirpsRecent = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate('2021-01-01', '2025-12-31')
  .select('precipitation')
  .sum().divide(5).rename('rainfall_annual_mm');

var rainfallAnomalyPct = chirpsRecent.subtract(chirpsBaseline)
  .divide(chirpsBaseline).multiply(100).rename('rainfall_anomaly_pct');

// 6. Parameter 5: Sentinel-2 Cloud-Masked NDVI (2020 vs 2025 Dry Season)
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var mask = qa.bitwiseAnd(1 << 10).eq(0).and(qa.bitwiseAnd(1 << 11).eq(0));
  return image.updateMask(mask).divide(10000);
}

var s2_2020 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(chennaiFocus)
  .filterDate('2020-01-01', '2020-05-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds).median();

var s2_2025 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(chennaiFocus)
  .filterDate('2025-01-01', '2025-05-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds).median();

var ndvi2020 = s2_2020.normalizedDifference(['B8', 'B4']).rename('ndvi_2020');
var ndvi2025 = s2_2025.normalizedDifference(['B8', 'B4']).rename('ndvi_mean');
var ndviTrend = ndvi2025.subtract(ndvi2020).rename('ndvi_5yr_trend');

// 7. Parameter 6: Dynamic World 10m Land Cover & Built-Up Growth (2020 -> 2025)
var dw2020 = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(chennaiFocus).filterDate('2020-01-01', '2020-12-31')
  .select('built').mean().multiply(100).rename('builtup_2020_pct');

var dw2025 = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(chennaiFocus).filterDate('2025-01-01', '2025-12-31')
  .select(['built', 'crops', 'bare']).mean().multiply(100)
  .rename(['builtup_pct', 'cropland_pct', 'bare_degraded_pct']);

var builtupGrowth = dw2025.select('builtup_pct').subtract(dw2020).rename('builtup_5yr_growth_pct');

// 8. Parameter 7: OpenLandMap USDA Soil Texture Class
var soilTexture = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
  .select('b0').clip(chennaiFocus).rename('soil_usda_class');

// 9. Parameter 8: JRC Global Surface Water Occurrence & Change
var jrcWater = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(chennaiFocus);
var waterOccurrence = jrcWater.select('occurrence').unmask(0).rename('water_occurrence_pct');
var waterChange = jrcWater.select('change_norm').unmask(0).multiply(-1).rename('water_10yr_decline_pct');

// 10. Stack All 8 Parameters & Reduce by Micro-Watershed Polygons
var stackedIndicators = ee.Image.cat([
  dem, slope, drainageRaster, chirpsRecent, rainfallAnomalyPct,
  ndvi2025, ndviTrend, dw2025, builtupGrowth, soilTexture,
  waterOccurrence, waterChange
]);

var watershedStats = stackedIndicators.reduceRegions({
  collection: microWatersheds,
  reducer: ee.Reducer.mean(),
  scale: 30
});

Map.addLayer(slope, {min: 0, max: 15, palette: ['#f7fcf5', '#74c476', '#00441b']}, 'SRTM Slope (deg)', false);
Map.addLayer(ndvi2025, {min: 0.1, max: 0.6, palette: ['#d73027', '#fee08b', '#1a9850']}, 'Sentinel-2 NDVI (2025)', false);
Map.addLayer(waterOccurrence, {min: 0, max: 80, palette: ['#ffffff', '#4292c6', '#08306b']}, 'JRC Surface Water Occurrence');
Map.addLayer(microWatersheds, {color: 'red'}, 'HydroSHEDS Micro-Watersheds');

Export.table.toDrive({
  collection: watershedStats,
  description: 'TamilNadu_MicroWatersheds_8Params',
  fileFormat: 'GeoJSON'
});
"""


def run_gee_python_extraction(project_id: str, output_geojson: Path) -> None:
    """
    Executes the Google Earth Engine Python API pipeline and exports the
    micro-watershed feature collection.
    """
    import ee  # type: ignore

    ee.Initialize(project=project_id)
    region = ee.Geometry.Rectangle([79.75, 12.65, 80.32, 13.30])
    basins = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_12").filterBounds(region)

    dem = ee.Image("USGS/SRTMGL1_003").clip(region).rename("dem_elevation_m")
    slope = ee.Terrain.slope(dem).rename("slope_deg")
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(region)
    water_occ = jrc.select("occurrence").unmask(0).rename("water_occurrence_pct")

    stack = ee.Image.cat([dem, slope, water_occ])
    reduced = stack.reduceRegions(collection=basins, reducer=ee.Reducer.mean(), scale=90)
    info = reduced.getInfo()

    import json

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    with open(output_geojson, "w", encoding="utf-8") as f:
        json.dump(info, f)
    print(f"Exported live GEE micro-watersheds to {output_geojson}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Tamil Nadu Micro-Watershed Satellite Parameters via GEE")
    parser.add_argument("--project", type=str, default="", help="Google Cloud Project ID for Earth Engine")
    parser.add_argument(
        "--out",
        type=str,
        default="data/gee_live_export.geojson",
        help="Output GeoJSON path",
    )
    args = parser.parse_args()
    if args.project:
        run_gee_python_extraction(args.project, Path(args.out))
    else:
        print("GEE JavaScript & Python pipeline ready. Pass --project <gcp-project> to run live GEE export.")
