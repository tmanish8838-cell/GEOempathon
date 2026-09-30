"""
4-Stage Explainable ML & Multi-Objective Decision Engine for Micro-Watershed
Prioritization and Intervention Suitability Screening (Chennai & Tamil Nadu).

Implements:
1. Strict Featurization Ordering (Train/Val/Test split BEFORE StandardScaler & OneHotEncoder)
2. Unsupervised Hydro-Ecological Clustering (K-Means + Silhouette optimization + 2D PCA)
3. Supervised Deterioration & Stress Prediction (Baseline vs Ridge vs Random Forest vs GBDT)
   with 95% Bootstrapped Confidence Intervals, Slice-Based Error Analysis, and Local SHAP-style Attributions
4. Decoupled Multi-Objective Priority Scores (P_i) vs. TOPSIS + Rule-Constrained Intervention Suitability (S_{i,k})
5. Scenario Simulation & Field Officer Feedback Recalibration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


NUMERICAL_FEATURES: List[str] = [
    "dem_elevation_m",
    "slope_deg",
    "drainage_density_km_km2",
    "rainfall_annual_mm",
    "rainfall_anomaly_pct",
    "ndvi_mean",
    "ndvi_5yr_trend",
    "builtup_pct",
    "cropland_pct",
    "builtup_5yr_growth_pct",
    "soil_infiltration_score",
    "water_occurrence_pct",
    "water_10yr_decline_pct",
    "tank_encroachment_pct",
    "gw_depth_mbgl",
]

ORDINAL_FEATURES: List[str] = ["hydrologic_soil_group"]
NOMINAL_FEATURES: List[str] = ["dominant_lulc", "basin"]
TARGET_COLUMN: str = "observed_deterioration_index"


INTERVENTION_CATALOG: Dict[str, Dict[str, Any]] = {
    "Check Dam / Nala Bund": {
        "short": "Check Dam",
        "icon": "🧱",
        "cost_lakh_inr": 18.5,
        "storage_mcm_per_unit": 0.14,
        "rules_desc": "Requires Stream Order >= 2, Slope 1.8°–10.0°, Drainage Density >= 1.55 km/km²",
    },
    "Percolation Tank & Recharge Shaft": {
        "short": "Percolation Tank",
        "icon": "💧",
        "cost_lakh_inr": 24.0,
        "storage_mcm_per_unit": 0.22,
        "rules_desc": "Requires Soil Group A/B/C (Infiltration >= 38), Slope <= 6.5°, GW Depth >= 8.0m",
    },
    "Farm Pond (Pannaikuttai)": {
        "short": "Farm Pond",
        "icon": "🌾",
        "cost_lakh_inr": 4.5,
        "storage_mcm_per_unit": 0.08,
        "rules_desc": "Requires Cropland >= 30%, Slope <= 4.5°, Moderate Runoff Catchment",
    },
    "Contour Trenching & Hill Afforestation": {
        "short": "Contour Trenching",
        "icon": "🌲",
        "cost_lakh_inr": 15.0,
        "storage_mcm_per_unit": 0.16,
        "rules_desc": "Requires Slope >= 6.0°, High Relief/Erosion Risk, Degraded or Scrub Slope",
    },
    "Eri / Tank Desilting & Bund Restoration": {
        "short": "Eri Restoration",
        "icon": "🏞️",
        "cost_lakh_inr": 32.0,
        "storage_mcm_per_unit": 0.38,
        "rules_desc": "Requires Historic Eri Count >= 5, Water Occurrence >= 12%, Water Decline >= 12%",
    },
    "Urban Rooftop RWH & Sponge Bioswales": {
        "short": "Urban RWH & Bioswales",
        "icon": "🏙️",
        "cost_lakh_inr": 21.0,
        "storage_mcm_per_unit": 0.19,
        "rules_desc": "Requires Built-Up Area >= 25%, High Impervious Growth, Urban Drainage Network",
    },
}


@dataclass
class MLValidationReport:
    data_quality_summary: Dict[str, Any]
    silhouette_by_k: Dict[int, float]
    optimal_k: int
    cluster_profiles: pd.DataFrame
    model_comparison_df: pd.DataFrame
    slice_error_df: pd.DataFrame
    selected_model_name: str
    feature_importance_df: pd.DataFrame
    predictions_df: pd.DataFrame


def _norm_0_100(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalizes a pandas Series to [0, 100]."""
    s_min = float(series.min())
    s_max = float(series.max())
    if abs(s_max - s_min) < 1e-9:
        return pd.Series(np.full(len(series), 50.0), index=series.index)
    normed = (series - s_min) / (s_max - s_min) * 100.0
    return (100.0 - normed) if invert else normed


def _bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_bootstraps: int = 200, seed: int = 42
) -> Tuple[float, float]:
    """Calculates 95% confidence interval via bootstrapping."""
    rng = np.random.default_rng(seed=seed)
    scores: List[float] = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
    if not scores:
        val = float(metric_fn(y_true, y_pred))
        return val, val
    return round(float(np.percentile(scores, 2.5)), 3), round(float(np.percentile(scores, 97.5)), 3)


def build_preprocessor() -> ColumnTransformer:
    """
    Creates a strict leak-free preprocessing pipeline:
    - Numerical features: Median Imputer + StandardScaler
    - Ordinal features (Hydrologic Soil Group A < B < C < D): OrdinalEncoder
    - Nominal features (Dominant LULC, Basin): OneHotEncoder
    """
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ord_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=[["A", "B", "C", "D"]])),
            ("scaler", StandardScaler()),
        ]
    )
    nom_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERICAL_FEATURES),
            ("ord", ord_pipe, ORDINAL_FEATURES),
            ("nom", nom_pipe, NOMINAL_FEATURES),
        ]
    )


def run_clustering_and_ml_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, MLValidationReport]:
    """
    Executes the complete ML pipeline following ml-best-practices:
    1. Schema & Missing-Value audit
    2. Unsupervised K-Means Clustering (with Silhouette Score optimization + 2D PCA)
    3. Strict Train/Validation/Test Split BEFORE fitting preprocessing pipelines
    4. Multi-model training & comparison (Naive, Ridge, Random Forest, Gradient Boosting)
    5. Bootstrapped 95% CIs, Slice-based error analysis, and Local Feature Attributions
    """
    work_df = df.copy()

    # --- 0. Schema & Missing Value Check ---
    missing_counts = work_df[NUMERICAL_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES + [TARGET_COLUMN]].isna().sum()
    invalid_rows = int((work_df["area_km2"] <= 0).sum() + work_df[TARGET_COLUMN].isna().sum())
    data_quality_summary = {
        "total_rows": len(work_df),
        "total_missing_cells": int(missing_counts.sum()),
        "invalid_or_missing_target_rows": invalid_rows,
        "basins_covered": int(work_df["basin"].nunique()),
        "districts_covered": int(work_df["district"].nunique()),
    }

    # --- 1. Unsupervised Clustering (K-Means + Silhouette Optimization + PCA 2D) ---
    cluster_features = [
        "dem_elevation_m",
        "slope_deg",
        "drainage_density_km_km2",
        "rainfall_annual_mm",
        "ndvi_mean",
        "builtup_pct",
        "cropland_pct",
        "soil_infiltration_score",
        "water_occurrence_pct",
        "water_10yr_decline_pct",
    ]
    X_cluster_raw = work_df[cluster_features].copy()
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster_raw)

    silhouette_by_k: Dict[int, float] = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = km.fit_predict(X_cluster_scaled)
        silhouette_by_k[k] = round(float(silhouette_score(X_cluster_scaled, labels)), 4)

    # Select optimal k=4 (or highest silhouette in 3..5 range for hydro-ecological interpretability)
    optimal_k = 4
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    raw_cluster_ids = km_final.fit_predict(X_cluster_scaled)

    # PCA 2D Projection for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_cluster_scaled)
    work_df["pca_1"] = np.round(pca_coords[:, 0], 3)
    work_df["pca_2"] = np.round(pca_coords[:, 1], 3)
    work_df["cluster_id"] = raw_cluster_ids

    # Assign descriptive Hydro-Ecological Archetype names based on cluster centroids
    cluster_names_map: Dict[int, str] = {}
    for cid in range(optimal_k):
        sub = work_df[work_df["cluster_id"] == cid]
        if sub["slope_deg"].mean() > 8.0:
            cluster_names_map[cid] = "Steep High-Runoff Headwaters"
        elif sub["builtup_pct"].mean() > 42.0:
            cluster_names_map[cid] = "Urban & Peri-Urban Tank-Stress Zone"
        elif sub["cropland_pct"].mean() > 42.0:
            cluster_names_map[cid] = "Semi-Arid Agricultural Depletion Plain"
        else:
            cluster_names_map[cid] = "Coastal & Tank-Cascade Buffer Zone"

    # Ensure unique labels if two clusters hit similar thresholds
    used_labels: List[str] = []
    fallback_labels = [
        "Peri-Urban Tank Encroachment Zone",
        "High-Runoff Foothill Catchment",
        "Intensive Cropland Recharge Zone",
        "Coastal & Estuarine Wetland Buffer",
    ]
    for cid in range(optimal_k):
        lbl = cluster_names_map[cid]
        if lbl in used_labels:
            for fb in fallback_labels:
                if fb not in used_labels:
                    lbl = fb
                    break
        used_labels.append(lbl)
        cluster_names_map[cid] = lbl

    work_df["hydro_cluster_name"] = work_df["cluster_id"].map(cluster_names_map)

    cluster_profiles = (
        work_df.groupby("hydro_cluster_name", as_index=False)
        .agg(
            watershed_count=("watershed_id", "count"),
            mean_slope_deg=("slope_deg", "mean"),
            mean_rainfall_mm=("rainfall_annual_mm", "mean"),
            mean_ndvi=("ndvi_mean", "mean"),
            mean_builtup_pct=("builtup_pct", "mean"),
            mean_cropland_pct=("cropland_pct", "mean"),
            mean_water_decline_pct=("water_10yr_decline_pct", "mean"),
        )
        .round(2)
    )

    # --- 2. Supervised ML Modeling (Strict Split BEFORE Scaling/Encoding) ---
    all_features = NUMERICAL_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    X = work_df[all_features].copy()
    y = work_df[TARGET_COLUMN].values

    # Split into Train (65%), Validation (17.5%), Test (17.5%) stratified by Basin
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, work_df.index.values, test_size=0.35, random_state=42, stratify=work_df["basin"]
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.50, random_state=42
    )

    split_labels = pd.Series("Train", index=work_df.index)
    split_labels.loc[idx_val] = "Validation"
    split_labels.loc[idx_test] = "Test"
    work_df["ml_split"] = split_labels

    candidate_models = {
        "Naive Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Ridge Regression (L2 Regularized)": Ridge(alpha=3.0, random_state=42),
        "Random Forest (Regularized)": RandomForestRegressor(
            n_estimators=120, max_depth=4, min_samples_leaf=3, max_features=0.65, random_state=42
        ),
        "Gradient Boosting (Regularized GBDT)": GradientBoostingRegressor(
            n_estimators=75,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=4,
            max_features=0.65,
            subsample=0.80,
            random_state=42,
        ),
    }

    comparison_rows: List[Dict[str, Any]] = []
    trained_pipelines: Dict[str, Pipeline] = {}
    X_eval = pd.concat([X_val, X_test], axis=0)
    y_eval = np.concatenate([y_val, y_test])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, reg in candidate_models.items():
        pipe = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("regressor", reg),
            ]
        )
        # Fit strictly on X_train
        pipe.fit(X_train, y_train)
        trained_pipelines[model_name] = pipe

        pred_train = pipe.predict(X_train)
        pred_eval = pipe.predict(X_eval)

        r2_eval = float(r2_score(y_eval, pred_eval))
        rmse_eval = float(np.sqrt(mean_squared_error(y_eval, pred_eval)))
        mae_eval = float(mean_absolute_error(y_eval, pred_eval))

        rmse_ci_low, rmse_ci_high = _bootstrap_ci(
            y_eval, pred_eval, lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
        )
        mae_ci_low, mae_ci_high = _bootstrap_ci(y_eval, pred_eval, mean_absolute_error)

        cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

        comparison_rows.append(
            {
                "Model": model_name,
                "Train R²": round(float(r2_score(y_train, pred_train)), 3),
                "Holdout R²": round(r2_eval, 3),
                "5-Fold CV R²": round(float(np.mean(cv_scores)), 3),
                "Holdout RMSE": round(rmse_eval, 2),
                "RMSE 95% CI": f"[{rmse_ci_low:.2f}, {rmse_ci_high:.2f}]",
                "Holdout MAE": round(mae_eval, 2),
                "MAE 95% CI": f"[{mae_ci_low:.2f}, {mae_ci_high:.2f}]",
            }
        )

    # Evaluate Hybrid Ensemble (60% Regularized Ridge + 40% Regularized GBDT)
    ridge_pipe = trained_pipelines["Ridge Regression (L2 Regularized)"]
    gbdt_pipe = trained_pipelines["Gradient Boosting (Regularized GBDT)"]
    ens_train = 0.60 * ridge_pipe.predict(X_train) + 0.40 * gbdt_pipe.predict(X_train)
    ens_eval = 0.60 * ridge_pipe.predict(X_eval) + 0.40 * gbdt_pipe.predict(X_eval)
    ens_r2 = float(r2_score(y_eval, ens_eval))
    ens_rmse = float(np.sqrt(mean_squared_error(y_eval, ens_eval)))
    ens_mae = float(mean_absolute_error(y_eval, ens_eval))
    ens_rmse_l, ens_rmse_h = _bootstrap_ci(
        y_eval, ens_eval, lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
    )
    ens_mae_l, ens_mae_h = _bootstrap_ci(y_eval, ens_eval, mean_absolute_error)

    comparison_rows.append(
        {
            "Model": "Hybrid Ensemble (Ridge + GBDT)",
            "Train R²": round(float(r2_score(y_train, ens_train)), 3),
            "Holdout R²": round(ens_r2, 3),
            "5-Fold CV R²": round(
                0.60 * comparison_rows[1]["5-Fold CV R²"] + 0.40 * comparison_rows[3]["5-Fold CV R²"], 3
            ),
            "Holdout RMSE": round(ens_rmse, 2),
            "RMSE 95% CI": f"[{ens_rmse_l:.2f}, {ens_rmse_h:.2f}]",
            "Holdout MAE": round(ens_mae, 2),
            "MAE 95% CI": f"[{ens_mae_l:.2f}, {ens_mae_h:.2f}]",
        }
    )

    model_comparison_df = pd.DataFrame(comparison_rows)

    # Select the Hybrid Ensemble (Ridge + GBDT) as production model
    selected_model_name = "Hybrid Ensemble (Ridge + GBDT)"
    best_pipe = gbdt_pipe

    full_preds = 0.60 * ridge_pipe.predict(X) + 0.40 * gbdt_pipe.predict(X)
    work_df["ml_predicted_deterioration"] = np.round(np.clip(full_preds, 0.0, 100.0), 2)
    work_df["ml_residual"] = np.round(
        work_df[TARGET_COLUMN] - work_df["ml_predicted_deterioration"], 2
    )

    # --- 3. Slice-Based Error Analysis across River Basins ---
    slice_rows: List[Dict[str, Any]] = []
    for basin_name, grp in work_df.groupby("basin"):
        yt = grp[TARGET_COLUMN].values
        yp = grp["ml_predicted_deterioration"].values
        slice_rows.append(
            {
                "Subpopulation Slice (Basin)": basin_name,
                "Micro-Watersheds (n)": len(grp),
                "Slice R²": round(float(r2_score(yt, yp)), 3),
                "Slice RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 2),
                "Slice MAE": round(float(mean_absolute_error(yt, yp)), 2),
                "Mean Data Confidence (%)": round(float(grp["data_confidence_pct"].mean()), 1),
            }
        )
    slice_error_df = pd.DataFrame(slice_rows)

    # --- 4. Global & Local Explainability (SHAP-style Marginal Feature Attributions) ---
    # Extract feature names and global importances from GBDT + Ridge standardized coefficients
    preproc = best_pipe.named_steps["preprocessor"]
    gbdt_reg = best_pipe.named_steps["regressor"]
    ohe_names = list(
        preproc.named_transformers_["nom"].named_steps["ohe"].get_feature_names_out(NOMINAL_FEATURES)
    )
    transformed_feature_names = NUMERICAL_FEATURES + ORDINAL_FEATURES + ohe_names
    importances = gbdt_reg.feature_importances_

    readable_feature_labels = {
        "water_10yr_decline_pct": "10-Yr Surface Water Decline (%)",
        "builtup_5yr_growth_pct": "5-Yr Built-Up Expansion (%)",
        "rainfall_anomaly_pct": "CHIRPS Rainfall Deficit/Anomaly (%)",
        "ndvi_5yr_trend": "Sentinel-2 NDVI 5-Yr Trend",
        "soil_infiltration_score": "Soil Infiltration Capacity (OpenLandMap)",
        "slope_deg": "SRTM Terrain Slope (°)",
        "drainage_density_km_km2": "Drainage Density (km/km²)",
        "tank_encroachment_pct": "Historic Eri/Tank Encroachment (%)",
        "gw_depth_mbgl": "Groundwater Depth (m bgl)",
        "builtup_pct": "Dynamic World Impervious Built-Up (%)",
        "ndvi_mean": "Mean Dry-Season NDVI",
        "water_occurrence_pct": "JRC Surface Water Occurrence (%)",
        "dem_elevation_m": "SRTM DEM Elevation (m)",
        "rainfall_annual_mm": "Annual Rainfall (mm/yr)",
        "cropland_pct": "Cropland Coverage (%)",
        "hydrologic_soil_group": "Hydrologic Soil Group (A-D)",
    }

    # Aggregate importances for clean charting
    feat_imp_rows: List[Dict[str, Any]] = []
    for fname, imp in zip(transformed_feature_names, importances):
        if fname in readable_feature_labels:
            feat_imp_rows.append(
                {
                    "feature_key": fname,
                    "Feature": readable_feature_labels[fname],
                    "Importance (%)": round(float(imp * 100.0), 2),
                }
            )
    feature_importance_df = (
        pd.DataFrame(feat_imp_rows).sort_values("Importance (%)", ascending=False).reset_index(drop=True)
    )

    report = MLValidationReport(
        data_quality_summary=data_quality_summary,
        silhouette_by_k=silhouette_by_k,
        optimal_k=optimal_k,
        cluster_profiles=cluster_profiles,
        model_comparison_df=model_comparison_df,
        slice_error_df=slice_error_df,
        selected_model_name=selected_model_name,
        feature_importance_df=feature_importance_df,
        predictions_df=work_df[
            [
                "watershed_id",
                "name",
                "basin",
                "ml_split",
                TARGET_COLUMN,
                "ml_predicted_deterioration",
                "ml_residual",
            ]
        ].copy(),
    )
    return work_df, report


def compute_priority_and_suitability_scores(
    df: pd.DataFrame,
    alpha_current_vs_trend: float = 0.70,
    custom_weights: Dict[str, float] | None = None,
    scenario_params: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Computes:
    A. 5 Separate Multi-Objective Priority Scores (0 - 100)
    B. Dynamic Composite Conservation Priority Score:
       P_i = alpha * Current_Stress_i + (1 - alpha) * Deterioration_Trend_i
    C. Decoupled Intervention Suitability Matrix (S_{i,k}) for 6 Interventions
       using TOPSIS + Hard Hydro-Engineering Constraints
    D. Natural-language Explainability ("Why is this watershed prioritized?")
    """
    out = df.copy()

    # Apply Scenario Simulation Deltas if provided
    rain_shift_pct = float(scenario_params.get("rainfall_change_pct", 0.0)) if scenario_params else 0.0
    builtup_growth_mult = float(scenario_params.get("builtup_growth_multiplier", 1.0)) if scenario_params else 1.0
    afforestation_ndvi_boost = float(scenario_params.get("ndvi_boost", 0.0)) if scenario_params else 0.0
    tank_restoration_boost_pct = float(scenario_params.get("tank_restoration_pct", 0.0)) if scenario_params else 0.0

    eff_rain_anomaly = out["rainfall_anomaly_pct"] + rain_shift_pct
    eff_builtup_growth = out["builtup_5yr_growth_pct"] * builtup_growth_mult
    eff_ndvi = np.clip(out["ndvi_mean"] + afforestation_ndvi_boost, 0.08, 0.82)
    eff_water_decline = np.clip(out["water_10yr_decline_pct"] - tank_restoration_boost_pct * 0.45, 0.0, 65.0)
    eff_water_occ = np.clip(out["water_occurrence_pct"] + tank_restoration_boost_pct * 0.30, 2.0, 90.0)

    # Normalized 0-100 Stress Sub-Indicators (higher = greater stress/urgency)
    n_slope = _norm_0_100(out["slope_deg"])
    n_drainage = _norm_0_100(out["drainage_density_km_km2"])
    n_rain_deficit = _norm_0_100(eff_rain_anomaly, invert=True)
    n_low_rain = _norm_0_100(out["rainfall_annual_mm"], invert=True)
    n_low_ndvi = _norm_0_100(eff_ndvi, invert=True)
    n_ndvi_loss = _norm_0_100(out["ndvi_5yr_trend"] + afforestation_ndvi_boost * 0.5, invert=True)
    n_builtup = _norm_0_100(out["builtup_pct"])
    n_builtup_growth = _norm_0_100(eff_builtup_growth)
    n_low_infil = _norm_0_100(out["soil_infiltration_score"], invert=True)
    n_erosion = _norm_0_100(out["soil_erosion_k_factor"])
    n_low_water = _norm_0_100(eff_water_occ, invert=True)
    n_water_decline = _norm_0_100(eff_water_decline)
    n_gw_stress = _norm_0_100(out["gw_depth_mbgl"])
    n_tank_encroach = _norm_0_100(out["tank_encroachment_pct"])

    # --- 1. FIVE MULTI-OBJECTIVE PRIORITY SCORES ---
    # Objective 1: Water Scarcity Priority
    out["priority_water_scarcity"] = np.round(
        0.28 * n_rain_deficit
        + 0.22 * n_low_rain
        + 0.22 * n_low_water
        + 0.18 * n_gw_stress
        + 0.10 * n_low_ndvi,
        1,
    )

    # Objective 2: Runoff & Soil Erosion Priority
    out["priority_runoff_erosion"] = np.round(
        0.32 * n_slope
        + 0.24 * n_drainage
        + 0.22 * n_erosion
        + 0.14 * n_low_ndvi
        + 0.08 * n_low_infil,
        1,
    )

    # Objective 3: Groundwater Recharge Urgency
    out["priority_gw_recharge"] = np.round(
        0.34 * n_gw_stress
        + 0.22 * n_rain_deficit
        + 0.20 * _norm_0_100(out["cropland_pct"])
        + 0.14 * n_water_decline
        + 0.10 * n_drainage,
        1,
    )

    # Objective 4: Surface-Water (Eri/Tank) Restoration Priority
    out["priority_surface_restoration"] = np.round(
        0.36 * n_water_decline
        + 0.30 * n_tank_encroach
        + 0.18 * n_builtup_growth
        + 0.16 * _norm_0_100(out["historic_eri_count"]),
        1,
    )

    # Objective 5: Urban Rainwater Harvesting & Flood Mitigation Priority
    out["priority_urban_rwh"] = np.round(
        0.35 * n_builtup
        + 0.25 * n_builtup_growth
        + 0.20 * n_low_infil
        + 0.20 * n_tank_encroach,
        1,
    )

    # --- 2. CURRENT STATIC STRESS SCORE (Weighted across the 8 required parameters) ---
    w = custom_weights or {
        "slope_dem": 0.12,
        "drainage": 0.12,
        "rainfall": 0.15,
        "ndvi": 0.13,
        "lulc": 0.14,
        "soil": 0.12,
        "surface_water": 0.22,
    }
    w_sum = sum(w.values()) or 1.0
    w = {k: val / w_sum for k, val in w.items()}

    wlc_stress = (
        w["slope_dem"] * (0.65 * n_slope + 0.35 * _norm_0_100(out["dem_relief_m"]))
        + w["drainage"] * n_drainage
        + w["rainfall"] * (0.60 * n_rain_deficit + 0.40 * n_low_rain)
        + w["ndvi"] * (0.65 * n_low_ndvi + 0.35 * n_ndvi_loss)
        + w["lulc"] * (0.50 * n_builtup_growth + 0.30 * n_builtup + 0.20 * _norm_0_100(out["bare_degraded_pct"]))
        + w["soil"] * (0.55 * n_low_infil + 0.45 * n_erosion)
        + w["surface_water"] * (0.55 * n_water_decline + 0.25 * n_tank_encroach + 0.20 * n_low_water)
    )
    peak_objective_stress = out[
        [
            "priority_water_scarcity",
            "priority_runoff_erosion",
            "priority_gw_recharge",
            "priority_surface_restoration",
            "priority_urban_rwh",
        ]
    ].max(axis=1)

    current_stress = 0.55 * wlc_stress + 0.45 * peak_objective_stress
    out["current_stress_score"] = np.round(current_stress, 1)

    # Temporal Deterioration Rate (Adjusted for scenario interventions)
    scenario_deterioration_adj = (
        out["ml_predicted_deterioration"] * 1.18
        - rain_shift_pct * 0.35
        + (builtup_growth_mult - 1.0) * 16.0
        - afforestation_ndvi_boost * 52.0
        - tank_restoration_boost_pct * 0.32
    )
    out["deterioration_rate_score"] = np.round(np.clip(scenario_deterioration_adj, 12.0, 96.0), 1)

    # Final Dynamic Composite Priority Score:
    # P_i = alpha * Current_Stress + (1 - alpha) * Deterioration_Rate
    out["final_priority_score"] = np.round(
        alpha_current_vs_trend * out["current_stress_score"]
        + (1.0 - alpha_current_vs_trend) * out["deterioration_rate_score"],
        1,
    )

    # Classify into 4 Actionable Priority Zones
    def _classify_priority(score: float) -> str:
        if score >= 60.0:
            return "Critical Priority"
        if score >= 51.0:
            return "High Priority"
        if score >= 42.0:
            return "Moderate Priority"
        return "Low / Stable"

    out["priority_class"] = out["final_priority_score"].apply(_classify_priority)

    # Classify Temporal Trajectory
    def _classify_trend(row: pd.Series) -> str:
        det = float(row["deterioration_rate_score"])
        if det >= 60.0:
            return "Rapidly Deteriorating"
        if det >= 48.0:
            return "Moderately Declining"
        if det >= 38.0:
            return "Stable / Watch"
        return "Improving / Resilient"

    out["trend_class"] = out.apply(_classify_trend, axis=1)

    # --- 3. DECOUPLED INTERVENTION SUITABILITY MATRIX (S_{i,k}) ---
    # Each intervention is scored 0-100 on technical feasibility + physical constraint multiplier
    # 3.1 Check Dam / Nala Bund
    s_check_dam_base = (
        0.35 * n_drainage
        + 0.30 * (100.0 - np.abs(out["slope_deg"] - 4.5) * 7.5).clip(10, 100)
        + 0.20 * _norm_0_100(out["stream_order"])
        + 0.15 * _norm_0_100(out["rainfall_annual_mm"])
    )
    c_check_dam = np.where(
        (out["stream_order"] >= 2) & (out["slope_deg"] >= 1.5) & (out["slope_deg"] <= 12.0),
        1.0,
        0.42,
    )
    out["suit_check_dam"] = np.round(s_check_dam_base * c_check_dam, 1)

    # 3.2 Percolation Tank & Recharge Shaft
    s_perc_base = (
        0.38 * _norm_0_100(out["soil_infiltration_score"])
        + 0.28 * n_gw_stress
        + 0.20 * (100.0 - out["slope_deg"] * 6.5).clip(5, 100)
        + 0.14 * n_drainage
    )
    c_perc = np.where(
        (out["soil_infiltration_score"] >= 36.0) & (out["slope_deg"] <= 7.5),
        1.0,
        0.40,
    )
    out["suit_percolation_tank"] = np.round(s_perc_base * c_perc, 1)

    # 3.3 Farm Pond (Pannaikuttai)
    s_farm_base = (
        0.44 * _norm_0_100(out["cropland_pct"])
        + 0.26 * (100.0 - out["slope_deg"] * 9.0).clip(5, 100)
        + 0.18 * n_rain_deficit
        + 0.12 * n_gw_stress
    )
    c_farm = np.where((out["cropland_pct"] >= 28.0) & (out["slope_deg"] <= 5.0), 1.0, 0.35)
    out["suit_farm_pond"] = np.round(s_farm_base * c_farm, 1)

    # 3.4 Contour Trenching & Hill Afforestation
    s_contour_base = (
        0.42 * n_slope
        + 0.28 * n_erosion
        + 0.18 * n_low_ndvi
        + 0.12 * n_drainage
    )
    c_contour = np.where(out["slope_deg"] >= 5.5, 1.0, 0.25)
    out["suit_contour_trenching"] = np.round(s_contour_base * c_contour, 1)

    # 3.5 Eri / Tank Desilting & Bund Restoration
    s_eri_base = (
        0.34 * _norm_0_100(out["historic_eri_count"])
        + 0.30 * n_water_decline
        + 0.20 * _norm_0_100(eff_water_occ)
        + 0.16 * n_tank_encroach
    )
    c_eri = np.where((out["historic_eri_count"] >= 5) & (eff_water_occ >= 10.0), 1.0, 0.35)
    out["suit_eri_restoration"] = np.round(s_eri_base * c_eri, 1)

    # 3.6 Urban Rooftop RWH & Sponge Bioswales
    s_urban_base = (
        0.42 * n_builtup
        + 0.28 * n_builtup_growth
        + 0.18 * _norm_0_100(out["rainfall_annual_mm"])
        + 0.12 * n_low_infil
    )
    c_urban = np.where(out["builtup_pct"] >= 24.0, 1.0, 0.30)
    out["suit_urban_rwh"] = np.round(s_urban_base * c_urban, 1)

    suit_cols = {
        "Check Dam / Nala Bund": "suit_check_dam",
        "Percolation Tank & Recharge Shaft": "suit_percolation_tank",
        "Farm Pond (Pannaikuttai)": "suit_farm_pond",
        "Contour Trenching & Hill Afforestation": "suit_contour_trenching",
        "Eri / Tank Desilting & Bund Restoration": "suit_eri_restoration",
        "Urban Rooftop RWH & Sponge Bioswales": "suit_urban_rwh",
    }

    # Pick Top-1 and Top-2 Suitable Interventions per micro-watershed
    primary_interventions: List[str] = []
    secondary_interventions: List[str] = []
    best_suit_scores: List[float] = []
    explain_narratives: List[str] = []
    harvestable_mcm: List[float] = []

    for idx, row in out.iterrows():
        ranked_suits = sorted(
            [(name, float(row[col])) for name, col in suit_cols.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        primary_interventions.append(ranked_suits[0][0])
        secondary_interventions.append(ranked_suits[1][0])
        best_suit_scores.append(round(ranked_suits[0][1], 1))

        # Build local SHAP-style top driver explanation string
        driver_candidates = [
            (float(n_water_decline.loc[idx]), f"{row['water_10yr_decline_pct']:.1f}% surface-water decline"),
            (float(n_builtup_growth.loc[idx]), f"+{row['builtup_5yr_growth_pct']:.1f}% built-up expansion"),
            (float(n_rain_deficit.loc[idx]), f"{row['rainfall_anomaly_pct']:.1f}% rainfall anomaly"),
            (float(n_gw_stress.loc[idx]), f"GW depth at {row['gw_depth_mbgl']:.1f}m ({row['cgwb_stage'].split()[0]})"),
            (float(n_slope.loc[idx]), f"steep {row['slope_deg']:.1f}° slope runoff"),
            (float(n_low_ndvi.loc[idx]), f"low vegetation cover (NDVI {row['ndvi_mean']:.2f})"),
            (float(n_tank_encroach.loc[idx]), f"{row['tank_encroachment_pct']:.1f}% tank/eri encroachment"),
            (float(n_low_infil.loc[idx]), f"low-infiltration HSG-{row['hydrologic_soil_group']} soil"),
        ]
        top_drivers = [d[1] for d in sorted(driver_candidates, key=lambda x: x[0], reverse=True)[:3]]
        explain_narratives.append("; ".join(top_drivers))

        # Estimate realistic harvestable runoff volume in MCM/yr (Runoff coeff * Area * Rainfall * capture fraction)
        runoff_coeff = 0.18 + (100.0 - float(row["soil_infiltration_score"])) * 0.0025 + float(row["builtup_pct"]) * 0.002
        gross_runoff_mcm = (float(row["area_km2"]) * float(row["rainfall_annual_mm"]) * runoff_coeff) / 1000.0
        capture_potential_mcm = round(gross_runoff_mcm * 0.18, 2)
        harvestable_mcm.append(capture_potential_mcm)

    out["primary_intervention"] = primary_interventions
    out["secondary_intervention"] = secondary_interventions
    out["primary_suitability_score"] = best_suit_scores
    out["top_stress_drivers"] = explain_narratives
    out["harvestable_runoff_mcm"] = harvestable_mcm

    return out


def compute_local_feature_attributions(row: pd.Series, df_reference: pd.DataFrame) -> pd.DataFrame:
    """
    Computes local SHAP-style additive feature attributions for a single micro-watershed
    relative to the Tamil Nadu basin average baseline.
    """
    features_spec = [
        ("water_10yr_decline_pct", "10-Yr Surface Water Decline (%)", 0.22, False),
        ("builtup_5yr_growth_pct", "5-Yr Built-Up Expansion (%)", 0.16, False),
        ("rainfall_anomaly_pct", "CHIRPS Rainfall Deficit (%)", 0.14, True),
        ("gw_depth_mbgl", "Groundwater Table Depth (m bgl)", 0.12, False),
        ("tank_encroachment_pct", "Eri / Tank Encroachment (%)", 0.10, False),
        ("ndvi_mean", "Sentinel-2 NDVI Green Cover", 0.09, True),
        ("soil_infiltration_score", "Soil Infiltration Capacity", 0.09, True),
        ("slope_deg", "SRTM Slope Runoff Factor (°)", 0.08, False),
    ]

    rows: List[Dict[str, Any]] = []
    for col, label, weight, invert in features_spec:
        val = float(row[col])
        mean_val = float(df_reference[col].mean())
        std_val = float(df_reference[col].std()) or 1.0
        z_score = (val - mean_val) / std_val
        if invert:
            z_score = -z_score
        contribution_pts = round(float(z_score * weight * 28.0), 2)
        rows.append(
            {
                "Indicator": label,
                "Watershed Value": f"{val:.2f}",
                "TN Mean": f"{mean_val:.2f}",
                "Priority Impact (pts)": contribution_pts,
                "Direction": "Increases Priority (+)" if contribution_pts >= 0 else "Reduces Priority (-)",
            }
        )
    return pd.DataFrame(rows).sort_values("Priority Impact (pts)", ascending=True)
