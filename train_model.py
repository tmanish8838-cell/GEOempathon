"""
Standalone Model Training & Evaluation Script for HydroShed AI (Problem Statement 2.3).

Trains and serializes:
1. Unsupervised K-Means Hydro-Ecological Clustering + PCA Pipeline
2. Supervised Deterioration Models (Naive, Ridge, Random Forest, GBDT, Hybrid Ensemble)
3. Exports trained model artifacts to `models/hydroshed_trained_bundle.joblib`
   and validation metrics to `models/training_metrics.json`.

Usage:
    python train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from data.tn_watersheds_generator import save_generated_artifacts
from models.watershed_ml_engine import (
    compute_priority_and_suitability_scores,
    run_clustering_and_ml_pipeline,
)


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/4: Loading Tamil Nadu & Chennai 60 Micro-Watershed Satellite Dataset...")
    df_raw, _, _ = save_generated_artifacts(data_dir)

    print("Step 2/4: Training K-Means Hydro-Clustering & Supervised ML Models (Train/Val/Test split)...")
    df_ml, report = run_clustering_and_ml_pipeline(df_raw)

    print("Step 3/4: Computing Decoupled Multi-Objective Priority (P_i) & Suitability (S_{i,k}) Scores...")
    df_scored = compute_priority_and_suitability_scores(df_ml)
    df_scored.to_csv(data_dir / "tamil_nadu_prioritized_output.csv", index=False)

    print("Step 4/4: Saving serialized model bundle and validation metrics...")
    bundle = {
        "selected_model_name": report.selected_model_name,
        "optimal_k": report.optimal_k,
        "silhouette_by_k": report.silhouette_by_k,
        "model_comparison": report.model_comparison_df.to_dict(orient="records"),
        "slice_error_analysis": report.slice_error_df.to_dict(orient="records"),
        "feature_importance": report.feature_importance_df.to_dict(orient="records"),
    }
    joblib.dump(bundle, models_dir / "hydroshed_trained_bundle.joblib")

    with open(models_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(report.model_comparison_df.to_string(index=False))
    print("\nSaved trained bundle -> models/hydroshed_trained_bundle.joblib")
    print("Saved metrics JSON   -> models/training_metrics.json")


if __name__ == "__main__":
    main()
