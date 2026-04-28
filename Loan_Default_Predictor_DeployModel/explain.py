"""
explain.py
XAI (Explainable AI) module using SHAP for the Loan Default Prediction model.
Generates SHAP summary plots, waterfall plots, and feature importance.
All artifacts are saved locally and intended to be logged to MLflow.
"""

import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def run_shap_explanation(pipeline, X_test: pd.DataFrame, feature_names: list, experiment_name: str) -> str:
    """
    Compute SHAP values for the XGBoost classifier inside the pipeline.
    Saves summary bar plot + beeswarm plot.

    Returns path to the summary plot artifact.
    """
    print(f"  [SHAP] Computing explanations for {experiment_name}...")

    # Extract preprocessor and classifier from ImbPipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform test data (use a sample for speed)
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    X_transformed = preprocessor.transform(X_sample)
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # SHAP TreeExplainer (fast for XGBoost)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed_df)

    # ── Plot 1: Bar Summary (global feature importance) ──────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed_df, plot_type="bar",
                      max_display=20, show=False)
    plt.title(f"SHAP Feature Importance (Bar) - {experiment_name}")
    plt.tight_layout()
    bar_path = os.path.join(ARTIFACT_DIR, f"shap_bar_{experiment_name}.png")
    plt.savefig(bar_path, bbox_inches="tight", dpi=120)
    plt.close()

    # ── Plot 2: Beeswarm Summary (direction + magnitude) ─────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed_df, plot_type="dot",
                      max_display=20, show=False)
    plt.title(f"SHAP Summary (Beeswarm) - {experiment_name}")
    plt.tight_layout()
    beeswarm_path = os.path.join(ARTIFACT_DIR, f"shap_beeswarm_{experiment_name}.png")
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=120)
    plt.close()

    # ── Plot 3: Waterfall for a single high-risk prediction ───────────────
    try:
        probs = pipeline.predict_proba(X_sample)[:, 1]
        high_risk_idx = np.argmax(probs)
        shap_exp = shap.Explanation(
            values=shap_values[high_risk_idx],
            base_values=explainer.expected_value,
            data=X_transformed_df.iloc[high_risk_idx].values,
            feature_names=feature_names
        )
        fig3 = plt.figure(figsize=(12, 6))
        shap.waterfall_plot(shap_exp, max_display=15, show=False)
        plt.title(f"SHAP Waterfall - Highest Risk Case - {experiment_name}")
        waterfall_path = os.path.join(ARTIFACT_DIR, f"shap_waterfall_{experiment_name}.png")
        plt.savefig(waterfall_path, bbox_inches="tight", dpi=120)
        plt.close()
    except Exception as e:
        print(f"  [SHAP] Waterfall plot skipped: {e}")

    # ── Feature importance as CSV ─────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    shap_csv_path = os.path.join(ARTIFACT_DIR, f"shap_importance_{experiment_name}.csv")
    shap_df.to_csv(shap_csv_path, index=False)

    print(f"  [SHAP] Top 5 features:\n{shap_df.head(5).to_string(index=False)}")
    print(f"  [SHAP] Artifacts saved to {ARTIFACT_DIR}/")

    return bar_path


def explain_single_prediction(pipeline, input_df: pd.DataFrame, feature_names: list) -> dict:
    """
    Explain a single prediction. Used by app.py for inference-time XAI.
    Returns a dict of feature -> shap_value, sorted by |importance|.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(input_df)
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_df)

    shap_dict = dict(zip(feature_names, shap_values[0]))
    shap_sorted = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    return shap_sorted