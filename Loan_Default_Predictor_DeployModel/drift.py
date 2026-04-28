"""
drift.py
Data Drift detection module for Loan Default Prediction.
Implements:
  - PSI  (Population Stability Index)   — overall distribution shift
  - CSI  (Characteristic Stability Index) — per-feature PSI variant
  - KS   (Kolmogorov-Smirnov test)      — feature distribution comparison

All metrics are designed to be logged to MLflow.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PSI Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_psi_single(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute PSI for one numeric feature.
    PSI < 0.1  → No significant change
    PSI 0.1–0.2 → Moderate change, investigate
    PSI > 0.2  → Significant change, likely data drift
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Use expected distribution to define bins
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # deduplicate

    if len(breakpoints) < 3:
        return np.nan  # not enough unique values to bin

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Avoid division by zero / log(0) with small smoothing
    expected_pct = (expected_counts / len(expected)) + 1e-8
    actual_pct = (actual_counts / len(actual)) + 1e-8

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def _interpret_psi(psi: float) -> str:
    if np.isnan(psi):
        return "N/A"
    if psi < 0.1:
        return "Stable"
    elif psi < 0.2:
        return "Moderate Shift"
    else:
        return "Significant Drift"


# ─────────────────────────────────────────────────────────────────────────────
#  KS Test Helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ks_single(expected: np.ndarray, actual: np.ndarray):
    """Run KS test for one numeric feature. Returns (statistic, p_value)."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 5 or len(actual) < 5:
        return np.nan, np.nan
    stat, pval = stats.ks_2samp(expected, actual)
    return float(stat), float(pval)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Drift Function
# ─────────────────────────────────────────────────────────────────────────────

def compute_drift_metrics(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    experiment_name: str,
    top_n_features: int = 20
) -> dict:
    """
    Compute PSI, CSI, and KS statistics for all numeric features.

    Parameters
    ----------
    reference_df : training/reference distribution (post-preprocessing)
    current_df   : test/current distribution (post-preprocessing)
    experiment_name : used for artifact naming
    top_n_features  : how many features to show in the drift plot

    Returns
    -------
    dict with scalar summary metrics and a 'details' DataFrame.
    """
    print(f"  [Drift] Computing PSI/CSI/KS for {experiment_name}...")

    feature_cols = reference_df.columns.tolist()
    results = []

    for col in feature_cols:
        ref_vals = reference_df[col].values.astype(float)
        cur_vals = current_df[col].values.astype(float)

        psi = _compute_psi_single(ref_vals, cur_vals)
        ks_stat, ks_pval = _compute_ks_single(ref_vals, cur_vals)

        results.append({
            "feature": col,
            "psi": psi,
            "csi": psi,          # CSI per-feature = PSI computed per feature
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "psi_interpretation": _interpret_psi(psi),
            "drift_detected": (
                (not np.isnan(psi) and psi > 0.2) or
                (not np.isnan(ks_pval) and ks_pval < 0.05)
            )
        })

    details_df = pd.DataFrame(results).sort_values("psi", ascending=False, na_position="last")

    # ── Summary scalars ───────────────────────────────────────────────────
    valid_psi = details_df["psi"].dropna()
    n_drifted = details_df["drift_detected"].sum()
    mean_psi = float(valid_psi.mean()) if len(valid_psi) > 0 else np.nan
    max_psi = float(valid_psi.max()) if len(valid_psi) > 0 else np.nan

    # ── Drift summary plot ────────────────────────────────────────────────
    plot_df = details_df.dropna(subset=["psi"]).head(top_n_features)
    if len(plot_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # PSI bar chart
        colors = ["#e74c3c" if v > 0.2 else "#f39c12" if v > 0.1 else "#2ecc71"
                  for v in plot_df["psi"]]
        axes[0].barh(plot_df["feature"], plot_df["psi"], color=colors)
        axes[0].axvline(x=0.1, color="orange", linestyle="--", label="Moderate (0.1)")
        axes[0].axvline(x=0.2, color="red", linestyle="--", label="High (0.2)")
        axes[0].set_title(f"PSI / CSI per Feature\n{experiment_name}")
        axes[0].set_xlabel("PSI Value")
        axes[0].legend()

        # KS statistic bar chart
        ks_df = plot_df.dropna(subset=["ks_statistic"]).head(top_n_features)
        ks_colors = ["#e74c3c" if p < 0.05 else "#2ecc71"
                     for p in ks_df["ks_pvalue"]]
        axes[1].barh(ks_df["feature"], ks_df["ks_statistic"], color=ks_colors)
        axes[1].set_title(f"KS Statistic per Feature\n(red = p<0.05 → drift)")
        axes[1].set_xlabel("KS Statistic")

        plt.tight_layout()
        drift_plot_path = os.path.join(ARTIFACT_DIR, f"drift_plot_{experiment_name}.png")
        plt.savefig(drift_plot_path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  [Drift] Plot saved: {drift_plot_path}")

    print(f"  [Drift] Mean PSI={mean_psi:.4f} | Max PSI={max_psi:.4f} | Features drifted: {n_drifted}/{len(details_df)}")

    return {
        "drift_mean_psi": mean_psi,
        "drift_max_psi": max_psi,
        "drift_n_features_drifted": int(n_drifted),
        "drift_pct_features_drifted": round(n_drifted / max(len(details_df), 1), 4),
        "details": details_df
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    ref = pd.DataFrame(np.random.randn(1000, 5), columns=[f"f{i}" for i in range(5)])
    cur = pd.DataFrame(np.random.randn(1000, 5) + [0, 0.5, 1.0, 0, 0],
                       columns=[f"f{i}" for i in range(5)])
    result = compute_drift_metrics(ref, cur, "test_drift")
    print(result["details"])