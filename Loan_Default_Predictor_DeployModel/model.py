"""
model.py
Full ML lifecycle for Loan Default Prediction.
Handles preprocessing, feature engineering, class imbalance,
training two experiments with different hyperparameters, and MLflow tracking.
"""

import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from explain import run_shap_explanation
from drift import compute_drift_metrics

warnings.filterwarnings("ignore")

DATA_PATH = "data/Dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  1. DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────

def load_and_engineer(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # --- Type coercion (mixed-type columns come in as object) ---
    numeric_cols = [
        "Client_Income", "Credit_Amount", "Loan_Annuity",
        "Population_Region_Relative", "Age_Days", "Employed_Days",
        "Registration_Days", "ID_Days", "Score_Source_3"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Drop ID (not predictive) ---
    df.drop(columns=["ID"], inplace=True, errors="ignore")

    # --- Derived features ---
    df["Age_Years"] = df["Age_Days"].abs() / 365
    df["Employed_Years"] = df["Employed_Days"].abs() / 365
    df["Income_per_FamilyMember"] = df["Client_Income"] / (df["Client_Family_Members"].replace(0, 1))
    df["Debt_Income_Ratio"] = df["Credit_Amount"] / (df["Client_Income"].replace(0, np.nan))
    df["Annuity_Income_Ratio"] = df["Loan_Annuity"] / (df["Client_Income"].replace(0, np.nan))
    df["Credit_Annuity_Ratio"] = df["Credit_Amount"] / (df["Loan_Annuity"].replace(0, np.nan))
    df["Score_Mean"] = df[["Score_Source_1", "Score_Source_2", "Score_Source_3"]].mean(axis=1)
    df["Score_Std"] = df[["Score_Source_1", "Score_Source_2", "Score_Source_3"]].std(axis=1)
    df["Has_All_Scores"] = df[["Score_Source_1", "Score_Source_2", "Score_Source_3"]].notna().all(axis=1).astype(int)
    df["Phone_Reachability"] = df["Mobile_Tag"] + df["Homephone_Tag"] + df["Workphone_Working"]
    df["ID_Registration_Ratio"] = df["ID_Days"] / (df["Registration_Days"].replace(0, np.nan))

    return df


def get_column_types(df: pd.DataFrame):
    target = "Default"
    drop_cols = [target]
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    return num_cols, cat_cols


# ─────────────────────────────────────────────
#  2. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def build_preprocessor(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    # Encode categoricals manually (ColumnTransformer + OrdinalEncoder)
    from sklearn.preprocessing import OrdinalEncoder
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    return preprocessor


# ─────────────────────────────────────────────
#  3. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, prefix=""):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        f"{prefix}roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        f"{prefix}avg_precision": round(average_precision_score(y_test, y_prob), 4),
        f"{prefix}f1": round(f1_score(y_test, y_pred), 4),
        f"{prefix}precision": round(precision_score(y_test, y_pred), 4),
        f"{prefix}recall": round(recall_score(y_test, y_pred), 4),
    }
    return metrics, y_pred, y_prob


def run_experiment(experiment_name: str, params: dict, df: pd.DataFrame):
    """Run a single MLflow experiment with given hyperparameters."""

    mlflow.set_experiment(experiment_name)

    num_cols, cat_cols = get_column_types(df)
    X = df.drop(columns=["Default"])
    y = df["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)

    xgb_params = {k: v for k, v in params.items() if k != "smote_k"}
    smote_k = params.get("smote_k", 5)

    model = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=smote_k)),
        ("classifier", XGBClassifier(
            **xgb_params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    with mlflow.start_run(run_name=experiment_name):
        # Log params
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("feature_count", X.shape[1])

        # Fit
        model.fit(X_train, y_train)

        # Evaluate on val & test
        val_metrics, _, _ = evaluate_model(model, X_val, y_val, prefix="val_")
        test_metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, prefix="test_")

        mlflow.log_metrics({**val_metrics, **test_metrics})
        print(f"\n[{experiment_name}] Test ROC-AUC: {test_metrics['test_roc_auc']}")
        print(classification_report(y_test, y_pred))

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {experiment_name}")
        cm_path = f"artifacts/confusion_matrix_{experiment_name}.png"
        os.makedirs("artifacts", exist_ok=True)
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path)

        # SHAP explanation
        try:
            shap_path = run_shap_explanation(model, X_test, num_cols + cat_cols, experiment_name)
            mlflow.log_artifact(shap_path)
        except Exception as e:
            print(f"SHAP warning: {e}")

        # Data Drift (train vs test distribution)
        try:
            preprocessor_fitted = model.named_steps["preprocessor"]
            X_train_proc = preprocessor_fitted.transform(X_train)
            X_test_proc = preprocessor_fitted.transform(X_test)
            all_feature_names = num_cols + cat_cols
            drift_results = compute_drift_metrics(
                pd.DataFrame(X_train_proc, columns=all_feature_names),
                pd.DataFrame(X_test_proc, columns=all_feature_names),
                experiment_name
            )
            for k, v in drift_results.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            drift_path = f"artifacts/drift_{experiment_name}.csv"
            drift_results["details"].to_csv(drift_path, index=False)
            mlflow.log_artifact(drift_path)
        except Exception as e:
            print(f"Drift warning: {e}")

        # Save model
        model_path = os.path.join(MODEL_DIR, f"model_{experiment_name}.pkl")
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(model_path)

    return model, X_test, y_test


# ─────────────────────────────────────────────
#  4. MAIN: TWO EXPERIMENTS
# ─────────────────────────────────────────────

def main():
    print("Loading and engineering features...")
    df = load_and_engineer()
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['Default'].mean():.2%}")

    # Save a reference copy for drift baseline
    df.to_csv("data/processed_dataset.csv", index=False)

    # ── Experiment 1: Conservative XGBoost ──────────────────────────────
    params_exp1 = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": 11,   # ~ratio of negatives/positives
        "smote_k": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    print("\n=== Experiment 1: Conservative XGBoost ===")
    model1, X_test1, y_test1 = run_experiment("Exp1_Conservative_XGB", params_exp1, df)

    # ── Experiment 2: Aggressive XGBoost ────────────────────────────────
    params_exp2 = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 3,
        "scale_pos_weight": 11,
        "smote_k": 3,
        "gamma": 0.0,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    }
    print("\n=== Experiment 2: Deeper XGBoost ===")
    model2, X_test2, y_test2 = run_experiment("Exp2_Deeper_XGB", params_exp2, df)

    print("\n✅ Both experiments complete. Run `mlflow ui` to view results.")


if __name__ == "__main__":
    main()