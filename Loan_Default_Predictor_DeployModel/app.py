"""
app.py - Flask REST API for Loan Default Prediction
"""

import os
import json
import traceback
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_Exp1_Conservative_XGB.pkl")
model = None

def _load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model not found at {MODEL_PATH}. Run model.py first.")
        return
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded: {MODEL_PATH}")

_load_model()


def _engineer(df):
    numeric_cols = [
        "Client_Income", "Credit_Amount", "Loan_Annuity",
        "Population_Region_Relative", "Age_Days", "Employed_Days",
        "Registration_Days", "ID_Days", "Score_Source_3"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["ID", "Default"], errors="ignore")

    n = len(df)
    df["Age_Years"] = df["Age_Days"].abs() / 365 if "Age_Days" in df.columns else np.nan
    df["Employed_Years"] = df["Employed_Days"].abs() / 365 if "Employed_Days" in df.columns else np.nan

    ci = df["Client_Income"].replace(0, np.nan) if "Client_Income" in df.columns else pd.Series([np.nan]*n)
    ca = df["Credit_Amount"] if "Credit_Amount" in df.columns else pd.Series([np.nan]*n)
    la = df["Loan_Annuity"].replace(0, np.nan) if "Loan_Annuity" in df.columns else pd.Series([np.nan]*n)
    fm = df["Client_Family_Members"].replace(0, 1) if "Client_Family_Members" in df.columns else pd.Series([1]*n)

    df["Debt_Income_Ratio"] = ca.values / ci.values
    df["Annuity_Income_Ratio"] = la.values / ci.values
    df["Credit_Annuity_Ratio"] = ca.values / la.values
    df["Income_per_FamilyMember"] = ci.values / fm.values

    score_cols = [c for c in ["Score_Source_1", "Score_Source_2", "Score_Source_3"] if c in df.columns]
    df["Score_Mean"] = df[score_cols].mean(axis=1) if score_cols else np.nan
    df["Score_Std"] = df[score_cols].std(axis=1) if score_cols else np.nan
    df["Has_All_Scores"] = df[score_cols].notna().all(axis=1).astype(int) if score_cols else 0

    tag_cols = [c for c in ["Mobile_Tag", "Homephone_Tag", "Workphone_Working"] if c in df.columns]
    df["Phone_Reachability"] = df[tag_cols].sum(axis=1) if tag_cols else 0

    id_d = df["ID_Days"] if "ID_Days" in df.columns else pd.Series([np.nan]*n)
    reg_d = df["Registration_Days"].replace(0, np.nan) if "Registration_Days" in df.columns else pd.Series([np.nan]*n)
    df["ID_Registration_Ratio"] = id_d.values / reg_d.values

    return df


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run model.py first."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        input_df = _engineer(pd.DataFrame([data]))
        prob = float(model.predict_proba(input_df)[:, 1][0])
        pred = int(model.predict(input_df)[0])
        return jsonify({
            "prediction": pred,
            "probability_of_default": round(prob, 4),
            "risk_label": "High Risk" if prob > 0.5 else "Low Risk"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array"}), 400
    try:
        input_df = _engineer(pd.DataFrame(data))
        probs = model.predict_proba(input_df)[:, 1].tolist()
        preds = model.predict(input_df).tolist()
        results = [{"index": i, "prediction": int(preds[i]),
                    "probability_of_default": round(probs[i], 4),
                    "risk_label": "High Risk" if probs[i] > 0.5 else "Low Risk"}
                   for i in range(len(preds))]
        return jsonify({"predictions": results, "count": len(results)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)