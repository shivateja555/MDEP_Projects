"""
app.py - Flask REST API + Web UI for spam classification with explainability.

Endpoints
---------
GET  /              -> serves the web app (index.html)
GET  /health        -> liveness check
POST /predict       -> classify text + explain prediction
POST /predict/batch -> classify multiple texts at once
"""

import os
import logging

from flask import Flask, request, jsonify, render_template # type: ignore

from src.model   import load_model, predict as model_predict
from src.explain import explain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app      = Flask(__name__)
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        logger.info("Loading model ...")
        pipeline = load_model()
    return pipeline


# ── Web UI ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the web app."""
    return render_template("Index.html")


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body or "text" not in body:
        return jsonify({"error": "Request body must contain a 'text' field."}), 400

    text   = str(body["text"]).strip()

    if not text:
        return jsonify({"error": "'text' must not be empty."}), 400

    pipe        = get_pipeline()
    prediction  = model_predict(text, pipe)
    explanation = explain(text, pipe)

    return jsonify({
        "text":        text,
        "label":       prediction["label"],
        "probability": prediction["probability"],
        "explanation": explanation,
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    body = request.get_json(silent=True)
    if not body or "texts" not in body:
        return jsonify({"error": "Request body must contain a 'texts' list."}), 400

    texts  = body["texts"]

    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list."}), 400

    pipe    = get_pipeline()
    results = []
    for text in texts:
        text        = str(text).strip()
        prediction  = model_predict(text, pipe)
        explanation = explain(text, pipe)
        results.append({
            "text":        text,
            "label":       prediction["label"],
            "probability": prediction["probability"],
            "explanation": explanation,
        })

    return jsonify({"results": results, "count": len(results)})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    get_pipeline()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)