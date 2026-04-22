"""
explain.py - Explainability layer for the spam classifier.

Provides one complementary methods:
    Feature importance  - fast, always available, works via TF-IDF weights.
"""

import re
import logging
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _tokenize(text: str):
    """Simple whitespace + punctuation tokenizer (mirrors sklearn defaults)."""
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


# ---------------------------------------------------------------------------
# Method : TF-IDF feature importance
# ---------------------------------------------------------------------------

def feature_importance_explanation(text: str, pipeline: Pipeline, top_n: int = 10) -> dict:
    """
    Return the top-N words that pushed the model toward its prediction.

    Works by looking at the TF-IDF weight * logistic-regression coefficient
    for each token present in the input.

    Returns
    -------
    {
        "method":      "feature_importance",
        "label":       "spam" | "ham",
        "top_words":   [{"word": str, "score": float}, …],
        "explanation": str          # human-readable summary
    }
    """
    vectorizer: Any = pipeline.named_steps["tfidf"]
    clf: Any        = pipeline.named_steps["clf"]
    classes         = clf.classes_.tolist()

    # TF-IDF transform for this single document
    tfidf_matrix = vectorizer.transform([text])            # (1, vocab)
    feature_names = vectorizer.get_feature_names_out()

    # Logistic regression coefficients for the positive class
    spam_idx    = classes.index("spam")
    coef_vector = clf.coef_[spam_idx] if len(classes) > 2 else clf.coef_[0]

    # Element-wise product: tfidf_weight * lr_coef
    tfidf_dense  = tfidf_matrix.toarray()[0]               # (vocab,)
    contribution = tfidf_dense * coef_vector               # (vocab,)

    # Predict label for this text
    pred_label   = pipeline.predict([text])[0]

    # Pick sign based on prediction: spam → positive contribs, ham → negative
    sign = 1 if pred_label == "spam" else -1
    top_indices = np.argsort(sign * contribution)[::-1][:top_n]

    top_words = [
        {"word": feature_names[i], "score": round(float(contribution[i]), 5)}
        for i in top_indices
        if tfidf_dense[i] > 0          # only words actually in the message
    ]

    if not top_words:
        top_words = [{"word": "(no strong signal)", "score": 0.0}]

    word_list = ", ".join(w["word"] for w in top_words[:5])
    explanation = (
        f"This message was classified as **{pred_label}** mainly because of "
        f"the following words: {word_list}."
    )

    return {
        "method":      "feature_importance",
        "label":       pred_label,
        "top_words":   top_words,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def explain(text: str, pipeline: Pipeline, top_n: int = 10) -> dict:
    """
    Explain a prediction using the requested method.

    Parameters
    ----------
    text     : input message
    pipeline : fitted sklearn Pipeline
    method   : "feature_importance" (default) | "lime"
    top_n    : number of top words to return
    """
    return feature_importance_explanation(text, pipeline, top_n)