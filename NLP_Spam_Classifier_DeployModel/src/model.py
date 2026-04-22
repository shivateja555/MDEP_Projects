"""
model.py - Training and inference logic for spam classification.

Dataset columns:
  - text : the email/message content
  - spam : 1 = spam, 0 = ham (not spam)
"""

import os
import pickle
import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("models", "spam_model.pkl")
DATA_PATH  = os.path.join("data", "spamEmails.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the spam CSV dataset and normalise column names."""
    df = pd.read_csv(path)

    # Dataset has columns: 'text' and 'spam' (1=spam, 0=ham)
    df = df[["text", "spam"]].dropna()
    df["label"] = df["spam"].apply(lambda x: "spam" if int(x) == 1 else "ham")

    logger.info("Loaded %d samples — %s", len(df), dict(df["label"].value_counts()))
    return df


def build_pipeline() -> Pipeline:
    """Build the TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> Pipeline:
    """Train the model and save it to disk. Returns the fitted pipeline."""
    df = load_data(data_path)

    X = df["text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    logger.info("Training pipeline ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Test accuracy: %.4f", acc)
    logger.info("\n%s", classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Model saved to %s", model_path)

    return pipeline


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    """Load a previously trained model from disk, or train a new one."""
    if not os.path.exists(model_path):
        logger.info("No model found at %s — training now ...", model_path)
        return train(model_path=model_path)

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    logger.info("Model loaded from %s", model_path)
    return pipeline


def predict(text: str, pipeline: Pipeline) -> dict:
    """
    Run inference on a single text string.

    Returns:
        {
            "label":       "spam" | "ham",
            "probability": {"spam": float, "ham": float}
        }
    """
    proba   = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_.tolist()
    prob_dict = dict(zip(classes, [round(float(p), 4) for p in proba]))
    label     = classes[int(proba.argmax())]
    return {"label": label, "probability": prob_dict}


if __name__ == "__main__":
    train()