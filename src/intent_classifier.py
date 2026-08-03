"""Specialized local intent classifier for VibeTrace AI.

This is a small, reproducible classroom "specialization" experiment: a
TF-IDF + Logistic Regression model trained on a hand-written synthetic dataset
(``data/intent_training.json``). It is NOT a production language model. A simple
keyword baseline is included so we can measure whether the trained model
actually helps on a held-out split.

Everything is deterministic: a fixed random seed and a fixed train/test split.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

RANDOM_SEED = 42
DEFAULT_TRAINING_PATH = "data/intent_training.json"
DEFAULT_MODEL_PATH = "models/intent_model.joblib"

INTENTS = [
    "discover",
    "study",
    "workout",
    "relax",
    "compare",
    "explain",
    "diversify",
    "out_of_scope",
]


@dataclass
class Prediction:
    """One classification result with calibrated-ish probability estimates."""

    intent: str
    confidence: float
    alternatives: List[Tuple[str, float]]

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "confidence": round(float(self.confidence), 4),
            "alternatives": [
                {"intent": i, "confidence": round(float(c), 4)}
                for i, c in self.alternatives
            ],
        }


def load_training_data(path: str = DEFAULT_TRAINING_PATH) -> Tuple[List[str], List[str]]:
    """Load (texts, labels) from the synthetic training JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    texts = [ex["text"] for ex in data["examples"]]
    labels = [ex["intent"] for ex in data["examples"]]
    return texts, labels


# ---------------------------------------------------------------------------
# Keyword baseline (for the specialization comparison)
# ---------------------------------------------------------------------------

# Ordered so earlier, more specific intents win ties.
_KEYWORD_RULES: List[Tuple[str, List[str]]] = [
    ("out_of_scope", ["weather", "stock", "invest", "recipe", "diagnos", "depress",
                       "anxiet", "medication", "medicine", "cure", "flight", "homework",
                       "quantum", "football", "dentist", "tax"]),
    ("compare", ["compare", "versus", " vs ", "difference between", "side by side",
                 "which is better", "contrast", "choose between"]),
    ("explain", ["explain", "why did", "why is", "why does", "reason", "justify",
                 "break down", "how did you"]),
    ("diversify", ["avoid repeating", "no repeat", "variety", "diverse", "mix it up",
                   "different artists", "different genres", "eclectic", "varied"]),
    ("study", ["study", "studying", "focus", "concentrat", "homework", "read", "essay",
               "coding at night", "productive"]),
    ("workout", ["workout", "gym", "run", "running", "cardio", "exercise", "lifting",
                 "treadmill", "hype", "pump", "training", "spin class"]),
    ("relax", ["relax", "unwind", "chill", "calm down", "soothing", "de-stress",
               "wind down", "peaceful", "cozy", "tranquil", "rest"]),
    ("discover", ["surprise me", "discover", "recommend", "suggestion", "suggest",
                  "new music", "find me", "something new", "what should i listen"]),
]


class KeywordBaseline:
    """A transparent rule-based classifier used only as a comparison baseline."""

    def predict(self, text: str) -> str:
        low = " " + text.lower() + " "
        for intent, keywords in _KEYWORD_RULES:
            if any(kw in low for kw in keywords):
                return intent
        # Default guess when no keyword matches.
        return "discover"

    def score(self, texts: List[str], labels: List[str]) -> float:
        correct = sum(1 for t, y in zip(texts, labels) if self.predict(t) == y)
        return correct / len(texts) if texts else 0.0


# ---------------------------------------------------------------------------
# Trained classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """TF-IDF + Logistic Regression intent classifier."""

    def __init__(self) -> None:
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = []

    # -- training / persistence ------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        # Word n-grams capture whole-word cues; character n-grams add robustness
        # to paraphrases, typos, and word variants. Together they generalize far
        # better than a fixed keyword ruleset on this small, varied dataset.
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                     sublinear_tf=True, stop_words=None)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=1, sublinear_tf=True)),
        ])
        return Pipeline(
            steps=[
                ("features", features),
                ("clf", LogisticRegression(
                    max_iter=4000,
                    C=8.0,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                )),
            ]
        )

    def train(self, texts: List[str], labels: List[str]) -> "IntentClassifier":
        """Fit the pipeline on all provided examples."""
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def is_trained(self) -> bool:
        return self.pipeline is not None

    def save(self, path: str = DEFAULT_MODEL_PATH) -> str:
        if not self.is_trained():
            raise RuntimeError("Cannot save an untrained classifier.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.pipeline, path)
        return path

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "IntentClassifier":
        obj = cls()
        obj.pipeline = joblib.load(path)
        obj.classes_ = list(obj.pipeline.named_steps["clf"].classes_)
        return obj

    @classmethod
    def load_or_train(
        cls,
        model_path: str = DEFAULT_MODEL_PATH,
        training_path: str = DEFAULT_TRAINING_PATH,
    ) -> "IntentClassifier":
        """Load a saved model if present, otherwise train from JSON in-memory.

        Training on the tiny synthetic dataset takes a fraction of a second, so
        the system works offline with no pre-built artifact required.
        """
        if os.path.exists(model_path):
            try:
                return cls.load(model_path)
            except Exception:
                pass  # Fall back to fresh training on any load error.
        texts, labels = load_training_data(training_path)
        return cls().train(texts, labels)

    # -- inference -------------------------------------------------------
    def predict(self, text: str) -> str:
        return self.predict_with_confidence(text).intent

    def predict_with_confidence(self, text: str, top_alt: int = 2) -> Prediction:
        if not self.is_trained():
            raise RuntimeError("Classifier is not trained.")
        probs = self.pipeline.predict_proba([text])[0]
        ranked = sorted(zip(self.classes_, probs), key=lambda kv: kv[1], reverse=True)
        best_intent, best_conf = ranked[0]
        alternatives = [(i, float(c)) for i, c in ranked[1 : 1 + top_alt]]
        return Prediction(
            intent=best_intent,
            confidence=float(best_conf),
            alternatives=alternatives,
        )


# ---------------------------------------------------------------------------
# Specialization experiment
# ---------------------------------------------------------------------------

N_FOLDS = 4


def run_specialization_experiment(
    training_path: str = DEFAULT_TRAINING_PATH,
) -> Dict:
    """Compare the keyword baseline against the trained model via cross-validation.

    Uses deterministic 4-fold stratified cross-validation (seed=42) so every
    example is scored while held out. Cross-validation is more honest than a
    single tiny holdout because it averages over folds instead of one lucky or
    unlucky split. Fully reproducible.
    """
    texts, labels = load_training_data(training_path)
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Baseline: out-of-fold predictions from the fixed keyword rules.
    baseline = KeywordBaseline()
    baseline_pred = [baseline.predict(t) for t in texts]

    # Model: out-of-fold predictions so no example is scored by a model that
    # was trained on it.
    pipeline = IntentClassifier()._build_pipeline()
    model_pred = list(cross_val_predict(pipeline, texts, y, cv=skf))

    def _acc(preds):
        return sum(1 for p, t in zip(preds, labels) if p == t) / len(labels)

    baseline_acc = _acc(baseline_pred)
    model_acc = _acc(model_pred)

    per_intent = {}
    for intent in INTENTS:
        idx = [i for i, t in enumerate(labels) if t == intent]
        if not idx:
            continue
        b = sum(1 for i in idx if baseline_pred[i] == labels[i]) / len(idx)
        m = sum(1 for i in idx if model_pred[i] == labels[i]) / len(idx)
        per_intent[intent] = {"baseline": b, "model": m, "support": len(idx)}

    return {
        "n_examples": len(texts),
        "n_folds": N_FOLDS,
        "baseline_accuracy": baseline_acc,
        "model_accuracy": model_acc,
        "improvement": model_acc - baseline_acc,
        "per_intent": per_intent,
    }
