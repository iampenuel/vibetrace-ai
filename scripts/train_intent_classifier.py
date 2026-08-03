"""Train the VibeTrace AI intent classifier and run the specialization experiment.

This script:
1. Runs a deterministic held-out comparison of the keyword baseline vs. the
   trained TF-IDF + Logistic Regression model.
2. Trains a final model on the full dataset and saves it to models/.
3. Writes outputs/specialization_comparison.txt.

Run:
    python scripts/train_intent_classifier.py
"""

from __future__ import annotations

import os
import sys

# Allow running as `python scripts/train_intent_classifier.py` from the repo root.
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.intent_classifier import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    IntentClassifier,
    load_training_data,
    run_specialization_experiment,
)

OUTPUT_PATH = "outputs/specialization_comparison.txt"


def format_report(metrics: dict) -> str:
    lines = []
    lines.append("VibeTrace AI — Intent Classifier Specialization Experiment")
    lines.append("=" * 62)
    lines.append("")
    lines.append(f"Setup: deterministic {metrics['n_folds']}-fold stratified cross-validation")
    lines.append(f"(seed=42) over a hand-written synthetic dataset of {metrics['n_examples']} examples")
    lines.append("across 8 intents. Every example is scored while held out of training.")
    lines.append("This is a small classroom specialization, NOT a production language model.")
    lines.append("")
    lines.append("Overall cross-validated accuracy")
    lines.append("-" * 62)
    lines.append(f"  Keyword baseline        : {metrics['baseline_accuracy'] * 100:6.2f}%")
    lines.append(f"  Specialized classifier  : {metrics['model_accuracy'] * 100:6.2f}%")
    lines.append(f"  Improvement             : {metrics['improvement'] * 100:+6.2f} points")
    lines.append("")
    lines.append("Per-intent cross-validated accuracy (baseline -> model)")
    lines.append("-" * 62)
    for intent, row in metrics["per_intent"].items():
        lines.append(
            f"  {intent:<14} support={row['support']}  "
            f"{row['baseline'] * 100:6.2f}% -> {row['model'] * 100:6.2f}%"
        )
    lines.append("")
    lines.append("Interpretation: the trained model generalizes to paraphrases the")
    lines.append("keyword rules miss (e.g. novel wording for 'study' or 'relax'), while")
    lines.append("the baseline only fires on exact keywords. Both are transparent.")
    return "\n".join(lines)


def main() -> int:
    metrics = run_specialization_experiment()
    report = format_report(metrics)
    print(report)

    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print(f"\nWrote {OUTPUT_PATH}")

    # Train and persist a final model on the full dataset.
    texts, labels = load_training_data()
    model = IntentClassifier().train(texts, labels)
    saved = model.save(DEFAULT_MODEL_PATH)
    print(f"Saved trained model to {saved} (gitignored; agent can also train in-memory).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
