"""Tests for the specialized intent classifier and the keyword baseline."""

import pytest

from src.intent_classifier import (
    IntentClassifier,
    KeywordBaseline,
    Prediction,
    load_training_data,
    run_specialization_experiment,
)


@pytest.fixture(scope="module")
def trained():
    texts, labels = load_training_data()
    return IntentClassifier().train(texts, labels)


def test_training_succeeds(trained):
    assert trained.is_trained()
    assert set(trained.classes_) >= {"discover", "study", "workout", "out_of_scope"}


@pytest.mark.parametrize("text,expected", [
    ("calm quiet instrumental music for late night studying", "study"),
    ("high energy fast songs for my gym workout", "workout"),
    ("soothing music to relax and unwind before bed", "relax"),
    ("what is the weather forecast tomorrow", "out_of_scope"),
])
def test_known_examples_classify_correctly(trained, text, expected):
    assert trained.predict(text) == expected


def test_confidence_is_numeric_and_bounded(trained):
    pred = trained.predict_with_confidence("songs for studying")
    assert isinstance(pred, Prediction)
    assert 0.0 <= pred.confidence <= 1.0
    assert all(0.0 <= c <= 1.0 for _i, c in pred.alternatives)


def test_alternatives_are_returned(trained):
    pred = trained.predict_with_confidence("music for the gym", top_alt=2)
    assert len(pred.alternatives) == 2
    # The chosen intent should out-rank its alternatives.
    assert all(pred.confidence >= c for _i, c in pred.alternatives)


def test_clear_query_more_confident_than_gibberish(trained):
    clear = trained.predict_with_confidence("energetic workout running music").confidence
    vague = trained.predict_with_confidence("zxqw plok").confidence
    assert clear > vague


def test_keyword_baseline_runs():
    kb = KeywordBaseline()
    assert kb.predict("i need focus music for studying") == "study"
    assert kb.predict("what is the weather") == "out_of_scope"
    # Unknown phrasing falls back to the default guess.
    assert kb.predict("zzz") == "discover"


def test_specialized_model_outperforms_baseline():
    metrics = run_specialization_experiment()
    assert metrics["model_accuracy"] > metrics["baseline_accuracy"]
    # And by a meaningful margin on this dataset.
    assert metrics["improvement"] >= 0.05


def test_save_and_load_roundtrip(tmp_path, trained):
    path = tmp_path / "model.joblib"
    trained.save(str(path))
    loaded = IntentClassifier.load(str(path))
    assert loaded.predict("songs for a workout") == trained.predict("songs for a workout")
