"""Tests for the multi-source TF-IDF retriever."""

import os

import pytest

from src.recommender import load_songs
from src.retriever import MultiSourceRetriever, RetrieverError

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
KNOW = os.path.join(os.path.dirname(__file__), "..", "knowledge")
HIST = os.path.join(os.path.dirname(__file__), "..", "data", "sample_user_history.json")


@pytest.fixture(scope="module")
def retriever():
    songs = load_songs(DATA)
    return MultiSourceRetriever().build_index(songs, knowledge_dir=KNOW, history_path=HIST)


def test_index_builds(retriever):
    assert retriever.is_built()


def test_song_retrieval_works(retriever):
    hits = retriever.retrieve("calm acoustic study music", top_k=5, sources=["song"])
    assert hits
    assert all(e.source_type == "song" for e in hits)
    assert all(e.source_id.startswith("song:") for e in hits)


def test_document_retrieval_works(retriever):
    hits = retriever.retrieve("what does energy and tempo mean", top_k=3, sources=["doc"])
    assert hits
    assert all(e.source_type == "doc" for e in hits)
    assert all(e.source_id.startswith("doc:") and "#" in e.source_id for e in hits)


def test_history_retrieval_works(retriever):
    hits = retriever.retrieve("late night coding lofi focus", top_k=1, sources=["history"])
    assert hits
    assert hits[0].source_type == "history"
    assert hits[0].source_id.startswith("history:")


def test_evidence_ids_are_stable(retriever):
    # Library Rain is song id 4 in the fixture catalog.
    hits = retriever.retrieve("library rain lofi", top_k=3, sources=["song"])
    ids = {e.source_id for e in hits}
    assert "song:4" in ids


def test_study_query_retrieves_study_context(retriever):
    hits = retriever.retrieve(
        "quiet instrumental focus music for studying", top_k=4, sources=["doc"]
    )
    joined = " ".join(e.source_id for e in hits)
    assert "studying" in joined or "energy" in joined


def test_workout_query_retrieves_energetic_evidence(retriever):
    songs = retriever.retrieve("high energy fast workout gym", top_k=3, sources=["song"])
    # Top workout songs should have high energy in their metadata-linked text.
    assert songs
    assert any("energy" in s.text or "workout" in s.text for s in songs)


def test_multi_source_returns_mixed_types(retriever):
    ev = retriever.retrieve_by_source(
        "calm study music", {"song": 5, "doc": 2, "history": 1}
    )
    types = {e.source_type for e in ev}
    assert "song" in types and "doc" in types


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("", top_k=5) == []


def test_missing_index_fails_gracefully():
    r = MultiSourceRetriever()
    with pytest.raises(RetrieverError):
        r.retrieve("anything", top_k=3)


def test_missing_knowledge_dir_is_graceful():
    songs = load_songs(DATA)
    r = MultiSourceRetriever().build_index(
        songs, knowledge_dir="does_not_exist_dir", history_path=None
    )
    # Still works with songs only.
    assert r.retrieve("pop happy", top_k=3, sources=["song"])
    assert r.retrieve("anything", top_k=3, sources=["doc"]) == []
