"""Tests for the output verifier and heuristic confidence scoring."""

from src.models import Recommendation
from src.verifier import compute_confidence, verify_output


def _rec(song_id=1, score=5.0, evidence=None):
    return Recommendation(
        song_id=song_id, title=f"T{song_id}", artist="A", genre="pop", mood="happy",
        score=score, reasons="genre match: pop (+3.00)",
        evidence_ids=evidence if evidence is not None else [f"song:{song_id}"],
    )


VALID_EVIDENCE = {"song:1", "song:2", "song:3", "doc:genre_guide.md#pop"}
CATALOG = {1, 2, 3}


def _verify(recs, response="1. T1\nEvidence: [song:1]", **kw):
    kwargs = dict(
        intent="discover", status="ok", response_text=response,
        recommendations=recs, valid_evidence_ids=VALID_EVIDENCE, catalog_ids=CATALOG,
        requested_k=len(recs), explicit_ok=None, explicit_song_ids=set(),
        low_confidence=False, warnings=[],
    )
    kwargs.update(kw)
    return verify_output(**kwargs)


def test_valid_grounded_response_passes():
    report = _verify([_rec(1, 5.0), _rec(2, 4.0)])
    assert report.passed
    assert report.pass_rate == 1.0


def test_detects_nonexistent_evidence_citation():
    report = _verify([_rec(1, 5.0, evidence=["song:999"])])
    assert not report.checks["evidence_ids_exist"]
    assert not report.passed


def test_detects_nonexistent_song_id():
    report = _verify([_rec(99, 5.0, evidence=["song:1"])])
    assert not report.checks["songs_exist"]


def test_detects_unordered_scores():
    report = _verify([_rec(1, 3.0), _rec(2, 9.0)])
    assert not report.checks["scores_ordered"]


def test_detects_ungrounded_recommendation():
    report = _verify([_rec(1, 5.0, evidence=[])])
    assert not report.checks["recommendations_grounded"]


def test_explicit_preference_violation_detected():
    report = _verify(
        [_rec(1, 5.0)], explicit_ok=False, explicit_song_ids={1},
    )
    assert not report.checks["explicit_respected"]


def test_banned_medical_claim_detected():
    report = _verify([_rec(1, 5.0)], response="This music will cure your anxiety.")
    assert not report.checks["no_unsupported_claims"]


def test_low_confidence_must_be_acknowledged():
    without_ack = _verify([_rec(1, 5.0)], low_confidence=True, warnings=[])
    assert not without_ack.checks["low_confidence_acknowledged"]
    with_ack = _verify([_rec(1, 5.0)], low_confidence=True,
                       warnings=["I wasn't sure, used a safe approach"])
    assert with_ack.checks["low_confidence_acknowledged"]


def test_guardrail_answer_skips_recommendation_checks():
    report = verify_output(
        intent="out_of_scope", status="guardrail",
        response_text="I only help with music.", recommendations=[],
        valid_evidence_ids=VALID_EVIDENCE, catalog_ids=CATALOG, requested_k=0,
        explicit_ok=None, explicit_song_ids=set(), low_confidence=False, warnings=[],
    )
    assert report.passed
    assert report.checks["safe_no_ranking"]


def test_confidence_is_bounded():
    recs = [_rec(1, 9.0), _rec(2, 4.0)]
    c = compute_confidence(0.8, 0.5, recs, 1.0)
    assert 0.0 <= c <= 1.0


def test_confidence_monotonic_in_verifier_rate():
    recs = [_rec(1, 9.0), _rec(2, 4.0)]
    high = compute_confidence(0.8, 0.5, recs, 1.0)
    low = compute_confidence(0.8, 0.5, recs, 0.5)
    assert high > low


def test_confidence_zero_with_no_signal():
    c = compute_confidence(0.0, 0.0, [], 0.0)
    assert c == 0.0
