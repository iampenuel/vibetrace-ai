"""End-to-end tests for the VibeTrace agent and the planner."""

import pytest

from src import planner
from src.agent import VibeTraceAgent

# Forbidden hidden-reasoning fields that must never appear in a trace.
FORBIDDEN_TRACE_FIELDS = {
    "chain_of_thought", "cot", "reasoning", "hidden_reasoning",
    "scratchpad", "thoughts", "internal_monologue",
}


@pytest.fixture(scope="module")
def agent():
    return VibeTraceAgent()


def _run(agent, query, **kw):
    kw.setdefault("log_path", None)
    return agent.run(query, **kw)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def test_discover_plan_ranks_candidates():
    plan = planner.plan_for_intent("discover")
    assert "rank_candidates" in plan
    assert "verify_output" == plan[-1]


def test_compare_plan_uses_comparison_step():
    plan = planner.plan_for_intent("compare")
    assert "identify_two_candidates" in plan
    assert "compare_candidates" in plan
    assert "rank_candidates" not in plan


def test_explain_plan_explains_ranking():
    plan = planner.plan_for_intent("explain")
    assert "explain_ranking" in plan


def test_out_of_scope_plan_skips_ranking():
    plan = planner.plan_for_intent("out_of_scope")
    assert "rank_candidates" not in plan
    assert "return_guardrail_response" in plan


def test_no_retrieval_toggle_changes_plan():
    with_r = planner.plan_for_intent("study", use_retrieval=True)
    without_r = planner.plan_for_intent("study", use_retrieval=False)
    assert "retrieve_context" in with_r
    assert "retrieve_context" not in without_r


# ---------------------------------------------------------------------------
# Agent end-to-end
# ---------------------------------------------------------------------------

def test_study_query_returns_grounded_recommendations(agent):
    r = _run(agent, "calm low-energy music for late-night studying", top_k=3)
    assert r.status == "ok"
    assert r.intent == "study"
    assert len(r.recommendations) == 3
    assert all(rec.evidence_ids for rec in r.recommendations)


def test_workout_differs_from_study(agent):
    study = _run(agent, "calm low-energy music for studying", top_k=3)
    workout = _run(agent, "high energy fast songs for my gym workout", top_k=3)
    study_titles = [x.title for x in study.recommendations]
    workout_titles = [x.title for x in workout.recommendations]
    assert study_titles != workout_titles


def test_compare_query_returns_two(agent):
    r = _run(agent, "Compare Library Rain and Midnight Coding for studying", top_k=2)
    assert r.intent == "compare"
    assert len(r.recommendations) == 2


def test_out_of_scope_query_is_safe(agent):
    r = _run(agent, "diagnose my anxiety and prescribe medication", top_k=3)
    assert r.status == "guardrail"
    assert r.intent == "out_of_scope"
    assert r.recommendations == []


def test_empty_query_is_guardrailed(agent):
    r = _run(agent, "", top_k=3)
    assert r.status == "guardrail"


def test_invalid_top_k_is_guardrailed(agent):
    r = _run(agent, "songs for studying", top_k=0)
    assert r.status == "guardrail"


def test_output_includes_evidence_ids(agent):
    r = _run(agent, "upbeat clean workout songs", top_k=3)
    assert any(rec.evidence_ids for rec in r.recommendations)
    assert "[song:" in r.response_text


def test_explicit_preference_respected(agent):
    r = _run(agent, "clean high energy workout songs with no explicit lyrics", top_k=5)
    explicit = [rec.song_id for rec in r.recommendations
                if agent._by_id[rec.song_id]["explicit"]]
    assert explicit == []


def test_diversify_avoids_repeating_artists(agent):
    r = _run(agent, "surprise me with variety but avoid repeating artists", top_k=5)
    artists = [rec.artist for rec in r.recommendations]
    assert len(artists) == len(set(artists))


def test_no_retrieval_removes_context_evidence(agent):
    with_r = _run(agent, "calm study music", top_k=3, use_retrieval=True)
    without_r = _run(agent, "calm study music", top_k=3, use_retrieval=False)
    with_docs = [e for e in with_r.evidence if e.source_type == "doc"]
    without_docs = [e for e in without_r.evidence if e.source_type == "doc"]
    assert with_docs and not without_docs


def test_trace_has_required_structured_fields(agent):
    r = _run(agent, "calm study music", top_k=3)
    for field in ["request_id", "timestamp", "intent", "plan", "components_called",
                  "retrieved_evidence_ids", "recommendations", "guardrail_decisions",
                  "verifier_checks", "confidence", "status"]:
        assert field in r.trace, f"missing trace field {field}"


def test_trace_has_no_hidden_chain_of_thought(agent):
    r = _run(agent, "calm study music", top_k=3)
    assert FORBIDDEN_TRACE_FIELDS.isdisjoint(r.trace.keys())


def test_confidence_is_bounded(agent):
    r = _run(agent, "upbeat workout songs", top_k=3)
    assert 0.0 <= r.confidence <= 1.0


def test_verifier_passes_on_normal_queries(agent):
    for q in ["calm study music", "high energy workout", "relaxing evening songs"]:
        r = _run(agent, q, top_k=3)
        assert r.verifier["passed"], f"verifier failed for {q}"
