"""Tests for input guardrails."""

from src import guardrails


def test_empty_input_blocked():
    d = guardrails.validate_query("")
    assert not d.ok and d.code == "empty_input"


def test_whitespace_only_blocked():
    d = guardrails.validate_query("    \n\t  ")
    assert not d.ok and d.code == "empty_input"


def test_none_input_blocked():
    d = guardrails.validate_query(None)
    assert not d.ok


def test_normal_input_accepted():
    d = guardrails.validate_query("calm study music")
    assert d.ok and d.code == "input_ok"
    assert d.details["sanitized"] == "calm study music"


def test_long_input_truncated():
    long = "songs " * 200
    d = guardrails.validate_query(long)
    assert d.ok and d.code == "input_truncated"
    assert len(d.details["sanitized"]) == guardrails.MAX_QUERY_LENGTH


def test_out_of_domain_detection():
    assert guardrails.looks_out_of_domain("diagnose my anxiety and depression")
    assert guardrails.looks_out_of_domain("what stocks should i buy")
    assert not guardrails.looks_out_of_domain("calm songs for studying")


def test_invalid_top_k():
    assert not guardrails.enforce_topk(0).ok
    assert not guardrails.enforce_topk(-2).ok
    assert guardrails.enforce_topk(3).ok
    assert guardrails.enforce_topk(3).details["k"] == 3


def test_invalid_top_k_non_integer():
    assert not guardrails.enforce_topk("abc").ok


def test_explicit_preference_detection():
    assert guardrails.detect_explicit_preference("clean songs with no explicit lyrics") is False
    assert guardrails.detect_explicit_preference("family-friendly workout music") is False
    assert guardrails.detect_explicit_preference("explicit is fine") is True
    assert guardrails.detect_explicit_preference("some upbeat pop songs") is None


def test_low_confidence_flagged():
    low = guardrails.check_low_confidence(0.2)
    assert low.code == "low_confidence"
    high = guardrails.check_low_confidence(0.9)
    assert high.code == "confidence_ok"


def test_summarize_input_is_bounded():
    s = guardrails.summarize_input("word " * 100, limit=50)
    assert len(s) <= 51
