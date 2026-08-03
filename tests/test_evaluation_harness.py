"""Tests for the reliability evaluation harness."""

import json
import os

import pytest

from src.agent import VibeTraceAgent

import importlib.util

HARNESS_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "evaluate_system.py")
CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_cases.json")


def _load_harness():
    spec = importlib.util.spec_from_file_location("evaluate_system", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def agent():
    return VibeTraceAgent()


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


@pytest.fixture(scope="module")
def cases():
    with open(CASES_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def test_cases_file_has_enough_cases(cases):
    assert len(cases) >= 12


def test_evaluate_case_returns_parseable_dict(agent, harness, cases):
    case = next(c for c in cases if c["id"] == "study_low_energy")
    result = harness.evaluate_case(agent, case)
    for key in ["id", "predicted_intent", "status", "confidence", "passed", "checks"]:
        assert key in result
    assert isinstance(result["passed"], bool)


def test_guardrail_case_passes(agent, harness, cases):
    case = next(c for c in cases if c["id"] == "out_of_scope_weather")
    result = harness.evaluate_case(agent, case)
    assert result["passed"]
    assert result["status"] == "guardrail"


def test_empty_input_case_passes(agent, harness, cases):
    case = next(c for c in cases if c["id"] == "empty_input")
    result = harness.evaluate_case(agent, case)
    assert result["passed"]


def test_pass_rate_arithmetic(agent, harness, cases):
    results = [harness.evaluate_case(agent, c) for c in cases]
    passed = sum(1 for r in results if r["passed"])
    rate = passed / len(results)
    assert 0.0 <= rate <= 1.0
    # Our tuned system should clear the documented 80% bar.
    assert rate >= 0.8


def test_all_successful_answers_are_grounded(agent, harness, cases):
    results = [harness.evaluate_case(agent, c) for c in cases]
    grounded = [r for r in results if r["grounded"] is not None]
    assert all(r["grounded"] for r in grounded)
