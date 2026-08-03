"""VibeTrace AI — reliability evaluation harness.

Runs every predefined case in data/evaluation_cases.json through the real agent
and reports parseable metrics: intent accuracy, retrieval evidence hit rate,
end-to-end pass rate, guardrail pass rate, explanation grounding pass rate,
average heuristic confidence, and error count.

Critical thresholds (documented):
  * All safety / guardrail cases must pass.
  * End-to-end pass rate must be >= 80%.
  * Grounding pass rate for successful answers must be 100%.

Exit code is non-zero only if a critical threshold fails.

Run:
    python scripts/evaluate_system.py
    python scripts/evaluate_system.py | tee outputs/evaluation_summary.txt
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.agent import VibeTraceAgent  # noqa: E402

CASES_PATH = "data/evaluation_cases.json"
RESULTS_JSON = "outputs/evaluation_results.json"

E2E_THRESHOLD = 0.80


def _op(value: float, op: str, target: float) -> bool:
    if op == "lte":
        return value <= target
    if op == "gte":
        return value >= target
    if op == "eq":
        return value == target
    return False


def evaluate_case(agent: VibeTraceAgent, case: dict) -> dict:
    """Run one case and return a dict of per-case pass/fail signals."""
    top_k = case.get("top_k", 3)
    result = agent.run(case["query"], top_k=top_k, log_path=None)

    reasons = []
    checks = {}

    expected_guardrail = case.get("expected_guardrail")
    if expected_guardrail:
        # Guardrail cases: expect status 'guardrail' (and out_of_scope intent).
        checks["guardrail_triggered"] = result.status == "guardrail"
        if not checks["guardrail_triggered"]:
            reasons.append(f"expected guardrail '{expected_guardrail}', got status {result.status}")
        if expected_guardrail == "out_of_scope":
            checks["intent_out_of_scope"] = result.intent == "out_of_scope"
            if not checks["intent_out_of_scope"]:
                reasons.append(f"expected out_of_scope intent, got {result.intent}")
    else:
        checks["status_ok"] = result.status == "ok"
        if not checks["status_ok"]:
            reasons.append(f"expected ok status, got {result.status}")

        if case.get("expected_intent"):
            checks["intent_match"] = result.intent == case["expected_intent"]
            if not checks["intent_match"]:
                reasons.append(f"intent {result.intent} != {case['expected_intent']}")

        if "min_recommendations" in case:
            n = len(result.recommendations)
            checks["min_recs"] = n >= case["min_recommendations"]
            if not checks["min_recs"]:
                reasons.append(f"{n} recs < {case['min_recommendations']}")

        if case.get("required_evidence_source"):
            src = case["required_evidence_source"]
            got = any(e.source_type == src for e in result.evidence)
            checks["evidence_source"] = got
            if not got:
                reasons.append(f"no '{src}' evidence retrieved")

        if case.get("forbid_explicit"):
            explicit = [r.song_id for r in result.recommendations
                        if agent._by_id.get(r.song_id, {}).get("explicit")]
            checks["no_explicit"] = len(explicit) == 0
            if explicit:
                reasons.append(f"explicit songs present: {explicit}")

        if case.get("distinct_artists"):
            artists = [r.artist for r in result.recommendations]
            checks["distinct_artists"] = len(artists) == len(set(artists))
            if not checks["distinct_artists"]:
                reasons.append("duplicate artists in recommendations")

        if case.get("top_feature") and result.recommendations:
            spec = case["top_feature"]
            top = agent._by_id.get(result.recommendations[0].song_id, {})
            val = float(top.get(spec["feature"], 0.0))
            ok = _op(val, spec["op"], spec["value"])
            checks["top_feature"] = ok
            if not ok:
                reasons.append(
                    f"top {spec['feature']}={val} fails {spec['op']} {spec['value']}"
                )

    # Grounding signal (only meaningful for successful recommendation answers).
    grounded = None
    if result.status == "ok" and result.recommendations:
        grounded = result.verifier.get("checks", {}).get("recommendations_grounded", False)

    passed = all(checks.values())

    return {
        "id": case["id"],
        "query": case["query"],
        "predicted_intent": result.intent,
        "expected_intent": case.get("expected_intent"),
        "status": result.status,
        "confidence": result.confidence,
        "verifier_passed": result.verifier.get("passed"),
        "grounded": grounded,
        "checks": checks,
        "passed": passed,
        "critical": bool(case.get("critical")),
        "reasons": reasons,
    }


def main() -> int:
    with open(CASES_PATH, "r", encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]

    agent = VibeTraceAgent()
    results = [evaluate_case(agent, c) for c in cases]

    # Aggregate metrics.
    intent_cases = [r for r in results if r["expected_intent"]]
    intent_correct = sum(1 for r in intent_cases if r["predicted_intent"] == r["expected_intent"])
    intent_acc = intent_correct / len(intent_cases) if intent_cases else 0.0

    retr_cases = [r for r in results if "evidence_source" in r["checks"]]
    retr_hits = sum(1 for r in retr_cases if r["checks"]["evidence_source"])
    retr_rate = retr_hits / len(retr_cases) if retr_cases else 1.0

    guardrail_cases = [r for r in results if r["critical"]]
    guardrail_pass = sum(1 for r in guardrail_cases if r["passed"])
    guardrail_rate = guardrail_pass / len(guardrail_cases) if guardrail_cases else 1.0

    grounded_cases = [r for r in results if r["grounded"] is not None]
    grounded_pass = sum(1 for r in grounded_cases if r["grounded"])
    grounded_rate = grounded_pass / len(grounded_cases) if grounded_cases else 1.0

    ok_cases = [r for r in results if r["status"] == "ok"]
    avg_conf = sum(r["confidence"] for r in ok_cases) / len(ok_cases) if ok_cases else 0.0

    e2e_pass = sum(1 for r in results if r["passed"])
    e2e_rate = e2e_pass / len(results)

    errors = sum(1 for r in results if r["status"] == "error")

    # Critical checks.
    critical_ok = all(r["passed"] for r in guardrail_cases)
    e2e_ok = e2e_rate >= E2E_THRESHOLD
    grounding_ok = grounded_rate >= 1.0
    overall_ok = critical_ok and e2e_ok and grounding_ok

    # -- report ----------------------------------------------------------
    print("=" * 66)
    print("VibeTrace AI — Reliability Evaluation Summary")
    print("=" * 66)
    print(f"Cases run: {len(results)}")
    print("")
    print(f"Intent accuracy            : {intent_acc * 100:6.2f}%  ({intent_correct}/{len(intent_cases)})")
    print(f"Retrieval evidence hit rate: {retr_rate * 100:6.2f}%  ({retr_hits}/{len(retr_cases)})")
    print(f"End-to-end pass rate       : {e2e_rate * 100:6.2f}%  ({e2e_pass}/{len(results)})")
    print(f"Guardrail pass rate        : {guardrail_rate * 100:6.2f}%  ({guardrail_pass}/{len(guardrail_cases)})")
    print(f"Grounding pass rate        : {grounded_rate * 100:6.2f}%  ({grounded_pass}/{len(grounded_cases)})")
    print(f"Average heuristic confidence: {avg_conf:5.2f}")
    print(f"Errors                     : {errors}")
    print("")
    print("Per-case results")
    print("-" * 66)
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        crit = " [critical]" if r["critical"] else ""
        print(f"  [{mark}] {r['id']:<28} intent={r['predicted_intent']:<12} conf={r['confidence']:.2f}{crit}")
        for reason in r["reasons"]:
            print(f"         - {reason}")
    print("")
    print("Critical thresholds")
    print("-" * 66)
    print(f"  All safety/guardrail cases pass : {'YES' if critical_ok else 'NO'}")
    print(f"  End-to-end >= {int(E2E_THRESHOLD*100)}%             : {'YES' if e2e_ok else 'NO'}")
    print(f"  Grounding == 100%               : {'YES' if grounding_ok else 'NO'}")
    print("")
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 66)

    # Machine-readable results.
    os.makedirs("outputs", exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump({
            "metrics": {
                "intent_accuracy": intent_acc,
                "retrieval_hit_rate": retr_rate,
                "end_to_end_pass_rate": e2e_rate,
                "guardrail_pass_rate": guardrail_rate,
                "grounding_pass_rate": grounded_rate,
                "average_confidence": avg_conf,
                "errors": errors,
                "overall_pass": overall_ok,
            },
            "cases": results,
        }, handle, indent=2)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
