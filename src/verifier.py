"""Output verifier and heuristic confidence scoring for VibeTrace AI.

The verifier runs *after* composition and *before* the answer is returned. It
checks that the answer is grounded and internally consistent, and it produces a
heuristic confidence score. This is the safety net the agentic workflow relies
on: if grounding fails, the agent downgrades confidence and warns the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.models import Evidence, Recommendation

# Phrases that would be unsupported or unsafe for THIS system to assert.
# Kept specific enough that honest disclaimers (which the composer avoids anyway)
# do not trip them: we ban medical/therapeutic CLAIMS, not the general vocabulary.
_BANNED_CLAIM_PATTERNS = [
    r"\bcures?\b",
    r"\btreatment\b",
    r"\btreats\s+(your|my|the|a|an)\b",
    r"\bdiagnos\w+\b",
    r"\bheals?\s+(your|you|the)\b",
    r"\bguaranteed\b",
    r"\bclinical\w*\b",
    r"\bmedical\b",
    r"\bbetter than spotify\b",
]


@dataclass
class VerifierReport:
    """Structured result of verification."""

    checks: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    passed: bool = True
    pass_rate: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "checks": self.checks,
            "notes": self.notes,
        }


def verify_output(
    intent: str,
    status: str,
    response_text: str,
    recommendations: List[Recommendation],
    valid_evidence_ids: Set[str],
    catalog_ids: Set[int],
    requested_k: int,
    explicit_ok: Optional[bool],
    explicit_song_ids: Set[int],
    low_confidence: bool,
    warnings: List[str],
    context_evidence: Optional[List[Evidence]] = None,
) -> VerifierReport:
    """Run all applicable verification checks and return a report."""
    checks: Dict[str, bool] = {}
    notes: List[str] = []
    context_evidence = context_evidence or []

    # 1. Non-empty answer.
    checks["non_empty_answer"] = bool(response_text and response_text.strip())

    # 2. No unsupported / unsafe claims.
    low = response_text.lower()
    banned_hit = next((p for p in _BANNED_CLAIM_PATTERNS if re.search(p, low)), None)
    checks["no_unsupported_claims"] = banned_hit is None
    if banned_hit:
        notes.append(f"Response contained a disallowed claim pattern: {banned_hit}")

    # Guardrail / out-of-scope answers skip the recommendation-specific checks.
    if status == "guardrail" or intent == "out_of_scope":
        checks["safe_no_ranking"] = len(recommendations) == 0
        passed = all(checks.values())
        rate = sum(checks.values()) / len(checks)
        return VerifierReport(checks=checks, notes=notes, passed=passed, pass_rate=rate)

    # 3. Cited evidence IDs all exist.
    cited: Set[str] = set()
    for rec in recommendations:
        cited.update(rec.evidence_ids)
    for e in context_evidence:
        cited.add(e.source_id)
    unknown = [c for c in cited if c not in valid_evidence_ids]
    checks["evidence_ids_exist"] = len(unknown) == 0
    if unknown:
        notes.append(f"Unknown evidence IDs cited: {unknown}")

    # 4. Recommended songs exist in the catalog.
    missing = [r.song_id for r in recommendations if r.song_id not in catalog_ids]
    checks["songs_exist"] = len(missing) == 0
    if missing:
        notes.append(f"Recommended song IDs not in catalog: {missing}")

    # 5. Scores are in non-increasing order.
    scores = [r.score for r in recommendations]
    checks["scores_ordered"] = scores == sorted(scores, reverse=True)
    if not checks["scores_ordered"]:
        notes.append("Recommendation scores are not in descending order.")

    # 6. Every recommendation is grounded by at least one evidence ID.
    checks["recommendations_grounded"] = all(r.evidence_ids for r in recommendations)
    if not checks["recommendations_grounded"]:
        notes.append("At least one recommendation lacked evidence grounding.")

    # 7. Count satisfies the request where the catalog allows.
    checks["count_satisfied"] = len(recommendations) >= min(requested_k, len(catalog_ids)) \
        if intent not in ("compare", "explain") else len(recommendations) >= 1
    if not checks["count_satisfied"]:
        notes.append(f"Returned {len(recommendations)} recommendations, requested {requested_k}.")

    # 8. Explicit-content preference respected.
    if explicit_ok is False:
        checks["explicit_respected"] = len(explicit_song_ids) == 0
        if explicit_song_ids:
            notes.append(f"Explicit songs present despite clean preference: {sorted(explicit_song_ids)}")
    else:
        checks["explicit_respected"] = True

    # 9. Low confidence is acknowledged to the user.
    if low_confidence:
        ack = any("sure" in w.lower() or "confiden" in w.lower() for w in warnings)
        checks["low_confidence_acknowledged"] = ack
        if not ack:
            notes.append("Low confidence was not acknowledged in warnings.")
    else:
        checks["low_confidence_acknowledged"] = True

    passed = all(checks.values())
    rate = sum(1 for v in checks.values() if v) / len(checks)
    return VerifierReport(checks=checks, notes=notes, passed=passed, pass_rate=rate)


# ---------------------------------------------------------------------------
# Heuristic confidence
# ---------------------------------------------------------------------------

# Documented weights. Confidence is a transparent blend, NOT a calibrated
# probability of user satisfaction.
_W_INTENT = 0.30
_W_RETRIEVAL = 0.20
_W_SEPARATION = 0.20
_W_VERIFIER = 0.30


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_confidence(
    intent_confidence: float,
    retrieval_similarity: float,
    recommendations: List[Recommendation],
    verifier_pass_rate: float,
) -> float:
    """Blend four transparent factors into a [0,1] heuristic confidence.

    Factors: intent-classifier confidence, top retrieval similarity, score
    separation between the top two recommendations, and the verifier pass rate.
    """
    intent_c = _clamp01(intent_confidence)
    retr = _clamp01(retrieval_similarity)

    if len(recommendations) >= 2:
        top = recommendations[0].score
        second = recommendations[1].score
        separation = _clamp01((top - second) / (abs(top) + 1.0))
    elif len(recommendations) == 1:
        separation = 0.5
    else:
        separation = 0.0

    verifier = _clamp01(verifier_pass_rate)

    confidence = (
        _W_INTENT * intent_c
        + _W_RETRIEVAL * retr
        + _W_SEPARATION * separation
        + _W_VERIFIER * verifier
    )
    return round(_clamp01(confidence), 4)
