"""Grounded response composer for VibeTrace AI.

Deterministic, template-based composition. The composer never invents song
attributes: every reason string it prints comes from the recommender's scoring
output, and every evidence ID comes from the retriever. Its job is to arrange
that grounded material into a readable, transparent answer.
"""

from __future__ import annotations

from typing import List, Optional

from src.models import Evidence, Recommendation

_INTENT_INTERPRETATION = {
    "discover": "a music discovery request",
    "study": "a calm, focus-friendly study request",
    "workout": "a high-energy workout request",
    "relax": "a relaxing, low-key listening request",
    "compare": "a request to compare two songs",
    "explain": "a request to explain a recommendation",
    "diversify": "a request for a varied, non-repetitive selection",
    "out_of_scope": "a request outside music discovery",
}

_LIMITATION_NOTE = "This catalog is synthetic and small; scores reflect transparent feature matching, not a promise of enjoyment."


def interpret(intent: str, query: str) -> str:
    """One-sentence restatement of what the system understood."""
    label = _INTENT_INTERPRETATION.get(intent, "a music request")
    return f"I interpreted this as {label}."


def _confidence_line(confidence: float) -> str:
    return f"Confidence: {confidence:.2f} (heuristic, not a calibrated probability)"


def compose_recommendation_response(
    intent: str,
    query: str,
    recommendations: List[Recommendation],
    context_evidence: Optional[List[Evidence]] = None,
    confidence: float = 0.0,
    warnings: Optional[List[str]] = None,
    diversified: bool = True,
) -> str:
    """Compose a grounded answer for the recommendation-style intents."""
    warnings = warnings or []
    context_evidence = context_evidence or []
    lines: List[str] = [interpret(intent, query), ""]

    if not recommendations:
        lines.append("I couldn't find any songs that match this request well.")
        lines.append(_confidence_line(confidence))
        return "\n".join(lines)

    for rank, rec in enumerate(recommendations, start=1):
        lines.append(f"{rank}. {rec.title} — {rec.artist} (score {rec.score:.2f})")
        lines.append(f"   Why: {rec.reasons}")
        if rec.evidence_ids:
            lines.append("   Evidence: " + " ".join(f"[{eid}]" for eid in rec.evidence_ids))

    if context_evidence:
        lines.append("")
        cite = " ".join(e.ref() for e in context_evidence)
        lines.append(f"Context used: {cite}")

    lines.append("")
    if diversified:
        lines.append("Note: diversity reranking is on, so artists and genres are spread out.")
    lines.append(f"Note: {_LIMITATION_NOTE}")
    lines.append(_confidence_line(confidence))
    for w in warnings:
        lines.append(f"⚠ {w}")
    return "\n".join(lines)


def compose_comparison_response(
    query: str,
    first: Recommendation,
    second: Recommendation,
    context_evidence: Optional[List[Evidence]] = None,
    confidence: float = 0.0,
    warnings: Optional[List[str]] = None,
) -> str:
    """Compose a grounded side-by-side comparison of two songs."""
    warnings = warnings or []
    context_evidence = context_evidence or []
    lines: List[str] = [interpret("compare", query), ""]

    for label, rec in (("A", first), ("B", second)):
        lines.append(f"Song {label}: {rec.title} — {rec.artist} "
                     f"({rec.genre}, {rec.mood}, score {rec.score:.2f})")
        lines.append(f"   Why: {rec.reasons}")
        if rec.evidence_ids:
            lines.append("   Evidence: " + " ".join(f"[{eid}]" for eid in rec.evidence_ids))

    winner = first if first.score >= second.score else second
    lines.append("")
    lines.append(
        f"For this request, {winner.title} fits slightly better based on the "
        f"scored features above (score {winner.score:.2f})."
    )
    if context_evidence:
        cite = " ".join(e.ref() for e in context_evidence)
        lines.append(f"Feature definitions used: {cite}")

    lines.append("")
    lines.append(f"Note: {_LIMITATION_NOTE}")
    lines.append(_confidence_line(confidence))
    for w in warnings:
        lines.append(f"⚠ {w}")
    return "\n".join(lines)


def compose_explain_response(
    query: str,
    rec: Recommendation,
    context_evidence: Optional[List[Evidence]] = None,
    confidence: float = 0.0,
    warnings: Optional[List[str]] = None,
) -> str:
    """Compose a grounded explanation of a single song's ranking."""
    warnings = warnings or []
    context_evidence = context_evidence or []
    lines = [interpret("explain", query), ""]
    lines.append(f"{rec.title} — {rec.artist} ({rec.genre}, {rec.mood}) "
                 f"scored {rec.score:.2f}.")
    lines.append(f"Why: {rec.reasons}")
    if rec.evidence_ids:
        lines.append("Evidence: " + " ".join(f"[{eid}]" for eid in rec.evidence_ids))
    if context_evidence:
        cite = " ".join(e.ref() for e in context_evidence)
        lines.append(f"Feature definitions used: {cite}")
    lines.append("")
    lines.append(f"Note: {_LIMITATION_NOTE}")
    lines.append(_confidence_line(confidence))
    for w in warnings:
        lines.append(f"⚠ {w}")
    return "\n".join(lines)


def compose_out_of_scope_response(
    message: str, confidence: float = 0.0
) -> str:
    """Compose the safe out-of-domain response."""
    return f"{message}\n{_confidence_line(confidence)}"
