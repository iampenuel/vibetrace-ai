"""Input guardrails and safety checks for VibeTrace AI.

Guardrails run at the edges of the pipeline. They fail gracefully, return a
useful message, and produce a structured :class:`GuardrailDecision` that the
agent records in its trace. They never raise stack traces at normal users.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Longest query we will process. Anything past this is truncated and flagged.
MAX_QUERY_LENGTH = 500
# Below this intent-classifier confidence we treat the request as ambiguous.
LOW_CONFIDENCE_THRESHOLD = 0.45

# Phrases that signal the user wants clean (non-explicit) music.
_CLEAN_PATTERNS = [
    r"\bclean\b",
    r"\bno explicit\b",
    r"\bnon[- ]?explicit\b",
    r"\bfamily[- ]?friendly\b",
    r"\bno swearing\b",
    r"\bno curs\w*\b",
    r"\bno profanity\b",
    r"\bkid[- ]?friendly\b",
]
# Phrases that signal explicit content is acceptable.
_ALLOW_EXPLICIT_PATTERNS = [
    r"\bexplicit (is )?(ok|fine|allowed)\b",
    r"\ballow explicit\b",
    r"\bexplicit ok\b",
]

# Obvious out-of-domain signals used as a cheap pre-filter alongside the
# intent classifier's ``out_of_scope`` class. Kept deliberately narrow so we do
# not accidentally reject real music requests.
_OUT_OF_DOMAIN_PATTERNS = [
    r"\b(diagnos\w+|depress\w+|anxiet\w+|therapy|medication|medicine|cure)\b",
    r"\b(stocks?|invest\w*|crypto|bitcoin)\b",
    r"\b(weather|forecast)\b",
    r"\b(recipe|lasagna|cook\w*)\b",
    r"\b(flight|book (me )?a)\b",
]


@dataclass
class GuardrailDecision:
    """One guardrail outcome. ``ok`` False means the pipeline should stop or
    adjust. ``code`` is a short machine-readable label used in traces."""

    ok: bool
    code: str
    message: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "ok": self.ok,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def validate_query(query: Optional[str]) -> GuardrailDecision:
    """Validate raw user input for emptiness and length.

    Returns a blocking decision for empty/whitespace input, a non-blocking
    (ok=True) decision that truncates over-long input, or a clean pass.
    """
    if query is None or not str(query).strip():
        return GuardrailDecision(
            ok=False,
            code="empty_input",
            message=(
                "Your request looks empty. Try something like: "
                "'calm low-energy songs for late-night studying'."
            ),
        )

    text = str(query).strip()
    if len(text) > MAX_QUERY_LENGTH:
        truncated = text[:MAX_QUERY_LENGTH]
        return GuardrailDecision(
            ok=True,
            code="input_truncated",
            message=(
                f"Your request was longer than {MAX_QUERY_LENGTH} characters and "
                "was truncated before processing."
            ),
            details={"original_length": len(text), "sanitized": truncated},
        )

    return GuardrailDecision(ok=True, code="input_ok", message="Input accepted.",
                             details={"sanitized": text})


def looks_out_of_domain(query: str) -> bool:
    """Cheap keyword pre-filter for clearly non-music requests."""
    low = query.lower()
    return any(re.search(p, low) for p in _OUT_OF_DOMAIN_PATTERNS)


def detect_explicit_preference(query: str) -> Optional[bool]:
    """Infer an explicit-content preference from the query text.

    Returns ``False`` (avoid explicit) if the user asked for clean music,
    ``True`` if they explicitly allowed it, or ``None`` if unstated.
    """
    low = query.lower()
    if any(re.search(p, low) for p in _ALLOW_EXPLICIT_PATTERNS):
        return True
    if any(re.search(p, low) for p in _CLEAN_PATTERNS):
        return False
    return None


def enforce_topk(k: int) -> GuardrailDecision:
    """Validate the requested number of recommendations."""
    try:
        k_int = int(k)
    except (TypeError, ValueError):
        return GuardrailDecision(
            ok=False, code="invalid_top_k",
            message=f"top-k must be an integer, got {k!r}.",
        )
    if k_int < 1:
        return GuardrailDecision(
            ok=False, code="invalid_top_k",
            message=f"top-k must be at least 1, got {k_int}.",
        )
    return GuardrailDecision(ok=True, code="top_k_ok", message="top-k accepted.",
                             details={"k": k_int})


def check_low_confidence(
    confidence: float, threshold: float = LOW_CONFIDENCE_THRESHOLD
) -> GuardrailDecision:
    """Flag (but do not block) low-confidence intent classification."""
    if confidence < threshold:
        return GuardrailDecision(
            ok=True,
            code="low_confidence",
            message=(
                "I wasn't fully sure what you meant, so I used a safe, balanced "
                "recommendation approach. Feel free to rephrase for a sharper match."
            ),
            details={"confidence": round(float(confidence), 4), "threshold": threshold},
        )
    return GuardrailDecision(ok=True, code="confidence_ok", message="Confidence acceptable.")


def out_of_scope_message() -> str:
    """User-facing message for out-of-domain requests."""
    return (
        "VibeTrace AI only helps with music discovery — finding, comparing, and "
        "explaining songs for moods and activities. I can't help with that request, "
        "but I can suggest music for studying, working out, relaxing, and more."
    )


def summarize_input(query: str, limit: int = 120) -> str:
    """Return a short, log-safe summary of the input (no raw dumps)."""
    collapsed = re.sub(r"\s+", " ", str(query)).strip()
    if len(collapsed) > limit:
        return collapsed[:limit] + "…"
    return collapsed


def combine(decisions: List[GuardrailDecision]) -> List[Dict]:
    """Serialize a list of guardrail decisions for the trace."""
    return [d.to_dict() for d in decisions]
