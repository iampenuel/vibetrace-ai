"""Shared data models for the VibeTrace AI applied-AI pipeline.

These small dataclasses are the "nouns" that flow between the components
(retriever, recommender, composer, verifier, agent). They are deliberately
plain and beginner-readable. The Project 3 ``Song`` and ``UserProfile`` models
live in :mod:`src.recommender` and are re-exported here for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Re-export the base-project models so callers can import everything from one
# place without duplicating the recommendation data schema.
from src.recommender import Song, UserProfile  # noqa: F401


@dataclass
class Evidence:
    """One retrieved piece of grounding evidence.

    ``source_type`` is one of ``"song"``, ``"doc"``, or ``"history"``.
    ``source_id`` is the inner identifier without brackets, e.g. ``"song:12"``,
    ``"doc:mood_and_energy_guide.md#energy"``, or ``"history:night_owl_coder"``.
    """

    source_type: str
    source_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ref(self) -> str:
        """Return the bracketed citation form, e.g. ``[song:12]``."""
        return f"[{self.source_id}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "score": round(float(self.score), 4),
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class Recommendation:
    """A single ranked recommendation with its truthful reasoning."""

    song_id: int
    title: str
    artist: str
    genre: str
    mood: str
    score: float
    reasons: str
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "song_id": self.song_id,
            "title": self.title,
            "artist": self.artist,
            "genre": self.genre,
            "mood": self.mood,
            "score": round(float(self.score), 4),
            "reasons": self.reasons,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass
class AgentResult:
    """The structured result returned by :meth:`src.agent.VibeTraceAgent.run`."""

    query: str
    status: str  # "ok" | "guardrail" | "error"
    intent: str
    intent_confidence: float
    plan: List[str]
    response_text: str
    recommendations: List[Recommendation] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)
    verifier: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "status": self.status,
            "intent": self.intent,
            "intent_confidence": round(float(self.intent_confidence), 4),
            "plan": list(self.plan),
            "response_text": self.response_text,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": round(float(self.confidence), 4),
            "warnings": list(self.warnings),
            "verifier": self.verifier,
            "trace": self.trace,
        }
