"""Structured, high-level trace logging for VibeTrace AI.

Traces are decision records, never hidden chain-of-thought. Each record captures
*what* the agent decided and *which* components ran — intent, plan, evidence IDs,
scores, guardrail and verifier outcomes, and final status — but never any private
step-by-step reasoning text.

Logs are written as JSON Lines (one JSON object per line) so they are easy to
append, diff, and parse in the evaluation harness.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

# Field names that must never appear in a trace record. Enforced at write time.
_FORBIDDEN_FIELDS = {
    "chain_of_thought", "cot", "reasoning", "hidden_reasoning",
    "scratchpad", "thoughts", "internal_monologue",
}


def make_request_id(query: str, counter: int) -> str:
    """Deterministic, log-safe request id from a query hash and a counter.

    Uses a content hash rather than randomness so the same demo run is
    reproducible. Not a security token — purely a trace correlation id.
    """
    digest = hashlib.sha1(f"{counter}:{query}".encode("utf-8")).hexdigest()[:8]
    return f"req_{counter:04d}_{digest}"


def utc_timestamp() -> str:
    """Current UTC timestamp in ISO-8601 form (for trace ordering only)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_trace(record: Dict[str, Any]) -> Dict[str, Any]:
    """Drop any forbidden hidden-reasoning fields before writing."""
    return {k: v for k, v in record.items() if k not in _FORBIDDEN_FIELDS}


def write_trace(record: Dict[str, Any], path: str = "logs/agent_trace.jsonl") -> None:
    """Append one sanitized trace record to ``path`` as a JSON line."""
    clean = sanitize_trace(record)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean, ensure_ascii=False) + "\n")


def read_traces(path: str) -> list:
    """Read all trace records from a JSONL file (empty list if missing)."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
