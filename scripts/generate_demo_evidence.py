"""Generate real execution evidence for VibeTrace AI.

Runs four demonstration queries plus a retrieval ablation and writes their
actual output to outputs/*.txt. Also appends the real high-level traces to
logs/agent_trace_examples.jsonl for inspection.

Run:
    python scripts/generate_demo_evidence.py
"""

from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.agent import VibeTraceAgent  # noqa: E402
from src.logging_utils import write_trace  # noqa: E402
from src.main import print_report  # noqa: E402

EXAMPLE_TRACE_PATH = "logs/agent_trace_examples.jsonl"


def capture_report(agent: VibeTraceAgent, query: str, top_k=3, show_trace=True,
                   **kwargs) -> tuple:
    """Run the agent and capture the CLI-style report as a string."""
    result = agent.run(query, top_k=top_k, log_path=None, **kwargs)
    buf = io.StringIO()
    stdout = sys.stdout
    try:
        sys.stdout = buf
        print_report(result, show_trace=show_trace)
    finally:
        sys.stdout = stdout
    return result, buf.getvalue()


def write_output(path: str, text: str) -> None:
    os.makedirs("outputs", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"Wrote {path}")


def main() -> int:
    agent = VibeTraceAgent()

    demos = [
        ("outputs/demo_discovery.txt",
         "I need upbeat, clean songs for a 30-minute workout.", 3, {}),
        ("outputs/demo_study.txt",
         "Give me calm, low-energy music for late-night studying and explain each choice.", 3, {}),
        ("outputs/demo_comparison.txt",
         "Compare two suitable songs for a reflective nighttime study session.", 2, {}),
        ("outputs/demo_guardrail.txt",
         "Can you diagnose my anxiety and recommend medication?", 3, {}),
    ]

    # Reset the example trace file so it only holds these representative runs.
    if os.path.exists(EXAMPLE_TRACE_PATH):
        os.remove(EXAMPLE_TRACE_PATH)

    for path, query, k, kwargs in demos:
        result, report = capture_report(agent, query, top_k=k, **kwargs)
        write_output(path, report)
        write_trace(result.trace, EXAMPLE_TRACE_PATH)

    # Also capture the empty-input guardrail as part of the guardrail demo.
    empty_result, empty_report = capture_report(agent, "", top_k=3)
    with open("outputs/demo_guardrail.txt", "a", encoding="utf-8") as handle:
        handle.write("\n\n### Additional guardrail: empty input ###\n\n")
        handle.write(empty_report)
    write_trace(empty_result.trace, EXAMPLE_TRACE_PATH)

    # -- RAG before/after comparison ------------------------------------
    query = "calm instrumental music for late-night studying"
    _, with_r = capture_report(agent, query, top_k=3, show_trace=False, use_retrieval=True)
    _, without_r = capture_report(agent, query, top_k=3, show_trace=False, use_retrieval=False)
    comparison = (
        "VibeTrace AI — Retrieval (RAG) Before/After Comparison\n"
        + "=" * 62 + "\n\n"
        f"Query: {query!r}\n\n"
        "This ablation shows how multi-source retrieval changes the system's\n"
        "behavior. WITHOUT retrieval, no knowledge/context evidence is cited and\n"
        "no retrieval-relevance bonus is applied to ranking. WITH retrieval, the\n"
        "answer cites [doc:...] context passages and blends a retrieval-relevance\n"
        "signal into the song scores, producing richer, evidence-grounded output.\n\n"
        + "#" * 62 + "\n# WITHOUT multi-source retrieval (--no-retrieval)\n" + "#" * 62 + "\n\n"
        + without_r
        + "\n\n" + "#" * 62 + "\n# WITH multi-source retrieval (default)\n" + "#" * 62 + "\n\n"
        + with_r
    )
    write_output("outputs/retrieval_comparison.txt", comparison)

    print(f"\nAppended {5} representative traces to {EXAMPLE_TRACE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
