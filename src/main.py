"""VibeTrace AI — command-line interface.

Runs the full agentic pipeline (guardrails -> intent -> plan -> retrieval ->
ranking -> composition -> verification) and prints a professional, readable
report. This CLI shares the exact same :class:`~src.agent.VibeTraceAgent` logic
as the Streamlit app.

Examples
--------
    python -m src.main --query "I need upbeat clean music for a workout"
    python -m src.main --query "calm low-energy songs for studying" --top-k 3 --show-trace
    python -m src.main --query "Compare Library Rain and Midnight Coding for studying"
    python -m src.main --query "Surprise me, but avoid repeating artists" --mode balanced
    python -m src.main --query ""
"""

from __future__ import annotations

import argparse
import json
from typing import List

from tabulate import tabulate

from src.agent import VibeTraceAgent
from src.models import AgentResult
from src.recommender import available_modes

# Sample profile names available via --profile (from data/sample_user_history.json).
PROFILE_CHOICES = ["night_owl_coder", "gym_enthusiast", "weekend_relaxer"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VibeTrace AI — explainable, retrieval-grounded music copilot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", default="", help="Your natural-language music request.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of songs to return.")
    parser.add_argument("--mode", default="balanced", choices=available_modes(),
                        help="Ranking strategy (scoring weights).")
    parser.add_argument("--profile", default=None, choices=PROFILE_CHOICES,
                        help="Optional sample listening profile for history grounding.")
    parser.add_argument("--show-trace", action="store_true",
                        help="Print the high-level decision trace.")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="Disable knowledge/context retrieval (ablation).")
    parser.add_argument("--no-diversity", action="store_true",
                        help="Disable diversity reranking.")
    parser.add_argument("--log-path", default="logs/agent_trace.jsonl",
                        help="Where to append the JSONL trace ('none' to disable).")
    parser.add_argument("--data", default="data/songs.csv", help="Path to songs CSV.")
    parser.add_argument("--json", action="store_true",
                        help="Print the full structured result as JSON instead.")
    return parser


def _render_recommendation_table(result: AgentResult) -> str:
    rows = []
    for rank, rec in enumerate(result.recommendations, start=1):
        rows.append([rank, rec.title, rec.artist, rec.genre, rec.mood,
                     f"{rec.score:.2f}", " ".join(f"[{e}]" for e in rec.evidence_ids)])
    headers = ["#", "Title", "Artist", "Genre", "Mood", "Score", "Evidence"]
    return tabulate(rows, headers=headers, tablefmt="github")


def print_report(result: AgentResult, show_trace: bool) -> None:
    print("=" * 74)
    print(f"QUERY: {result.query!r}")
    print(f"Intent: {result.intent}  (classifier confidence {result.intent_confidence:.2f})")
    print(f"Status: {result.status}   |   System confidence: {result.confidence:.2f}")
    print(f"Plan: {' -> '.join(result.plan)}")
    print("-" * 74)

    if result.recommendations:
        print(_render_recommendation_table(result))
        print()

    print(result.response_text)

    if result.evidence:
        print("-" * 74)
        print("Retrieved evidence:")
        for e in result.evidence:
            label = e.metadata.get("heading") or e.metadata.get("title") \
                or e.metadata.get("display_name") or ""
            print(f"  {e.ref():<48} sim={e.score:.3f}  {label}")

    v = result.verifier
    print("-" * 74)
    passed = "PASSED" if v.get("passed") else "FLAGGED"
    print(f"Verifier: {passed}  (pass rate {v.get('pass_rate', 0):.2f})")
    failed = [k for k, ok in v.get("checks", {}).items() if not ok]
    if failed:
        print(f"  Failed checks: {failed}")

    if result.warnings:
        print("-" * 74)
        for w in result.warnings:
            print(f"⚠ {w}")

    if show_trace:
        print("-" * 74)
        print("High-Level Decision Trace (no hidden chain-of-thought):")
        print(json.dumps(result.trace, indent=2))
    print("=" * 74)


def main(argv: List[str] = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = None if str(args.log_path).lower() == "none" else args.log_path

    agent = VibeTraceAgent(data_path=args.data)
    result = agent.run(
        query=args.query,
        top_k=args.top_k,
        mode=args.mode,
        profile=args.profile,
        use_retrieval=not args.no_retrieval,
        use_diversity=not args.no_diversity,
        log_path=log_path,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_report(result, show_trace=args.show_trace)

    # Non-zero exit only for hard internal errors, so scripting can detect them.
    return 0 if result.status in ("ok", "guardrail") else 1


if __name__ == "__main__":
    raise SystemExit(main())
