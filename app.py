"""VibeTrace AI — Streamlit interface.

A thin front-end over :class:`src.agent.VibeTraceAgent`. It duplicates NO
recommendation, retrieval, or verification logic — it calls the same ``run``
method the CLI uses and renders the structured result.

Run with:
    python -m streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agent import VibeTraceAgent

PROFILE_LABELS = {
    None: "None (no listening history)",
    "night_owl_coder": "Night Owl Coder",
    "gym_enthusiast": "Gym Enthusiast",
    "weekend_relaxer": "Weekend Relaxer",
}
MODES = ["balanced", "genre_first", "mood_first", "energy_focused"]


@st.cache_resource
def get_agent() -> VibeTraceAgent:
    """Build the agent once and cache it across reruns."""
    return VibeTraceAgent()


def main() -> None:
    st.set_page_config(page_title="VibeTrace AI", page_icon="🎧", layout="wide")
    st.title("🎧 VibeTrace AI")
    st.caption(
        "An explainable, retrieval-grounded music discovery copilot. It classifies "
        "your intent, retrieves catalog + knowledge evidence, ranks diverse songs, "
        "and verifies that its explanations are grounded — fully offline, no API key."
    )

    agent = get_agent()

    with st.sidebar:
        st.header("Controls")
        profile = st.selectbox(
            "Sample listening profile",
            options=list(PROFILE_LABELS.keys()),
            format_func=lambda k: PROFILE_LABELS[k],
        )
        mode = st.selectbox("Ranking mode", MODES)
        top_k = st.slider("Number of songs (top-k)", 1, 8, 4)
        use_retrieval = st.checkbox("Use knowledge retrieval (RAG)", value=True)
        use_diversity = st.checkbox("Diversity reranking", value=True)
        st.markdown("---")
        with st.expander("System limitations"):
            st.write(
                "- Catalog is small and synthetic.\n"
                "- Retrieved documents are educational context written for this project.\n"
                "- Recommendation quality is subjective; confidence is a heuristic.\n"
                "- This is not Spotify and makes no health claims."
            )

    query = st.text_area(
        "What are you in the mood for?",
        value="Give me calm, low-energy music for late-night studying.",
        height=90,
    )
    col_run, col_clear = st.columns([1, 1])
    run = col_run.button("Run VibeTrace", type="primary")
    if col_clear.button("Clear / Reset"):
        st.rerun()

    if not run:
        st.info("Enter a request and press **Run VibeTrace**.")
        return

    result = agent.run(
        query=query, top_k=top_k, mode=mode, profile=profile,
        use_retrieval=use_retrieval, use_diversity=use_diversity,
        log_path="logs/agent_trace.jsonl",
    )

    # Status / intent / confidence row.
    c1, c2, c3 = st.columns(3)
    c1.metric("Intent", result.intent, f"conf {result.intent_confidence:.2f}")
    c2.metric("System confidence", f"{result.confidence:.2f}")
    verdict = "PASSED" if result.verifier.get("passed") else "FLAGGED"
    c3.metric("Verifier", verdict, f"pass rate {result.verifier.get('pass_rate', 0):.2f}")

    if result.status == "guardrail":
        st.warning(result.response_text)
    else:
        st.success(f"Interpreted intent: **{result.intent}**")

    # Recommendations table.
    if result.recommendations:
        st.subheader("Recommendations")
        df = pd.DataFrame([
            {
                "#": i,
                "Title": r.title,
                "Artist": r.artist,
                "Genre": r.genre,
                "Mood": r.mood,
                "Score": round(r.score, 2),
                "Evidence": " ".join(f"[{e}]" for e in r.evidence_ids),
            }
            for i, r in enumerate(result.recommendations, start=1)
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)

    # Grounded explanation.
    st.subheader("Grounded explanation")
    st.text(result.response_text)

    # Evidence section.
    if result.evidence:
        st.subheader("Retrieved evidence")
        ev = pd.DataFrame([
            {
                "Evidence ID": e.source_id,
                "Type": e.source_type,
                "Similarity": round(e.score, 3),
                "Detail": e.metadata.get("heading") or e.metadata.get("title")
                          or e.metadata.get("display_name") or "",
            }
            for e in result.evidence
        ])
        st.dataframe(ev, hide_index=True, use_container_width=True)

    # Warnings.
    for w in result.warnings:
        st.warning(w)

    # High-level trace.
    with st.expander("High-Level Decision Trace (no hidden chain-of-thought)"):
        st.json(result.trace)


if __name__ == "__main__":
    main()
