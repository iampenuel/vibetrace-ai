"""Optional Streamlit interface for the Music Recommender Simulation.

Run with:

    streamlit run src/app.py

This UI is a thin front-end: it calls exactly the same recommender logic as
the CLI (``load_songs`` / ``recommend_songs``) rather than duplicating any
scoring math.
"""

import pandas as pd
import streamlit as st

try:  # Support running from the repo root.
    from src.recommender import available_modes, load_songs, recommend_songs
    from src.profiles import DISPLAY_NAMES, get_profile, profile_slugs
except ImportError:  # pragma: no cover
    from recommender import available_modes, load_songs, recommend_songs
    from profiles import DISPLAY_NAMES, get_profile, profile_slugs

DATA_PATH = "data/songs.csv"


@st.cache_data
def _load(path: str):
    return load_songs(path)


def main() -> None:
    st.set_page_config(page_title="VibeScope Recommender", page_icon="🎧")
    st.title("🎧 VibeScope Recommender")
    st.caption(
        "A small, fully explainable content-based music recommender. "
        "This UI reuses the same logic as `python -m src.main`."
    )

    songs = _load(DATA_PATH)

    with st.sidebar:
        st.header("Controls")
        slug = st.selectbox(
            "Profile",
            profile_slugs(),
            format_func=lambda s: DISPLAY_NAMES.get(s, s),
        )
        mode = st.selectbox("Ranking mode", available_modes())
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)
        diversify = st.checkbox("Diversity reranking", value=True)

    profile = get_profile(slug)

    st.subheader("Profile preferences")
    prefs = profile.to_prefs()
    st.json(prefs)

    results = recommend_songs(
        prefs, songs, k=top_k, mode=mode, diversify=diversify
    )

    st.subheader(f"Recommendations — {DISPLAY_NAMES[slug]} ({mode})")
    table = pd.DataFrame(
        [
            {
                "#": i,
                "Title": song["title"],
                "Artist": song["artist"],
                "Genre": song["genre"],
                "Mood": song["mood"],
                "Score": round(score, 2),
            }
            for i, (song, score, _why) in enumerate(results, start=1)
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.caption(
        "Diversity reranking is "
        + ("**ON** (artist-repeat and genre-concentration penalties applied)."
           if diversify
           else "**OFF** (pure score order).")
    )

    st.subheader("Why these songs?")
    for i, (song, score, why) in enumerate(results, start=1):
        with st.expander(f"{i}. {song['title']} — {score:.2f}"):
            for fragment in why.split(";"):
                st.write("•", fragment.strip())


if __name__ == "__main__":
    main()
