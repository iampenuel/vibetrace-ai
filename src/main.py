"""Command-line runner for the Music Recommender Simulation.

Examples
--------
    python -m src.main --profile high-energy-pop --mode balanced --top-k 5
    python -m src.main --profile acoustic-chill --mode mood_first --top-k 5
    python -m src.main --all-profiles --mode balanced --top-k 5
    python -m src.main --compare-modes --profile high-energy-pop --top-k 5
    python -m src.main --profile edm-workout --mode energy_focused --no-diversity

The CLI is non-interactive: running ``python -m src.main`` with no arguments
prints a useful default recommendation for the High-Energy Pop profile.
"""

import argparse
from typing import Dict, List, Tuple

from tabulate import tabulate

try:  # Support both `python -m src.main` and `python src/main.py`.
    from src.recommender import (
        UserProfile,
        available_modes,
        load_songs,
        recommend_songs,
    )
    from src.profiles import (
        DISPLAY_NAMES,
        get_profile,
        profile_slugs,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from recommender import (
        UserProfile,
        available_modes,
        load_songs,
        recommend_songs,
    )
    from profiles import DISPLAY_NAMES, get_profile, profile_slugs

DEFAULT_DATA_PATH = "data/songs.csv"


def concise_reasons(explanation: str, limit: int = 3) -> str:
    """Pick the most informative reason fragments for a compact table cell.

    We prefer contributions that actually moved the score (drop the
    ``+0.00`` mismatch notes) and always keep any penalty lines.
    """
    fragments = [frag.strip() for frag in explanation.split(";") if frag.strip()]
    kept: List[str] = []
    for frag in fragments:
        is_penalty = "-" in frag and frag.rstrip(")").endswith(")")
        is_zero = "+0.00" in frag
        if is_zero and not is_penalty:
            continue
        kept.append(frag)
    chosen = kept[:limit]
    if not chosen:  # Fall back to whatever we have.
        chosen = fragments[:limit]
    return "\n".join(chosen)


def summarise_profile(profile: UserProfile) -> str:
    """Return a one-line summary of a profile's active preferences."""
    prefs = profile.to_prefs()
    parts = []
    if prefs.get("genres"):
        parts.append(f"genres={'/'.join(prefs['genres'])}")
    if prefs.get("moods"):
        parts.append(f"moods={'/'.join(prefs['moods'])}")
    for key in ("energy", "tempo", "acousticness", "danceability", "popularity"):
        if prefs.get(key) is not None:
            parts.append(f"{key}={prefs[key]}")
    if prefs.get("allow_explicit") is not None:
        parts.append(f"explicit_ok={prefs['allow_explicit']}")
    return ", ".join(parts)


def render_table(
    results: List[Tuple[Dict, float, str]], diversify: bool
) -> str:
    """Render recommendations as a formatted table."""
    rows = []
    for rank, (song, score, explanation) in enumerate(results, start=1):
        rows.append(
            [
                rank,
                song["title"],
                song["artist"],
                song["genre"],
                song["mood"],
                f"{score:.2f}",
                concise_reasons(explanation),
            ]
        )
    headers = ["#", "Title", "Artist", "Genre", "Mood", "Score", "Why"]
    table = tabulate(rows, headers=headers, tablefmt="grid")
    legend = (
        "Diversity reranking: ENABLED "
        "(artist-repeat and genre-concentration penalties applied)"
        if diversify
        else "Diversity reranking: DISABLED (pure score order)"
    )
    return f"{table}\n{legend}"


def run_single(
    slug: str, songs: List[Dict], mode: str, top_k: int, diversify: bool
) -> None:
    profile = get_profile(slug)
    results = recommend_songs(
        profile.to_prefs(), songs, k=top_k, mode=mode, diversify=diversify
    )
    print(f"\n=== {DISPLAY_NAMES[slug]}  |  mode={mode}  |  top-{top_k} ===")
    print(f"Preferences: {summarise_profile(profile)}")
    print(render_table(results, diversify))


def run_all_profiles(
    songs: List[Dict], mode: str, top_k: int, diversify: bool
) -> None:
    for slug in profile_slugs():
        run_single(slug, songs, mode, top_k, diversify)


def run_compare_modes(
    slug: str, songs: List[Dict], top_k: int, diversify: bool
) -> None:
    profile = get_profile(slug)
    print(f"\n=== Mode comparison for {DISPLAY_NAMES[slug]} (top-{top_k}) ===")
    print(f"Preferences: {summarise_profile(profile)}")
    for mode in available_modes():
        results = recommend_songs(
            profile.to_prefs(), songs, k=top_k, mode=mode, diversify=diversify
        )
        ranking = " > ".join(f"{s['title']} ({sc:.2f})" for s, sc, _ in results)
        print(f"\n[{mode}]")
        print(f"  {ranking}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Music Recommender Simulation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        default="high-energy-pop",
        choices=profile_slugs(),
        help="Named user taste profile to recommend for.",
    )
    parser.add_argument(
        "--mode",
        default="balanced",
        choices=available_modes(),
        help="Ranking strategy (scoring weights).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of songs to return.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to songs CSV.")
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Show recommendations for every named profile.",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Compare all ranking modes for the selected profile.",
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Disable diversity reranking (use pure score order).",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    diversify = not args.no_diversity

    songs = load_songs(args.data)
    print(f"Loaded {len(songs)} songs from {args.data}")

    if args.compare_modes:
        run_compare_modes(args.profile, songs, args.top_k, diversify)
    elif args.all_profiles:
        run_all_profiles(songs, args.mode, args.top_k, diversify)
    else:
        run_single(args.profile, songs, args.mode, args.top_k, diversify)


if __name__ == "__main__":
    main()
