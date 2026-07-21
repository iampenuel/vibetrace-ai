"""Named example user profiles for the Music Recommender Simulation.

Each profile is a :class:`~src.recommender.UserProfile` that differs from the
others across several preferences (not just genre), so switching profiles or
ranking modes produces visibly different recommendations.
"""

from typing import Dict, List

try:  # Support both `python -m src.main` and `python src/main.py`.
    from src.recommender import UserProfile
except ImportError:  # pragma: no cover - fallback for direct execution
    from recommender import UserProfile


# Ordered mapping of CLI slug -> (display name, profile).
PROFILES: Dict[str, "UserProfile"] = {
    "high-energy-pop": UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        preferred_moods=["energetic"],
        target_energy=0.90,
        target_tempo=124,
        target_valence=0.82,
        target_danceability=0.85,
        target_acousticness=0.12,
        target_popularity=82,
        preferred_decade=2020,
        preferred_language="English",
        allow_explicit=False,
    ),
    "acoustic-chill": UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        preferred_genres=["folk", "jazz", "ambient"],
        preferred_moods=["relaxed"],
        target_energy=0.33,
        target_tempo=76,
        target_valence=0.60,
        target_danceability=0.50,
        target_acousticness=0.88,
        target_instrumentalness=0.70,
        target_popularity=48,
        allow_explicit=False,
    ),
    "intense-rock": UserProfile(
        favorite_genre="rock",
        favorite_mood="intense",
        preferred_genres=["metal"],
        preferred_moods=["dark"],
        target_energy=0.92,
        target_tempo=150,
        target_valence=0.40,
        target_danceability=0.60,
        target_acousticness=0.10,
        target_popularity=62,
        allow_explicit=True,
    ),
    "edm-workout": UserProfile(
        favorite_genre="edm",
        favorite_mood="energetic",
        target_energy=0.95,
        target_tempo=132,
        target_valence=0.72,
        target_danceability=0.90,
        target_acousticness=0.05,
        target_liveness=0.25,
        target_popularity=75,
        preferred_language="English",
        allow_explicit=True,
    ),
}

# Human-friendly display names.
DISPLAY_NAMES: Dict[str, str] = {
    "high-energy-pop": "High-Energy Pop",
    "acoustic-chill": "Acoustic Chill",
    "intense-rock": "Intense Rock",
    "edm-workout": "EDM Workout",
}


def profile_slugs() -> List[str]:
    """Return the available profile slugs in display order."""
    return list(PROFILES.keys())


def get_profile(slug: str) -> "UserProfile":
    """Return the profile for ``slug`` or raise a clear error."""
    try:
        return PROFILES[slug]
    except KeyError:
        raise ValueError(
            f"Unknown profile: {slug!r}. Valid profiles: {profile_slugs()}"
        )
