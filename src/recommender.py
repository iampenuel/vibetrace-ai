"""
Music Recommender Simulation — core logic.

This module contains a small, fully explainable, content-based music
recommender. "Content-based" means we compare each song's attributes against
a user's taste profile and score how well they match. There is no machine
learning and no listening history — every number the system produces can be
traced back to a simple, deterministic formula.

The module exposes two matching APIs so it works with both the starter tests
and the CLI:

* A **functional** API built around plain dictionaries:
    - ``load_songs(csv_path)``
    - ``score_song(user_prefs, song, mode="balanced")``
    - ``recommend_songs(user_prefs, songs, k=5, mode="balanced", diversify=True)``

* An **object-oriented** API built around dataclasses:
    - ``Song`` and ``UserProfile`` dataclasses
    - ``Recommender`` class

Both paths share the exact same scoring math (see ``_score_song_core``) so a
recommendation never depends on which API you called.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

# Columns that must be present in every dataset row.
REQUIRED_COLUMNS: List[str] = [
    "id",
    "title",
    "artist",
    "genre",
    "mood",
    "energy",
    "tempo_bpm",
    "valence",
    "danceability",
    "acousticness",
    # Additional (stretch) attributes:
    "popularity",
    "release_decade",
    "instrumentalness",
    "speechiness",
    "liveness",
    "language",
    "explicit",
    "duration_seconds",
]

# Numeric features that must stay inside the 0.0 - 1.0 range.
UNIT_INTERVAL_FEATURES: List[str] = [
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
]

# Integer columns.
INT_COLUMNS = ["id", "tempo_bpm", "popularity", "release_decade", "duration_seconds"]
# Float columns.
FLOAT_COLUMNS = list(UNIT_INTERVAL_FEATURES)
# String columns.
STR_COLUMNS = ["title", "artist", "genre", "mood", "language"]
# Boolean columns.
BOOL_COLUMNS = ["explicit"]

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}
_FALSE_STRINGS = {"false", "0", "no", "n", "f", ""}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """A single song and its audio/metadata attributes.

    Every attribute maps directly to a column in ``data/songs.csv``.
    """

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Additional (stretch) attributes:
    popularity: int = 50
    release_decade: int = 2020
    instrumentalness: float = 0.0
    speechiness: float = 0.0
    liveness: float = 0.0
    language: str = "English"
    explicit: bool = False
    duration_seconds: int = 210

    @classmethod
    def from_dict(cls, data: Dict) -> "Song":
        """Build a ``Song`` from a (validated) dictionary row."""
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)

    def to_dict(self) -> Dict:
        """Return a plain-dictionary view of this song."""
        return asdict(self)


@dataclass
class UserProfile:
    """A user's taste profile.

    All fields are optional so a minimal profile still works. The original
    four starter fields (``favorite_genre``, ``favorite_mood``,
    ``target_energy``, ``likes_acoustic``) are preserved for backwards
    compatibility with the starter tests, and richer preferences can be
    layered on top.
    """

    favorite_genre: Optional[str] = None
    favorite_mood: Optional[str] = None
    target_energy: Optional[float] = None
    likes_acoustic: Optional[bool] = None

    # Richer, optional preferences (used when provided):
    preferred_genres: List[str] = field(default_factory=list)
    preferred_moods: List[str] = field(default_factory=list)
    target_tempo: Optional[float] = None
    target_valence: Optional[float] = None
    target_danceability: Optional[float] = None
    target_acousticness: Optional[float] = None
    target_popularity: Optional[float] = None
    preferred_decade: Optional[int] = None
    target_instrumentalness: Optional[float] = None
    target_speechiness: Optional[float] = None
    target_liveness: Optional[float] = None
    preferred_language: Optional[str] = None
    allow_explicit: Optional[bool] = None

    def to_prefs(self) -> Dict:
        """Translate this profile into the ``user_prefs`` dict the scoring
        functions understand.  Only preferences that were actually set are
        included, so unset preferences simply do not affect the score."""
        prefs: Dict = {}

        genres = list(self.preferred_genres)
        if self.favorite_genre:
            genres.append(self.favorite_genre)
        if genres:
            prefs["genres"] = genres

        moods = list(self.preferred_moods)
        if self.favorite_mood:
            moods.append(self.favorite_mood)
        if moods:
            prefs["moods"] = moods

        if self.target_energy is not None:
            prefs["energy"] = self.target_energy
        if self.target_tempo is not None:
            prefs["tempo"] = self.target_tempo
        if self.target_valence is not None:
            prefs["valence"] = self.target_valence
        if self.target_danceability is not None:
            prefs["danceability"] = self.target_danceability

        # Explicit target acousticness wins; otherwise derive from likes_acoustic.
        if self.target_acousticness is not None:
            prefs["acousticness"] = self.target_acousticness
        elif self.likes_acoustic is not None:
            prefs["acousticness"] = 0.8 if self.likes_acoustic else 0.2

        if self.target_popularity is not None:
            prefs["popularity"] = self.target_popularity
        if self.preferred_decade is not None:
            prefs["decade"] = self.preferred_decade
        if self.target_instrumentalness is not None:
            prefs["instrumentalness"] = self.target_instrumentalness
        if self.target_speechiness is not None:
            prefs["speechiness"] = self.target_speechiness
        if self.target_liveness is not None:
            prefs["liveness"] = self.target_liveness
        if self.preferred_language is not None:
            prefs["language"] = self.preferred_language
        if self.allow_explicit is not None:
            prefs["allow_explicit"] = self.allow_explicit

        return prefs


# ---------------------------------------------------------------------------
# Dataset loading + validation
# ---------------------------------------------------------------------------

class DatasetError(ValueError):
    """Raised when the songs dataset is malformed."""


def _to_float(value: str, column: str, row_id) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise DatasetError(
            f"Row {row_id}: column '{column}' expected a number but got {value!r}"
        )


def _to_int(value: str, column: str, row_id) -> int:
    text = str(value).strip()
    try:
        # Allow "2020" and "2020.0" style values.
        return int(float(text))
    except (TypeError, ValueError):
        raise DatasetError(
            f"Row {row_id}: column '{column}' expected an integer but got {value!r}"
        )


def _to_bool(value: str, column: str, row_id) -> bool:
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    raise DatasetError(
        f"Row {row_id}: column '{column}' expected a boolean but got {value!r}"
    )


def _parse_row(raw: Dict[str, str]) -> Dict:
    """Validate and type-convert a single CSV row into a clean song dict."""
    # Normalise whitespace in keys and values.
    row = {
        (k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
        for k, v in raw.items()
    }

    row_id = row.get("id", "?")
    missing = [c for c in REQUIRED_COLUMNS if c not in row]
    if missing:
        raise DatasetError(f"Row {row_id}: missing required columns: {missing}")

    song: Dict = {}
    for col in STR_COLUMNS:
        text = row[col]
        if not text:
            raise DatasetError(f"Row {row_id}: column '{col}' must not be empty")
        song[col] = text

    for col in INT_COLUMNS:
        song[col] = _to_int(row[col], col, row_id)

    for col in FLOAT_COLUMNS:
        song[col] = _to_float(row[col], col, row_id)

    for col in BOOL_COLUMNS:
        song[col] = _to_bool(row[col], col, row_id)

    # Range checks.
    for col in UNIT_INTERVAL_FEATURES:
        if not 0.0 <= song[col] <= 1.0:
            raise DatasetError(
                f"Row {row_id}: column '{col}' must be within 0.0-1.0 "
                f"(got {song[col]})"
            )
    if not 0 <= song["popularity"] <= 100:
        raise DatasetError(
            f"Row {row_id}: 'popularity' must be within 0-100 (got {song['popularity']})"
        )
    if song["tempo_bpm"] <= 0:
        raise DatasetError(
            f"Row {row_id}: 'tempo_bpm' must be positive (got {song['tempo_bpm']})"
        )

    return song


def load_songs(csv_path: str) -> List[Dict]:
    """Load and validate songs from ``csv_path``.

    Returns a list of clean song dictionaries with correct Python types.
    Raises :class:`DatasetError` with a clear message if the data is invalid.
    """
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"{csv_path}: file is empty")
        header = [h.strip() for h in reader.fieldnames]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise DatasetError(f"{csv_path}: missing required columns: {missing}")

        songs: List[Dict] = []
        seen_ids = set()
        for raw in reader:
            song = _parse_row(raw)
            if song["id"] in seen_ids:
                raise DatasetError(f"Duplicate song id: {song['id']}")
            seen_ids.add(song["id"])
            songs.append(song)

    if not songs:
        raise DatasetError(f"{csv_path}: no song rows found")
    return songs


# ---------------------------------------------------------------------------
# Scoring strategies (Strategy design pattern)
# ---------------------------------------------------------------------------

class ScoringStrategy(ABC):
    """Abstract base for a scoring strategy.

    A strategy is nothing more than a *named set of feature weights*. All
    strategies use the identical scoring math; they only disagree on how much
    each feature matters. This keeps the different ranking "modes" free of
    tangled ``if/elif`` branches — to add a mode you add a small class.
    """

    name: str = "base"

    @property
    @abstractmethod
    def weights(self) -> Dict[str, float]:
        """Return the per-feature weights for this strategy."""


class BalancedStrategy(ScoringStrategy):
    """Every feature contributes; genre and mood lead but nothing dominates."""

    name = "balanced"

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "genre": 3.0,
            "mood": 2.0,
            "energy": 2.0,
            "tempo": 1.5,
            "valence": 1.0,
            "danceability": 1.0,
            "acousticness": 1.5,
            "popularity": 1.0,
            "decade": 1.0,
            "instrumentalness": 1.0,
            "speechiness": 0.5,
            "liveness": 0.5,
            "language": 0.5,
            "explicit": 1.0,
        }


class GenreFirstStrategy(ScoringStrategy):
    """Genre match overwhelmingly drives the ranking."""

    name = "genre_first"

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "genre": 6.0,
            "mood": 1.5,
            "energy": 1.0,
            "tempo": 0.75,
            "valence": 0.5,
            "danceability": 0.5,
            "acousticness": 0.75,
            "popularity": 0.5,
            "decade": 0.5,
            "instrumentalness": 0.5,
            "speechiness": 0.25,
            "liveness": 0.25,
            "language": 0.5,
            "explicit": 1.0,
        }


class MoodFirstStrategy(ScoringStrategy):
    """Mood match overwhelmingly drives the ranking."""

    name = "mood_first"

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "genre": 1.5,
            "mood": 6.0,
            "energy": 1.5,
            "tempo": 0.75,
            "valence": 1.5,
            "danceability": 0.75,
            "acousticness": 1.0,
            "popularity": 0.5,
            "decade": 0.5,
            "instrumentalness": 0.5,
            "speechiness": 0.25,
            "liveness": 0.25,
            "language": 0.5,
            "explicit": 1.0,
        }


class EnergyFocusedStrategy(ScoringStrategy):
    """Rewards songs whose energy, tempo and danceability match the target."""

    name = "energy_focused"

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "genre": 1.0,
            "mood": 1.0,
            "energy": 5.0,
            "tempo": 3.0,
            "valence": 0.75,
            "danceability": 2.0,
            "acousticness": 0.5,
            "popularity": 0.5,
            "decade": 0.25,
            "instrumentalness": 0.5,
            "speechiness": 0.25,
            "liveness": 0.5,
            "language": 0.25,
            "explicit": 1.0,
        }


# Registry / factory mapping mode name -> strategy instance.
_STRATEGIES: Dict[str, ScoringStrategy] = {
    s.name: s
    for s in (
        BalancedStrategy(),
        GenreFirstStrategy(),
        MoodFirstStrategy(),
        EnergyFocusedStrategy(),
    )
}


def available_modes() -> List[str]:
    """Return the list of valid ranking-mode names."""
    return list(_STRATEGIES.keys())


def get_strategy(mode: str) -> ScoringStrategy:
    """Look up a scoring strategy by mode name, raising on unknown modes."""
    try:
        return _STRATEGIES[mode]
    except KeyError:
        raise ValueError(
            f"Unknown ranking mode: {mode!r}. Valid modes: {available_modes()}"
        )


# ---------------------------------------------------------------------------
# Scoring core (shared by both the functional and OOP APIs)
# ---------------------------------------------------------------------------

# Numeric closeness features: pref_key -> (song_key, value_range)
# similarity = 1 - |target - value| / range, clamped to [0, 1].
_NUMERIC_FEATURES: Dict[str, Tuple[str, float]] = {
    "energy": ("energy", 1.0),
    "tempo": ("tempo_bpm", 120.0),
    "valence": ("valence", 1.0),
    "danceability": ("danceability", 1.0),
    "acousticness": ("acousticness", 1.0),
    "popularity": ("popularity", 100.0),
    "instrumentalness": ("instrumentalness", 1.0),
    "speechiness": ("speechiness", 1.0),
    "liveness": ("liveness", 1.0),
}

# Decade closeness fades to 0 across four decades of distance.
_DECADE_RANGE = 40.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _closeness(target: float, actual: float, value_range: float) -> float:
    """Bounded numeric similarity in [0, 1]: closer values score higher."""
    return _clamp01(1.0 - abs(target - actual) / value_range)


def _score_song_core(
    user_prefs: Dict, song: Dict, strategy: ScoringStrategy
) -> Tuple[float, List[str]]:
    """Compute a song's score and truthful reason list under ``strategy``.

    Every reason string is generated directly from the number that was added
    to the score, so explanations can never drift away from the math.
    """
    weights = strategy.weights
    score = 0.0
    reasons: List[str] = []

    # --- Genre (categorical match) --------------------------------------
    genres = _requested_categories(user_prefs, "genres", "genre")
    if genres:
        weight = weights["genre"]
        if song["genre"] in genres:
            contribution = weight * 1.0
            reasons.append(f"genre match: {song['genre']} (+{contribution:.2f})")
        else:
            contribution = 0.0
            reasons.append(
                f"genre mismatch: {song['genre']} not in {sorted(genres)} (+0.00)"
            )
        score += contribution

    # --- Mood (categorical match) ---------------------------------------
    moods = _requested_categories(user_prefs, "moods", "mood")
    if moods:
        weight = weights["mood"]
        if song["mood"] in moods:
            contribution = weight * 1.0
            reasons.append(f"mood match: {song['mood']} (+{contribution:.2f})")
        else:
            contribution = 0.0
            reasons.append(
                f"mood mismatch: {song['mood']} not in {sorted(moods)} (+0.00)"
            )
        score += contribution

    # --- Numeric closeness features -------------------------------------
    for pref_key, (song_key, value_range) in _NUMERIC_FEATURES.items():
        if user_prefs.get(pref_key) is None:
            continue
        target = float(user_prefs[pref_key])
        actual = float(song[song_key])
        similarity = _closeness(target, actual, value_range)
        contribution = weights[pref_key] * similarity
        score += contribution
        reasons.append(
            f"{pref_key} similarity: {similarity:.2f} (+{contribution:.2f})"
        )

    # --- Decade preference ----------------------------------------------
    if user_prefs.get("decade") is not None:
        target_decade = float(user_prefs["decade"])
        similarity = _closeness(target_decade, float(song["release_decade"]), _DECADE_RANGE)
        contribution = weights["decade"] * similarity
        score += contribution
        reasons.append(
            f"decade similarity ({song['release_decade']}s): "
            f"{similarity:.2f} (+{contribution:.2f})"
        )

    # --- Language match -------------------------------------------------
    if user_prefs.get("language") is not None:
        weight = weights["language"]
        if song["language"] == user_prefs["language"]:
            contribution = weight
            reasons.append(
                f"preferred language match: {song['language']} (+{contribution:.2f})"
            )
        else:
            contribution = 0.0
            reasons.append(
                f"language mismatch: {song['language']} (+0.00)"
            )
        score += contribution

    # --- Explicit-content compatibility (can subtract) ------------------
    if user_prefs.get("allow_explicit") is not None:
        allow = bool(user_prefs["allow_explicit"])
        if song["explicit"] and not allow:
            contribution = -weights["explicit"]
            score += contribution
            reasons.append(f"explicit-content mismatch ({contribution:.2f})")
        elif not song["explicit"] and not allow:
            reasons.append("clean-content match (+0.00)")

    return score, reasons


def _requested_categories(prefs: Dict, list_key: str, single_key: str) -> set:
    """Collect requested categorical values from either a list or single key."""
    values = set()
    listed = prefs.get(list_key)
    if listed:
        values.update(listed)
    single = prefs.get(single_key)
    if single:
        values.add(single)
    return values


def score_song(
    user_prefs: Dict, song: Dict, mode: str = "balanced"
) -> Tuple[float, List[str]]:
    """Score a single song dict against ``user_prefs`` under ``mode``.

    Returns ``(score, reasons)`` where ``score`` is a float and ``reasons`` is
    a list of human-readable contribution strings.
    """
    strategy = get_strategy(mode)
    return _score_song_core(user_prefs, song, strategy)


# ---------------------------------------------------------------------------
# Diversity / novelty reranking
# ---------------------------------------------------------------------------

# Penalty applied for each earlier pick by the same artist.
ARTIST_REPEAT_PENALTY = 1.5
# Penalty applied for each earlier pick sharing the same genre.
GENRE_REPEAT_PENALTY = 0.4


def _diversity_penalty(
    song: Dict, artist_counts: Dict[str, int], genre_counts: Dict[str, int]
) -> Tuple[float, List[str]]:
    """Return the penalty (>= 0) and reason strings for adding ``song`` given
    what has already been selected."""
    penalty = 0.0
    reasons: List[str] = []

    already_artist = artist_counts.get(song["artist"], 0)
    if already_artist > 0:
        amount = ARTIST_REPEAT_PENALTY * already_artist
        penalty += amount
        reasons.append(f"artist-repeat penalty: {song['artist']} (-{amount:.2f})")

    already_genre = genre_counts.get(song["genre"], 0)
    if already_genre > 0:
        amount = GENRE_REPEAT_PENALTY * already_genre
        penalty += amount
        reasons.append(f"genre-concentration penalty: {song['genre']} (-{amount:.2f})")

    return penalty, reasons


def _sort_key(item: Tuple[Dict, float, List[str]]):
    """Deterministic ranking: score desc, then title asc, then id asc."""
    song, base_score, _ = item
    return (-base_score, song["title"], song["id"])


def _diversify(
    scored: List[Tuple[Dict, float, List[str]]], k: int
) -> List[Tuple[Dict, float, List[str]]]:
    """Greedy diversity reranking.

    At each step pick the remaining candidate with the highest score *after*
    subtracting artist-repeat and genre-concentration penalties for what has
    already been chosen. The applied penalty is folded into the final score and
    reported in the reasons, so the output stays fully explainable and
    deterministic.
    """
    remaining = list(scored)
    selected: List[Tuple[Dict, float, List[str]]] = []
    artist_counts: Dict[str, int] = {}
    genre_counts: Dict[str, int] = {}

    while remaining and len(selected) < k:
        best_index = None
        best_adjusted = None
        best_penalty = 0.0
        best_penalty_reasons: List[str] = []

        for index, (song, base_score, _reasons) in enumerate(remaining):
            penalty, penalty_reasons = _diversity_penalty(
                song, artist_counts, genre_counts
            )
            adjusted = base_score - penalty
            candidate_key = (-adjusted, song["title"], song["id"])
            if best_adjusted is None or candidate_key < best_adjusted:
                best_adjusted = candidate_key
                best_index = index
                best_penalty = penalty
                best_penalty_reasons = penalty_reasons

        song, base_score, reasons = remaining.pop(best_index)
        final_reasons = list(reasons) + best_penalty_reasons
        selected.append((song, base_score - best_penalty, final_reasons))

        artist_counts[song["artist"]] = artist_counts.get(song["artist"], 0) + 1
        genre_counts[song["genre"]] = genre_counts.get(song["genre"], 0) + 1

    return selected


# ---------------------------------------------------------------------------
# Recommendation (functional API)
# ---------------------------------------------------------------------------

def _reasons_to_text(reasons: Sequence[str]) -> str:
    """Join reason fragments into a single human-readable explanation."""
    return "; ".join(reasons) if reasons else "no active preferences"


# ---------------------------------------------------------------------------
# Public ranking helpers (used by the Project 4 agent to reuse P3 math)
# ---------------------------------------------------------------------------

def reasons_to_text(reasons: Sequence[str]) -> str:
    """Public alias so callers can format reason lists the same way the CLI does."""
    return _reasons_to_text(reasons)


def sort_scored(
    scored: List[Tuple[Dict, float, List[str]]]
) -> List[Tuple[Dict, float, List[str]]]:
    """Deterministically sort ``(song, score, reasons)`` tuples (score desc)."""
    ordered = list(scored)
    ordered.sort(key=_sort_key)
    return ordered


def diversify_scored(
    scored: List[Tuple[Dict, float, List[str]]], k: int
) -> List[Tuple[Dict, float, List[str]]]:
    """Public wrapper around the greedy diversity reranker.

    Lets the Project 4 agent apply artist/genre diversity to a list of songs it
    has already scored (for example, after adding a retrieval bonus) without
    duplicating the diversity math from :func:`recommend_songs`.
    """
    return _diversify(list(scored), k)


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    mode: str = "balanced",
    diversify: bool = True,
) -> List[Tuple[Dict, float, str]]:
    """Rank ``songs`` for ``user_prefs`` and return up to the top ``k``.

    Each result is ``(song_dict, final_score, explanation)``. The input list is
    never mutated. Results are deterministic and sorted by final adjusted
    score (descending), with title then id as tie-breakers.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    # Validate the mode early so bad modes fail clearly.
    strategy = get_strategy(mode)

    if not songs:
        return []

    # Score every song against a defensive copy so we never mutate the input.
    scored: List[Tuple[Dict, float, List[str]]] = []
    for song in songs:
        song_copy = dict(song)
        base_score, reasons = _score_song_core(user_prefs, song_copy, strategy)
        scored.append((song_copy, base_score, reasons))

    scored.sort(key=_sort_key)

    if diversify:
        ranked = _diversify(scored, k)
    else:
        ranked = scored[:k]

    return [
        (song, round(score, 4), _reasons_to_text(reasons))
        for song, score, reasons in ranked
    ]


# ---------------------------------------------------------------------------
# Recommender (OOP API)
# ---------------------------------------------------------------------------

class Recommender:
    """Object-oriented wrapper around the same scoring core.

    Accepts a list of :class:`Song` objects and produces recommendations. It
    delegates all scoring to :func:`recommend_songs` so the OOP and functional
    paths can never disagree.
    """

    def __init__(self, songs: List[Song], mode: str = "balanced"):
        self.songs = songs
        self.mode = mode
        self._by_id = {song.id: song for song in songs}

    def _prefs(self, user: UserProfile) -> Dict:
        return user.to_prefs()

    def recommend(
        self,
        user: UserProfile,
        k: int = 5,
        mode: Optional[str] = None,
        diversify: bool = True,
    ) -> List[Song]:
        """Return the top ``k`` songs (as :class:`Song` objects)."""
        detailed = self.recommend_detailed(user, k=k, mode=mode, diversify=diversify)
        return [song for song, _score, _why in detailed]

    def recommend_detailed(
        self,
        user: UserProfile,
        k: int = 5,
        mode: Optional[str] = None,
        diversify: bool = True,
    ) -> List[Tuple[Song, float, str]]:
        """Like :meth:`recommend` but also returns scores and explanations."""
        chosen_mode = mode or self.mode
        song_dicts = [song.to_dict() for song in self.songs]
        results = recommend_songs(
            self._prefs(user), song_dicts, k=k, mode=chosen_mode, diversify=diversify
        )
        detailed: List[Tuple[Song, float, str]] = []
        for song_dict, score, explanation in results:
            detailed.append((self._by_id[song_dict["id"]], score, explanation))
        return detailed

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a truthful, human-readable explanation for one song."""
        score, reasons = score_song(self._prefs(user), song.to_dict(), self.mode)
        return (
            f"{song.title} by {song.artist} scored {score:.2f} "
            f"[{self.mode} mode]: {_reasons_to_text(reasons)}"
        )
