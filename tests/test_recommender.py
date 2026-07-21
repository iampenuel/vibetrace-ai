"""Deterministic test suite for the Music Recommender Simulation.

Covers dataset loading/validation, the scoring core, ranking, the Strategy
pattern modes, diversity reranking, and explanations. The two original
starter tests are preserved at the top.
"""

import os

import pytest

from src.recommender import (
    DatasetError,
    Recommender,
    Song,
    UserProfile,
    available_modes,
    load_songs,
    recommend_songs,
    score_song,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


# ---------------------------------------------------------------------------
# Starter tests (preserved)
# ---------------------------------------------------------------------------

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def songs():
    return load_songs(DATA_PATH)


BASE_PREFS = {
    "genre": "pop",
    "mood": "happy",
    "energy": 0.85,
    "tempo": 120,
    "valence": 0.8,
    "danceability": 0.8,
    "acousticness": 0.2,
    "popularity": 80,
    "decade": 2020,
    "instrumentalness": 0.1,
    "speechiness": 0.05,
    "liveness": 0.15,
    "language": "English",
    "allow_explicit": False,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def test_csv_loads(songs):
    assert isinstance(songs, list)
    assert len(songs) >= 20


def test_new_attributes_present(songs):
    required_new = [
        "popularity",
        "release_decade",
        "instrumentalness",
        "speechiness",
        "liveness",
        "language",
        "explicit",
    ]
    for attr in required_new:
        assert attr in songs[0], f"missing new attribute {attr}"


def test_numeric_types(songs):
    s = songs[0]
    assert isinstance(s["id"], int)
    assert isinstance(s["tempo_bpm"], int)
    assert isinstance(s["popularity"], int)
    assert isinstance(s["energy"], float)
    assert isinstance(s["acousticness"], float)


def test_boolean_field_loads(songs):
    assert all(isinstance(s["explicit"], bool) for s in songs)
    assert any(s["explicit"] for s in songs)
    assert any(not s["explicit"] for s in songs)


def test_ids_unique(songs):
    ids = [s["id"] for s in songs]
    assert len(ids) == len(set(ids))


def test_feature_ranges(songs):
    unit = [
        "energy",
        "valence",
        "danceability",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "liveness",
    ]
    for s in songs:
        for f in unit:
            assert 0.0 <= s[f] <= 1.0
        assert 0 <= s["popularity"] <= 100
        assert s["tempo_bpm"] > 0


def test_dataset_diversity(songs):
    assert len({s["genre"] for s in songs}) >= 10
    assert len({s["mood"] for s in songs}) >= 8
    assert len({s["artist"] for s in songs}) >= 15


def test_load_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,title,artist\n1,X,Y\n")
    with pytest.raises(DatasetError):
        load_songs(str(bad))


def test_load_bad_numeric_raises(tmp_path):
    header = ",".join(
        [
            "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability",
            "acousticness,popularity,release_decade,instrumentalness,speechiness",
            "liveness,language,explicit,duration_seconds",
        ]
    )
    row = "1,X,Y,pop,happy,notanumber,120,0.8,0.7,0.2,80,2020,0.1,0.05,0.1,English,False,200"
    bad = tmp_path / "bad.csv"
    bad.write_text(header + "\n" + row + "\n")
    with pytest.raises(DatasetError):
        load_songs(str(bad))


def test_load_out_of_range_raises(tmp_path):
    header = ",".join(
        [
            "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability",
            "acousticness,popularity,release_decade,instrumentalness,speechiness",
            "liveness,language,explicit,duration_seconds",
        ]
    )
    row = "1,X,Y,pop,happy,1.9,120,0.8,0.7,0.2,80,2020,0.1,0.05,0.1,English,False,200"
    bad = tmp_path / "bad.csv"
    bad.write_text(header + "\n" + row + "\n")
    with pytest.raises(DatasetError):
        load_songs(str(bad))


def test_load_duplicate_id_raises(tmp_path):
    header = ",".join(
        [
            "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability",
            "acousticness,popularity,release_decade,instrumentalness,speechiness",
            "liveness,language,explicit,duration_seconds",
        ]
    )
    row = "1,X,Y,pop,happy,0.8,120,0.8,0.7,0.2,80,2020,0.1,0.05,0.1,English,False,200"
    bad = tmp_path / "bad.csv"
    bad.write_text(header + "\n" + row + "\n" + row + "\n")
    with pytest.raises(DatasetError):
        load_songs(str(bad))


def test_whitespace_normalized(tmp_path):
    header = ",".join(
        [
            "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability",
            "acousticness,popularity,release_decade,instrumentalness,speechiness",
            "liveness,language,explicit,duration_seconds",
        ]
    )
    row = "1,  Spaced Title  ,  Artist X ,pop,happy, 0.8 ,120,0.8,0.7,0.2,80,2020,0.1,0.05,0.1,English,False,200"
    good = tmp_path / "good.csv"
    good.write_text(header + "\n" + row + "\n")
    loaded = load_songs(str(good))
    assert loaded[0]["title"] == "Spaced Title"
    assert loaded[0]["artist"] == "Artist X"
    assert loaded[0]["energy"] == 0.8


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _song(**over):
    base = {
        "id": 1,
        "title": "T",
        "artist": "A",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "tempo_bpm": 120,
        "valence": 0.8,
        "danceability": 0.8,
        "acousticness": 0.2,
        "popularity": 80,
        "release_decade": 2020,
        "instrumentalness": 0.1,
        "speechiness": 0.05,
        "liveness": 0.15,
        "language": "English",
        "explicit": False,
        "duration_seconds": 200,
    }
    base.update(over)
    return base


def test_score_song_returns_number_and_reasons():
    score, reasons = score_song(BASE_PREFS, _song())
    assert isinstance(score, float)
    assert isinstance(reasons, list)
    assert all(isinstance(r, str) for r in reasons)
    assert reasons


def test_genre_preference_changes_score():
    match = score_song({"genre": "pop"}, _song(genre="pop"))[0]
    miss = score_song({"genre": "pop"}, _song(genre="rock"))[0]
    assert match > miss


def test_mood_preference_changes_score():
    match = score_song({"mood": "happy"}, _song(mood="happy"))[0]
    miss = score_song({"mood": "happy"}, _song(mood="dark"))[0]
    assert match > miss


def test_closer_energy_scores_higher():
    prefs = {"energy": 0.9}
    close = score_song(prefs, _song(energy=0.88))[0]
    far = score_song(prefs, _song(energy=0.30))[0]
    assert close > far


def test_closer_tempo_scores_higher():
    prefs = {"tempo": 120}
    close = score_song(prefs, _song(tempo_bpm=122))[0]
    far = score_song(prefs, _song(tempo_bpm=60))[0]
    assert close > far


def test_acoustic_preference_changes_score():
    prefs = {"acousticness": 0.9}
    close = score_song(prefs, _song(acousticness=0.88))[0]
    far = score_song(prefs, _song(acousticness=0.05))[0]
    assert close > far


def test_popularity_preference_changes_score():
    prefs = {"popularity": 90}
    close = score_song(prefs, _song(popularity=88))[0]
    far = score_song(prefs, _song(popularity=20))[0]
    assert close > far


def test_decade_preference_changes_score():
    prefs = {"decade": 2020}
    close = score_song(prefs, _song(release_decade=2020))[0]
    far = score_song(prefs, _song(release_decade=1960))[0]
    assert close > far


def test_language_preference_changes_score():
    prefs = {"language": "English"}
    match = score_song(prefs, _song(language="English"))[0]
    miss = score_song(prefs, _song(language="Spanish"))[0]
    assert match > miss


def test_explicit_compatibility_changes_score():
    prefs = {"allow_explicit": False}
    clean = score_song(prefs, _song(explicit=False))[0]
    explicit = score_song(prefs, _song(explicit=True))[0]
    assert clean > explicit


def test_all_songs_score_without_error(songs):
    for s in songs:
        score, reasons = score_song(BASE_PREFS, s)
        assert isinstance(score, float)
        assert isinstance(reasons, list)


def test_reasons_contain_contribution_text():
    _score, reasons = score_song({"genre": "pop"}, _song(genre="pop"))
    assert any("genre match" in r and "+" in r for r in reasons)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_recommendations_sorted_by_score(songs):
    results = recommend_songs(BASE_PREFS, songs, k=5)
    scores = [score for _s, score, _why in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_returns_requested_number(songs):
    assert len(recommend_songs(BASE_PREFS, songs, k=3)) == 3


def test_top_k_larger_than_dataset(songs):
    results = recommend_songs(BASE_PREFS, songs, k=1000)
    assert len(results) == len(songs)


def test_empty_dataset():
    assert recommend_songs(BASE_PREFS, [], k=5) == []


def test_invalid_k_raises(songs):
    with pytest.raises(ValueError):
        recommend_songs(BASE_PREFS, songs, k=0)
    with pytest.raises(ValueError):
        recommend_songs(BASE_PREFS, songs, k=-3)


def test_different_profiles_differ(songs):
    pop = recommend_songs({"genre": "pop", "mood": "happy"}, songs, k=5)
    metal = recommend_songs({"genre": "metal", "mood": "intense"}, songs, k=5)
    pop_titles = [s["title"] for s, _, _ in pop]
    metal_titles = [s["title"] for s, _, _ in metal]
    assert pop_titles != metal_titles


def test_input_not_mutated(songs):
    snapshot = [dict(s) for s in songs]
    recommend_songs(BASE_PREFS, songs, k=5)
    assert songs == snapshot


def test_incomplete_prefs_work(songs):
    results = recommend_songs({"genre": "pop"}, songs, k=5)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def test_all_modes_run(songs):
    for mode in available_modes():
        results = recommend_songs(BASE_PREFS, songs, k=5, mode=mode)
        assert len(results) == 5


def test_unknown_mode_raises(songs):
    with pytest.raises(ValueError):
        recommend_songs(BASE_PREFS, songs, k=5, mode="does_not_exist")
    with pytest.raises(ValueError):
        score_song(BASE_PREFS, _song(), mode="nope")


def test_mode_changes_ranking(songs):
    prefs = {"genre": "pop", "mood": "energetic", "energy": 0.95, "tempo": 130}
    genre_first = [s["title"] for s, _, _ in recommend_songs(prefs, songs, k=5, mode="genre_first")]
    energy_focused = [s["title"] for s, _, _ in recommend_songs(prefs, songs, k=5, mode="energy_focused")]
    assert genre_first != energy_focused


def test_oop_recommender_compatible(songs):
    song_objs = [Song.from_dict(s) for s in songs]
    rec = Recommender(song_objs)
    user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.85)
    results = rec.recommend(user, k=5)
    assert len(results) == 5
    assert all(isinstance(s, Song) for s in results)


def test_score_song_defaults_to_balanced(songs):
    default = score_song(BASE_PREFS, songs[0])
    balanced = score_song(BASE_PREFS, songs[0], mode="balanced")
    assert default == balanced


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def test_same_artist_repetition_penalised():
    prefs = {"genre": "pop"}
    catalog = [
        _song(id=1, title="A1", artist="Solo", genre="pop", popularity=90),
        _song(id=2, title="A2", artist="Solo", genre="pop", popularity=88),
        _song(id=3, title="B1", artist="Other", genre="pop", popularity=50),
    ]
    results = recommend_songs(prefs, catalog, k=3, diversify=True)
    # The second "Solo" song must carry an artist-repeat penalty in its reasons.
    solo_second = [why for s, _, why in results if s["artist"] == "Solo"][1]
    assert "artist-repeat penalty" in solo_second


def test_genre_concentration_penalised():
    prefs = {"genre": "pop"}
    catalog = [
        _song(id=1, title="A", artist="One", genre="pop"),
        _song(id=2, title="B", artist="Two", genre="pop"),
        _song(id=3, title="C", artist="Three", genre="pop"),
    ]
    results = recommend_songs(prefs, catalog, k=3, diversify=True)
    joined = " ".join(why for _s, _sc, why in results)
    assert "genre-concentration penalty" in joined


def test_disabling_diversity_preserves_score_order():
    prefs = {"genre": "pop"}
    catalog = [
        _song(id=1, title="A1", artist="Solo", genre="pop", popularity=90),
        _song(id=2, title="A2", artist="Solo", genre="pop", popularity=88),
        _song(id=3, title="B1", artist="Other", genre="rock", popularity=50),
    ]
    results = recommend_songs(prefs, catalog, k=3, diversify=False)
    scores = [sc for _s, sc, _w in results]
    assert scores == sorted(scores, reverse=True)
    # Two Solo pop songs stay adjacent at the top with no penalties.
    assert [s["artist"] for s, _, _ in results][:2] == ["Solo", "Solo"]
    assert all("penalty" not in why for _s, _sc, why in results)


def test_diversity_can_improve_artist_variety():
    prefs = {"genre": "pop"}
    catalog = [
        _song(id=1, title="A1", artist="Solo", genre="pop", popularity=95),
        _song(id=2, title="A2", artist="Solo", genre="pop", popularity=93),
        _song(id=3, title="B1", artist="Other", genre="pop", popularity=80),
    ]
    plain = recommend_songs(prefs, catalog, k=2, diversify=False)
    diverse = recommend_songs(prefs, catalog, k=2, diversify=True)
    plain_artists = {s["artist"] for s, _, _ in plain}
    diverse_artists = {s["artist"] for s, _, _ in diverse}
    assert plain_artists == {"Solo"}
    assert diverse_artists == {"Solo", "Other"}


def test_diversified_results_sorted_by_final_score(songs):
    results = recommend_songs({"genre": "pop", "mood": "happy"}, songs, k=6, diversify=True)
    scores = [sc for _s, sc, _w in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def test_at_least_three_non_empty_explanations(songs):
    results = recommend_songs(BASE_PREFS, songs, k=5)
    non_empty = [why for _s, _sc, why in results if why.strip()]
    assert len(non_empty) >= 3


def test_explanation_matches_features_used(songs):
    results = recommend_songs({"genre": "pop", "energy": 0.9}, songs, k=5)
    top_song, _score, why = results[0]
    if top_song["genre"] == "pop":
        assert "genre match: pop" in why
    assert "energy similarity" in why


def test_scores_are_deterministic(songs):
    a = recommend_songs(BASE_PREFS, songs, k=5)
    b = recommend_songs(BASE_PREFS, songs, k=5)
    assert [(s["id"], sc) for s, sc, _ in a] == [(s["id"], sc) for s, sc, _ in b]
