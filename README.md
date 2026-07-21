# 🎵 Music Recommender Simulation — VibeScope

A small, fully **explainable, content-based** music recommender. Given a user
"taste profile" and a catalog of songs, VibeScope scores every song with a
deterministic weighted formula, ranks them, optionally re-ranks for variety,
and explains *exactly* why each song was chosen — every number in an
explanation is generated from the number that was actually added to the score.

There is no machine learning and no listening history here. This is a
classroom simulation designed to make the moving parts of a recommender
(features → preferences → scoring → ranking → selection) visible and honest.

---

## How real music recommendation systems work

Production platforms (Spotify, Apple Music, YouTube Music) decide what to play
next by combining many signals:

- **Behavioral signals** — likes, skips, saves, repeat plays, listen/watch
  time, playlist adds, and search activity.
- **Context** — time of day, device, whether you are working out or winding
  down.
- **Item metadata / audio features** — genre, tempo, energy, "danceability",
  and other measured characteristics of the audio itself.

Two big families of algorithms turn those signals into recommendations:

- **Collaborative filtering** learns from *patterns across many users*: "people
  who liked what you liked also liked X." It needs lots of user history and can
  surface songs that share no obvious attributes with your favorites.
- **Content-based filtering** compares *item attributes* against a profile of
  what a single user is known to like. It needs no other users, works from day
  one, but tends to recommend "more of the same."

It helps to separate four stages that every recommender shares:

1. **Input data** — describes the songs (and, in real systems, user behavior).
2. **User preferences** — what the system *believes* a person likes.
3. **Scoring** — converts features + preferences into a numeric relevance value.
4. **Ranking & selection** — sorts candidates by relevance and picks the final
   top results, sometimes applying diversity, business, safety, or freshness
   rules on top.

**This project is a small content-based simulation, not a production Spotify
model.** It implements stages 1–4 with transparent, hand-tuned weights so you
can read every decision.

---

## How this classroom simulation works

- Each **song** is a row of attributes (genre, mood, energy, tempo, valence,
  danceability, acousticness, plus popularity, decade, instrumentalness,
  speechiness, liveness, language, and an explicit flag).
- A **user profile** stores target values for those same attributes (e.g.
  "energy ≈ 0.9, genre = pop, acousticness ≈ 0.1").
- The **recommender** scores each song by measuring how *close* it is to the
  user's targets, multiplies each closeness by a per-feature weight, and sums
  them into a single score.
- **Selection** sorts by score and returns the top *k*, optionally re-ranking
  to avoid recommending too many songs by the same artist or in the same genre.

---

## Dataset overview

`data/songs.csv` contains **28 fictional songs** (fictional titles and
artists — no real lyrics or artists). It spans **16 genres**, **11 moods**,
**6 languages**, and **7 release decades (1960s–2020s)**, and no artist appears
more than twice.

### Base attributes (from the starter)

`id`, `title`, `artist`, `genre`, `mood`, `energy`, `tempo_bpm`, `valence`,
`danceability`, `acousticness`.

### Additional stretch attributes (Agentic AI feature — 7 new)

| Attribute | Range | Used in scoring as |
|---|---|---|
| `popularity` | 0–100 | closeness to target popularity |
| `release_decade` | 1960–2020 | closeness to preferred decade |
| `instrumentalness` | 0.0–1.0 | closeness to target |
| `speechiness` | 0.0–1.0 | closeness to target |
| `liveness` | 0.0–1.0 | closeness to target |
| `language` | e.g. English/Spanish/… | exact-match bonus |
| `explicit` | True/False | penalty when user disallows explicit |

`duration_seconds` is also stored but is **not** counted among the seven
scored attributes.

All rows are validated on load (`load_songs`): required columns must exist,
numeric fields must parse, 0–1 features must stay in range, popularity must be
0–100, ids must be unique, booleans are parsed consistently, and leading/
trailing whitespace is stripped. Bad data raises a clear `DatasetError`.

---

## Complete feature list

- Content-based weighted scoring over 14 scored features.
- Deterministic ranking with stable tie-breaking (score ↓, title ↑, id ↑).
- Truthful explanations built directly from score contributions.
- **Four ranking modes** via a Strategy design pattern.
- **Diversity / novelty reranking** to reduce filter bubbles (toggleable).
- Dataset validation with clear errors.
- `argparse` CLI with four named profiles, mode switching, and a formatted
  table (`tabulate`).
- Optional **Streamlit** UI (`src/app.py`) reusing the same logic.
- 47 deterministic tests.

---

## Scoring formula

For each feature the user expressed a preference on:

```
numeric features:   similarity = clamp01(1 - |target - value| / range)
                    contribution = weight[mode][feature] * similarity

categorical (genre, mood, language):
                    contribution = weight if the value matches, else 0

explicit content:   contribution = -weight  if song is explicit and the
                                            user disallows explicit content

final_score = sum(contributions) - diversity_penalties
```

`range` normalizes each feature (1.0 for 0–1 features, 120 bpm for tempo, 100
for popularity, 40 years for decade). Similarity is always clamped to
`[0.0, 1.0]`, so the system rewards *closeness* to the target — a song is never
rewarded simply for having higher energy.

---

## Ranking modes and the Strategy pattern

A "mode" is just a named set of feature weights. Each mode is a small concrete
class implementing the `ScoringStrategy` base, registered in a factory
(`get_strategy`). This keeps mode logic out of one giant `if/elif` block — to
add a mode you add a class.

| Mode | What it emphasizes |
|---|---|
| `balanced` (default) | every feature contributes; genre & mood lead |
| `genre_first` | genre match dominates (weight 6.0) |
| `mood_first` | mood match dominates (weight 6.0) |
| `energy_focused` | energy (5.0), tempo (3.0), danceability (2.0) |

Unknown modes raise a clear `ValueError`.

---

## Diversity and filter-bubble mitigation

After base scoring, `recommend_songs(..., diversify=True)` performs a
transparent **greedy reranking**. As it fills the top-k list it subtracts:

- `ARTIST_REPEAT_PENALTY = 1.5` for each earlier pick by the same artist, and
- `GENRE_REPEAT_PENALTY = 0.4` for each earlier pick sharing the same genre.

These penalties are folded into the final score and reported in the
explanation, and the result stays fully deterministic. Set `diversify=False`
(or pass `--no-diversity`) to get pure score order. See `model_card.md` for why
this reduces repetition but does **not** guarantee fairness.

---

## Project structure

```
music-recommender/
├── data/songs.csv          # 28-song validated catalog
├── src/
│   ├── recommender.py      # models, loader, scoring, strategies, diversity
│   ├── profiles.py         # four named user profiles
│   ├── main.py             # argparse CLI + tabulate table
│   └── app.py              # optional Streamlit UI
├── tests/test_recommender.py
├── outputs/                # captured real CLI runs
├── requirements.txt
├── README.md
├── model_card.md
└── ai_interactions.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Run instructions

```bash
python -m src.main                      # default: High-Energy Pop, balanced
python -m src.main --all-profiles --mode balanced --top-k 5
streamlit run src/app.py                # optional UI
```

### Ranking-mode CLI commands

```bash
python -m src.main --profile high-energy-pop --mode balanced --top-k 5
python -m src.main --profile acoustic-chill --mode mood_first --top-k 5
python -m src.main --profile intense-rock --mode genre_first --top-k 5
python -m src.main --profile edm-workout --mode energy_focused --top-k 5
python -m src.main --compare-modes --profile high-energy-pop --top-k 5
python -m src.main --profile high-energy-pop --no-diversity
```

---

## Profile experiments

Four distinct profiles are defined in `src/profiles.py`; each differs across
several preferences, not just genre. The outputs below are the **actual**
captured runs saved under `outputs/`.

### 1. High-Energy Pop — `--mode balanced`

```
=== High-Energy Pop  |  mode=balanced  |  top-5 ===
Preferences: genres=pop, moods=energetic/happy, energy=0.9, tempo=124, acousticness=0.12, danceability=0.85, popularity=82, explicit_ok=False
+-----+---------------+---------------+---------+-----------+---------+---------------------------------+
|   # | Title         | Artist        | Genre   | Mood      |   Score | Why                             |
+=====+===============+===============+=========+===========+=========+=================================+
|   1 | Sunrise City  | Neon Echo     | pop     | happy     |   14.07 | genre match: pop (+3.00)        |
|     |               |               |         |           |         | mood match: happy (+2.00)       |
|     |               |               |         |           |         | energy similarity: 0.92 (+1.84) |
+-----+---------------+---------------+---------+-----------+---------+---------------------------------+
|   2 | Seoul Nights  | Hana Kim      | pop     | energetic |   13.38 | genre match: pop (+3.00)        |
|     |               |               |         |           |         | mood match: energetic (+2.00)   |
|     |               |               |         |           |         | energy similarity: 0.96 (+1.92) |
+-----+---------------+---------------+---------+-----------+---------+---------------------------------+
|   3 | Cielo Abierto | Los Vientos   | pop     | happy     |   12.7  | genre match: pop (+3.00)        |
|     |               |               |         |           |         | mood match: happy (+2.00)       |
|     |               |               |         |           |         | energy similarity: 0.90 (+1.80) |
+-----+---------------+---------------+---------+-----------+---------+---------------------------------+
|   4 | Neon Pulse    | Circuit Bloom | edm     | energetic |   11.01 | mood match: energetic (+2.00)   |
|     |               |               |         |           |         | energy similarity: 0.95 (+1.90) |
|     |               |               |         |           |         | tempo similarity: 0.97 (+1.45)  |
+-----+---------------+---------------+---------+-----------+---------+---------------------------------+
|   5 | Gym Hero      | Max Pulse     | pop     | intense   |   10.81 | genre match: pop (+3.00)        |
|     |               |               |         |           |         | energy similarity: 0.97 (+1.94) |
|     |               |               |         |           |         | tempo similarity: 0.93 (+1.40)  |
+-----+---------------+---------------+---------+-----------+---------+---------------------------------+
Diversity reranking: ENABLED (artist-repeat and genre-concentration penalties applied)
```

*Why:* pop / happy / high-energy tracks dominate. Note `Neon Pulse` (edm) sneaks
into #4 above a fourth pop song — the diversity reranker penalizes over-
concentration in one genre.

### 2. Acoustic Chill — `--mode mood_first`

```
=== Acoustic Chill  |  mode=mood_first  |  top-5 ===
Preferences: genres=folk/jazz/ambient/lofi, moods=relaxed/chill, energy=0.33, tempo=76, acousticness=0.88, danceability=0.5, popularity=48, explicit_ok=False
+-----+---------------------+----------------+---------+---------+---------+---------------------------------+
|   # | Title               | Artist         | Genre   | Mood    |   Score | Why                             |
+=====+=====================+================+=========+=========+=========+=================================+
|   1 | Sunday Slowdown     | Slow Stereo    | jazz    | relaxed |   13.78 | genre match: jazz (+1.50)       |
|     |                     |                |         |         |         | mood match: relaxed (+6.00)     |
|     |                     |                |         |         |         | energy similarity: 0.99 (+1.48) |
+-----+---------------------+----------------+---------+---------+---------+---------------------------------+
|   2 | Library Rain        | Paper Lanterns | lofi    | chill   |   13.77 | genre match: lofi (+1.50)       |
|     |                     |                |         |         |         | mood match: chill (+6.00)       |
|     |                     |                |         |         |         | energy similarity: 0.98 (+1.47) |
+-----+---------------------+----------------+---------+---------+---------+---------------------------------+
|   3 | Spacewalk Thoughts  | Orbit Bloom    | ambient | chill   |   13.48 | genre match: ambient (+1.50)    |
|     |                     |                |         |         |         | mood match: chill (+6.00)       |
|     |                     |                |         |         |         | energy similarity: 0.95 (+1.42) |
+-----+---------------------+----------------+---------+---------+---------+---------------------------------+
|   4 | Midnight Coding     | LoRoom         | lofi    | chill   |   13.02 | genre match: lofi (+1.50)       |
|     |                     |                |         |         |         | mood match: chill (+6.00)       |
|     |                     |                |         |         |         | energy similarity: 0.91 (+1.36) |
+-----+---------------------+----------------+---------+---------+---------+---------------------------------+
|   5 | Coffee Shop Stories | Slow Stereo    | jazz    | relaxed |   11.68 | genre match: jazz (+1.50)       |
|     |                     |                |         |         |         | mood match: relaxed (+6.00)     |
|     |                     |                |         |         |         | energy similarity: 0.96 (+1.44) |
+-----+---------------------+----------------+---------+---------+---------+---------------------------------+
Diversity reranking: ENABLED (artist-repeat and genre-concentration penalties applied)
```

*Why:* the profile shifts hard toward **low-energy, high-acousticness** chill/
relaxed tracks. In `mood_first` mode the mood match (+6.00) dominates, so quiet
jazz/lofi/ambient tracks top the list — a completely different world from the
pop profile.

### 3. Intense Rock — `--mode genre_first`

```
=== Intense Rock  |  mode=genre_first  |  top-5 ===
Preferences: genres=metal/rock, moods=dark/intense, energy=0.92, tempo=150, acousticness=0.1, danceability=0.6, popularity=62, explicit_ok=True
+-----+----------------+---------------+---------+---------+---------+---------------------------------+
|   # | Title          | Artist        | Genre   | Mood    |   Score | Why                             |
+=====+================+===============+=========+=========+=========+=================================+
|   1 | Storm Runner   | Voltline      | rock    | intense |   11.36 | genre match: rock (+6.00)       |
|     |                |               |         |         |         | mood match: intense (+1.50)     |
|     |                |               |         |         |         | energy similarity: 0.99 (+0.99) |
+-----+----------------+---------------+---------+---------+---------+---------------------------------+
|   2 | Iron Veins     | Blackforge    | metal   | intense |   11.17 | genre match: metal (+6.00)      |
|     |                |               |         |         |         | mood match: intense (+1.50)     |
|     |                |               |         |         |         | energy similarity: 0.95 (+0.95) |
+-----+----------------+---------------+---------+---------+---------+---------------------------------+
|   3 | Rust and Bone  | Voltline      | rock    | dark    |    9.42 | genre match: rock (+6.00)       |
|     |                |               |         |         |         | mood match: dark (+1.50)        |
|     |                |               |         |         |         | energy similarity: 0.93 (+0.93) |
+-----+----------------+---------------+---------+---------+---------+---------------------------------+
|   4 | Bass Cathedral | Circuit Bloom | edm     | dark    |    5.2  | mood match: dark (+1.50)        |
|     |                |               |         |         |         | energy similarity: 0.96 (+0.96) |
|     |                |               |         |         |         | tempo similarity: 0.92 (+0.69)  |
+-----+----------------+---------------+---------+---------+---------+---------------------------------+
|   5 | Gym Hero       | Max Pulse     | pop     | intense |    4.99 | mood match: intense (+1.50)     |
|     |                |               |         |         |         | energy similarity: 0.99 (+0.99) |
|     |                |               |         |         |         | tempo similarity: 0.85 (+0.64)  |
+-----+----------------+---------------+---------+---------+---------+---------------------------------+
Diversity reranking: ENABLED (artist-repeat and genre-concentration penalties applied)
```

*Why:* in `genre_first` mode the +6.00 genre bonus makes rock/metal tracks pull
far ahead (scores ~11 vs ~5 for off-genre songs). The steep score cliff between
#3 and #4 shows the genre weight doing the heavy lifting.

### Comparing the profiles

The three profiles above share the **same catalog and the same scoring engine**
yet produce almost no overlap in their top 5. Pop favors bright, danceable,
mainstream tracks; Acoustic Chill collapses onto quiet, acoustic, instrumental
tracks; Intense Rock climbs the genre/energy axis. That is content-based
filtering working as intended: the *profile*, not the catalog, drives the
result.

---

## Mode-comparison experiment

`python -m src.main --compare-modes --profile high-energy-pop --top-k 5`:

```
[balanced]
  Sunrise City (14.07) > Seoul Nights (13.38) > Cielo Abierto (12.70) > Neon Pulse (11.01) > Gym Hero (10.81)

[genre_first]
  Sunrise City (12.29) > Seoul Nights (11.49) > Cielo Abierto (10.95) > Gym Hero (9.56) > Neon Pulse (6.25)

[mood_first]
  Sunrise City (14.20) > Seoul Nights (13.41) > Cielo Abierto (12.88) > Neon Pulse (12.57) > Electric Youth (12.40)

[energy_focused]
  Seoul Nights (13.64) > Sunrise City (13.12) > Neon Pulse (12.64) > Cielo Abierto (12.49) > Electric Youth (12.48)
```

*Observation:* switching modes changes which features dominate. `energy_focused`
promotes `Seoul Nights` to #1 (its energy/tempo hug the targets), and
`genre_first` pushes the off-genre `Neon Pulse` down to #5 (6.25) while
`mood_first` pulls it up to #4 (12.57). Same songs, different lens.

---

## Diversity experiment

Same profile/mode, diversity **off** (`--no-diversity`):

```
1. Sunrise City  (14.07) pop
2. Seoul Nights  (13.78) pop
3. Cielo Abierto (13.50) pop
4. Gym Hero      (12.02) pop
5. Neon Pulse    (11.01) edm
```

With diversity **on** (default), the top 5 becomes:

```
1. Sunrise City  (14.07) pop
2. Seoul Nights  (13.38) pop   <- -0.40 genre-concentration penalty
3. Cielo Abierto (12.70) pop   <- -0.80 genre-concentration penalty
4. Neon Pulse    (11.01) edm   <- promoted above a 4th pop song
5. Gym Hero      (10.81) pop   <- pushed to #5
```

*Observation:* the diversity reranker leaves the strong #1 untouched, gently
lowers each additional pop track, and lets `Neon Pulse` (edm) climb — trading a
little relevance for more genre variety without randomizing the list.

---

## Testing

```bash
python -m compileall src tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v
```

Actual passing output (full log in `test_results.txt`):

```
============================= test session starts ==============================
platform darwin -- Python 3.12.1, pytest-9.1.1, pluggy-1.6.0
collecting ... collected 47 items

tests/test_recommender.py::test_recommend_returns_songs_sorted_by_score PASSED
tests/test_recommender.py::test_explain_recommendation_returns_non_empty_string PASSED
... (43 more) ...
tests/test_recommender.py::test_scores_are_deterministic PASSED

============================== 47 passed in 0.03s ==============================
```

---

## Recommendation explanations (real examples)

1. **Sunrise City** (High-Energy Pop, balanced, 14.07):
   `genre match: pop (+3.00); mood match: happy (+2.00); energy similarity: 0.92 (+1.84)`
2. **Sunday Slowdown** (Acoustic Chill, mood_first, 13.78):
   `genre match: jazz (+1.50); mood match: relaxed (+6.00); energy similarity: 0.99 (+1.48)`
3. **Iron Veins** (Intense Rock, genre_first, 11.17):
   `genre match: metal (+6.00); mood match: intense (+1.50); energy similarity: 0.95 (+0.95)`

Each explanation is generated from the exact contributions summed into the
score — including any diversity penalty, e.g.
`genre-concentration penalty: pop (-0.40)`.

---

## Limitations and risks

- Tiny synthetic catalog (28 songs) with hand-authored metadata.
- Weights are hand-tuned, not learned from real feedback.
- No collaborative filtering and no listening history — pure content-based.
- Genre/mood are single coarse labels; real taste is fuzzier and multi-label.
- Diversity penalties are a heuristic; they reduce repetition but do not
  guarantee fairness.
- Rewarding popularity can introduce mainstream bias.

A deeper treatment is in the model card.

---

## Model card

See [**model_card.md**](model_card.md) for the full VibeScope Recommender 1.0
model card (intended use, evaluation, limitations, fairness, and future work).

---

## AI collaboration summary

This project was built with the Claude Code agent driving repository setup,
dataset expansion, implementation, testing, documentation, and Git history. All
CLI output, test output, and evaluation results in this README are copied from
real runs (see `outputs/` and `test_results.txt`) — none are fabricated. The
Strategy pattern and additional-attribute work are documented in
[ai_interactions.md](ai_interactions.md).

## Reflection

Building VibeScope made the "magic" of recommenders concrete: a recommendation
is just a weighted sum of how closely a song matches a stored preference, then
a sort. Making the scoring explainable forced every design choice into the
open — which is also where bias hides. Because the weights and the popularity
signal are chosen by hand, the system can quietly favor mainstream, English-
language, non-explicit music unless you deliberately design against it. The
diversity reranker helped me see that "relevance" and "variety" genuinely pull
in opposite directions, and that mitigating a filter bubble is a tradeoff you
tune, not a box you check.
