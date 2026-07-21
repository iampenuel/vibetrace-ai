# 🎧 Model Card: VibeScope Recommender 1.0

## 1. Model name

**VibeScope Recommender 1.0** — an explainable, content-based music
recommender simulation.

## 2. Goal / task

Given a user "taste profile" and a catalog of songs, rank the catalog by how
well each song matches the profile and return an explainable top-*k* list.

## 3. Intended use

- Classroom exploration of how recommender systems turn data into predictions.
- Demonstrating content-based scoring, ranking strategies, diversity reranking,
  and transparent explanations.
- A teaching artifact where every score can be traced back to the formula.

## 4. Non-intended use

- **Not** for production or real end-user recommendations.
- Not a source of truth about real songs, artists, or listening behavior (the
  catalog is fictional and synthetic).
- Not suitable for any decision with real stakes; it has no real user data and
  no evaluation against real preferences.

## 5. Dataset description

`data/songs.csv` is a hand-authored, **fictional** catalog. Titles and artists
are invented to avoid using copyrighted lyrics or making claims about real
artists. It deliberately spreads across many genres, moods, languages, and
decades so that different profiles produce different results.

## 6. Dataset size

**28 songs**, **16 genres**, **11 moods**, **6 languages**, **7 release
decades (1960s–2020s)**. No artist appears more than twice.

## 7. Base and additional attributes

- **Base:** `id`, `title`, `artist`, `genre`, `mood`, `energy`, `tempo_bpm`,
  `valence`, `danceability`, `acousticness`.
- **Additional (7 scored):** `popularity` (0–100), `release_decade`,
  `instrumentalness`, `speechiness`, `liveness` (0–1), `language`, `explicit`.
- **Stored but not scored:** `duration_seconds`.

All rows are validated on load (schema, numeric ranges, unique ids, boolean
parsing, whitespace normalization).

## 8. Algorithm summary (plain language)

Imagine a checklist of song qualities — how energetic, how acoustic, what
genre, what mood, and so on. The user profile writes down a target for each
quality. For every song we measure how close it is to each target, give each
quality an importance weight, and add it all up. Songs closest to what the user
wants across the important qualities get the highest total. We then sort by that
total and hand back the top few. Nothing is learned or hidden — it is closeness
× importance, summed.

## 9. Ranking modes

Four modes (a **Strategy** design pattern — each is a small class of weights):

- `balanced` — every feature contributes; genre and mood lead.
- `genre_first` — genre match dominates.
- `mood_first` — mood match dominates.
- `energy_focused` — energy, tempo, and danceability dominate.

Switching modes measurably changes rankings (see README mode-comparison
experiment).

## 10. Diversity reranking

After scoring, a greedy reranker subtracts a penalty for each already-selected
song by the same **artist** (1.5) or in the same **genre** (0.4). Penalties are
included in the final score and shown in the explanation. It can be disabled
with `diversify=False` / `--no-diversity`.

## 11. Strengths

- Fully explainable: every number in an explanation comes from the score math.
- Deterministic and reproducible (stable tie-breaking; identical runs match).
- Cold-start friendly: needs no user history to work.
- Cleanly separated stages (features → preferences → scoring → ranking →
  selection) and easy-to-extend modes.
- Gives intuitively reasonable results for clear profiles (see evaluation).

## 12. Evaluation process

Evaluation is qualitative and reproducible rather than metric-based. We ran the
CLI for multiple named profiles and ranking modes, saved the real output under
`outputs/`, and checked that (a) each profile's top results matched the
profile's intent, (b) switching modes changed which features dominated, and
(c) enabling diversity changed the selected list in the expected direction.
A 47-test suite (`tests/test_recommender.py`) additionally asserts scoring
direction, ranking, edge cases, mode behavior, and diversity penalties.

## 13. Profiles tested

- **High-Energy Pop** — pop, happy/energetic, high energy/danceability.
- **Acoustic Chill** — lofi/folk/jazz/ambient, chill/relaxed, high acousticness.
- **Intense Rock** — rock/metal, intense/dark, high energy/tempo.
- **EDM Workout** — edm, energetic, very high energy/tempo/danceability.

## 14. Experiment results

- **High-Energy Pop (balanced):** pop/happy/energetic tracks led; `Sunrise City`
  scored 14.07. A non-pop track (`Neon Pulse`, edm) was promoted into the top-5
  by the diversity reranker.
- **Acoustic Chill (mood_first):** the list collapsed onto quiet, acoustic
  jazz/lofi/ambient tracks; the +6.00 mood weight made mood the deciding factor.
- **Intense Rock (genre_first):** rock/metal tracks pulled far ahead (~11 vs ~5)
  because of the +6.00 genre weight.
- **Mode comparison:** `energy_focused` promoted `Seoul Nights` to #1;
  `genre_first` pushed the off-genre `Neon Pulse` to #5 (6.25) while `mood_first`
  raised it to #4 (12.57).
- **Diversity:** with diversity off, the top-5 was four pop songs then one edm;
  with diversity on, additional pop tracks took small penalties and `Neon Pulse`
  rose above the fourth pop song.

## 15. Profile-output comparisons

The three headline profiles share the same catalog and scoring engine yet have
essentially non-overlapping top-5 lists. This is the signature of content-based
filtering: the **profile**, not the catalog, drives the outcome. Pop selects
bright/danceable/mainstream tracks; Acoustic Chill selects quiet/acoustic/
instrumental tracks; Intense Rock climbs the genre-and-intensity axis.

## 16. Limitations and biases

- Tiny synthetic catalog (28 songs); conclusions do not generalize.
- Hand-crafted weights, not learned from data.
- No real listening history and no collaborative filtering.
- Simplified single-label mood/genre categorization that cannot capture the
  full range or cultural nuance of real musical taste.
- Synthetic metadata (popularity, audio features) invented by hand.
- Explanations reveal the rules but do **not** prove recommendation *quality*.

## 17. Filter-bubble risk

Content-based scoring inherently recommends "more of the same," which can trap a
user in a narrow slice of taste. Rewarding `popularity` compounds this by
nudging toward mainstream tracks. The diversity reranker is a partial mitigation
(it penalizes artist and genre repetition), but it only reshuffles within the
already-similar candidate set — it cannot introduce genuinely novel taste.

## 18. Fairness discussion

The diversity reranker reduces repetition, but it is a heuristic and does **not
guarantee fairness**. It does not model artist equity, catalog representation,
cultural balance, or exposure fairness, and the popularity signal can
systematically disadvantage less-popular or non-English-language tracks. Fair
recommendation would require explicit fairness objectives and measurement, not
just an anti-repetition penalty.

## 19. Improvement ideas

1. **Learn weights from feedback** (likes/skips) instead of hand-tuning.
2. **Add larger, multilingual catalogs** and multi-label genres/moods.
3. **Use embeddings or real audio features** rather than scalar attributes;
   optionally add user history and a collaborative-filtering signal.
4. **Measure diversity and relevance quantitatively** and run human evaluation.

## 20. Personal reflection

The most useful lesson was how ordinary a recommendation is underneath: measure
closeness, weight it, sort. Forcing the model to explain itself made the design
choices — and their biases — impossible to hide, and building the diversity
reranker made the relevance-vs-variety tradeoff tangible rather than abstract.
Mitigating a filter bubble turned out to be a dial you tune with eyes open, not
a feature you simply switch on.
