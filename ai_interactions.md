# AI Interactions Log

This project was completed with the **Claude Code** agent (Opus 4.8) working
autonomously across the whole workflow. The notes below are a truthful record
of what the agent did and how the work was verified.

---

## Agentic Workflow (SF8)

### What task did the agent perform?

Complete CodePath AI110 Project 3 ("Music Recommender Simulation") end to end:
fork and clone the starter, inspect it, expand and validate the dataset,
implement the recommender (models, scoring, ranking modes, diversity), build a
CLI and an optional Streamlit UI, write a test suite, run real evaluation
experiments, write the README / model card / this log, and commit and push to
the public fork.

### Prompts used (excerpt)

The main instruction (summarized):

> "Complete CodePath AI110 Project 3 autonomously from repository setup through
> final push. Earn every rubric point (21 required + 8 stretch). Every claimed
> feature must be functional, tested, demonstrated with real command output,
> documented accurately, and committed through meaningful Git history. Inspect
> before editing; never modify or push to the CodePath upstream; never
> force-push; do not commit `.venv`/caches/secrets; do not fabricate CLI or test
> output; keep scoring deterministic and explainable; preserve the starter
> tests. Complete all four stretch categories after the core system passes."

### What the agent changed / created

- **Repository:** forked `codepath/...-starter` to `iampenuel/...-starter`
  (via `gh repo fork --clone=false`), cloned the fork into
  `music-recommender/`, added a read-only `upstream` remote, and disabled push
  to upstream as a safety measure.
- **Data:** expanded `data/songs.csv` from 10 → 28 fictional songs (16 genres,
  11 moods, 6 languages, 7 decades) with 7 new scored attributes.
- **Code:** rewrote `src/recommender.py` (dataclasses, validating loader,
  scoring core, Strategy modes, diversity reranker), reworked `src/main.py`
  into an argparse CLI, added `src/profiles.py` and an optional `src/app.py`
  (Streamlit).
- **Tests:** grew `tests/test_recommender.py` from 2 → 47 tests.
- **Docs:** rewrote `README.md`, `model_card.md`, and this file.
- **Config:** added `tabulate` to `requirements.txt`; broadened `.gitignore`.

### Commands run (all actually executed)

- Environment: `git --version`, `gh auth status`, `gh repo fork ...`,
  `git clone ...`, `git remote -v`.
- Setup: `python3 -m venv .venv`, `pip install -r requirements.txt`.
- Baseline + checks: `python -m compileall src tests`,
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v`, `python -m src.main`.
- Evaluation: `python -m src.main --profile ... --mode ... --top-k 5 | tee
  outputs/...`, `--compare-modes`, `--no-diversity`, `--all-profiles`.
- Streamlit smoke test: launched headless, confirmed `HTTP 200` and
  `/_stcore/health = ok`.

### Verification performed

- **Automated checks run:** `compileall` clean; **47/47 pytest tests pass**
  (captured in `test_results.txt`).
- **Output inspected:** every README/model-card figure was copied from real
  captured runs in `outputs/`; the agent re-read the outputs before quoting
  them.
- **Issues the agent found and corrected during the build:**
  - The starter `src/main.py` used `from recommender import ...`, which breaks
    under `python -m src.main`; fixed with package-qualified imports plus a
    fallback.
  - Verified the starter tests still pass against the new code (the OOP
    `Recommender.recommend` returns `List[Song]` and preserves the pop/happy
    ordering the starter test expects).
  - Confirmed diversity actually changes selection (not just scores) by
    comparing `--no-diversity` against the default run.
- **Remaining human review recommendation:** a human should skim the dataset
  for tone/label quality and confirm the fork URL is submitted in CodePath.
  The agent did not claim any manual human verification that did not occur.

---

## Design Pattern (SF10)

### Which design pattern?

The **Strategy pattern** for ranking modes.

### Why it was selected

The four ranking modes differ only in *how much each feature matters* — the
scoring math is identical. Encoding that as one large `if mode == ...`
conditional would duplicate the scoring loop and grow unmanageably with each new
mode. Strategy lets each mode be a small, self-contained set of weights while a
single scoring core stays shared, so the modes cannot silently diverge from one
another.

### How AI helped

The agent proposed Strategy over alternatives (a flat weight dict, or nested
conditionals), argued that a beginner can explain "each mode is just a different
set of weights," and implemented the minimal version: an abstract base plus four
tiny concrete classes and a registry — deliberately avoiding heavier machinery
(no config files, no dynamic plugin loading) to keep it readable.

### How the pattern appears in the final code (`src/recommender.py`)

- `ScoringStrategy(ABC)` — abstract base declaring a `weights` property.
- `BalancedStrategy`, `GenreFirstStrategy`, `MoodFirstStrategy`,
  `EnergyFocusedStrategy` — concrete strategies (each just returns a weight
  dict).
- `_STRATEGIES` registry + `get_strategy(mode)` factory, which raises a clear
  `ValueError` on unknown modes.
- `_score_song_core(prefs, song, strategy)` — the single shared scoring
  routine used by both `score_song` (functional) and `Recommender` (OOP).

### What was simplified to stay beginner-readable

Strategies hold only weights (no per-strategy custom logic), and all similarity
math lives in one place, so the difference between modes is literally a table of
numbers.

---

## Additional Song Attributes via Agentic AI

### New CSV columns

`popularity` (0–100), `release_decade`, `instrumentalness` (0–1),
`speechiness` (0–1), `liveness` (0–1), `language`, `explicit`. (`duration_seconds`
was also added but is intentionally **not** scored.)

### How scoring uses them

- `popularity`, `instrumentalness`, `speechiness`, `liveness` → numeric
  closeness (`1 - |target - value| / range`) when the user sets a target.
- `release_decade` → closeness that fades over ~4 decades.
- `language` → exact-match bonus.
- `explicit` → a **negative** contribution when a song is explicit and the user
  disallows explicit content.

Each attribute is wired into `_NUMERIC_FEATURES` / the categorical and explicit
branches of `_score_song_core`, and each has a mode weight, so the attributes
are functional rather than decorative.

### Validation performed

`load_songs` checks required columns, parses numeric/boolean types, enforces
0–1 and 0–100 ranges, requires unique ids, and normalizes whitespace, raising a
clear `DatasetError` otherwise.

### Row-count / schema check (actually run)

```
rows: 28
cols: 18
genres: 16   moods: 11   languages: 6   decades: 7
unique ids: True
```

### Relevant tests

`test_new_attributes_present`, `test_numeric_types`, `test_boolean_field_loads`,
`test_feature_ranges`, `test_popularity_preference_changes_score`,
`test_decade_preference_changes_score`, `test_language_preference_changes_score`,
`test_explicit_compatibility_changes_score`, and the malformed-data loaders
(`test_load_bad_numeric_raises`, `test_load_out_of_range_raises`,
`test_load_duplicate_id_raises`, `test_whitespace_normalized`).
