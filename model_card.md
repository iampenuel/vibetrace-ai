# Model Card — VibeTrace AI

## 1. System name and version
**VibeTrace AI**, version 1.0 — an explainable, retrieval-grounded music
discovery copilot. This is a rule-and-classifier hybrid *system*, not a single
trained model. No paid API key is required; it runs locally and offline.

## 2. Base project
Extends **Project 3 — Music Recommender Simulation**. The Project 3 explainable
scoring engine (`src/recommender.py`) is reused directly by the agent.

## 3. Intended purpose
Educational demonstration of applied, responsible AI: turning a natural-language
listening goal into transparent, evidence-grounded, verified song
recommendations over a small synthetic catalog.

## 4. Intended users
Students, instructors, and portfolio reviewers evaluating an applied-AI project.
Secondary: anyone learning how RAG, agentic workflows, guardrails, and
verification fit together in a small, readable codebase.

## 5. Non-intended uses
Not for real-world music-service deployment, not a source of truth about real
songs or artists (the catalog is fictional), and **not** for any medical,
psychological, or emotional-health decision-making.

## 6. System architecture
CLI / Streamlit → guardrails → intent classifier → planner → multi-source
retriever → recommendation engine → diversity reranker → grounded composer →
verifier → structured logger → evaluation harness. See
`diagrams/architecture.mmd`.

## 7. Data sources
- `data/songs.csv` — 28 synthetic songs with audio-style features (from P3).
- `knowledge/*.md` — four original educational documents (genre, mood/energy,
  listening contexts, system limits), chunked by `## section`.
- `data/sample_user_history.json` — three fictional listening profiles.
- `data/intent_training.json` — 192 synthetic labeled intent examples.

## 8. Retrieval method
Single TF-IDF vector space over songs, knowledge chunks, and history profiles;
cosine similarity; deterministic tie-breaking by evidence ID. Each result is an
`Evidence` object with a stable citation (`[song:id]`, `[doc:file#section]`,
`[history:name]`). Missing sources degrade gracefully.

## 9. Intent classifier
TF-IDF (word 1–2 grams + char 3–5 grams) + Logistic Regression
(`C=8`, balanced classes, seed=42) over 8 intents. A keyword baseline is included
for comparison. **A small classroom specialization experiment, not a production
language model.** Low-confidence predictions (< 0.45) trigger a safe balanced
fallback and a user-facing warning.

## 10. Recommendation method
The Project 3 weighted, explainable scorer (`score_song`) computes a base score
and truthful per-feature reasons. When retrieval is enabled, a retrieval
relevance bonus (`2.5 × cosine similarity`) is blended in. Diversity reranking
(artist-repeat and genre-concentration penalties) is reused from Project 3.

## 11. Agentic workflow
`VibeTraceAgent.run` plans a per-intent sequence of steps and executes real
components. `out_of_scope` skips ranking entirely; `compare` identifies and
scores two candidates; `explain` scores one target. The verifier gates the final
answer and can downgrade confidence and add warnings.

## 12. Confidence scoring
Heuristic blend (documented weights): intent confidence 0.30, top retrieval
similarity 0.20, top-two score separation 0.20, verifier pass rate 0.30, clamped
to [0, 1]. **This is a transparent heuristic, not a calibrated probability of
user satisfaction.**

## 13. Guardrails
Empty/whitespace input, over-length truncation, out-of-domain detection, invalid
top-k, explicit-content **hard filtering**, and low-confidence fallback. All
decisions are recorded in the trace; users never see raw stack traces.

## 14. Evaluation process
`scripts/evaluate_system.py` runs 14 predefined cases (`data/evaluation_cases.json`)
through the real agent and computes intent accuracy, retrieval hit rate,
end-to-end pass rate, guardrail pass rate, grounding pass rate, average
confidence, and error count. Critical thresholds: all safety cases pass;
end-to-end ≥ 80%; grounding = 100% for successful answers.

## 15. Actual evaluation results

Reliability harness (`outputs/evaluation_summary.txt`):

```text
Intent accuracy            : 100.00%  (11/11)
Retrieval evidence hit rate: 100.00%  (2/2)
End-to-end pass rate       : 100.00%  (14/14)
Guardrail pass rate        : 100.00%  (4/4)
Grounding pass rate        : 100.00%  (10/10)
Average heuristic confidence:  0.59
Errors                     : 0
OVERALL: PASS
```

Specialization experiment (4-fold CV, seed=42,
`outputs/specialization_comparison.txt`):

```text
Keyword baseline        :  58.85%
Specialized classifier  :  74.48%   (+15.62 points)
```

Automated tests: **117 passed** (`test_results.txt`).

## 16. Strengths
Fully transparent and explainable; every recommendation cites evidence; verifier
prevents ungrounded or unsafe answers; reproducible (fixed seeds, deterministic
ranking); no external services or secrets; reuses proven Project 3 math.

## 17. Limitations
Small synthetic catalog (28 songs); handcrafted scoring weights; synthetic
intent-training examples; English-focused queries; imperfect genre/mood proxies;
no real user feedback; diversity heuristics can reduce raw relevance; TF-IDF
misses semantic paraphrases; confidence is heuristic, not calibrated; the
retrieved documents were written specifically for this project.

## 18. Biases
The catalog's genre/mood/energy balance reflects the author's choices, so
recommendations inherit that distribution. Intent examples are English and
reflect one writer's phrasing. Keyword and TF-IDF features favor exact/character
overlap over meaning, which can disadvantage unusual phrasings or dialects.
Popularity as a feature can create a mild rich-get-richer effect.

## 19. Misuse risks
- Treating subjective recommendations as objective truth.
- Interpreting the heuristic confidence as certainty.
- Using synthetic catalog data as evidence about real artists.
- Reading mood labels ("calm", "happy") as mental-health inferences.
- Over-relying on a narrow catalog as if it were comprehensive.

## 20. Mitigations
Limitations shown directly in outputs; evidence IDs on every claim; guardrails
and out-of-scope refusals; explicit-content hard filtering; no health claims (a
verifier check bans medical/therapeutic claim patterns); no real private user
data; confidence warnings on low-confidence runs; recommended human review
before any downstream use.

## 21. Human oversight
Automated verification runs on every response; output files are inspected
programmatically; the evaluation harness enforces critical thresholds. **A final
human review is recommended before submission.** The system does not claim a
human manually reviewed every generated output.

## 22. Privacy
No real user data is collected or stored. The only "history" is three fictional
sample profiles, used solely for optional retrieval grounding. Traces store
short, sanitized input summaries and high-level decision records — never hidden
reasoning.

## 23. Future improvements
Dense-embedding retrieval for paraphrase robustness; calibrated confidence;
larger and more diverse catalog and intent data; multilingual support; real
user-feedback signals; an optional external-LLM adapter that composes phrasing
*only* on top of the same grounded evidence.

## 24. AI collaboration reflection

This project was built with AI-assisted planning (Claude Code). Three
architectures were considered before implementation (see `ai_interactions.md`):
(A) external-LLM-only, (B) retrieval + deterministic composer only, and (C) a
local hybrid. Option C was selected.

**A helpful suggestion that was accepted:** adding a **verifier** that checks
evidence IDs, catalog membership, score ordering, and guardrail compliance
*before* returning the final answer. This became the backbone of the system's
reliability and directly produces the grounding metric.

**A flawed suggestion that was modified/rejected:** using an external LLM for
every response. That would have required private credentials, hurt
reproducibility, and increased the risk of unsupported claims. It was modified
into a fully local, deterministic pipeline, with any external LLM relegated to a
strictly optional future adapter. *(This was a design suggestion considered
during AI-assisted planning, not a runtime factual error.)*

**How correctness was checked:** code inspection; 117 unit and end-to-end tests;
the evaluation harness; execution logs and committed trace samples; Git diffs;
and reproducible CLI demonstrations whose real output is saved under `outputs/`.

### What are the limitations or biases?
See §17–§18. In short: a small synthetic catalog, handcrafted weights, synthetic
English-only intent data, imperfect feature proxies, no real feedback, and a
heuristic (uncalibrated) confidence.

### Could the system be misused?
See §19–§20. Chiefly by mistaking subjective, synthetic recommendations for
objective truth or reading mood labels as health signals — mitigated by visible
limitations, evidence IDs, guardrails, and the no-health-claims verifier check.

### What surprised me in reliability testing?
Two real observations from the runs above: (1) the *keyword baseline* was
initially very strong because its rules matched the training vocabulary, so an
honest comparison required expanding the dataset with paraphrases and switching
to cross-validation — after which the trained model led by ~15 points; and
(2) a broadly-scoped banned-phrase check first flagged the system's **own honest
disclaimer** ("treat scores as…"), which taught me to make safety checks specific
to *claims* rather than vocabulary.
