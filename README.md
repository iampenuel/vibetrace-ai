# 🎧 VibeTrace AI

**An explainable, retrieval-grounded music discovery copilot.**

VibeTrace AI takes a natural-language listening goal ("calm low-energy music for
late-night studying"), classifies the intent, retrieves supporting evidence from
a song catalog and a knowledge base, plans a workflow, ranks diverse songs by
reusing a transparent Project 3 scoring engine, composes a grounded explanation
with citations, verifies that every claim is backed by evidence, applies
guardrails, and returns a transparent recommendation with a confidence score.

> **No paid API key is required.** The entire required system runs locally and
> offline after `pip install`. There is no OpenAI / Anthropic / Gemini dependency.

---

## Why this problem matters

Most recommender systems are black boxes: they tell you *what* to play but never
*why*, and they can quietly overstate confidence or drift into claims they can't
support. VibeTrace AI is a small, honest counter-example — a system that shows
its reasoning at every step, cites the evidence behind each recommendation,
refuses out-of-scope or unsafe requests, and reports a heuristic confidence
rather than pretending to certainty. It is a study in *applied, responsible AI
engineering* rather than raw recommendation accuracy.

---

## Original project

**Base project: Project 3 — Music Recommender Simulation.**

> VibeTrace AI extends my Module 3 project, Music Recommender Simulation. The
> original system loaded a structured song catalog, converted user preferences
> into explainable feature scores, ranked songs, and produced transparent
> recommendation reasons, with multiple named profiles and ranking modes. This
> final project adds natural-language intent recognition, retrieval from multiple
> sources, an agentic planning and verification pipeline, guardrails, trace
> logging, and structured reliability evaluation.

### How Project 4 extends Project 3

| Project 3 (preserved) | Project 4 (added) |
|---|---|
| CSV catalog loading + validation | Natural-language intent classification |
| Weighted explainable scoring (`score_song`) | Multi-source TF-IDF retrieval (songs + docs + history) |
| Ranking + diversity reranking | Agentic planner / executor / verifier workflow |
| Strategy-pattern ranking modes | Input guardrails + safety refusals |
| Deterministic reasons | Grounded composer with evidence IDs |
| Pytest coverage | Structured JSONL traces + evaluation harness |

The Project 3 recommendation math is **reused, not duplicated**: the agent calls
`score_song`, `sort_scored`, and `diversify_scored` from `src/recommender.py`.

---

## Core capabilities

- **Query interface:** CLI (`python -m src.main`) and Streamlit UI (`app.py`),
  both backed by the *same* `VibeTraceAgent`.
- **Input guardrails:** empty/whitespace, over-length, out-of-domain, invalid
  top-k, explicit-content enforcement (hard filter), low-confidence fallback.
- **Specialized intent classifier:** local TF-IDF + Logistic Regression over 8
  intents, with a keyword baseline for comparison.
- **Agent planner:** intent-specific high-level plans (compare ≠ discover ≠
  out-of-scope).
- **Multi-source retriever:** songs, knowledge documents, and sample listening
  profiles in one TF-IDF space, each result carrying a stable evidence ID.
- **Recommendation engine:** the Project 3 scorer, extended with a retrieval
  relevance bonus and diversity reranking.
- **Grounded composer:** deterministic answers with evidence citations.
- **Verifier:** checks evidence existence, catalog membership, score ordering,
  grounding, explicit compliance, and safety.
- **Structured logging + evaluation harness** for reliability metrics.

---

## Applied-AI feature overview

### RAG — multi-source retrieval (required + stretch)

`MultiSourceRetriever` indexes three source types into one TF-IDF vector space:

- **Song catalog** → `[song:12]`
- **Knowledge documents** (`knowledge/*.md`, split by `## section`) → `[doc:listening_contexts.md#studying]`
- **Sample listening history** → `[history:night_owl_coder]`

Retrieval runs **before** composition and changes behavior in two concrete ways:
retrieved knowledge passages are cited as context, and a retrieval-relevance
signal is blended into song scores (`base_score + 2.5 × similarity`). Disabling
retrieval (`--no-retrieval`) removes both — see
[`outputs/retrieval_comparison.txt`](outputs/retrieval_comparison.txt).

### Agentic planner / executor / verifier (stretch)

`VibeTraceAgent.run` executes:

```
validate → classify → plan → retrieve → build preferences → rank → diversify → compose → verify → log
```

Different intents produce different plans. An `out_of_scope` request never runs
ranking; a `compare` request identifies two candidates and compares them instead
of ranking the full catalog. The **verifier** runs before the answer is returned
and can downgrade confidence and add warnings if grounding fails.

### Specialized intent classifier (stretch)

A local TF-IDF (word 1–2 grams **+** character 3–5 grams) + Logistic Regression
model trained on `data/intent_training.json` (192 hand-written synthetic
examples, 8 intents). A keyword baseline is included for comparison. This is a
**small classroom specialization experiment, not a production language model.**

### Evaluation harness (stretch)

`scripts/evaluate_system.py` runs 14 predefined cases and reports intent
accuracy, retrieval hit rate, end-to-end pass rate, guardrail pass rate,
grounding pass rate, average confidence, and error count, with documented
critical thresholds.

---

## Reliability & guardrails

Confidence is a **heuristic**, not a calibrated probability. It blends four
transparent factors with fixed weights: intent-classifier confidence (0.30),
top retrieval similarity (0.20), score separation between the top two picks
(0.20), and the verifier pass rate (0.30). Guardrails fail gracefully with a
useful message and are recorded in the trace; explicit-content requests are
hard-filtered, not merely penalized.

---

## Architecture overview

```
User → CLI / Streamlit
     → Guardrails → Intent Classifier → Planner
     → Multi-Source Retriever (Songs | Knowledge Docs | User History)
     → Recommendation Engine → Diversity Reranker
     → Grounded Composer → Verifier (can warn / downgrade) → Output
     → Structured Logger → Evaluation Harness → Human review
```

Mermaid source: [`diagrams/architecture.mmd`](diagrams/architecture.mmd).

## Project structure

```
vibetrace-ai/
├── README.md  model_card.md  ai_interactions.md  PRESENTATION.md  portfolio_blurb.md
├── requirements.txt  .gitignore  app.py
├── data/            songs.csv, intent_training.json, evaluation_cases.json, sample_user_history.json
├── knowledge/       genre_guide.md, mood_and_energy_guide.md, listening_contexts.md, system_limits.md
├── diagrams/        architecture.mmd
├── logs/            agent_trace_examples.jsonl (committed samples)
├── outputs/         demo_*.txt, retrieval_comparison.txt, specialization_comparison.txt, evaluation_summary.txt
├── scripts/         evaluate_system.py, generate_demo_evidence.py, train_intent_classifier.py
├── src/             agent.py, recommender.py, models.py, intent_classifier.py, retriever.py,
│                    planner.py, composer.py, verifier.py, guardrails.py, logging_utils.py, main.py
└── tests/           test_recommender, test_intent_classifier, test_retriever, test_guardrails,
                     test_verifier, test_agent, test_evaluation_harness
```

---

## Setup

```bash
cd vibetrace-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## CLI usage

```bash
python -m src.main --query "I need upbeat, clean songs for a 30-minute workout" --top-k 3
python -m src.main --query "Give me calm low-energy music for late-night studying" --top-k 3 --show-trace
python -m src.main --query "Compare Library Rain and Midnight Coding for studying"
python -m src.main --query "Surprise me, but avoid repeating artists" --mode balanced
python -m src.main --query ""            # guardrail demo
```

Options: `--query --top-k --mode --profile --show-trace --no-retrieval --no-diversity --log-path --json`.

## Streamlit usage

```bash
python -m streamlit run app.py
```

## Evaluation usage

```bash
python scripts/train_intent_classifier.py            # specialization experiment
python scripts/evaluate_system.py | tee outputs/evaluation_summary.txt
python scripts/generate_demo_evidence.py             # regenerate demo outputs + traces
```

---

## Sample interactions (real execution output)

### 1. Study request (grounded, with retrieved context)

```text
QUERY: 'Give me calm, low-energy music for late-night studying and explain each choice.'
Intent: study  (classifier confidence 0.56)
Status: ok   |   System confidence: 0.50
Plan: validate_input -> classify_intent -> retrieve_catalog -> retrieve_context -> build_preferences -> rank_candidates -> apply_diversity -> compose_grounded_answer -> verify_output

| # | Title              | Artist         | Genre   | Mood    | Score | Evidence  |
|---|--------------------|----------------|---------|---------|-------|-----------|
| 1 | Library Rain       | Paper Lanterns | lofi    | chill   | 9.77  | [song:4]  |
| 2 | Sunday Slowdown    | Slow Stereo    | jazz    | relaxed | 9.67  | [song:25] |
| 3 | Spacewalk Thoughts | Orbit Bloom    | ambient | chill   | 9.65  | [song:6]  |

1. Library Rain — Paper Lanterns (score 9.77)
   Why: mood match: chill (+2.00); energy similarity: 0.95 (+1.90); acousticness similarity: 0.99 (+1.48); instrumentalness similarity: 0.90 (+0.90); retrieval relevance: 0.10 (+0.26)
   Evidence: [song:4]

Context used: [doc:listening_contexts.md#reflective-listening] [doc:mood_and_energy_guide.md#energy] [doc:listening_contexts.md#studying]
Note: This catalog is synthetic and small; scores reflect transparent feature matching, not a promise of enjoyment.
Confidence: 0.50 (heuristic, not a calibrated probability)
Verifier: PASSED  (pass rate 1.00)
```

### 2. Guardrail (out-of-domain refusal)

```text
QUERY: 'Can you diagnose my anxiety and recommend medication?'
Intent: out_of_scope  (classifier confidence 0.80)
Status: guardrail   |   System confidence: 0.00
Plan: validate_input -> classify_intent -> return_guardrail_response -> verify_output
VibeTrace AI only helps with music discovery — finding, comparing, and explaining
songs for moods and activities. I can't help with that request, but I can suggest
music for studying, working out, relaxing, and more.
Verifier: PASSED  (pass rate 1.00)
```

Full outputs: [`outputs/demo_study.txt`](outputs/demo_study.txt),
[`outputs/demo_discovery.txt`](outputs/demo_discovery.txt),
[`outputs/demo_comparison.txt`](outputs/demo_comparison.txt),
[`outputs/demo_guardrail.txt`](outputs/demo_guardrail.txt).

### Retrieved-evidence behavior

Every recommendation carries a `[song:id]` evidence ID; when retrieval is on, the
answer also cites `[doc:...]` knowledge passages and, with a profile,
`[history:...]`. The verifier confirms all cited IDs exist before returning.

### High-level trace example (no hidden chain-of-thought)

```json
{
  "request_id": "req_0002_1726caa8",
  "intent": "study",
  "intent_confidence": 0.5615,
  "plan": ["validate_input","classify_intent","retrieve_catalog","retrieve_context","build_preferences","rank_candidates","apply_diversity","compose_grounded_answer","verify_output"],
  "components_called": ["guardrails","intent_classifier","planner","retriever","recommender","composer","verifier"],
  "retrieved_evidence_ids": ["doc:listening_contexts.md#reflective-listening","doc:mood_and_energy_guide.md#energy","doc:listening_contexts.md#studying"],
  "recommendations": [{"id":4,"title":"Library Rain","score":9.766}, {"id":25,"title":"Sunday Slowdown","score":9.665}],
  "verifier_checks": {"non_empty_answer": true, "evidence_ids_exist": true, "songs_exist": true, "scores_ordered": true, "recommendations_grounded": true, "explicit_respected": true},
  "verifier_passed": true,
  "confidence": 0.5037,
  "status": "ok"
}
```

More samples: [`logs/agent_trace_examples.jsonl`](logs/agent_trace_examples.jsonl).

---

## Reliability summary (actual `scripts/evaluate_system.py` output)

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

Critical thresholds (documented): all safety/guardrail cases must pass;
end-to-end ≥ 80%; grounding = 100% for successful answers.

### Retrieval before/after comparison

With retrieval **off**, the study answer cites no `[doc:...]` context and applies
no retrieval-relevance bonus. With retrieval **on**, it cites
`[doc:listening_contexts.md#studying]` and `[doc:mood_and_energy_guide.md#energy]`
and blends retrieval relevance into scoring — richer, evidence-grounded output.
Full ablation: [`outputs/retrieval_comparison.txt`](outputs/retrieval_comparison.txt).

### Specialization: baseline vs. trained model (4-fold CV, seed=42)

```text
Keyword baseline        :  58.85%
Specialized classifier  :  74.48%
Improvement             : +15.62 points
```

The trained model generalizes to paraphrases the keyword rules miss (biggest
gains on `compare`, `explain`, `diversify`, `relax`). Full report:
[`outputs/specialization_comparison.txt`](outputs/specialization_comparison.txt).

---

## Testing summary

**117 automated tests pass** across recommendation, classifier, retriever,
guardrails, verifier, end-to-end agent, and evaluation-harness suites — including
all preserved Project 3 tests.

```bash
python -m compileall src scripts tests app.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v | tee test_results.txt
```

```text
======================= 117 passed, 18 warnings in 6.45s =======================
```

Full log: [`test_results.txt`](test_results.txt).

---

## Design decisions and tradeoffs

- **Local hybrid over external LLM.** A fully local, deterministic pipeline is
  reproducible, gradeable without secrets, and less prone to unsupported claims.
  Tradeoff: less flexible language understanding than a large model.
- **TF-IDF retrieval.** Transparent and fast, but misses semantic paraphrases a
  dense embedding model would catch.
- **Word + character n-grams** in the classifier: robust on a tiny dataset;
  still small and classroom-scale.
- **Verifier-in-the-loop.** Cheap insurance against ungrounded or unsafe output;
  it can only downgrade/warn, never fabricate.
- **Reuse, not rewrite.** The agent calls the Project 3 scorer directly.

## Limitations

Small synthetic catalog; handcrafted scoring weights; synthetic intent data;
English-only; imperfect genre/mood proxies; no real user feedback; diversity may
reduce raw relevance; TF-IDF misses paraphrases; confidence is heuristic, not
calibrated; retrieved documents were written for this project.

## Responsible use

VibeTrace AI is **not** Spotify or a production recommender, and makes **no**
medical, psychological, or emotional-health claims. Mood labels describe musical
character, not a listener's state. It uses no real private listening data — only
optional synthetic sample profiles. See
[`knowledge/system_limits.md`](knowledge/system_limits.md).

## Presentation & portfolio

- Presentation plan: [`PRESENTATION.md`](PRESENTATION.md)
- Portfolio blurb: [`portfolio_blurb.md`](portfolio_blurb.md)

## Portfolio reflection

Building VibeTrace AI meant thinking like an *AI systems engineer*, not just a
model user: composing intent classification, retrieval, planning, and
verification into one honest pipeline; making reliability measurable; and being
disciplined about what the system may and may not claim. The most valuable
component turned out to be the verifier — a small adversarial check that keeps
the rest of the system honest.

## Model card

Full details, evaluation results, biases, misuse analysis, and the AI
collaboration reflection: [`model_card.md`](model_card.md).
