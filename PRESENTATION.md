# VibeTrace AI — Presentation Plan (5–7 minutes)

Speaker notes are concise talking points, not a script. Times are approximate.

---

## 1. Title (0:20)
**VibeTrace AI — an explainable, retrieval-grounded music discovery copilot.**
- Built for CodePath AI110 Project 4; evolves my Project 3 recommender.
- Runs fully locally, no paid API key.

## 2. Problem (0:40)
- Recommenders usually tell you *what*, never *why*, and can overstate certainty.
- Goal: a small, honest system that shows its reasoning, cites evidence, refuses
  out-of-scope requests, and reports calibrated-*style* confidence.
- This is an applied **responsible-AI** exercise, not a Spotify clone.

## 3. Original Project 3 (0:40)
- Loaded a validated song catalog, turned preferences into weighted feature
  scores, ranked songs, applied diversity, and explained every pick.
- VibeTrace AI **reuses that scoring engine directly** — no rewrite.

## 4. Architecture (1:00)
- Show `diagrams/architecture.mmd`.
- Flow: guardrails → intent classifier → planner → multi-source retriever →
  recommender → diversity → grounded composer → verifier → logger → evaluation.
- Same agent powers both the CLI and the Streamlit UI.

## 5. RAG demonstration (1:00)
- Three retrieval sources in one TF-IDF space: `[song:id]`, `[doc:file#section]`,
  `[history:name]`.
- Live: run a study query; point out the cited `[doc:listening_contexts.md#studying]`.
- Show the ablation: `outputs/retrieval_comparison.txt` — with vs. without
  retrieval changes citations and ranking.

## 6. Agentic workflow (1:00)
- Different intents → different plans (compare ≠ discover ≠ out-of-scope).
- The **verifier** gates the answer: checks evidence IDs, catalog membership,
  score order, grounding, explicit compliance, and banned claims; it can
  downgrade confidence and warn.
- Show a committed trace from `logs/agent_trace_examples.jsonl` — high-level
  only, no hidden chain-of-thought.

## 7. Reliability results (0:50)
- Evaluation harness (14 cases): intent 100%, retrieval hit 100%, end-to-end
  100%, guardrail 100%, grounding 100%, avg confidence 0.59, 0 errors.
- Specialization (4-fold CV): keyword baseline 58.85% → trained model 74.48%
  (+15.62 points).
- 117 automated tests pass.

## 8. Responsible AI (0:40)
- Guardrails + out-of-scope refusals; explicit-content hard filtering.
- No medical/psychological claims (a verifier check enforces this).
- No real private user data; limitations shown in every output.

## 9. Lessons learned (0:40)
- An honest baseline can be *too* strong — fair comparison needed more varied
  data and cross-validation.
- A safety check flagged my own disclaimer, so I made checks target *claims*, not
  vocabulary.
- Verification-in-the-loop was the highest-leverage component.

## 10. Closing (0:20)
- VibeTrace AI: transparent, grounded, verified, reproducible — applied AI done
  responsibly.
- Repo: `github.com/iampenuel/vibetrace-ai`. Thank you — questions?
