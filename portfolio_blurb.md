# Portfolio Blurb — VibeTrace AI

**Project title:** VibeTrace AI — an explainable, retrieval-grounded music
discovery copilot

**GitHub:** https://github.com/iampenuel/vibetrace-ai

**Technologies:** Python, scikit-learn (TF-IDF, Logistic Regression), pandas,
Streamlit, pytest, joblib, tabulate; a local retrieval-augmented, agentic
architecture with no external LLM or paid API key.

---

## Description (100–150 words)

VibeTrace AI turns a natural-language listening goal — "calm low-energy music for
late-night studying" — into transparent, evidence-grounded song recommendations.
It classifies intent with a specialized local classifier, retrieves supporting
evidence from a song catalog, a knowledge base, and sample listening profiles,
plans an intent-specific workflow, and reuses an explainable Project 3 scoring
engine to rank diverse songs. A grounded composer cites an evidence ID for every
claim, and a verifier confirms each citation exists, scores are ordered, and no
unsupported or medical claims slip through before the answer is returned.
Guardrails handle empty, over-long, out-of-domain, and explicit-content requests.
An evaluation harness reports reliability metrics (100% grounding and guardrail
pass rates across 14 cases), and 117 automated tests keep the pipeline honest —
all running locally and offline.

---

## What this project says about me as an AI engineer

I build AI *systems*, not just prompts. VibeTrace AI shows that I can compose
intent classification, retrieval, planning, and verification into one coherent
pipeline; measure reliability instead of asserting it; and hold a hard line on
what a system may and may not claim. I care about reproducibility (fixed seeds,
deterministic ranking, no hidden secrets), about honesty (I fixed a classifier
that lost to its own baseline rather than hide it), and about safety (guardrails,
grounding checks, and a no-health-claims verifier). Most of all, I treat
verification as a first-class feature — an adversarial check that keeps the rest
of the system trustworthy.
