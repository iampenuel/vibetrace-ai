"""Agent planner: turns a classified intent into a high-level, ordered plan.

The plan is a list of named steps the executor (the agent) will carry out.
Different intents produce meaningfully different plans — for example an
``out_of_scope`` request never runs recommendation ranking, while ``compare``
identifies two candidates instead of ranking a full list.

These are high-level decision records, not hidden reasoning: each step names a
component or action, nothing more.
"""

from __future__ import annotations

from typing import List

# Reusable step vocabulary (kept as plain strings for transparent logging).
VALIDATE = "validate_input"
CLASSIFY = "classify_intent"
RETRIEVE_CATALOG = "retrieve_catalog"
RETRIEVE_CONTEXT = "retrieve_context"
RETRIEVE_HISTORY = "retrieve_history"
BUILD_PREFERENCES = "build_preferences"
RANK = "rank_candidates"
DIVERSIFY = "apply_diversity"
IDENTIFY_PAIR = "identify_two_candidates"
IDENTIFY_ONE = "identify_target_song"
COMPARE = "compare_candidates"
EXPLAIN = "explain_ranking"
COMPOSE = "compose_grounded_answer"
VERIFY = "verify_output"
GUARDRAIL_RESPONSE = "return_guardrail_response"


def plan_for_intent(intent: str, use_retrieval: bool = True,
                    use_diversity: bool = True, use_history: bool = False) -> List[str]:
    """Return the ordered high-level plan for ``intent``.

    Toggles (retrieval / diversity / history) are reflected in the plan so the
    trace honestly shows what will run.
    """
    plan: List[str] = [VALIDATE, CLASSIFY]

    if intent == "out_of_scope":
        # Safety first: no ranking work for out-of-domain requests.
        plan.append(GUARDRAIL_RESPONSE)
        plan.append(VERIFY)
        return plan

    # Retrieval steps (shared by the recommendation-style intents).
    plan.append(RETRIEVE_CATALOG)
    if use_retrieval:
        plan.append(RETRIEVE_CONTEXT)
    if use_history:
        plan.append(RETRIEVE_HISTORY)

    if intent == "compare":
        plan += [IDENTIFY_PAIR, BUILD_PREFERENCES, COMPARE, COMPOSE, VERIFY]
        return plan

    if intent == "explain":
        plan += [IDENTIFY_ONE, BUILD_PREFERENCES, RANK, EXPLAIN, COMPOSE, VERIFY]
        return plan

    # discover / study / workout / relax / diversify all rank candidates.
    plan.append(BUILD_PREFERENCES)
    plan.append(RANK)
    # diversify intent always diversifies; others honor the toggle.
    if intent == "diversify" or use_diversity:
        plan.append(DIVERSIFY)
    plan += [COMPOSE, VERIFY]
    return plan
