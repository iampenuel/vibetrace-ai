"""VibeTrace AI agent — the planner / executor / verifier orchestrator.

``VibeTraceAgent.run`` is the single entry point shared by the CLI and the
Streamlit UI. It executes the transparent workflow:

    validate -> classify -> plan -> retrieve -> build preferences -> rank ->
    diversify -> compose -> verify -> log -> return

and returns a structured :class:`src.models.AgentResult`. Every run emits one
high-level (non-chain-of-thought) trace record.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from src import guardrails, planner
from src.composer import (
    compose_comparison_response,
    compose_explain_response,
    compose_out_of_scope_response,
    compose_recommendation_response,
)
from src.intent_classifier import IntentClassifier
from src.logging_utils import make_request_id, utc_timestamp, write_trace
from src.models import AgentResult, Evidence, Recommendation
from src.recommender import (
    diversify_scored,
    load_songs,
    reasons_to_text,
    score_song,
    sort_scored,
)
from src.retriever import MultiSourceRetriever
from src.verifier import compute_confidence, verify_output

# How strongly a query<->song retrieval similarity boosts a song's base score.
# Large enough to reorder near-ties, small enough not to override genre/mood fit.
RETRIEVAL_BONUS = 2.5

# Per-intent preference presets. These translate an intent into the numeric
# targets the Project 3 scorer understands.
_INTENT_PRESETS: Dict[str, Dict] = {
    "study": {
        "energy": 0.30, "tempo": 76, "acousticness": 0.85, "instrumentalness": 0.80,
        "valence": 0.55, "danceability": 0.40, "moods": ["chill", "focused", "relaxed"],
    },
    "relax": {
        "energy": 0.32, "tempo": 80, "acousticness": 0.80, "valence": 0.62,
        "danceability": 0.45, "moods": ["relaxed", "chill", "romantic"],
    },
    "workout": {
        "energy": 0.93, "tempo": 132, "acousticness": 0.06, "danceability": 0.88,
        "valence": 0.72, "moods": ["energetic", "intense"],
    },
    "discover": {
        "energy": 0.62, "valence": 0.70, "popularity": 72, "danceability": 0.70,
    },
    "diversify": {
        "energy": 0.60, "valence": 0.68, "popularity": 68,
    },
}
# compare / explain reuse a sub-context preset detected from the query.
_DEFAULT_PRESET = {"energy": 0.55, "valence": 0.65}


class VibeTraceAgent:
    """Loads all components once and answers queries via :meth:`run`."""

    def __init__(
        self,
        data_path: str = "data/songs.csv",
        knowledge_dir: str = "knowledge",
        history_path: str = "data/sample_user_history.json",
        model_path: str = "models/intent_model.joblib",
        training_path: str = "data/intent_training.json",
    ) -> None:
        self.songs: List[Dict] = load_songs(data_path)
        self._by_id: Dict[int, Dict] = {s["id"]: s for s in self.songs}
        self.catalog_ids = set(self._by_id)

        self.retriever = MultiSourceRetriever().build_index(
            self.songs, knowledge_dir=knowledge_dir, history_path=history_path
        )
        self.valid_evidence_ids = {d.source_id for d in self.retriever._docs}

        self.classifier = IntentClassifier.load_or_train(
            model_path=model_path, training_path=training_path
        )
        self.history_path = history_path
        self._run_counter = 0

    # -- helpers ---------------------------------------------------------
    def _catalog_genres(self) -> set:
        return {s["genre"] for s in self.songs}

    def _catalog_moods(self) -> set:
        return {s["mood"] for s in self.songs}

    def _detect_subcontext(self, query: str) -> str:
        low = query.lower()
        if any(w in low for w in ("study", "focus", "concentrat", "read", "coding")):
            return "study"
        if any(w in low for w in ("workout", "gym", "run", "cardio", "exercise", "lift")):
            return "workout"
        if any(w in low for w in ("relax", "chill", "unwind", "calm", "sooth", "reflect")):
            return "relax"
        return "discover"

    def _build_preferences(
        self, intent: str, query: str, profile: Optional[Dict]
    ) -> Tuple[Dict, Optional[bool]]:
        """Translate intent + query + optional profile into scorer preferences."""
        preset_intent = intent
        if intent in ("compare", "explain"):
            preset_intent = self._detect_subcontext(query)
        prefs: Dict = dict(_INTENT_PRESETS.get(preset_intent, _DEFAULT_PRESET))

        # Query keyword overlays: genre and mood mentions from the catalog.
        low = query.lower()
        genres = [g for g in self._catalog_genres() if g.lower() in low]
        if genres:
            prefs["genres"] = sorted(set(prefs.get("genres", [])) | set(genres))
        moods = [m for m in self._catalog_moods() if m.lower() in low]
        if moods:
            prefs["moods"] = sorted(set(prefs.get("moods", [])) | set(moods))

        # Explicit-content preference (query wins over profile default).
        explicit_pref = guardrails.detect_explicit_preference(query)

        # Optional profile overlay.
        if profile:
            prefs.setdefault("genres", [])
            prefs["genres"] = sorted(set(prefs["genres"]) | set(profile.get("top_genres", [])))
            prefs.setdefault("moods", [])
            prefs["moods"] = sorted(set(prefs["moods"]) | set(profile.get("top_moods", [])))
            if "typical_energy" in profile:
                prefs["energy"] = profile["typical_energy"]
            if explicit_pref is None and profile.get("avoid_explicit"):
                explicit_pref = False

        if explicit_pref is not None:
            prefs["allow_explicit"] = explicit_pref
        return prefs, explicit_pref

    def _load_profile(self, profile_name: Optional[str]) -> Optional[Dict]:
        if not profile_name:
            return None
        if not os.path.exists(self.history_path):
            return None
        with open(self.history_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("profiles", {}).get(profile_name)

    def _rank(
        self, prefs: Dict, mode: str, retrieved_sims: Dict[int, float],
        top_k: int, diversify: bool, allow_explicit: Optional[bool] = None,
    ) -> List[Recommendation]:
        """Score every song (P3 math), add a retrieval bonus, rank, diversify.

        When the user asked for clean music (``allow_explicit is False``),
        explicit tracks are hard-filtered out — a guardrail guarantee rather
        than the soft scoring penalty alone.
        """
        candidates = self.songs
        if allow_explicit is False:
            candidates = [s for s in self.songs if not s["explicit"]]
        scored: List[Tuple[Dict, float, List[str]]] = []
        for song in candidates:
            base, reasons = score_song(prefs, song, mode)
            sim = retrieved_sims.get(song["id"], 0.0)
            reasons = list(reasons)
            if sim > 0:
                bonus = RETRIEVAL_BONUS * sim
                reasons.append(f"retrieval relevance: {sim:.2f} (+{bonus:.2f})")
                base += bonus
            scored.append((dict(song), base, reasons))

        scored = sort_scored(scored)
        ranked = diversify_scored(scored, top_k) if diversify else scored[:top_k]

        recs: List[Recommendation] = []
        for song, score, reasons in ranked:
            recs.append(Recommendation(
                song_id=song["id"], title=song["title"], artist=song["artist"],
                genre=song["genre"], mood=song["mood"], score=round(score, 4),
                reasons=reasons_to_text(reasons),
                evidence_ids=[f"song:{song['id']}"],
            ))
        return recs

    def _score_one(self, song: Dict, prefs: Dict, mode: str) -> Recommendation:
        base, reasons = score_song(prefs, song, mode)
        return Recommendation(
            song_id=song["id"], title=song["title"], artist=song["artist"],
            genre=song["genre"], mood=song["mood"], score=round(base, 4),
            reasons=reasons_to_text(reasons), evidence_ids=[f"song:{song['id']}"],
        )

    def _find_named_songs(self, query: str) -> List[Dict]:
        low = query.lower()
        found = [s for s in self.songs if s["title"].lower() in low]
        # Preserve query order of appearance.
        found.sort(key=lambda s: low.find(s["title"].lower()))
        return found

    # -- main entry point ------------------------------------------------
    def run(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "balanced",
        profile: Optional[str] = None,
        use_retrieval: bool = True,
        use_diversity: bool = True,
        log_path: Optional[str] = "logs/agent_trace.jsonl",
    ) -> AgentResult:
        self._run_counter += 1
        request_id = make_request_id(query or "", self._run_counter)
        guardrail_records: List[Dict] = []
        warnings: List[str] = []

        # 1. Validate input.
        v = guardrails.validate_query(query)
        guardrail_records.append(v.to_dict())
        if not v.ok:
            return self._finish_guardrail(
                request_id, query or "", "empty", v.message, guardrail_records,
                log_path, intent="out_of_scope", intent_conf=1.0,
            )
        clean_query = v.details.get("sanitized", str(query).strip())
        if v.code == "input_truncated":
            warnings.append(v.message)

        # 2. Validate top-k.
        tk = guardrails.enforce_topk(top_k)
        guardrail_records.append(tk.to_dict())
        if not tk.ok:
            return self._finish_guardrail(
                request_id, clean_query, "invalid_top_k", tk.message,
                guardrail_records, log_path, intent="out_of_scope", intent_conf=1.0,
            )
        top_k = tk.details["k"]

        # 3. Classify intent (with out-of-domain pre-filter).
        prediction = self.classifier.predict_with_confidence(clean_query)
        intent = prediction.intent
        intent_conf = prediction.confidence
        if guardrails.looks_out_of_domain(clean_query):
            intent = "out_of_scope"

        low_conf = guardrails.check_low_confidence(intent_conf)
        low_confidence = low_conf.code == "low_confidence"
        if low_confidence and intent != "out_of_scope":
            guardrail_records.append(low_conf.to_dict())
            warnings.append(low_conf.message)
            # Safe fallback: ambiguous requests become a balanced discovery.
            intent = "discover" if intent == "out_of_scope" else intent

        # 4. Plan.
        use_history = profile is not None
        plan = planner.plan_for_intent(
            intent, use_retrieval=use_retrieval,
            use_diversity=use_diversity, use_history=use_history,
        )

        # 5. Out-of-scope short-circuit.
        if intent == "out_of_scope":
            msg = guardrails.out_of_scope_message()
            decision = guardrails.GuardrailDecision(
                ok=False, code="out_of_scope",
                message="Request classified as out of music-discovery domain.",
            )
            guardrail_records.append(decision.to_dict())
            return self._finish_guardrail(
                request_id, clean_query, "out_of_scope", msg, guardrail_records,
                log_path, intent="out_of_scope", intent_conf=intent_conf,
                explicit_pref=None,
            )

        # 6. Retrieval.
        profile_data = self._load_profile(profile)
        retrieved_sims: Dict[int, float] = {}
        context_evidence: List[Evidence] = []
        song_evidence: List[Evidence] = []
        if use_retrieval:
            song_hits = self.retriever.retrieve(clean_query, top_k=12, sources=["song"])
            song_evidence = song_hits
            for e in song_hits:
                retrieved_sims[e.metadata["id"]] = e.score
            context_evidence = self.retriever.retrieve(
                clean_query, top_k=3, sources=["doc"], min_score=0.0
            )
            if profile_data:
                hist = self.retriever.retrieve(clean_query, top_k=1, sources=["history"])
                context_evidence = context_evidence + hist

        # 7. Build preferences.
        prefs, explicit_pref = self._build_preferences(intent, clean_query, profile_data)

        # 8/9. Branch by intent: compare / explain / ranking. Each branch only
        # produces the ranked recommendation(s); the text is composed after the
        # confidence value is known so the printed confidence line is accurate.
        recommendations: List[Recommendation] = []
        if intent == "compare":
            recommendations = self._handle_compare(clean_query, prefs, mode, retrieved_sims)
        elif intent == "explain":
            recommendations = self._handle_explain(clean_query, prefs, mode, retrieved_sims)
        else:
            diversify = use_diversity or intent == "diversify"
            recommendations = self._rank(prefs, mode, retrieved_sims, top_k,
                                         diversify, allow_explicit=explicit_pref)

        # 10. Confidence.
        top_song_sim = max(retrieved_sims.values(), default=0.0)
        top_doc_sim = max((e.score for e in context_evidence), default=0.0)
        retrieval_similarity = max(top_song_sim, top_doc_sim)

        # Confidence depends on the verifier pass rate, but the final text is
        # composed only after confidence is known. We resolve this by running the
        # grounding/consistency checks first on a neutral stand-in string (these
        # checks are about recommendations and evidence, not wording), then
        # verifying the real text once it exists.
        explicit_song_ids = {
            r.song_id for r in recommendations
            if self._by_id.get(r.song_id, {}).get("explicit")
        }

        # Compose (for ranking intents) now that we know evidence/confidence needs.
        diversified = (intent not in ("compare", "explain")) and (use_diversity or intent == "diversify")

        # First verification pass (grounding/consistency) to feed confidence.
        pre_report = verify_output(
            intent=intent, status="ok", response_text="pending final composition",
            recommendations=recommendations, valid_evidence_ids=self.valid_evidence_ids,
            catalog_ids=self.catalog_ids, requested_k=top_k, explicit_ok=explicit_pref,
            explicit_song_ids=explicit_song_ids, low_confidence=low_confidence,
            warnings=warnings, context_evidence=context_evidence,
        )
        confidence = compute_confidence(
            intent_confidence=intent_conf, retrieval_similarity=retrieval_similarity,
            recommendations=recommendations, verifier_pass_rate=pre_report.pass_rate,
        )

        # Compose the final text now that confidence is known.
        if intent == "compare" and len(recommendations) >= 2:
            response_text = compose_comparison_response(
                clean_query, recommendations[0], recommendations[1],
                context_evidence=context_evidence, confidence=confidence,
                warnings=warnings,
            )
        elif intent == "explain" and recommendations:
            response_text = compose_explain_response(
                clean_query, recommendations[0], context_evidence=context_evidence,
                confidence=confidence, warnings=warnings,
            )
        else:
            response_text = compose_recommendation_response(
                intent=intent, query=clean_query, recommendations=recommendations,
                context_evidence=context_evidence, confidence=confidence,
                warnings=warnings, diversified=diversified,
            )

        # 11. Final verification on the actual text.
        report = verify_output(
            intent=intent, status="ok", response_text=response_text,
            recommendations=recommendations, valid_evidence_ids=self.valid_evidence_ids,
            catalog_ids=self.catalog_ids, requested_k=top_k, explicit_ok=explicit_pref,
            explicit_song_ids=explicit_song_ids, low_confidence=low_confidence,
            warnings=warnings, context_evidence=context_evidence,
        )
        if not report.passed:
            warnings.append("Automated verification flagged a consistency issue; interpret results carefully.")
            confidence = round(confidence * 0.8, 4)

        # 12. Build result + trace, log, return.
        all_evidence = self._dedupe_evidence(song_evidence + context_evidence)
        result = AgentResult(
            query=clean_query, status="ok", intent=intent,
            intent_confidence=intent_conf, plan=plan, response_text=response_text,
            recommendations=recommendations, evidence=all_evidence,
            confidence=confidence, warnings=warnings, verifier=report.to_dict(),
        )
        result.trace = self._build_trace(
            request_id, clean_query, intent, intent_conf, plan, use_retrieval,
            use_diversity, recommendations, context_evidence, guardrail_records,
            report, confidence, "ok",
        )
        if log_path:
            write_trace(result.trace, log_path)
        return result

    # -- intent handlers -------------------------------------------------
    def _handle_compare(self, query, prefs, mode, retrieved_sims) -> List[Recommendation]:
        """Identify two candidate songs and score both. Returns [first, second]."""
        named = self._find_named_songs(query)
        candidates = named[:2]
        if len(candidates) < 2:
            # Fall back to the two best retrieved/ranked songs.
            ranked = self._rank(prefs, mode, retrieved_sims, top_k=5, diversify=False)
            picked_ids = [c["id"] for c in candidates]
            for r in ranked:
                if r.song_id not in picked_ids:
                    candidates.append(self._by_id[r.song_id])
                    picked_ids.append(r.song_id)
                if len(candidates) >= 2:
                    break
        first = self._score_one(candidates[0], prefs, mode)
        second = self._score_one(candidates[1], prefs, mode)
        # Present the higher-scoring song first for a stable, ordered comparison.
        return sorted([first, second], key=lambda r: r.score, reverse=True)

    def _handle_explain(self, query, prefs, mode, retrieved_sims) -> List[Recommendation]:
        """Identify one target song and score it for explanation."""
        named = self._find_named_songs(query)
        if named:
            target = named[0]
        else:
            ranked = self._rank(prefs, mode, retrieved_sims, top_k=1, diversify=False)
            target = self._by_id[ranked[0].song_id] if ranked else self.songs[0]
        return [self._score_one(target, prefs, mode)]

    # -- finishing helpers ----------------------------------------------
    def _dedupe_evidence(self, evidence: List[Evidence]) -> List[Evidence]:
        seen = set()
        out = []
        for e in evidence:
            if e.source_id in seen:
                continue
            seen.add(e.source_id)
            out.append(e)
        return out

    def _finish_guardrail(
        self, request_id, query, code, message, guardrail_records, log_path,
        intent, intent_conf, explicit_pref=None,
    ) -> AgentResult:
        response_text = compose_out_of_scope_response(message, confidence=0.0)
        report = verify_output(
            intent=intent, status="guardrail", response_text=response_text,
            recommendations=[], valid_evidence_ids=self.valid_evidence_ids,
            catalog_ids=self.catalog_ids, requested_k=0, explicit_ok=explicit_pref,
            explicit_song_ids=set(), low_confidence=False, warnings=[],
        )
        result = AgentResult(
            query=query, status="guardrail", intent=intent,
            intent_confidence=intent_conf,
            plan=planner.plan_for_intent("out_of_scope"),
            response_text=response_text, recommendations=[], evidence=[],
            confidence=0.0, warnings=[message], verifier=report.to_dict(),
        )
        result.trace = self._build_trace(
            request_id, query, intent, intent_conf, result.plan, False, False,
            [], [], guardrail_records, report, 0.0, "guardrail",
        )
        if log_path:
            write_trace(result.trace, log_path)
        return result

    def _build_trace(
        self, request_id, query, intent, intent_conf, plan, use_retrieval,
        use_diversity, recommendations, context_evidence, guardrail_records,
        report, confidence, status,
    ) -> Dict:
        components = ["guardrails", "intent_classifier", "planner"]
        if status == "ok":
            if use_retrieval:
                components.append("retriever")
            components += ["recommender", "composer", "verifier"]
        else:
            components += ["composer", "verifier"]
        return {
            "request_id": request_id,
            "timestamp": utc_timestamp(),
            "input_summary": guardrails.summarize_input(query),
            "intent": intent,
            "intent_confidence": round(float(intent_conf), 4),
            "plan": list(plan),
            "components_called": components,
            "retrieved_evidence_ids": [e.source_id for e in context_evidence],
            "recommendations": [
                {"id": r.song_id, "title": r.title, "score": r.score}
                for r in recommendations
            ],
            "guardrail_decisions": guardrail_records,
            "verifier_checks": report.checks,
            "verifier_passed": report.passed,
            "confidence": round(float(confidence), 4),
            "status": status,
        }
