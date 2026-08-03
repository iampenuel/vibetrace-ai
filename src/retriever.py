"""Multi-source TF-IDF retrieval for VibeTrace AI.

Retrieval grounds the system in evidence *before* it composes an answer. Three
source types are indexed into a single TF-IDF space:

* ``song``    — one document per catalog track (searchable feature descriptors)
* ``doc``     — one document per ``## section`` of each knowledge/*.md file
* ``history`` — one document per synthetic sample listening profile (optional)

Each retrieved item is returned as an :class:`src.models.Evidence` object with a
stable identifier such as ``song:12``, ``doc:listening_contexts.md#studying``,
or ``history:night_owl_coder``.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import Evidence


class RetrieverError(RuntimeError):
    """Raised when the retriever is used before an index is built."""


def _slugify(heading: str) -> str:
    slug = heading.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug or "section"


# ---------------------------------------------------------------------------
# Source -> searchable text helpers
# ---------------------------------------------------------------------------

def song_to_text(song: Dict) -> str:
    """Turn a song row into a keyword-rich, searchable description.

    Numeric audio features are expanded into descriptive words (e.g. a low
    ``energy`` becomes "low energy calm") so natural-language queries such as
    "calm study music" can match the right tracks.
    """
    parts: List[str] = [
        song["title"], song["artist"], song["genre"], song["mood"],
        f"{song['genre']} {song['mood']}",
    ]

    energy = float(song["energy"])
    if energy >= 0.75:
        parts.append("high energy energetic intense powerful driving")
    elif energy >= 0.5:
        parts.append("moderate energy")
    else:
        parts.append("low energy calm mellow soft gentle quiet")

    tempo = float(song["tempo_bpm"])
    if tempo >= 120:
        parts.append("fast tempo upbeat quick")
    elif tempo <= 90:
        parts.append("slow tempo laid back")

    if float(song["acousticness"]) >= 0.6:
        parts.append("acoustic organic unplugged")
    if float(song["instrumentalness"]) >= 0.6:
        parts.append("instrumental no lyrics vocal free")
    if float(song["danceability"]) >= 0.7:
        parts.append("danceable groovy")
    valence = float(song["valence"])
    if valence >= 0.65:
        parts.append("happy positive cheerful uplifting")
    elif valence <= 0.4:
        parts.append("moody melancholic dark reflective")

    # Coarse context tags derived from feature combinations. These help study /
    # workout / relax queries retrieve the right tracks.
    if energy < 0.45 and (float(song["acousticness"]) >= 0.6
                          or float(song["instrumentalness"]) >= 0.6):
        parts.append("studying focus concentration reading relaxing calm")
    if energy >= 0.8 and tempo >= 120:
        parts.append("workout gym running exercise cardio training")
    if energy < 0.5 and valence >= 0.55:
        parts.append("relaxing unwind chill soothing")

    if song.get("explicit"):
        parts.append("explicit content")
    else:
        parts.append("clean content")

    return " ".join(str(p) for p in parts)


def parse_doc_chunks(knowledge_dir: str) -> List[Dict]:
    """Parse every ``## section`` of every markdown file in ``knowledge_dir``."""
    chunks: List[Dict] = []
    if not os.path.isdir(knowledge_dir):
        return chunks
    for path in sorted(glob.glob(os.path.join(knowledge_dir, "*.md"))):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        # Split on level-2 headings, keeping the heading with its body.
        sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
        for section in sections[1:]:  # sections[0] is the file's H1 preamble
            lines = section.splitlines()
            heading = lines[0].strip()
            body = " ".join(l.strip() for l in lines[1:] if l.strip())
            if not body:
                continue
            slug = _slugify(heading)
            chunks.append({
                "source_id": f"doc:{filename}#{slug}",
                "heading": heading,
                "filename": filename,
                "text": f"{heading}. {body}",
            })
    return chunks


def parse_history_profiles(history_path: str) -> List[Dict]:
    """Parse synthetic sample listening profiles into searchable documents."""
    records: List[Dict] = []
    if not history_path or not os.path.exists(history_path):
        return records
    with open(history_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for name, profile in data.get("profiles", {}).items():
        text = " ".join([
            profile.get("display_name", name),
            profile.get("summary", ""),
            " ".join(profile.get("top_genres", [])),
            " ".join(profile.get("top_moods", [])),
            profile.get("notes", ""),
        ])
        records.append({
            "source_id": f"history:{name}",
            "name": name,
            "text": text,
            "profile": profile,
        })
    return records


# ---------------------------------------------------------------------------
# The retriever
# ---------------------------------------------------------------------------

@dataclass
class _Doc:
    source_type: str
    source_id: str
    text: str
    metadata: Dict


class MultiSourceRetriever:
    """Builds one TF-IDF index over songs, knowledge docs, and history profiles."""

    def __init__(self) -> None:
        self._docs: List[_Doc] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

    def build_index(
        self,
        songs: List[Dict],
        knowledge_dir: str = "knowledge",
        history_path: Optional[str] = "data/sample_user_history.json",
    ) -> "MultiSourceRetriever":
        """Index all available sources. Missing doc/history sources are skipped
        gracefully so retrieval still works with songs alone."""
        docs: List[_Doc] = []

        for song in songs:
            docs.append(_Doc(
                source_type="song",
                source_id=f"song:{song['id']}",
                text=song_to_text(song),
                metadata={
                    "id": song["id"], "title": song["title"],
                    "artist": song["artist"], "genre": song["genre"],
                    "mood": song["mood"],
                },
            ))

        for chunk in parse_doc_chunks(knowledge_dir):
            docs.append(_Doc(
                source_type="doc",
                source_id=chunk["source_id"],
                text=chunk["text"],
                metadata={"filename": chunk["filename"], "heading": chunk["heading"]},
            ))

        for rec in parse_history_profiles(history_path):
            docs.append(_Doc(
                source_type="history",
                source_id=rec["source_id"],
                text=rec["text"],
                metadata={"name": rec["name"],
                          "display_name": rec["profile"].get("display_name", rec["name"])},
            ))

        if not docs:
            raise RetrieverError("No documents to index.")

        self._docs = docs
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                           sublinear_tf=True, stop_words="english")
        self._matrix = self._vectorizer.fit_transform([d.text for d in docs])
        return self

    def is_built(self) -> bool:
        return self._matrix is not None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        sources: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[Evidence]:
        """Return up to ``top_k`` evidence items ranked by cosine similarity.

        ``sources`` optionally restricts the search to given source types
        (e.g. ``["doc"]``). Returns an empty list for empty queries; raises
        :class:`RetrieverError` if no index has been built.
        """
        if not self.is_built():
            raise RetrieverError("Index has not been built. Call build_index() first.")
        if not query or not str(query).strip():
            return []

        allowed = set(sources) if sources else None
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]

        scored = []
        for idx, doc in enumerate(self._docs):
            if allowed is not None and doc.source_type not in allowed:
                continue
            score = float(sims[idx])
            if score <= min_score:
                continue
            scored.append((score, idx, doc))

        # Sort by score desc, then source_id for deterministic ties.
        scored.sort(key=lambda t: (-t[0], t[2].source_id))

        results: List[Evidence] = []
        for score, _idx, doc in scored[:top_k]:
            results.append(Evidence(
                source_type=doc.source_type,
                source_id=doc.source_id,
                text=doc.text,
                score=round(score, 4),
                metadata=dict(doc.metadata),
            ))
        return results

    def retrieve_by_source(
        self, query: str, per_source: Dict[str, int]
    ) -> List[Evidence]:
        """Retrieve a fixed number of items per source type.

        Example: ``{"song": 8, "doc": 3, "history": 1}`` returns up to 8 songs,
        3 doc passages, and 1 history record. Useful for balanced grounding.
        """
        out: List[Evidence] = []
        for source, k in per_source.items():
            if k <= 0:
                continue
            out.extend(self.retrieve(query, top_k=k, sources=[source]))
        return out
