"""
STELLA Skill Retriever
Hybrid 3-stage skill retrieval: tag/pattern match → embedding similarity → quality re-rank.
"""

import re
import fnmatch
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skill_schema import Skill
from skill_store import SkillStore


class SkillRetriever:
    """
    Three-stage hybrid retrieval system for skills.

    Stage 1 — Tag & Pattern Matching (fast filter)
      Match query against skill.tags and skill.applicable_queries (glob patterns).

    Stage 2 — Embedding Similarity (semantic ranking)
      TF-IDF + cosine similarity on skill descriptions.
      (Pluggable: can swap to dense embeddings in future.)

    Stage 3 — Quality-Weighted Re-ranking
      final_score = α × similarity + β × success_rate + γ × recency
      Default: α=0.5, β=0.3, γ=0.2
    """

    def __init__(
        self,
        skill_store: SkillStore,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        recency_decay_days: float = 90.0,
    ):
        self.store = skill_store
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.recency_decay_days = recency_decay_days

        # TF-IDF vectorizer for Stage 2
        self._vectorizer = TfidfVectorizer(
            stop_words="english", max_features=2000, ngram_range=(1, 2)
        )
        self._skill_ids: List[str] = []
        self._skill_vectors = None

        # Build initial index
        self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def _build_index(self):
        """Build TF-IDF index from current skill descriptions."""
        descriptions = self.store.get_skill_descriptions()
        if not descriptions:
            self._skill_ids = []
            self._skill_vectors = None
            return

        self._skill_ids = list(descriptions.keys())
        texts = []
        for sid in self._skill_ids:
            skill = self.store.get(sid)
            # Combine description + tags + applicable_queries for richer matching
            parts = [skill.description]
            parts.extend(skill.tags)
            parts.extend(skill.applicable_queries)
            # Add workflow step descriptions
            for step in skill.workflow:
                parts.append(step.description)
            texts.append(" ".join(parts))

        self._skill_vectors = self._vectorizer.fit_transform(texts)

    def rebuild_index(self):
        """Rebuild the index after skills change."""
        self._build_index()

    # ------------------------------------------------------------------
    # Main retrieval entry point
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: str = None,
        min_score: float = 0.1,
    ) -> List[Dict]:
        """
        Retrieve skills matching a query using 3-stage hybrid retrieval.

        Returns a list of dicts:
          [{"skill": Skill, "score": float, "stage1_match": bool, "similarity": float, ...}, ...]
        """
        if not self._skill_ids:
            return []

        # Stage 1: Fast filter by tags & patterns
        stage1_candidates = self._stage1_tag_pattern_match(query, domain)

        # Stage 2: Semantic similarity (on all active skills, not just stage1)
        stage2_scores = self._stage2_embedding_similarity(query)

        # Merge: boost stage1 candidates
        combined = {}
        for sid in self._skill_ids:
            skill = self.store.get(sid)
            if skill is None or skill.status != "active":
                continue
            if domain and skill.domain != domain:
                continue

            sim_score = stage2_scores.get(sid, 0.0)
            is_stage1 = sid in stage1_candidates

            # Stage 3: Quality re-ranking
            success_rate = skill.quality_metrics.success_rate
            recency_score = self._compute_recency(skill.quality_metrics.last_used)

            # If stage1 matched, give a similarity floor boost
            if is_stage1:
                sim_score = max(sim_score, 0.3)

            final_score = (
                self.alpha * sim_score
                + self.beta * success_rate
                + self.gamma * recency_score
            )

            if final_score >= min_score:
                combined[sid] = {
                    "skill": skill,
                    "score": round(final_score, 4),
                    "similarity": round(sim_score, 4),
                    "success_rate": round(success_rate, 4),
                    "recency": round(recency_score, 4),
                    "stage1_match": is_stage1,
                }

        # Sort by score descending
        results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Stage 1: Tag & Pattern Matching
    # ------------------------------------------------------------------
    def _stage1_tag_pattern_match(self, query: str, domain: str = None) -> set:
        """Fast filter: match query tokens against tags and glob patterns."""
        candidates = set()
        query_lower = query.lower()
        query_tokens = set(re.findall(r"\w+", query_lower))

        # Check tags
        all_tags = self.store.get_skill_tags()
        for sid, tags in all_tags.items():
            if domain:
                skill = self.store.get(sid)
                if skill and skill.domain != domain:
                    continue
            tag_set = set(t.lower() for t in tags)
            overlap = query_tokens & tag_set
            if len(overlap) >= 1:
                candidates.add(sid)

        # Check applicable_queries (glob patterns)
        all_patterns = self.store.get_skill_patterns()
        for sid, patterns in all_patterns.items():
            for pattern in patterns:
                # Convert glob pattern to regex-friendly
                if fnmatch.fnmatch(query_lower, pattern.lower()):
                    candidates.add(sid)
                    break
                # Also check if query contains key terms from pattern
                pattern_tokens = set(re.findall(r"\w+", pattern.lower())) - {"*"}
                if pattern_tokens and pattern_tokens.issubset(query_tokens):
                    candidates.add(sid)
                    break

        return candidates

    # ------------------------------------------------------------------
    # Stage 2: Embedding Similarity
    # ------------------------------------------------------------------
    def _stage2_embedding_similarity(self, query: str) -> Dict[str, float]:
        """Compute TF-IDF cosine similarity between query and all skills."""
        if self._skill_vectors is None or len(self._skill_ids) == 0:
            return {}

        try:
            query_vec = self._vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self._skill_vectors)[0]
        except Exception:
            return {}

        return {
            sid: float(similarities[i])
            for i, sid in enumerate(self._skill_ids)
        }

    # ------------------------------------------------------------------
    # Recency scoring
    # ------------------------------------------------------------------
    def _compute_recency(self, last_used: str) -> float:
        """Exponential decay recency score. 1.0 = today, decays over recency_decay_days."""
        if not last_used:
            return 0.3  # neutral for skills never used

        try:
            last_dt = datetime.strptime(last_used, "%Y-%m-%d")
            days_ago = (datetime.now() - last_dt).days
            return max(0.0, np.exp(-days_ago / self.recency_decay_days))
        except (ValueError, TypeError):
            return 0.3

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def find_similar_skills(self, skill_id: str, top_k: int = 3) -> List[Dict]:
        """Find skills similar to a given skill (for deduplication)."""
        skill = self.store.get(skill_id)
        if not skill:
            return []
        return [r for r in self.retrieve(skill.description, top_k=top_k + 1) if r["skill"].skill_id != skill_id][:top_k]

    def get_retrieval_explanation(self, query: str, top_k: int = 3) -> str:
        """Human-readable explanation of retrieval results."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No matching skills found."

        lines = [f"Skill retrieval for: '{query}'", ""]
        for i, r in enumerate(results, 1):
            s = r["skill"]
            lines.append(
                f"{i}. {s.name} (v{s.version})"
                f"\n   Score: {r['score']:.3f} "
                f"(sim={r['similarity']:.2f}, success={r['success_rate']:.0%}, recency={r['recency']:.2f})"
                f"\n   Domain: {s.domain} | Tags: {', '.join(s.tags[:5])}"
                f"\n   Stage 1 match: {'Yes' if r['stage1_match'] else 'No'}"
            )
        return "\n".join(lines)
