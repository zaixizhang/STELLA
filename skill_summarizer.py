"""
STELLA Skill Summarizer
Auto-extracts structured skills from successful agent runs.
"""

import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

from skill_schema import (
    Skill,
    SkillWorkflowStep,
    SkillToolDependency,
    SkillQualityMetrics,
    SkillProvenance,
    SkillVersionEntry,
)
from skill_store import SkillStore
from skill_retriever import SkillRetriever


# Default similarity threshold for deduplication
DEDUP_THRESHOLD = 0.85


class SkillSummarizer:
    """
    Extracts structured skills from successful runs.

    Workflow:
      1. Receive run details (query, steps taken, tools used, result, critic score).
      2. Use LLM to summarize run into a structured skill YAML.
      3. Deduplication: if similar skill exists (similarity > threshold), update it.
      4. Otherwise, create a new skill.
    """

    def __init__(
        self,
        skill_store: SkillStore,
        skill_retriever: SkillRetriever,
        llm_call=None,
        dedup_threshold: float = DEDUP_THRESHOLD,
    ):
        self.store = skill_store
        self.retriever = skill_retriever
        self.llm_call = llm_call  # function(prompt) -> str
        self.dedup_threshold = dedup_threshold

    def summarize_run(
        self,
        query: str,
        steps_taken: List[Dict[str, Any]],
        tools_used: List[str],
        result_summary: str,
        critic_score: float,
        domain: str = "general",
        agent: str = "manager",
        min_critic_score: float = 0.6,
    ) -> Optional[Skill]:
        """
        Attempt to create or update a skill from a successful run.

        Args:
            query: The original user query.
            steps_taken: List of dicts with 'action', 'description', 'tools'.
            tools_used: All tools used during the run.
            result_summary: Summary of the outcome.
            critic_score: Score from the Critic Agent (0.0 - 1.0).
            domain: Domain tag for the skill.
            agent: Which agent executed.
            min_critic_score: Minimum score to consider the run "successful".

        Returns:
            The created or updated Skill, or None if the run didn't qualify.
        """
        # Gate: only create skills from high-quality runs
        if critic_score < min_critic_score:
            print(f"Skill summarizer: critic_score {critic_score:.2f} below threshold {min_critic_score}. Skipping.")
            return None

        # Check for similar existing skills
        existing_matches = self.retriever.retrieve(query, top_k=1, min_score=0.0)
        if existing_matches and existing_matches[0]["similarity"] >= self.dedup_threshold:
            # Update existing skill instead of creating new one
            return self._update_existing_skill(
                existing_matches[0]["skill"],
                query, steps_taken, tools_used, result_summary, critic_score,
            )

        # Create new skill
        return self._create_new_skill(
            query, steps_taken, tools_used, result_summary,
            critic_score, domain, agent,
        )

    def _create_new_skill(
        self,
        query: str,
        steps_taken: List[Dict],
        tools_used: List[str],
        result_summary: str,
        critic_score: float,
        domain: str,
        agent: str,
    ) -> Skill:
        """Create a brand new skill from a run."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now_str = datetime.now().strftime("%Y-%m-%d")

        # Generate skill metadata using LLM if available
        skill_meta = self._generate_skill_metadata(query, steps_taken, tools_used, result_summary, domain)

        # Build workflow steps
        workflow = []
        for i, step in enumerate(steps_taken, 1):
            workflow.append(SkillWorkflowStep(
                step=i,
                action=step.get("action", f"step_{i}"),
                description=step.get("description", ""),
                tools_required=step.get("tools", []),
                agent=step.get("agent", agent),
            ))

        # If no structured steps, create a single-step workflow
        if not workflow:
            workflow = [SkillWorkflowStep(
                step=1,
                action="execute",
                description=f"Execute: {query}",
                tools_required=tools_used[:5],
                agent=agent,
            )]

        # Build tool dependencies
        tool_deps = [
            SkillToolDependency(tool_id=t, version=">=1.0.0", optional=False)
            for t in tools_used
        ]

        skill_id = skill_meta.get("skill_id", f"auto_{uuid.uuid4().hex[:8]}")
        skill = Skill(
            skill_id=skill_id,
            version="1.0.0",
            name=skill_meta.get("name", query[:80]),
            domain=domain,
            description=skill_meta.get("description", result_summary[:200]),
            workflow=workflow,
            tools_required=tool_deps,
            tags=skill_meta.get("tags", self._extract_tags(query)),
            applicable_queries=skill_meta.get("applicable_queries", [f"*{query.split()[0].lower()}*"]),
            quality_metrics=SkillQualityMetrics(
                success_count=1,
                failure_count=0,
                avg_completion_time_sec=0,
                avg_critic_score=critic_score,
                last_used=now_str,
                created_at=now_str,
                last_updated=now_str,
            ),
            provenance=SkillProvenance(
                created_by="auto_summarizer",
                source_run_id=run_id,
                changelog=[SkillVersionEntry(
                    version="1.0.0",
                    date=now_str,
                    changes="Auto-created from successful run",
                )],
            ),
            status="active",
        )

        # Save to store
        self.store.save(skill)
        # Record the run
        self.store.record_run(
            skill_id=skill.skill_id,
            success=True,
            critic_score=critic_score,
            query=query,
            result_summary=result_summary[:500],
            tools_used=tools_used,
            agent=agent,
        )
        # Rebuild retriever index
        self.retriever.rebuild_index()

        print(f"New skill created: {skill.skill_id} (v{skill.version})")
        return skill

    def _update_existing_skill(
        self,
        existing: Skill,
        query: str,
        steps_taken: List[Dict],
        tools_used: List[str],
        result_summary: str,
        critic_score: float,
    ) -> Skill:
        """Update an existing skill with data from a new successful run."""
        # Merge new tools into the skill's tool dependencies
        existing_tool_ids = {t.tool_id for t in existing.tools_required}
        for t in tools_used:
            if t not in existing_tool_ids:
                existing.tools_required.append(
                    SkillToolDependency(tool_id=t, version=">=1.0.0", optional=True)
                )

        # Merge new tags
        new_tags = self._extract_tags(query)
        existing_tags = set(existing.tags)
        for tag in new_tags:
            if tag not in existing_tags:
                existing.tags.append(tag)

        # Record the run (metrics update handled by store)
        self.store.record_run(
            skill_id=existing.skill_id,
            success=True,
            critic_score=critic_score,
            query=query,
            result_summary=result_summary[:500],
            tools_used=tools_used,
        )

        # Save updated skill
        self.store.save(existing)
        print(f"Existing skill updated: {existing.skill_id}")
        return existing

    def _generate_skill_metadata(
        self,
        query: str,
        steps_taken: List[Dict],
        tools_used: List[str],
        result_summary: str,
        domain: str,
    ) -> Dict[str, Any]:
        """Use LLM to generate rich skill metadata, or fall back to heuristics."""
        if self.llm_call:
            try:
                prompt = f"""Analyze this successful agent run and generate structured skill metadata.

Query: {query}
Domain: {domain}
Tools used: {', '.join(tools_used)}
Steps taken: {len(steps_taken)}
Result summary: {result_summary[:300]}

Return a JSON object with these fields:
- skill_id: short snake_case identifier (e.g., "gene_resistance_analysis")
- name: human-readable name (e.g., "Multi-Database Gene Resistance Analysis")
- description: 1-2 sentence description of the reusable workflow
- tags: list of 5-8 keyword tags
- applicable_queries: list of 3-5 glob patterns with * wildcards that would match similar future queries

Return ONLY valid JSON, no other text."""

                response = self.llm_call(prompt)
                # Parse JSON from response
                import json
                # Try to extract JSON from response
                json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception as e:
                print(f"LLM skill metadata generation failed: {e}")

        # Fallback: heuristic metadata
        return {
            "skill_id": f"auto_{self._slugify(query)[:40]}",
            "name": query[:80],
            "description": result_summary[:200] if result_summary else query,
            "tags": self._extract_tags(query),
            "applicable_queries": [f"*{w}*" for w in query.lower().split()[:3] if len(w) > 3],
        }

    def _extract_tags(self, text: str) -> List[str]:
        """Extract keyword tags from text."""
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "can", "shall",
                      "to", "of", "in", "for", "on", "with", "at", "by", "from",
                      "and", "or", "but", "not", "this", "that", "these", "those",
                      "it", "its", "i", "me", "my", "we", "our", "you", "your",
                      "he", "she", "they", "them", "what", "which", "who", "how",
                      "find", "get", "use", "make", "help", "please"}
        words = re.findall(r"\w+", text.lower())
        tags = [w for w in words if len(w) > 2 and w not in stop_words]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique[:8]

    def _slugify(self, text: str) -> str:
        """Convert text to a slug suitable for skill_id."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")
