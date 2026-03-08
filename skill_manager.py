"""
STELLA Skill Manager
Unified interface for skill lifecycle: create, retrieve, execute, evaluate, update.
Replaces the old template/memory system.
"""

import os
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable

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
from skill_summarizer import SkillSummarizer
from tool_governance import ToolIndex

STELLA_DIR = os.path.dirname(os.path.abspath(__file__))


class SkillManager:
    """
    Unified skill lifecycle manager for STELLA.

    Responsibilities:
      - Skill retrieval (find relevant skills for a query)
      - Skill execution tracking (record runs, update quality metrics)
      - Skill creation (auto-summarize successful runs into skills)
      - Skill governance (versioning, deprecation, deduplication)
      - Tool resolution (check tool availability for skills via ToolIndex)

    Replaces:
      - KnowledgeBase (template storage/retrieval)
      - MemoryManager (3-tier memory)
      - retrieve_similar_templates() / save_successful_template() in stella_core.py
    """

    def __init__(
        self,
        skills_dir: str = None,
        db_path: str = None,
        manifests_dir: str = None,
        llm_call: Callable = None,
    ):
        skills_dir = skills_dir or os.path.join(STELLA_DIR, "skills")
        db_path = db_path or os.path.join(STELLA_DIR, "data", "skill_runs.db")
        manifests_dir = manifests_dir or os.path.join(STELLA_DIR, "new_tools", "manifests")

        # Core components
        self.store = SkillStore(skills_dir=skills_dir, db_path=db_path)
        self.retriever = SkillRetriever(skill_store=self.store)
        self.summarizer = SkillSummarizer(
            skill_store=self.store,
            skill_retriever=self.retriever,
            llm_call=llm_call,
        )
        self.tool_index = ToolIndex(manifests_dir=manifests_dir)

        # Active run tracking
        self._active_runs: Dict[str, Dict] = {}

        print(f"SkillManager initialized: {len(self.store.list_all())} skills, "
              f"{len(self.tool_index.list_all())} tools indexed")

    # ------------------------------------------------------------------
    # Skill Retrieval (replaces retrieve_similar_templates)
    # ------------------------------------------------------------------
    def retrieve_skills(
        self,
        query: str,
        top_k: int = 3,
        domain: str = None,
        check_tools: bool = True,
    ) -> str:
        """
        Retrieve relevant skills for a query.
        Returns a formatted string suitable for agent consumption.

        This replaces the old retrieve_similar_templates() function.
        """
        results = self.retriever.retrieve(query, top_k=top_k, domain=domain)
        if not results:
            return "No matching skills found. Consider creating a new approach."

        lines = [f"Found {len(results)} relevant skills:\n"]
        for i, r in enumerate(results, 1):
            skill = r["skill"]
            lines.append(f"--- Skill {i}: {skill.name} (v{skill.version}) ---")
            lines.append(f"Domain: {skill.domain}")
            lines.append(f"Score: {r['score']:.3f} (similarity={r['similarity']:.2f}, "
                        f"success_rate={r['success_rate']:.0%}, recency={r['recency']:.2f})")
            lines.append(f"Description: {skill.description}")

            # Workflow steps
            if skill.workflow:
                lines.append("Workflow:")
                for step in skill.workflow:
                    tools_str = f" [tools: {', '.join(step.tools_required)}]" if step.tools_required else ""
                    lines.append(f"  {step.step}. {step.action}: {step.description}{tools_str}")

            # Tool availability check
            if check_tools:
                available, missing = self.tool_index.check_skill_tool_availability(skill)
                if missing:
                    lines.append(f"Missing tools: {', '.join(missing)}")
                else:
                    lines.append(f"All {len(available)} tools available")

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Run Tracking (replaces AutoMemory for skill-level tracking)
    # ------------------------------------------------------------------
    def start_run(self, skill_id: str, query: str) -> str:
        """Start tracking a skill execution. Returns run tracking ID."""
        import uuid
        track_id = f"track_{uuid.uuid4().hex[:8]}"
        self._active_runs[track_id] = {
            "skill_id": skill_id,
            "query": query,
            "start_time": time.time(),
            "tools_used": [],
            "steps": [],
        }
        return track_id

    def record_step(self, track_id: str, action: str, description: str, tools: List[str] = None, agent: str = "manager"):
        """Record a step during skill execution."""
        run = self._active_runs.get(track_id)
        if not run:
            return
        step_num = len(run["steps"]) + 1
        run["steps"].append({
            "step": step_num,
            "action": action,
            "description": description,
            "tools": tools or [],
            "agent": agent,
        })
        if tools:
            run["tools_used"].extend(tools)

    def record_tool_use(self, track_id: str, tool_name: str):
        """Record a tool usage during a run."""
        run = self._active_runs.get(track_id)
        if run:
            run["tools_used"].append(tool_name)

    def end_run(
        self,
        track_id: str,
        success: bool,
        critic_score: float = None,
        result_summary: str = "",
    ) -> Optional[str]:
        """
        End a tracked run. Records to store and optionally creates/updates skills.
        Returns the skill_id if a skill was created/updated, None otherwise.
        """
        run = self._active_runs.pop(track_id, None)
        if not run:
            return None

        completion_time = time.time() - run["start_time"]
        skill_id = run["skill_id"]
        tools_used = list(set(run["tools_used"]))

        # Record the run in store
        if skill_id and self.store.exists(skill_id):
            self.store.record_run(
                skill_id=skill_id,
                success=success,
                critic_score=critic_score,
                completion_time_sec=completion_time,
                query=run["query"],
                result_summary=result_summary,
                tools_used=tools_used,
            )
            # Record tool usage in tool index
            for tool_name in tools_used:
                self.tool_index.record_tool_use(tool_name, success)
            return skill_id

        # If no existing skill was used, try to create one from the run
        if success and critic_score and critic_score >= 0.6:
            skill = self.summarizer.summarize_run(
                query=run["query"],
                steps_taken=run["steps"],
                tools_used=tools_used,
                result_summary=result_summary,
                critic_score=critic_score,
            )
            if skill:
                return skill.skill_id

        return None

    # ------------------------------------------------------------------
    # Skill Creation (replaces save_successful_template)
    # ------------------------------------------------------------------
    def save_skill_from_run(
        self,
        query: str,
        reasoning_process: str,
        result_summary: str,
        tools_used: List[str] = None,
        domain: str = "general",
        critic_score: float = 0.7,
    ) -> str:
        """
        Save a successful run as a skill. Drop-in replacement for save_successful_template().

        Returns a status message.
        """
        steps = [{"action": "execute", "description": reasoning_process[:500], "tools": tools_used or []}]
        skill = self.summarizer.summarize_run(
            query=query,
            steps_taken=steps,
            tools_used=tools_used or [],
            result_summary=result_summary,
            critic_score=critic_score,
            domain=domain,
        )
        if skill:
            return f"Skill saved: {skill.skill_id} (v{skill.version})"
        return "Skill not created (quality threshold not met)"

    # ------------------------------------------------------------------
    # Skill Governance
    # ------------------------------------------------------------------
    def get_skill_status(self) -> str:
        """Get overall skill system status. Replaces list_knowledge_base_status()."""
        store_summary = self.store.get_summary()
        run_stats = self.store.get_all_run_stats()

        lines = [
            "=== STELLA Skill System Status ===",
            store_summary,
            "",
            f"Run Statistics:",
            f"  Total runs: {run_stats['total_runs']}",
            f"  Success rate: {run_stats['success_rate']:.0%}" if run_stats['total_runs'] > 0 else "  No runs yet",
            f"  Skills used: {run_stats['skills_used']}",
            "",
            self.tool_index.get_summary(),
        ]
        return "\n".join(lines)

    def search_skills(self, keyword: str, limit: int = 5) -> str:
        """Search skills by keyword. Replaces search_templates_by_keyword()."""
        results = self.retriever.retrieve(keyword, top_k=limit, min_score=0.05)
        if not results:
            return f"No skills found matching '{keyword}'"

        lines = [f"Skills matching '{keyword}':"]
        for r in results:
            s = r["skill"]
            lines.append(f"  - {s.skill_id}: {s.name} (score={r['score']:.3f}, domain={s.domain})")
        return "\n".join(lines)

    def deprecate_low_quality_skills(self, min_runs: int = 10, max_success_rate: float = 0.3) -> List[str]:
        """Flag and deprecate consistently failing skills."""
        deprecated = []
        for skill in self.store.list_all(status="active"):
            m = skill.quality_metrics
            if m.total_runs >= min_runs and m.success_rate < max_success_rate:
                self.store.deprecate_skill(
                    skill.skill_id,
                    reason=f"Low success rate: {m.success_rate:.0%} over {m.total_runs} runs",
                )
                deprecated.append(skill.skill_id)
        if deprecated:
            self.retriever.rebuild_index()
        return deprecated

    # ------------------------------------------------------------------
    # Tool-Skill Integration
    # ------------------------------------------------------------------
    def check_tools_for_query(self, query: str) -> str:
        """Check which tools would be needed for a query based on matching skills."""
        results = self.retriever.retrieve(query, top_k=3)
        if not results:
            return "No matching skills found. Tools will need to be loaded dynamically."

        all_tools = set()
        for r in results:
            all_tools.update(r["skill"].get_all_tool_ids())

        available, missing = [], []
        for tool_id in all_tools:
            if self.tool_index.get(tool_id):
                available.append(tool_id)
            else:
                missing.append(tool_id)

        lines = [f"Tools needed for query (based on {len(results)} matching skills):"]
        if available:
            lines.append(f"  Available: {', '.join(available)}")
        if missing:
            lines.append(f"  Missing: {', '.join(missing)}")
        return "\n".join(lines)

    def export_environment_for_skill(self, skill_id: str, format: str = "requirements") -> str:
        """Export reproducible environment for a specific skill's tools."""
        skill = self.store.get(skill_id)
        if not skill:
            return f"Skill {skill_id} not found"
        tool_ids = skill.get_all_tool_ids()
        return self.tool_index.export_environment(tool_ids, format=format)

    # ------------------------------------------------------------------
    # Migration from old system
    # ------------------------------------------------------------------
    def migrate_from_knowledge_base(self, kb_json_path: str) -> int:
        """
        Migrate templates from the old agent_knowledge_base.json to skills.

        Returns number of skills created.
        """
        if not os.path.exists(kb_json_path):
            print(f"Knowledge base file not found: {kb_json_path}")
            return 0

        try:
            with open(kb_json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load knowledge base: {e}")
            return 0

        templates = data if isinstance(data, list) else data.get("templates", [])
        created = 0

        for tmpl in templates:
            if not isinstance(tmpl, dict):
                continue
            task = tmpl.get("task", "")
            reasoning = tmpl.get("key_reasoning", "")
            domain = tmpl.get("domain", "general")

            if not task:
                continue

            # Create a skill from each template
            skill = self.summarizer._create_new_skill(
                query=task,
                steps_taken=[{"action": "execute", "description": reasoning, "tools": []}],
                tools_used=[],
                result_summary=reasoning,
                critic_score=0.7,
                domain=domain,
                agent="migrated",
            )
            if skill:
                created += 1

        if created:
            self.retriever.rebuild_index()
        print(f"Migrated {created} templates to skills")
        return created
