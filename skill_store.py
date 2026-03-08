"""
STELLA Skill Store
Persistence layer for skills: YAML files + SQLite for run tracking.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from skill_schema import Skill, SkillQualityMetrics, SkillVersionEntry

# Base directories
STELLA_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(STELLA_DIR, "skills")
PREBUILT_DIR = os.path.join(SKILLS_DIR, "prebuilt")
AUTO_GEN_DIR = os.path.join(SKILLS_DIR, "auto_generated")
DATA_DIR = os.path.join(STELLA_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "skill_runs.db")


class SkillStore:
    """
    Manages skill persistence:
      - Skills stored as YAML files in skills/ directory
      - Run history and metrics stored in SQLite (data/skill_runs.db)
    """

    def __init__(self, skills_dir: str = SKILLS_DIR, db_path: str = DB_PATH):
        self.skills_dir = skills_dir
        self.prebuilt_dir = os.path.join(skills_dir, "prebuilt")
        self.auto_gen_dir = os.path.join(skills_dir, "auto_generated")
        self.db_path = db_path

        # Ensure directories exist
        for d in [self.skills_dir, self.prebuilt_dir, self.auto_gen_dir, os.path.dirname(db_path)]:
            os.makedirs(d, exist_ok=True)

        # In-memory skill index: skill_id -> Skill
        self._skills: Dict[str, Skill] = {}
        # Track file paths: skill_id -> file_path
        self._paths: Dict[str, str] = {}

        # Initialize SQLite
        self._init_db()
        # Load all skills from disk
        self._load_all_skills()

    # ------------------------------------------------------------------
    # SQLite setup
    # ------------------------------------------------------------------
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS skill_runs (
                run_id TEXT PRIMARY KEY,
                skill_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                success INTEGER NOT NULL,
                critic_score REAL,
                completion_time_sec REAL,
                query TEXT,
                result_summary TEXT,
                tools_used TEXT,
                agent TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS skill_events (
                event_id TEXT PRIMARY KEY,
                skill_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                details TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Skill loading
    # ------------------------------------------------------------------
    def _load_all_skills(self):
        """Load all skill YAML files from prebuilt/ and auto_generated/."""
        for directory in [self.prebuilt_dir, self.auto_gen_dir]:
            if not os.path.exists(directory):
                continue
            for fname in os.listdir(directory):
                if fname.endswith((".yaml", ".yml")):
                    path = os.path.join(directory, fname)
                    try:
                        skill = Skill.from_yaml_file(path)
                        self._skills[skill.skill_id] = skill
                        self._paths[skill.skill_id] = path
                    except Exception as e:
                        print(f"Warning: failed to load skill from {path}: {e}")

    def reload(self):
        """Reload all skills from disk."""
        self._skills.clear()
        self._paths.clear()
        self._load_all_skills()

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def list_all(self, status: str = None, domain: str = None) -> List[Skill]:
        skills = list(self._skills.values())
        if status:
            skills = [s for s in skills if s.status == status]
        if domain:
            skills = [s for s in skills if s.domain == domain]
        return skills

    def save(self, skill: Skill, directory: str = None) -> str:
        """Save a skill to YAML. Returns the file path."""
        if directory is None:
            # Auto-generated skills go to auto_generated/
            if skill.provenance.created_by == "prebuilt":
                directory = self.prebuilt_dir
            else:
                directory = self.auto_gen_dir

        filename = f"{skill.skill_id}.yaml"
        path = os.path.join(directory, filename)
        skill.save_yaml(path)
        self._skills[skill.skill_id] = skill
        self._paths[skill.skill_id] = path
        return path

    def delete(self, skill_id: str) -> bool:
        """Delete a skill from store and disk."""
        if skill_id in self._paths:
            path = self._paths[skill_id]
            if os.path.exists(path):
                os.remove(path)
            del self._paths[skill_id]
            del self._skills[skill_id]
            return True
        return False

    def exists(self, skill_id: str) -> bool:
        return skill_id in self._skills

    # ------------------------------------------------------------------
    # Run tracking
    # ------------------------------------------------------------------
    def record_run(
        self,
        skill_id: str,
        success: bool,
        critic_score: float = None,
        completion_time_sec: float = None,
        query: str = "",
        result_summary: str = "",
        tools_used: List[str] = None,
        agent: str = "manager",
    ) -> str:
        """Record a skill execution run. Returns run_id."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO skill_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                skill_id,
                timestamp,
                1 if success else 0,
                critic_score,
                completion_time_sec,
                query,
                result_summary[:500] if result_summary else "",
                json.dumps(tools_used or []),
                agent,
            ),
        )
        conn.commit()
        conn.close()

        # Update in-memory skill metrics
        self._update_skill_metrics(skill_id, success, critic_score, completion_time_sec)

        return run_id

    def _update_skill_metrics(
        self, skill_id: str, success: bool, critic_score: float = None, completion_time: float = None
    ):
        """Update skill quality metrics after a run."""
        skill = self._skills.get(skill_id)
        if not skill:
            return

        m = skill.quality_metrics
        if success:
            m.success_count += 1
        else:
            m.failure_count += 1

        m.last_used = datetime.now().strftime("%Y-%m-%d")
        m.last_updated = datetime.now().strftime("%Y-%m-%d")

        # Update running averages
        total = m.total_runs
        if critic_score is not None and total > 0:
            m.avg_critic_score = (
                (m.avg_critic_score * (total - 1) + critic_score) / total
            )
        if completion_time is not None and total > 0:
            m.avg_completion_time_sec = (
                (m.avg_completion_time_sec * (total - 1) + completion_time) / total
            )

        # Persist updated metrics
        if skill_id in self._paths:
            skill.save_yaml(self._paths[skill_id])

    def get_run_history(self, skill_id: str, limit: int = 20) -> List[Dict]:
        """Get recent runs for a skill."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT * FROM skill_runs WHERE skill_id = ? ORDER BY timestamp DESC LIMIT ?",
            (skill_id, limit),
        )
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_all_run_stats(self) -> Dict[str, Any]:
        """Get aggregate run statistics across all skills."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM skill_runs")
        total_runs = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM skill_runs WHERE success = 1")
        total_success = c.fetchone()[0]
        c.execute("SELECT COUNT(DISTINCT skill_id) FROM skill_runs")
        skills_used = c.fetchone()[0]
        conn.close()
        return {
            "total_runs": total_runs,
            "total_success": total_success,
            "success_rate": total_success / total_runs if total_runs > 0 else 0.0,
            "skills_used": skills_used,
            "total_skills": len(self._skills),
        }

    # ------------------------------------------------------------------
    # Skill versioning
    # ------------------------------------------------------------------
    def version_skill(self, skill_id: str, new_version: str, changes: str) -> bool:
        """Bump skill version and record changelog."""
        skill = self._skills.get(skill_id)
        if not skill:
            return False

        entry = SkillVersionEntry(
            version=new_version,
            date=datetime.now().strftime("%Y-%m-%d"),
            changes=changes,
        )
        skill.provenance.changelog.append(entry)
        skill.version = new_version
        skill.quality_metrics.last_updated = datetime.now().strftime("%Y-%m-%d")

        # Save
        if skill_id in self._paths:
            skill.save_yaml(self._paths[skill_id])
        return True

    def deprecate_skill(self, skill_id: str, reason: str = "") -> bool:
        """Mark a skill as deprecated."""
        skill = self._skills.get(skill_id)
        if not skill:
            return False
        skill.status = "deprecated"
        entry = SkillVersionEntry(
            version=skill.version,
            date=datetime.now().strftime("%Y-%m-%d"),
            changes=f"Deprecated: {reason}" if reason else "Deprecated",
        )
        skill.provenance.changelog.append(entry)
        if skill_id in self._paths:
            skill.save_yaml(self._paths[skill_id])
        return True

    # ------------------------------------------------------------------
    # Search helpers (used by SkillRetriever)
    # ------------------------------------------------------------------
    def get_skill_descriptions(self) -> Dict[str, str]:
        """Return {skill_id: description} for embedding."""
        return {sid: s.description for sid, s in self._skills.items() if s.status == "active"}

    def get_skill_tags(self) -> Dict[str, List[str]]:
        """Return {skill_id: tags} for fast filtering."""
        return {sid: s.tags for sid, s in self._skills.items() if s.status == "active"}

    def get_skill_patterns(self) -> Dict[str, List[str]]:
        """Return {skill_id: applicable_queries} for pattern matching."""
        return {sid: s.applicable_queries for sid, s in self._skills.items() if s.status == "active"}

    def get_summary(self) -> str:
        """Human-readable summary of the skill store."""
        active = [s for s in self._skills.values() if s.status == "active"]
        deprecated = [s for s in self._skills.values() if s.status == "deprecated"]
        domains = set(s.domain for s in active)
        lines = [
            f"Skill Store: {len(active)} active, {len(deprecated)} deprecated",
            f"Domains: {', '.join(sorted(domains))}",
        ]
        for s in sorted(active, key=lambda x: x.quality_metrics.success_rate, reverse=True)[:10]:
            sr = f"{s.quality_metrics.success_rate:.0%}" if s.quality_metrics.total_runs > 0 else "N/A"
            lines.append(f"  - {s.skill_id} v{s.version} ({s.domain}) success={sr}")
        return "\n".join(lines)
