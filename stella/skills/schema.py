"""
STELLA Skill & Tool Manifest Schemas
Pydantic models for structured skill and tool governance.
"""

import os
import yaml
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict


# ============================================================
# Tool Manifest Schema
# ============================================================

@dataclass
class ToolParameter:
    """A single parameter in a tool's interface."""
    name: str
    type: str
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass
class ToolInterface:
    """The function interface contract for a tool."""
    function_name: str
    module_path: str
    parameters: List[ToolParameter] = field(default_factory=list)
    return_type: str = "str"
    return_description: str = ""


@dataclass
class ToolDependencies:
    """Dependencies required by a tool."""
    python_packages: List[str] = field(default_factory=list)
    system_packages: List[str] = field(default_factory=list)
    conda_channels: List[str] = field(default_factory=lambda: ["conda-forge"])
    env_vars_required: List[str] = field(default_factory=list)


@dataclass
class ToolValidation:
    """Validation metadata for a tool."""
    test_function: str = ""
    test_module: str = ""
    last_validated: str = ""
    validation_status: str = "untested"  # untested, passed, failed


@dataclass
class ToolChangelogEntry:
    """A single changelog entry."""
    version: str
    date: str
    changes: str


@dataclass
class ToolProvenance:
    """Provenance tracking for a tool."""
    created_at: str = ""
    last_modified: str = ""
    source: str = "predefined"  # predefined, auto_created, imported
    changelog: List[ToolChangelogEntry] = field(default_factory=list)


@dataclass
class ToolUsageStats:
    """Runtime usage statistics for a tool."""
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class ToolManifest:
    """Complete tool manifest with governance metadata."""
    tool_id: str
    version: str
    name: str
    description: str
    category: str  # database, search, analysis, visualization, environment, etc.
    author: str = "stella_team"

    interface: ToolInterface = field(default_factory=lambda: ToolInterface("", ""))
    dependencies: ToolDependencies = field(default_factory=ToolDependencies)
    validation: ToolValidation = field(default_factory=ToolValidation)
    provenance: ToolProvenance = field(default_factory=ToolProvenance)
    usage_stats: ToolUsageStats = field(default_factory=ToolUsageStats)

    # Status
    status: str = "active"  # active, deprecated, experimental

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> "ToolManifest":
        """Create a ToolManifest from a dictionary (parsed YAML)."""
        # Reconstruct nested dataclasses
        if "interface" in data and isinstance(data["interface"], dict):
            iface = data["interface"]
            params = [ToolParameter(**p) for p in iface.get("parameters", [])]
            data["interface"] = ToolInterface(
                function_name=iface.get("function_name", ""),
                module_path=iface.get("module_path", ""),
                parameters=params,
                return_type=iface.get("return_type", "str"),
                return_description=iface.get("return_description", ""),
            )
        if "dependencies" in data and isinstance(data["dependencies"], dict):
            data["dependencies"] = ToolDependencies(**data["dependencies"])
        if "validation" in data and isinstance(data["validation"], dict):
            data["validation"] = ToolValidation(**data["validation"])
        if "provenance" in data and isinstance(data["provenance"], dict):
            prov = data["provenance"]
            changelog = [ToolChangelogEntry(**c) for c in prov.get("changelog", [])]
            data["provenance"] = ToolProvenance(
                created_at=prov.get("created_at", ""),
                last_modified=prov.get("last_modified", ""),
                source=prov.get("source", "predefined"),
                changelog=changelog,
            )
        if "usage_stats" in data and isinstance(data["usage_stats"], dict):
            data["usage_stats"] = ToolUsageStats(**data["usage_stats"])
        return cls(**data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "ToolManifest":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# ============================================================
# Skill Schema
# ============================================================

@dataclass
class SkillWorkflowStep:
    """A single step in a skill's workflow."""
    step: int
    action: str
    description: str
    tools_required: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    agent: str = "manager"  # manager, dev, critic


@dataclass
class SkillToolDependency:
    """A tool dependency for a skill."""
    tool_id: str
    version: str = ">=1.0.0"
    optional: bool = False


@dataclass
class SkillQualityMetrics:
    """Quality and success tracking for a skill."""
    success_count: int = 0
    failure_count: int = 0
    avg_completion_time_sec: float = 0.0
    avg_critic_score: float = 0.0
    last_used: str = ""
    created_at: str = ""
    last_updated: str = ""

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def total_runs(self) -> int:
        return self.success_count + self.failure_count


@dataclass
class SkillVersionEntry:
    """A changelog entry for skill versioning."""
    version: str
    date: str
    changes: str


@dataclass
class SkillProvenance:
    """Provenance tracking for a skill."""
    created_by: str = "user"  # user, auto_summarizer, imported
    source_run_id: str = ""
    parent_skill_id: Optional[str] = None
    changelog: List[SkillVersionEntry] = field(default_factory=list)


@dataclass
class Skill:
    """Complete skill definition — the core unit replacing templates."""
    skill_id: str
    version: str
    name: str
    domain: str
    description: str

    # Structured workflow
    workflow: List[SkillWorkflowStep] = field(default_factory=list)

    # Tool dependencies
    tools_required: List[SkillToolDependency] = field(default_factory=list)

    # Retrieval metadata
    tags: List[str] = field(default_factory=list)
    applicable_queries: List[str] = field(default_factory=list)  # glob-style patterns

    # Quality tracking
    quality_metrics: SkillQualityMetrics = field(default_factory=SkillQualityMetrics)

    # Provenance
    provenance: SkillProvenance = field(default_factory=SkillProvenance)

    # Status
    status: str = "active"  # active, deprecated, experimental, draft

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        """Create a Skill from a dictionary (parsed YAML)."""
        if "workflow" in data:
            data["workflow"] = [
                SkillWorkflowStep(**s) if isinstance(s, dict) else s
                for s in data["workflow"]
            ]
        if "tools_required" in data:
            data["tools_required"] = [
                SkillToolDependency(**t) if isinstance(t, dict) else t
                for t in data["tools_required"]
            ]
        if "quality_metrics" in data and isinstance(data["quality_metrics"], dict):
            data["quality_metrics"] = SkillQualityMetrics(**data["quality_metrics"])
        if "provenance" in data and isinstance(data["provenance"], dict):
            prov = data["provenance"]
            changelog = [SkillVersionEntry(**c) for c in prov.get("changelog", [])]
            data["provenance"] = SkillProvenance(
                created_by=prov.get("created_by", "user"),
                source_run_id=prov.get("source_run_id", ""),
                parent_skill_id=prov.get("parent_skill_id"),
                changelog=changelog,
            )
        return cls(**data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "Skill":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_all_tool_ids(self) -> List[str]:
        """Get all tool IDs needed (from both tools_required and workflow steps)."""
        tool_ids = set()
        for dep in self.tools_required:
            tool_ids.add(dep.tool_id)
        for step in self.workflow:
            for t in step.tools_required:
                tool_ids.add(t)
        return list(tool_ids)
