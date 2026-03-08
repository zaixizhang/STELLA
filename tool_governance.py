"""
STELLA Tool Governance
Tool Index, validation, versioning, dependency management, and reproducible environments.
"""

import os
import re
import ast
import yaml
import json
import importlib
import importlib.util
import inspect
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

from skill_schema import (
    ToolManifest,
    ToolInterface,
    ToolParameter,
    ToolDependencies,
    ToolProvenance,
    ToolChangelogEntry,
    ToolValidation,
    ToolUsageStats,
)

STELLA_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFESTS_DIR = os.path.join(STELLA_DIR, "new_tools", "manifests")
NEW_TOOLS_DIR = os.path.join(STELLA_DIR, "new_tools")


class ToolIndex:
    """
    Centralized, governed tool registry.
    Replaces the old dynamic_tools_registry dict with structured governance.
    """

    def __init__(self, manifests_dir: str = MANIFESTS_DIR):
        self.manifests_dir = manifests_dir
        os.makedirs(manifests_dir, exist_ok=True)

        # tool_id -> ToolManifest
        self._tools: Dict[str, ToolManifest] = {}
        # tool_id -> file path of manifest
        self._manifest_paths: Dict[str, str] = {}
        # Dependency graph: package -> set of tool_ids
        self._dep_graph: Dict[str, Set[str]] = defaultdict(set)

        # Load manifests
        self._load_all_manifests()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_all_manifests(self):
        """Load all tool manifest YAML files from manifests directory."""
        if not os.path.exists(self.manifests_dir):
            return

        for fname in os.listdir(self.manifests_dir):
            if not fname.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(self.manifests_dir, fname)
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)

                if data is None:
                    continue

                # Handle grouped manifests (multiple tools in one file)
                if "tools" in data and isinstance(data["tools"], list):
                    for tool_data in data["tools"]:
                        self._register_from_dict(tool_data, path)
                else:
                    # Single tool manifest
                    self._register_from_dict(data, path)
            except Exception as e:
                print(f"Warning: failed to load manifest {path}: {e}")

    def _register_from_dict(self, data: dict, path: str):
        """Register a tool from a parsed manifest dict."""
        try:
            manifest = ToolManifest.from_dict(data)
            self._tools[manifest.tool_id] = manifest
            self._manifest_paths[manifest.tool_id] = path
            # Build dependency graph
            for pkg in manifest.dependencies.python_packages:
                pkg_name = re.split(r"[><=!]", pkg)[0].strip()
                self._dep_graph[pkg_name].add(manifest.tool_id)
        except Exception as e:
            tool_id = data.get("tool_id", "unknown")
            print(f"Warning: failed to parse manifest for {tool_id}: {e}")

    def reload(self):
        """Reload all manifests from disk."""
        self._tools.clear()
        self._manifest_paths.clear()
        self._dep_graph.clear()
        self._load_all_manifests()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, manifest: ToolManifest, save: bool = True) -> bool:
        """Register a tool with optional persistence."""
        self._tools[manifest.tool_id] = manifest
        # Build dep graph
        for pkg in manifest.dependencies.python_packages:
            pkg_name = re.split(r"[><=!]", pkg)[0].strip()
            self._dep_graph[pkg_name].add(manifest.tool_id)

        if save:
            path = os.path.join(self.manifests_dir, f"{manifest.tool_id}.yaml")
            manifest.save_yaml(path)
            self._manifest_paths[manifest.tool_id] = path
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get(self, tool_id: str) -> Optional[ToolManifest]:
        return self._tools.get(tool_id)

    def list_all(self, category: str = None, status: str = None) -> List[ToolManifest]:
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        if status:
            tools = [t for t in tools if t.status == status]
        return tools

    def search(self, query: str, category: str = None, max_results: int = 10) -> List[ToolManifest]:
        """Search tools by query string matching name/description."""
        query_lower = query.lower()
        query_tokens = set(re.findall(r"\w+", query_lower))

        scored = []
        for tid, manifest in self._tools.items():
            if manifest.status == "deprecated":
                continue
            if category and manifest.category != category:
                continue
            # Score by token overlap
            text = f"{manifest.name} {manifest.description} {manifest.category}".lower()
            text_tokens = set(re.findall(r"\w+", text))
            overlap = len(query_tokens & text_tokens)
            if overlap > 0:
                scored.append((manifest, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:max_results]]

    def get_tools_for_skill(self, skill) -> Dict[str, Optional[ToolManifest]]:
        """Resolve all tools needed for a skill. Returns {tool_id: manifest_or_None}."""
        result = {}
        for tool_id in skill.get_all_tool_ids():
            result[tool_id] = self._tools.get(tool_id)
        return result

    def check_skill_tool_availability(self, skill) -> Tuple[List[str], List[str]]:
        """Check tool availability for a skill. Returns (available, missing)."""
        available = []
        missing = []
        for tool_id in skill.get_all_tool_ids():
            if tool_id in self._tools:
                available.append(tool_id)
            else:
                missing.append(tool_id)
        return available, missing

    # ------------------------------------------------------------------
    # Tool Usage Tracking
    # ------------------------------------------------------------------
    def record_tool_use(self, tool_id: str, success: bool, execution_time_ms: float = 0):
        """Record a tool execution for governance stats."""
        manifest = self._tools.get(tool_id)
        if not manifest:
            return

        stats = manifest.usage_stats
        stats.total_calls += 1
        if success:
            stats.success_count += 1
        else:
            stats.failure_count += 1

        # Update running average execution time
        if execution_time_ms > 0 and stats.total_calls > 0:
            stats.avg_execution_time_ms = (
                (stats.avg_execution_time_ms * (stats.total_calls - 1) + execution_time_ms)
                / stats.total_calls
            )

    # ------------------------------------------------------------------
    # Dependency Management
    # ------------------------------------------------------------------
    def check_dependency_conflicts(self, new_packages: List[str]) -> Dict[str, Any]:
        """
        Check if new packages conflict with existing tool dependencies.

        Returns:
            {
                "conflicts": [{"package": ..., "new_version": ..., "existing_version": ..., "tools": [...]}],
                "safe": bool
            }
        """
        conflicts = []
        existing_packages = {}

        # Collect all existing package version specs
        for manifest in self._tools.values():
            for pkg_spec in manifest.dependencies.python_packages:
                pkg_name = re.split(r"[><=!]", pkg_spec)[0].strip()
                if pkg_name not in existing_packages:
                    existing_packages[pkg_name] = []
                existing_packages[pkg_name].append({
                    "spec": pkg_spec,
                    "tool_id": manifest.tool_id,
                })

        # Check new packages against existing
        for new_spec in new_packages:
            pkg_name = re.split(r"[><=!]", new_spec)[0].strip()
            if pkg_name in existing_packages:
                existing = existing_packages[pkg_name]
                # Simple conflict detection: different version specs
                existing_specs = set(e["spec"] for e in existing)
                if new_spec not in existing_specs and len(existing_specs) > 0:
                    conflicts.append({
                        "package": pkg_name,
                        "new_spec": new_spec,
                        "existing_specs": list(existing_specs),
                        "affected_tools": [e["tool_id"] for e in existing],
                    })

        return {
            "conflicts": conflicts,
            "safe": len(conflicts) == 0,
        }

    def get_dependency_report(self) -> str:
        """Generate a human-readable dependency report."""
        all_packages = defaultdict(set)
        for manifest in self._tools.values():
            for pkg_spec in manifest.dependencies.python_packages:
                pkg_name = re.split(r"[><=!]", pkg_spec)[0].strip()
                all_packages[pkg_name].add(manifest.tool_id)

        lines = [f"Tool Dependency Report ({len(all_packages)} packages, {len(self._tools)} tools)", ""]

        # Sort by number of tools using each package
        for pkg, tools in sorted(all_packages.items(), key=lambda x: len(x[1]), reverse=True):
            lines.append(f"  {pkg}: used by {len(tools)} tools")

        return "\n".join(lines)

    def export_environment(self, tool_ids: List[str] = None, format: str = "requirements") -> str:
        """
        Export reproducible environment specification for given tools.

        Args:
            tool_ids: specific tools to include (None = all active tools)
            format: "requirements" (pip) or "conda" (environment.yml)

        Returns:
            File content string.
        """
        packages = set()
        channels = set()

        tools = (
            [self._tools[tid] for tid in tool_ids if tid in self._tools]
            if tool_ids
            else [t for t in self._tools.values() if t.status == "active"]
        )

        for manifest in tools:
            for pkg in manifest.dependencies.python_packages:
                packages.add(pkg)
            for ch in manifest.dependencies.conda_channels:
                channels.add(ch)

        if format == "requirements":
            lines = [
                "# Auto-generated by STELLA Tool Governance",
                f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"# Tools included: {len(tools)}",
                "",
            ]
            for pkg in sorted(packages):
                lines.append(pkg)
            return "\n".join(lines)

        elif format == "conda":
            env = {
                "name": "stella_tools",
                "channels": sorted(channels) or ["conda-forge", "defaults"],
                "dependencies": sorted(packages),
            }
            return yaml.dump(env, default_flow_style=False)

        return ""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_tool(self, tool_id: str) -> Dict[str, Any]:
        """
        Validate a tool against its manifest.

        Checks:
        1. Function exists at declared module_path
        2. Function signature matches manifest parameters
        3. Dependencies are importable
        """
        manifest = self._tools.get(tool_id)
        if not manifest:
            return {"valid": False, "errors": [f"Tool {tool_id} not found in index"]}

        errors = []
        warnings = []

        # Check 1: Module and function exist
        iface = manifest.interface
        if iface.module_path and iface.function_name:
            try:
                module_path = iface.module_path.replace("/", ".")
                if module_path.endswith(".py"):
                    module_path = module_path[:-3]
                mod = importlib.import_module(module_path)
                func = getattr(mod, iface.function_name, None)
                if func is None:
                    errors.append(f"Function '{iface.function_name}' not found in {iface.module_path}")
                else:
                    # Check 2: Signature match
                    sig = inspect.signature(func)
                    manifest_params = {p.name for p in iface.parameters}
                    actual_params = set(sig.parameters.keys())
                    missing = manifest_params - actual_params
                    extra = actual_params - manifest_params
                    if missing:
                        warnings.append(f"Manifest declares params not in function: {missing}")
                    if extra:
                        warnings.append(f"Function has params not in manifest: {extra}")
            except ImportError as e:
                errors.append(f"Cannot import module {iface.module_path}: {e}")
            except Exception as e:
                errors.append(f"Validation error: {e}")

        # Check 3: Key dependencies importable
        for pkg_spec in manifest.dependencies.python_packages[:5]:  # Check top 5
            pkg_name = re.split(r"[><=!]", pkg_spec)[0].strip()
            # Map pip package names to import names
            import_map = {
                "googlesearch-python": "googlesearch",
                "beautifulsoup4": "bs4",
                "scikit-learn": "sklearn",
                "PyPDF2": "PyPDF2",
                "python-dotenv": "dotenv",
            }
            import_name = import_map.get(pkg_name, pkg_name)
            try:
                importlib.import_module(import_name)
            except ImportError:
                warnings.append(f"Package '{pkg_name}' not importable")

        # Update validation status
        status = "passed" if not errors else "failed"
        manifest.validation.validation_status = status
        manifest.validation.last_validated = datetime.now().strftime("%Y-%m-%d")

        return {
            "valid": len(errors) == 0,
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "tool_id": tool_id,
        }

    def validate_all(self) -> Dict[str, Any]:
        """Validate all registered tools."""
        results = {}
        passed = 0
        failed = 0
        for tool_id in self._tools:
            result = self.validate_tool(tool_id)
            results[tool_id] = result
            if result["valid"]:
                passed += 1
            else:
                failed += 1
        return {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "results": results,
        }

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------
    def version_tool(self, tool_id: str, new_version: str, changes: str) -> bool:
        """Bump a tool's version."""
        manifest = self._tools.get(tool_id)
        if not manifest:
            return False

        entry = ToolChangelogEntry(
            version=new_version,
            date=datetime.now().strftime("%Y-%m-%d"),
            changes=changes,
        )
        manifest.provenance.changelog.append(entry)
        manifest.version = new_version
        manifest.provenance.last_modified = datetime.now().strftime("%Y-%m-%d")

        # Save
        if tool_id in self._manifest_paths:
            manifest.save_yaml(self._manifest_paths[tool_id])
        return True

    def deprecate_tool(self, tool_id: str, replacement_id: str = None, reason: str = "") -> bool:
        """Mark a tool as deprecated."""
        manifest = self._tools.get(tool_id)
        if not manifest:
            return False

        manifest.status = "deprecated"
        msg = f"Deprecated"
        if reason:
            msg += f": {reason}"
        if replacement_id:
            msg += f". Replaced by {replacement_id}"

        entry = ToolChangelogEntry(
            version=manifest.version,
            date=datetime.now().strftime("%Y-%m-%d"),
            changes=msg,
        )
        manifest.provenance.changelog.append(entry)

        if tool_id in self._manifest_paths:
            manifest.save_yaml(self._manifest_paths[tool_id])
        return True

    # ------------------------------------------------------------------
    # Auto-manifest generation
    # ------------------------------------------------------------------
    def auto_generate_manifest(
        self,
        function_name: str,
        module_path: str,
        category: str = "general",
        version: str = "1.0.0",
    ) -> Optional[ToolManifest]:
        """
        Auto-generate a manifest from a Python function.
        Inspects the function signature and docstring.
        """
        try:
            mod_import = module_path.replace("/", ".").replace(".py", "")
            mod = importlib.import_module(mod_import)
            func = getattr(mod, function_name, None)
            if func is None:
                return None

            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or ""

            # Parse parameters
            params = []
            for pname, param in sig.parameters.items():
                ptype = "Any"
                if param.annotation != inspect.Parameter.empty:
                    ptype = getattr(param.annotation, "__name__", str(param.annotation))
                default = None
                required = True
                if param.default != inspect.Parameter.empty:
                    default = param.default
                    required = False
                params.append(ToolParameter(
                    name=pname,
                    type=ptype,
                    required=required,
                    default=default,
                ))

            manifest = ToolManifest(
                tool_id=function_name,
                version=version,
                name=function_name.replace("_", " ").title(),
                description=doc.split("\n")[0] if doc else function_name,
                category=category,
                interface=ToolInterface(
                    function_name=function_name,
                    module_path=module_path,
                    parameters=params,
                ),
                provenance=ToolProvenance(
                    created_at=datetime.now().strftime("%Y-%m-%d"),
                    last_modified=datetime.now().strftime("%Y-%m-%d"),
                    source="auto_generated",
                    changelog=[ToolChangelogEntry(
                        version=version,
                        date=datetime.now().strftime("%Y-%m-%d"),
                        changes="Auto-generated manifest",
                    )],
                ),
            )
            return manifest

        except Exception as e:
            print(f"Auto-manifest generation failed for {function_name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def get_summary(self) -> str:
        """Human-readable summary of the tool index."""
        active = [t for t in self._tools.values() if t.status == "active"]
        deprecated = [t for t in self._tools.values() if t.status == "deprecated"]
        categories = defaultdict(int)
        for t in active:
            categories[t.category] += 1

        lines = [
            f"Tool Index: {len(active)} active, {len(deprecated)} deprecated",
            "Categories: " + ", ".join(f"{cat}({n})" for cat, n in sorted(categories.items())),
        ]

        # Top tools by usage
        by_usage = sorted(active, key=lambda t: t.usage_stats.total_calls, reverse=True)[:5]
        if by_usage and by_usage[0].usage_stats.total_calls > 0:
            lines.append("Top tools by usage:")
            for t in by_usage:
                sr = f"{t.usage_stats.success_rate:.0%}" if t.usage_stats.total_calls > 0 else "N/A"
                lines.append(f"  - {t.tool_id} v{t.version}: {t.usage_stats.total_calls} calls, success={sr}")

        return "\n".join(lines)
