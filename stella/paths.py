"""Shared paths for bundled resources and writable workspace data.

Source checkouts use the repository as their workspace. Installed packages use
the current working directory. Environment variables can override either path.
"""

import os
import shutil
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
_checkout = PACKAGE_DIR.parent
WORKSPACE_DIR = Path(os.environ.get(
    "STELLA_WORKSPACE", _checkout if (_checkout / "pyproject.toml").is_file() else Path.cwd()
)).expanduser().resolve()
DATA_DIR = Path(os.environ.get("STELLA_DATA_DIR", WORKSPACE_DIR / "data")).expanduser().resolve()
RESOURCE_DIR = Path(os.environ.get("STELLA_RESOURCE_DIR", WORKSPACE_DIR / "resource")).expanduser().resolve()
BUNDLED_RESOURCES = PACKAGE_DIR / "resources"
PROMPTS_DIR = BUNDLED_RESOURCES / "prompts"
SKILLS_DIR = DATA_DIR / "skills"
MANIFESTS_DIR = DATA_DIR / "tool_manifests"
GENERATED_TOOLS_DIR = DATA_DIR / "tools"
BIOMEDICAL_TOOLS_DIR = PACKAGE_DIR / "tools" / "biomedical"


def seed_directory(source: Path, destination: Path) -> None:
    """Copy bundled defaults into writable storage without replacing user edits."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(path, target)
