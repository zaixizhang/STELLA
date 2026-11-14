"""Regression checks for package paths, writable state and documented entry points."""

import ast
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run_isolated(code, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), STELLA_WORKSPACE=str(tmp_path),
               STELLA_DATA_DIR=str(tmp_path / "data"))
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_resources_and_state_outside_checkout(tmp_path):
    run_isolated('''
from pathlib import Path
from stella.paths import DATA_DIR, PROMPTS_DIR, BUNDLED_RESOURCES
from stella.skills.manager import SkillManager
assert (PROMPTS_DIR / "Stella_prompt_bioml.yaml").is_file()
assert (BUNDLED_RESOURCES / "assets/stella_logo.png").read_bytes().startswith(bytes([137]) + b"PNG")
before = {p: p.read_bytes() for p in BUNDLED_RESOURCES.rglob("*.yaml")}
manager = SkillManager()
assert len(manager.store.list_all()) == 20
assert len(manager.tool_index.list_all()) > 0
skill = manager.store.list_all()[0]
manager.store.record_run(skill.skill_id, success=True, query="offline regression")
saved = Path(manager.store.save(skill))
assert saved.is_relative_to(DATA_DIR)
manager2 = SkillManager()
assert manager2.store.get(skill.skill_id).quality_metrics.success_count == skill.quality_metrics.success_count
assert all(p.read_bytes() == data for p, data in before.items())
assert (DATA_DIR / "skill_runs.db").is_file()
''', tmp_path)


def test_generated_tools_support_package_imports(tmp_path):
    run_isolated('''
from stella.paths import GENERATED_TOOLS_DIR
from stella.tools.loading import load_tool_module
from stella.tools.governance import ToolIndex
GENERATED_TOOLS_DIR.mkdir(parents=True)
(GENERATED_TOOLS_DIR / "helper.py").write_text("VALUE = 42\\n")
(GENERATED_TOOLS_DIR / "example.py").write_text(
    "from .helper import VALUE\\ndef answer() -> int:\\n    return VALUE\\n")
module = load_tool_module("example")
assert module.answer() == 42
assert module.__name__ == "stella.tools.generated.example"
index = ToolIndex()
manifest = index.auto_generate_manifest("answer", module.__name__)
assert manifest is not None
index.register(manifest)
assert index.validate_tool("answer")["valid"]
assert load_tool_module("protein_seq_featurizer").__name__ == "stella.tools.biomedical.protein_seq_featurizer"
''', tmp_path)


def test_tool_manifest_modules_and_functions_exist():
    for path in (ROOT / "stella/resources/tool_manifests").glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        for tool in data.get("tools", [data]):
            interface = tool["interface"]
            spec = importlib.util.find_spec(interface["module_path"])
            assert spec is not None, (path, interface["module_path"])
            tree = ast.parse(Path(spec.origin).read_text())
            names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            assert interface["function_name"] in names, (path, interface["function_name"])


def test_benchmark_default_answer_key_is_readable():
    from benchmarks.tool_creation.simple_set.score_results import DEFAULT_GROUND_TRUTH_CSV, load_ground_truth
    assert len(load_ground_truth(DEFAULT_GROUND_TRUTH_CSV)) > 0


def test_cli_help_without_api_key(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run([sys.executable, "-m", "stella", "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--enable_tool_creation" in result.stdout
    assert "--disable_tool_creation" in result.stdout


def test_local_documentation_links():
    paths = [ROOT / "README.md", *ROOT.glob("docs/*.md"),
             ROOT / "stella-rsi/README.md",
             ROOT / "docker/README.md", ROOT / "benchmarks/tool_creation/README.md"]
    for path in paths:
        text = path.read_text()
        links = re.findall(r"\]\(([^)]+)\)", text) + re.findall(r'<img[^>]+src="([^"]+)"', text)
        for link in links:
            if re.match(r"[a-z]+:", link) or link.startswith("#"):
                continue
            target = link.split("#", 1)[0]
            assert (path.parent / target).exists(), f"{path.relative_to(ROOT)}: {target}"
