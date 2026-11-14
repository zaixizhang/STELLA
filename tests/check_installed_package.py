"""Check an installed wheel with ``python -I tests/check_installed_package.py``.

Isolated Python mode ensures the source checkout cannot satisfy package imports.
No API keys, model calls, GPU, or optional biomedical dependencies are required.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    with tempfile.TemporaryDirectory(prefix="stella-wheel-") as directory:
        os.chdir(directory)
        os.environ["STELLA_WORKSPACE"] = directory
        os.environ["STELLA_DATA_DIR"] = str(Path(directory) / "data")
        from stella.paths import BUNDLED_RESOURCES, PACKAGE_DIR, PROMPTS_DIR
        from stella.skills.manager import SkillManager
        from stella.self_evolve.reward import score_trajectory
        from stella.tools.loading import load_tool_module

        assert "site-packages" in PACKAGE_DIR.parts, PACKAGE_DIR
        assert (PROMPTS_DIR / "Stella_prompt_bioml.yaml").is_file()
        assert (BUNDLED_RESOURCES / "assets/stella_logo.png").read_bytes().startswith(b"\x89PNG")
        manager = SkillManager()
        assert len(manager.store.list_all()) == 20
        assert len(manager.tool_index.list_all()) == 82
        assert callable(score_trajectory)
        assert load_tool_module("protein_seq_featurizer") is not None
        assert (Path(directory) / "data/skill_runs.db").is_file()
        subprocess.run([sys.executable, "-I", "-m", "stella", "--help"], check=True,
                       stdout=subprocess.DEVNULL)
        print("Installed wheel: imports, CLI, logo, prompts, 20 skills and 82 tools OK")


if __name__ == "__main__":
    main()
