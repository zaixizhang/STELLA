# Repository layout and development

All application imports use the `stella` package. From the repository root:

```bash
pip install -e .
python -m stella --help
python -m stella
```

The installed commands `stella` and `stella-web` launch the main interface and
the alternate web launcher respectively. `python -m stella.ui.launcher` is the
module equivalent of `stella-web`.

## Migration from the flat layout

| Previous location or command | Current location or command |
| --- | --- |
| `stella_core.py` | `stella/core.py`; launch with `python -m stella` |
| `from stella_core import initialize_stella` | `from stella.core import initialize_stella` |
| `stella_ui_english.py` | `stella/ui/web.py` |
| `start_stella_web.py` | `python -m stella.ui.launcher` |
| `skill_*.py` | `stella/skills/{manager,store,schema,retriever,summarizer}.py` |
| `predefined_tools.py` | `stella/tools/predefined.py` |
| `tool_governance.py` | `stella/tools/governance.py` |
| `new_tools/*.py` | `stella/tools/biomedical/*.py` |
| `self_evolve/` | `stella/self_evolve/`; run `python -m stella.self_evolve.<module>` |
| `prompts/` | `stella/resources/prompts/` |
| `skills/prebuilt/` | `stella/resources/skills/prebuilt/` |
| `new_tools/manifests/` | `stella/resources/tool_manifests/` |
| `cases/` | `examples/cases/` |
| `Tool_Creation_Benchmark/` | `benchmarks/tool_creation/` |
| `self_evolve/tests/` | `tests/self_evolve/` |
| `asset/` | `docs/assets/`; UI logo in `stella/resources/assets/` |
| `requirements-self-evolve.txt` | `requirements/self-evolve.txt` |

The root `Dockerfile` is retained so the existing release workflow can build
the image without changing its build context or workflow permissions.

Update external scripts that import the old top-level modules. Use module
commands rather than running files inside `stella/` directly. For example:

```python
from stella import initialize_stella
from stella.skills.manager import SkillManager
from stella.self_evolve.dataset_builder import build_datasets
```

## Bundled resources and writable data

The UI logo, prompts, prebuilt workflow skills, and tool manifests ship inside both the wheel
and source distribution. Paths are defined in [`stella/paths.py`](../stella/paths.py).
They do not depend on the directory from which a module happens to be imported.

For a source checkout, the workspace defaults to the repository root. For an
installed wheel, it defaults to the directory where the application is launched.

| Variable | Default | Purpose |
| --- | --- | --- |
| `STELLA_WORKSPACE` | Checkout root or launch directory | Workspace and `.env` location |
| `STELLA_DATA_DIR` | `<workspace>/data` | Writable skills, manifests, generated tools and SQLite database |
| `STELLA_RESOURCE_DIR` | `<workspace>/resource` | Optional downloaded biomedical datasets |
| `STELLA_SELF_EVOLVE_DATA` | `<workspace>/self_evolve_data` | Trajectories and training datasets |

Set these variables before starting Python. On first use, bundled skills and
manifests are copied into `data/skills/` and `data/tool_manifests/`. Existing files
are preserved so recorded metrics and local edits survive restarts. To pick up
updated bundled definitions, compare them with the writable copies and merge
changes explicitly. Generated Python tools are saved to `data/tools/` and loaded
as `stella.tools.generated.<name>`; the installed package is never modified.

If upgrading an existing workspace, copy your locally generated skill YAMLs from
`skills/auto_generated/` into `data/skills/auto_generated/`, and custom tool Python
files from `new_tools/` into `data/tools/`. Update their imports to use `stella.*`.
Copy custom manifests into `data/tool_manifests/` and update their `module_path`
fields to `stella.tools.generated.<name>` (or `stella.tools.biomedical.<name>` for
bundled tools). For modified prebuilt skills, copy them into `data/skills/prebuilt/`.
The existing `data/skill_runs.db`, `resource/`, and `self_evolve_data/` locations
remain valid.

## Validation

These checks do not run paid model calls or GPU training:

```bash
pip install pytest build
python -m pytest -q
python -m tests.self_evolve.test_pipeline
python -m compileall -q stella examples benchmarks tests
python -m build
pip install --force-reinstall --no-deps dist/*.whl
python -I tests/check_installed_package.py
```

The tests cover the self-evolution pipeline, benchmark scoring, package imports,
resource lookup, generated tool loading, and local documentation links. The
pipeline's checks also raise assertions under pytest.

The alternate interface uses Gradio 4.44–5.x; installation constraints exclude
Gradio 6, which removed parameters used by that interface.

The hard benchmark runners require the external BioMLBench task bundles, runner
scripts, and configurations described in the [benchmark guide](../benchmarks/tool_creation/README.md).
Those external assets are not added by the directory reorganization.
