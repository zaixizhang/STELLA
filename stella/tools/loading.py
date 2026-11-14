"""Load bundled and generated tools using their full package names."""

import importlib
import importlib.util
import sys

from stella.paths import GENERATED_TOOLS_DIR


def load_tool_module(name: str):
    if not name.isidentifier():
        raise ValueError(f"Invalid tool module name: {name!r}")
    path = GENERATED_TOOLS_DIR / f"{name}.py"
    if not path.is_file():
        return importlib.import_module(f"stella.tools.biomedical.{name}")
    module_name = f"stella.tools.generated.{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module
