"""STELLA: Self-Evolving Multimodal Agents for Biomedical Research.

Run ``python -m stella`` or use ``from stella import initialize_stella``.
Core imports are lazy so offline data processing needs no API key.
"""

__version__ = "0.1.0"
__author__ = "Zaixi Zhang"
__license__ = "Apache 2.0"


def __getattr__(name):
    if name in ("initialize_stella", "AutoMemory"):
        from stella import core
        return getattr(core, name)
    if name == "SkillManager":
        from stella.skills.manager import SkillManager
        return SkillManager
    raise AttributeError(f"module 'stella' has no attribute {name!r}")


__all__ = ["__version__", "initialize_stella", "AutoMemory", "SkillManager"]
