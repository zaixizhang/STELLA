"""
STELLA: Self-Evolving Multimodal Agents for Biomedical Research

A multi-agent AI framework for autonomous biomedical research tasks,
featuring self-evolving knowledge bases, dynamic tool creation, and
integrated biomedical databases.

Quick start:
    from stella import MemoryManager, KnowledgeBase
    # Then run: stella-web  (to launch the Gradio interface)

GitHub: https://github.com/zaixizhang/STELLA
"""

__version__ = "0.1.0"
__author__ = "Zaixi Zhang"
__license__ = "Apache 2.0"

# Public API — lazy imports to avoid heavy dependencies at import time
def __getattr__(name):
    if name in ("MemoryManager", "BaseMemoryComponent",
                "KnowledgeMemory", "CollaborationMemory", "SessionMemory"):
        from memory_manager import (
            MemoryManager, BaseMemoryComponent,
            KnowledgeMemory, CollaborationMemory, SessionMemory,
        )
        return locals()[name]

    if name in ("KnowledgeBase", "Mem0EnhancedKnowledgeBase"):
        from Knowledge_base import KnowledgeBase, Mem0EnhancedKnowledgeBase
        return locals()[name]

    if name == "AutoMemory":
        from stella_core import AutoMemory
        return AutoMemory

    raise AttributeError(f"module 'stella' has no attribute {name!r}")


__all__ = [
    "__version__",
    "MemoryManager",
    "BaseMemoryComponent",
    "KnowledgeMemory",
    "CollaborationMemory",
    "SessionMemory",
    "KnowledgeBase",
    "Mem0EnhancedKnowledgeBase",
    "AutoMemory",
]
