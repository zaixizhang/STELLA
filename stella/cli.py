"""Command-line entry point; parse options before loading models and tools."""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="STELLA biomedical research assistant")
    parser.add_argument("--use_template", action="store_true", default=True,
                        help="Enable workflow skills (default)")
    parser.add_argument("--no_template", action="store_true", help="Disable workflow skills")
    parser.add_argument("--use_mem0", action="store_true", help="Enable optional Mem0 memory")
    parser.add_argument("--use_default_prompts", action="store_true",
                        help="Use default smolagents prompts")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--enable_tool_creation", action="store_true",
                        help="Enable dynamic tool creation (default: disabled)")
    parser.add_argument("--disable_tool_creation", action="store_true",
                        help="Disable tool creation, overriding --enable_tool_creation")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from stella.core import main as run
    return run(args)
