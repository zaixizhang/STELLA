"""Run after installing STELLA and stella-rsi; each invocation starts a session."""

import argparse

from stella.core import initialize_stella
from stella_rsi.capture import InteractionRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="stella-rsi/data/raw")
    args = parser.parse_args()
    manager = initialize_stella(use_template=True, enable_tool_creation=False)
    if manager is None:
        raise RuntimeError("STELLA could not initialize")
    recorder = InteractionRecorder(manager, args.output_dir)
    print(f"Session {recorder.session_id}; type /quit to end.")
    while True:
        try:
            task = input("You: ").strip()
        except EOFError:
            break
        if task == "/quit":
            break
        if task:
            print(recorder.run(task, reset=False))


if __name__ == "__main__":
    main()
