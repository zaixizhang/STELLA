#!/usr/bin/env python3
"""
STELLA Case Study: Novel NK-Cell Negative Regulators in AML
============================================================
Reproduces the NK-cell/AML target-discovery case in the paper (Fig. 3a-c)
from the single prompt reported there. No candidate families, exclusion
lists, scoring formulas or output schema are supplied: STELLA's Manager/Dev/
Critic agents select tools from the Tool Ocean, build the candidate pool,
search the literature and define/apply the evidence-based ranking themselves.

Reproducibility note: the exact gene list can vary between runs (LLM
stochasticity, literature index freshness, evolving templates/tools).

Output: examples/cases/output/nk_aml_negative_regulators_raw.txt  (full agent output)
"""

import os
import sys
import json
import csv as csv_module
import datetime
import re
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / '.env')

print("NK-Cell Negative Regulators in AML — STELLA Discovery")
print("=" * 70)

print("\nInitializing STELLA...")
from stella.core import initialize_stella
if not initialize_stella(use_template=True):
    print("STELLA initialization failed!")
    sys.exit(1)

from stella.core import manager_agent
print(f"Manager Agent: {'Ready' if manager_agent else 'FAILED'}")

# ─────────────────────────────────────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────────────────────────────────────

query = "Identify novel, previously unreported negative regulators of NK-cell function in the context of AML."

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

print("\nRunning STELLA manager_agent...")
print("=" * 70)
start_time = time.time()

try:
    result = manager_agent.run(query)
    execution_time = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"Completed in {execution_time:.1f}s ({execution_time/60:.1f} min)")
    print("=" * 70)

    result_str = str(result) if result is not None else ""

    print(result_str[:5000])

    raw_file = os.path.join(output_dir, 'nk_aml_negative_regulators_raw.txt')
    with open(raw_file, 'w') as f:
        f.write(f"Execution time: {execution_time:.1f}s\n{'='*60}\n\n{result_str}")
    print(f"[OK] Raw: {raw_file}")

except Exception as e:
    import traceback
    print(f"\nError: {e}")
    traceback.print_exc()
    with open(os.path.join(output_dir, 'error.log'), 'w') as f:
        f.write(f"Error: {e}\n")
        traceback.print_exc(file=f)

print("\n" + "=" * 70)
print("NK-AML Negative Regulator Discovery — DONE")
print("=" * 70)
