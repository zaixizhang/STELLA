#!/usr/bin/env python3
"""
STELLA Case Study: Strictosidine Synthase (RsSTR, UniProt P68175) Round 2
==========================================================================
Reproduces the second round of the multi-round directed-evolution case in the
paper (Fig. 4). Round 1 is initiated by the paper's prompt alone; for Round 2
the only additional input is the Round 1 wet-lab HPLC measurements, which
STELLA uses as feedback (fine-tuning its ESM2 activity model), as in the paper.
No hot-spot hints, mandatory variants or dead-zone rules are supplied: STELLA
derives the workflow (structure modelling, pocket identification, library
construction, ESM2 prediction, FoldX/RosettaDDG filtering, Critic review) from
its template library and tools.

Reproducibility note: exact variants can vary between runs (LLM stochasticity,
model availability, evolving templates/tools).

Output: cases/output/strictosidine_synthase_round2_raw.txt  (full agent output)
"""

import os
import sys
import json
import csv as csv_module
import datetime
import re
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

print("Strictosidine Synthase Round 2 Optimization — STELLA")
print("=" * 70)

print("\nInitializing STELLA...")
from stella_core import initialize_stella
if not initialize_stella(use_template=True):
    print("STELLA initialization failed!")
    sys.exit(1)

from stella_core import manager_agent
print(f"Manager Agent: {'Ready' if manager_agent else 'FAILED'}")

# ─────────────────────────────────────────────────────────────────────────────
# ROUND 1 WET-LAB DATA (experimental feedback)
# ─────────────────────────────────────────────────────────────────────────────

ROUND1_DATA = """
WT baseline: Area1=150.5, Area2=276.9, Area3=257.5

Variant       Area1   Area2   Area3   Status
W149F         0       0       0       Dead
V176W         147.6   42.3    42.1    Significantly Reduced
V176R         0       0       0       Dead
V176Q         16.8    19.9    19.9    Significantly Reduced
V176L         29.9    36.2    35.7    Significantly Reduced
V176K         0       0       0       Dead
P253V         0       0       0       Dead
M276R         258.9   448.5   347.6   IMPROVED (HIT)
M276F         41.6    50.7    50.6    Reduced
M180Y         0       0       5.5     Dead
M180I         19.7    23.8    36.2    Significantly Reduced
M180F         30.7    45.4    50.6    Significantly Reduced
I179V         18.4    29.5    18.8    Significantly Reduced
H307R         0       0       0       Dead
H307F         0       17.5    15.3    Significantly Reduced
G210V         0       0       0       Dead
G210T         0       0       0       Dead
G210L         0       0       0       Dead
G210A         7.4     5.6     0       Dead
F226L         5.3     5.7     8       Significantly Reduced
E306V         41.7    45.2    42.2    Reduced
E306R         35.2    73.7    73.8    Reduced
E306L         23      69.4    68.4    Reduced
E306I         28.5    33.2    23.7    Reduced
E306F         35.9    45.9    56.7    Reduced
E271R         0       6.1     10.5    Dead
E271N         20.6    20.8    21.7    Significantly Reduced
E271K         0       0       5.3     Dead
E271I         0       0       0       Dead
E271D         17.2    9.1     21      Significantly Reduced
"""

# ─────────────────────────────────────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────────────────────────────────────

query = f"""Optimize the catalytic activity of strictosidine synthase (UniProt ID: P68175) for its natural reaction involving tryptamine and secologanin.

Round 1 wet-lab results (HPLC peak areas, three replicates; WT = reference) are provided as experimental feedback:
{ROUND1_DATA}
"""

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

    raw_file = os.path.join(output_dir, 'strictosidine_synthase_round2_raw.txt')
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
print("Strictosidine Synthase Round 2 Optimization — DONE")
print("=" * 70)
