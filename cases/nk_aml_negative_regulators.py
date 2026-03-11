#!/usr/bin/env python3
"""
STELLA Case Study: Novel NK-Cell Negative Regulators in AML
============================================================
Task: Identify 40 negative regulators of NK-cell function in AML (Acute Myeloid
      Leukemia) via autonomous family-based literature mining and multi-source scoring.

Methodology:
  1. Mine 19 niche receptor/pathway families for candidate genes
  2. Score each candidate on: literature novelty, mechanism evidence,
     NK-cell expression, and AML relevance (PubMed + web)
  3. Rank by composite score; enforce family diversity

Reproducibility note: Exact gene members may vary across runs (~30-35% gene-level
overlap to manually curated benchmarks) due to LLM stochasticity and literature
index freshness. The gene FAMILIES and scoring methodology are deterministic.
An example output is provided in nk_aml_negative_regulators_example.csv.

Output: cases/output/nk_aml_negative_regulators.csv  (rank, gene, scores, rationale)
        cases/output/nk_aml_negative_regulators.txt  (detailed report)
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

print("NK-Cell Negative Regulators in AML — STELLA Discovery")
print("=" * 70)

print("\nInitializing STELLA...")
from stella_core import initialize_stella
if not initialize_stella(use_template=False):
    print("STELLA initialization failed!")
    sys.exit(1)

from stella_core import manager_agent
print(f"Manager Agent: {'Ready' if manager_agent else 'FAILED'}")

# ─────────────────────────────────────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────────────────────────────────────

query = """Identify 40 negative regulators of NK-cell function in AML (Acute Myeloid Leukemia).
Score candidates from PubMed + web and rank by composite score.
SPEED REQUIREMENT: finish within 10 minutes — use parallel calls aggressively.

EXCLUSIONS: TIGIT, TIM3, PD-1, LAG3, NKG2A, LILRB1, LILRB2, LILRB4, TGFB1, LGALS9.

=== STEP 1: Load Tools ===
Call analyze_query_and_load_relevant_tools() for NK biology, AML immune evasion,
inhibitory receptors, GPCR/cAMP, lncRNA.

=== STEP 2: Build Candidate Pool (no tool calls needed — use biomedical knowledge) ===
From your knowledge of immunology, nominate 2-3 specific gene members per family.
Choose members expressed on NK cells or in AML bone marrow but LESS studied than the
headline gene of that family. Output as a Python list called `candidates`.

Families to cover:
A) Butyrophilin BTN3A subfamily — phosphoantigen/NK crosstalk
B) Orphan & adhesion GPCRs on NK cells — ADGRG/ADGRE subfamilies, cAMP-raising
C) KLR niche — KLRE, KLRB, KLRC (non-NKG2A), KLRG, KLRA human orthologs
D) SLAM family (SLAMF1-9) — members with ambiguous/absent activating adaptors in AML
E) CD300 inhibitory family — phospholipid-binding ITIM members
F) CD200 receptor variants — CD200R and decoy isoforms
G) LAIR family — membrane ITIM form + soluble decoy isoform
H) Paired Ig-like receptors — ITIM-bearing isoforms on NK cells
I) CEACAM family — ITIM members engaging AML blasts
J) Siglec family — members binding AML-expressed sialic acids
K) IFN-stimulated negative-feedback genes — IFI44/IFI44L, IFIT, IFI6 subfamilies
L) Adenylate cyclase ADCY isoforms in lymphocytes + inhibitory GNAI subunits
M) Kinesin KIF members for NK lytic granule transport
N) lncRNAs — antisense or cancer-associated, IFN-regulatory (not MALAT1/NEAT1)
O) Scaffold/threshold proteins — THEMIS-like, SASH family
P) Secreted macroglobulin-family immunosuppressors in bone marrow
Q) MS4A family (not MS4A1) expressed on NK/myeloid cells
R) DNA repair genes whose dysfunction alters NKG2D-ligand exposure in AML
S) Munc13/UNC13 or Munc18/STXBP degranulation limiters

=== STEP 3: Score All Candidates in ONE Parallel Batch ===
Build a single tool_calls list with 2 calls per gene:
  - query_pubmed(query="GENE NK cell AML", max_papers=5)
  - multi_source_search(query="GENE inhibitory NK cell mechanism AML", sources="google")

Call execute_tools_in_parallel(tool_calls=tool_calls, max_workers=10) ONCE for all genes.

Then score in Python only (no more tool calls):
  S_lit  = 3 if 0 pubmed hits, 2 if 1-2 hits, 1 if 3-5, 0 if >5
  S_mech = 3 if web result confirms ITIM/inhibitory/cAMP/lncRNA mechanism,
           2 if inferred, 1 if plausible, 0 if none
  S_expr = 2 if pubmed mentions NK cell expression, 1 if inferred, 0 if absent
  S_aml  = 2 if pubmed/web mentions AML or myeloid, 1 if indirect, 0 if none
  composite = S_lit + S_mech + S_expr + S_aml  (max 10)

=== STEP 4: Rank and Select Top 40 ===
Sort by composite score descending. Enforce ≥1 gene per family (A-S).
novelty_status = "novel" if S_lit >= 2 else "reported"

=== STEP 5: Return JSON ===
import json
final_answer(json.dumps([
  {"rank": 1, "gene_symbol": "SYMBOL", "novelty_status": "novel",
   "mechanism_category": "family label",
   "composite_score": 9, "s_lit": 3, "s_mech": 3, "s_expr": 2, "s_aml": 1,
   "rationale": "2-3 sentences on AML+NK suppression mechanism",
   "pubmed_search_query": "SYMBOL NK cell AML",
   "pubmed_result_summary": "brief summary"},
  ...  # all 40
]))
Do NOT use open() — return JSON string only.
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

    # Parse JSON from agent result
    genes = None
    for chunk in [result_str, result_str.strip()]:
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, list) and parsed and 'gene_symbol' in parsed[0]:
                genes = parsed
                break
        except Exception:
            pass
    if genes is None:
        m = re.search(r'(\[.*\])', result_str, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, list) and parsed and 'gene_symbol' in parsed[0]:
                    genes = parsed
            except Exception:
                pass

    if genes:
        for i, g in enumerate(genes):
            g['rank'] = i + 1

        fieldnames = ['rank', 'gene_symbol', 'novelty_status', 'mechanism_category',
                      'composite_score', 's_lit', 's_mech', 's_expr', 's_aml',
                      'rationale', 'pubmed_search_query', 'pubmed_result_summary']

        csv_file = os.path.join(output_dir, 'nk_aml_negative_regulators.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(genes)
        print(f"[OK] CSV: {csv_file} ({os.path.getsize(csv_file):,} bytes)")

        txt_file = os.path.join(output_dir, 'nk_aml_negative_regulators.txt')
        with open(txt_file, 'w') as f:
            f.write("STELLA NK-AML NEGATIVE REGULATOR DISCOVERY\n")
            f.write(f"Date: {datetime.date.today()}\n")
            f.write("=" * 60 + "\n\n")
            for g in genes:
                f.write(f"GENE {g['rank']}: {g['gene_symbol']}\n")
                f.write(f"Novelty:  {g.get('novelty_status','')}\n")
                f.write(f"Category: {g.get('mechanism_category','')}\n")
                f.write(f"Score:    {g.get('composite_score','')} "
                        f"(lit={g.get('s_lit','')} mech={g.get('s_mech','')} "
                        f"expr={g.get('s_expr','')} aml={g.get('s_aml','')})\n")
                f.write(f"Rationale: {g.get('rationale','')}\n")
                f.write(f"Evidence:  {g.get('pubmed_result_summary','')}\n")
                f.write("-" * 60 + "\n")
        print(f"[OK] TXT: {txt_file} ({os.path.getsize(txt_file):,} bytes)")

        print("\nTop 40 discovered genes:")
        print(f"{'Rank':>4}  {'Gene':<14} {'Novelty':<10} {'Score':>5}  Category")
        print("-" * 70)
        for g in genes:
            print(f"{g['rank']:>4}  {g['gene_symbol']:<14} "
                  f"{g.get('novelty_status',''):<10} "
                  f"{str(g.get('composite_score','')):>5}  "
                  f"{g.get('mechanism_category','')}")
    else:
        print("Could not parse JSON from agent result.")
        print(result_str[:3000])

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
