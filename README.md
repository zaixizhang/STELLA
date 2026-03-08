# STELLA: Self-Evolving Intelligent Laboratory Assistant

<div align="center">

![STELLA Logo](Stella.png)

**Multi-Agent AI Research Assistant for Biomedical Science**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Framework](https://img.shields.io/badge/Framework-Smolagents-orange.svg)
![AI](https://img.shields.io/badge/AI-Multi--Agent-purple.svg)

</div>

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=sk-or-v1-...       # Required: OpenRouter API key for LLM access
SERPAPI_API_KEY=...                    # Optional: SerpAPI for enhanced web search
```

All LLM calls are routed through [OpenRouter](https://openrouter.ai/), so a single API key gives access to GPT, Gemini, Claude, and other models.

### 3. Download resources (optional)

Large biomedical datasets (~9.7 GB) are hosted on Google Drive. These are only needed for tasks that query local datasets (e.g., gene expression, coexpression networks).

[Download STELLA Resources](https://drive.google.com/file/d/1n8I-gkM58wL1OZysdpYhr-Q9ZApd9yU4/view?usp=sharing)

Extract to `resource/` in the project root.

### 4. Launch STELLA

**Web UI (recommended):**
```bash
python3 stella_core.py
```
Open http://localhost:7860 in your browser. A public Gradio link is also generated automatically.

**With options:**
```bash
# Disable skill retrieval
python3 stella_core.py --no_template

# Enable dynamic tool creation agent
python3 stella_core.py --enable_tool_creation

# Custom port
python3 stella_core.py --port 8080

# Use default smolagents prompts instead of custom STELLA prompts
python3 stella_core.py --use_default_prompts
```

**Alternative launchers:**
```bash
python3 start_stella_web.py     # Web launcher with preset config
python3 start_stella_basic.py   # Minimal launcher
```

### 5. Use STELLA programmatically

```python
from stella_core import initialize_stella

manager_agent = initialize_stella(
    use_template=True,
    enable_tool_creation=True,
)
result = manager_agent.run("What genes are associated with breast cancer resistance?")
```

---

## Architecture

```
                         User Query
                             |
                     +-------v--------+
                     | Manager Agent  |
                     |  - query analysis
                     |  - skill retrieval
                     |  - tool selection
                     |  - task delegation
                     +--+-----+-----+-+
                        |     |     |
           +------------+  +--+--+  +------------+
           |               |     |               |
    +------v------+  +-----v---+ +------v------+ |
    |  Dev Agent  |  | Critic  | | Tool Create | |
    |  - code exec|  | Agent   | |   Agent     | |
    |  - analysis |  | - eval  | | - new tools | |
    +------+------+  +----+----+ +------+------+ |
           |              |             |         |
    +------v--------------v-------------v---------v----+
    |                  Tool Ocean                       |
    |  predefined_tools.py  |  new_tools/*.py           |
    |  (search, shell, I/O) |  (databases, screening)  |
    +---------------------------+-----------------------+
                                |
    +---------------------------v-----------------------+
    |               Skill System                        |
    |  SkillStore -> SkillRetriever -> SkillSummarizer  |
    |  skills/prebuilt/  |  skills/auto_generated/      |
    +---------------------------------------------------+
```

### Multi-agent roles

| Agent | Default Model | Responsibility |
|-------|--------------|----------------|
| Manager | Configurable (OpenRouter) | Query analysis, skill retrieval, tool selection, task delegation |
| Dev | Configurable (OpenRouter) | Code execution, data analysis, environment management |
| Critic | Configurable (OpenRouter) | Quality evaluation, scoring, improvement suggestions |
| Tool Creation | Configurable (OpenRouter) | Dynamic tool development, validation, testing |

Models are configurable via OpenRouter. Edit `stella_core.py` model variables to switch providers (GPT, Claude, Gemini, etc.).

### Skill system

STELLA uses a structured **skill management system** to learn from experience:

1. **Skill retrieval** -- Before each task, the manager retrieves relevant skills from the store using 3-stage hybrid retrieval (tag matching, TF-IDF similarity, quality-weighted re-ranking).
2. **Skill-guided execution** -- Retrieved skills provide workflow steps, recommended tools, and past success patterns to guide the agent.
3. **Skill creation** -- After a successful run (critic score >= 0.6), the summarizer auto-extracts a new skill or updates an existing one (dedup threshold 0.85).
4. **Prebuilt skills** -- Curated YAML skill files in `skills/prebuilt/` cover common biomedical workflows.

### Tool governance

Each tool has a YAML manifest in `new_tools/manifests/` that records:
- Interface definition (parameters, return type)
- Dependencies and version constraints
- Validation rules and test commands
- Provenance (author, creation date, changelog)
- Usage statistics

The `ToolIndex` class validates tools against their manifests, detects dependency conflicts, and can export reproducible environments.

---

## Project Structure

```
STELLA/
|-- stella_core.py                  # Main entry point & multi-agent orchestration
|-- predefined_tools.py             # Core system tools (search, shell, file I/O)
|-- skill_manager.py                # Unified skill management interface
|-- skill_store.py                  # Skill persistence (YAML + SQLite)
|-- skill_retriever.py              # 3-stage hybrid skill retrieval
|-- skill_summarizer.py             # Auto-create skills from successful runs
|-- skill_schema.py                 # Dataclass schemas for skills & tool manifests
|-- tool_governance.py              # Tool validation, versioning, dependency management
|
|-- prompts/
|   |-- Stella_prompt_bioml.yaml    # Default system prompt (BioML workflow)
|   |-- Stella_prompt_modified.yaml # Alternative prompt with skill workflow
|   |-- Stella_prompt.yaml          # Original base prompt
|   |-- code_agent.yaml             # Code agent prompt template
|   |-- toolcalling_agent.yaml      # Tool-calling agent prompt template
|   +-- structured_code_agent.yaml  # Structured code agent prompt
|
|-- skills/
|   |-- prebuilt/                   # Curated skill definitions (YAML)
|   |   |-- gene_resistance_analysis.yaml
|   |   |-- protein_structure_analysis.yaml
|   |   |-- drug_screening_pipeline.yaml
|   |   |-- literature_review.yaml
|   |   |-- expression_data_analysis.yaml
|   |   +-- crispr_experiment_design.yaml
|   +-- auto_generated/             # Skills created from successful runs
|
|-- new_tools/                      # Biomedical tool library (60+ tools)
|   |-- database_tools.py           # ClinVar, Ensembl, OpenTargets, UniProt, PDB, etc.
|   |-- literature_tools.py         # PubMed, ArXiv, Google Scholar
|   |-- virtual_screening_tools.py  # Molecular descriptors, ADMET, docking
|   |-- enzyme_tools.py             # Enzyme function & pathway analysis
|   |-- llm.py                      # LLM client (OpenRouter multi-model)
|   |-- biosecurity_alignment_guard.py  # Biosafety checks
|   +-- manifests/                  # Tool governance manifests (YAML)
|
|-- start_stella_web.py             # Web launcher with preset config
|-- start_stella_basic.py           # Minimal launcher
|-- stella_ui_english.py            # Gradio web interface
|-- requirements.txt                # Python dependencies
|-- resource/                       # Large biomedical datasets (download separately)
+-- docker_files/                   # Docker deployment configs
```

### Key modules

| File | Role |
|------|------|
| `stella_core.py` | Creates the Manager/Dev/Critic/ToolCreation agents, wires tools, launches Gradio UI |
| `predefined_tools.py` | ~20 general-purpose tools: web search, shell commands, file operations, PDF extraction |
| `new_tools/database_tools.py` | 30+ biomedical database query tools (ClinVar, Ensembl, UniProt, PDB, STRING, ChEMBL, TCGA, GTEx, etc.) |
| `skill_manager.py` | Wraps SkillStore + SkillRetriever + SkillSummarizer + ToolIndex into one interface |
| `skill_retriever.py` | 3-stage retrieval: tag/pattern match -> TF-IDF similarity -> quality-weighted re-ranking |
| `tool_governance.py` | ToolIndex: manifest loading, validation, versioning, dependency conflict detection |

---

## Tool Library (60+ Tools)

### Literature & Search
- **PubMed**: Advanced medical literature search and analysis
- **ArXiv**: Academic paper discovery and content extraction
- **Google Scholar**: Citation analysis and research trends
- **Web Search**: Enhanced scientific web search (Google, SerpAPI)

### Biomedical Databases
- **UniProt**: Protein sequence and annotation data
- **PDB**: Protein structure analysis
- **KEGG**: Pathway analysis and metabolic networks
- **Ensembl**: Genomic data access and gene annotation
- **STRING**: Protein-protein interaction networks
- **ChEMBL**: Chemical bioactivity database
- **TCGA**: Cancer genomics data analysis
- **GTEx**: Gene expression tissue atlas
- **ClinVar**: Clinical variant interpretation
- **OpenTargets**: Drug target evidence

### Virtual Screening & Drug Discovery
- **Molecular Descriptors**: Chemical property calculation (RDKit)
- **ADMET Prediction**: Drug-like property assessment
- **Molecular Docking**: Protein-ligand interaction modeling
- **Ligand-based Screening**: Similarity search and pharmacophore analysis

### Specialized Analysis
- **Enzyme Function**: Biochemical pathway analysis
- **Protein Embeddings**: ESM-2 protein language model features
- **Gene Expression**: RNA-seq and microarray analysis
- **Biosafety**: Research ethics and safety guidelines

---

## Example Queries

```
Analyze the latest CRISPR-Cas9 developments in cancer therapy from recent PubMed papers
```

```
Retrieve the crystal structure of human insulin from PDB, analyze binding sites, and predict drug interactions
```

```
Screen potential inhibitors for EGFR kinase using virtual docking and predict ADMET properties
```

```
Which genes are associated with intervertebral disc disease according to DisGeNet but not OMIM?
```

```
Analyze gene coexpression networks in cancer datasets and identify potential biomarkers
```

---

## Benchmark Results

Evaluated on [LAB-Bench](https://github.com/Future-House/LAB-Bench) DbQA (database question answering):

| Subtask | Accuracy |
|---------|----------|
| Gene-disease association (DisGeNet) | 1/1 |
| Gene location (Ensembl) | 2/2 |
| miRNA targets | 1/1 |
| Mouse tumor gene sets (MGI) | 3/3 |
| Oncogenic signatures (MSigDB) | 2/2 |
| ClinVar variant lookup | 1/1 |
| Vaccine response | 2/2 |
| Multi-sequence variant | 1/3 |
| **Overall** | **13/15 = 87%** |

---

## Docker Deployment

See `docker_files/README-Docker.md` for Docker-based deployment instructions.

```bash
cd docker_files
docker-compose up
```

---

## License

MIT License -- see LICENSE file for details.
