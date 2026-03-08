<div align="center">

<img src="asset/stella_logo.png" width="200"/>

# STELLA

### Self-Evolving Multimodal Agents for Biomedical Research

<p>
    <em>A self-evolving multi-agent framework that continuously learns and adapts — integrating dynamic knowledge bases, reasoning templates, and self-correction to accelerate biomedical discovery.</em>
</p>

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/zaixizhang/STELLA)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/bioRxiv-2025.07.01.662467v2-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.07.01.662467v2)
[![arXiv](https://img.shields.io/badge/arXiv-2507.02004-b31b1b.svg)](https://arxiv.org/abs/2507.02004)
[![Web](https://img.shields.io/badge/Web-stella--agent.com-orange.svg)](https://stella-agent.com/)

[**Paper**](https://www.biorxiv.org/content/10.1101/2025.07.01.662467v2) | [**Online Demo**](https://stella-agent.com/) | [**arXiv**](https://arxiv.org/abs/2507.02004) | [**Twitter**](https://x.com/BiologyAIDaily/status/1941122955850993966)

### :rocket: [Try STELLA Online — No Installation, No API Keys, One Click to Start!](https://stella-agent.com/)
**[:globe_with_meridians: stella-agent.com](https://stella-agent.com/)**

**Authors:**
Ruofan Jin<sup>1</sup>, Mingyang Xu<sup>2</sup>, Fei Meng<sup>3,4</sup>, Guancheng Wan<sup>5</sup>, Qingran Cai<sup>6</sup>, Yize Jiang<sup>7</sup>, Jin Han<sup>8</sup>, Yuanyuan Chen<sup>9</sup>, Wanqing Lu<sup>9</sup>, Mengyang Wang<sup>10</sup>, Zhiqian Lan<sup>11</sup>, Yuxuan Jiang<sup>11</sup>, Junhong Liu<sup>7,✉</sup>, Dongyao Wang<sup>3,4,✉</sup>, Le Cong<sup>12,✉</sup>, and Zaixi Zhang<sup>1,✉</sup>

<sup>1</sup>Princeton University, <sup>2</sup>University of Michigan, <sup>3</sup>The First Affiliated Hospital, USTC, <sup>4</sup>National Key Laboratory of Immune Response and Immunotherapy
<sup>5</sup>UCLA, <sup>6</sup>Shanghai Jiao Tong University, <sup>7</sup>Microcyto, <sup>8</sup>Nanjing University, <sup>9</sup>Tianjin University of Science and Technology
<sup>10</sup>Peking University, <sup>11</sup>The University of Hong Kong, <sup>12</sup>Stanford University
<sup>✉</sup> liujunhong@microcyto.cn, dywsn@ustc.edu.cn, congle@stanford.edu, zz8680@princeton.edu

</div>

---

## Contents

- [Overview](#overview)
- [What's New in v2](#whats-new-in-v2)
- [Demo Video](#demo-video)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Resource Download](#resource-download--setup)
- [Project Structure](#project-structure)
- [API Keys](#api-keys)
- [Updates & Collaboration](#updates--collaboration)
- [Citation](#citation)
- [Related Projects](#related-projects)
- [License](#license)

## Overview

Recent advancements in Large Language Models (LLMs) have demonstrated their potential in specialized fields such as biomedical research. However, their performance is often limited by the lack of domain-specific knowledge and reasoning abilities. **STELLA** addresses this by introducing a self-evolving mechanism that enables the agent to continuously learn and adapt from new data and experiences.

The agent architecture integrates a dynamic knowledge base, a reasoning module, and a self-correction component, allowing it to perform intricate tasks such as literature analysis, experimental design, and data interpretation. STELLA achieves superior performance over existing state-of-the-art models on comprehensive biomedical benchmarks.

**Try STELLA online at [stella-agent.com](https://stella-agent.com/) — no installation required.**

## What's New in v2

This revision introduces a **structured skill management system** that replaces the previous template/memory architecture:

- **Skill System** — YAML-based skill definitions with 3-stage hybrid retrieval (tag matching → TF-IDF similarity → quality-weighted re-ranking). Skills are auto-created from successful runs and deduplicated.
- **Tool Governance** — Every tool now has a YAML manifest recording its interface, dependencies, validation rules, and usage statistics. The `ToolIndex` validates tools and detects dependency conflicts.
- **Expanded Tool Library** — 11 new biomedical tools including molecular docking, ligand-based screening, KEGG pathway search, ESM-2 protein embeddings, and protocol evaluation.
- **6 Prebuilt Skills** — Curated workflows for gene resistance analysis, protein structure analysis, drug screening, literature review, expression data analysis, and CRISPR experiment design.
- **Docker Support** — Dockerfile and docker-compose for containerized deployment.
- **Cleaner Codebase** — Removed legacy memory system, hardcoded API keys, and unused configs. All API keys are now loaded from environment variables.

## Demo Video

[![Watch the demo](video_image.png)](https://drive.google.com/file/d/1a6PoJWZMMix8zyccWVOZxU47NKhABO_W/view?usp=sharing)

*Click the thumbnail above to watch the STELLA demonstration video on Google Drive.*

## Key Results

<img src="asset/Stella_result.png" width="800"/>

*Performance of STELLA on various benchmarks. (A) Comparison of STELLA with other LLMs on Humanity's Last Exam (HLE) Biomedicine, LAB-Bench (DBQA), and LAB-Bench (LitQA). (B) Self-evolving performance of STELLA on the same benchmarks with increasing computation budget.*

Key achievements:
- **4.01/5** score on custom benchmark with **100% task completion**
- Identified **BTN3A1** as a novel regulator in acute myeloid leukemia, verified through CRISPR studies
- Developed enzyme variants showing **>2x improvement** in catalytic activity
- Robotic laboratory automation with success rates improving from **17% to 82%**

## System Architecture

<img src="asset/stella_illustration.png" width="800"/>

*Overview of the STELLA framework. The framework consists of four main components: a manager agent, a dev agent, a critic agent, and a tool ocean. The skill system provides structured workflows learned from successful runs.*

| Component | Description |
|-----------|-------------|
| **Manager Agent** | Decomposes scientific objectives, retrieves relevant skills, orchestrates sub-agents |
| **Dev Agent** | Executes bioinformatics analyses, runs code, queries databases and literature |
| **Critic Agent** | Evaluates result quality, identifies gaps, recommends improvements |
| **Tool Creation Agent** | Dynamically creates new tools when existing ones are insufficient |
| **Tool Ocean** | 60+ predefined and self-evolving tools for literature search, databases, virtual screening |
| **Skill System** | Retrieves, applies, and auto-creates structured workflow skills from successful runs |

Models are configurable via [OpenRouter](https://openrouter.ai/). Edit model variables in `stella_core.py` to switch between GPT, Claude, Gemini, and other providers.

## Quick Start

**Option 1: Use the online version (recommended)**

Visit [stella-agent.com](https://stella-agent.com/) to use STELLA directly in your browser.

**Option 2: Run locally**

```bash
# Clone and install
git clone https://github.com/zaixizhang/STELLA.git
cd STELLA
pip install -r requirements.txt

# Configure API key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Launch web UI
python stella_core.py
```

Open http://localhost:7860 in your browser. A public Gradio link is also generated automatically.

### Usage Options

| Mode | Command | Description |
|------|---------|-------------|
| **Web UI** | `python stella_core.py` | Full Gradio interface with skill retrieval (recommended) |
| **No Skills** | `python stella_core.py --no_template` | Disable skill/template retrieval |
| **Tool Creation** | `python stella_core.py --enable_tool_creation` | Enable dynamic tool creation agent |
| **Custom Port** | `python stella_core.py --port 8080` | Run on a different port |
| **Web Launcher** | `python start_stella_web.py` | Alternative launcher with preset config |
| **Basic** | `python start_stella_basic.py` | Minimal launcher |

### Programmatic Usage

```python
from stella_core import initialize_stella

manager_agent = initialize_stella(
    use_template=True,
    enable_tool_creation=True,
)
result = manager_agent.run("What genes are associated with breast cancer resistance?")
```

## Installation

### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/zaixizhang/STELLA.git
cd STELLA

# Create conda environment
conda create -n stella python=3.12 -y
conda activate stella

# Install scientific packages via conda
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn -y

# Install remaining dependencies via pip
pip install -r requirements.txt
```

### Option 2: Using pip only
```bash
pip install -r requirements.txt
```

### Option 3: Using Docker
```bash
cd docker_files
docker-compose up
```

See `docker_files/README-Docker.md` for details.

### Optional Dependencies
```bash
# For biomedical analysis
pip install biopython rdkit-pypi

# For MCP tools integration
pip install mcp uvx
```

## Resource Download & Setup

**Large Resource Files**: Due to the size of biomedical datasets, additional resources are available for download:

```bash
# Download the resource package (optional for basic usage)
# Manual download from: https://drive.google.com/file/d/1n8I-gkM58wL1OZysdpYhr-Q9ZApd9yU4/view?usp=sharing
# File: resource_backup_20250719_055729.zip (2.0GB compressed)

# Create resource directory and extract
mkdir -p resource/
unzip resource_backup_20250719_055729.zip -d resource/
```

> **Note**: Resources are optional for basic STELLA functionality but enhance performance for complex biomedical tasks.

## Project Structure

```
STELLA/
├── stella_core.py                  # Core multi-agent system & Gradio UI entry point
├── predefined_tools.py             # Core system tools (search, shell, file I/O)
├── skill_manager.py                # Unified skill lifecycle management
├── skill_store.py                  # Skill persistence (YAML files + SQLite run history)
├── skill_retriever.py              # 3-stage hybrid skill retrieval engine
├── skill_summarizer.py             # Auto-create skills from successful runs
├── skill_schema.py                 # Dataclass schemas for skills & tool manifests
├── tool_governance.py              # Tool validation, versioning, dependency management
│
├── new_tools/                      # Biomedical tool library (60+ tools)
│   ├── database_tools.py           # ClinVar, Ensembl, OpenTargets, UniProt, PDB, STRING, etc.
│   ├── literature_tools.py         # PubMed, ArXiv, Google Scholar
│   ├── virtual_screening_tools.py  # Molecular descriptors, ADMET, docking
│   ├── enzyme_tools.py             # Enzyme function & pathway analysis
│   ├── llm.py                      # LLM client (OpenRouter multi-model)
│   ├── esm2_embedder.py            # ESM-2 protein embeddings
│   ├── kegg_pathway_search.py      # KEGG pathway queries
│   ├── molecular_docking_tool.py   # Molecular docking
│   ├── ligand_based_screening_tool.py  # Ligand-based virtual screening
│   ├── protein_seq_featurizer.py   # Protein sequence features
│   ├── uniprot_query.py            # UniProt protein lookup
│   ├── biosecurity_tool.py         # Biosafety mechanisms
│   └── manifests/                  # Tool governance manifests (YAML)
│
├── skills/
│   ├── prebuilt/                   # 6 curated skill definitions (YAML)
│   └── auto_generated/             # Skills created from successful runs
│
├── prompts/                        # Agent prompt templates (YAML)
├── asset/                          # Images and figures
├── start_stella_web.py             # Web launcher with preset config
├── start_stella_basic.py           # Minimal launcher
├── stella_ui_english.py            # Gradio web interface
├── docker_files/                   # Docker deployment configs
├── Tool_Creation_Benchmark/        # Tool creation evaluation benchmark
├── resource/                       # Large biomedical datasets (download separately)
└── requirements.txt                # Python dependencies
```

### Key Modules

| File | Role |
|------|------|
| `stella_core.py` | Creates Manager/Dev/Critic/ToolCreation agents, wires tools, launches Gradio UI |
| `predefined_tools.py` | ~20 general-purpose tools: web search, shell commands, file operations, PDF extraction |
| `new_tools/database_tools.py` | 30+ biomedical database query tools (ClinVar, Ensembl, UniProt, PDB, STRING, ChEMBL, TCGA, GTEx) |
| `skill_manager.py` | Wraps SkillStore + SkillRetriever + SkillSummarizer + ToolIndex into one interface |
| `skill_retriever.py` | 3-stage retrieval: tag/pattern match → TF-IDF similarity → quality-weighted re-ranking |
| `tool_governance.py` | ToolIndex: manifest loading, validation, versioning, dependency conflict detection |

## API Keys

> **Don't have API keys?** No worries — you can directly try our online version at **[stella-agent.com](https://stella-agent.com/)**. No installation, no API keys needed — just one click and start using STELLA right away!

For local deployment, configure the following API keys:

| Key | Required | Purpose |
|-----|:--------:|---------|
| `OPENROUTER_API_KEY` | **Yes** | Powers all LLM agents via OpenRouter |
| `SERPAPI_API_KEY` | No | Enhanced web search results |
| `PAPERQA_API_KEY` | No | Academic literature analysis |

```bash
# Create .env file with your API keys
echo "OPENROUTER_API_KEY=your_openrouter_api_key_here" > .env
echo "SERPAPI_API_KEY=your_serpapi_key_here" >> .env
```

**Get API Keys:**
- OpenRouter: https://openrouter.ai/
- SERPAPI: https://serpapi.com/

## Updates & Collaboration

* **Wet-Lab Verification**: We are currently in the process of conducting wet-lab experiments to verify the key findings and predictions generated by STELLA.
* **Call for Collaboration**: We welcome both wet-lab collaborations (target discovery, antibody/protein/RNA optimization, etc.) and community contributions including: new tools & algorithms, datasets & knowledge bases, software integration, benchmarks & metrics, tutorials & use cases. Please reach out to zz8680@princeton.edu or submit pull requests!

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{jin2025stella,
  title={STELLA: Towards a Biomedical World Model with Self-Evolving Multimodal Agents},
  author={Jin, Ruofan and Xu, Mingyang and Meng, Fei and Wan, Guancheng and Cai, Qingran and Jiang, Yize and Han, Jin and Chen, Yuanyuan and Lu, Wanqing and Wang, Mengyang and Lan, Zhiqian and Jiang, Yuxuan and Liu, Junhong and Wang, Dongyao and Cong, Le and Zhang, Zaixi},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.07.01.662467}
}
```

## Related Projects

- **[LabOS](https://github.com/zaixizhang/LabOS)** — The AI-XR co-scientist that builds upon STELLA for wet-lab integration
- **[stella-agent.com](https://stella-agent.com/)** — Online web version of STELLA

## License

Apache 2.0 — see [LICENSE](LICENSE).
