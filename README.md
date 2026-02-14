<div align="center">

<img src="asset/stella_logo.png" width="200"/>

# STELLA

### Self-Evolving Multimodal Agents for Biomedical Research

<p>
    <em>A self-evolving multi-agent framework that continuously learns and adapts — integrating dynamic knowledge bases, reasoning templates, and self-correction to accelerate biomedical discovery.</em>
</p>

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/zaixizhang/STELLA)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/bioRxiv-2025.07.01.662467v2-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.07.01.662467v2)
[![arXiv](https://img.shields.io/badge/arXiv-2507.02004-b31b1b.svg)](https://arxiv.org/abs/2507.02004)
[![Web](https://img.shields.io/badge/Web-stella--agent.com-orange.svg)](https://stella-agent.com/)

[**Paper**](https://www.biorxiv.org/content/10.1101/2025.07.01.662467v2) | [**Online Demo**](https://stella-agent.com/) | [**arXiv**](https://arxiv.org/abs/2507.02004) | [**Twitter**](https://x.com/BiologyAIDaily/status/1941122955850993966)

**Authors:**
Ruofan Jin (1\*), Mingyang Xu (1\*), Fei Meng (1\*), Guancheng Wan (2), Qingran Cai (1), Yize Jiang (1), Jin Han (1), Yuanyuan Chen (1), Wanqing Lu (1), Mengyang Wang (1), Zhiqian Lan (1), Yuxuan Jiang (1), Junhong Liu (1), Dongyao Wang (1), Le Cong (3), Zaixi Zhang (1,†)

<sup>1: Princeton University, 2: UIUC, 3: Stanford School of Medicine</sup>
<sup>* Equal Contribution, † Corresponding Author</sup>

</div>

---

## Contents

- [Overview](#overview)
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

*Overview of the STELLA framework. The framework consists of four main components: a reasoning template, a manager agent, a dev agent, and a critic agent. The tool ocean provides a set of predefined and self-evolving tools for the agents to use.*

| Component | Description |
|-----------|-------------|
| **Manager Agent** | Decomposes scientific objectives, orchestrates sub-agents, plans multi-step workflows |
| **Dev Agent** | Executes bioinformatics analyses, runs code, queries databases and literature |
| **Critic Agent** | Evaluates result quality, identifies gaps, recommends improvements |
| **Tool Ocean** | Predefined and self-evolving tools for literature search, databases, virtual screening, and more |

**Self-Evolution Loop:** STELLA continuously improves by learning from new data, refining its reasoning templates, and autonomously creating new tools when existing ones are insufficient.

## Quick Start

**Option 1: Use the online version (recommended)**

Visit [stella-agent.com](https://stella-agent.com/) to use STELLA directly in your browser.

**Option 2: Run locally**

```bash
# Simple mode (basic functionality)
python stella_core.py

# Memory-enhanced mode (recommended)
python stella_core.py --use_template --use_mem0

# Web interface (user-friendly)
python start_stella_web.py
```

### Usage Options

| Mode | Command | Description |
|------|---------|-------------|
| **Basic** | `python stella_core.py` | Core STELLA functionality with standard agents |
| **Memory Enhanced** | `python stella_core.py --use_template --use_mem0` | Adds template learning and enhanced memory via Mem0 |
| **Web Interface** | `python start_stella_web.py` | User-friendly Gradio interface |

## Installation

### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/zaixizhang/STELLA.git
cd STELLA

# Create conda environment with Python 3.12
conda create -n stella python=3.12 -y
conda activate stella

# Install scientific packages via conda
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn -y

# Install remaining dependencies via pip
pip install -r requirements.txt
```

### Option 2: Using pip only
```bash
# Python 3.8+ required
python --version

# Core dependencies
pip install gradio>=4.0.0
pip install 'smolagents[mcp]'
pip install numpy pandas scikit-learn
pip install requests beautifulsoup4 markdownify
```

### Optional Dependencies
```bash
# For Mem0 enhanced memory (recommended)
pip install mem0ai

# For biomedical analysis
pip install biopython rdkit-pypi
pip install pymed arxiv scholarly

# For MCP tools integration
pip install uvx
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
├── README.md                       # This documentation
├── stella_core.py                  # Core multi-agent system
├── start_stella_web.py             # Web interface launcher
├── stella_ui_english.py            # Web UI implementation
├── memory_manager.py               # Memory management system
├── Knowledge_base.py               # Knowledge base system
├── predefined_tools.py             # Core system tools
│
├── new_tools/                      # Professional tool library
│   ├── literature_tools.py         # Literature search tools
│   ├── database_tools.py           # Biomedical databases
│   ├── virtual_screening_tools.py  # Drug discovery tools
│   ├── enzyme_tools.py             # Biochemical analysis
│   ├── llm.py                      # LLM integration utilities
│   └── biosecurity_tool.py         # Safety mechanisms
│
├── asset/                          # Images and figures
│   ├── stella_logo.png             # STELLA logo
│   ├── stella_illustration.png     # Framework diagram
│   └── Stella_result.png           # Performance results
│
└── prompts/                        # Prompt templates
```

## API Keys

| Key | Required | Purpose |
|-----|:--------:|---------|
| `OPENROUTER_API_KEY` | **Yes** | Powers all LLM agents via OpenRouter |
| `SERPAPI_API_KEY` | No | Enhanced web search results |
| `PAPERQA_API_KEY` | No | Academic literature analysis |

```bash
# Create .env file with your API keys
touch .env
echo "OPENROUTER_API_KEY=your_openrouter_api_key_here" >> .env
echo "SERPAPI_API_KEY=your_serpapi_key_here" >> .env
echo "PAPERQA_API_KEY=your_paperqa_key_here" >> .env
```

**Get API Keys:**
- OpenRouter: https://openrouter.ai/
- SERPAPI: https://serpapi.com/
- PaperQA: https://paperqa.ai/

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
