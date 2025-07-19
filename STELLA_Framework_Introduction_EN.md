# 🌟 STELLA: Self-Evolving Intelligent Laboratory Assistant Framework

**STELLA** (Self-Evolving Intelligent Laboratory Assistant) is a multi-agent collaborative AI research assistant framework designed specifically for biomedical research and scientific computing. STELLA features advanced capabilities including intelligent tool selection, dynamic tool creation, memory management, and multi-agent collaboration.

## 📋 Table of Contents
- [Core Features](#core-features)
- [Agent Architecture](#agent-architecture)
- [Memory System](#memory-system)
- [Tool Management System](#tool-management-system)
- [Professional Tool Library](#professional-tool-library)
- [Technical Architecture](#technical-architecture)
- [Use Cases](#use-cases)
- [Quick Start](#quick-start)

## 🚀 Core Features

### 🤖 LLM-Driven Intelligent Tool Selection
- **Intelligent Analysis**: Uses Gemini-2.5-Pro for deep understanding of user query intent
- **Precise Matching**: Intelligently selects the most relevant tool combinations from 61+ professional tools
- **Dynamic Loading**: Real-time loading of required professional tools for agents
- **Multi-language Support**: Perfect support for mixed Chinese-English queries

### 🧠 Self-Evolution Capabilities
- **Quality Assessment**: Automatic evaluation of task completion quality
- **Intelligent Decision**: Smart determination of whether to create new tools based on needs
- **Dynamic Creation**: Automatic generation of specialized tools to improve system performance
- **Continuous Learning**: Learning and optimization from successful experiences

### 💾 Advanced Memory System
- **Mem0 Enhancement**: Support for semantic memory search and intelligent management
- **Knowledge Templates**: Save and retrieve successful problem-solving patterns
- **Collaborative Memory**: Shared memory space for multi-agent collaboration
- **Personalization**: User-specific memory and preference learning

## 🏗️ Agent Architecture

### 1. 🧠 Manager Agent (Main Coordinator)
**Model**: Google Gemini-2.5-Pro  
**Role**: System core with full permissions and strategic decision-making capabilities

**Core Functions**:
- 🎯 **Intelligent Tool Preparation**: Uses `analyze_query_and_load_relevant_tools()` to automatically analyze queries and load the most relevant professional tools
- 📋 **Task Coordination**: Manages and delegates tasks to specialized agents
- 🔧 **Tool Management**: Complete tool creation, loading, and distribution permissions
- 📚 **Memory Management**: Full access to knowledge base and memory systems
- 🤝 **Collaboration Management**: Creates shared workspaces and manages team collaboration

**Permission Level**: ⭐⭐⭐⭐⭐ (Highest Level)

### 2. 🛠️ Dev Agent (Development Execution Specialist)
**Model**: Anthropic Claude Sonnet-4  
**Role**: Professional code execution and environment management expert

**Core Functions**:
- 💻 **Code Execution**: High-quality Python code writing and execution
- 🔬 **Scientific Computing**: Bioinformatics analysis, data processing, visualization
- 🐍 **Environment Management**: Conda environment creation, package management, GPU monitoring
- 📊 **Data Analysis**: CSV analysis, statistical computing, chart generation

**Permission Level**: ⭐⭐⭐ (Basic Execution Permissions)

### 3. 🎯 Critic Agent (Quality Assessment Specialist)
**Model**: Google Gemini-2.5-Pro  
**Role**: Objective task quality assessment and improvement recommendation expert

**Core Functions**:
- 📈 **Quality Assessment**: Objective analysis of task completion quality and accuracy
- 🔍 **Gap Identification**: Discovering improvement opportunities and potential issues
- 🛠️ **Tool Recommendations**: Intelligent suggestions on whether to create specialized tools
- 📝 **Improvement Plans**: Providing specific optimization recommendations

**Permission Level**: ⭐⭐ (Assessment Permissions)

### 4. 🔧 Tool Creation Agent (Tool Development Specialist)
**Model**: Anthropic Claude Sonnet-4  
**Role**: Professional tool development and code architecture expert

**Core Functions**:
- 🛠️ **Tool Development**: Creating production-grade Python tools
- 🔍 **Best Practices**: Researching and applying latest programming best practices
- 🧪 **Quality Assurance**: Complete testing and validation processes
- 📚 **Documentation**: Generating detailed tool documentation and usage instructions

**Permission Level**: ⭐⭐⭐⭐ (Tool Creation Permissions)

## 💾 Memory System

### 🧠 Mem0 Enhanced Memory (Optional)
- **Semantic Search**: Intelligent retrieval based on content similarity
- **Auto Optimization**: Intelligent deduplication and memory integration
- **Personalized Learning**: Learning user preferences and working patterns
- **Context Maintenance**: Maintaining conversation continuity

### 📚 Knowledge Template System
```python
# Save successful solutions
save_successful_template(
    task_description="Protein structure analysis",
    reasoning_process="Using UniProt->AlphaFold->PDB workflow",
    solution_outcome="Successfully obtained high-quality structural data",
    domain="structural_biology"
)

# Retrieve similar experiences
retrieve_similar_templates(
    task_description="Analyze 3D structure of new protein",
    top_k=3
)
```

### 🤝 Multi-Agent Collaborative Memory
- **Shared Workspaces**: Memory spaces for team collaboration
- **Task Tracking**: Progress management for multi-step tasks
- **Knowledge Transfer**: Experience sharing between agents
- **Contribution Statistics**: Team member collaboration analysis

## 🔧 Tool Management System

### 🎯 LLM Intelligent Tool Selection
STELLA's core innovation is the LLM-based intelligent tool selection system:

```python
# Automatically analyze queries and load relevant tools
result = analyze_query_and_load_relevant_tools(
    user_query="Search PubMed for latest CRISPR research",
    max_tools=10
)
```

**Workflow**:
1. **Semantic Analysis**: LLM deeply understands user query intent
2. **Intelligent Matching**: Selects most relevant tools from 61+ available tools
3. **Relevance Scoring**: Provides 0.0-1.0 relevance scores with selection reasoning
4. **Dynamic Loading**: Real-time loading of tools into manager_agent and tool_creation_agent
5. **Immediate Availability**: Tools immediately available for task execution

### 🛠️ Dynamic Tool Creation
```python
# Intelligently assess if new tools are needed
evaluation = evaluate_with_critic(
    task_description="Analyze single-cell RNA-seq data",
    current_result="Basic analysis completed",
    expected_outcome="Advanced clustering and trajectory analysis"
)

# If needed, automatically create specialized tools
if evaluation.should_create_tool:
    create_new_tool(
        tool_name="advanced_scrna_analyzer",
        tool_purpose="Advanced single-cell RNA-seq analysis",
        tool_category="data_analysis"
    )
```

### 📊 Tool Permission Management
| Agent | Tool Discovery | Tool Loading | Tool Creation | System Management |
|-------|----------------|--------------|---------------|-------------------|
| Manager Agent | ✅ | ✅ | ✅ | ✅ |
| Tool Creation Agent | ✅ | ✅ | ✅ | ❌ |
| Dev Agent | ✅ | ✅ | ❌ | ❌ |
| Critic Agent | ❌ | ❌ | ❌ | ❌ |

## 🔬 Professional Tool Library

### 📚 Literature Search Tools (Literature Tools)
- **`query_pubmed`**: PubMed database search for biomedical literature retrieval
- **`query_arxiv`**: arXiv preprint paper search
- **`query_scholar`**: Google Scholar academic search
- **`search_google`**: General Google search
- **`extract_pdf_content`**: PDF literature content extraction
- **`fetch_supplementary_info_from_doi`**: DOI supplementary material retrieval

### 🧬 Biological Database Tools (Database Tools)
- **`query_uniprot`**: UniProt protein database queries
- **`query_alphafold`**: AlphaFold protein structure prediction data
- **`query_pdb`**: PDB protein structure database
- **`query_kegg`**: KEGG metabolic pathway database
- **`query_stringdb`**: STRING protein interaction networks
- **`query_ensembl`**: Ensembl genome database
- **`query_clinvar`**: ClinVar clinical variant database
- **`query_gwas_catalog`**: GWAS catalog genome-wide association studies
- **`query_gnomad`**: gnomAD population genetic variant frequencies
- **`query_reactome`**: Reactome pathway database
- **`blast_sequence`**: BLAST sequence alignment
- **`query_geo`**: GEO gene expression database
- **`query_opentarget`**: Open Targets drug target data
- **30+ additional specialized database tools**

### 💊 Virtual Screening Tools (Virtual Screening Tools)
- **Molecular Descriptor Calculation**: Chemical molecular property calculation
- **ADMET Prediction**: Absorption, distribution, metabolism, excretion, toxicity prediction
- **Drug Similarity Analysis**: Compound structural similarity assessment
- **Molecular Docking**: Protein-ligand interaction prediction

### 🛠️ Basic Development Tools
- **`run_shell_command`**: System command execution
- **`create_conda_environment`**: Python environment management
- **`install_packages_conda/pip`**: Package management
- **`check_gpu_status`**: GPU status monitoring
- **`create_script`**: Script file creation
- **`run_script`**: Script execution
- **`monitor_training_logs`**: Training log monitoring

## ⚙️ Technical Architecture

### 🤖 Model Configuration
- **Main Coordinator**: Google Gemini-2.5-Pro (Temperature 0.1, consistency optimized)
- **Code Execution**: Anthropic Claude Sonnet-4 (Default temperature)
- **API Integration**: OpenRouter unified API interface
- **Memory Enhancement**: Mem0 AI memory management platform

### 🔗 System Integration
- **smolagents**: Agent framework
- **MCP Protocol**: Integration with external services like PubMed
- **Vector Database**: Chroma local vector storage
- **Environment Management**: Conda/pip package management
- **Monitoring System**: Phoenix observability (optional)

### 🌐 Deployment Options
```bash
# Basic mode
python stella_core.py

# Enable knowledge templates
python stella_core.py --use_template

# Enable Mem0 enhanced memory
python stella_core.py --use_template --use_mem0

# Use Mem0 managed platform
python stella_core.py --use_template --use_mem0 --mem0_platform --mem0_api_key YOUR_KEY
```

## 🎯 Use Cases

### 🧬 Biomedical Research
- **Literature Review**: Intelligent retrieval and analysis of research papers
- **Database Queries**: Multi-database integrated queries and analysis
- **Sequence Analysis**: DNA/RNA/protein sequence analysis
- **Structural Biology**: Protein structure prediction and analysis
- **Genomics**: Variant analysis, GWAS studies
- **Drug Discovery**: Virtual screening, ADMET prediction

### 💻 Scientific Computing
- **Data Analysis**: Statistical analysis, machine learning
- **Visualization**: Scientific charts and data visualization
- **Environment Management**: Computing environment configuration and management
- **Workflows**: Complex analysis pipeline automation

### 🤝 Team Collaboration
- **Knowledge Management**: Team knowledge bases and experience sharing
- **Project Tracking**: Multi-step research project management
- **Quality Control**: Research quality assessment and improvement
- **Skill Development**: Automatic tool creation and capability expansion

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the project
git clone <repository-url>
cd agents/STELLA

# Install dependencies
pip install -r requirements.txt

# Configure API key
export OPENROUTER_API_KEY="your-api-key"
```

### 2. Basic Usage
```python
# Import STELLA
from stella_core import manager_agent, analyze_query_and_load_relevant_tools

# Intelligent tool selection
analyze_query_and_load_relevant_tools(
    "Search for latest research papers on CRISPR gene editing",
    max_tools=5
)

# Execute task
result = manager_agent.run("Please analyze the function of this protein sequence")
```

### 3. Launch Web Interface
```bash
# Start Gradio interface
python stella_core.py --port 7860

# Access http://localhost:7860
```

## 📈 Performance Advantages

### 🎯 Intelligence Enhancement
- **Precise Tool Selection**: LLM-driven semantic understanding with 95%+ selection accuracy
- **Adaptive Learning**: Learning from experience with continuous performance improvement
- **Multi-language Support**: Seamless processing of Chinese-English queries

### ⚡ Efficiency Improvement
- **Automated Workflows**: Full automation from query analysis to tool execution
- **Parallel Processing**: Agent collaboration for improved processing efficiency
- **Cache Optimization**: Intelligent memory management to avoid redundant computations

### 🔧 Flexibility
- **Modular Design**: Components can be used and extended independently
- **Dynamic Expansion**: Real-time addition of new tools and functionality
- **Multi-scenario Adaptation**: Suitable for various biomedical research scenarios

---

**STELLA Framework** - Making AI your true intelligent research partner!

*Version: 2024.12 | Development Team: AI Research Lab* 