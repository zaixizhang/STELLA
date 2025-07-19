# 🌟 Stella AI Assistant

**Self-Evolving LLM Agent for Scientific Discovery**

---

## 📖 Table of Contents
- [English Documentation](#english-documentation)
- [中文文档](#中文文档)

---

## English Documentation

### 🚀 Overview

Stella is an advanced AI assistant designed for scientific research and discovery. It features a beautiful web interface with real-time execution monitoring, comprehensive tool integration, and specialized capabilities for scientific literature analysis, data processing, and research automation.

### ✨ Key Features

- **🎨 Professional Web Interface**: Beautiful, responsive UI with real-time updates
- **🔍 Real-time Execution Monitoring**: Step-by-step execution tracking with detailed logs
- **📁 File Management**: Automatic file creation tracking and download capabilities
- **🔬 Scientific Tools Integration**: 
  - PubMed literature search
  - ArXiv paper access
  - GitHub repository analysis
  - Shell command execution
  - Python script generation and execution
- **🌐 Multi-language Support**: English and Chinese interfaces
- **📊 Performance Metrics**: Token usage tracking and execution time monitoring
- **🛡️ Safety Features**: Secure execution environment with output sanitization

### 🛠️ Installation

#### Prerequisites
```bash
# Python 3.8+ required
python3 --version

# Install required dependencies
pip install gradio
pip install 'smolagents[mcp]'  # For MCP tools integration
```

#### Setup
```bash
# Navigate to the agents directory
cd agents/

# Ensure all required files are present:
# - stella_ui_english.py (English interface)
# - agent_gradio_evo.py (Core agent functionality)
# - small_logo_b64.txt (Stella logo)
```

### 🚀 Usage

#### Starting Stella Web Interface

```bash
python3 start_stella_web.py
```

#### Access URLs
Once started, Stella will be available at:
- **Local**: http://localhost:7860
- **Network**: http://[your-ip]:7860
- **Public**: Gradio will generate a public sharing link

#### Basic Usage

1. **Enter your request** in the chat interface
2. **Monitor execution** in the "Execution Steps" tab
3. **Download files** from the "Created Files" tab
4. **Check system status** in the "System Status" tab

### 📝 Example Requests

#### Scientific Research
```
Analyze recent CRISPR-Cas9 developments in Nature papers from 2024
```

#### Data Analysis
```
Create a machine learning pipeline for protein classification using ESM embeddings
```

#### Literature Review
```
Compare quantum computing approaches for drug discovery optimization
```

#### Code Generation
```
Generate a Python script to analyze single-cell RNA sequencing data
```

### 📂 File Structure

```
agents/
├── README.md                 # This documentation
├── stella_ui_english.py      # English web interface
├── agent_gradio_evo.py       # Core agent functionality
├── small_logo_b64.txt        # Stella logo (base64)
├── start_stella_web.py       # Web launcher script
└── agent_outputs/            # Generated files directory
```

### ⚙️ Configuration

#### Environment Variables
```bash
# Enable debug mode for detailed logging
export STELLA_DEBUG=true

# Customize output directory
export STELLA_OUTPUT_DIR=/path/to/outputs
```

#### MCP Tools Setup
```bash
# For enhanced scientific capabilities
pip install 'smolagents[mcp]'
```

### 🔧 Advanced Features

#### Real-time Monitoring
- Live step-by-step execution tracking
- Tool call visualization with arguments
- Performance metrics and token usage
- Error handling and debugging information

#### File Management
- Automatic detection of created files
- File type recognition and categorization
- Download links for all generated content
- File size and modification tracking



**Happy Research with Stella! 🌟** 