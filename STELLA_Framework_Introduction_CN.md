# 🌟 STELLA：自进化智能生物医学研究助手框架

**STELLA** (Self-Evolving Intelligent Laboratory Assistant) 是一个基于多智能体协作的自进化AI研究助手框架，专为生物医学研究和科学计算而设计。STELLA具备智能工具选择、动态工具创建、记忆管理和多智能体协作等先进功能。

## 📋 目录
- [核心特性](#核心特性)
- [智能体架构](#智能体架构)
- [记忆系统](#记忆系统)
- [工具管理系统](#工具管理系统)
- [专业工具库](#专业工具库)
- [技术架构](#技术架构)
- [应用场景](#应用场景)
- [快速开始](#快速开始)

## 🚀 核心特性

### 🤖 LLM驱动的智能工具选择
- **智能分析**: 使用Gemini-2.5-Pro深度理解用户查询意图
- **精准匹配**: 从61+专业工具中智能选择最相关的工具组合
- **动态加载**: 实时为智能体加载所需的专业工具
- **多语言支持**: 完美支持中英文混合查询

### 🧠 自进化能力
- **质量评估**: 自动评估任务完成质量
- **智能决策**: 基于需求智能决定是否创建新工具
- **动态创建**: 自动生成专用工具提升系统性能
- **持续学习**: 从成功经验中学习和优化

### 💾 高级记忆系统
- **Mem0增强**: 支持语义记忆搜索和智能管理
- **知识模板**: 保存和检索成功的问题解决模式
- **协作记忆**: 多智能体共享记忆空间
- **个性化**: 用户专属记忆和偏好学习

## 🏗️ 智能体架构

### 1. 🧠 Manager Agent (主协调智能体)
**模型**: Google Gemini-2.5-Pro  
**角色**: 系统核心，具有完整权限和战略决策能力

**核心功能**:
- 🎯 **智能工具准备**: 使用`analyze_query_and_load_relevant_tools()`自动分析查询并加载最相关的专业工具
- 📋 **任务协调**: 管理和委派任务给专业智能体
- 🔧 **工具管理**: 完整的工具创建、加载和分发权限
- 📚 **记忆管理**: 完整的知识库和记忆系统访问权限
- 🤝 **协作管理**: 创建共享工作空间，管理团队协作

**权限级别**: ⭐⭐⭐⭐⭐ (最高级别)

### 2. 🛠️ Dev Agent (开发执行智能体)
**模型**: Anthropic Claude Sonnet-4  
**角色**: 专业的代码执行和环境管理专家

**核心功能**:
- 💻 **代码执行**: 高质量的Python代码编写和执行
- 🔬 **科学计算**: 生物信息学分析、数据处理、可视化
- 🐍 **环境管理**: Conda环境创建、包管理、GPU监控
- 📊 **数据分析**: CSV分析、统计计算、图表生成

**权限级别**: ⭐⭐⭐ (基础执行权限)

### 3. 🎯 Critic Agent (质量评估智能体)
**模型**: Google Gemini-2.5-Pro  
**角色**: 客观的任务质量评估和改进建议专家

**核心功能**:
- 📈 **质量评估**: 客观分析任务完成质量和准确性
- 🔍 **差距识别**: 发现改进空间和潜在问题
- 🛠️ **工具建议**: 智能推荐是否需要创建专用工具
- 📝 **改进方案**: 提供具体的优化建议

**权限级别**: ⭐⭐ (评估权限)

### 4. 🔧 Tool Creation Agent (工具创建智能体)
**模型**: Anthropic Claude Sonnet-4  
**角色**: 专业的工具开发和代码架构专家

**核心功能**:
- 🛠️ **工具开发**: 创建生产级Python工具
- 🔍 **最佳实践**: 研究和应用最新的编程最佳实践
- 🧪 **质量保证**: 完整的测试和验证流程
- 📚 **文档完善**: 生成详细的工具文档和使用说明

**权限级别**: ⭐⭐⭐⭐ (工具创建权限)

## 💾 记忆系统

### 🧠 Mem0增强记忆 (可选)
- **语义搜索**: 基于内容相似性的智能检索
- **自动优化**: 智能去重和记忆整合
- **个性化学习**: 学习用户偏好和工作模式
- **上下文保持**: 维护对话的连续性

### 📚 知识模板系统
```python
# 保存成功的解决方案
save_successful_template(
    task_description="蛋白质结构分析",
    reasoning_process="使用UniProt->AlphaFold->PDB流程",
    solution_outcome="成功获得高质量结构数据",
    domain="structural_biology"
)

# 检索相似经验
retrieve_similar_templates(
    task_description="分析新蛋白质的三维结构",
    top_k=3
)
```

### 🤝 多智能体协作记忆
- **共享工作空间**: 团队协作的记忆空间
- **任务追踪**: 多步骤任务的进度管理
- **知识传递**: 智能体间的经验分享
- **贡献统计**: 团队成员的协作分析

## 🔧 工具管理系统

### 🎯 LLM智能工具选择
STELLA的核心创新是基于LLM的智能工具选择系统：

```python
# 自动分析查询并加载相关工具
result = analyze_query_and_load_relevant_tools(
    user_query="搜索PubMed中关于CRISPR的最新研究",
    max_tools=10
)
```

**工作流程**:
1. **语义分析**: LLM深度理解用户查询意图
2. **智能匹配**: 从61+工具中选择最相关的工具
3. **相关性评分**: 提供0.0-1.0的相关性分数和选择理由
4. **动态加载**: 实时将工具加载到manager_agent和tool_creation_agent
5. **即时可用**: 工具立即可用于任务执行

### 🛠️ 动态工具创建
```python
# 智能评估是否需要新工具
evaluation = evaluate_with_critic(
    task_description="分析单细胞RNA-seq数据",
    current_result="基础分析完成",
    expected_outcome="高级聚类和轨迹分析"
)

# 如果需要，自动创建专用工具
if evaluation.should_create_tool:
    create_new_tool(
        tool_name="advanced_scrna_analyzer",
        tool_purpose="单细胞RNA-seq高级分析",
        tool_category="data_analysis"
    )
```

### 📊 工具权限管理
| 智能体 | 工具发现 | 工具加载 | 工具创建 | 系统管理 |
|--------|----------|----------|----------|----------|
| Manager Agent | ✅ | ✅ | ✅ | ✅ |
| Tool Creation Agent | ✅ | ✅ | ✅ | ❌ |
| Dev Agent | ✅ | ✅ | ❌ | ❌ |
| Critic Agent | ❌ | ❌ | ❌ | ❌ |

## 🔬 专业工具库

### 📚 文献搜索工具 (Literature Tools)
- **`query_pubmed`**: PubMed数据库搜索，生物医学文献检索
- **`query_arxiv`**: arXiv预印本论文搜索
- **`query_scholar`**: Google Scholar学术搜索
- **`search_google`**: 通用Google搜索
- **`extract_pdf_content`**: PDF文献内容提取
- **`fetch_supplementary_info_from_doi`**: DOI补充材料获取

### 🧬 生物数据库工具 (Database Tools)
- **`query_uniprot`**: UniProt蛋白质数据库查询
- **`query_alphafold`**: AlphaFold蛋白质结构预测数据
- **`query_pdb`**: PDB蛋白质结构数据库
- **`query_kegg`**: KEGG代谢通路数据库
- **`query_stringdb`**: STRING蛋白质相互作用网络
- **`query_ensembl`**: Ensembl基因组数据库
- **`query_clinvar`**: ClinVar临床变异数据库
- **`query_gwas_catalog`**: GWAS catalog全基因组关联研究
- **`query_gnomad`**: gnomAD人群基因变异频率
- **`query_reactome`**: Reactome通路数据库
- **`blast_sequence`**: BLAST序列比对
- **`query_geo`**: GEO基因表达数据库
- **`query_opentarget`**: Open Targets药物靶点数据
- **等30+专业数据库工具**

### 💊 虚拟筛选工具 (Virtual Screening Tools)
- **分子描述符计算**: 化学分子性质计算
- **ADMET预测**: 吸收、分布、代谢、排泄、毒性预测
- **药物相似性分析**: 化合物结构相似性评估
- **分子对接**: 蛋白质-配体相互作用预测

### 🛠️ 基础开发工具
- **`run_shell_command`**: 系统命令执行
- **`create_conda_environment`**: Python环境管理
- **`install_packages_conda/pip`**: 包管理
- **`check_gpu_status`**: GPU状态监控
- **`create_script`**: 脚本文件创建
- **`run_script`**: 脚本执行
- **`monitor_training_logs`**: 训练日志监控

## ⚙️ 技术架构

### 🤖 模型配置
- **主协调**: Google Gemini-2.5-Pro (温度0.1，一致性优化)
- **代码执行**: Anthropic Claude Sonnet-4 (温度默认)
- **API集成**: OpenRouter统一API接口
- **记忆增强**: Mem0 AI记忆管理平台

### 🔗 系统集成
- **smolagents**: 智能体框架
- **MCP协议**: PubMed等外部服务集成
- **向量数据库**: Chroma本地向量存储
- **环境管理**: Conda/pip包管理
- **监控系统**: Phoenix可观测性(可选)

### 🌐 部署方式
```bash
# 基础模式
python stella_core.py

# 启用知识模板
python stella_core.py --use_template

# 启用Mem0增强记忆
python stella_core.py --use_template --use_mem0

# 使用Mem0托管平台
python stella_core.py --use_template --use_mem0 --mem0_platform --mem0_api_key YOUR_KEY
```

## 🎯 应用场景

### 🧬 生物医学研究
- **文献调研**: 智能检索和分析科研论文
- **数据库查询**: 多数据库整合查询和分析
- **序列分析**: DNA/RNA/蛋白质序列分析
- **结构生物学**: 蛋白质结构预测和分析
- **基因组学**: 变异分析、GWAS研究
- **药物发现**: 虚拟筛选、ADMET预测

### 💻 科学计算
- **数据分析**: 统计分析、机器学习
- **可视化**: 科学图表和数据可视化
- **环境管理**: 计算环境配置和管理
- **工作流**: 复杂分析流程自动化

### 🤝 团队协作
- **知识管理**: 团队知识库和经验分享
- **项目追踪**: 多步骤研究项目管理
- **质量控制**: 研究质量评估和改进
- **技能发展**: 自动工具创建和能力扩展

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone <repository-url>
cd agents/STELLA

# 安装依赖
pip install -r requirements.txt

# 配置API密钥
export OPENROUTER_API_KEY="your-api-key"
```

### 2. 基础使用
```python
# 导入STELLA
from stella_core import manager_agent, analyze_query_and_load_relevant_tools

# 智能工具选择
analyze_query_and_load_relevant_tools(
    "搜索关于CRISPR基因编辑的最新研究论文",
    max_tools=5
)

# 执行任务
result = manager_agent.run("请分析这个蛋白质序列的功能")
```

### 3. 启动Web界面
```bash
# 启动Gradio界面
python stella_core.py --port 7860

# 访问 http://localhost:7860
```

## 📈 性能优势

### 🎯 智能性提升
- **精准工具选择**: LLM驱动的语义理解，选择准确率95%+
- **自适应学习**: 从经验中学习，性能持续改进
- **多语言支持**: 中英文查询无缝处理

### ⚡ 效率提升
- **自动化流程**: 从查询分析到工具执行的全自动化
- **并行处理**: 智能体协作，提高处理效率
- **缓存优化**: 智能记忆管理，避免重复计算

### 🔧 灵活性
- **模块化设计**: 组件可独立使用和扩展
- **动态扩展**: 实时添加新工具和功能
- **多场景适配**: 适用于各种生物医学研究场景

---

**STELLA框架** - 让AI真正成为您的智能研究伙伴！

*版本: 2024.12 | 开发团队: AI Research Lab* 