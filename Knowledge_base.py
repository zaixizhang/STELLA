import os
import re
import requests
import subprocess
import json
import time
import sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Smolagents imports for proper message handling
from smolagents import ChatMessage, MessageRole

# Mem0 integration for enhanced memory management
try:
    from mem0 import Memory, MemoryClient
    MEM0_AVAILABLE = True
    print("✅ Mem0 library available - enhanced memory features enabled")
except ImportError:
    MEM0_AVAILABLE = False
    print("⚠️ Mem0 library not installed - using traditional knowledge base")
    print("💡 Install with: pip install mem0ai")

# --- Knowledge Base System ---
class KnowledgeBase:
    """知识库系统 - 存储和检索成功的思维模板"""
    
    def __init__(self, gemini_model=None):
        self.templates = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.template_vectors = None
        self.knowledge_file = Path("/home/ubuntu/agent_outputs/agent_knowledge_base.json")
        self.gemini_model = gemini_model  # 添加 Gemini 模型支持
        
        # 加载已有知识库
        self.load_knowledge_base()
    
    def summarize_reasoning_process(self, question_text, detailed_reasoning, correct_answer):
        """使用LLM总结推理过程的关键步骤"""
        summarization_prompt = f"""Please summarize the key reasoning steps from the following detailed analysis. 
Focus on the essential logical steps and principles that led to the successful solution, without revealing the specific answer.

Task/Question: {question_text}

Detailed Reasoning Process:
{detailed_reasoning}

Please provide a concise summary of 4-5 key reasoning principles and methodological approaches that were crucial for solving this type of problem. Do not include the final answer.

Key Reasoning Summary:"""

        try:
            print("🧠 调用模型进行推理总结...")
            
            # Use gemini model for summarization
            response = self.gemini_model.generate(
                messages=[
                    ChatMessage(role=MessageRole.USER, content=summarization_prompt)
                ]
            )
            
            # Handle different response formats
            if hasattr(response, 'content'):
                summary = response.content
            elif isinstance(response, dict) and 'content' in response:
                summary = response['content']
            elif isinstance(response, str):
                summary = response
            else:
                summary = str(response)
            
            # Clean and validate the summary
            if summary and isinstance(summary, str):
                summary = summary.strip()
                # Remove common non-informative starts
                if summary.lower().startswith(('key reasoning summary:', 'summary:', 'the key reasoning')):
                    lines = summary.split('\n')
                    summary = '\n'.join(lines[1:]).strip() if len(lines) > 1 else summary
                
                # Ensure it's informative (not just generic text)
                if len(summary) > 50 and 'systematic analysis' not in summary.lower():
                    # Limit length
                    if len(summary) > 800:
                        summary = summary[:800] + "..."
                    print(f"✅ 成功生成推理总结: {summary[:100]}...")
                    return summary
                else:
                    print(f"⚠️ 模型生成的总结太通用或太短: {summary[:100]}")
                    return self._generate_manual_summary(question_text, detailed_reasoning)
            else:
                print(f"⚠️ 意外的模型响应类型: {type(summary)}")
                return self._generate_manual_summary(question_text, detailed_reasoning)
                
        except Exception as e:
            print(f"❌ 推理总结失败: {str(e)}")
            import traceback
            print(f"   详细错误: {traceback.format_exc()}")
            return self._generate_manual_summary(question_text, detailed_reasoning)
    
    def _generate_manual_summary(self, question_text, detailed_reasoning):
        """手动生成推理总结作为备用方案"""
        try:
            # Extract some key concepts from the question and reasoning
            question_lower = question_text.lower()
            reasoning_lower = detailed_reasoning.lower() if detailed_reasoning else ""
            
            # Identify domain-specific approaches
            if any(term in question_lower for term in ['data', 'analysis', 'csv', 'plot', 'graph']):
                return "Applied systematic data analysis with visualization and statistical interpretation approaches."
            elif any(term in question_lower for term in ['code', 'script', 'programming', 'function']):
                return "Used systematic programming approach with modular design and error handling principles."
            elif any(term in question_lower for term in ['search', 'research', 'find', 'information']):
                return "Applied comprehensive information retrieval with source verification and synthesis methods."
            elif any(term in question_lower for term in ['biomedical', 'biology', 'medical', 'health']):
                return "Applied biomedical reasoning with evidence-based analysis and scientific methodology."
            else:
                return "Applied systematic problem-solving approach with logical reasoning and evidence-based analysis."
        except:
            return "Applied systematic problem-solving approach with methodical analysis."

    def add_template(self, task_description, thought_process, solution_outcome, domain):
        """添加成功的思维模板到知识库（使用LLM总结，不存储具体答案）"""
        
        # 使用LLM总结关键推理步骤
        print("🧠 正在总结推理过程...")
        key_reasoning = self.summarize_reasoning_process(task_description, thought_process, solution_outcome)
        
        template = {
            'task': task_description,
            'key_reasoning': key_reasoning,  # 存储总结的关键推理步骤
            'domain': domain,
            'keywords': self.extract_keywords(task_description),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.templates.append(template)
        
        # 重新计算向量
        self.rebuild_vectors()
        
        # 限制知识库大小
        if len(self.templates) > 1000:
            self.templates = self.templates[-1000:]  # 保留最新1000个
            self.rebuild_vectors()
        
        # 保存到文件
        self.save_knowledge_base()
        
        print(f"💾 知识库新增模板（已总结），总数: {len(self.templates)}")
        
        # 返回成功状态
        return {
            "success": True,
            "message": f"Template added successfully. Total templates: {len(self.templates)}",
            "template_id": len(self.templates) - 1
        }
    
    def extract_keywords(self, text):
        """提取关键词 - 使用 Gemini 模型或回退到静态关键词"""
        
        # 如果有 Gemini 模型，使用 AI 进行关键词提取
        if self.gemini_model:
            try:
                keyword_prompt = f"""Extract 3-8 most important keywords from the following text, focusing on technical, scientific, medical, biological, data analysis, programming, and related professional terms.

Please return only keywords, separated by commas, without any other explanations or punctuation.

Text: {text}

Keywords:"""
                
                response = self.gemini_model.generate(
                    messages=[ChatMessage(role=MessageRole.USER, content=keyword_prompt)]
                )
                
                # 处理响应
                if hasattr(response, 'content'):
                    keywords_str = response.content.strip()
                elif isinstance(response, dict) and 'content' in response:
                    keywords_str = response['content'].strip()
                elif isinstance(response, str):
                    keywords_str = response.strip()
                else:
                    keywords_str = str(response).strip()
                
                # 解析关键词
                keywords = [kw.strip().lower() for kw in keywords_str.split(',')]
                keywords = [kw for kw in keywords if kw and len(kw) > 2]  # 过滤空白和太短的词
                
                if keywords:
                    return keywords
                    
            except Exception as e:
                print(f"⚠️ Gemini 关键词提取失败，回退到静态方法: {str(e)}")
        
        # 回退到原始的静态关键词方法
        tech_keywords = [
            # 通用技术关键词
            'data', 'analysis', 'code', 'script', 'programming', 'algorithm',
            'visualization', 'plot', 'chart', 'graph', 'statistics', 'model',
            'database', 'search', 'information', 'literature', 'paper',
            
            # 生物学通用术语
            'biomedical', 'biology', 'medical', 'research', 'scientific',
            'bioinformatics', 'computational', 'experimental', 'clinical',
            
            # 基于 Biomni 工具分类的专业术语
            'molecular_biology', 'molecular', 'cell_biology', 'cellular',
            'genetics', 'genetic', 'genomics', 'genome', 'genomic',
            'biochemistry', 'biochemical', 'immunology', 'immune', 'immunological',
            'microbiology', 'microbiological', 'microbial', 'bacterial',
            'cancer_biology', 'cancer', 'oncology', 'tumor', 'malignancy',
            'pathology', 'pathological', 'disease', 'disorder',
            'pharmacology', 'pharmacological', 'drug', 'therapeutic', 'treatment',
            'physiology', 'physiological', 'function', 'metabolism',
            'systems_biology', 'systems', 'network', 'pathway',
            'synthetic_biology', 'synthetic', 'engineering', 'design',
            'bioengineering', 'biomedical_engineering', 'biotechnology',
            'biophysics', 'biophysical', 'structural', 'protein', 'enzyme',
            
            # 实验技术和方法
            'sequencing', 'pcr', 'microscopy', 'imaging', 'assay',
            'chromatography', 'electrophoresis', 'blotting', 'culture',
            'transfection', 'transformation', 'cloning', 'expression',
            'purification', 'crystallization', 'spectroscopy',
            
            # 数据分析和计算
            'machine_learning', 'deep_learning', 'neural_network',
            'classification', 'clustering', 'regression', 'prediction',
            'simulation', 'modeling', 'optimization', 'annotation'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in tech_keywords if kw in text_lower]
        return found_keywords
    
    def rebuild_vectors(self):
        """重建向量表示"""
        if len(self.templates) == 0:
            return
        
        # 组合任务文本、关键推理和关键词
        texts = []
        for t in self.templates:
            # 处理新旧格式兼容性
            reasoning = t.get('key_reasoning', t.get('thought_process', ''))
            text = f"{t['task']} {reasoning} {' '.join(t['keywords'])}"
            texts.append(text)
        
        try:
            self.template_vectors = self.vectorizer.fit_transform(texts)
        except Exception as e:
            print(f"⚠️ 向量重建失败: {str(e)}")
            self.template_vectors = None
    
    def retrieve_similar_templates(self, task_description, top_k=3):
        """检索相似的思维模板"""
        if len(self.templates) == 0 or self.template_vectors is None:
            return []
        
        try:
            # 向量化当前任务
            task_vector = self.vectorizer.transform([task_description])
            
            # 计算相似度
            similarities = cosine_similarity(task_vector, self.template_vectors).flatten()
            
            # 获取top_k个最相似的模板
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar_templates = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # 相似度阈值
                    template = self.templates[idx].copy()
                    template['similarity'] = similarities[idx]
                    similar_templates.append(template)
            
            print(f"🔍 找到 {len(similar_templates)} 个相似模板")
            return similar_templates
            
        except Exception as e:
            print(f"⚠️ 模板检索失败: {str(e)}")
            return []
    
    def search_templates_by_keyword(self, keyword):
        """按关键词搜索模板"""
        matching_templates = []
        keyword_lower = keyword.lower()
        
        for template in self.templates:
            # 在任务描述、关键推理和关键词中搜索
            if (keyword_lower in template['task'].lower() or
                keyword_lower in template.get('key_reasoning', '').lower() or
                keyword_lower in ' '.join(template['keywords']).lower()):
                matching_templates.append(template)
        
        print(f"🔍 关键词 '{keyword}' 匹配到 {len(matching_templates)} 个模板")
        return matching_templates
    
    def save_knowledge_base(self):
        """保存知识库到文件"""
        try:
            # 确保输出目录存在
            self.knowledge_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存知识库失败: {str(e)}")
    
    def load_knowledge_base(self):
        """从文件加载知识库"""
        try:
            if self.knowledge_file.exists():
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    self.templates = json.load(f)
                
                # 重建向量
                self.rebuild_vectors()
                
                print(f"✅ 成功加载知识库，包含 {len(self.templates)} 个模板")
            else:
                print("📚 知识库文件不存在，从空白开始")
        except Exception as e:
            print(f"⚠️ 加载知识库失败: {str(e)}")
            self.templates = []


# --- Mem0 Enhanced Knowledge Base ---
class Mem0EnhancedKnowledgeBase:
    """使用 Mem0 的增强知识库系统 - 提供语义记忆和智能检索"""
    
    def __init__(self, gemini_model=None, use_mem0_platform=False, mem0_api_key=None, openrouter_api_key=None):
        self.gemini_model = gemini_model
        self.use_mem0_platform = use_mem0_platform
        self.mem0_api_key = mem0_api_key
        self.openrouter_api_key = openrouter_api_key
        self.fallback_kb = None  # 传统知识库作为后备
        
        # 初始化 Mem0
        if MEM0_AVAILABLE:
            try:
                if use_mem0_platform and mem0_api_key:
                    # 使用托管平台
                    print("🔗 初始化 Mem0 托管平台...")
                    self.memory = MemoryClient(api_key=mem0_api_key)
                else:
                    # 使用自托管版本
                    print("🏠 初始化 Mem0 自托管版本...")
                    config = self._get_mem0_config()
                    self.memory = Memory.from_config(config)
                
                print("✅ Mem0 初始化成功")
                self.mem0_enabled = True
            except Exception as e:
                print(f"❌ Mem0 初始化失败: {str(e)}")
                print("🔄 回退到传统知识库...")
                self.mem0_enabled = False
                self._init_fallback()
        else:
            print("📚 Mem0 不可用，使用传统知识库")
            self.mem0_enabled = False
            self._init_fallback()
    
    def _get_mem0_config(self):
        """获取 Mem0 配置"""
        config = {
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": self.openrouter_api_key  # 使用传入的API密钥
                }
            },
            "llm": {
                "provider": "openai", 
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": self.openrouter_api_key,
                    "openai_base_url": "https://openrouter.ai/api/v1"
                }
            },
            "vector_store": {
                "provider": "chroma",  # 使用本地Chroma数据库
                "config": {
                    "collection_name": "stella_knowledge_base",
                    "path": "/home/ubuntu/agent_outputs/mem0_db"
                }
            }
        }
        return config
    
    def _init_fallback(self):
        """初始化传统知识库作为后备"""
        self.fallback_kb = KnowledgeBase(gemini_model=self.gemini_model)
    
    def add_template(self, task_description, thought_process, solution_outcome, domain="general", user_id="agent_team"):
        """添加成功的思维模板到知识库"""
        if self.mem0_enabled:
            try:
                # 使用 Mem0 存储记忆
                conversation = [
                    {"role": "user", "content": f"Task: {task_description}"},
                    {"role": "assistant", "content": f"Reasoning: {thought_process}"},
                    {"role": "user", "content": f"Outcome: {solution_outcome}"}
                ]
                
                # 添加元数据
                metadata = {
                    "domain": domain,
                    "task_type": "problem_solving_template",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "keywords": self.extract_keywords(task_description)
                }
                
                result = self.memory.add(conversation, user_id=user_id, metadata=metadata)
                print(f"💾 Mem0: 成功保存模板到记忆系统")
                return result
                
            except Exception as e:
                print(f"⚠️ Mem0 保存失败，使用后备方法: {str(e)}")
                if self.fallback_kb:
                    return self.fallback_kb.add_template(task_description, thought_process, solution_outcome, domain)
                else:
                    return {"success": False, "message": f"Failed to add template: {str(e)}"}
        else:
            # 使用传统方法
            if self.fallback_kb:
                return self.fallback_kb.add_template(task_description, thought_process, solution_outcome, domain)
    
    def retrieve_similar_templates(self, task_description, top_k=3, user_id="agent_team"):
        """检索相似的思维模板"""
        if self.mem0_enabled:
            try:
                # 使用 Mem0 语义搜索
                results = self.memory.search(
                    query=task_description,
                    user_id=user_id,
                    limit=top_k
                )
                
                # 转换为兼容格式
                similar_templates = []
                for result in results.get('results', []):
                    template = {
                        'task': task_description,
                        'key_reasoning': result.get('memory', ''),
                        'domain': result.get('metadata', {}).get('domain', 'general'),
                        'keywords': result.get('metadata', {}).get('keywords', []),
                        'timestamp': result.get('metadata', {}).get('timestamp', ''),
                        'similarity': result.get('score', 0.0),
                        'memory_id': result.get('id', '')
                    }
                    similar_templates.append(template)
                
                return similar_templates
                
            except Exception as e:
                print(f"⚠️ Mem0 检索失败，使用后备方法: {str(e)}")
                if self.fallback_kb:
                    return self.fallback_kb.retrieve_similar_templates(task_description, top_k)
                return []
        else:
            # 使用传统方法
            if self.fallback_kb:
                return self.fallback_kb.retrieve_similar_templates(task_description, top_k)
            return []
    
    def search_memories_by_keyword(self, keyword, user_id="agent_team", limit=5):
        """按关键词搜索记忆"""
        if self.mem0_enabled:
            try:
                results = self.memory.search(
                    query=keyword,
                    user_id=user_id,
                    limit=limit
                )
                return results.get('results', [])
            except Exception as e:
                print(f"⚠️ Mem0 关键词搜索失败: {str(e)}")
                return []
        else:
            # 后备方法：使用传统搜索
            if self.fallback_kb and hasattr(self.fallback_kb, 'search_templates_by_keyword'):
                return self.fallback_kb.search_templates_by_keyword(keyword)
            return []
    
    def get_user_memories(self, user_id="agent_team"):
        """获取用户的所有记忆"""
        if self.mem0_enabled:
            try:
                memories = self.memory.get_all(user_id=user_id)
                return memories
            except Exception as e:
                print(f"⚠️ 获取用户记忆失败: {str(e)}")
                return []
        else:
            return []
    
    def delete_memory(self, memory_id, user_id="agent_team"):
        """删除特定记忆"""
        if self.mem0_enabled:
            try:
                self.memory.delete(memory_id=memory_id)
                print(f"🗑️ 已删除记忆: {memory_id}")
                return True
            except Exception as e:
                print(f"⚠️ 删除记忆失败: {str(e)}")
                return False
        return False
    
    def update_memory(self, memory_id, new_data):
        """更新现有记忆"""
        if self.mem0_enabled:
            try:
                result = self.memory.update(memory_id=memory_id, data=new_data)
                print(f"✏️ 已更新记忆: {memory_id}")
                return result
            except Exception as e:
                print(f"⚠️ 更新记忆失败: {str(e)}")
                return None
        return None
    
    def get_memory_stats(self, user_id="agent_team"):
        """获取记忆统计信息"""
        if self.mem0_enabled:
            try:
                memories = self.get_user_memories(user_id)
                return {
                    'total_memories': len(memories),
                    'backend': 'Mem0 Enhanced',
                    'user_id': user_id
                }
            except Exception as e:
                print(f"⚠️ 获取记忆统计失败: {str(e)}")
                return {'total_memories': 0, 'backend': 'Error', 'user_id': user_id}
        else:
            if self.fallback_kb:
                return {
                    'total_memories': len(self.fallback_kb.templates),
                    'backend': 'Traditional KnowledgeBase',
                    'user_id': user_id
                }
            return {'total_memories': 0, 'backend': 'No backend', 'user_id': user_id}
    
    def extract_keywords(self, text):
        """提取关键词 - 使用 Gemini 模型或回退到静态关键词"""
        if self.fallback_kb:
            return self.fallback_kb.extract_keywords(text)
        else:
            # 简单的静态关键词提取
            return [word.lower() for word in text.split() if len(word) > 3]

    # --- Multi-Agent Collaboration Layer ---
    
    def create_shared_workspace(self, workspace_id: str, task_description: str, participating_agents: list = None):
        """创建智能体团队的共享工作空间"""
        if not self.mem0_enabled:
            return {"success": False, "message": "Mem0 not available for shared workspace"}
        
        try:
            participating_agents = participating_agents or ["dev_agent", "manager_agent", "critic_agent", "tool_creation_agent"]
            
            workspace_memory = [{
                "role": "system", 
                "content": f"Shared workspace '{workspace_id}' created for collaborative task"
            }, {
                "role": "assistant", 
                "content": f"Task: {task_description}\nParticipating agents: {', '.join(participating_agents)}"
            }]
            
            metadata = {
                "memory_type": "shared_workspace",
                "workspace_id": workspace_id,
                "task_description": task_description,
                "participating_agents": participating_agents,
                "status": "active",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result = self.memory.add(workspace_memory, user_id="shared_workspace", metadata=metadata)
            
            return {
                "success": True,
                "workspace_id": workspace_id,
                "memory_id": result.get('id', ''),
                "participating_agents": participating_agents
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error creating workspace: {str(e)}"}
    
    def add_workspace_memory(self, workspace_id: str, agent_name: str, content: str, memory_type: str = "observation"):
        """向共享工作空间添加记忆"""
        if not self.mem0_enabled:
            return {"success": False, "message": "Mem0 not available"}
        
        try:
            memory_entry = [{
                "role": "user",
                "content": f"Agent: {agent_name}"
            }, {
                "role": "assistant", 
                "content": content
            }]
            
            metadata = {
                "memory_type": memory_type,  # observation, discovery, result, question
                "workspace_id": workspace_id,
                "agent_name": agent_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result = self.memory.add(memory_entry, user_id="shared_workspace", metadata=metadata)
            
            return {
                "success": True,
                "memory_id": result.get('id', ''),
                "workspace_id": workspace_id,
                "agent_name": agent_name
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error adding workspace memory: {str(e)}"}
    
    def get_workspace_memories(self, workspace_id: str, memory_type: str = "all", limit: int = 20):
        """获取共享工作空间的记忆"""
        if not self.mem0_enabled:
            return {"success": False, "memories": []}
        
        try:
            # 搜索特定工作空间的记忆
            results = self.memory.search(
                query=f"workspace {workspace_id}",
                user_id="shared_workspace",
                limit=limit * 2  # 获取更多以便过滤
            )
            
            # 过滤匹配的记忆
            workspace_memories = []
            for result in results.get('results', []):
                metadata = result.get('metadata', {})
                if metadata.get('workspace_id') == workspace_id:
                    if memory_type == "all" or metadata.get('memory_type') == memory_type:
                        workspace_memories.append(result)
            
            # 按时间排序（最新的在前）
            workspace_memories.sort(
                key=lambda x: x.get('metadata', {}).get('timestamp', ''), 
                reverse=True
            )
            
            return {
                "success": True,
                "memories": workspace_memories[:limit],
                "total_found": len(workspace_memories)
            }
            
        except Exception as e:
            return {"success": False, "memories": [], "message": str(e)}
    
    # --- Task Decomposition and Tracking ---
    
    def create_task_breakdown(self, task_id: str, main_task: str, subtasks: list, agent_assignments: dict = None):
        """创建任务分解和追踪记录"""
        if not self.mem0_enabled:
            return {"success": False, "message": "Mem0 not available"}
        
        try:
            agent_assignments = agent_assignments or {}
            
            # 创建主任务记录
            task_memory = [{
                "role": "user",
                "content": f"Task Breakdown for: {main_task}"
            }, {
                "role": "assistant",
                "content": f"Subtasks: {json.dumps(subtasks, ensure_ascii=False, indent=2)}\nAssignments: {json.dumps(agent_assignments, ensure_ascii=False, indent=2)}"
            }]
            
            metadata = {
                "memory_type": "task_breakdown",
                "task_id": task_id,
                "main_task": main_task,
                "subtasks": subtasks,
                "agent_assignments": agent_assignments,
                "status": "planned",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result = self.memory.add(task_memory, user_id="task_tracking", metadata=metadata)
            
            # 为每个子任务创建状态追踪
            for i, subtask in enumerate(subtasks):
                subtask_memory = [{
                    "role": "system",
                    "content": f"Subtask {i+1} of {task_id}"
                }, {
                    "role": "assistant",
                    "content": subtask
                }]
                
                subtask_metadata = {
                    "memory_type": "subtask_status",
                    "task_id": task_id,
                    "subtask_index": i,
                    "subtask_content": subtask,
                    "status": "pending",
                    "assigned_agent": agent_assignments.get(str(i), "unassigned"),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.memory.add(subtask_memory, user_id="task_tracking", metadata=subtask_metadata)
            
            return {
                "success": True,
                "task_id": task_id,
                "memory_id": result.get('id', ''),
                "subtasks_created": len(subtasks)
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error creating task breakdown: {str(e)}"}
    
    def update_subtask_status(self, task_id: str, subtask_index: int, new_status: str, agent_name: str, progress_notes: str = ""):
        """更新子任务状态"""
        if not self.mem0_enabled:
            return {"success": False, "message": "Mem0 not available"}
        
        try:
            # 搜索特定子任务
            results = self.memory.search(
                query=f"task {task_id} subtask {subtask_index}",
                user_id="task_tracking",
                limit=10
            )
            
            # 找到对应的子任务记录
            target_memory = None
            for result in results.get('results', []):
                metadata = result.get('metadata', {})
                if (metadata.get('task_id') == task_id and 
                    metadata.get('subtask_index') == subtask_index and
                    metadata.get('memory_type') == 'subtask_status'):
                    target_memory = result
                    break
            
            if not target_memory:
                return {"success": False, "message": f"Subtask {subtask_index} not found for task {task_id}"}
            
            # 创建状态更新记录
            update_memory = [{
                "role": "user",
                "content": f"Status update for subtask {subtask_index} of {task_id}"
            }, {
                "role": "assistant",
                "content": f"New status: {new_status}\nUpdated by: {agent_name}\nNotes: {progress_notes}"
            }]
            
            update_metadata = {
                "memory_type": "subtask_update",
                "task_id": task_id,
                "subtask_index": subtask_index,
                "previous_status": target_memory.get('metadata', {}).get('status', 'unknown'),
                "new_status": new_status,
                "updated_by": agent_name,
                "progress_notes": progress_notes,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result = self.memory.add(update_memory, user_id="task_tracking", metadata=update_metadata)
            
            return {
                "success": True,
                "task_id": task_id,
                "subtask_index": subtask_index,
                "new_status": new_status,
                "memory_id": result.get('id', '')
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error updating subtask status: {str(e)}"}
    
    def get_task_progress(self, task_id: str):
        """获取任务进度概览"""
        if not self.mem0_enabled:
            return {"success": False, "progress": {}}
        
        try:
            # 搜索任务相关的所有记忆
            results = self.memory.search(
                query=f"task {task_id}",
                user_id="task_tracking",
                limit=50
            )
            
            # 分析任务进度
            main_task_info = None
            subtask_statuses = {}
            latest_updates = []
            
            for result in results.get('results', []):
                metadata = result.get('metadata', {})
                if metadata.get('task_id') != task_id:
                    continue
                
                memory_type = metadata.get('memory_type')
                
                if memory_type == 'task_breakdown':
                    main_task_info = metadata
                elif memory_type == 'subtask_update':
                    subtask_idx = metadata.get('subtask_index')
                    if subtask_idx is not None:
                        # 保留最新的状态更新
                        timestamp = metadata.get('timestamp', '')
                        if (subtask_idx not in subtask_statuses or 
                            timestamp > subtask_statuses[subtask_idx].get('timestamp', '')):
                            subtask_statuses[subtask_idx] = metadata
                        latest_updates.append(metadata)
            
            # 计算总体进度
            total_subtasks = len(main_task_info.get('subtasks', [])) if main_task_info else 0
            completed_count = sum(1 for status in subtask_statuses.values() 
                                 if status.get('new_status') == 'completed')
            in_progress_count = sum(1 for status in subtask_statuses.values() 
                                   if status.get('new_status') == 'in_progress')
            
            progress_percentage = (completed_count / total_subtasks * 100) if total_subtasks > 0 else 0
            
            return {
                "success": True,
                "progress": {
                    "task_id": task_id,
                    "main_task": main_task_info.get('main_task', '') if main_task_info else '',
                    "total_subtasks": total_subtasks,
                    "completed": completed_count,
                    "in_progress": in_progress_count,
                    "pending": total_subtasks - completed_count - in_progress_count,
                    "progress_percentage": round(progress_percentage, 1),
                    "subtask_details": subtask_statuses,
                    "recent_updates": sorted(latest_updates, 
                                           key=lambda x: x.get('timestamp', ''), 
                                           reverse=True)[:5]
                }
            }
            
        except Exception as e:
            return {"success": False, "progress": {}, "message": str(e)}
    
    # --- Cross-Agent Knowledge Transfer ---
    
    def share_discovery(self, agent_name: str, discovery_title: str, discovery_content: str, tags: list = None, related_task: str = ""):
        """智能体分享发现和经验"""
        if not self.mem0_enabled:
            return {"success": False, "message": "Mem0 not available"}
        
        try:
            tags = tags or []
            
            discovery_memory = [{
                "role": "user",
                "content": f"Discovery by {agent_name}: {discovery_title}"
            }, {
                "role": "assistant",
                "content": discovery_content
            }]
            
            metadata = {
                "memory_type": "agent_discovery",
                "agent_name": agent_name,
                "discovery_title": discovery_title,
                "tags": tags,
                "related_task": related_task,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result = self.memory.add(discovery_memory, user_id="knowledge_sharing", metadata=metadata)
            
            return {
                "success": True,
                "discovery_id": result.get('id', ''),
                "agent_name": agent_name,
                "title": discovery_title
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error sharing discovery: {str(e)}"}
    
    def search_discoveries(self, query: str = "", agent_name: str = "", tags: list = None, limit: int = 10):
        """搜索其他智能体的发现和经验"""
        if not self.mem0_enabled:
            return {"success": False, "discoveries": []}
        
        try:
            # 构建搜索查询
            search_query = query if query else "discovery"
            if agent_name:
                search_query += f" {agent_name}"
            
            results = self.memory.search(
                query=search_query,
                user_id="knowledge_sharing",
                limit=limit * 2
            )
            
            # 过滤和整理发现
            discoveries = []
            for result in results.get('results', []):
                metadata = result.get('metadata', {})
                if metadata.get('memory_type') != 'agent_discovery':
                    continue
                
                # 按智能体过滤
                if agent_name and metadata.get('agent_name') != agent_name:
                    continue
                
                # 按标签过滤
                if tags:
                    result_tags = metadata.get('tags', [])
                    if not any(tag in result_tags for tag in tags):
                        continue
                
                discoveries.append({
                    "discovery_id": result.get('id', ''),
                    "agent_name": metadata.get('agent_name', ''),
                    "title": metadata.get('discovery_title', ''),
                    "content": result.get('memory', ''),
                    "tags": metadata.get('tags', []),
                    "related_task": metadata.get('related_task', ''),
                    "timestamp": metadata.get('timestamp', ''),
                    "relevance_score": result.get('score', 0.0)
                })
            
            # 按相关性排序
            discoveries.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                "success": True,
                "discoveries": discoveries[:limit],
                "total_found": len(discoveries)
            }
            
        except Exception as e:
            return {"success": False, "discoveries": [], "message": str(e)}
    
    def get_agent_contributions(self, agent_name: str):
        """获取特定智能体的贡献统计"""
        if not self.mem0_enabled:
            return {"success": False, "contributions": {}}
        
        try:
            results = self.memory.search(
                query=f"agent {agent_name}",
                user_id="knowledge_sharing",
                limit=100
            )
            
            discoveries = 0
            workspace_contributions = 0
            task_updates = 0
            
            for result in results.get('results', []):
                metadata = result.get('metadata', {})
                if metadata.get('agent_name') == agent_name:
                    memory_type = metadata.get('memory_type', '')
                    if memory_type == 'agent_discovery':
                        discoveries += 1
                    elif memory_type in ['observation', 'discovery', 'result']:
                        workspace_contributions += 1
                    elif memory_type == 'subtask_update':
                        task_updates += 1
            
            return {
                "success": True,
                "contributions": {
                    "agent_name": agent_name,
                    "discoveries_shared": discoveries,
                    "workspace_contributions": workspace_contributions,
                    "task_updates": task_updates,
                    "total_contributions": discoveries + workspace_contributions + task_updates
                }
            }
            
        except Exception as e:
            return {"success": False, "contributions": {}, "message": str(e)}

    def search_templates_by_keyword(self, keyword):
        """按关键词搜索模板 - 兼容传统知识库接口"""
        if self.mem0_enabled:
            try:
                results = self.memory.search(
                    query=keyword,
                    user_id="agent_team",
                    limit=10
                )
                return results.get('results', [])
            except Exception as e:
                print(f"⚠️ Mem0 关键词搜索失败: {str(e)}")
                return []
        else:
            # 后备方法：使用传统搜索
            if self.fallback_kb and hasattr(self.fallback_kb, 'search_templates_by_keyword'):
                return self.fallback_kb.search_templates_by_keyword(keyword)
            return [] 