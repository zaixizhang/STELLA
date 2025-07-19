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

# Mem0 integration for enhanced memory management
try:
    from mem0 import Memory, MemoryClient
    MEM0_AVAILABLE = True
    print("✅ Mem0 library available - enhanced memory features enabled")
except ImportError:
    MEM0_AVAILABLE = False
    print("⚠️ Mem0 library not installed - using traditional knowledge base")
    print("💡 Install with: pip install mem0ai")

# Import traditional KnowledgeBase for fallback
from Knowledge_base import KnowledgeBase


# --- Base Memory Component ---
class BaseMemoryComponent:
    """所有记忆组件的基类"""
    
    def __init__(self, component_name: str, gemini_model=None, mem0_config=None):
        self.component_name = component_name
        self.gemini_model = gemini_model
        self.mem0_config = mem0_config
        self.memory = None
        self.mem0_enabled = False
        
        # 初始化 Mem0
        if MEM0_AVAILABLE and mem0_config:
            try:
                if mem0_config.get('use_platform', False):
                    # 使用托管平台
                    self.memory = MemoryClient(api_key=mem0_config.get('api_key'))
                else:
                    # 使用自托管版本
                    config = self._get_component_config()
                    self.memory = Memory.from_config(config)
                
                self.mem0_enabled = True
                print(f"✅ {self.component_name} Mem0 初始化成功")
                
            except Exception as e:
                print(f"❌ {self.component_name} Mem0 初始化失败: {str(e)}")
                self.mem0_enabled = False
    
    def _get_component_config(self):
        """获取组件特定的 Mem0 配置"""
        base_config = {
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": self.mem0_config.get('openrouter_api_key')
                }
            },
            "llm": {
                "provider": "openai", 
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": self.mem0_config.get('openrouter_api_key'),
                    "openai_base_url": "https://openrouter.ai/api/v1"
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": f"stella_{self.component_name}",
                    "path": f"/home/ubuntu/agent_outputs/mem0_db/{self.component_name}"
                }
            }
        }
        return base_config


# --- 1. Knowledge Base Component ---
class KnowledgeMemory(BaseMemoryComponent):
    """专门管理思维模板和解题经验的记忆组件"""
    
    def __init__(self, gemini_model=None, mem0_config=None):
        super().__init__("knowledge", gemini_model, mem0_config)
        self.fallback_kb = None
        
        # 如果 Mem0 不可用，初始化传统知识库
        if not self.mem0_enabled:
            self.fallback_kb = KnowledgeBase(gemini_model=gemini_model)
    
    def add_template(self, task_description: str, thought_process: str, solution_outcome: str, 
                    domain: str = "general", user_id: str = "agent_team"):
        """添加成功的思维模板"""
        if self.mem0_enabled:
            try:
                conversation = [
                    {"role": "user", "content": f"Task: {task_description}"},
                    {"role": "assistant", "content": f"Reasoning: {thought_process}"},
                    {"role": "user", "content": f"Outcome: {solution_outcome}"}
                ]
                
                metadata = {
                    "type": "problem_solving_template",
                    "domain": domain,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "keywords": self._extract_keywords(task_description)
                }
                
                result = self.memory.add(conversation, user_id=user_id, metadata=metadata)
                print(f"💾 KnowledgeMemory: 成功保存思维模板")
                return {"success": True, "memory_id": result.get('id', '')}
                
            except Exception as e:
                print(f"⚠️ KnowledgeMemory 保存失败: {str(e)}")
                if self.fallback_kb:
                    return self.fallback_kb.add_template(task_description, thought_process, solution_outcome, domain)
        else:
            if self.fallback_kb:
                return self.fallback_kb.add_template(task_description, thought_process, solution_outcome, domain)
        
        return {"success": False, "message": "No available backend"}
    
    def search_templates(self, task_description: str, top_k: int = 3, user_id: str = "agent_team"):
        """搜索相似的思维模板"""
        if self.mem0_enabled:
            try:
                results = self.memory.search(
                    query=task_description,
                    user_id=user_id,
                    limit=top_k
                )
                
                templates = []
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
                    templates.append(template)
                
                return {"success": True, "templates": templates}
                
            except Exception as e:
                print(f"⚠️ KnowledgeMemory 检索失败: {str(e)}")
                if self.fallback_kb:
                    return {"success": True, "templates": self.fallback_kb.retrieve_similar_templates(task_description, top_k)}
        else:
            if self.fallback_kb:
                return {"success": True, "templates": self.fallback_kb.retrieve_similar_templates(task_description, top_k)}
        
        return {"success": False, "templates": []}
    
    def get_stats(self, user_id: str = "agent_team"):
        """获取知识库统计信息"""
        if self.mem0_enabled:
            try:
                # 获取用户的所有记忆
                all_results = self.memory.search(
                    query="template problem solving",
                    user_id=user_id,
                    limit=1000  # 大数量来获取统计
                )
                return {
                    'component': 'KnowledgeMemory',
                    'backend': 'Mem0 Enhanced',
                    'total_templates': len(all_results.get('results', [])),
                    'user_id': user_id
                }
            except Exception as e:
                return {
                    'component': 'KnowledgeMemory',
                    'backend': 'Error',
                    'total_templates': 0,
                    'error': str(e)
                }
        else:
            if self.fallback_kb:
                return {
                    'component': 'KnowledgeMemory',
                    'backend': 'Traditional KnowledgeBase',
                    'total_templates': len(self.fallback_kb.templates),
                    'user_id': user_id
                }
        
        return {
            'component': 'KnowledgeMemory',
            'backend': 'No backend',
            'total_templates': 0
        }
    
    def _extract_keywords(self, text):
        """提取关键词"""
        if self.fallback_kb:
            return self.fallback_kb.extract_keywords(text)
        else:
            # 简单的关键词提取
            return [word.lower() for word in text.split() if len(word) > 3]


# --- 2. Collaboration Memory Component ---
class CollaborationMemory(BaseMemoryComponent):
    """专门管理多智能体协作记忆的组件"""
    
    def __init__(self, gemini_model=None, mem0_config=None):
        super().__init__("collaboration", gemini_model, mem0_config)
    
    def create_workspace(self, workspace_id: str, task_description: str, 
                        participating_agents: list = None):
        """创建智能体团队的共享工作空间"""
        if not self.mem0_enabled:
            return {"success": False, "message": "Mem0 not available for collaboration"}
        
        try:
            participating_agents = participating_agents or ["dev_agent", "manager_agent", "critic_agent"]
            
            workspace_memory = [{
                "role": "system", 
                "content": f"Shared workspace '{workspace_id}' created for collaborative task"
            }, {
                "role": "assistant", 
                "content": f"Task: {task_description}\nParticipating agents: {', '.join(participating_agents)}"
            }]
            
            metadata = {
                "type": "workspace_creation",
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
    
    def add_agent_observation(self, workspace_id: str, agent_name: str, content: str, 
                             observation_type: str = "discovery"):
        """智能体添加观察或发现到共享工作空间"""
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
                "type": "agent_observation",
                "observation_type": observation_type,  # discovery, result, question, insight
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
            return {"success": False, "message": f"Error adding observation: {str(e)}"}
    
    def get_workspace_context(self, workspace_id: str, agent_perspective: str = "all", 
                             limit: int = 20):
        """获取工作空间的协作上下文"""
        if not self.mem0_enabled:
            return {"success": False, "observations": []}
        
        try:
            # 搜索特定工作空间的记忆
            results = self.memory.search(
                query=f"workspace {workspace_id}",
                user_id="shared_workspace",
                limit=limit * 2
            )
            
            # 过滤和组织观察
            observations = []
            for result in results.get('results', []):
                metadata = result.get('metadata', {})
                if metadata.get('workspace_id') == workspace_id:
                    if agent_perspective == "all" or metadata.get('agent_name') == agent_perspective:
                        observations.append({
                            "memory_id": result.get('id', ''),
                            "agent_name": metadata.get('agent_name', ''),
                            "content": result.get('memory', ''),
                            "observation_type": metadata.get('observation_type', ''),
                            "timestamp": metadata.get('timestamp', ''),
                            "score": result.get('score', 0.0)
                        })
            
            # 按时间排序
            observations.sort(
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            )
            
            return {
                "success": True,
                "workspace_id": workspace_id,
                "observations": observations[:limit],
                "total_found": len(observations)
            }
            
        except Exception as e:
            return {"success": False, "observations": [], "message": str(e)}
    
    def get_agent_contributions(self, workspace_id: str, agent_name: str, limit: int = 10):
        """获取特定智能体在工作空间的贡献"""
        context = self.get_workspace_context(workspace_id, agent_name, limit)
        if context["success"]:
            return {
                "success": True,
                "agent_name": agent_name,
                "workspace_id": workspace_id,
                "contributions": context["observations"]
            }
        return {"success": False, "contributions": []}


# --- 3. Session Memory Component ---
class SessionMemory(BaseMemoryComponent):
    """专门管理多轮对话上下文的记忆组件"""
    
    def __init__(self, gemini_model=None, mem0_config=None):
        super().__init__("session", gemini_model, mem0_config)
    
    def start_session(self, session_id: str, user_id: str, initial_context: str = ""):
        """开始新的会话"""
        if not self.mem0_enabled:
            return {"success": False, "message": "Mem0 not available for sessions"}
        
        try:
            session_start = [{
                "role": "system",
                "content": f"Session {session_id} started"
            }, {
                "role": "assistant",
                "content": f"Initial context: {initial_context}"
            }]
            
            metadata = {
                "type": "session_start",
                "session_id": session_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "active"
            }
            
            result = self.memory.add(session_start, user_id=user_id, metadata=metadata)
            
            return {
                "success": True,
                "session_id": session_id,
                "memory_id": result.get('id', '')
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error starting session: {str(e)}"}
    
    def add_conversation_turn(self, session_id: str, user_id: str, user_message: str, 
                             assistant_response: str, turn_type: str = "normal"):
        """添加对话轮次"""
        if not self.mem0_enabled:
            return {"success": False, "message": "Mem0 not available"}
        
        try:
            conversation_turn = [{
                "role": "user",
                "content": user_message
            }, {
                "role": "assistant",
                "content": assistant_response
            }]
            
            metadata = {
                "type": "conversation_turn",
                "turn_type": turn_type,  # normal, task_step, question, result
                "session_id": session_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result = self.memory.add(conversation_turn, user_id=user_id, metadata=metadata)
            
            return {
                "success": True,
                "memory_id": result.get('id', ''),
                "session_id": session_id
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error adding turn: {str(e)}"}
    
    def get_session_context(self, session_id: str, user_id: str, limit: int = 10):
        """获取会话上下文"""
        if not self.mem0_enabled:
            return {"success": False, "context": []}
        
        try:
            # 搜索特定会话的记忆
            results = self.memory.search(
                query=f"session {session_id}",
                user_id=user_id,
                limit=limit * 2
            )
            
            # 过滤和组织会话上下文
            context = []
            for result in results.get('results', []):
                metadata = result.get('metadata', {})
                if metadata.get('session_id') == session_id:
                    context.append({
                        "memory_id": result.get('id', ''),
                        "content": result.get('memory', ''),
                        "turn_type": metadata.get('turn_type', 'normal'),
                        "timestamp": metadata.get('timestamp', ''),
                        "score": result.get('score', 0.0)
                    })
            
            # 按时间排序
            context.sort(key=lambda x: x.get('timestamp', ''))
            
            return {
                "success": True,
                "session_id": session_id,
                "context": context[:limit],
                "total_turns": len(context)
            }
            
        except Exception as e:
            return {"success": False, "context": [], "message": str(e)}
    
    def get_user_preferences(self, user_id: str):
        """获取用户偏好和历史行为模式"""
        if not self.mem0_enabled:
            return {"success": False, "preferences": {}}
        
        try:
            # 搜索用户的所有会话记忆
            results = self.memory.search(
                query="conversation preference pattern",
                user_id=user_id,
                limit=50
            )
            
            # 分析偏好（简单实现）
            preferences = {
                "total_sessions": len(results.get('results', [])),
                "common_topics": [],
                "interaction_style": "standard"
            }
            
            return {
                "success": True,
                "user_id": user_id,
                "preferences": preferences
            }
            
        except Exception as e:
            return {"success": False, "preferences": {}, "message": str(e)}


# --- Main Memory Manager ---
class MemoryManager:
    """统一的内存管理器 - 协调所有记忆组件"""
    
    def __init__(self, gemini_model=None, use_mem0=False, mem0_platform=False, 
                 mem0_api_key=None, openrouter_api_key=None):
        
        self.gemini_model = gemini_model
        self.mem0_config = {
            'use_platform': mem0_platform,
            'api_key': mem0_api_key,
            'openrouter_api_key': openrouter_api_key
        } if use_mem0 else None
        
        # 初始化各个记忆组件
        print("🧠 初始化统一内存管理系统...")
        
        self.knowledge = KnowledgeMemory(gemini_model, self.mem0_config)
        self.collaboration = CollaborationMemory(gemini_model, self.mem0_config)
        self.session = SessionMemory(gemini_model, self.mem0_config)
        
        print("✅ MemoryManager 初始化完成")
        self._print_stats()
    
    def _print_stats(self):
        """打印各组件状态"""
        knowledge_stats = self.knowledge.get_stats()
        print(f"📚 知识记忆: {knowledge_stats['backend']} - {knowledge_stats['total_templates']} 个模板")
        
        if self.collaboration.mem0_enabled:
            print(f"🤝 协作记忆: Mem0 Enhanced - 已启用")
        else:
            print(f"🤝 协作记忆: 不可用")
            
        if self.session.mem0_enabled:
            print(f"💬 会话记忆: Mem0 Enhanced - 已启用")
        else:
            print(f"💬 会话记忆: 不可用")
    
    def get_overall_stats(self):
        """获取整体统计信息"""
        return {
            "knowledge": self.knowledge.get_stats(),
            "collaboration_enabled": self.collaboration.mem0_enabled,
            "session_enabled": self.session.mem0_enabled,
            "manager_version": "v1.0"
        }
    
    # --- 便捷方法用于向后兼容 ---
    def add_template(self, *args, **kwargs):
        """向后兼容的模板添加方法"""
        return self.knowledge.add_template(*args, **kwargs)
    
    def retrieve_similar_templates(self, *args, **kwargs):
        """向后兼容的模板检索方法"""
        result = self.knowledge.search_templates(*args, **kwargs)
        return result.get('templates', []) if result['success'] else []
    
    def get_memory_stats(self, *args, **kwargs):
        """向后兼容的统计方法"""
        return self.knowledge.get_stats(*args, **kwargs) 