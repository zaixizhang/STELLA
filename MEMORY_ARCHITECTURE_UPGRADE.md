# STELLA 内存架构升级指南

## 🚀 新架构概述

新的 **MemoryManager** 采用清晰的职责分离设计，提供更好的可维护性和扩展性：

```
MemoryManager (主管理器)
├── KnowledgeMemory    # 专门管理思维模板和解题经验
├── CollaborationMemory # 专门管理多智能体协作记忆  
└── SessionMemory      # 专门管理多轮对话上下文
```

## 📊 架构对比

### 原架构问题
- `global_knowledge_base` 承担过多职责
- 模板管理与协作记忆混合存储
- 难以独立优化不同类型的记忆
- 命名空间冲突

### 新架构优势
- **清晰职责分离**：每个组件专注单一职责
- **独立配置**：不同记忆类型可使用不同存储策略
- **更好扩展性**：易于添加新的记忆类型
- **向后兼容**：保持现有 API 接口

## 🛠️ 使用方法

### 1. 基本初始化

```python
from memory_manager import MemoryManager

# 创建内存管理器
memory_manager = MemoryManager(
    gemini_model=gemini_model,
    use_mem0=True,
    mem0_platform=False,  # 使用自托管
    openrouter_api_key="your-api-key"
)

# 获取整体状态
stats = memory_manager.get_overall_stats()
print(f"知识记忆: {stats['knowledge']['backend']}")
print(f"协作记忆: {'启用' if stats['collaboration_enabled'] else '禁用'}")
print(f"会话记忆: {'启用' if stats['session_enabled'] else '禁用'}")
```

### 2. 知识模板管理

```python
# 添加思维模板
result = memory_manager.knowledge.add_template(
    task_description="优化 React 组件性能",
    thought_process="使用 React.memo 和 useMemo 避免不必要的重渲染",
    solution_outcome="组件渲染性能提升 60%",
    domain="前端开发",
    user_id="developer_001"
)

# 搜索相似模板
templates = memory_manager.knowledge.search_templates(
    task_description="React 性能问题",
    top_k=3,
    user_id="developer_001"
)

if templates["success"]:
    for template in templates["templates"]:
        print(f"模板: {template['task']}")
        print(f"推理: {template['key_reasoning']}")
```

### 3. 多智能体协作记忆

```python
# 创建共享工作空间
workspace = memory_manager.collaboration.create_workspace(
    workspace_id="project_alpha",
    task_description="开发新功能模块",
    participating_agents=["dev_agent", "test_agent", "review_agent"]
)

# 智能体添加观察
memory_manager.collaboration.add_agent_observation(
    workspace_id="project_alpha",
    agent_name="dev_agent",
    content="发现性能瓶颈在数据库查询",
    observation_type="discovery"
)

# 获取工作空间上下文
context = memory_manager.collaboration.get_workspace_context(
    workspace_id="project_alpha",
    limit=10
)

for obs in context["observations"]:
    print(f"{obs['agent_name']}: {obs['content']}")
```

### 4. 会话记忆管理

```python
# 开始新会话
session = memory_manager.session.start_session(
    session_id="chat_001",
    user_id="user_123",
    initial_context="讨论 AI 项目架构设计"
)

# 添加对话轮次
memory_manager.session.add_conversation_turn(
    session_id="chat_001",
    user_id="user_123",
    user_message="如何设计可扩展的 AI 系统？",
    assistant_response="建议采用微服务架构...",
    turn_type="technical_discussion"
)

# 获取会话上下文
context = memory_manager.session.get_session_context(
    session_id="chat_001",
    user_id="user_123",
    limit=5
)
```

## 🔄 迁移步骤

### 步骤 1：更新导入
```python
# 旧代码
from Knowledge_base import KnowledgeBase, Mem0EnhancedKnowledgeBase

# 新代码
from memory_manager import MemoryManager
```

### 步骤 2：替换全局变量
```python
# 旧代码
global_knowledge_base = Mem0EnhancedKnowledgeBase(...)

# 新代码
global_memory_manager = MemoryManager(...)
```

### 步骤 3：更新函数调用
```python
# 旧代码 - 模板操作
global_knowledge_base.add_template(task, process, outcome, domain, user_id)
global_knowledge_base.retrieve_similar_templates(task, top_k, user_id)

# 新代码 - 分离的组件操作
global_memory_manager.knowledge.add_template(task, process, outcome, domain, user_id)
global_memory_manager.knowledge.search_templates(task, top_k, user_id)

# 旧代码 - 协作功能
global_knowledge_base.create_shared_workspace(workspace_id, task, agents)
global_knowledge_base.add_workspace_memory(workspace_id, agent, content, type)

# 新代码 - 专门的协作记忆
global_memory_manager.collaboration.create_workspace(workspace_id, task, agents)
global_memory_manager.collaboration.add_agent_observation(workspace_id, agent, content, type)
```

## 📈 性能优化建议

### 1. 组件独立配置
```python
# 为不同组件使用不同的数据库配置
memory_manager = MemoryManager(
    gemini_model=gemini_model,
    use_mem0=True,
    mem0_config={
        'knowledge': {'collection_name': 'stella_knowledge', 'path': '/db/knowledge'},
        'collaboration': {'collection_name': 'stella_collab', 'path': '/db/collab'},
        'session': {'collection_name': 'stella_session', 'path': '/db/session'}
    }
)
```

### 2. 批量操作
```python
# 批量添加模板
templates = [
    {"task": "任务1", "process": "过程1", "outcome": "结果1"},
    {"task": "任务2", "process": "过程2", "outcome": "结果2"},
]

for template in templates:
    memory_manager.knowledge.add_template(**template, user_id="batch_user")
```

## 🔧 故障排除

### 常见问题

1. **Mem0 初始化失败**
   - 检查 API 密钥配置
   - 确认网络连接
   - 查看错误日志

2. **向后兼容性问题**
   ```python
   # 使用向后兼容方法
   result = memory_manager.add_template(...)  # 自动路由到 knowledge 组件
   templates = memory_manager.retrieve_similar_templates(...)
   ```

3. **性能问题**
   - 使用适当的限制参数
   - 考虑异步操作
   - 监控内存使用

## 🚧 待完成功能

- [ ] 完成所有工具函数的迁移
- [ ] 添加异步操作支持  
- [ ] 实现更细粒度的权限控制
- [ ] 添加内存使用监控
- [ ] 支持自定义存储后端

## 📝 示例脚本

### 完整使用示例
```python
#!/usr/bin/env python3
"""STELLA 新内存架构使用示例"""

import sys
sys.path.append('/home/ubuntu/agents/STELLA')

from memory_manager import MemoryManager
from smolagents import OpenAIServerModel

def main():
    # 初始化模型
    gemini_model = OpenAIServerModel(
        model_id="google/gemini-2.5-pro",
        api_base="https://openrouter.ai/api/v1",
        api_key="your-api-key",
        temperature=0.1,
    )
    
    # 创建内存管理器
    memory_manager = MemoryManager(
        gemini_model=gemini_model,
        use_mem0=True,
        mem0_platform=False,
        openrouter_api_key="your-api-key"
    )
    
    # 测试知识模板
    print("=== 测试知识模板 ===")
    result = memory_manager.knowledge.add_template(
        task_description="解决数据库连接超时问题",
        thought_process="增加连接池大小，设置合理的超时时间，添加重试机制",
        solution_outcome="连接成功率从 85% 提升到 99.5%",
        domain="后端开发",
        user_id="dev_team"
    )
    print(f"添加模板结果: {result}")
    
    # 测试协作记忆
    print("\n=== 测试协作记忆 ===")
    workspace = memory_manager.collaboration.create_workspace(
        workspace_id="db_optimization",
        task_description="数据库性能优化项目",
        participating_agents=["dba_agent", "dev_agent", "ops_agent"]
    )
    print(f"创建工作空间: {workspace}")
    
    # 测试会话记忆
    print("\n=== 测试会话记忆 ===")
    session = memory_manager.session.start_session(
        session_id="debug_session_001",
        user_id="dev_team",
        initial_context="数据库性能问题排查"
    )
    print(f"开始会话: {session}")
    
    # 获取整体统计
    print("\n=== 系统统计 ===")
    stats = memory_manager.get_overall_stats()
    print(f"知识记忆: {stats['knowledge']['total_templates']} 个模板")
    print(f"协作记忆: {'✅ 启用' if stats['collaboration_enabled'] else '❌ 禁用'}")
    print(f"会话记忆: {'✅ 启用' if stats['session_enabled'] else '❌ 禁用'}")

if __name__ == "__main__":
    main()
```

运行示例：
```bash
cd /home/ubuntu/agents/STELLA
python memory_architecture_example.py
```

---

## 📞 支持

如有问题，请查看：
1. [Knowledge_base.py](./Knowledge_base.py) - 原始实现参考
2. [memory_manager.py](./memory_manager.py) - 新架构实现
3. [stella_core.py](./stella_core.py) - 主系统集成

**新架构设计目标**: 🎯 更清晰、更可维护、更可扩展的内存管理系统 