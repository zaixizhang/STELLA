# STELLA + Mem0 增强记忆系统集成指南

## 🌟 概述

STELLA现在支持Mem0增强记忆系统，提供比传统知识库更强大的语义记忆管理能力。Mem0集成带来了26%的准确性提升、91%的响应速度提升和90%的token使用率降低。

## 🚀 主要特性

### 🧠 Mem0 增强功能
- **语义记忆搜索**: 基于向量相似性的智能检索
- **用户个性化记忆**: 每用户独立记忆空间
- **会话上下文管理**: 对话连续性保持
- **偏好学习分析**: 自动学习和记住用户偏好
- **智能记忆更新**: 自动去重和记忆优化

### 🔄 向后兼容性
- 完全兼容传统知识库模式
- 如果Mem0不可用，自动回退到传统模式
- 所有现有功能保持不变

## 📦 安装要求

### 基础安装
```bash
pip install mem0ai
```

### 可选依赖 (推荐)
```bash
# 如果使用本地Chroma向量数据库
pip install chromadb

# 如果需要其他向量数据库支持
pip install qdrant-client  # Qdrant
pip install weaviate-client  # Weaviate
```

## 🛠️ 配置选项

### 1. 自托管模式 (推荐用于开发)
```bash
python stella_core.py --use_template --use_mem0
```

### 2. 托管平台模式 (推荐用于生产)
```bash
python stella_core.py --use_template --use_mem0 --mem0_platform --mem0_api_key YOUR_API_KEY
```

### 3. 传统模式 (兼容性)
```bash
python stella_core.py --use_template
```

## 🎯 启动参数详解

| 参数 | 描述 | 必需 |
|------|------|------|
| `--use_template` | 启用知识库功能 | ✅ (使用记忆功能时) |
| `--use_mem0` | 启用Mem0增强记忆 | ❌ |
| `--mem0_platform` | 使用托管平台 | ❌ |
| `--mem0_api_key` | Mem0 API密钥 | ❌ (托管模式需要) |
| `--port` | Gradio界面端口 | ❌ (默认7860) |

## 🧠 新增工具功能

### 基础记忆管理
- `retrieve_similar_templates()` - 检索相似模板 (支持user_id)
- `save_successful_template()` - 保存成功模板 (支持user_id)
- `list_knowledge_base_status()` - 获取记忆统计 (支持user_id)
- `search_templates_by_keyword()` - 关键词搜索 (支持user_id)

### Mem0专用功能
- `get_user_memories()` - 获取用户所有记忆
- `delete_memory_by_id()` - 删除特定记忆
- `update_memory_by_id()` - 更新记忆内容

### 会话管理
- `start_user_session()` - 启动用户会话
- `get_conversation_context()` - 获取对话上下文
- `add_conversation_memory()` - 添加对话记忆
- `get_user_preferences()` - 分析用户偏好
- `search_contextual_memories()` - 上下文记忆搜索

## 💡 使用示例

### 启动增强记忆系统
```bash
# 自托管模式
python stella_core.py --use_template --use_mem0

# 托管平台模式 (需要API密钥)
python stella_core.py --use_template --use_mem0 --mem0_platform --mem0_api_key sk-mem0-xxxxx
```

### 代理使用示例
```python
# Manager agent 会自动使用增强的记忆功能
# 1. 启动用户会话
manager_agent.run("start_user_session(user_id='alice', session_context='生物信息学研究')")

# 2. 检索相似经验
manager_agent.run("retrieve_similar_templates('分析基因表达数据', user_id='alice')")

# 3. 获取用户偏好
manager_agent.run("get_user_preferences(user_id='alice')")

# 4. 保存成功经验
manager_agent.run("save_successful_template('RNA-seq分析', '使用DESeq2方法', '找到了15个差异表达基因', 'bioinformatics', user_id='alice')")
```

## 🏗️ 架构改进

### 类结构
- `KnowledgeBase` - 传统知识库 (保持不变)
- `Mem0EnhancedKnowledgeBase` - 新增Mem0增强类
- 自动回退机制确保稳定性

### 工具权限分离
- **Dev Agent**: 基础工具权限 (查看/加载/刷新)
- **Manager Agent**: 完整权限 + Mem0增强功能
- **新增**: 会话管理和用户个性化功能

### 数据流
```
用户输入 → Manager Agent → Mem0语义搜索 → 个性化响应
                       ↓
                   记忆更新 ← 任务完成 ← Dev Agent
```

## 🔧 配置详解

### 自托管配置
系统自动配置：
- **向量数据库**: Chroma (本地存储)
- **嵌入模型**: text-embedding-3-small
- **LLM**: gpt-4o-mini (通过OpenRouter)
- **存储路径**: `/home/ubuntu/agent_outputs/mem0_db`

### 托管平台配置
- 使用Mem0官方托管服务
- 自动更新和安全性保障
- 企业级合规性 (SOC 2, HIPAA)

## 📊 性能对比

| 指标 | 传统知识库 | Mem0增强 | 改进 |
|------|------------|----------|------|
| 检索准确性 | 基准 | +26% | ⬆️ |
| 响应速度 | 基准 | +91% | ⬆️ |
| Token使用 | 基准 | -90% | ⬇️ |
| 个性化支持 | ❌ | ✅ | 🆕 |
| 会话连续性 | ❌ | ✅ | 🆕 |

## 🐛 故障排除

### 常见问题

1. **Mem0导入失败**
   ```bash
   pip install mem0ai
   ```

2. **向量数据库错误**
   ```bash
   # 删除现有数据库重新初始化
   rm -rf /home/ubuntu/agent_outputs/mem0_db
   ```

3. **API密钥问题**
   - 检查Mem0 API密钥格式
   - 确保网络连接正常
   - 验证账户余额

4. **权限错误**
   ```bash
   # 确保输出目录权限
   chmod 755 /home/ubuntu/agent_outputs/
   ```

### 调试模式
添加环境变量开启详细日志：
```bash
export MEM0_DEBUG=1
python stella_core.py --use_template --use_mem0
```

## 🔮 迁移指南

### 从传统模式迁移
1. 现有数据自动兼容
2. 首次启动时自动转换模板
3. 传统JSON文件保持不变作为备份

### 数据导出
```python
# 导出所有记忆为JSON
memories = global_knowledge_base.get_user_memories("user_id")
with open("backup.json", "w") as f:
    json.dump(memories, f, indent=2)
```

## 📈 最佳实践

### 用户ID管理
- 使用有意义的用户标识符
- 保持用户ID一致性
- 考虑隐私保护需求

### 记忆组织
- 使用描述性的context_type
- 定期清理过时记忆
- 利用偏好学习功能

### 性能优化
- 限制检索结果数量
- 使用适当的相似度阈值
- 定期更新重要记忆

## 🔗 相关资源

- [Mem0官方文档](https://docs.mem0.ai/)
- [STELLA项目主页](./README.md)
- [API参考文档](./API_Reference.md)
- [故障排除指南](./Troubleshooting.md)

## 📝 更新日志

### v1.0.0 - Mem0集成
- ✅ 完整Mem0集成
- ✅ 用户个性化记忆
- ✅ 会话管理功能
- ✅ 向后兼容性
- ✅ 自动回退机制

---

**注意**: 这是STELLA系统的重大升级，建议在生产环境使用前充分测试Mem0功能。 