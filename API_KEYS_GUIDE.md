# STELLA API Keys 配置指南

## 🔒 安全的API Key管理

STELLA现在使用环境变量来管理API keys，不再在代码中硬编码敏感信息。

## 📁 文件说明

- `.env.example` - API keys模板文件，包含所有可用的配置选项
- `.env` - 实际的API keys文件（已自动添加到.gitignore）
- `.gitignore` - 防止API keys被意外提交到git

## 🚀 快速开始

### 1. 检查现有配置
当前已经为您配置了OpenRouter API key，可以立即使用：

```bash
python stella_core.py
```

### 2. 添加额外的API keys（可选）
编辑 `.env` 文件添加其他服务的API keys：

```bash
# 必需的 API Keys
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key  # ✅ 已配置

# 可选的 API Keys
MEM0_API_KEY=your-mem0-key                        # 增强记忆功能
GEMINI_API_KEY=your-gemini-key                    # Google Gemini模型
TAVILY_API_KEY=your-tavily-key                    # 增强搜索功能
PHOENIX_API_KEY=your-phoenix-key                  # 监控和观察功能
```

## 🛡️ 安全最佳实践

1. **永远不要** 在代码中硬编码API keys
2. **永远不要** 将 `.env` 文件提交到git仓库
3. **定期轮换** API keys
4. **使用最小权限** 原则申请API keys

## 🔧 故障排除

### 缺少API Key错误
如果看到类似错误：
```
❌ Missing required API key: OPENROUTER_API_KEY
```

解决方法：
1. 确保 `.env` 文件存在
2. 检查 `.env` 文件中是否有对应的API key
3. 确保API key格式正确（不含额外空格）

### API Key无效错误
如果API调用失败：
1. 检查API key是否正确复制
2. 确认API key权限和额度
3. 查看API服务商的使用文档

## 📚 获取API Keys

- **OpenRouter**: https://openrouter.ai/
- **Mem0**: https://mem0.ai/
- **Google Gemini**: https://cloud.google.com/
- **Tavily**: https://tavily.com/
- **Phoenix**: https://phoenix.arize.com/

## ✅ 验证配置

运行以下命令验证API keys配置：

```bash
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('OpenRouter:', '✅' if os.getenv('OPENROUTER_API_KEY') else '❌')
print('Mem0:', '✅' if os.getenv('MEM0_API_KEY') else '⚠️')
"
```

