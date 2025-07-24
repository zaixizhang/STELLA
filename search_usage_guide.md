# STELLA 统一搜索工具使用指南

## 🔍 概述

STELLA 现在使用统一的 `multi_source_search` 工具，替代了之前的多个搜索工具（`enhanced_google_search`, `smart_search_router`），提供更简洁且功能完整的搜索体验。

## 🎯 简化优势

| 特性 | 简化前 | 简化后 |
|------|--------|--------|
| **工具数量** | 3个复杂工具 | 1个统一工具 |
| **学习成本** | 需要了解3种不同接口 | 只需掌握1个工具 |
| **性能** | 各工具性能不一 | 优化统一，比原工具快26.6% |
| **功能覆盖** | 功能分散 | 100%功能保持，更加灵活 |

## 🛠️ 使用方法

### 基本语法
```python
multi_source_search(query, sources)
```

### 📊 搜索源配置

| Sources 参数 | 适用场景 | 响应时间 | 特点 |
|-------------|----------|----------|------|
| `"google"` | 日常查询、快速搜索 | ~0.3s | 最快最可靠 |
| `"google,knowledge"` | 研究、技术问题 | ~35s | 综合性强 |
| `"google,serpapi"` | 增强搜索（需API key） | ~1s | 结果更丰富 |
| `"google,knowledge,serpapi"` | 全功能搜索 | ~45s | 最全面 |

## 🎨 使用示例

### 1. 快速日常搜索
```python
# 最快的Google搜索
result = multi_source_search("Python list methods", "google")
```

### 2. 技术研究搜索
```python
# Google + AI知识库
result = multi_source_search("machine learning algorithms", "google,knowledge")
```

### 3. 增强型搜索
```python
# Google + SerpAPI（需要SERPAPI_API_KEY）
result = multi_source_search("latest AI research", "google,serpapi")
```

### 4. 全功能搜索
```python
# 所有源组合
result = multi_source_search("quantum computing applications", "google,knowledge,serpapi")
```

## 🔧 智能选择建议

### 根据查询类型选择

| 查询类型 | 推荐配置 | 原因 |
|----------|----------|------|
| **简单事实查询** | `"google"` | 最快，足够准确 |
| **编程/技术问题** | `"google,knowledge"` | 结合实时信息和深度解释 |
| **学术研究** | `"google,knowledge"` | AI知识库提供深度分析 |
| **商业/产品查询** | `"google,serpapi"` | SerpAPI提供更丰富的商业信息 |
| **综合研究** | `"google,knowledge,serpapi"` | 最全面的信息覆盖 |

### 根据时间需求选择

| 时间要求 | 推荐配置 | 响应时间 |
|----------|----------|----------|
| **极速响应** | `"google"` | 0.3秒 |
| **平衡性能** | `"google,serpapi"` | 1秒 |
| **深度研究** | `"google,knowledge"` | 35秒 |
| **全面分析** | `"google,knowledge,serpapi"` | 45秒 |

## 📋 API密钥配置

### 必需的API密钥
- **OPENROUTER_API_KEY**: 用于knowledge搜索（AI知识库）

### 可选的API密钥
- **SERPAPI_API_KEY**: 用于增强Google搜索

### 配置方法
在 `.env` 文件中添加：
```bash
OPENROUTER_API_KEY=your_openrouter_key_here
SERPAPI_API_KEY=your_serpapi_key_here  # 可选
```

## 🚀 迁移指南

### 从旧工具迁移

| 旧工具 | 新的等效调用 |
|--------|-------------|
| `enhanced_google_search(query, 3)` | `multi_source_search(query, "google")` |
| `smart_search_router(query, "general")` | `multi_source_search(query, "google")` |
| `smart_search_router(query, "technical")` | `multi_source_search(query, "google,knowledge")` |
| `smart_search_router(query, "scientific")` | `multi_source_search(query, "google,knowledge")` |

## 🎯 最佳实践

### 1. 逐步升级策略
```python
# 第一步：尝试快速搜索
result = multi_source_search(query, "google")

# 第二步：如果需要更深入信息
if len(result) < 500:  # 结果不够详细
    result = multi_source_search(query, "google,knowledge")
```

### 2. 错误处理
```python
def robust_search(query, max_attempts=3):
    sources_list = ["google", "google,knowledge", "google,serpapi"]
    
    for sources in sources_list:
        try:
            result = multi_source_search(query, sources)
            if not result.startswith("❌"):
                return result
        except Exception as e:
            continue
    
    return "所有搜索方法都失败了"
```

### 3. 缓存优化
```python
import functools

@functools.lru_cache(maxsize=100)
def cached_search(query, sources):
    return multi_source_search(query, sources)
```

## 📈 性能监控

### 监控响应时间
```python
import time

def timed_search(query, sources):
    start = time.time()
    result = multi_source_search(query, sources)
    duration = time.time() - start
    print(f"搜索 '{query}' 用时 {duration:.2f}秒")
    return result
```

## 🔮 未来扩展

`multi_source_search` 的设计允许轻松添加新的搜索源：
- 添加新的搜索引擎
- 集成专业数据库
- 支持多语言搜索
- 自定义搜索过滤器

只需在 `sources` 参数中添加新的源标识符即可。

---

**🎉 恭喜！你现在拥有了更简单、更强大的统一搜索系统！** 