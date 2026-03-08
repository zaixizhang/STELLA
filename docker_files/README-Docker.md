# 🐳 STELLA Core Docker 部署指南

## 📋 概述

这个 Docker 配置包含了 STELLA 的核心功能，去除了实验数据和扩展脚本，只保留最核心的 AI 代理功能。

## 🚀 快速开始

### 方法 1: 使用 Docker Compose (推荐)

```bash
# 1. 克隆或下载 STELLA 核心文件
# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，添加你的 API 密钥

# 3. 构建并运行
docker-compose up --build

# 4. 访问 STELLA
# 浏览器打开: http://localhost:7860
```

### 方法 2: 直接使用 Docker

```bash
# 1. 构建镜像
docker build -f Dockerfile.stella-core -t stella-core .

# 2. 运行容器
docker run -d \
  --name stella-core \
  -p 7860:7860 \
  -e OPENROUTER_API_KEY=your_api_key \
  -e SERPAPI_API_KEY=your_serpapi_key \
  -e PAPERQA_API_KEY=your_paperqa_key \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/agent_outputs:/app/agent_outputs \
  stella-core

# 3. 访问 STELLA
# 浏览器打开: http://localhost:7860
```

## 🔧 环境变量配置

创建 `.env` 文件：

```bash
# 必需的 API 密钥
OPENROUTER_API_KEY=your_openrouter_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
PAPERQA_API_KEY=your_paperqa_key_here

# 可选配置
STELLA_HOME=/app
PYTHONPATH=/app
```

## 📦 私有仓库部署

### 构建和推送

```bash
# 1. 修改 build-and-push.sh 中的 REGISTRY 变量
# 2. 运行构建脚本
chmod +x build-and-push.sh
./build-and-push.sh
```

### 从私有仓库拉取

```bash
# 拉取镜像
docker pull your-registry.com/stella-core:latest

# 运行容器
docker run -p 7860:7860 \
  -e OPENROUTER_API_KEY=your_key \
  your-registry.com/stella-core:latest
```

## 🏗️ 镜像内容

### 包含的核心文件：
- ✅ `stella_core.py` - 核心多代理系统
- ✅ `start_stella_web.py` - Web 界面启动器
- ✅ `stella_ui_english.py` - Web UI 实现
- ✅ `memory_manager.py` - 内存管理系统
- ✅ `Knowledge_base.py` - 知识库系统
- ✅ `predefined_tools.py` - 核心系统工具
- ✅ `new_tools/` - 专业工具库
- ✅ `requirements.txt` - Python 依赖

### 不包含的内容：
- ❌ BioML 实验环境
- ❌ 预处理数据
- ❌ 扩展脚本
- ❌ 缓存文件

## 🔍 健康检查

容器包含健康检查功能：

```bash
# 检查容器状态
docker ps

# 查看健康检查日志
docker inspect stella-core | grep -A 10 "Health"
```

## 📊 端口和网络

- **主端口**: 7860 (Gradio Web 界面)
- **内部端口**: 7860
- **协议**: HTTP

## 🛠️ 故障排除

### 常见问题：

1. **API 密钥未配置**
   ```bash
   # 检查环境变量
   docker exec stella-core env | grep API_KEY
   ```

2. **端口冲突**
   ```bash
   # 使用不同端口
   docker run -p 8080:7860 stella-core
   ```

3. **权限问题**
   ```bash
   # 检查文件权限
   docker exec stella-core ls -la /app
   ```

## 📝 日志和监控

```bash
# 查看容器日志
docker logs stella-core

# 实时监控日志
docker logs -f stella-core

# 进入容器调试
docker exec -it stella-core /bin/bash
```

## 🔒 安全建议

1. **使用私有仓库**：不要推送到公共 Docker Hub
2. **API 密钥安全**：使用环境变量，不要硬编码
3. **网络安全**：在生产环境中使用 HTTPS
4. **资源限制**：设置内存和 CPU 限制

## 📞 支持

如有问题，请检查：
1. Docker 版本 >= 20.10
2. 可用的内存 >= 4GB
3. 网络连接正常
4. API 密钥有效




