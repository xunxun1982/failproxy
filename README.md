# FailProxy - 永不失败的AI模型代理服务

FailProxy 是一个基于 FastAPI 实现的高可用 AI 模型代理服务，专为解决 AI 模型请求失败问题而设计。核心机制是自动错误切换和故障转移，即使多个模型端点依次失败，系统仍然可以找到可用的服务继续响应，确保请求"永不失败"。

## 核心特性

- **自动故障转移** - 当主端点无法响应时，系统会自动按配置顺序尝试备用端点，直到找到可用服务
- **多端点管理** - 支持同时配置多种模型和多个服务提供商，灵活应对不同场景需求
- **流式响应支持** - 完整支持流式和非流式响应处理，兼容各类前端应用需求
- **多模型类型支持** - 内置支持多种端点类型(OpenRouter, Chutes等)，易于扩展新的模型服务
- **智能请求修复** - 自动修复格式不规范的JSON请求，提高兼容性
- **细粒度日志** - 可配置的详细日志系统，便于排查问题和优化性能

## 项目结构

```
FailProxy/
├── autoapi.py          # 主程序文件，包含FastAPI应用和代理逻辑
├── model_config.py     # 模型配置管理，定义端点参数和加载配置
├── config/             # 各类模型组配置目录
│   ├── think.json      # 带推理能力的模型组
│   ├── nothink.json    # 不带推理能力的模型组
│   ├── think-stream.json # 带推理+流式的模型组
│   └── ...             # 其他配置文件
└── logs/               # 日志文件存储目录（自动创建）
```

## 安装与部署

### 环境要求

- Python 3.8+
- FastAPI
- Uvicorn
- Aiohttp
- Pydantic

### 安装步骤

1. 克隆仓库到本地
```bash
git clone https://github.com/yourusername/failproxy.git
cd failproxy
```

2. 安装依赖
```bash
pip install fastapi uvicorn aiohttp pydantic
```

3. 配置模型端点
编辑`config`目录下的JSON文件，添加你的API端点配置。

4. 启动服务
```bash
uvicorn autoapi:app --host 0.0.0.0 --port 8000
```

## 配置详解

### 端点配置文件格式

在`config`目录下创建JSON文件，每个文件包含一组按优先级排列的端点配置：

```json
[
  {
    "name": "主端点名称",
    "baseurl": "http://api.example.com/v1",
    "apikey": "your-api-key",
    "model": "model-name",
    "timeout": 30,
    "type": "endpoint-type",
    "temperature": 0.1,
    "max_tokens": 4096,
    "stream": true
  },
  {
    "name": "备用端点1",
    "baseurl": "http://backup1.example.com/v1",
    "apikey": "backup1-api-key",
    "model": "backup-model1",
    "timeout": 20,
    "type": "endpoint-type",
    "temperature": 0.2,
    "stream": true
  }
  // 可添加更多备用端点...
]
```

### 端点配置参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | string | 否 | 端点名称，用于日志和调试 |
| baseurl | string | 是 | API端点的基础URL |
| apikey | string | 是 | 访问API所需的密钥 |
| model | string | 是 | 使用的模型名称 |
| timeout | number | 是 | 请求超时时间（秒） |
| type | string | 否 | 端点类型，如'chutes'、'openrouter'等 |
| temperature | number | 否 | 生成文本的随机性参数(0-1) |
| max_tokens | number | 否 | 生成文本的最大令牌数 |
| stream | boolean | 否 | 是否使用流式响应 |

### 模型组命名规则

配置文件的命名遵循以下规则，用于区分不同能力和响应方式的模型组：

- `think-*.json` - 带有推理能力的模型组
- `nothink-*.json` - 不带推理能力的模型组
- `*-stream.json` - 使用流式响应的模型组
- `*-nostream.json` - 使用非流式响应的模型组

## API使用说明

### 基本请求格式

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "think-stream",
           "messages": [
             {"role": "user", "content": "你好，介绍一下自己"}
           ],
           "temperature": 0.1,
           "stream": true
         }'
```

### 请求参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | 是 | 模型组名称，对应config目录下的配置文件名(不含.json) |
| messages | array | 是 | 聊天消息数组，格式为[{"role": "user", "content": "消息内容"}] |
| prompt | string | 否 | 文本提示，当messages为空时使用 |
| temperature | number | 否 | 生成文本的随机性参数 |
| max_tokens | number | 否 | 生成文本的最大令牌数 |
| stream | boolean | 否 | 是否使用流式响应 |

## 错误自动切换机制

FailProxy 的核心价值在于错误自动切换机制，确保即使多个端点失败，请求也能成功完成：

1. **顺序尝试** - 系统会按照配置文件中端点的顺序依次尝试，直到找到可用的端点
2. **错误智能分类** - 系统自动识别不同类型的错误（网络错误、认证错误、超时等）
3. **透明切换** - 对客户端完全透明，客户端无需关心底层使用了哪个端点
4. **全程日志** - 详细记录每次切换过程，便于问题排查和优化配置

### 常见错误类型及处理

系统能够识别并处理以下类型的错误：

- **网络连接错误** - 当网络不通或目标服务器不可达时
- **认证错误** - 当API密钥无效时
- **权限错误** - 当缺少访问权限时
- **服务器内部错误** - 当后端服务异常时
- **超时错误** - 当请求超过设定的超时时间时

## 日志配置

通过环境变量`LOG_LEVEL`控制日志记录级别：

```bash
# 启用详细日志（调试模式）
LOG_LEVEL=true uvicorn autoapi:app

# 仅记录错误日志（生产模式）
LOG_LEVEL=false uvicorn autoapi:app
```

## 性能优化建议

1. **端点优先级排序** - 将响应速度快、稳定性高的端点放在配置列表前面
2. **合理设置超时** - 根据模型复杂度和实际网络环境调整超时时间
3. **多样化备份** - 使用不同服务商或不同地区的端点，避免单点故障
4. **定期更新配置** - 定期检查和更新端点状态，移除长期不可用的端点

## 故障排查

### 常见问题

1. **所有端点均失败** - 检查网络连接和API密钥是否有效
2. **响应速度慢** - 检查主端点是否可用，可能系统正在尝试多个备用端点
3. **流式响应中断** - 检查客户端网络稳定性和服务器超时设置

### 排查工具

- 查看`logs`目录下的日志文件，按日期命名
- 临时启用详细日志模式(`LOG_LEVEL=true`)获取更多信息

## 项目维护与贡献

欢迎提交Issues和Pull Requests，共同完善项目功能。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。