# Immortal AI API代理服务

一个基于FastAPI实现的API代理服务，用于转发请求到多个后端模型服务，支持自动故障转移和灵活的配置管理。通过错误自动顺序切换机制，实现永远的AI对话不失败，确保服务的高可用性和稳定性。

## 主要特性

- **多端点自动切换**: 支持配置多个备用端点，当主端点失败时自动按顺序切换到备用端点，直到找到可用服务，确保请求永不失败
- **流式响应支持**: 完整支持流式和非流式响应处理
- **灵活的端点类型**: 支持多种类型的端点(chutes, openrouter等)
- **详细的日志记录**: 提供可配置的日志级别，方便调试和问题排查
- **模型组配置**: 通过JSON文件灵活配置不同的模型组和端点参数

## 安装说明

1. 克隆项目到本地
2. 安装依赖包：
```bash
pip install fastapi uvicorn aiohttp pydantic
```

## 配置说明

### 端点配置

在`config`目录下通过JSON文件配置不同的模型组，每个模型组可以包含多个端点配置：

```json
{
  "baseurl": "http://api.example.com/v1",
  "apikey": "your-api-key",
  "model": "model-name",
  "timeout": 30,
  "type": "chutes",
  "temperature": 0.1,
  "max_tokens": 32768,
  "stream": true
}
```

### 日志配置

通过环境变量`LOG_LEVEL`控制日志级别：
- `LOG_LEVEL=true`: 启用详细日志记录
- `LOG_LEVEL=false`: 仅记录错误日志

## 使用示例

### 启动服务

```bash
uvicorn autoapi:app --host 0.0.0.0 --port 8000
```

### API调用示例

```python
import requests

# 文本生成请求
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "你好"}
        ],
        "model": "think",
        "temperature": 0.1,
        "stream": True
    }
)
```

## 项目结构

```
├── autoapi.py          # FastAPI应用主文件
├── model_config.py     # 模型配置管理
├── config/            # 模型组配置目录
│   ├── think.json
│   ├── nothink.json
│   └── ...
└── logs/              # 日志目录
```

## 注意事项

1. 请确保在使用前正确配置API密钥和端点URL
2. 建议通过环境变量管理敏感信息
3. 根据实际需求调整超时时间和其他参数
4. 定期检查日志文件，及时发现和处理异常情况

## 错误自动切换机制

### 工作原理

1. **顺序尝试**: 系统按照配置文件中定义的端点顺序依次尝试，直到找到可用的端点
2. **错误分类**: 系统会精确识别错误类型（网络错误、认证错误、超时等），便于故障排查
3. **自动重试**: 当一个端点失败时，系统会自动尝试下一个可用端点，无需人工干预
4. **完整日志**: 每次切换都会记录详细日志，包括错误类型、详情和解决方案建议

### 错误类型处理

系统能够识别并处理以下类型的错误：

- **网络连接错误**: 当网络不通或目标服务器不可达时
- **认证错误**: 当API密钥无效时
- **权限错误**: 当缺少访问权限时
- **服务器内部错误**: 当后端服务异常时
- **超时错误**: 当请求超过设定的超时时间时

### 配置优化建议

为确保错误自动切换机制的最佳效果，建议：

1. **合理排序**: 将最稳定、响应最快的端点放在配置列表的前面
2. **适当超时**: 根据网络环境和模型复杂度设置合理的超时时间
3. **多样化备份**: 配置不同服务商或不同区域的端点，避免单点故障
4. **定期更新**: 定期检查和更新端点配置，移除长期不可用的端点

### 实现细节

错误自动切换机制在`autoapi.py`文件的`forward_request`函数中实现：

```python
# 核心切换逻辑摘要
for endpoint_idx, endpoint in enumerate(endpoints):
    try:
        # 尝试请求当前端点
        # ...
        return result  # 成功则返回结果
    except Exception as e:
        # 错误分类和日志记录
        # ...
        continue  # 失败则继续尝试下一个端点

# 所有端点都失败时抛出异常
raise HTTPException(502, f"所有端点均失败: {last_error}")
```