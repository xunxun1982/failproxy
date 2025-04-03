# ====================================================
# 文件名: model_config.py
# 描述: FailProxy模型配置文件，定义端点配置和模型组
# 功能: 
#   1. 定义EndpointConfig类用于配置后端API端点
#   2. 配置MODEL_GROUPS字典，包含多个模型组和备用端点
#   3. 动态加载不同配置文件，支持自动故障转移机制
# ====================================================

# === 导入模块 ===
from typing import Dict, List, Optional  # 类型注解，提高代码可读性和IDE提示支持
from pydantic import BaseModel  # 数据验证和模型定义，用于端点配置的参数检查
import json  # JSON文件处理，用于读取配置文件
from pathlib import Path  # 路径操作，提供跨平台的文件路径处理

# === 端点配置模型 ===
# 维护指南：
# 1. 优先使用环境变量管理敏感信息
# 2. 修改配置后需重启API服务生效
# 3. 新增端点时需测试连通性
# 4. 保持各端点参数命名规范统一
class EndpointConfig(BaseModel):
    """定义后端API端点配置模型
    
    字段说明:
        name: 端点的名称，用于日志和调试
        baseurl: API端点的基础URL，如http://example.com/api
        apikey: 访问API所需的密钥
        model: 使用的模型名称
        timeout: 请求超时时间（秒）
        type: 端点类型，如'chutes'或'openrouter'，用于特定处理逻辑
        temperature: 生成文本的随机性参数，值越大随机性越高
        max_tokens: 生成文本的最大令牌数
        stream: 是否使用流式响应
    """
    name: Optional[str] = None  # 端点名称，用于日志和调试识别
    baseurl: str  # API端点的基础URL (必填项，例如: "http://api.example.com/v1")
    apikey: str  # 访问API所需的密钥 (必填项，建议通过环境变量注入，而非硬编码)
    model: str  # 使用的模型名称 (必填项，对应目标服务支持的模型名称)
    timeout: float  # 请求超时时间（秒），推荐值范围5-30秒，根据网络质量和模型响应时间调整
    type: Optional[str] = None  # 端点类型，如'chutes'或'openrouter'，用于特定处理逻辑
    temperature: Optional[float] = None  # 生成文本的随机性参数，0-1之间，值越大随机性越高
    max_tokens: Optional[int] = None  # 生成文本的最大令牌数 (示例：32768=32k tokens，适合长文本生成)
    stream: Optional[bool] = None  # 是否使用流式响应，流式可提供更好的用户体验，但需要特殊处理

# === 模型组配置 ===

# 配置文件目录路径，使用相对路径，便于部署
CONFIG_DIR = Path(__file__).parent / "config"

# 初始化模型组字典，用于存储不同类型的模型配置
MODEL_GROUPS: Dict[str, List[EndpointConfig]] = {}

# 动态加载JSON配置文件，支持多种配置组合
# 命名规则:
# - think/nothink: 是否带推理能力
# - stream/nostream: 是否使用流式响应
for config_file in [
    "think.json",         # 带推理能力
    "nothink.json",       # 不带推理能力
    "think-stream.json",  # 带推理能力+流式响应
    "think-nostream.json", # 带推理能力+非流式响应
    "nothink-stream.json", # 不带推理能力+流式响应
    "nothink-nostream.json", # 不带推理能力+非流式响应
    "search.json"         # 搜索专用模型组
]:
    try:
        # 构建完整的文件路径
        file_path = CONFIG_DIR / config_file
        # 打开并读取JSON配置文件
        with open(file_path, encoding='utf-8') as f:
            # 使用模型组名称作为键（去除.json后缀）
            group_name = config_file.rsplit('.', 1)[0]
            # 将JSON转换为EndpointConfig对象列表
            MODEL_GROUPS[group_name] = [EndpointConfig(**item) for item in json.load(f)]
    except Exception as e:
            # 记录配置文件加载失败的错误
            print(f"加载配置文件{config_file}失败: {str(e)}")