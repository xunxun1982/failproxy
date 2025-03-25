# ====================================================
# 文件名: model_config.py
# 描述: 模型配置文件，定义端点配置和模型组
# 功能: 
#   1. 定义EndpointConfig类用于配置后端API端点
#   2. 配置MODEL_GROUPS字典，包含多个模型组和备用端点
# ====================================================

# === 导入模块 ===
from typing import Dict, List, Optional  # 类型注解
from pydantic import BaseModel  # 数据验证和模型定义
import json  # JSON文件处理
from pathlib import Path  # 路径操作

# === 端点配置模型 ===
# 维护指南：
# 1. 优先使用环境变量管理敏感信息
# 2. 修改配置后需重启API服务生效
# 3. 新增端点时需测试连通性
# 4. 保持各端点参数命名规范统一
class EndpointConfig(BaseModel):
    """定义后端API端点配置模型
    
    字段说明:
        baseurl: API端点的基础URL，如http://example.com/api
        apikey: 访问API所需的密钥
        model: 使用的模型名称
        timeout: 请求超时时间（秒）
        type: 端点类型，如'chutes'或'openrouter'，用于特定处理逻辑
        temperature: 生成文本的随机性参数，值越大随机性越高
        max_tokens: 生成文本的最大令牌数
        stream: 是否使用流式响应
    """
    baseurl: str  # API端点的基础URL (示例: "http://api.example.com/v1")
    apikey: str  # 访问API所需的密钥 (建议通过环境变量注入，示例: "sk-******")
    model: str  # 使用的模型名称
    timeout: float  # 请求超时时间（秒），推荐值范围5-30秒，根据网络质量调整
    type: Optional[str] = None  # 端点类型，如'chutes'或'openrouter'
    temperature: Optional[float] = None  # 生成文本的随机性参数
    max_tokens: Optional[int] = None  # 生成文本的最大令牌数 (示例：32768=32k tokens，适合长文本生成)
    stream: Optional[bool] = None  # 是否使用流式响应

# === 模型组配置 ===

# 配置文件目录路径
CONFIG_DIR = Path(__file__).parent / "config"

# 初始化模型组字典
MODEL_GROUPS: Dict[str, List[EndpointConfig]] = {}

# 动态加载JSON配置文件
for config_file in [
    "think.json",
    "nothink.json",
    "think-stream.json", 
    "think-nostream.json",
    "nothink-stream.json",
    "nothink-nostream.json",
    "search.json"
]:
    try:
        file_path = CONFIG_DIR / config_file
        with open(file_path, encoding='utf-8') as f:
            # 使用模型组名称作为键（去除.json后缀）
            group_name = config_file.rsplit('.', 1)[0]
            MODEL_GROUPS[group_name] = [EndpointConfig(**item) for item in json.load(f)]
    except Exception as e:
            print(f"加载配置文件{config_file}失败: {str(e)}")