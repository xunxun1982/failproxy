# ====================================================
# 文件名: autoapi.py
# 描述: FastAPI实现的API代理服务，用于转发请求到多个后端模型服务
# 功能: 
#   1. 支持多个备用端点自动切换
#   2. 支持流式和非流式响应处理
#   3. 支持不同类型的端点(chutes, openrouter等)
#   4. 提供详细的日志记录
# ====================================================

# === 导入模块 ===
from fastapi import FastAPI, HTTPException  # FastAPI框架核心组件
from pydantic import BaseModel, field_validator, ValidationInfo  # 数据验证和模型定义
import asyncio  # 异步支持
import aiohttp  # 异步HTTP客户端
import logging  # 日志记录
import os  # 操作系统接口
from datetime import datetime  # 日期时间处理
from typing import Dict, List, Optional  # 类型注解
import json  # JSON数据处理
import traceback  # 异常追踪和堆栈信息
import uvicorn  # ASGI服务器
import random  # 随机数生成，用于随机化Chrome版本号
from model_config import MODEL_GROUPS  # 导入模型配置

# === 日志配置模块 ===

# 环境变量控制日志级别，可通过环境变量LOG_LEVEL控制是否启用详细日志
# 从环境变量获取LOG_LEVEL，默认为'true'，转换为布尔值
LOG_LEVEL = os.getenv('LOG_LEVEL', 'true').lower() == 'true'
# 强制设置为False，禁用详细日志记录（开发调试时可改为True）
LOG_LEVEL = False

# 创建全局logger实例，用于整个应用的日志记录
logger = logging.getLogger()

def should_log_details():
    """判断是否应该记录详细日志
    
    返回:
        bool: 如果LOG_LEVEL为True则返回True，否则返回False
    
    说明:
        此函数用于代码中判断是否需要记录详细的调试日志，
        通过集中控制可以避免在代码中直接使用LOG_LEVEL变量
    """
    return LOG_LEVEL or False

def setup_logger():
    """初始化日志配置，创建日志目录和文件/控制台处理器
    
    功能:
        1. 创建logs目录（如果不存在）
        2. 配置日志文件名（按日期命名）
        3. 设置日志级别（DEBUG或ERROR，取决于LOG_LEVEL）
        4. 创建文件和控制台处理器
        5. 设置日志格式
        6. 将处理器添加到logger
    """
    # 创建日志目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 按日期生成日志文件名
    log_filename = f"{log_dir}/{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # 配置logger级别 - 如果LOG_LEVEL为True则使用DEBUG级别，否则使用ERROR级别
    logger.setLevel(logging.DEBUG if LOG_LEVEL else logging.ERROR)
    
    # 创建文件handler - 将日志写入文件
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if LOG_LEVEL else logging.ERROR)
    
    # 创建控制台handler - 将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)  # 控制台只显示ERROR级别日志
    
    # 创建日志格式 - 包含时间、日志级别和消息
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers到logger - 确保不重复添加
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

setup_logger()

# === FastAPI应用实例 ===
app = FastAPI()  # 创建FastAPI核心应用实例

# === 请求数据模型 ===
class RequestModel(BaseModel):
    """客户端请求数据模型，支持消息和提示两种输入模式
    
    字段说明:
        messages: 聊天消息列表，格式为[{"role": "user", "content": "消息内容"}]等
        prompt: 文本提示，当messages为空时使用
        max_tokens: 生成文本的最大令牌数，默认4096
        temperature: 生成文本的随机性，值越大随机性越高，默认0.1
        stream: 是否使用流式响应，默认True
        model: 使用的模型名称，必填字段，用于路由到对应的模型组
    """
    # 定义模型字段及其默认值
    messages: Optional[List[Dict[str, str]]] = None  # 聊天消息列表，可选
    prompt: Optional[str] = ""  # 文本提示，可选
    max_tokens: Optional[int] = None  # 生成文本的最大令牌数
    temperature: float = 0.1  # 生成文本的随机性参数
    stream: Optional[bool] = None  # 是否使用流式响应，优先使用入参值，若未指定则使用配置值
    model: str  # 模型名称，必填字段，用于路由
    
    @field_validator('prompt', mode='before')
    @classmethod
    def set_prompt_from_messages(cls, v, info: ValidationInfo):
        """从messages字段自动生成prompt字段的验证器
        
        参数:
            v: 当前prompt值
            info: 包含所有字段数据的验证信息对象
            
        返回:
            str: 处理后的prompt值
            
        说明:
            如果prompt为空且messages不为空，则自动将messages转换为prompt格式
        """
        values = info.data
        if not v and values.get('messages'):
            # 将消息列表转换为文本格式: "role: content\nrole: content..."
            return '\n'.join(f"{msg['role']}: {msg['content']}" for msg in values['messages'])
        return v or ''  # 返回原始prompt或空字符串
    
    class Config:
        """Pydantic模型配置"""
        extra = "allow"  # 允许并保留所有未在模型中声明的额外参数

    def model_dump(self, **kwargs):
        """自定义模型序列化方法
        
        参数:
            **kwargs: 传递给父类model_dump方法的参数
            
        返回:
            Dict: 序列化后的模型数据
            
        说明:
            1. 当使用messages时，排除prompt字段
            2. 自动生成日志中的prompt信息（但不包含在输出中）
            3. 合并额外参数到输出中
        """
        # 当使用messages时排除prompt字段
        exclude_fields = {'prompt'} if self.messages else set()
        # 调用父类方法获取基础数据
        base_data = super().model_dump(exclude=exclude_fields, include={'messages','temperature','stream','model'}, exclude_none=True, **kwargs)
        
        # 当使用messages时自动生成prompt但不包含在最终参数中（仅用于日志记录）
        if base_data.get('messages'):
            prompt = '\n'.join(
                [f"{msg['role']}: {msg['content']}" 
                 for msg in base_data['messages']]
            )
            if should_log_details():
                logger.debug(f"生成的prompt: {prompt}")
        
        # 合并额外参数到输出数据中
        if self.__pydantic_extra__:
            base_data.update(self.__pydantic_extra__)
        return base_data



# === 核心请求转发逻辑 ===
async def forward_request(model_group: str, params: Dict, path: str):
    """转发请求到指定模型组的端点，支持自动故障转移
    
    参数:
        model_group: 模型组名称，用于从MODEL_GROUPS中获取对应的端点列表
        params: 请求参数字典，包含模型名称、温度等参数
        path: API路径，如"chat/completions"
        
    返回:
        Dict/List: 从成功的端点获取的响应数据
        
    异常:
        HTTPException: 当模型组不存在或所有端点均失败时抛出
        
    说明:
        1. 按顺序尝试模型组中的每个端点，直到成功或全部失败
        2. 对每个端点使用独立的参数副本，避免参数污染
        3. 支持流式和非流式响应处理
        4. 详细记录请求和响应过程（当LOG_LEVEL为True时）
        
    渠道切换机制:
        1. 当前端点请求失败时（网络错误、认证错误、超时等），自动尝试下一个端点
        2. 切换触发点在异常处理块的continue语句，确保循环继续执行
        3. 所有端点都失败时才会抛出异常，返回502错误
        4. 错误类型精确分类，便于排查问题和优化配置
    """
    # 从MODEL_GROUPS获取指定模型组的端点列表
    endpoints = MODEL_GROUPS.get(model_group)
    if not endpoints:
        # 如果模型组不存在，抛出400错误
        raise HTTPException(400, f"Invalid model group: {model_group}")

    # === 记录原始请求信息 ===
    original_model = params.get('model', 'unknown')  # 获取原始请求中的模型名称
    if should_log_details():
        logger.debug(f"原始请求模型: {original_model}, 使用模型组: {model_group}")

    # 用于存储最后一次错误信息
    last_error = None
    for endpoint_idx, endpoint in enumerate(endpoints):
        # === 修复点1：使用独立参数副本 ===
        endpoint_params = params.copy()
        endpoint_params['model'] = endpoint.model
        
        # === 参数合并逻辑 ===
        # 1. 基础参数：使用请求参数的副本
        final_params = endpoint_params.copy()
        
        # 2. 处理stream参数的优先级
        # - 如果入参中指定了stream，优先使用入参的值
        # - 如果入参未指定stream，但配置中有定义，则使用配置值
        # - 如果两者都未指定，则不传stream参数
        if 'stream' in endpoint_params:
            # 入参中指定了stream，保留入参的值
            pass
        elif hasattr(endpoint, 'stream') and endpoint.stream is not None:
            # 入参未指定，使用配置中的值
            final_params['stream'] = endpoint.stream
        else:
            # 入参和配置都未指定，移除stream参数
            final_params.pop('stream', None)
            
        # 3. 合并其他配置参数（temperature、max_tokens等）
        for param in ['max_tokens', 'temperature']:
            if hasattr(endpoint, param) and getattr(endpoint, param) is not None:
                final_params[param] = getattr(endpoint, param)
        # 移除未指定的参数
        final_params = {k: v for k, v in final_params.items() if v is not None}
        endpoint_params = final_params
            
        # 为openrouter类型的端点添加include_reasoning参数
        if endpoint.type == 'openrouter':
            if 'extra_body' not in endpoint_params:
                endpoint_params['extra_body'] = {}
            endpoint_params['extra_body']['include_reasoning'] = True
            if should_log_details():
                logger.debug(f"为openrouter端点添加include_reasoning参数: {endpoint_params['extra_body']}")
        
        # 记录尝试的端点信息
        if should_log_details():
            logger.debug(f"尝试连接端点 {endpoint_idx+1}/{len(endpoints)}: {endpoint.baseurl}")
        try:
            # 构造请求URL
            url = f"{endpoint.baseurl.rstrip('/')}/{path.lstrip('/')}"
            if should_log_details():
                logger.debug(f"构造请求地址: {url}")
            
            # 设置超时时间
            timeout = aiohttp.ClientTimeout(total=endpoint.timeout)
            if should_log_details():
                logger.debug(f"设置超时时间: {endpoint.timeout}秒")
            
            # 创建会话
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if should_log_details():
                    logger.debug("成功创建会话")
                
                # 构造请求头
                # 生成随机Chrome版本号
                chrome_version = f"{random.randint(70, 134)}.0.{random.randint(0, 9)}.{random.randint(0, 9)}"
                headers = {
                    "Authorization": f"Bearer {endpoint.apikey}",
                    "Content-Type": "application/json",
                    "User-Agent": f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36"  # 使用Linux版本的随机Chrome版本User-Agent
                }
                if should_log_details():
                    logger.debug(f"使用请求头: {headers}")
                
                # 记录更新后的请求参数
                # === 修复点2：更新请求参数使用副本 ===
                if should_log_details():
                    logger.debug(f"发送请求至: {url}")
                updated_params = endpoint_params.copy()  # 改用端点参数副本
                if should_log_details():
                    logger.debug(f"更新后的请求参数: {json.dumps(updated_params, ensure_ascii=False)}")
                
                # 发起POST请求并处理流式响应
                async with session.post(url, json=updated_params, headers=headers) as response:
                    if should_log_details():
                        logger.debug(f"收到响应状态码: {response.status}")
                    response.raise_for_status()  # 检查响应状态
                    
                    # === 响应内容类型检查 ===
                    # 获取响应的Content-Type头
                    content_type = response.headers.get('Content-Type', '')
                    
                    # === 处理流式响应（SSE格式）===
                    if 'text/event-stream' in content_type:  # 检查是否为流式响应
                        # 初始化结果列表，用于存储所有流式数据
                        result = []
                        # 异步迭代响应内容的每一行
                        async for line in response.content:
                            # 解码并清理每一行数据
                            decoded_line = line.decode('utf-8').strip()
                            # 检查是否为SSE数据行（以"data:"开头）
                            if decoded_line.startswith("data:"):
                                # 提取数据内容部分
                                data_content = decoded_line[5:].strip()
                                # 检查是否为流式传输结束标记
                                if data_content == "[DONE]":  
                                    if should_log_details():
                                        logger.debug("收到流式结束标记 [DONE]")
                                    continue  # 跳过结束标记
                                    
                                # 尝试解析JSON数据
                                try:
                                    # 将数据字符串解析为JSON对象
                                    data = json.loads(data_content)
                                    # 处理reasoning字段（如果存在）
                                    update_reasoning(data, endpoint.type)
                                    # 将处理后的数据添加到结果列表
                                    result.append(data)
                                    if should_log_details():
                                        logger.debug(f"收到流式数据: {data}")
                                except json.JSONDecodeError:
                                    # 记录无法解析的数据行
                                    logger.warning(f"无法解析的流式数据: {decoded_line}")
                                    
                        # 流式传输完成，记录成功信息
                        if should_log_details():
                            logger.debug(f"成功从 {endpoint.baseurl} 获取流式结果")
                        # 返回完整的流式数据结果列表
                        return result
                        
                    # === 处理普通JSON响应 ===
                    else:
                        # 解析响应体为JSON对象
                        result = await response.json()
                        # 处理reasoning字段（如果存在）
                        update_reasoning(result, endpoint.type)
                        if should_log_details():
                            logger.debug(f"收到响应数据: {result}")
                        
                        # 处理响应数据中的reasoning字段
                        if should_log_details():
                            logger.debug(f"开始处理出参数据: {result}")
                        # 再次调用处理函数（可能是冗余的，但保留原代码逻辑）
                        update_reasoning(result)  
                        if should_log_details():
                            logger.debug("已完成reasoning字段转换")
                            
                        # 记录成功信息并返回结果
                        if should_log_details():
                            logger.debug(f"成功从 {endpoint.baseurl} 获取JSON结果")
                        return result

        # === 错误处理部分 ===
        except Exception as e:  # 捕获请求过程中的所有异常
            # 错误分类逻辑 - 根据异常类型进行分类，便于故障排查和日志分析
            error_type = "unknown_error"  # 默认为未知错误类型
            error_detail = str(e)  # 获取异常的字符串表示
            solution = "请检查网络连接和服务状态"  # 默认解决方案

            # 识别具体异常类型 - 根据异常类的类型进行精确分类
            if isinstance(e, aiohttp.ClientConnectorError):
                # 网络连接错误 - 通常是网络不通或目标服务器不可达
                error_type = "network_error"
                solution = "请检查网络连接是否正常，目标地址是否可达"
            elif isinstance(e, aiohttp.ClientResponseError):
                # HTTP响应错误 - 服务器返回了错误状态码
                if e.status == 401:
                    # 认证错误 - API密钥无效
                    error_type = "authentication_error"
                    solution = "API密钥无效，请检查端点配置"
                elif e.status == 403:
                    # 权限错误 - 无权访问资源
                    error_type = "permission_denied"
                    solution = "缺少访问权限，请检查API密钥权限"
                elif e.status >= 500:
                    # 服务器内部错误
                    error_type = "server_error"
                    solution = "后端服务异常，请稍后重试"
            elif isinstance(e, asyncio.TimeoutError):
                # 超时错误 - 请求超过了设定的超时时间
                error_type = "timeout"
                solution = f"请求超时（{endpoint.timeout}秒），请尝试增加超时时间"

            # 构造错误信息 - 格式化错误消息，包含错误类型和详情
            error_msg = f"[{error_type}] 端点 {endpoint.baseurl} 请求失败: {error_detail}"
            
            # 增强错误日志输出 - 添加更多上下文信息便于排查
            endpoint_info = f"端点配置: baseurl={endpoint.baseurl}, apikey={endpoint.apikey}, model={endpoint.model}, timeout={endpoint.timeout}"
            params_info = f"请求参数: {json.dumps(updated_params, ensure_ascii=False)}"
            
            # 记录带分类的错误日志 - 输出到日志系统
            logger.error(f"{error_msg}\n解决方案: {solution}\n{endpoint_info}\n{params_info}")
            
            # 保存带分类的错误信息 - 用于在所有端点失败时提供详细错误信息
            last_error = f"{error_type}: {error_detail}\n解决方案: {solution}\n{endpoint_info}\n{params_info}"
            
            # 记录详细堆栈信息 - 便于开发人员排查问题
            logger.debug(f"详细错误追踪:\n{traceback.format_exc()}")
            
            # 继续尝试下一个端点 - 这是渠道切换的关键触发点
            # 当前端点失败后，循环会继续执行，尝试下一个备用端点
            # 这实现了自动故障转移功能，提高了系统的可靠性
            continue

    # 所有端点都失败时抛出异常，返回502错误（Bad Gateway）
    raise HTTPException(502, f"所有端点均失败: {last_error}")

# === 响应数据处理 ===
def update_reasoning(obj, endpoint_type=None):
    """
    递归处理响应数据中的reasoning字段
    
    参数:
        obj: 要处理的对象（字典、列表或其他值）
        endpoint_type: 端点类型，用于确定处理逻辑
        
    功能:
        1. 递归遍历响应数据的所有层级
        2. 对于openrouter类型的端点，将'reasoning'字段重命名为'reasoning_content'
        3. 保留原始数据结构，只修改字段名
        
    说明:
        此函数用于统一不同端点类型的响应格式，特别是处理openrouter端点的reasoning字段
        
    注意事项:
        1. 在流式和非流式响应处理中都会调用此函数
        2. 确保在渠道切换后，不同端点类型的响应格式保持一致
        3. 字段重命名而非删除，保留原始信息的完整性
        4. 递归处理确保嵌套结构中的字段也能被正确处理
    """
    # 处理字典类型对象
    if isinstance(obj, dict):
        # 对于openrouter类型的端点，处理reasoning字段
        if endpoint_type == 'openrouter' and "reasoning" in obj:
            # 将reasoning字段重命名为reasoning_content
            obj["reasoning_content"] = obj.pop("reasoning")
            if should_log_details():
                logger.debug(f"已将 'reasoning' 转换为 'reasoning_content': {obj}")
        
        # 递归处理字典中的所有值
        for key, value in obj.items():
            if should_log_details():
                logger.debug(f"递归处理字典键: {key}")
            # 特别关注delta字段，它通常包含流式响应的内容
            if key == "delta" and should_log_details():
                logger.debug("检测到 delta 字段，深入处理")
            # 递归处理每个值
            update_reasoning(value, endpoint_type)
    
    # 处理列表类型对象
    elif isinstance(obj, list):
        # 递归处理列表中的所有项
        for index, item in enumerate(obj):
            if should_log_details():
                logger.debug(f"递归处理列表索引: {index}")
            # 递归处理每个项
            update_reasoning(item, endpoint_type)

# === API路由 ===
@app.post("/v1/chat/completions")
async def chat_completions(request: RequestModel):
    """聊天补全主接口，根据请求的model参数动态选择模型组
    
    参数:
        request: 客户端请求数据，使用RequestModel进行验证和处理
        
    返回:
        Dict/List: 从模型服务获取的响应数据
        
    异常:
        HTTPException: 当模型组不存在、请求参数无效或服务器内部错误时抛出
        
    说明:
        1. 此接口兼容OpenAI API格式，支持流式和非流式响应
        2. 根据请求中的model字段选择对应的模型组
        3. 通过forward_request函数转发请求到后端服务
    """
    # 将请求模型转换为字典格式
    params = request.model_dump()
    try:
        # 从请求参数中获取model字段作为模型组名称，默认使用'list1'
        model_group = params.get('model', 'think-stream')  
        
        # 验证模型组是否存在
        if model_group not in MODEL_GROUPS:
            # 如果模型组不存在，返回400错误
            raise HTTPException(400, f"无效模型名称: {model_group}")
            
        # 转发请求到指定的模型组，使用chat/completions路径
        return await forward_request(model_group, params, "chat/completions")
    except HTTPException as he:
        # 直接传递HTTP异常，保留原始状态码和错误信息
        raise he
    except Exception as e:
        # 捕获所有其他异常，记录错误并返回500错误
        logger.error(f"未捕获异常: {str(e)}")
        raise HTTPException(500, "服务器内部错误")

# === 服务启动入口 ===
if __name__ == "__main__":
    """使用UVicorn启动FastAPI服务
    
    配置:
        host: 监听所有网络接口(0.0.0.0)
        port: 使用8000端口
        
    说明:
        此代码块仅在直接运行autoapi.py文件时执行
        可以通过命令行参数覆盖这些默认配置
    """
    # 启动ASGI服务器，加载FastAPI应用
    uvicorn.run(app, host="0.0.0.0", port=8000)
