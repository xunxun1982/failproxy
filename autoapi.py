# ====================================================
# 文件名: autoapi.py
# 描述: FailProxy - 基于FastAPI实现的AI模型代理服务，提供自动故障转移和请求转发功能
# 功能: 
#   1. 多端点自动切换: 当一个端点失败时自动尝试下一个端点，确保请求永不失败
#   2. 流式响应处理: 完整支持流式和非流式响应，适配不同场景需求
#   3. 多类型端点适配: 支持多种类型的AI接口(OpenRouter, Chutes等)
#   4. 详细日志记录: 提供可配置的日志级别，便于调试和问题诊断
#   5. 请求修复: 自动修复格式不正确的JSON请求，提高兼容性
# ====================================================

# === 导入模块 ===
from fastapi import FastAPI, HTTPException, Request  # FastAPI核心组件和请求处理
from pydantic import BaseModel, field_validator, ValidationInfo  # 数据验证和模型定义
import aiohttp  # 异步HTTP客户端，用于向后端服务发送请求
import logging  # 日志记录，用于跟踪请求和响应
import os  # 操作系统接口，用于环境变量和文件操作
from datetime import datetime  # 日期时间处理，用于日志文件命名
from typing import Dict, List, Optional  # 类型注解，提高代码可读性
import json  # JSON数据处理，用于请求和响应的序列化/反序列化
import traceback  # 异常追踪和堆栈信息，用于详细错误日志
import uvicorn  # ASGI服务器，用于运行FastAPI应用
import random  # 随机数生成，用于随机化Chrome版本号避免被屏蔽
from model_config import MODEL_GROUPS  # 导入模型配置，包含所有端点信息
import re  # 正则表达式支持，用于处理和修复JSON内容
import asyncio  # 异步IO支持，用于处理异步请求和超时异常

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
        通过集中控制可以避免在代码中直接使用LOG_LEVEL变量，
        便于未来扩展更复杂的日志级别控制逻辑
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

# === 添加请求预处理中间件 ===
@app.middleware("http")
async def fix_malformed_json(request: Request, call_next):
    """修复格式不正确的JSON请求
    
    参数:
        request: 客户端请求对象
        call_next: 下一个处理函数
        
    返回:
        Response: 处理后的响应对象
        
    说明:
        1. 读取请求体并尝试修复常见的JSON格式错误
        2. 特别处理content字段中不完整的JSON数组和特殊字符
        3. 修复后替换原始请求体并继续处理
        4. 支持多种常见错误修复，提高服务的容错能力
        5. 为AI聊天应用提供更好的兼容性支持
    """
    # 只处理POST请求和JSON内容类型
    if request.method == "POST" and "application/json" in request.headers.get("content-type", ""):
        # 读取原始请求体
        body = await request.body()
        body_str = body.decode()
        
        try:
            # 尝试解析JSON
            json.loads(body_str)
        except json.JSONDecodeError:
            # JSON解析失败，尝试修复
            logger.warning(f"检测到格式不正确的JSON: {body_str[:100]}...")
            
            # 修复常见的JSON格式错误
            fixed_body = body_str
            
            # 1. 处理content字段中的特殊字符和不完整结构
            try:
                # 使用正则表达式查找content字段
                content_pattern = r'"content"\s*:\s*"((?:\\"|[^"\u4e00-\u9fa5])*?[\u4e00-\u9fa5]+(?:\\"|[^"])*?)"(?=\s*[,}])'
                
                def fix_content_match(match):
                    content = match.group(1)
                    # 检查是否包含未转义的特殊字符
                    if any(char in content for char in '{}[]"'):  # 添加双引号到检查列表中
                        # 对特殊字符进行转义
                        escaped_content = content.replace('\\', '\\\\')  # 先处理已有的反斜杠
                        escaped_content = escaped_content.replace('"', '\\"')  # 处理双引号
                        # 使用正则表达式统一转义未转义的特殊字符
                        escaped_content = re.sub(r'(?<!\\)([\[\]"{}])', r'\\\1', escaped_content)
                        # 处理中文字符后的特殊符号
                        escaped_content = re.sub(r'([\u4e00-\u9fa5])([\[\]"{}])', r'\1\\\2', escaped_content)
                        logger.info("已转义content中的特殊字符: {} -> {}".format(content[:50], escaped_content[:50]))
                        return f'"content":"{escaped_content}"'
                    return match.group(0)
                
                # 应用修复
                fixed_body = re.sub(content_pattern, fix_content_match, fixed_body)
                
                # 2. 修复不完整的JSON数组 - 在content字段中
                array_pattern = r'"content"\s*:\s*"(\[.*?)(?<!\\)"'
                matches = re.findall(array_pattern, fixed_body)
                
                for match in matches:
                    if match.count('[') > match.count(']'):
                        # 数组开始但没有结束，添加结束括号和引号
                        fixed_match = match + ']"'
                        # 转义正则表达式中的特殊字符
                        escaped_match = re.escape(match)
                        # 替换原始字符串
                        fixed_body = re.sub(f'"{escaped_match}"', f'"{fixed_match}"', fixed_body)
                        logger.info(f"已修复不完整的JSON数组: {match} -> {fixed_match}")
                        
                # 3. 检查并修复缺失的引号和括号
                # 计算各种括号的数量
                open_braces = fixed_body.count('{')
                close_braces = fixed_body.count('}')
                open_brackets = fixed_body.count('[')
                close_brackets = fixed_body.count(']')
                
                # 修复缺失的括号
                if open_braces > close_braces:
                    fixed_body += '}' * (open_braces - close_braces)
                    logger.info(f"已添加缺失的右花括号: {open_braces - close_braces}个")
                if open_brackets > close_brackets:
                    fixed_body += ']' * (open_brackets - close_brackets)
                    logger.info(f"已添加缺失的右方括号: {open_brackets - close_brackets}个")
                
                # 4. 检查并修复JSON对象中的格式问题
                # 修复缺少逗号的情况
                fixed_body = re.sub(r'"\s*}\s*"', '","', fixed_body)
                fixed_body = re.sub(r'"\s*]\s*"', '","', fixed_body)
                
            except Exception as e:
                logger.error(f"修复JSON时发生错误: {str(e)}")
            
            # 尝试验证修复后的JSON
            try:
                json.loads(fixed_body)
                logger.info("JSON修复成功")
                
                # 创建新的请求体
                request._body = fixed_body.encode()
            except json.JSONDecodeError as e:
                logger.error(f"JSON修复失败: {str(e)}")
    
    # 继续处理请求
    response = await call_next(request)
    return response

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
    stream: Optional[bool] = True  # 将默认值设置为True，确保流式响应优先
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
        
        # 记录stream参数值
        if should_log_details():
            logger.debug(f"[参数处理] stream参数值: {base_data.get('stream', '未设置')}")
            
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
        5. 自动适配不同端点类型的请求和响应格式
        
    故障转移机制:
        1. 当前端点请求失败时（网络错误、认证错误、超时等），自动尝试下一个端点
        2. 切换触发点在异常处理块的continue语句，确保循环继续执行
        3. 所有端点都失败时才会抛出异常，返回502错误
        4. 错误类型精确分类，便于排查问题和优化配置
        5. 通过独立的参数副本确保每个端点使用正确的配置
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
                    # 记录响应状态码（无论是否为流式响应）
                    logger.info(f"收到响应状态码: {response.status}")
                    if should_log_details():
                        logger.debug(f"收到响应状态码: {response.status}")
                    response.raise_for_status()  # 检查响应状态
                    
                    # === 响应内容类型检查 ===
                    # 获取响应的Content-Type头
                    content_type = response.headers.get('Content-Type', '')
                    
                    # === 处理流式响应（SSE格式）===
                    if 'text/event-stream' in content_type:  # 检查是否为流式响应
                        # 初始化结果列表，直接返回原始SSE数据
                        result = []
                        async for chunk in response.content:
                            # 解码每个文本块
                            text = chunk.decode('utf-8')
                            if text.strip():
                                result.append(text)
                        
                        chunk_count = len(result)
                        if should_log_details():
                            logger.debug(f"流式响应完成，共接收 {chunk_count} 个数据块")
                        
                        # 添加更多调试日志，检查原始流式响应格式
                        if should_log_details():
                            logger.debug("------------- 流式响应格式分析开始 -------------")
                            # 检查前几个响应，看看格式是否符合预期
                            if isinstance(result, list) and len(result) > 0:
                                # 检查是否每个响应都以 "data: " 开头
                                all_start_with_data = all(isinstance(r, str) and r.strip().startswith("data:") for r in result[:5] if r.strip())
                                logger.debug(f"[原始数据分析] 所有数据块都以'data:'开头: {all_start_with_data}")
                                
                                # 检查是否每个响应都以换行结束
                                all_end_with_newline = all(isinstance(r, str) and r.endswith("\n\n") for r in result[:5] if r.strip())
                                logger.debug(f"[原始数据分析] 所有数据块都以换行结束: {all_end_with_newline}")
                                
                                # 检查是否包含JSON数据
                                sample_data = [r for r in result[:5] if r.strip()]
                                if sample_data:
                                    for i, data in enumerate(sample_data):
                                        logger.debug(f"[原始数据分析] 数据样本 #{i+1}: {data.strip()}")
                                        # 尝试提取JSON部分
                                        if data.startswith("data: "):
                                            json_part = data[6:].strip()
                                            try:
                                                _ = json.loads(json_part)
                                                logger.debug(f"[原始数据分析] 数据样本 #{i+1} 包含有效JSON")
                                            except json.JSONDecodeError:
                                                logger.debug(f"[原始数据分析] 数据样本 #{i+1} 不包含有效JSON: {json_part}")
                            logger.debug("------------- 流式响应格式分析结束 -------------")
                        
                        # 只对openrouter类型的端点进行特殊处理
                        if endpoint.type == 'openrouter':
                            if should_log_details():
                                logger.debug("对openrouter端点响应进行特殊处理")
                            # 处理openrouter的特殊字段，但保持原始格式不变
                            for i, chunk in enumerate(result):
                                if endpoint.type == 'openrouter' and "data:" in chunk and "data: [DONE]" not in chunk:
                                    try:
                                        # 使用正则表达式直接替换reasoning字段为reasoning_content
                                        # 这样可以避免JSON解析和序列化造成的转义问题
                                        if '"reasoning":' in chunk:
                                            # 记录原始数据
                                            if should_log_details():
                                                logger.debug(f"发现reasoning字段，原始数据: {chunk.strip()}")
                                            
                                            # 使用正则表达式替换字段名，但保持值不变
                                            modified_chunk = re.sub(r'"reasoning":', r'"reasoning_content":', chunk)
                                            result[i] = modified_chunk
                                            
                                            # 记录修改后的数据
                                            if should_log_details():
                                                logger.debug(f"替换reasoning字段后: {modified_chunk.strip()}")
                                    except Exception as e:
                                        # 处理失败时记录错误但不中断流程
                                        logger.error(f"处理openrouter响应时发生错误: {str(e)}")
                                        if should_log_details():
                                            logger.debug(f"详细错误信息: {traceback.format_exc()}")
                        
                        return result
                    else:
                        # === 处理非流式响应 ===
                        # 读取完整响应内容
                        response_text = await response.text()
                        
                        # 记录响应内容（限制长度以避免日志过大）
                        if should_log_details():
                            # 如果响应内容过长，只记录开头和结尾
                            if len(response_text) > 1000:
                                logger.debug(f"非流式响应(已截断): {response_text[:500]}...{response_text[-500:]}")
                            else:
                                logger.debug(f"非流式响应: {response_text}")
                            
                            # 检查原始响应是否包含markdown代码块标记
                            if "```json" in response_text:
                                logger.debug("原始非流式响应包含```json标记")
                            if "```" in response_text:
                                logger.debug("原始非流式响应包含```标记")
                        
                        # 只对openrouter类型的端点进行处理，其他类型直接返回原始文本
                        if endpoint.type == 'openrouter':
                            # 处理openrouter类型的特殊字段
                            if should_log_details():
                                logger.debug("对openrouter端点响应进行特殊处理")
                            
                            try:
                                # 解析JSON，替换reasoning字段，然后返回
                                json_response = json.loads(response_text)
                                
                                # 处理openrouter的特殊字段
                                update_reasoning(json_response, endpoint.type)
                                
                                # 记录特殊处理后的响应
                                if should_log_details():
                                    logger.debug(f"处理后的openrouter非流式响应: {json.dumps(json_response, ensure_ascii=False)[:100]}...")
                                
                                # 将处理后的JSON转换回字符串并返回
                                return json.dumps(json_response, ensure_ascii=False)
                            except json.JSONDecodeError:
                                # 解析失败，直接返回原始文本
                                if should_log_details():
                                    logger.debug("openrouter响应解析失败，直接返回原始文本")
                                return response_text
                        else:
                            # 非openrouter端点，直接返回原始文本
                            # 尝试解析JSON进行日志记录，但返回原始文本
                            if should_log_details():
                                try:
                                    # 尝试解析JSON，仅用于日志记录
                                    json_data = json.loads(response_text)
                                    
                                    # 记录content字段
                                    if isinstance(json_data, dict) and 'choices' in json_data and json_data['choices']:
                                        for choice in json_data['choices']:
                                            if 'message' in choice and 'content' in choice['message']:
                                                content = choice['message']['content']
                                                logger.debug(f"非流式响应content字段: {content[:100]}...")
                                                
                                                # 检查是否包含markdown代码块
                                                if "```json" in content:
                                                    logger.debug(f"content字段包含```json标记: {content[:50]}...{content[-50:] if len(content) > 50 else ''}")
                                                if "```" in content:
                                                    logger.debug(f"content字段包含```标记: {content[:50]}...{content[-50:] if len(content) > 50 else ''}")
                                except json.JSONDecodeError:
                                    logger.debug("非流式响应不是有效的JSON格式")
                            
                            # 无论JSON解析是否成功，都返回原始文本
                            if should_log_details():
                                logger.debug("直接返回原始响应文本")
                            return response_text
        except aiohttp.ClientError as e:
            error_msg = f"端点 {endpoint.baseurl} 请求失败 [name:{getattr(endpoint, 'name', 'unknown')}, model:{endpoint.model}, type:{endpoint.type}]: {str(e)}"
            logger.error(error_msg)
            last_error = error_msg
            continue
        except asyncio.TimeoutError as e:
            error_msg = f"端点 {endpoint.baseurl} 请求超时 [name:{getattr(endpoint, 'name', 'unknown')}, model:{endpoint.model}, type:{endpoint.type}]: {str(e)}"
            logger.error(error_msg)
            if should_log_details():
                logger.debug(f"详细错误信息: {traceback.format_exc()}")
            last_error = error_msg
            continue
        except Exception as e:
            error_msg = f"端点 {endpoint.baseurl} 发生未知错误 [name:{getattr(endpoint, 'name', 'unknown')}, model:{endpoint.model}, type:{endpoint.type}]: {str(e)}"
            logger.error(error_msg)
            if should_log_details():
                logger.debug(f"详细错误信息: {traceback.format_exc()}")
            last_error = error_msg
            continue

    # 所有端点都失败时抛出异常
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
        
    说明:
        此函数只处理openrouter端点的responses，对其他端点类型的响应不做任何修改
    """
    # 仅处理openrouter类型的端点
    if endpoint_type != 'openrouter':
        return
        
    # 处理字典类型对象
    if isinstance(obj, dict):
        # 处理reasoning字段
        if "reasoning" in obj:
            # 将reasoning字段重命名为reasoning_content
            obj["reasoning_content"] = obj.pop("reasoning")
            if should_log_details():
                logger.debug(f"已将 'reasoning' 转换为 'reasoning_content': {obj}")
        
        # 递归处理字典中的所有值
        for key, value in obj.items():
            if should_log_details():
                logger.debug(f"递归处理字典键: {key}")
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
        Dict/List/StreamingResponse: 从模型服务获取的响应数据，支持流式和非流式响应
        
    异常:
        HTTPException: 当模型组不存在、请求参数无效或服务器内部错误时抛出
        
    说明:
        1. 此接口兼容OpenAI API格式，支持流式和非流式响应
        2. 根据请求中的model字段选择对应的模型组
        3. 通过forward_request函数转发请求到后端服务
        4. 对于流式响应，使用FastAPI的StreamingResponse返回SSE格式数据
        5. 处理各种响应格式和错误情况，确保客户端始终能收到正确格式的响应
        6. 自动处理不同端点类型的格式差异，对客户端保持一致的接口
    """
    from fastapi.responses import StreamingResponse, Response
    
    # 将请求模型转换为字典格式
    params = request.model_dump()
    try:
        # 从请求参数中获取model字段作为模型组名称，默认使用'think-stream'
        model_group = params.get('model', 'think-stream')  
        
        # 验证模型组是否存在
        if model_group not in MODEL_GROUPS:
            # 如果模型组不存在，返回400错误
            raise HTTPException(400, f"无效模型名称: {model_group}")
        
        # 检查是否为流式请求
        is_stream = params.get('stream', False)
        if should_log_details():
            logger.debug(f"[请求类型] stream参数设置为: {is_stream}")
        
        # 转发请求到指定的模型组，使用chat/completions路径
        result = await forward_request(model_group, params, "chat/completions")
        
        # 添加更多日志来分析结果类型
        if should_log_details():
            logger.debug(f"[结果分析] 结果类型: {type(result).__name__}")
            if isinstance(result, list):
                logger.debug(f"[结果分析] 结果是列表，长度: {len(result)}")
                if len(result) > 0:
                    logger.debug(f"[结果分析] 第一个元素类型: {type(result[0]).__name__}")
                    if isinstance(result[0], str) and result[0].startswith("data:"):
                        logger.debug("[结果分析] 看起来是流式响应格式")
            else:
                logger.debug("[结果分析] 不是列表类型，看起来是非流式响应")
        
        # 处理流式响应 - 如果result是列表且包含类似流式数据的内容，则按流式处理
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], str) and result[0].startswith("data:"):
            if should_log_details():
                logger.debug("[流程控制] 检测到流式响应数据格式，进入流式处理逻辑")
            # 定义异步生成器函数，用于流式返回SSE数据
            async def generate():
                response_count = 0
                total_count = len(result)  # 获取数据块总数
                
                # 在函数开头记录整体的流式响应信息
                if should_log_details():
                    logger.debug(f"[原始数据总览] 准备发送流式响应，总计 {total_count} 个数据块")
                    # 检查第一个和最后一个数据块，帮助调试
                    if total_count > 0:
                        logger.debug(f"[原始数据总览] 第一个数据块: {result[0].strip()}")
                    if total_count > 1:
                        logger.debug(f"[原始数据总览] 最后一个数据块: {result[-1].strip()}")
                        
                    # 检查是否存在markdown代码块标记
                    markdown_found = False
                    for chunk in result:
                        if "```json" in chunk or "```" in chunk:
                            markdown_found = True
                            logger.debug(f"[原始数据总览] 发现markdown代码块标记在原始响应中: {chunk.strip()}")
                            break
                    if not markdown_found:
                        logger.debug("[原始数据总览] 没有在原始流式响应中找到markdown代码块标记")
                
                for chunk in result:
                    # 详细记录发送给客户端的内容
                    if should_log_details() and chunk.strip():
                        # 检查关键内容
                        if "content\":" in chunk:
                            logger.debug(f"[原始数据内容] 内容数据: {chunk.strip()}")
                            
                            # 检查是否包含markdown代码块
                            if "```json" in chunk:
                                logger.debug(f"[原始数据内容] 包含```json标记: {chunk.strip()}")
                            if "```" in chunk:
                                logger.debug(f"[原始数据内容] 包含```标记: {chunk.strip()}")
                        
                        # 记录所有数据块详情
                        logger.debug(f"[原始数据内容] 流式数据 #{response_count+1}/{total_count}: {chunk.strip()}")
                        response_count += 1
                    
                    # 处理JSON内容的转义
                    if chunk.startswith("data: "):
                        try:
                            # 检查是否是data: [DONE]
                            if chunk.strip() == "data: [DONE]":
                                yield chunk
                                continue
                                
                            # 直接使用字符串替换方法进行处理 - 避免JSON解析
                            if '"content":"[' in chunk or '"content":"[\\"' in chunk:
                                # 记录原始数据
                                if should_log_details():
                                    logger.debug(f"检测到需要转义的方括号: {chunk.strip()}")
                                
                                # 使用正则表达式查找content字段内容
                                content_pattern = r'"content":"(.*?)(?<!\\)"'
                                match = re.search(content_pattern, chunk)
                                
                                if match:
                                    original_content = match.group(1)
                                    # 检查内容是否包含未转义的方括号
                                    if (original_content.startswith('[') and not original_content.startswith('\\[')) or \
                                       ('"[' in original_content and not '\\"[' in original_content):
                                        
                                        # 对内容进行直接替换处理 - 确保方括号被正确转义
                                        escaped_content = original_content.replace('[', '\\[').replace(']', '\\]')
                                        # 替换回原始数据
                                        modified_chunk = chunk.replace(f'"content":"{original_content}"', f'"content":"{escaped_content}"')
                                        
                                        if should_log_details():
                                            logger.debug(f"[转义处理] 原始内容: {original_content}")
                                            logger.debug(f"[转义处理] 转义后内容: {escaped_content}")
                                            logger.debug(f"[转义处理] 处理后JSON: {modified_chunk.strip()}")
                                        
                                        # 记录传递给客户端的最终数据
                                        if should_log_details():
                                            logger.debug(f"[发送给客户端] 转义处理后: {modified_chunk.strip()}")
                                        
                                        yield modified_chunk
                                        continue
                            
                            # 如果没有进行特殊处理，则尝试JSON解析进行标准处理
                            json_str = chunk[6:].strip()
                            json_data = json.loads(json_str)
                            
                            # 序列化返回，确保中文正常显示
                            processed_chunk = f"data: {json.dumps(json_data, ensure_ascii=False)}\n\n"
                            yield processed_chunk
                        except Exception as e:
                            # 任何错误都返回原始数据
                            if should_log_details():
                                logger.debug(f"处理流式数据失败: {str(e)}, 返回原始数据")
                            yield chunk
                    else:
                        # 非data:前缀的数据直接返回
                        yield chunk
                
                # 在所有数据发送完成后，添加处理后数据的分析
                if should_log_details():
                    logger.debug("====================================================")
                    logger.debug("=========== 最终处理后流式响应分析开始 ===========")
                    logger.debug("====================================================")
                    # 记录最终发送的数据格式
                    for i, chunk in enumerate(result):  # 检查所有数据块
                        if isinstance(chunk, str) and chunk.startswith("data:") and "content" in chunk:
                            # 分析内容字段
                            content_match = re.search(r'"content":"(.*?)(?<!\\)"', chunk)
                            if content_match:
                                content = content_match.group(1)
                                # 检查是否包含方括号
                                has_brackets = '[' in content or ']' in content
                                has_escaped_brackets = '\\[' in content or '\\]' in content
                                
                                if has_brackets or has_escaped_brackets:
                                    logger.debug(f"[数据块 #{i+1}] 原始数据: {chunk.strip()}")
                                    logger.debug(f"[数据块 #{i+1}] 内容字段: {content}")
                                    logger.debug(f"[数据块 #{i+1}] 包含方括号: {has_brackets}, 包含转义方括号: {has_escaped_brackets}")
                    
                    logger.debug("====================================================")
                    logger.debug("=========== 最终处理后流式响应分析结束 ===========")
                    logger.debug("====================================================")
            
            # 返回StreamingResponse，设置正确的媒体类型和保持连接活跃
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        # 非流式响应，检查是否包含markdown块
        if should_log_details():
            # 记录原始响应是否包含markdown代码块
            if isinstance(result, str):
                if "```json" in result:
                    logger.debug("非流式原始响应中包含```json标记")
                if "```" in result:
                    logger.debug("非流式原始响应中包含```标记")
            
            # 尝试解析JSON进行日志记录，但确保返回原始响应
            if isinstance(result, str):
                try:
                    # 尝试解析JSON，仅用于日志记录
                    json_data = json.loads(result)
                    
                    # 记录content字段
                    if isinstance(json_data, dict) and 'choices' in json_data and json_data['choices']:
                        for choice in json_data['choices']:
                            if 'message' in choice and 'content' in choice['message']:
                                content = choice['message']['content']
                                logger.debug(f"非流式响应content字段摘要: {content[:100]}..." if len(content) > 100 else content)
                                
                                # 检查content字段是否包含markdown代码块
                                if "```json" in content:
                                    logger.debug("content字段包含```json标记")
                                if "```" in content:
                                    logger.debug("content字段包含```标记")
                except json.JSONDecodeError:
                    logger.debug("非流式响应不是有效的JSON格式，但这不影响返回")
                except Exception as e:
                    logger.debug(f"尝试解析非流式响应JSON时发生错误: {str(e)}")
                    
        # 直接返回原始响应，不做任何修改
        if should_log_details():
            logger.debug("直接返回原始非流式响应")
        
        # 对于字符串类型的响应，使用Response对象避免FastAPI进行二次序列化
        if isinstance(result, str):
            try:
                # 尝试解析JSON，确保响应是有效的JSON字符串
                json.loads(result)
                # 如果解析成功，返回带有正确内容类型的Response
                if should_log_details():
                    logger.debug("响应是有效的JSON字符串，使用Response对象返回避免二次序列化")
                return Response(content=result, media_type="application/json")
            except json.JSONDecodeError:
                # 不是有效的JSON字符串，直接返回原始内容
                if should_log_details():
                    logger.debug("响应不是有效的JSON字符串，直接返回原始内容")
        
        # 对于非字符串类型或JSON解析失败的情况，让FastAPI处理序列化
        return result
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
        在生产环境中，建议使用uvicorn命令启动，可以提供更多配置选项
    """
    # 启动ASGI服务器，加载FastAPI应用
    uvicorn.run(app, host="0.0.0.0", port=8000)
