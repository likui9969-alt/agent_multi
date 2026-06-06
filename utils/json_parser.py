"""
utils/json_parser.py — LLM JSON 输出解析器
============================================

从 LLM 返回的文本中鲁棒地提取并解析 JSON。
处理常见的 LLM 输出格式问题：代码块包裹、首尾噪音、截断、转义错误。
"""

import json
import re
import logging
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> str:
    """
    从 LLM 返回文本中提取 JSON 字符串。

    处理以下格式：
      1. ```json ... ```  代码块包裹
      2. ``` ... ```       无语言标记代码块
      3. { ... }           纯 JSON
      4. 首尾有说明文字    自动定位 JSON 起止

    Args:
        text: LLM 原始响应文本

    Returns:
        str: 提取出的 JSON 字符串

    Raises:
        ValueError: 无法提取有效 JSON
    """
    if not text or not text.strip():
        raise ValueError("LLM 返回空文本")

    # 1. 尝试提取 ```json ... ``` 代码块
    match = re.search(r"```json\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. 尝试提取 ``` ... ``` 代码块
    match = re.search(r"```\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. 定位首尾花括号
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1].strip()

    # 4. 定位首尾方括号（数组格式）
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return text[first_bracket:last_bracket + 1].strip()

    raise ValueError(f"无法从文本中提取 JSON: {text[:200]}...")


def parse_and_validate(text: str, model_cls: Type[T]) -> T:
    """
    从 LLM 文本中提取 JSON 并校验为 Pydantic 模型。

    容错处理：JSON 解析失败时尝试常见修复（尾部逗号、单引号等）。

    Args:
        text: LLM 原始响应文本
        model_cls: 目标 Pydantic 模型类

    Returns:
        Pydantic 模型实例

    Raises:
        ValueError: 无法解析或校验失败
    """
    json_str = extract_json(text)

    # 尝试直接解析
    try:
        data = json.loads(json_str)
        return model_cls(**data)
    except json.JSONDecodeError:
        pass

    # 修复尝试 1: 去除尾部逗号
    try:
        fixed = re.sub(r",\s*}", "}", json_str)
        fixed = re.sub(r",\s*]", "]", fixed)
        data = json.loads(fixed)
        return model_cls(**data)
    except (json.JSONDecodeError, ValidationError):
        pass

    # 修复尝试 2: 单引号 → 双引号
    try:
        fixed = json_str.replace("'", '"')
        data = json.loads(fixed)
        return model_cls(**data)
    except (json.JSONDecodeError, ValidationError):
        pass

    raise ValueError(
        f"JSON 解析和校验均失败。原文前 300 字符: {text[:300]}"
    )
