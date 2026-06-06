"""
backend/config/settings.py — 应用配置中心
==========================================

使用 Pydantic Settings 从 .env 文件和环境变量加载所有配置项。
配置分为 3 组：LLM、Redis、应用。

使用方式:
    from backend.config.settings import get_settings
    settings = get_settings()
    print(settings.llm_model)  # "deepseek-v3"
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 确保项目根目录的 .env 被加载
load_dotenv(
    os.path.join(os.path.dirname(__file__), "..", "..", ".env")
)


class Settings(BaseSettings):
    """
    应用全局配置，所有值从 .env / 环境变量自动注入。

    Pydantic Settings 按以下优先级读取：
      1. 环境变量（最高）
      2. .env 文件
      3. 字段默认值（最低）
    """

    # ── LLM 配置 ──
    llm_model: str = "deepseek-v3"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    # 通用 LLM 配置（优先级高于下面的具体厂商配置）
    llm_api_key: str = ""
    llm_base_url: str = ""

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    # ── Embedding 配置 ──
    embedding_model: str = "text-embedding-v3"
    dashscope_api_key: str = ""
    dashscope_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ── Tavily 搜索 ──
    tavily_api_key: str = ""

    # ── Redis ──
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # ── 应用 ──
    app_name: str = "AIGC Multi-Agent Platform"
    app_version: str = "0.1.0"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ── 工作流 ──
    max_retries: int = 3
    request_timeout: int = 300  # 秒

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # 忽略 .env 中未定义的额外字段（如 CHUNK_SIZE / JIRA_* 等）
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    获取 Settings 单例（带 LRU 缓存）。

    使用 @lru_cache 确保 .env 只解析一次，
    后续调用直接返回缓存的 Settings 实例。
    """
    return Settings()
