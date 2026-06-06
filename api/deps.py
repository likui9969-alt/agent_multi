"""
api/deps.py — FastAPI 依赖注入
===============================

使用 FastAPI 的 Depends() 机制，按需创建并注入：
  - Settings 配置
  - DeepSeek LLM 客户端
  - Tavily 搜索客户端
  - LangGraph 编译后的工作流

每项均为单例（请求级别复用），避免重复初始化。
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI

from backend.config.settings import get_settings, Settings

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Settings 单例（进程级别缓存）
# ══════════════════════════════════════════════════════════════════════

@lru_cache()
def _cached_settings() -> Settings:
    """进程级别的 Settings 缓存，.env 只加载一次。"""
    return get_settings()


# ══════════════════════════════════════════════════════════════════════
# LLM 客户端（请求级别单例）
# ══════════════════════════════════════════════════════════════════════

_llm_instance: Optional[ChatOpenAI] = None


def get_llm() -> ChatOpenAI:
    """
    获取 DeepSeek LLM 客户端（进程级别单例）。

    DeepSeek 使用 OpenAI 兼容协议，直接用 ChatOpenAI 连接。
    连接池由 httpx 内部管理，单例复用即可。
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    settings = _cached_settings()
    # 优先使用通用 LLM 配置，未设置时回退到 DeepSeek 配置
    api_key = settings.llm_api_key or settings.deepseek_api_key
    base_url = settings.llm_base_url or settings.deepseek_base_url
    _llm_instance = ChatOpenAI(
        model=settings.llm_model,
        api_key=api_key,
        base_url=base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    logger.info(
        f"[DI] LLM 已初始化: model={settings.llm_model} "
        f"base_url={settings.deepseek_base_url}"
    )
    return _llm_instance


# ══════════════════════════════════════════════════════════════════════
# Tavily 搜索客户端（请求级别单例）
# ══════════════════════════════════════════════════════════════════════

_tavily_instance = None  # Optional[TavilyClient]


def get_tavily_client():
    """
    获取 Tavily 搜索客户端（进程级别单例）。

    Returns:
        TavilyClient | None: 成功初始化返回客户端，失败返回 None（降级为纯 LLM 模式）。
    """
    global _tavily_instance
    if _tavily_instance is not None:
        return _tavily_instance

    settings = _cached_settings()
    if not settings.tavily_api_key:
        logger.warning("[DI] Tavily API Key 未配置，搜索增强已禁用")
        return None

    try:
        # Tavily Python SDK 导入
        from tavily import TavilyClient
        _tavily_instance = TavilyClient(api_key=settings.tavily_api_key)
        logger.info("[DI] Tavily 搜索客户端已初始化")
        return _tavily_instance
    except ImportError:
        logger.warning("[DI] tavily-python 未安装，搜索增强已禁用")
        return None
    except Exception as exc:
        logger.error(f"[DI] Tavily 初始化失败: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════
# Redis 客户端（进程级别单例）
# ══════════════════════════════════════════════════════════════════════

async def get_redis():
    """
    获取 Redis 异步客户端。

    FastAPI 路由中通过 Depends(get_redis) 注入。
    """
    from storage.redis_client import get_redis as _get_redis
    return await _get_redis()


# ══════════════════════════════════════════════════════════════════════
# LangGraph 工作流（进程级别单例）
# ══════════════════════════════════════════════════════════════════════

_graph_instance = None  # Optional[CompiledStateGraph]


def get_graph():
    """
    获取编译后的 LangGraph 工作流（进程级别单例）。

    首次调用时构建整个 StateGraph（含 4 个 Agent + 条件边），
    后续请求直接复用编译后的图实例。
    """
    global _graph_instance
    if _graph_instance is not None:
        return _graph_instance

    from graph.graph_builder import build_graph

    llm = get_llm()
    tavily = get_tavily_client()

    _graph_instance = build_graph(
        llm=llm,
        tavily_client=tavily,
        # redis_checkpointer 按需启用，此处先不配置
    )
    logger.info("[DI] LangGraph 工作流已编译并缓存")
    return _graph_instance
