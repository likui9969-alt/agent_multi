"""
storage/redis_client.py — Redis 客户端封装
===========================================

基于 redis-py (async) 的连接池管理和客户端单例。

Key 命名规范:
  task:{task_id}         — Hash, 任务状态和元信息
  task:{task_id}:result  — String, 完整脚本 Markdown（大文本单独存）
"""

import logging
from typing import Optional

import redis.asyncio as aioredis

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

# 进程级单例
_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """
    获取 Redis 异步客户端（进程级别单例）。

    首次调用时创建连接池，后续调用返回同一实例。
    """
    global _redis
    if _redis is not None:
        return _redis

    settings = get_settings()
    _redis = aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        decode_responses=True,   # 自动 bytes → str 解码
        max_connections=10,
        socket_timeout=5,
        retry_on_timeout=True,
    )
    # 验证连接
    await _redis.ping()
    logger.info(
        f"[Redis] 已连接 {settings.redis_host}:{settings.redis_port} "
        f"db={settings.redis_db}"
    )
    return _redis


async def close_redis():
    """关闭 Redis 连接（在应用 shutdown 时调用）"""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
        logger.info("[Redis] 连接已关闭")


# ══════════════════════════════════════════════════════════════════════
# Key 生成工具
# ══════════════════════════════════════════════════════════════════════

def task_key(task_id: str) -> str:
    """任务状态 Hash Key"""
    return f"task:{task_id}"


def task_result_key(task_id: str) -> str:
    """任务完整结果 String Key"""
    return f"task:{task_id}:result"


# TTL 配置
TASK_TTL = 60 * 60 * 24 * 7  # 任务数据保留 7 天
RESULT_TTL = 60 * 60 * 24 * 30  # 结果保留 30 天
