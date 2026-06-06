"""
services/task_service.py — 任务状态服务
========================================

基于 Redis 的任务生命周期管理服务。

职责:
  - 创建任务 → PENDING
  - 开始执行 → RUNNING
  - 更新中间状态（current_agent）
  - 完成 → COMPLETED（保存结果）
  - 失败 → FAILED（保存错误）
  - 查询任务状态 / 结果

设计原则:
  - 零侵入：不修改 agents/、graph/ 任何代码
  - 旁路模式：在路由层调用 service，graph 无感知
  - Redis 故障不阻塞业务：所有写操作 try/except 包裹
"""

import json
import logging
from typing import Optional
from datetime import datetime, timezone

from models.task import TaskInfo, TaskStatus
from storage.redis_client import (
    get_redis,
    task_key,
    task_result_key,
    TASK_TTL,
    RESULT_TTL,
)

logger = logging.getLogger(__name__)


class TaskService:
    """
    任务状态服务 —— 纯 Service 层，无状态，方法均为 async。

    使用方式:
        svc = TaskService()
        await svc.create(task_id, topic)       # 创建任务
        await svc.mark_running(task_id)         # 开始执行
        await svc.mark_completed(task_id, ...)  # 执行成功
        await svc.mark_failed(task_id, error)   # 执行失败
        info = await svc.get(task_id)           # 查询状态
    """

    # ════════════════════════════════════════════════════════════
    # 写入操作
    # ════════════════════════════════════════════════════════════

    async def create(self, task_id: str, topic: str) -> TaskInfo:
        """
        创建新任务，状态 = PENDING。

        Args:
            task_id: 任务唯一 ID
            topic: 用户创作主题

        Returns:
            TaskInfo: 创建的任务信息
        """
        info = TaskInfo(
            task_id=task_id,
            topic=topic,
            status=TaskStatus.PENDING,
        )
        await self._save(info)
        logger.info(f"[TaskService] 任务已创建: {task_id} topic='{topic[:50]}...'")
        return info

    async def mark_running(self, task_id: str, current_agent: str = "topic_analysis") -> None:
        """
        标记任务为 RUNNING。

        Args:
            task_id: 任务 ID
            current_agent: 当前执行的 Agent 节点名
        """
        await self._update(
            task_id,
            status=TaskStatus.RUNNING,
            current_agent=current_agent,
        )
        logger.info(f"[TaskService] 任务开始执行: {task_id} agent={current_agent}")

    async def update_agent(self, task_id: str, current_agent: str) -> None:
        """
        更新当前执行的 Agent 节点名（用于进度轮询）。

        调用时机: 每个 Agent 执行完毕后。

        Args:
            task_id: 任务 ID
            current_agent: 刚完成的 Agent 节点名
        """
        await self._update(task_id, current_agent=current_agent)
        logger.debug(f"[TaskService] 任务进度更新: {task_id} → {current_agent}")

    async def mark_completed(
        self,
        task_id: str,
        result_markdown: str,
    ) -> None:
        """
        标记任务为 COMPLETED，保存完整结果到独立 Key。

        大文本（final_script Markdown）存入 task:{id}:result String，
        task:{id} Hash 中仅存前 200 字预览 + result key 引用。

        Args:
            task_id: 任务 ID
            result_markdown: 最终视频脚本的 Markdown 文本
        """
        # 1. 存完整结果到独立 Key
        result_key = task_result_key(task_id)
        try:
            r = await get_redis()
            await r.set(result_key, result_markdown, ex=RESULT_TTL)
        except Exception as exc:
            logger.warning(f"[TaskService] Redis 结果存储失败: {exc}")

        # 2. 更新任务状态
        preview = result_markdown[:200] + "..." if len(result_markdown) > 200 else result_markdown
        await self._update(
            task_id,
            status=TaskStatus.COMPLETED,
            current_agent="END",
            result_preview=preview,
            result_full_key=result_key,
        )
        logger.info(f"[TaskService] 任务完成: {task_id} result_length={len(result_markdown)}")

    async def mark_failed(self, task_id: str, error: str) -> None:
        """
        标记任务为 FAILED。

        Args:
            task_id: 任务 ID
            error: 错误描述
        """
        await self._update(
            task_id,
            status=TaskStatus.FAILED,
            error=error,
        )
        logger.error(f"[TaskService] 任务失败: {task_id} error='{error[:100]}...'")

    # ════════════════════════════════════════════════════════════
    # 查询操作
    # ════════════════════════════════════════════════════════════

    async def get(self, task_id: str) -> Optional[TaskInfo]:
        """
        查询任务状态。

        Args:
            task_id: 任务 ID

        Returns:
            TaskInfo | None: 任务信息，不存在返回 None
        """
        try:
            r = await get_redis()
            key = task_key(task_id)
            data = await r.hgetall(key)
            if not data:
                return None
            return TaskInfo.from_redis_hash(data)
        except Exception as exc:
            logger.warning(f"[TaskService] Redis 查询失败 task_id={task_id}: {exc}")
            return None

    async def get_result(self, task_id: str) -> Optional[str]:
        """
        查询任务完整结果（final_script Markdown）。

        仅当任务状态为 COMPLETED 时返回有效结果。

        Args:
            task_id: 任务 ID

        Returns:
            str | None: 完整 Markdown 脚本，不存在或未完成返回 None
        """
        try:
            # 先查状态
            info = await self.get(task_id)
            if info is None or info.status != TaskStatus.COMPLETED:
                return None

            # 从 result key 读取完整文本
            result_key = info.result_full_key or task_result_key(task_id)
            r = await get_redis()
            result = await r.get(result_key)
            return result
        except Exception as exc:
            logger.warning(f"[TaskService] 结果查询失败 task_id={task_id}: {exc}")
            return None

    # ════════════════════════════════════════════════════════════
    # 内部方法
    # ════════════════════════════════════════════════════════════

    async def _save(self, info: TaskInfo) -> None:
        """存储完整 TaskInfo 到 Redis Hash"""
        try:
            r = await get_redis()
            key = task_key(info.task_id)
            await r.hset(key, mapping=info.to_redis_hash())
            await r.expire(key, TASK_TTL)
        except Exception as exc:
            logger.error(f"[TaskService] Redis 存储失败: {exc}")

    async def _update(self, task_id: str, **fields) -> None:
        """
        部分更新任务字段。

        先 GET 再 SET，避免覆盖未指定的字段。
        fields 中可含: status / current_agent / error / result_preview / result_full_key
        """
        try:
            r = await get_redis()
            key = task_key(task_id)

            # 读取当前 Hash
            current = await r.hgetall(key)
            if not current:
                logger.warning(f"[TaskService] 任务不存在: {task_id}")
                return

            # 合并更新
            updated_at = datetime.now(timezone.utc).isoformat()
            updates = {"updated_at": updated_at}
            for k, v in fields.items():
                if v is not None:
                    updates[k] = v.value if isinstance(v, TaskStatus) else v

            await r.hset(key, mapping=updates)
            # 刷新 TTL
            await r.expire(key, TASK_TTL)
        except Exception as exc:
            logger.error(f"[TaskService] Redis 更新失败 task_id={task_id}: {exc}")


# ══════════════════════════════════════════════════════════════════════
# 全局单例（进程级别，无状态，可安全共享）
# ══════════════════════════════════════════════════════════════════════

_task_service: Optional[TaskService] = None


def get_task_service() -> TaskService:
    """获取 TaskService 单例"""
    global _task_service
    if _task_service is None:
        _task_service = TaskService()
    return _task_service
