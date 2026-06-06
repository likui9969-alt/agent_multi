"""
api/routes/task.py — 任务状态查询路由
======================================

GET /task/{task_id}/status  — 查询任务执行进度
GET /task/{task_id}/result  — 获取任务完整结果
"""

import logging
from fastapi import APIRouter, HTTPException

from services.task_service import get_task_service
from models.task import TaskStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["task"])


@router.get(
    "/task/{task_id}/status",
    summary="查询任务状态",
    description="""
    返回任务的当前执行状态和进度信息。

    状态枚举:
      - pending:   已创建，等待执行
      - running:   正在执行中（current_agent 指示当前阶段）
      - completed: 已完成，可通过 /task/{task_id}/result 获取脚本
      - failed:    执行失败，error 字段包含原因

    轮询建议:
      - 前端每 2-5 秒轮询一次
      - current_agent 变化即进度更新
    """,
)
async def get_task_status(task_id: str) -> dict:
    """
    查询任务状态。

    Args:
        task_id: 任务唯一标识符

    Returns:
        {
            "task_id": "T2026...",
            "topic": "AI Agent",
            "status": "running",
            "current_agent": "content_review",
            "created_at": "2026-06-06T14:30:22.000Z",
            "updated_at": "2026-06-06T14:31:45.000Z",
            "error": null,
            "result_preview": null
        }
    """
    svc = get_task_service()
    info = await svc.get(task_id)

    if info is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    return {
        "task_id": info.task_id,
        "topic": info.topic,
        "status": info.status.value,
        "current_agent": info.current_agent,
        "created_at": info.created_at,
        "updated_at": info.updated_at,
        "error": info.error,
        "result_preview": info.result_preview,
    }


@router.get(
    "/task/{task_id}/result",
    summary="获取任务结果",
    description="""
    返回完整的视频脚本（Markdown 格式）。

    仅当任务状态为 completed 时返回有效结果，
    否则返回 404 或 409（任务未完成）。
    """,
)
async def get_task_result(task_id: str) -> dict:
    """
    获取完整视频脚本。

    Args:
        task_id: 任务唯一标识符

    Returns:
        {"task_id": "...", "topic": "...", "final_script": "# ..."}
    """
    svc = get_task_service()

    # 先查状态
    info = await svc.get(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    if info.status == TaskStatus.PENDING:
        raise HTTPException(status_code=409, detail="任务尚未开始执行")
    if info.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=409, detail=f"任务正在执行中（{info.current_agent}）")
    if info.status == TaskStatus.FAILED:
        raise HTTPException(status_code=422, detail=f"任务执行失败: {info.error}")

    # 读取完整结果
    result = await svc.get_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="结果数据丢失或已过期")

    return {
        "task_id": info.task_id,
        "topic": info.topic,
        "final_script": result,
    }
