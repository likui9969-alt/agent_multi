"""
api/schemas/response.py — 响应体 Pydantic 模型
===============================================

定义所有 API 端点的输出数据结构。
"""

from typing import Optional
from pydantic import BaseModel, Field


class GenerateResponse(BaseModel):
    """
    POST /generate 成功响应。

    返回 LangGraph 流水线执行完毕后的最终视频脚本。
    """
    task_id: str = Field(
        ...,
        description="任务唯一标识符（ULID），用于追踪和审计",
    )
    final_script: str = Field(
        ...,
        description="最终视频脚本，Markdown 格式，包含标题、分镜、旁白、字幕等",
    )
    topic: str = Field(
        ...,
        description="原始创作主题（回显）",
    )


class ErrorResponse(BaseModel):
    """
    通用错误响应，用于参数校验失败、LLM 异常、工作流超时等场景。
    """
    error: str = Field(
        ...,
        description="人类可读的错误描述",
    )
    detail: Optional[str] = Field(
        default=None,
        description="详细错误信息（调试用）",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="关联的任务 ID（如有）",
    )
