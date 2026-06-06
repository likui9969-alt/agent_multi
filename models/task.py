"""
models/task.py — 任务模型
==========================

定义任务状态枚举和任务信息数据结构。

任务生命周期:
  PENDING → RUNNING → COMPLETED
                    → FAILED
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timezone


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"        # 已创建，等待执行
    RUNNING = "running"         # 正在执行 LangGraph 工作流
    COMPLETED = "completed"     # 成功完成，final_script 可用
    FAILED = "failed"           # 执行失败，error 字段有详情


@dataclass
class TaskInfo:
    """
    任务完整信息。

    存储在 Redis Hash 中，key = task:{task_id}。
    字段序列化为 JSON 字符串存入 Hash field。
    """
    task_id: str
    topic: str
    status: TaskStatus = TaskStatus.PENDING
    current_agent: str = "START"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None
    # 以下字段在完成后填充
    result_preview: Optional[str] = None    # final_script 前 200 字符预览
    result_full_key: Optional[str] = None   # Redis key 指向完整结果

    def to_redis_hash(self) -> dict:
        """
        转为 Redis Hash 可存储的 dict（所有值必须是 str）。
        """
        data = {}
        for k, v in asdict(self).items():
            if v is None:
                data[k] = ""
            elif isinstance(v, Enum):
                data[k] = v.value
            else:
                data[k] = str(v) if not isinstance(v, str) else v
        return data

    @classmethod
    def from_redis_hash(cls, data: dict) -> "TaskInfo":
        """
        从 Redis Hash 返回的 bytes dict 反序列化。
        """
        def _decode(v):
            if isinstance(v, bytes):
                return v.decode("utf-8")
            return v

        return cls(
            task_id=_decode(data.get("task_id", "")),
            topic=_decode(data.get("topic", "")),
            status=TaskStatus(_decode(data.get("status", "pending"))),
            current_agent=_decode(data.get("current_agent", "START")),
            created_at=_decode(data.get("created_at", "")),
            updated_at=_decode(data.get("updated_at", "")),
            error=_decode(data.get("error", "")) or None,
            result_preview=_decode(data.get("result_preview", "")) or None,
            result_full_key=_decode(data.get("result_full_key", "")) or None,
        )
