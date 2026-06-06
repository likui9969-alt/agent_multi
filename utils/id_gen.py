"""
utils/id_gen.py — 任务 ID 生成器
=================================

生成全局唯一、时间有序的任务标识符。

当前实现：基于时间戳 + 随机数的 26 字符 ID
格式：T + 年月日时分秒(14) + 随机hex(8) + 毫秒(3)
示例：T20260606143022A3F8C1D2005

替代方案：
  - ulid 库（pip install python-ulid）→ 更标准的 ULID
  - uuid7  → 时间有序 UUID
"""

import random
from datetime import datetime, timezone


def generate_task_id() -> str:
    """
    生成任务唯一标识符。

    Returns:
        str: 26字符 ID，格式 T{YYYYMMDDHHmmss}{8位随机hex}{毫秒3位}
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d%H%M%S")           # 14字符 时间
    random_hex = format(random.getrandbits(32), "08x")  #  8字符 随机
    millis = str(now.microsecond // 1000).zfill(3)      #  3字符 毫秒
    return f"T{timestamp}{random_hex}{millis}"           # 26字符 总计
