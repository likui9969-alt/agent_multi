"""
api/schemas/request.py — 请求体 Pydantic 模型
==============================================

定义所有 API 端点的输入数据结构，包含自动校验和示例。
"""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """
    POST /generate 请求体。

    示例:
        {"topic": "2025年AI Agent发展趋势"}
        {"topic": "如何用Python构建RAG系统", "max_retries": 5}
    """
    topic: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="用户输入的创作主题",
        examples=["2025年AI Agent技术发展趋势与行业落地案例"],
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="每个阶段的最大重试次数，默认 3，范围 1-10",
    )
