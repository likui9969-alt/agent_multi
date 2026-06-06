"""
agents/base.py — Agent 抽象基类
================================

所有 Agent 的统一接口和行为契约。

每个 Agent 都是 LangGraph 节点函数：
  async def __call__(state: GraphState) -> dict

子类职责：
  1. 设置 self.output_model (Pydantic 模型类)
  2. 编写 System Prompt
  3. 实现 _build_messages(state) → 构建 LLM 输入
  4. 实现 _to_state_update(result, state) → 转换输出到 State dict
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Any, Type

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from graph.state import GraphState, AuditEntry
from utils.json_parser import parse_and_validate

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Agent 抽象基类。

    封装了 LLM 调用、JSON 解析、错误处理、审计日志等通用逻辑。
    子类只需设置 output_model、实现 Prompt 构建和输出转换。

    使用方式:
        class WriterAgent(BaseAgent):
            output_model = WriterOutput  # Pydantic 模型类

            @property
            def agent_name(self) -> str:
                return "content_generate"

            def _build_messages(self, state): ...
            def _to_state_update(self, result, state): ...
    """

    # 子类必须覆盖此属性，指定输出的 Pydantic 模型类
    output_model: Type[BaseModel]

    def __init__(
        self,
        llm: ChatOpenAI,
        tavily_client: Optional[Any] = None,
    ):
        """
        Args:
            llm: LangChain ChatOpenAI 实例
            tavily_client: Tavily 搜索客户端（可选）
        """
        self.llm = llm
        self.tavily_client = tavily_client

    # ── 子类必须实现 ──────────────────────────────────────────

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """返回 Agent 名称，用于 current_agent / 审计日志 / 日志输出"""
        ...

    @abstractmethod
    def _build_messages(self, state: GraphState) -> list:
        """构建发送给 LLM 的消息列表（SystemMessage + HumanMessage）"""
        ...

    @abstractmethod
    def _to_state_update(self, result: Any, state: GraphState) -> dict:
        """将 LLM 返回的 Pydantic 模型转换为 State 部分更新 dict"""
        ...

    # ── LLM 调用 + JSON 解析（子类无需覆盖） ───────────────────

    async def _invoke_and_parse(self, messages: list) -> BaseModel:
        """
        调用 LLM → 提取 JSON → 校验为 Pydantic 模型。

        替代 with_structured_output()，兼容所有 LLM 提供商。
        """
        response = await self.llm.ainvoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        logger.debug(f"[{self.agent_name}] LLM 响应前 200 字符: {text[:200]}")
        return parse_and_validate(text, self.output_model)

    # ── LangGraph 节点入口（子类通常无需覆盖） ─────────────────

    async def __call__(self, state: GraphState) -> dict:
        """
        LangGraph 节点函数统一入口。

        流程:
          1. 构建消息
          2. 调用 LLM + JSON 解析
          3. 转换结果 → State 更新
          4. 异常处理 → 返回 error
        """
        logger.info(f"[{self.agent_name}] 开始执行")

        messages = self._build_messages(state)

        try:
            result = await self._invoke_and_parse(messages)
        except Exception as exc:
            logger.error(f"[{self.agent_name}] 执行失败: {exc}")
            return {
                "current_agent": self.agent_name,
                "error": f"{self.agent_name} 执行异常: {str(exc)}",
                "audit_log": [self._make_audit("error", str(exc))],
            }

        state_update = self._to_state_update(result, state)
        logger.info(f"[{self.agent_name}] 执行完成")
        return state_update

    # ── 搜索增强（子类按需调用） ──────────────────────────────

    async def _search(self, query: str, max_results: int = 5) -> str:
        """Tavily 搜索，返回摘要文本。失败时降级为空字符串。"""
        if self.tavily_client is None:
            return ""

        try:
            response = await self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
            )
            snippets = []
            for i, r in enumerate(response.get("results", []), 1):
                snippets.append(
                    f"[{i}] {r.get('title','')}\n{r.get('content','')}\n{r.get('url','')}"
                )
            return "\n\n".join(snippets)
        except Exception as exc:
            logger.warning(f"[{self.agent_name}] 搜索失败: {exc}")
            return ""

    # ── 审计日志辅助 ─────────────────────────────────────────

    def _make_audit(self, action: str, detail: str) -> AuditEntry:
        return AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=self.agent_name,
            action=action,
            detail=detail,
        )
