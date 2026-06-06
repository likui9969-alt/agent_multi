"""
agents/writer_agent.py — 内容生成 Agent
========================================

LangGraph 节点：content_generate
输入: State.outline + State.topic (+ State.review_result.feedback 回退场景)
输出: State.draft（完整文案 + 段落拆分 + SEO 关键词 + 参考来源）

职责:
  - 基于结构化大纲撰写 Markdown 格式完整文案
  - 搜索补充资料增强内容深度
  - 回退重写时读取审核反馈修正内容
  - 自动统计字数、拆分段落、提取 SEO 标签
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from graph.state import GraphState, Draft, AuditEntry
from agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic 输出模型
# ═══════════════════════════════════════════════════════════════════════════════

class WriterOutput(BaseModel):
    """WriterAgent 结构化输出 — 映射到 State.draft"""
    title: str = Field(
        description="文章/视频文案标题，吸引眼球，不超过 30 字"
    )
    full_text: str = Field(
        description="完整文案正文，Markdown 格式，按大纲章节展开，800-3000 字"
    )
    paragraphs: list[str] = Field(
        description="按自然段拆分的文案片段列表，每段 100-300 字，用于后续分镜映射"
    )
    seo_keywords: list[str] = Field(
        description="SEO 关键词/标签列表，5-10 个，覆盖核心话题和长尾词"
    )
    references: list[str] = Field(
        description="引用的信息来源或推荐参考资料的 URL 列表，可为空"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
你是一名资深新媒体内容创作者，擅长将结构化大纲转化为引人入胜的完整文案。

## 你的写作风格
- 开篇钩子：前 3 句抓住注意力（设问/数据冲击/故事引入）
- 正文：逻辑清晰，段落分明，每段有明确的中心思想
- 语言：口语化但不失专业，适合视频口播
- 结尾：有力总结 + 引导互动

## 写作要求
1. 严格按照提供的大纲结构展开，不遗漏任何章节
2. 每段 100-300 字，段落之间自然过渡
3. 关键数据和观点需标注来源或说明依据
4. 全文 Markdown 格式，使用 ##/### 标题层级
5. 中文为主，专业术语可保留英文

## JSON 输出格式（必须严格遵守，输出纯JSON，不要包裹在```json```中）
{
    "title": "文章标题",
    "full_text": "## 章节1\\n\\n段落内容...\\n\\n## 章节2\\n\\n段落内容...",
    "paragraphs": ["段落1文本...", "段落2文本...", "段落3文本..."],
    "seo_keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
    "references": ["https://example.com/ref1", "https://example.com/ref2"]
}
"""

HUMAN_TEMPLATE = """\
## 创作主题
{topic}

## 内容大纲
{outline_text}

## 参考搜索资料
{search_context}

## 审核反馈（可能为空，非空时必须针对性修改）
{feedback}

## 任务
请基于上述大纲撰写完整文案，输出为结构化 JSON。
{feedback_note}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# WriterAgent
# ═══════════════════════════════════════════════════════════════════════════════

class WriterAgent(BaseAgent):

    output_model = WriterOutput

    @property
    def agent_name(self) -> str:
        return "content_generate"

    def __init__(
        self,
        llm: ChatOpenAI,
        tavily_client: Optional[Any] = None,
    ):
        super().__init__(llm, tavily_client)

    # ── Prompt 构建 ──────────────────────────────────────────────

    def _build_messages(self, state: GraphState) -> list:
        """
        构建 LLM 输入消息。

        从 state 中提取:
          - topic: 创作主题（回显）
          - outline: 结构化大纲（序列化为可读文本）
          - review_result.feedback: 审核反馈（回退场景）
        """
        topic = state.get("topic", "")
        outline = state.get("outline")

        # 大纲 → 可读文本
        outline_text = self._format_outline(outline) if outline else "（大纲缺失，请基于主题自由发挥）"

        # 审核反馈（回退重写场景）
        review = state.get("review_result")
        feedback = ""
        feedback_note = ""
        if review and not review.get("passed", True):
            feedback = review.get("feedback", "")
            feedback_note = "⚠️ 这是修改稿，请务必根据「审核反馈」修正上一版的问题。"

        # 搜索补充资料
        search_context = ""  # 按需搜索，避免每次大量 API 调用
        # search_context = await self._search(f"{topic} 深度分析 数据")

        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        human_msg = HumanMessage(content=HUMAN_TEMPLATE.format(
            topic=topic,
            outline_text=outline_text,
            search_context=search_context or "（未提供搜索资料，基于你的知识创作）",
            feedback=feedback or "（首次创作，无审核反馈）",
            feedback_note=feedback_note,
        ))
        return [system_msg, human_msg]

    # ── 输出转换 ─────────────────────────────────────────────────

    def _to_state_update(self, result: WriterOutput, state: GraphState) -> dict:
        """
        WriterOutput Pydantic 模型 → State.draft TypedDict。

        自动计算实际字数。
        """
        word_count = len(result.full_text.replace(" ", "").replace("\n", ""))

        draft: Draft = {
            "full_text": result.full_text,
            "word_count": word_count,
            "paragraphs": result.paragraphs,
            "seo_keywords": result.seo_keywords,
            "references": result.references,
        }

        retry_count = state.get("retry_count", 0)
        review = state.get("review_result")
        is_retry = review and not review.get("passed", True)

        return {
            "draft": draft,
            "current_agent": self.agent_name,
            "retry_count": 0 if not is_retry else retry_count,
            "audit_log": [self._make_audit(
                "complete",
                f"字数={word_count}, 段落={len(result.paragraphs)}段, "
                f"SEO关键词={len(result.seo_keywords)}个"
                + (" (回退重写)" if is_retry else "")
            )],
        }

    # ── 大纲格式化 ───────────────────────────────────────────────

    @staticmethod
    def _format_outline(outline: dict) -> str:
        """将 Outline TypedDict 格式化为 LLM 可读的文本。"""
        lines = []
        sections = outline.get("sections", [])
        for sec in sections:
            level = sec.get("level", 1)
            prefix = "#" * level
            title = sec.get("title", "")
            lines.append(f"{prefix} {title}")
            for kp in sec.get("key_points", []):
                lines.append(f"  - {kp}")
            lines.append("")

        lines.append(f"预计时长: {outline.get('estimated_duration', 0)} 秒")
        lines.append(f"预计字数: {outline.get('estimated_word_count', 0)} 字")
        logic = outline.get("logic_flow", "")
        if logic:
            lines.append(f"逻辑脉络: {logic}")

        return "\n".join(lines)
