"""
TopicAgent — 选题分析智能体
============================

LangGraph 节点：topic_analysis
读取 State.topic → 搜索研究 → LLM 分析 → 输出结构化大纲

职责边界：
  - 研究话题热度、受众画像、竞争格局
  - 生成结构化内容大纲（层级标题 + 核心要点）
  - 评估话题可行性（0-10 评分，< 6.0 触发条件边回退重试）
  - 输出逻辑脉络（起承转合），为 content_generate 提供骨架

技术栈：LangChain + DeepSeek (OpenAI 兼容协议) + Tavily Search
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from graph.state import GraphState, Outline, OutlineItem, AuditEntry
from utils.json_parser import parse_and_validate

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic 输出模型 — 与 State TypedDict 一一对应，用于 with_structured_output()
# ═══════════════════════════════════════════════════════════════════════════════

class OutlineSectionModel(BaseModel):
    """大纲单项 — 对应 OutlineItem TypedDict"""
    level: int = Field(
        description="标题层级：1=视频总标题/核心主题，2=大章节，3=小节要点"
    )
    title: str = Field(
        description="该节标题，简洁有力，不超过 20 字"
    )
    key_points: list[str] = Field(
        description="该节核心要点，3-5 个，每个不超过 30 字，用于展开讲解"
    )


class TopicAnalysisOutput(BaseModel):
    """选题分析完整输出 — LLM 结构化返回，映射到 State.outline + 附加元数据"""
    # ── 大纲核心（映射到 GraphState.outline） ──
    sections: list[OutlineSectionModel] = Field(
        description="内容大纲的层级结构，至少包含 1 个一级标题和 3-6 个二级标题"
    )
    estimated_duration: int = Field(
        description="预计视频总时长（秒），建议 300-1200 秒（5-20 分钟）"
    )
    estimated_word_count: int = Field(
        description="预计口播文案总字数，中文，建议 800-3000 字"
    )
    logic_flow: str = Field(
        description="逻辑脉络描述，说明内容的起承转合结构，50-100 字"
    )

    # ── 分析元数据（附加信息，用于条件边判断和下游 Agent 上下文） ──
    topic_score: float = Field(
        description="话题可行性综合评分 0.0-10.0：<4 冷门/高风险，4-6 一般，6-8 优质，>8 爆款潜力"
    )
    keywords: list[str] = Field(
        description="核心关键词列表，5-10 个，用于 SEO 和内容标签"
    )
    target_audience: str = Field(
        description="目标受众画像，包含年龄段/兴趣/知识水平，30-50 字"
    )
    competitor_analysis: str = Field(
        description="同类内容竞争分析，当前平台覆盖度、差异化切入点，50-100 字"
    )
    recommendation: str = Field(
        description="综合选题建议：角度选择、风格建议、注意事项，50-100 字"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
你是一名资深的内容策略师和视频选题分析师，拥有 10 年新媒体内容策划经验。

## 你的能力
- 分析话题热度、受众匹配度、内容可行性
- 设计结构清晰、逻辑连贯的内容大纲
- 预判内容爆款潜力并给出优化建议

## 工作流程
1. 理解用户提供的主题
2. 结合搜索结果（如提供）补充行业背景
3. 生成结构化大纲（层级标题 + 核心要点）
4. 评估话题可行性（0-10 分）
5. 给出目标受众、竞争分析和选题建议

## 输出要求
- 所有输出必须严格遵循指定的 JSON 格式
- sections 至少包含 1 个一级标题（level=1）和 3-6 个二级标题（level=2）
- key_points 每项 3-5 个，简洁有信息量
- topic_score 客观评估，不虚高
- 所有文本字段使用中文

## JSON 输出格式（必须严格遵守）
{
    "sections": [
        {"level": 1, "title": "主标题", "key_points": ["要点1", "要点2", "要点3"]},
        {"level": 2, "title": "章节标题", "key_points": ["要点1", "要点2", "要点3"]}
    ],
    "estimated_duration": 600,
    "estimated_word_count": 1500,
    "logic_flow": "起：引入话题 → 承：展开分析 → 转：深入探讨 → 合：总结展望",
    "topic_score": 7.5,
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"],
    "target_audience": "对AI技术感兴趣的20-35岁互联网从业者",
    "competitor_analysis": "同类内容较多，建议从垂直细分角度切入",
    "recommendation": "建议结合具体案例增强说服力"
}
"""

HUMAN_TEMPLATE = """\
## 用户提交的创作主题
{topic}

## 搜索结果参考（可能为空）
{search_context}

## 任务
请对上述主题进行全面的选题分析，包含：
1. 话题可行性评估（热度、受众规模、竞争程度）
2. 结构化内容大纲（层级分明，逻辑连贯）
3. 目标受众画像
4. 差异化切入建议

请严格按照 JSON Schema 输出结构化结果。
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TopicAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TopicAgent:
    """
    选题分析 Agent — 独立实现，不继承 BaseAgent（因搜索流程特殊）。

    使用方式：
        agent = TopicAgent(llm=chat_deepseek, tavily_client=tavily)
        graph.add_node("topic_analysis", agent)
    """

    output_model = TopicAnalysisOutput

    def __init__(
        self,
        llm: ChatOpenAI,
        tavily_client: Optional[Any] = None,
    ):
        self.llm = llm
        self.tavily_client = tavily_client

    # ── LangGraph 节点调用入口 ──────────────────────────────────────
    async def __call__(self, state: GraphState) -> dict:
        """
        LangGraph 节点函数。覆盖基类以支持搜索增强和附加元数据返回。
        """
        topic = state.get("topic", "")
        logger.info(f"[TopicAgent] 开始选题分析，topic='{topic[:50]}...'")

        # 步骤 1: 搜索研究
        search_context = await self._research(topic)

        # 步骤 2: 构建消息 + 调用 LLM + JSON 解析
        messages = self._build_messages(topic, search_context)
        try:
            response = await self.llm.ainvoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            result = parse_and_validate(text, self.output_model)
        except Exception as exc:
            logger.error(f"[TopicAgent] 执行失败: {exc}")
            return {
                "current_agent": "topic_analysis",
                "error": f"选题分析异常: {str(exc)}",
                "audit_log": [self._make_audit("error", str(exc))],
            }

        # 步骤 3: 转换输出
        outline = self._to_outline(result)

        # 步骤 4: 审计日志
        audit_entry = self._make_audit(
            "complete",
            f"评分={result.topic_score:.1f}，"
            f"关键词={len(result.keywords)}个，"
            f"大纲={len(result.sections)}节，"
            f"预计{result.estimated_word_count}字/{result.estimated_duration}秒",
        )

        logger.info(
            f"[TopicAgent] 选题分析完成 → score={result.topic_score:.1f}, "
            f"sections={len(result.sections)}"
        )

        return {
            "outline": outline,
            "current_agent": "topic_analysis",
            "retry_count": state.get("retry_count", 0),
            "topic_score": result.topic_score,
            "keywords": result.keywords,
            "audit_log": [audit_entry],
        }

    # ── 搜索研究 ──────────────────────────────────────────────────
    async def _research(self, topic: str) -> str:
        """
        使用 Tavily 搜索话题相关信息，生成搜索上下文注入 Prompt。

        仅当 Tavily 客户端已配置且 API Key 可用时才执行搜索。
        搜索失败不阻塞流程，降级返回空字符串（纯 LLM 分析模式）。

        Args:
            topic: 用户创作主题

        Returns:
            str: 搜索结果摘要文本，失败或无客户端时返回 ""
        """
        if self.tavily_client is None:
            logger.info("[TopicAgent] 未配置 Tavily 客户端，跳过搜索增强")
            return ""

        try:
            # Tavily 语义搜索 —— 返回 AI 优化的搜索结果
            response = await self.tavily_client.search(
                query=f"{topic} 热门 趋势 分析",
                search_depth="advanced",      # "basic" | "advanced"
                max_results=5,                # 控制上下文长度
                include_domains=[],            # 可限定域名，空=不限
            )

            # 组装搜索上下文
            snippets = []
            for idx, result in enumerate(response.get("results", []), start=1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                snippets.append(f"[来源{idx}] {title}\n{content}\n🔗 {url}")

            context = "\n\n".join(snippets) if snippets else ""
            logger.info(f"[TopicAgent] Tavily 搜索完成，获得 {len(snippets)} 条结果")
            return context

        except Exception as exc:
            logger.warning(f"[TopicAgent] Tavily 搜索失败，降级为无搜索模式：{exc}")
            return ""

    # ── Prompt 构建 ────────────────────────────────────────────────
    def _build_messages(self, topic: str, search_context: str) -> list:
        """
        构建发送给 LLM 的消息列表。

        Args:
            topic: 用户主题
            search_context: 搜索上下文（可能为空字符串）

        Returns:
            list[SystemMessage, HumanMessage]
        """
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        human_msg = HumanMessage(
            content=HUMAN_TEMPLATE.format(
                topic=topic,
                search_context=search_context or "（未提供搜索结果，请基于你的知识进行分析）",
            )
        )
        return [system_msg, human_msg]

    # ── 输出转换 ───────────────────────────────────────────────────
    def _to_outline(self, result: TopicAnalysisOutput) -> Outline:
        """
        将 LLM 返回的 Pydantic 模型转换为 State 所需的 TypedDict 格式。

        Outline TypedDict 结构：
            sections: list[OutlineItem]   — 层级标题+要点
            estimated_duration: int       — 预计时长
            estimated_word_count: int     — 预计字数
            logic_flow: str               — 逻辑脉络

        Args:
            result: LLM 结构化输出的 Pydantic 实例

        Returns:
            Outline: 符合 GraphState.outline 字段类型的 TypedDict
        """
        # Pydantic v2 用 model_dump()，v1 用 dict()
        data = result.model_dump()

        outline_items: list[OutlineItem] = []
        for sec in data["sections"]:
            item: OutlineItem = {
                "level": sec["level"],
                "title": sec["title"],
                "key_points": sec["key_points"],
            }
            outline_items.append(item)

        outline: Outline = {
            "sections": outline_items,
            "estimated_duration": data["estimated_duration"],
            "estimated_word_count": data["estimated_word_count"],
            "logic_flow": data["logic_flow"],
        }
        return outline

    def _make_audit(self, action: str, detail: str) -> AuditEntry:
        """构造审计日志条目（BaseAgent 同名方法，因不继承而在此独立定义）。"""
        return AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent="topic_analysis",
            action=action,
            detail=detail,
        )

    def get_topic_metadata(self, result: TopicAnalysisOutput) -> dict:
        """
        提取选题分析的元数据（不在 Outline 中的额外信息），
        供条件边 check_topic_score 和下游 Agent 使用。

        这些字段当前不直接存入 GraphState（State 中无对应字段），
        通过此方法暴露给调用者按需使用。

        Args:
            result: LLM 结构化输出

        Returns:
            dict: {topic_score, keywords, target_audience,
                   competitor_analysis, recommendation}
        """
        return {
            "topic_score": result.topic_score,
            "keywords": result.keywords,
            "target_audience": result.target_audience,
            "competitor_analysis": result.competitor_analysis,
            "recommendation": result.recommendation,
        }

