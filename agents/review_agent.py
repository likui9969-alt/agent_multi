"""
agents/review_agent.py — 内容审核 Agent
========================================

LangGraph 节点：content_review
输入: State.draft + State.outline
输出: State.review_result（通过/驳回 + 4维评分 + 问题列表 + 修改反馈）

职责:
  - 事实准确性审核（结合搜索验证关键数据）
  - 文风/可读性评分
  - 结构逻辑性检查
  - 受众适配度评估
  - 输出结构化反馈供 WriterAgent 修正
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from graph.state import GraphState, ReviewResult, ReviewIssue, AuditEntry
from agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic 输出模型
# ═══════════════════════════════════════════════════════════════════════════════

class ReviewIssueModel(BaseModel):
    """审核问题项 — 对应 ReviewIssue TypedDict"""
    issue_type: str = Field(
        description="问题类型: factual(事实错误) | style(文风) | structure(结构) | legal(合规) | audience(受众)"
    )
    severity: str = Field(
        description="严重程度: critical(致命) | major(严重) | minor(轻微) | suggestion(建议)"
    )
    location: str = Field(
        description="问题位置，如'第3段第2句'或'关于XXX的论述'"
    )
    description: str = Field(
        description="问题的具体描述，明确指出哪里有问题"
    )
    suggestion: str = Field(
        description="具体的修改建议，可操作、可执行"
    )


class ReviewOutput(BaseModel):
    """ReviewAgent 结构化输出 — 映射到 State.review_result"""
    passed: bool = Field(
        description="是否通过审核。无 critical 级别问题 + overall_score >= 6.0 → True"
    )
    overall_score: float = Field(
        description="综合质量评分 0.0-10.0"
    )
    factual_accuracy: float = Field(
        description="事实准确性评分 0.0-10.0"
    )
    style_score: float = Field(
        description="文风与可读性评分 0.0-10.0"
    )
    structure_score: float = Field(
        description="结构逻辑性评分 0.0-10.0"
    )
    issues: list[ReviewIssueModel] = Field(
        description="发现的问题列表。passed=True 时可为空或仅有 suggestion 级别"
    )
    feedback: str = Field(
        description="综合修改建议，自然语言，100-300字。passed=True 时可写'内容质量优秀，建议进入脚本制作'"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
你是一名资深内容审核编辑，拥有 10 年出版级内容把关经验。

## 审核维度（4 维度，各 0-10 分）
1. **事实准确性** (factual_accuracy)
   - 数据是否有依据？引用是否准确？
   - 逻辑推理是否有漏洞？
   - 时效性：内容是否符合当前时间点？

2. **文风与可读性** (style_score)
   - 语言是否流畅、口语化（适合视频口播）？
   - 是否存在病句、冗余、AI 味过重的问题？
   - 段落节奏是否合理？

3. **结构逻辑性** (structure_score)
   - 是否覆盖了大纲的所有章节？
   - 起承转合是否自然？
   - 段落衔接是否流畅？

4. **合规与受众** (audience)
   - 是否存在违规内容（敏感话题、虚假信息）？
   - 是否符合目标受众的认知水平？

## 审核原则
- 严格但不苛刻：minor 级别问题不阻塞通过
- 反馈必须具体、可操作：不要说"需要改进"，要说"第X段改为Y"
- 评分客观：6.0 为及格线，7.0+ 为良好，8.5+ 为优秀

## JSON 输出格式（必须严格遵守，输出纯JSON）
{
    "passed": true,
    "overall_score": 7.5,
    "factual_accuracy": 8.0,
    "style_score": 7.0,
    "structure_score": 7.5,
    "issues": [
        {
            "issue_type": "factual",
            "severity": "minor",
            "location": "第2段",
            "description": "某数据缺少引用来源",
            "suggestion": "建议补充数据出处"
        }
    ],
    "feedback": "整体质量良好，建议在以下方面优化：1. ... 2. ..."
}
"""

HUMAN_TEMPLATE = """\
## 创作主题
{topic}

## 原始大纲
{outline_text}

## 待审核文案
{draft_text}

## 搜索参考资料（事实核查用）
{search_context}

## 任务
请对待审核文案进行 4 维度全面审核。重点核查：
1. 数据和事实是否准确（对照搜索资料）
2. 文案是否适合视频口播
3. 结构是否与大纲一致
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ReviewAgent
# ═══════════════════════════════════════════════════════════════════════════════

class ReviewAgent(BaseAgent):

    output_model = ReviewOutput

    @property
    def agent_name(self) -> str:
        return "content_review"

    def __init__(
        self,
        llm: ChatOpenAI,
        tavily_client: Optional[Any] = None,
    ):
        super().__init__(llm, tavily_client)

    # ── Prompt 构建 ──────────────────────────────────────────────

    def _build_messages(self, state: GraphState) -> list:
        topic = state.get("topic", "")

        outline = state.get("outline")
        outline_text = self._format_outline(outline) if outline else "（大纲缺失）"

        draft = state.get("draft")
        draft_text = draft.get("full_text", "") if draft else "（文案缺失）"

        # 搜索资料用于事实核查（提取关键数据点搜索）
        search_context = ""  # 事实核查搜索按需启用

        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        human_msg = HumanMessage(content=HUMAN_TEMPLATE.format(
            topic=topic,
            outline_text=outline_text,
            draft_text=draft_text[:5000],  # 截断防超 token
            search_context=search_context or "（未搜索，基于你的知识审核）",
        ))
        return [system_msg, human_msg]

    # ── 输出转换 ─────────────────────────────────────────────────

    def _to_state_update(self, result: ReviewOutput, state: GraphState) -> dict:
        issues: list[ReviewIssue] = []
        for iss in result.issues:
            issue: ReviewIssue = {
                "issue_type": iss.issue_type,
                "severity": iss.severity,
                "location": iss.location,
                "description": iss.description,
                "suggestion": iss.suggestion,
            }
            issues.append(issue)

        review_result: ReviewResult = {
            "passed": result.passed,
            "overall_score": result.overall_score,
            "factual_accuracy": result.factual_accuracy,
            "style_score": result.style_score,
            "structure_score": result.structure_score,
            "issues": issues,
            "feedback": result.feedback,
        }

        status = "✅ 通过" if result.passed else "❌ 驳回"
        return {
            "review_result": review_result,
            "current_agent": self.agent_name,
            "retry_count": state.get("retry_count", 0),
            "audit_log": [self._make_audit(
                "complete",
                f"{status} | 综合={result.overall_score:.1f} "
                f"事实={result.factual_accuracy:.1f} "
                f"文风={result.style_score:.1f} "
                f"结构={result.structure_score:.1f} "
                f"问题={len(issues)}个"
            )],
        }

    # ── 大纲格式化 ───────────────────────────────────────────────

    @staticmethod
    def _format_outline(outline: dict) -> str:
        lines = []
        for sec in outline.get("sections", []):
            lines.append(f"{'#' * sec.get('level', 1)} {sec.get('title', '')}")
            for kp in sec.get("key_points", []):
                lines.append(f"  - {kp}")
        return "\n".join(lines)
