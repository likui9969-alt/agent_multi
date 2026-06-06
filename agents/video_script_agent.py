"""
agents/video_script_agent.py — 视频脚本生成 Agent
==================================================

LangGraph 节点：video_script
输入: State.draft + State.review_result + State.topic
输出: State.final_script（标题 + 分镜列表 + 旁白/字幕 + B-roll + 配乐 + CTA）

职责:
  - 将文字文案转化为可视化分镜脚本
  - 按段落拆分场景，分配时长
  - 为每个场景设计画面描述、旁白、字幕、转场
  - 输出 B-roll 素材建议和整体配乐风格
  - 生成片尾 Call-to-Action
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from graph.state import GraphState, FinalScript, Scene, AuditEntry
from agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic 输出模型
# ═══════════════════════════════════════════════════════════════════════════════

class SceneModel(BaseModel):
    """分镜单项 — 对应 Scene TypedDict"""
    scene_id: int = Field(description="分镜序号，从 1 开始递增")
    duration_sec: int = Field(description="该分镜时长（秒），建议 10-60 秒")
    visual_description: str = Field(
        description="画面描述：构图、主体、色调、动作。供画师/AI绘图/AE合成使用"
    )
    narration: str = Field(
        description="旁白/配音台词，口语化，与画面同步"
    )
    subtitles: str = Field(
        description="字幕文本，精简版旁白，每行不超过 20 字"
    )
    transition: str = Field(
        description="转场效果: cut(硬切) | fade(淡入淡出) | dissolve(叠化) | wipe(划像)"
    )
    background_music: str = Field(
        description="该分镜的配乐/音效说明，如'轻快电子乐渐入'、'悬疑弦乐'"
    )


class VideoScriptOutput(BaseModel):
    """VideoScriptAgent 结构化输出 — 映射到 State.final_script"""
    title: str = Field(
        description="视频最终标题，吸引点击，不超过 40 字"
    )
    total_duration_sec: int = Field(
        description="预计视频总时长（秒），所有分镜时长之和"
    )
    target_platform: str = Field(
        description="最适合的发布平台: bilibili | douyin | youtube | 通用"
    )
    style: str = Field(
        description="视频风格: 口播 | 纪录片 | 动画 | 混合"
    )
    scenes: list[SceneModel] = Field(
        description="分镜列表，至少 5 个分镜，覆盖文案所有段落"
    )
    b_roll_suggestions: list[str] = Field(
        description="B-roll 素材建议，5-10 条，如'城市航拍延时'、'代码编辑器屏幕录制'"
    )
    music_style: str = Field(
        description="整体配乐风格描述，如'科技感电子音乐，中速节奏'"
    )
    call_to_action: str = Field(
        description="片尾 CTA，引导观众点赞/关注/评论/分享，不超过 50 字"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
你是一名资深视频制作导演，擅长将文字文案转化为专业的视频分镜脚本。

## 你的能力
- 文字 → 画面：将抽象描述转化为具体的视觉画面
- 节奏控制：合理分配每个分镜的时长
- 视听设计：为每个场景匹配音乐、音效、转场
- 平台适配：根据内容特点推荐最佳发布平台

## 分镜设计原则
1. 每个自然段对应 1-3 个分镜
2. 开篇 3 秒抓眼球（冲击力画面 + 钩子文案）
3. 分镜时长 10-60 秒，总时长控制在文案总字数 / 3.5 秒左右
4. 旁白 = 文案原句（口语化微调），字幕 = 旁白精简版
5. 转场默认 cut，情感转折处用 fade/dissolve
6. B-roll 素材建议具体可搜索，如"4K 城市夜景延时"

## 输出要求
- 分镜数量 ≥ 5 个
- 每个分镜的 narration 必须来自原文案对应段落
- visual_description 具体到构图、色调、主体动作

## JSON 输出格式（必须严格遵守，输出纯JSON）
{
    "title": "视频标题",
    "total_duration_sec": 300,
    "target_platform": "bilibili",
    "style": "口播",
    "scenes": [
        {
            "scene_id": 1,
            "duration_sec": 30,
            "visual_description": "主持人正面出镜，科技感背景",
            "narration": "大家好，今天我们来聊聊...",
            "subtitles": "今天我们来聊聊...",
            "transition": "cut",
            "background_music": "轻快电子乐渐入"
        }
    ],
    "b_roll_suggestions": ["城市航拍延时", "数据图表动画"],
    "music_style": "科技感电子音乐，中速节奏",
    "call_to_action": "如果觉得有用，记得点赞关注哦！"
}
"""

HUMAN_TEMPLATE = """\
## 视频主题
{topic}

## 完整文案（分镜的依据）
{draft_text}

## 审核评分
综合 {overall}/10 | 事实准确性 {factual}/10 | 文风 {style}/10 | 结构 {structure}/10

## 任务
请将上述文案转化为完整的视频分镜脚本。
每个文案段落至少对应 1 个分镜，输出结构化 JSON。
"""


# ═══════════════════════════════════════════════════════════════════════════════
# VideoScriptAgent
# ═══════════════════════════════════════════════════════════════════════════════

class VideoScriptAgent(BaseAgent):

    output_model = VideoScriptOutput

    @property
    def agent_name(self) -> str:
        return "video_script"

    def __init__(
        self,
        llm: ChatOpenAI,
        tavily_client: Optional[Any] = None,
    ):
        super().__init__(llm, tavily_client)

    # ── Prompt 构建 ──────────────────────────────────────────────

    def _build_messages(self, state: GraphState) -> list:
        topic = state.get("topic", "")

        draft = state.get("draft")
        draft_text = draft.get("full_text", "") if draft else "（文案缺失）"

        review = state.get("review_result")
        overall = review.get("overall_score", 0) if review else 0
        factual = review.get("factual_accuracy", 0) if review else 0
        style = review.get("style_score", 0) if review else 0
        structure = review.get("structure_score", 0) if review else 0

        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        human_msg = HumanMessage(content=HUMAN_TEMPLATE.format(
            topic=topic,
            draft_text=draft_text[:6000],
            overall=overall,
            factual=factual,
            style=style,
            structure=structure,
        ))
        return [system_msg, human_msg]

    # ── 输出转换 ─────────────────────────────────────────────────

    def _to_state_update(self, result: VideoScriptOutput, state: GraphState) -> dict:
        scenes: list[Scene] = []
        for sc in result.scenes:
            scene: Scene = {
                "scene_id": sc.scene_id,
                "duration_sec": sc.duration_sec,
                "visual_description": sc.visual_description,
                "narration": sc.narration,
                "subtitles": sc.subtitles,
                "transition": sc.transition,
                "background_music": sc.background_music,
            }
            scenes.append(scene)

        final_script: FinalScript = {
            "title": result.title,
            "total_duration_sec": result.total_duration_sec,
            "target_platform": result.target_platform,
            "style": result.style,
            "scenes": scenes,
            "b_roll_suggestions": result.b_roll_suggestions,
            "music_style": result.music_style,
            "call_to_action": result.call_to_action,
        }

        total_dur = result.total_duration_sec
        minutes = total_dur // 60
        seconds = total_dur % 60

        return {
            "final_script": final_script,
            "current_agent": self.agent_name,
            "retry_count": 0,
            "audit_log": [self._make_audit(
                "complete",
                f"标题='{result.title[:30]}' | "
                f"时长={minutes}分{seconds}秒 | "
                f"分镜={len(scenes)}个 | "
                f"平台={result.target_platform} | "
                f"风格={result.style}"
            )],
        }
