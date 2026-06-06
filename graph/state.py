"""
LangGraph 工作流 State 定义
============================

定义贯穿整个 Multi-Agent 内容生成流水线的全局状态对象。
每个 Graph 节点读取 State，返回部分字段的更新（Partial Update），
LangGraph 自动执行浅合并（shallow merge），特殊字段通过 Annotated reducer 控制合并策略。

State 流转路径：
  input_processor → topic_analysis → content_generate → content_review → video_script → END
                     ↑                                    ↓                ↓
                     └────────── 条件回退（重试）──────────┘                │
                                                                           │
                                                                           └──→ END
"""

from typing import TypedDict, Annotated, Optional, Union
from operator import add


# ═══════════════════════════════════════════════════════════════════════════════
# 嵌套子结构 TypedDict — 每个复杂字段的类型定义
# ═══════════════════════════════════════════════════════════════════════════════

class OutlineItem(TypedDict):
    """大纲单项"""
    level: int                          # 层级（1=一级标题, 2=二级标题, 3=三级标题）
    title: str                          # 标题文本
    key_points: list[str]               # 该节核心要点（3-5个）


class Outline(TypedDict):
    """内容大纲"""
    sections: list[OutlineItem]         # 大纲结构
    estimated_duration: int             # 预计总时长（秒）
    estimated_word_count: int           # 预计总字数
    logic_flow: str                     # 逻辑脉络简述（起承转合）


class Draft(TypedDict):
    """初稿内容"""
    full_text: str                      # 完整文案正文（Markdown格式）
    word_count: int                     # 实际字数
    paragraphs: list[str]               # 按段落拆分（用于分镜映射）
    seo_keywords: list[str]             # SEO 关键词/标签
    references: list[str]               # 引用的信息来源 URL 列表


class ReviewIssue(TypedDict):
    """审核发现的具体问题"""
    issue_type: str                     # 问题类型: "factual" | "style" | "structure" | "legal" | "audience"
    severity: str                       # 严重程度: "critical" | "major" | "minor" | "suggestion"
    location: str                       # 问题位置描述（如"第3段第2句"）
    description: str                    # 问题具体描述
    suggestion: str                     # 修改建议


class ReviewResult(TypedDict):
    """审核结果"""
    passed: bool                        # 是否通过审核（True=进入视频脚本, False=回退修改）
    overall_score: float                # 综合质量评分（0.0-10.0）
    factual_accuracy: float             # 事实准确性评分
    style_score: float                  # 文风/可读性评分
    structure_score: float              # 结构合理性评分
    issues: list[ReviewIssue]           # 发现的问题列表
    feedback: str                       # 综合反馈意见（自然语言，传递回 content_generate）


class Scene(TypedDict):
    """单个分镜"""
    scene_id: int                       # 分镜序号（从1开始）
    duration_sec: int                   # 该分镜时长（秒）
    visual_description: str             # 画面描述（给画师/AI绘图/AE合成的提示）
    narration: str                      # 旁白/配音台词
    subtitles: str                      # 字幕文本
    transition: str                     # 转场效果: "cut" | "fade" | "dissolve" | "wipe"
    background_music: str               # 该分镜的配乐/音效说明


class FinalScript(TypedDict):
    """最终视频脚本"""
    title: str                          # 视频标题
    total_duration_sec: int             # 预计总时长（秒）
    target_platform: str                # 目标平台: "bilibili" | "douyin" | "youtube" | "通用"
    style: str                          # 视频风格: "口播" | "纪录片" | "动画" | "混合"
    scenes: list[Scene]                 # 分镜列表
    b_roll_suggestions: list[str]       # B-roll 素材建议
    music_style: str                    # 整体配乐风格
    call_to_action: str                 # 片尾 CTA（关注/点赞/评论引导语）


class AuditEntry(TypedDict):
    """审计日志条目（每条记录一次状态变更）"""
    timestamp: str                      # ISO 8601 时间戳
    agent: str                          # 执行节点名称（topic_analysis / content_generate / content_review / video_script）
    action: str                         # 动作: "start" | "complete" | "error" | "retry"
    detail: str                         # 简要描述（如 "选题分析完成，评分 8.5，进入内容生成"）


# ═══════════════════════════════════════════════════════════════════════════════
# 主 State — GraphState
# ═══════════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    """
    Multi-Agent 内容生成流水线的全局状态。

    使用 TypedDict 定义，LangGraph 的 StateGraph 基于此类型进行节点间数据传递。
    每个节点函数返回 dict（部分字段），LangGraph 自动浅合并到当前 State。

    特殊字段说明：
      - audit_log: 使用 Annotated[list, add] reducer，强制追加而非覆盖
      - error: Optional[str]，为 None 时表示无错误
      - retry_count: 跨节点共享，用于条件边判断是否超过最大重试次数
    """

    # ── 阶段 0: 输入 ─────────────────────────────────────────────
    topic: str
    """
    用户输入的创作主题（原始文本）。

    示例:
      "2025年AI Agent技术发展趋势与行业落地案例"
      "如何用Python构建一个RAG系统"

    由 POST /api/v1/workflow/run 传入，input_processor 节点写入 State。
    全局只读，各 Agent 从此字段提取上下文。
    """

    task_id: str
    """
    任务唯一标识符。

    格式: ULID（大写，26字符，如 "01HX3K9M8P2Q7R5V0W4Y6Z8A1B"）
    用途: 关联 API 响应、Redis 缓存 Key、审计日志主键、请求追踪。
    生成时机: workflow_service.run() 接收到请求后立即生成。
    """

    # ── 阶段 1: 选题分析 ─────────────────────────────────────────
    outline: Optional[Outline]
    """
    选题分析 + 大纲规划的结果。

    生成者: topic_analysis Agent
    消费者: content_generate Agent（作为内容生成的骨架输入）
    设置为 Optional: 流水线启动时为空，topic_analysis 节点执行后填充。

    包含: 结构化大纲（层级标题+要点）、预计时长/字数、逻辑脉络。
    """

    topic_score: float
    """
    选题可行性综合评分 0.0-10.0。

    生成者: topic_analysis Agent
    消费者: check_topic_score 条件边（评分 ≥ 6.0 通过，否则重试）
    默认值: 0.0（未评分状态，条件边会判定为不通过）
    """

    keywords: list[str]
    """
    核心关键词列表。

    生成者: topic_analysis Agent
    消费者: content_generate Agent（作为 SEO 标签的候选词库）
    默认值: 空列表
    """

    # ── 阶段 2: 内容生成 ─────────────────────────────────────────
    draft: Optional[Draft]
    """
    基于 outline 生成的初稿内容。

    生成者: content_generate Agent
    消费者: content_review Agent（审核对象）
    回退时: 若审核不通过，content_generate 基于 feedback 重新生成并覆盖此字段。

    包含: 完整文案（Markdown）、分段列表、SEO 关键词、参考来源。
    """

    # ── 阶段 3: 内容审核 ─────────────────────────────────────────
    review_result: Optional[ReviewResult]
    """
    对 draft 的多维度审核结果。

    生成者: content_review Agent
    消费者: 条件边 check_review_result（决定路由）
    回退时: 若 passed=False，feedback 字段作为提示词注入 content_generate 的重试上下文。

    包含: 通过标记、4维度评分、问题列表、综合反馈。
    """

    # ── 阶段 4: 视频脚本 ─────────────────────────────────────────
    final_script: Optional[FinalScript]
    """
    最终输出的视频脚本（整个流水线的交付物）。

    生成者: video_script Agent
    消费者: API 响应（GET /workflow/{task_id}/result 直接返回此字段）
    终态标志: 此字段非空 + error 为空 → 流水线成功完成。

    包含: 标题、分镜列表、旁白、字幕、B-roll、配乐、CTA。
    """

    # ── 执行控制 ─────────────────────────────────────────────────
    current_agent: str
    """
    当前正在执行的 Agent 节点名称。

    合法值:
      "input_processor"   — 输入预处理
      "topic_analysis"    — 选题分析
      "content_generate"  — 内容生成
      "content_review"    — 内容审核
      "video_script"      — 视频脚本生成
      "END"               — 流水线结束

    用途:
      1. API status 端点返回当前进度给前端轮询展示
      2. 条件边路由判断"从哪个节点来"
      3. Redis checkpoint 恢复时定位断点
      4. 审计日志的 agent 字段
    """

    retry_count: int
    """
    当前节点已重试次数（跨节点共享）。

    初始值: 0
    递增时机: 条件边判定"回退重试"时 +1
    重置时机: 成功进入下一节点时重置为 0
    上限判断: retry_count >= max_retries → 中止流水线，设置 error
    """

    max_retries: int
    """
    每个节点的最大重试次数上限。

    默认值: 3
    可配置: 通过 API 请求参数覆盖（例如要求更高容错时可设为 5）
    用途: 条件边函数读取此值判断是否应该终止而非继续重试。
    """

    error: Optional[str]
    """
    错误信息。

    为 None 时: 流水线正常运行
    非空时:   流水线异常终止，内容为人类可读的错误描述

    触发场景:
      - retry_count 超过 max_retries（"选题分析连续3次评分不足"）
      - LLM 调用异常（"DeepSeek API 超时"）
      - JSON 解析失败（"Agent 返回格式错误，无法解析"）
      - Tavily 搜索异常（"搜索服务不可用"）

    消费者:
      - API 响应中作为 error_message 返回
      - 条件边 check_for_error 检测此字段 → 路由到 END
    """

    # ── 审计追踪 ─────────────────────────────────────────────────
    audit_log: Annotated[list[AuditEntry], add]
    """
    全流程审计日志，记录每一步状态变更。

    使用 Annotated[list, add] reducer:
      节点返回 {"audit_log": [new_entry]} 时
      → LangGraph 自动 append 到现有列表，而非覆盖
      → 无需在节点中手动读取旧列表再拼接

    记录时机:
      - 每个 Agent 启动时:    action="start"
      - 每个 Agent 完成时:    action="complete"
      - 条件边判定重试时:    action="retry"
      - 发生错误时:          action="error"

    持久化: Redis List（key: audit:{task_id}）
    用途:   事后追溯、调试、成本核算（统计每步耗时和 Token 消耗）
    """
