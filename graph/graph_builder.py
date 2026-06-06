"""
graph_builder.py — LangGraph StateGraph 构建器
================================================

组装 Multi-Agent 内容生成流水线，包含 4 个 Agent 节点、
条件边质量门控、自动重试与 Redis 检查点。

流水线拓扑：
  START
    ↓
  topic_analysis (TopicAgent)
    ↓ [check_topic_score: score≥6? → 通过 / 重试 / 终止]
  content_generate (WriterAgent)
    ↓ [check_content_quality: word_count≥300? → 通过 / 重试 / 终止]
  content_review (ReviewAgent)
    ↓ [check_review_result: passed? → 生成脚本 / 回退修改]
  video_script (VideoScriptAgent)
    ↓ [check_script_complete: scenes非空? → 完成 / 重试]
  END

条件边覆盖：
  - 正常通过 → 进入下一阶段
  - 质量不足 → 回退当前节点重试（retry_count + 1）
  - 超过重试上限 → 设置 error，路由到 END
  - 审核驳回 → 携带 feedback 回退到 content_generate
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Any

from langgraph.graph import StateGraph, START, END

from graph.state import (
    GraphState,
    Outline,
    Draft,
    ReviewResult,
    FinalScript,
    AuditEntry,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 条件边路由函数（Conditional Edges）
# ═══════════════════════════════════════════════════════════════════════════════
# 每个函数签名: (state: GraphState) -> str
# 返回值: 下一个节点的名称字符串
# LangGraph 根据返回值查路由表决定下一跳
# ═══════════════════════════════════════════════════════════════════════════════

def check_topic_score(state: GraphState) -> str:
    """
    选题评分门控 —— topic_analysis 节点之后执行。

    判断逻辑:
      1. error 非空 → 直接终止
      2. topic_score >= 6.0 → 通过，进入内容生成
      3. retry_count < max_retries → 回退 topic_analysis 重试
      4. 否则 → 终止并设置 error

    评分含义:
      0-4:  冷门/高风险话题，建议放弃
      4-6:  一般话题，可尝试但需优化角度
      6-8:  优质话题，有爆款潜力
      8-10: 顶级话题，高热度+低竞争

    Returns:
        "content_generate" | "topic_analysis" | "END"
    """
    # 最高优先级：有错误直接终止
    if state.get("error"):
        logger.warning(f"[Router] 检测到上游错误，终止: {state['error']}")
        return END

    score = state.get("topic_score", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if score >= 6.0:
        logger.info(f"[Router] 选题评分 {score:.1f} ≥ 6.0，进入内容生成")
        return "content_generate"

    if retry_count < max_retries:
        logger.warning(
            f"[Router] 选题评分 {score:.1f} < 6.0，"
            f"第 {retry_count + 1}/{max_retries} 次重试"
        )
        return "topic_analysis"

    # 超过最大重试次数，放弃
    logger.error(f"[Router] 选题评分 {score:.1f}，已重试 {max_retries} 次，终止流水线")
    return END


def check_content_quality(state: GraphState) -> str:
    """
    内容质量门控 —— content_generate 节点之后执行。

    判断逻辑:
      1. error 非空 → 终止
      2. outline 为空（异常） → 终止
      3. draft.word_count >= 800 且 paragraphs 非空 → 通过，进入审核
      4. retry_count < max_retries → 回退 content_generate 重试
      5. 否则 → 终止

    注意：若 review_result 有 feedback（审核驳回重写场景），
         重试时 WriterAgent 应读取 feedback 修正内容。

    Returns:
        "content_review" | "content_generate" | "END"
    """
    if state.get("error"):
        logger.warning(f"[Router] 检测到错误，终止: {state['error']}")
        return END

    outline = state.get("outline")
    if outline is None:
        logger.error("[Router] outline 为空，无法继续")
        return END

    draft = state.get("draft")
    if draft and draft.get("word_count", 0) >= 800 and draft.get("paragraphs"):
        wc = draft["word_count"]
        logger.info(f"[Router] 内容质量通过 → {wc} 字，进入审核")
        return "content_review"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        logger.warning(
            f"[Router] 内容质量不足（字数或段落缺失），"
            f"第 {retry_count + 1}/{max_retries} 次重试"
        )
        return "content_generate"

    logger.error(f"[Router] 内容生成已达最大重试 {max_retries} 次，终止")
    return END


def check_review_result(state: GraphState) -> str:
    """
    审核结果路由 —— content_review 节点之后执行。

    判断逻辑:
      1. error 非空 → 终止
      2. review_result 为空（异常） → 终止
      3. passed = True → 通过，进入视频脚本
      4. retry_count < max_retries → 回退 content_generate（带 feedback）
      5. 否则 → 终止

    回退时 feedback 由 content_generate 读取，用于修正内容。

    Returns:
        "video_script" | "content_generate" | "END"
    """
    if state.get("error"):
        logger.warning(f"[Router] 检测到错误，终止: {state['error']}")
        return END

    review = state.get("review_result")
    if review is None:
        logger.error("[Router] review_result 为空，无法路由")
        return END

    if review.get("passed"):
        score = review.get("overall_score", 0)
        logger.info(f"[Router] 审核通过（{score:.1f}分），进入视频脚本生成")
        return "video_script"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        feedback = review.get("feedback", "（无具体反馈）")
        logger.warning(
            f"[Router] 审核驳回，第 {retry_count + 1}/{max_retries} 次回退修改"
            f"\n  feedback: {feedback[:100]}..."
        )
        return "content_generate"

    logger.error(f"[Router] 审核 {max_retries} 次仍未通过，终止")
    return END


def check_script_complete(state: GraphState) -> str:
    """
    脚本完整性检查 —— video_script 节点之后执行。

    判断逻辑:
      1. error 非空 → 终止
      2. final_script.scenes 非空 → 成功完成，路由到 END
      3. retry_count < max_retries → 回退 video_script 重试
      4. 否则 → 终止

    Returns:
        END | "video_script"
    """
    if state.get("error"):
        logger.warning(f"[Router] 检测到错误，终止: {state['error']}")
        return END

    script = state.get("final_script")
    if script and script.get("scenes"):
        scene_count = len(script["scenes"])
        logger.info(f"[Router] 脚本生成完成 → {scene_count} 个分镜，流水线结束")
        return END

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        logger.warning(
            f"[Router] 脚本不完整（scenes 为空），"
            f"第 {retry_count + 1}/{max_retries} 次重试"
        )
        return "video_script"

    logger.error(f"[Router] 脚本生成已达最大重试 {max_retries} 次，终止")
    return END


# ═══════════════════════════════════════════════════════════════════════════════
# 条件边路由表
# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph.add_conditional_edges(source, condition, path_map)
# path_map: 将 condition 函数的返回值映射到目标节点名

TOPIC_SCORE_ROUTES = {
    "content_generate": "content_generate",
    "topic_analysis": "topic_analysis",
    END: END,
}

CONTENT_QUALITY_ROUTES = {
    "content_review": "content_review",
    "content_generate": "content_generate",
    END: END,
}

REVIEW_RESULT_ROUTES = {
    "video_script": "video_script",
    "content_generate": "content_generate",
    END: END,
}

SCRIPT_COMPLETE_ROUTES = {
    "video_script": "video_script",
    END: END,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 占位 Agent 节点函数（用于尚未实现的 Agent，保持图结构完整）
# ═══════════════════════════════════════════════════════════════════════════════

async def _writer_agent(state: GraphState) -> dict:
    """
    [占位] WriterAgent — 内容生成节点。

    TODO: 替换为 agents/writer_agent.py 的真实实现。
    当前行为: 透传 state，打印警告日志。

    预期输入: state.outline
    预期输出: state.draft (Draft TypedDict)
    """
    logger.warning(
        "[Placeholder] WriterAgent 尚未实现，请在 agents/writer_agent.py 中完成"
    )
    return {
        "current_agent": "content_generate",
        "retry_count": state.get("retry_count", 0),
        "audit_log": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "content_generate",
            "action": "start",
            "detail": "[占位] 内容生成节点未实现，跳过",
        }],
    }


async def _review_agent(state: GraphState) -> dict:
    """
    [占位] ReviewAgent — 内容审核节点。

    TODO: 替换为 agents/review_agent.py 的真实实现。
    当前行为: 透传 state，打印警告日志。

    预期输入: state.draft + state.outline
    预期输出: state.review_result (ReviewResult TypedDict)
    """
    logger.warning(
        "[Placeholder] ReviewAgent 尚未实现，请在 agents/review_agent.py 中完成"
    )
    return {
        "current_agent": "content_review",
        "retry_count": state.get("retry_count", 0),
        "audit_log": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "content_review",
            "action": "start",
            "detail": "[占位] 内容审核节点未实现，跳过",
        }],
    }


async def _video_script_agent(state: GraphState) -> dict:
    """
    [占位] VideoScriptAgent — 视频脚本生成节点。

    TODO: 替换为 agents/video_script_agent.py 的真实实现。
    当前行为: 透传 state，打印警告日志。

    预期输入: state.draft + state.review_result
    预期输出: state.final_script (FinalScript TypedDict)
    """
    logger.warning(
        "[Placeholder] VideoScriptAgent 尚未实现，请在 agents/video_script_agent.py 中完成"
    )
    return {
        "current_agent": "video_script",
        "retry_count": state.get("retry_count", 0),
        "audit_log": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "video_script",
            "action": "start",
            "detail": "[占位] 视频脚本节点未实现，跳过",
        }],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 节点包装器 — 统一的重试计数器管理
# ═══════════════════════════════════════════════════════════════════════════════
# 每个 Agent 执行后需要维护 retry_count:
#   - 成功: 重置为 0（下一阶段从零开始）
#   - 失败/重试: 递增（由条件边函数读取后+1 注入）
#
# 包装器负责在 agent 返回后调整 retry_count。

def _wrap_node(node_func):
    """
    节点包装器装饰逻辑说明（已内联到 build_graph 的 node 定义中）：

    原始行为：node(state) → dict
    增强行为：
      1. 调用 node(state)
      2. 若返回 error → retry_count += 1
      3. 若正常 → retry_count = 0 (进入下一阶段时重置)
    """
    return node_func


# ═══════════════════════════════════════════════════════════════════════════════
# 图构建工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(
    llm: Any,
    tavily_client: Optional[Any] = None,
    redis_checkpointer: Optional[Any] = None,
    topic_agent: Optional[Any] = None,
    writer_agent: Optional[Any] = None,
    review_agent: Optional[Any] = None,
    video_script_agent: Optional[Any] = None,
) -> StateGraph:
    """
    构建并编译 Multi-Agent 内容生成流水线的 StateGraph。

    Args:
        llm: LangChain ChatOpenAI 实例（配置好 DeepSeek base_url/api_key），
             所有 Agent 共享同一个 LLM 客户端。
        tavily_client: Tavily 搜索客户端（可选），传入 None 则 TopicAgent 使用纯 LLM 模式。
        redis_checkpointer: RedisSaver 实例（可选），传入则启用 Redis 检查点持久化。
        topic_agent: 自定义 TopicAgent 实例（可选），不传则自动创建。
        writer_agent: 自定义 WriterAgent 实例（可选），不传则使用占位节点。
        review_agent: 自定义 ReviewAgent 实例（可选），不传则使用占位节点。
        video_script_agent: 自定义 VideoScriptAgent 实例（可选），不传则使用占位节点。

    Returns:
        CompiledStateGraph: 已编译的 LangGraph 工作流，可直接调用 .ainvoke() 执行。

    使用示例:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(base_url="https://api.deepseek.com", api_key="sk-...")
        >>> graph = build_graph(llm=llm)
        >>> result = await graph.ainvoke({"topic": "AI Agent 发展趋势", ...})
    """
    # ── 1. 初始化 Agent ────────────────────────────────────────────
    if topic_agent is not None:
        ta = topic_agent
    else:
        from agents.topic_agent import TopicAgent
        ta = TopicAgent(llm=llm, tavily_client=tavily_client)
        logger.info("[GraphBuilder] TopicAgent 已初始化")

    if writer_agent is not None:
        wa = writer_agent
    else:
        from agents.writer_agent import WriterAgent
        wa = WriterAgent(llm=llm, tavily_client=tavily_client)
        logger.info("[GraphBuilder] WriterAgent 已初始化")

    if review_agent is not None:
        ra = review_agent
    else:
        from agents.review_agent import ReviewAgent
        ra = ReviewAgent(llm=llm, tavily_client=tavily_client)
        logger.info("[GraphBuilder] ReviewAgent 已初始化")

    if video_script_agent is not None:
        vsa = video_script_agent
    else:
        from agents.video_script_agent import VideoScriptAgent
        vsa = VideoScriptAgent(llm=llm)
        logger.info("[GraphBuilder] VideoScriptAgent 已初始化")

    # ── 2. 创建 StateGraph ────────────────────────────────────────
    workflow = StateGraph(GraphState)

    # ── 3. 添加 4 个 Agent 节点 ───────────────────────────────────
    # 节点名         节点函数          对应 Agent           阶段产出
    # ────────────  ────────────────  ──────────────────   ────────────
    workflow.add_node("topic_analysis", ta)        # TopicAgent         → outline
    workflow.add_node("content_generate", wa)      # WriterAgent        → draft
    workflow.add_node("content_review", ra)        # ReviewAgent        → review_result
    workflow.add_node("video_script", vsa)         # VideoScriptAgent   → final_script

    logger.info(
        "[GraphBuilder] 已添加 4 个节点: "
        "topic_analysis, content_generate, content_review, video_script"
    )

    # ── 4. 添加边（入口 + 条件边 + 回退边） ──────────────────────

    # 4a. START → topic_analysis（入口边，无条件）
    workflow.add_edge(START, "topic_analysis")

    # 4b. topic_analysis → 条件路由 → {content_generate | topic_analysis(重试) | END(终止)}
    workflow.add_conditional_edges(
        "topic_analysis",
        check_topic_score,
        TOPIC_SCORE_ROUTES,
    )

    # 4c. content_generate → 条件路由 → {content_review | content_generate(重试) | END(终止)}
    workflow.add_conditional_edges(
        "content_generate",
        check_content_quality,
        CONTENT_QUALITY_ROUTES,
    )

    # 4d. content_review → 条件路由 → {video_script | content_generate(回退修改) | END(终止)}
    workflow.add_conditional_edges(
        "content_review",
        check_review_result,
        REVIEW_RESULT_ROUTES,
    )

    # 4e. video_script → 条件路由 → {END(完成) | video_script(重试)}
    workflow.add_conditional_edges(
        "video_script",
        check_script_complete,
        SCRIPT_COMPLETE_ROUTES,
    )

    logger.info("[GraphBuilder] 已添加条件边: START → topic → content → review → script → END")

    # ── 5. 编译图 ──────────────────────────────────────────────────
    if redis_checkpointer is not None:
        compiled = workflow.compile(checkpointer=redis_checkpointer)
        logger.info("[GraphBuilder] 图编译完成，已启用 Redis Checkpointer")
    else:
        compiled = workflow.compile()
        logger.info("[GraphBuilder] 图编译完成（无持久化）")

    return compiled


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷函数：带默认 LLM 配置的快速构建
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph_from_env(
    tavily_client: Optional[Any] = None,
    redis_checkpointer: Optional[Any] = None,
) -> StateGraph:
    """
    从环境变量读取 DeepSeek 配置，一键构建图。
    适合快速启动和本地开发。

    读取的环境变量（在 .env 中配置）：
      DEEPSEEK_API_KEY   — API 密钥
      DEEPSEEK_BASE_URL  — API 地址（默认 https://api.deepseek.com）
      LLM_MODEL          — 模型名（默认 deepseek-v3）
      LLM_TEMPERATURE    — 温度（默认 0.1）
      LLM_MAX_TOKENS     — 最大输出 Token（默认 4096）

    Returns:
        CompiledStateGraph
    """
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-v3"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
    )

    logger.info(
        f"[GraphBuilder] LLM: {os.getenv('LLM_MODEL', 'deepseek-v3')} "
        f"@ {os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')}"
    )

    return build_graph(
        llm=llm,
        tavily_client=tavily_client,
        redis_checkpointer=redis_checkpointer,
    )
