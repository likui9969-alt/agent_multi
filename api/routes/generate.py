"""
api/routes/generate.py — POST /generate 路由
=============================================

接收用户主题 → 调用 LangGraph 工作流 → 返回最终视频脚本。

请求:
    POST /generate
    Body: {"topic": "AI Agent", "max_retries": 3}

成功响应 (200):
    {
        "task_id": "01HX3K9M8P2Q7R5V0W4Y6Z8A1B",
        "final_script": "# 视频标题\n\n## 分镜1\n...",
        "topic": "AI Agent"
    }

错误响应:
    400 - 参数校验失败（topic 为空或过长）
    500 - LLM 调用异常 / 工作流执行错误
    504 - 工作流超时（超过 request_timeout 秒）
"""

import logging
import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends

from api.schemas.request import GenerateRequest
from api.schemas.response import GenerateResponse, ErrorResponse
from api.deps import get_graph, get_settings
from graph.state import AuditEntry
from utils.id_gen import generate_task_id
from services.task_service import get_task_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "请求参数错误"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"},
        504: {"model": ErrorResponse, "description": "工作流执行超时"},
    },
    summary="生成视频脚本",
    description="""
    提交创作主题，自动执行「选题分析→内容生成→内容审核→视频脚本」全流程，
    返回 Markdown 格式的最终视频脚本。

    ## 执行流程
    1. **选题分析** — 搜索话题热度，生成结构化大纲
    2. **内容生成** — 基于大纲撰写完整文案
    3. **内容审核** — 多维度质量审查，不通过自动回退修改（最多 3 轮）
    4. **视频脚本** — 文案转分镜脚本（画面/旁白/字幕/配乐）

    ## 注意事项
    - 单次请求最长等待 300 秒（可配置）
    - 若 3 轮审核仍未通过，返回错误信息而非低质量脚本
    """,
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    POST /generate — 提交主题，获取完整视频脚本。

    Args:
        request: GenerateRequest {topic, max_retries?}

    Returns:
        GenerateResponse {task_id, final_script, topic}

    Raises:
        HTTPException(400): 参数校验失败
        HTTPException(500): LLM 或工作流执行异常
        HTTPException(504): 工作流超时
    """
    # ── 1. 生成任务 ID ─────────────────────────────────────────
    task_id = generate_task_id()
    logger.info(
        f"[API] 收到生成请求 task_id={task_id} topic='{request.topic[:80]}...'"
    )

    # ── 1b. Redis: 创建任务（PENDING） ──────────────────────────
    task_svc = get_task_service()
    await task_svc.create(task_id, request.topic)

    # ── 2. 初始化 State ───────────────────────────────────────
    settings = get_settings()
    initial_state = {
        "topic": request.topic,
        "task_id": task_id,
        "current_agent": "START",
        "retry_count": 0,
        "max_retries": request.max_retries,
        # 以下字段由各 Agent 逐步填充，初始为 None
        "outline": None,
        "topic_score": 0.0,
        "keywords": [],
        "draft": None,
        "review_result": None,
        "final_script": None,
        "error": None,
        # audit_log 会自动初始化（Annotated[list, add] 从空列表开始）
    }

    # ── 3. 执行 LangGraph 工作流 ──────────────────────────────
    # Redis: 标记 RUNNING
    await task_svc.mark_running(task_id, "topic_analysis")

    graph = get_graph()

    try:
        # 设置超时（防止 LLM 长时间无响应）
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=settings.request_timeout,
        )
        logger.info(f"[API] 工作流执行完成 task_id={task_id}")

    except asyncio.TimeoutError:
        logger.error(f"[API] 工作流超时 task_id={task_id} timeout={settings.request_timeout}s")
        await task_svc.mark_failed(task_id, f"工作流超时（>{settings.request_timeout}s）")
        raise HTTPException(
            status_code=504,
            detail={
                "error": "工作流执行超时",
                "detail": f"超过 {settings.request_timeout} 秒未完成，请尝试更短的主题或稍后重试",
                "task_id": task_id,
            },
        )
    except Exception as exc:
        logger.exception(f"[API] 工作流执行异常 task_id={task_id}")
        await task_svc.mark_failed(task_id, str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "工作流执行失败",
                "detail": str(exc),
                "task_id": task_id,
            },
        )

    # ── 4. 检查执行结果 ──────────────────────────────────────
    # 优先检查 error 字段（Agent 内部或条件边设置的错误）
    if result.get("error"):
        error_msg = result["error"]
        logger.error(f"[API] 工作流返回错误 task_id={task_id}: {error_msg}")
        await task_svc.mark_failed(task_id, error_msg)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "内容生成失败",
                "detail": error_msg,
                "task_id": task_id,
            },
        )

    # 检查 final_script 是否存在
    final_script = result.get("final_script")
    if final_script is None:
        logger.error(f"[API] final_script 为空 task_id={task_id}")
        await task_svc.mark_failed(task_id, "工作流完成但未产出 final_script")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "脚本生成失败",
                "detail": "工作流完成但未产出 final_script",
                "task_id": task_id,
            },
        )

    # ── 5. 格式化最终脚本为 Markdown ─────────────────────────
    markdown_output = _format_script_markdown(final_script)

    # ── 5b. Redis: 标记 COMPLETED + 保存结果 ─────────────────
    await task_svc.mark_completed(task_id, markdown_output)

    # ── 6. 返回 ──────────────────────────────────────────────
    logger.info(
        f"[API] 请求完成 task_id={task_id} "
        f"scenes={len(final_script.get('scenes', []))} "
        f"duration={final_script.get('total_duration_sec', 0)}s"
    )
    return GenerateResponse(
        task_id=task_id,
        final_script=markdown_output,
        topic=request.topic,
    )


# ══════════════════════════════════════════════════════════════════════
# 格式化工具函数
# ══════════════════════════════════════════════════════════════════════

def _format_script_markdown(script: dict) -> str:
    """
    将 FinalScript TypedDict 序列化为可读的 Markdown 字符串。

    输入: GraphState.final_script (dict)
    输出: Markdown 格式的完整视频脚本

    Args:
        script: FinalScript 字典

    Returns:
        str: Markdown 格式的脚本文本
    """
    lines = []

    # 标题
    title = script.get("title", "未命名视频")
    lines.append(f"# {title}")
    lines.append("")

    # 元信息
    platform = script.get("target_platform", "通用")
    style = script.get("style", "口播")
    duration = script.get("total_duration_sec", 0)
    music = script.get("music_style", "无")
    lines.append(f"> **平台**: {platform} | **风格**: {style} | **时长**: {duration}秒 | **配乐**: {music}")
    lines.append("")

    # 分镜
    scenes = script.get("scenes", [])
    lines.append(f"## 分镜列表（共 {len(scenes)} 个）")
    lines.append("")

    for scene in scenes:
        sid = scene.get("scene_id", "?")
        dur = scene.get("duration_sec", 0)
        lines.append(f"### 分镜 {sid}（{dur}秒）")
        lines.append("")

        visual = scene.get("visual_description", "")
        if visual:
            lines.append(f"**🎬 画面**: {visual}")
            lines.append("")

        narration = scene.get("narration", "")
        if narration:
            lines.append(f"**🎙️ 旁白**: {narration}")
            lines.append("")

        subtitles = scene.get("subtitles", "")
        if subtitles:
            lines.append(f"**💬 字幕**: {subtitles}")
            lines.append("")

        transition = scene.get("transition", "cut")
        bgm = scene.get("background_music", "")
        lines.append(f"*转场: {transition}*")
        if bgm:
            lines.append(f"*音效: {bgm}*")
        lines.append("")
        lines.append("---")
        lines.append("")

    # B-roll 建议
    b_rolls = script.get("b_roll_suggestions", [])
    if b_rolls:
        lines.append("## B-roll 素材建议")
        lines.append("")
        for br in b_rolls:
            lines.append(f"- {br}")
        lines.append("")

    # CTA
    cta = script.get("call_to_action", "")
    if cta:
        lines.append("## 片尾 CTA")
        lines.append("")
        lines.append(f"> {cta}")
        lines.append("")

    return "\n".join(lines)
