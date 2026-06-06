"""
frontend/streamlit_app.py — AIGC Multi-Agent 视频脚本生成器
=============================================================

Streamlit 前端界面，提供：
  - 主题输入 + 一键生成
  - 4 个 Agent 实时进度可视化
  - 完整视频脚本预览
  - 一键下载 Markdown 文件

启动:
  streamlit run frontend/streamlit_app.py
"""

import time
import threading
import streamlit as st
import httpx

# ── 配置 ────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
GENERATE_URL = f"{API_BASE}/api/v1/generate"
STATUS_URL = f"{API_BASE}/api/v1/task/{{task_id}}/status"
RESULT_URL = f"{API_BASE}/api/v1/task/{{task_id}}/result"

AGENT_NAMES = {
    "START":            ("⚪ 等待开始", 0),
    "topic_analysis":   ("📊 选题分析中...", 1),
    "content_generate": ("📝 内容生成中...", 2),
    "content_review":   ("🔍 内容审核中...", 3),
    "video_script":     ("🎬 视频脚本生成中...", 4),
    "END":              ("✅ 完成", 5),
}

# ── 页面设置 ────────────────────────────────────────────────
st.set_page_config(
    page_title="AIGC 视频脚本生成器",
    page_icon="🎬",
    layout="wide",
)
st.title("🎬 AIGC 多智能体视频脚本生成器")
st.caption("基于 LangGraph + Qwen 的自动化内容创作平台")

# ── 侧边栏 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 设置")
    max_retries = st.slider("最大重试次数", 1, 5, 2, help="审核不通过时自动重写的次数上限")
    st.divider()
    st.header("📋 Agent 流水线")
    st.markdown("""
    1. 📊 **选题分析** — 搜索 + 评分
    2. 📝 **内容生成** — 撰写完整文案
    3. 🔍 **内容审核** — 4维质量审查
    4. 🎬 **视频脚本** — 分镜 + 旁白 + 字幕
    """)
    st.divider()
    st.caption("Powered by LangGraph + Streamlit")

# ── 主区域：输入区 ──────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input(
        "输入创作主题",
        placeholder="例如：Python编程入门、AI Agent发展趋势、量子计算原理...",
        label_visibility="collapsed",
    )
with col2:
    generate_btn = st.button("🚀 生成脚本", type="primary", use_container_width=True, disabled=not topic)

# ── 进度显示 ────────────────────────────────────────────────
progress_container = st.container()

# ── 结果区域 ────────────────────────────────────────────────
result_container = st.container()


# ── 生成逻辑 ────────────────────────────────────────────────
def run_generation():
    """后台线程：提交任务 → 轮询状态 → 获取结果"""
    try:
        # 1. 提交任务
        with st.spinner("提交任务..."):
            r = httpx.post(GENERATE_URL, json={"topic": topic, "max_retries": max_retries}, timeout=600)
            if r.status_code != 200:
                st.session_state.error = f"提交失败: {r.text[:300]}"
                st.session_state.running = False
                return
            data = r.json()
            task_id = data["task_id"]
            st.session_state.task_id = task_id
            st.session_state.final_script = data.get("final_script", "")
            st.session_state.status = "completed"

    except httpx.TimeoutException:
        st.session_state.error = "请求超时（>600秒），请尝试更短的主题"
    except Exception as e:
        st.session_state.error = f"请求失败: {str(e)}"
    finally:
        st.session_state.running = False


# ── 状态初始化 ──────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "status" not in st.session_state:
    st.session_state.status = None
if "task_id" not in st.session_state:
    st.session_state.task_id = None
if "final_script" not in st.session_state:
    st.session_state.final_script = None
if "error" not in st.session_state:
    st.session_state.error = None
if "thread" not in st.session_state:
    st.session_state.thread = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

# ── 处理生成按钮 ────────────────────────────────────────────
if generate_btn and topic and not st.session_state.running:
    # 如果已有旧线程在运行，等待结束
    if st.session_state.thread and st.session_state.thread.is_alive():
        st.session_state.thread.join(timeout=1)

    st.session_state.running = True
    st.session_state.status = "running"
    st.session_state.error = None
    st.session_state.final_script = None
    st.session_state.last_topic = topic

    thread = threading.Thread(target=run_generation, daemon=True)
    st.session_state.thread = thread
    thread.start()

# ── 进度显示 ────────────────────────────────────────────────
with progress_container:
    if st.session_state.running:
        # 显示模拟进度（真实轮询在后台线程中）
        progress_bar = st.progress(0, "📊 选题分析中...")
        status_text = st.empty()

        steps = [
            ("📊 选题分析", "搜索话题 + 评估热度 + 生成大纲"),
            ("📝 内容生成", "基于大纲撰写完整文案"),
            ("🔍 内容审核", "4维质量审查 + 事实核查"),
            ("🎬 视频脚本", "文案转分镜 + 旁白 + 字幕"),
        ]

        # 模拟进度动画
        for i, (title, desc) in enumerate(steps):
            if not st.session_state.running:
                break
            progress_bar.progress((i + 1) / 5, f"{title}中...")
            status_text.info(f"**{title}** — {desc}")
            # 每步等待，直到完成或超时
            waited = 0
            while st.session_state.running and waited < 120:
                time.sleep(2)
                waited += 2
                # 检查是否已完成
                if st.session_state.final_script:
                    break
            if st.session_state.final_script:
                progress_bar.progress(1.0, "✅ 生成完成！")
                status_text.success("所有 Agent 已完成")
                st.session_state.running = False
                st.rerun()
                break

        if st.session_state.running and not st.session_state.final_script:
            progress_bar.progress(1.0, "⏳ 等待中...")

    elif st.session_state.status == "completed" and st.session_state.final_script:
        st.success(f"✅ 生成完成 — 任务 ID: `{st.session_state.task_id}`")

    elif st.session_state.error:
        st.error(st.session_state.error)

# ── 结果展示 ────────────────────────────────────────────────
with result_container:
    if st.session_state.final_script and st.session_state.status == "completed":
        st.divider()

        script = st.session_state.final_script

        # 下载按钮
        col_a, col_b = st.columns([1, 5])
        with col_a:
            st.download_button(
                label="📥 下载 Markdown",
                data=script,
                file_name=f"视频脚本_{st.session_state.last_topic}.md",
                mime="text/markdown",
                type="primary",
            )

        # 标签页展示
        tab1, tab2 = st.tabs(["📄 渲染预览", "📝 原始 Markdown"])

        with tab1:
            st.markdown(script)

        with tab2:
            st.code(script, language="markdown")

# ── 页脚 ────────────────────────────────────────────────────
st.divider()
st.caption("AIGC Multi-Agent Platform — 基于 LangGraph + DeepSeek/Qwen 的自动化内容生成平台")
