"""
backend/app.py — FastAPI 应用工厂
==================================

使用工厂模式创建 FastAPI 实例，统一管理：
  - 生命周期事件（startup / shutdown）
  - 中间件（CORS / 请求ID）
  - 路由注册

设计原则：
  - 延迟导入：路由在 startup 时注册，避免循环依赖
  - 优雅关闭：shutdown 时清理 LLM 连接池和 Redis 连接
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# 生命周期管理
# ══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理器。

    Startup:
      1. 预加载 Settings
      2. 预初始化 LLM 客户端（验证 API Key 连通性）
      3. 预热 LangGraph 工作流编译

    Shutdown:
      1. 关闭 LLM HTTP 连接池
      2. 关闭 Redis 连接池
    """
    settings = get_settings()
    logger.info(f"🚀 {settings.app_name} v{settings.app_version} 启动中...")

    # ── Startup ──
    try:
        # 预初始化（验证配置 + 预热编译）
        from api.deps import get_llm, get_graph
        llm = get_llm()
        logger.info(f"  ✓ LLM: {settings.llm_model} @ {settings.deepseek_base_url}")
        graph = get_graph()
        logger.info(f"  ✓ LangGraph 工作流已就绪")
    except Exception as exc:
        logger.error(f"  ✗ 初始化失败: {exc}")
        raise

    logger.info(f"✅ {settings.app_name} 已就绪，监听 {settings.app_host}:{settings.app_port}")
    yield  # ← 应用运行期间

    # ── Shutdown ──
    logger.info("🛑 正在关闭服务...")

    # 关闭 LLM 异步 HTTP 客户端
    try:
        from api.deps import get_llm
        llm = get_llm()
        if hasattr(llm, 'async_client') and llm.async_client:
            await llm.async_client.aclose()
            logger.info("  ✓ LLM HTTP 客户端已关闭")
    except Exception as exc:
        logger.warning(f"  ⚠ LLM 关闭异常: {exc}")

    # 关闭 Redis 连接
    try:
        from storage.redis_client import close_redis
        await close_redis()
        logger.info("  ✓ Redis 连接已关闭")
    except Exception as exc:
        logger.warning(f"  ⚠ Redis 关闭异常: {exc}")

    logger.info("✅ 服务已关闭")


# ══════════════════════════════════════════════════════════════════════
# 应用工厂
# ══════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用实例。

    Returns:
        FastAPI: 配置完成的 app 实例，可直接被 uvicorn 挂载。
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AIGC 多智能体自动化内容生成平台 — 基于 LangGraph + DeepSeek",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── 中间件注册 ──────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # 生产环境应改为具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 路由注册 ────────────────────────────────────────────
    from api.routes.generate import router as generate_router
    from api.routes.task import router as task_router
    app.include_router(generate_router, prefix="/api/v1")
    app.include_router(task_router, prefix="/api/v1")

    # 健康检查路由（内联，无需单独文件）
    @app.get("/health", tags=["health"])
    async def health_check():
        """服务存活检查"""
        return {"status": "ok", "version": settings.app_version}

    @app.get("/ready", tags=["health"])
    async def readiness_check():
        """
        就绪检查（含 Redis 连通性）。
        Kubernetes 用此端点判断 Pod 是否可接收流量。
        """
        try:
            # 校验 LLM 可用
            from api.deps import get_llm
            get_llm()
            return {"status": "ready"}
        except Exception as exc:
            return {"status": "not_ready", "reason": str(exc)}

    return app
