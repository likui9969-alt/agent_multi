"""
backend/main.py — 应用启动入口
===============================

使用 uvicorn 启动 FastAPI 服务。

启动方式:
    python backend/main.py              # 默认 0.0.0.0:8000
    python backend/main.py --port 9000  # 自定义端口
    uvicorn backend.main:app --reload   # 开发模式（热重载）

Docker 启动:
    CMD ["python", "backend/main.py"]
"""

import sys
import os

# 将项目根目录加入 sys.path，确保所有 import 从项目根解析
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from backend.app import create_app

# ── 日志配置 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── 创建应用 ──────────────────────────────────────────────────
app = create_app()

# ── 启动入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from backend.config.settings import get_settings

    settings = get_settings()
    logger = logging.getLogger(__name__)
    logger.info(f"启动 {settings.app_name} v{settings.app_version}")

    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="info",
    )
