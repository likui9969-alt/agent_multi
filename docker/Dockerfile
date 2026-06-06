# ============================================================
# AIGC Multi-Agent Platform — 生产镜像
# ============================================================
# 构建: docker build -t agent-multi:latest -f docker/Dockerfile .
# ============================================================

# ── Stage 1: Builder（安装依赖） ──────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# 安装编译依赖（redis 需要 gcc）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单，利用 Docker 缓存层
COPY requirements.txt .

# 安装到临时目录，避免污染最终镜像
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime（最小运行时） ────────────────────────
FROM python:3.11-slim-bookworm

# 元数据
LABEL org.opencontainers.image.title="AIGC Multi-Agent Platform"
LABEL org.opencontainers.image.description="基于 LangGraph + DeepSeek 的自动化内容生成平台"
LABEL org.opencontainers.image.version="0.1.0"

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# 从 builder 复制已安装的依赖
COPY --from=builder /install /usr/local

# 复制应用代码
COPY . .

# 修正权限
RUN chown -R appuser:appuser /app

# 切换到非 root 用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "backend/main.py"]
