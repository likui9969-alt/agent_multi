

# Agent 多媒体脚本生成系统

## 项目简介
- 基于 FastAPI + Redis 的多 Agent 系统
- 能生成视频分镜脚本、旁白、字幕等
- 支持多主题生成

## 技术栈
- Python 3.11+
- FastAPI
- Redis
- Docker / docker-compose
- Streamlit（可选前端）

## 系统架构
1. **TopicAgent**：选题评分
2. **WriterAgent**：内容生成
3. **ReviewAgent**：审核
4. **VideoScriptAgent**：生成分镜脚本
5. **Redis**：状态存储

## 安装与运行
### Docker 一键启动
```bash
git clone https://github.com/likui9969-alt/agent_multi.git
cd agent_multi
docker-compose up --build
````

### 本地运行

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

### 访问接口

* Swagger UI: `http://127.0.0.1:8000/docs`
* 健康检查: `http://127.0.0.1:8000/health`
* 核心接口: `/api/v1/generate`

## 使用示例

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/generate" \
-H "Content-Type: application/json" \
-d '{"topic": "Python是什么", "max_retries": 2}'
```

## 项目效果

* 成功生成 Markdown 格式视频脚本
* 每个分镜包含：画面描述 + 旁白台词 + 字幕文本 + 转场效果 + 音效
* 末尾附 B-roll 素材建议和片尾 CTA

```

---

记得配置一个自己的.env文件，需要配置api_key
