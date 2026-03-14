# AI Sales Trainer

基于 LangGraph 多智能体架构的 AI 销售训练系统。系统模拟真实客户角色，让销售人员在对话中练习价值传递技巧，并提供多维度实时评估和训练报告。

## 系统架构

```
用户输入 → Evaluator（评估） → StateUpdater（状态更新） → StrategistPersona（客户回复）
                ↓                       ↓                        ↓
          语义点匹配评分          难度/进度/态度更新          基于人设的客户回复
```

**核心技术栈：**
- **后端**: FastAPI + LangGraph + LiteLLM + SQLAlchemy(async) + SQLite
- **前端**: React 19 + TypeScript + Vite + Tailwind CSS + Recharts
- **LLM**: 通过 LiteLLM 支持 OpenAI / Gemini / Claude 等任意兼容 API

## 功能特性

- 多场景训练（医药销售、企业 SaaS、云安全、财富管理、豪华汽车、K12 教育等）
- 三级难度系统（简单 / 普通 / 困难）
- 四维评分（数据引用、客户相关性、FAB 结构、互动技巧）
- 语义点覆盖率追踪
- SSE 实时流式对话
- 训练报告 + 技能雷达图
- 个人 Dashboard（趋势分析、弱项诊断、历史记录）
- RBAC 权限系统（用户 / 管理员 / 超级管理员）
- 管理员后台（用户统计、场景统计、角色管理）

## 快速开始

### 前置要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 包管理器
- Node.js 18+（前端开发）

### 1. 安装后端依赖

```bash
cd ai-sales-trainer
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 LLM API 配置：

```env
TRAINER_LLM_MODEL=openai/gpt-4o
TRAINER_LLM_API_KEY=sk-your-api-key
TRAINER_LLM_API_BASE=https://api.openai.com/v1
```

> 支持任何 OpenAI 兼容 API（如 yunwu.ai 代理、Azure OpenAI 等），只需设置对应的 base URL 和 key。

### 3. 启动后端

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

首次启动会自动创建 SQLite 数据库和加载内置场景。

### 4. 安装并启动前端

```bash
cd web
npm install
npm run dev
```

前端默认运行在 `http://localhost:5173`，自动代理 `/api` 请求到后端。

### 5. 开始使用

1. 打开 `http://localhost:5173`，注册账号（**第一个注册的用户自动成为超级管理员**）
2. 选择训练场景和难度，开始对话练习
3. 完成训练后查看多维度报告
4. 在 Dashboard 查看个人训练趋势和弱项分析

## 生产部署

### 后端

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

建议配合 nginx 反向代理使用。

### 前端

```bash
cd web
npm run build
```

构建产物在 `web/dist/`，可用 nginx 托管静态文件，并将 `/api` 代理到后端：

```nginx
server {
    listen 80;

    location / {
        root /path/to/web/dist;
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;           # SSE 需要关闭缓冲
    }
}
```

## API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/auth/register` | POST | 注册 |
| `/api/v1/auth/login` | POST | 登录 |
| `/api/v1/auth/users` | GET | 用户列表（管理员） |
| `/api/v1/auth/users/{id}/role` | PUT | 修改角色（超管） |
| `/api/v1/scenarios` | GET | 场景列表 |
| `/api/v1/scenarios` | POST | 创建场景（管理员） |
| `/api/v1/sessions` | POST | 创建训练会话 |
| `/api/v1/sessions/{id}/messages` | POST | 发送消息（SSE 流） |
| `/api/v1/sessions/{id}/report` | GET | 训练报告 |
| `/api/v1/dashboard/me` | GET | 个人看板 |
| `/api/v1/dashboard/admin` | GET | 管理员看板 |

## 项目结构

```
ai-sales-trainer/
├── app/
│   ├── api/            # FastAPI 路由（auth, sessions, messages, scenarios, dashboard, reports, feedback）
│   ├── graph/          # LangGraph 图定义和节点
│   │   ├── builder.py  # StateGraph 编排
│   │   ├── state.py    # 会话状态定义
│   │   └── nodes/      # Evaluator, StateUpdater, StrategistPersona
│   ├── llm/            # LiteLLM 封装
│   ├── models/         # SQLAlchemy 模型 + 场景 Pydantic 模型
│   ├── prompts/        # 版本化 prompt 模板（v1/）
│   ├── services/       # 业务逻辑（session, report, dashboard）
│   └── data/scenarios/ # 内置训练场景 JSON
├── web/                # React 前端（详见 web/README.md）
├── tests/              # 测试用例
├── .env.example        # 环境变量模板
└── pyproject.toml      # Python 依赖
```

## 测试

```bash
uv run pytest tests/ -v
```

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `TRAINER_LLM_MODEL` | 是 | LiteLLM 模型 ID |
| `TRAINER_LLM_API_KEY` | 是 | API Key |
| `TRAINER_LLM_API_BASE` | 是 | API Base URL |
| `TRAINER_LITE_MODEL` | 否 | 轻量模型（评估/压缩用） |
| `TRAINER_JWT_SECRET` | 否 | JWT 密钥（默认自动生成） |
| `TRAINER_SUPER_ADMIN_USERNAME` | 否 | 指定超管用户名 |
