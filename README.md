# 🏀 投篮智能评分系统（Shooting Scoring System）

一个结合 **计算机视觉 + 时序建模 + RAG 智能问答** 的篮球投篮分析系统。  
支持从视频中自动识别动作，并给出评分与专业训练建议。

---

## ✨ 项目亮点

- 🎯 自动投篮评分：基于姿态识别 + 时序模型（LSTM/BiLSTM），输出 5 档评分
- 📊 可解释分析：关键帧、关节角度变化、动作阶段分析
- 🤖 智能教练问答（RAG）：结合知识库，提供专业训练建议
- 🧠 记忆能力：支持短期对话记忆 + 长期用户习惯学习
- ⚙️ 双 Agent 架构：
  - 基线方案（快速落地）
  - 多智能体方案（可扩展）

---

## 🧩 系统架构（简述）

视频输入
  ↓
OpenPose 姿态提取
  ↓
特征工程（角度 / 关键帧）
  ↓
LSTM/BiLSTM 模型
  ↓
评分 + 技术分析
  ↓
RAG + Agent
  ↓
自然语言反馈（教练建议）

---

## 🚀 快速开始

### 1️⃣ 环境准备

git clone https://github.com/popsnake/Basketball_System.git
cd base
pip install -r requirements.txt

---

### 2️⃣ 启动服务

uvicorn apps.api.main:app --reload

默认地址：
http://localhost:8000

---

## 📡 核心接口

### 🎯 投篮评分（序列）

POST /v1/score/sequence

输入：
{
  "sequence": [[...], [...]]
}

输出：
{
  "grade": 4,
  "probabilities": [0.1, 0.2, 0.5, 0.15, 0.05],
  "coaching_hints": ["手肘角度偏大", "出手点略低"]
}

---

### 🎥 视频评分（异步）

POST /v1/score/video  
GET /v1/jobs/{id}

---

### 💬 智能对话

POST /v1/chat

示例：
{
  "session_id": "abc",
  "content": "我这个投篮哪里有问题？",
  "agent_mode": "baseline"
}

---

## 🧠 Agent 模式

### Baseline（默认）

- 单链路处理
- 快速稳定
- 推荐初期使用

### Upgraded（可选）

- 多智能体（Planner / Critic / RAG）
- 支持 MCP 工具调用
- 更强复杂任务能力

---

## 📚 知识库（RAG）

系统内置篮球训练知识库：

- 投篮动作标准
- 常见错误分析
- 训练方法建议

支持：

- 文档上传
- 自动分块
- 向量检索
- 查询重写

---

## 🧠 记忆系统

| 类型 | 说明 |
|------|------|
| 短期记忆 | Redis（对话上下文） |
| 长期记忆 | 向量数据库（用户习惯、偏好） |

---

## 📁 项目结构

base/
├── apps/
├── packages/
├── configs/
└── docs/

---

## 📄 License

MIT License
