# 投篮智能评分系统 v2 — 架构规格（落盘版）

> 本文档为仓库固定架构说明：`docs/ARCHITECTURE_V2.md`  
> 需求来源：`readme.md`（原系统）、`willing.md`（拓展：知识库必选，MCP 多智能体可选）。

---

## 1. 总览

| 子系统 | 架构策略 |
|--------|----------|
| **模型训练与输出判别** | **基础架构**：延续 readme 中 OpenPose → 序列/角度/关键帧 → **LSTM/BiLSTM 时序分类（5 档）** 路径；训练、导出、推理判别不引入多智能体复杂度。 |
| **智能问答与知识库** | **必选**：RAG + 查询重写 + 语义分块 + Redis 短期记忆 + 向量长期记忆（见第 4、6 节）。 |
| **Agent** | **两套方案**：**基线**（单链路、易落地）与**升级**（MCP + 多角色编排，可插拔）。 |

---

## 2. 模型训练与输出判别（基础架构）

### 2.1 范围声明

训练与判别仅依赖「姿态流水线 + 手工/规则特征 + 深度时序模型」，**不绑定** Agent、MCP 或 LLM。Agent 层通过 **HTTP/内部 SDK** 调用判别服务即可。

### 2.2 数据与特征（与 readme 对齐）

- **输入**：视频 → OpenPose `body_25` → 按「动态抽帧」策略得到约 **T×25×3**（readme 中约 30 帧量级，实现可配置为与 `train.py` 一致的 **T×75** 展平）。
- **特征工程**：关节角（左右手臂夹角、左右肘角）、四条角度曲线、**k = k1+k2+k3+k4** 关键帧及邻域最值；用于可解释文案与标签构造。
- **数据增强**：对姿态矩阵平移、翻转等（readme 所述）；小样本下配合 Dropout / L2。
- **标签**：5 档分类（与现有 Softmax 输出一致）。

### 2.3 模型形态（基础）

- **结构**：以 **BiLSTM/LSTM + Dropout + Dense** 为主干（与历史 `train.py` 思路一致；层数/单元数以验证集为准防过拟合）。
- **训练**：`sparse_categorical_crossentropy` + Adam；批大小、轮次、是否正则化等与 readme 对照实验结论一致即可。
- **制品**：`model.h5` 或导出的 **ONNX**（推荐便于部署）；**固定输入形状** `(T, 75)` 及 **归一化系数**（如历史 `/1500`）写入 `configs/model.yaml`，推理与训练必须同源。

### 2.4 判别输出（对外最小契约）

- **必选字段**：`grade`（1–5）、`probabilities[5]`。
- **可选字段**：`keyframe_index`、`angles_summary`、`coaching_hints`（由规则特征与阈值模板生成，不依赖 LLM）。

### 2.5 代码落位（建议）

- `packages/pose_pipeline/` — 抽帧、OpenPose 封装、序列 Schema。  
- `packages/features/` — 角度、关键帧、k 值、指导意见数据。  
- `packages/models_dl/` — 训练脚本、导出、**纯推理** `predict(sequence) -> ScoringResult`。  
- `apps/api/routers/scoring.py` — 仅转发到 `models_dl`，无 Agent 逻辑。

---

## 3. 模块与文件结构（全仓）

```text
base/
├── docs/
│   └── ARCHITECTURE_V2.md          # 本文件（固定架构说明）
├── apps/
│   ├── api/
│   │   ├── routers/
│   │   │   ├── scoring.py         # 基础判别 API
│   │   │   ├── chat.py            # 对话（装配记忆+RAG）
│   │   │   ├── knowledge.py
│   │   │   └── memory.py
│   │   └── settings.py
│   ├── worker/                     # 重视频、OpenPose 异步任务
│   └── mcp_server/                 # 【可选】仅升级 Agent 方案需要
├── packages/
│   ├── pose_pipeline/
│   ├── features/
│   ├── models_dl/                  # 基础训练与推理
│   ├── rag/                        # 分块、重写、检索
│   ├── memory/                     # Redis 短期 + 向量长期
│   ├── agents/
│   │   ├── baseline/               # 基线 Agent（单编排器）
│   │   └── upgraded/               # 升级 Agent（图/MCP）
│   └── orchestration/              # 可选：共享的 tool 协议 DTO
├── configs/
│   ├── model.yaml
│   ├── rag.yaml
│   ├── memory.yaml
│   └── agents.yaml                 # baseline | upgraded 开关与角色参数
├── legacy/
│   ├── train.py
│   └── judge_colab.py
└── tests/
```

---

## 4. Agent：基线方案 vs 升级架构

### 4.1 共同能力边界

两套方案均需对接：

- **知识库 RAG**（语义分块、向量召回、查询重写）；  
- **短期记忆**（滑动窗口 + 摘要 + Redis）；  
- **长期记忆**（主动录入、向量化、语义召回）；  
- **基础判别服务**（第 2 节）：输入序列或 `job_id` 结果，只读调用。

差异集中在：**编排复杂度、是否 MCP、是否多角色与自愈循环**。

---

### 4.2 基线方案（Baseline Agent）

**定位**：单进程、**单编排链路**（一个「教练助手」逻辑），最快打通 willing 中的必选能力。

**拓扑**（逻辑）：

```text
用户消息
  → 读 Redis（窗口 + 摘要）
  → 可选：长期记忆检索
  → 查询重写
  → KB 向量检索（+ 可选关键词）
  → （可选）调用 scoring API：仅当用户上传序列/关联 job_id 需要数值档位时
  → 单次 LLM 调用生成回复
  → 写回 Redis；必要时异步写入长期记忆向量
```

**特点**：

| 项 | 说明 |
|----|------|
| 角色 | 无独立 Planner/Critic 进程，**一个 system prompt** 定义教练+知识引用风格。 |
| 工具 | 以内置函数/SDK 调用 `scoring`、`kb_search`，**不暴露 MCP**。 |
| 适用 | 开发联调、低成本部署、无 Cursor/外部 Agent Host 场景。 |
| 风险 | 复杂任务（多步推理+多工具）易在长对话中漂移；需靠 **RAG 引用约束 + 摘要** 缓解。 |

**配置**：`configs/agents.yaml` 中 `mode: baseline`。

**代码落位**：`packages/agents/baseline/` — `pipeline.py`（顺序步骤）、`prompts.py`。

---

### 4.3 升级架构（Upgraded Agent）

**定位**：**MCP（可选）+ Multi-Agent**，适合工具多、宿主为 Cursor/自研 Agent 平台、要与外部系统解耦的场景。

**拓扑**（逻辑）：

```text
                ┌─────────────┐
  用户 ────────►│  Orchestrator│（Router / Planner）
                └──────┬──────┘
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
  ┌─────────┐    ┌───────────┐   ┌────────────┐
  │ KB Agent │    │Score Tool │   │Memory Tool │
  │(RAG子图) │    │(MCP/HTTP) │   │(Redis+Vec) │
  └─────────┘    └───────────┘   └────────────┘
       │               │               │
       └───────────────┴───────────────┘
                       ▼
                 ┌───────────┐
                 │  Critic   │（引用检查、禁忌、是否再检索）
                 └─────┬─────┘
                       ▼
                   最终回复
```

**角色建议**：

| 角色 | 职责 |
|------|------|
| **Router** | 意图分类：闲聊 / 查 KB / 要评分 / 记长期事实。 |
| **Planner** | 多步计划：先检索再答，或先 `score` 再生成建议。 |
| **KB/RAG 子 Agent** | 封装重写、分块检索、片段格式化。 |
| **Critic** | 校验是否引用 `citations`、是否与 `ScoringResult` 矛盾；不通过则退回补充检索或改写。 |

**MCP**：

- 独立进程 `apps/mcp_server/`：`score_sequence`、`kb_search`、`session_context_get/set`、`long_term_memory_append` 等与 API 对齐。  
- Multi-Agent 宿主通过 MCP 调工具，**业务实现仍在 `packages/`**，避免重复逻辑。

**配置**：`configs/agents.yaml` 中 `mode: upgraded`，`mcp.enabled: true/false`。

**代码落位**：`packages/agents/upgraded/` — `graph.py`（状态机或简单 DAG）、`roles.py`、`mcp_tools_mapping.py`。

---

### 4.4 基线 vs 升级对照表

| 维度 | 基线方案 | 升级架构 |
|------|----------|----------|
| 进程数 | 少（通常仅 API + Redis + 向量库） | 多（可加 MCP Server、Worker） |
| 编排 | 线性 pipeline | 多角色 + 可选循环（Critic） |
| 工具协议 | 进程内函数 | MCP + HTTP 统一 Tool 层 |
| 运维与调试 | 简单 | 需 trace、工具超时与权限 |
| 与 willing 对齐 | 必选项全覆盖 | 必选 + **可选项 MCP multiagent** |

---

## 5. TodoList（实施顺序）

### Phase 0 — 契约

- [ ] 冻结 `PoseSequence` / `ScoringResult` JSON Schema 与 `configs/model.yaml`。  
- [ ] 基础推理服务：`POST /v1/score/sequence` 可独立通过集成测试。

### Phase 1 — 知识库与记忆（willing 必选）

- [ ] RAG：语义分块、嵌入、向量库、查询重写。  
- [ ] Redis：滑动窗口 + 滚动摘要。  
- [ ] 长期记忆：录入 API + 向量召回 + 与 KB 分 collection/前缀。

### Phase 2 — 基线 Agent

- [ ] `packages/agents/baseline` 串联记忆 → 重写 → 检索 → LLM。  
- [ ] `POST /v1/chat` 默认 `agents.mode=baseline`。

### Phase 3 — 升级 Agent（可选）

- [ ] `packages/agents/upgraded` 多角色图 + Critic。  
- [ ] `apps/mcp_server` 与 `agents.yaml` 联动；文档说明 Cursor/外部连接方式。

### Phase 4 — 视频端到端

- [ ] Worker + `POST /v1/score/video`；结果写入对话上下文供基线/升级复用。

---

## 6. 接口摘要（与判别基础架构分离）

| 域 | 方法 | 路径 | 说明 |
|----|------|------|------|
| 判别（基础） | POST | `/v1/score/sequence` | 张量/JSON 序列 → 档位与概率 |
| 判别（基础） | POST/GET | `/v1/score/video`, `/v1/jobs/{id}` | 异步视频任务 |
| 对话 | POST | `/v1/chat` | `agent_mode`: `baseline` \| `upgraded`（若实现） |
| 知识库 | POST/DELETE | `/v1/kb/documents`, … | 文档与重索引 |
| 长期记忆 | POST/GET/DELETE | `/v1/memory/long_term` | 主动录入与删除 |

详细字段与错误码可在实现阶段补充至 `docs/API.md`（可选）。

---

## 7. 核心模型（数据与运行时）

### 7.1 判别（基础架构）

- **输入**：`(T, 75)` float，与训练归一化一致。  
- **输出**：5 类 logits / softmax；可选规则衍生 `coaching_hints`。

### 7.2 知识库与检索

- **Document / Chunk / Embedding**；**RewriteResult**；**RetrievalHit**。

### 7.3 记忆

- **SessionState**（Redis）：`messages[]`、`summary`、`ttl`。  
- **LongTermMemoryRecord**：`user_id`、`type`、`content`、向量、`expires_at`。

### 7.4 Agent 运行时（两套共用 DTO）

- **ChatRequest**：`session_id`、`user_id`、`content`、`agent_mode`、`use_kb`、`use_long_term_memory`。  
- **ChatResponse**：`reply`、`citations[]`、`memory_used`、`agent_mode`。  
- **升级专有**：内部 `PlanStep[]`、`ToolCall[]`（可记录于日志，不必全部对外暴露）。

---

## 8. 修订记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v2.0 | 2026-04-18 | 初版落盘：训练/判别基础架构；Agent 基线 + 升级双方案；模块与 Todo、接口摘要。 |
