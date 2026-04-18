# ShootCoach 技术方案说明

## 1. 项目定位

ShootCoach 是一个面向篮球投篮训练场景的智能系统，当前聚焦两条主线：

1. 动作评分链路
基于姿态序列、规则特征和时序模型，对投篮动作进行等级判断与动作建议输出。

2. 智能问答链路
基于知识库检索、短期记忆、长期记忆和问答编排，为用户提供带上下文的训练问答能力。

当前项目已经形成“可运行的 baseline 架构”，并在此基础上逐步演进为“可升级架构”。

---

## 2. 当前整体架构

系统目前可以分为 5 个层次：

1. API 层
- 提供评分、问答、知识库管理、长期记忆管理等 HTTP 接口
- 入口见 [main.py](D:/base/packages/apps/api/main.py)

2. 模型与特征层
- 封装评分模型推理、姿态预处理、角度特征、关键帧与 coaching hints
- 代码位于 [models_dl](D:/base/packages/models_dl) 和 [features](D:/base/packages/features)

3. RAG 层
- 负责知识文档切块、入库、检索、查询重写
- 代码位于 [packages/rag](D:/base/packages/rag)

4. Memory 层
- 负责短期记忆和长期记忆管理
- 代码位于 [packages/memory](D:/base/packages/memory)

5. Agent 层
- baseline agent：当前主要问答编排链路
- upgraded agent：已建立骨架，后续可扩展到 router / planner / critic
- 代码位于 [packages/agents](D:/base/packages/agents)

---

## 3. 当前目录结构

当前已经落地的关键目录如下：

```text
docs/
  ARCHITECTURE_V2.md
  TECHNICAL_OVERVIEW.md

configs/
  model.yaml
  rag.yaml
  memory.yaml
  agents.yaml

packages/
  apps/api/
  agents/baseline/
  agents/upgraded/
  rag/
  memory/
  models_dl/
  features/
  pose_pipeline/

data/
  knowledge/
  kb.sqlite3
  ltm.sqlite3
```

---

## 4. 动作评分链路

### 4.1 输入输出

当前正式对外的评分接口是：

- `POST /v1/score/sequence`

输入为形状 `(T, 75)` 的姿态时序特征，输出为：

- `grade`
- `probabilities`
- `keyframe_index`
- `angles_summary`
- `coaching_hints`

相关实现：

- [scoring.py](D:/base/packages/apps/api/routers/scoring.py)
- [inference.py](D:/base/packages/models_dl/inference.py)
- [schemas.py](D:/base/packages/models_dl/schemas.py)

### 4.2 当前策略

评分链路目前遵循“基础模型独立、问答系统通过接口调用”的思路：

- 评分逻辑不依赖 Agent
- Agent 可以按需调用评分结果
- 评分模型仍然保持独立可测试、可部署

### 4.3 当前状态

已完成：

- `sequence -> model -> result` 基础推理链路
- 规则特征与动作建议输出
- FastAPI 对外接口

未完成：

- `video -> pose -> sequence -> score` 的正式端到端链路
- `/v1/score/video`
- 异步任务与 worker

---

## 5. 智能问答与 Agent 架构

### 5.1 baseline agent

baseline agent 是当前实际工作的问答引擎。

核心流程：

1. 读取短期记忆
2. 查询重写
3. 检索知识库
4. 检索长期记忆
5. 组装上下文
6. 生成回答
7. 写回短期记忆

核心实现：

- [pipeline.py](D:/base/packages/agents/baseline/pipeline.py)
- [chat.py](D:/base/packages/apps/api/routers/chat.py)

### 5.2 upgraded agent

upgraded agent 当前还是骨架，但已经建立好了升级入口。

目标方向：

- Router：识别请求类型
- Planner：决定检索、记忆、评分的执行顺序
- Critic：检查回答是否有引用、是否有幻觉、是否需要补检索

当前实现：

- [graph.py](D:/base/packages/agents/upgraded/graph.py)
- [roles.py](D:/base/packages/agents/upgraded/roles.py)

现阶段说明：

- `agent_mode` 已接入 `/v1/chat`
- `upgraded` 当前先复用 baseline 能力
- 后续可以在不改接口的前提下替换内部编排方式

---

## 6. 知识库与 RAG 设计

### 6.1 当前知识库输入方式

当前知识库主要支持：

- 本地 `txt / md` 文档
- API 直接新增文本

相关目录：

- [data/knowledge](D:/base/data/knowledge)

相关接口：

- `GET /v1/kb/documents`
- `POST /v1/kb/documents`
- `DELETE /v1/kb/documents/{doc_id}`
- `POST /v1/kb/documents/reindex`

接口实现：

- [kb.py](D:/base/packages/apps/api/routers/kb.py)

### 6.2 当前 RAG 流程

当前流程如下：

1. 文档读取
2. 分块
3. 向量化
4. 写入 SQLite 知识库
5. 查询重写
6. hybrid 检索
7. 阈值过滤、文档去重
8. 输出更短的引用片段

核心代码：

- [ingest.py](D:/base/packages/rag/ingest.py)
- [retriever.py](D:/base/packages/rag/retriever.py)
- [store.py](D:/base/packages/rag/store.py)
- [rewrite.py](D:/base/packages/rag/rewrite.py)

### 6.3 当前检索策略

目前的 RAG 检索策略不是单一向量召回，而是更偏工程可用的 hybrid 方案：

- 向量检索
- lexical fallback
- rerank 混合排序
- 最低分阈值过滤
- 文档级去重
- 截短后的 citation 片段

相关配置见：

- [rag.yaml](D:/base/configs/rag.yaml)

包含参数：

- `chunk_size`
- `chunk_overlap`
- `min_score`
- `max_hits_per_doc`
- `snippet_chars`

### 6.4 当前状态

已完成：

- 本地知识库管理接口
- 文档切块与重建索引
- hybrid 检索和基础重排
- citation 输出

未完成：

- 真正的语义分块模型
- 正式的向量数据库
- 更高级的 reranker
- 文档权限、版本与审计

---

## 7. 短期记忆设计

### 7.1 当前目标

短期记忆用于维持会话上下文连续性，避免每轮问答都只看单次输入。

### 7.2 当前实现

短期记忆采用两级策略：

1. Redis 版本
- 用于正式短期记忆存储
- 记录最近消息窗口和摘要

2. 内存 fallback
- 当 Redis 不可用时，退化到本地内存存储

相关代码：

- [short_term.py](D:/base/packages/memory/short_term.py)
- [redis_memory.py](D:/base/packages/agents/baseline/redis_memory.py)
- [memory.py](D:/base/packages/agents/baseline/memory.py)

### 7.3 当前摘要策略

最近已升级为“滚动摘要”：

- 保留已有摘要
- 合并最近若干轮重要信息
- 控制摘要长度
- 避免无限膨胀

实现见：

- [summary.py](D:/base/packages/memory/summary.py)

### 7.4 当前状态

已完成：

- 滑动窗口
- Redis 存储
- 规则滚动摘要

未完成：

- 更高质量的语义摘要
- 重要轮次抽取
- 摘要质量评估

---

## 8. 长期记忆设计

### 8.1 当前目标

长期记忆用于存储“用户稳定偏好、事实、训练目标”等跨会话信息。

### 8.2 当前实现

长期记忆现在支持：

- 主动录入
- SQLite 持久化
- 向量化存储
- 相似度召回
- 重要度权重
- 过期控制
- 删除接口

接口：

- `POST /v1/memory/long_term`
- `GET /v1/memory/long_term`
- `DELETE /v1/memory/long_term/{memory_id}`

相关代码：

- [ltm_sqlite.py](D:/base/packages/agents/baseline/ltm_sqlite.py)
- [long_term.py](D:/base/packages/memory/long_term.py)
- [chat.py](D:/base/packages/apps/api/routers/chat.py)

### 8.3 当前长期记忆字段

当前长期记忆记录已经包含：

- `id`
- `user_id`
- `type`
- `content`
- `tags`
- `importance`
- `created_at`
- `expires_at`

### 8.4 当前状态

已完成：

- 主动录入
- 向量化
- 语义召回
- 删除
- 重要度加权
- 过期过滤

未完成：

- 更复杂的记忆类型治理
- 自动沉淀记忆
- 记忆冲突解决策略
- 长期记忆可视化管理界面

---

## 9. 当前 API 一览

### 9.1 评分相关

- `GET /healthz`
- `GET /v1/version`
- `POST /v1/score/sequence`
- `POST /v1/demo/video_score`

### 9.2 问答相关

- `POST /v1/chat`

支持：

- `agent_mode=baseline`
- `agent_mode=upgraded`

### 9.3 知识库相关

- `GET /v1/kb/documents`
- `POST /v1/kb/documents`
- `DELETE /v1/kb/documents/{doc_id}`
- `POST /v1/kb/documents/reindex`

### 9.4 长期记忆相关

- `POST /v1/memory/long_term`
- `GET /v1/memory/long_term`
- `DELETE /v1/memory/long_term/{memory_id}`

---

## 10. Demo 页面能力

当前 `localhost` 页面已不是单纯的视频上传页面，而是一个本地联调控制台。

页面支持：

1. 上传视频并做 demo 评分
2. 基于评分和知识库继续问答
3. 查看知识库文档列表
4. 新增知识库文档
5. 删除知识库文档
6. 重建知识库索引

前端代码：

- [index.html](D:/base/packages/apps/api/static/index.html)
- [app.js](D:/base/packages/apps/api/static/app.js)
- [styles.css](D:/base/packages/apps/api/static/styles.css)

---

## 11. 当前技术策略总结

### 11.1 总体策略

当前项目采用“先做稳定 baseline，再逐步升级”的演进策略：

- 评分系统先独立成基础服务
- 问答系统先做 baseline agent
- 记忆和 RAG 先跑通，再逐步增强
- 先保留统一接口，再替换内部能力

### 11.2 为什么这样设计

这样做的主要好处是：

- 能快速形成可演示系统
- 各模块职责清晰
- 可以边开发边验证
- 升级不需要推翻已有接口

### 11.3 当前阶段的核心特点

目前项目的特点不是“功能已经全部做完”，而是：

- 主链路已贯通
- 模块边界已开始稳定
- 架构已经具备升级空间
- baseline 可以支撑本地调试和产品演示

---

## 12. 当前不足与后续方向

### 12.1 当前不足

目前仍然存在以下不足：

- 视频到姿态序列的正式链路未完成
- upgraded agent 仍是骨架
- MCP server 未落地
- 检索排序仍可继续优化
- 源码中部分旧文件仍有乱码问题
- 自动化测试覆盖还不够完整

### 12.2 后续推荐路线

建议后续优先级如下：

1. 完善问答前端
- 在页面中展示 citations 和 retrieval_summary

2. 继续增强检索质量
- 标题加权
- FAQ 优先
- 更强 rerank

3. 完成视频异步评分链路
- `/v1/score/video`
- `jobs`
- worker

4. 落地 upgraded agent
- router
- planner
- critic

5. 视情况引入 MCP
- 对接外部 Agent Host
- 统一工具调用协议

---

## 13. 一句话介绍项目

ShootCoach 当前是一个围绕“篮球投篮训练”的智能系统原型，已经具备基础动作评分、知识库问答、短长期记忆和本地知识库管理能力，并正在从 baseline 架构逐步升级为支持多策略检索和可扩展智能体编排的技术体系。
