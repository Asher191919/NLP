# OntoBridg — 竞赛与论文战略

> 版本：v0.1 | 日期：2026-09-19 | 主方向：LLMs4OL | 备选：OAEI 2027

---

## 1. Timeline 与目标年

### 1.1 已确认时间事实

| 事件 | 日期 | 状态（as of 2026-09-19） |
|------|------|---------------------------|
| OAEI 2026 registration | 2026-07-31 | 已过 |
| OAEI 2026 execution deadline | **2026-09-05** | **已过，无法参赛** |
| OAEI 2026 alignments due | 2026-09-22 | 临近；仅作观察 |
| OM workshop @ ISWC Bari | 2026-10-25 | 可听可读，不提交参赛系统 |
| LLMs4OL 系列 | 自 2024 年起 | **2026 已结束**（论文截止 7/28，camera-ready 8/31，workshop 10/25–26 Bari）；**目标 2027** |

### 1.2 目标年决策

| 项目 | 决策 |
|------|------|
| **主竞赛目标** | **LLMs4OL 2027**（若 2026 late entry/observational 通道意外开放则评估，不阻塞主线） |
| **启动时间** | **现在（2026-09）开始 build**，不等官方 2027 announcement |
| **观察义务** | 跟踪 LLMs4OL 官方站点、arXiv、ISWC/OM 通告，任务/数据集/指标变更需在 48h 内记入 `competition/README.md` |
| **备选窗口** | OAEI 2027（约 2027 年年中 registration/execution）；仅当主路径受阻时切换 |

### 1.3 为何现在就要 build

- 2027 公告后到 execution 通常只有数周；系统、数据管道、污染评测协议必须提前就绪。
- 论文 workshop 窗口（OM 2026 投稿可能已过/不适用；2027 workshop 或其他 workshop）依赖系统可运行。
- Contamination 协议与 baseline 复现本身需要时间，且是论文核心，不能压缩成赛前冲刺。

---

## 2. Task Prioritization（以 **LLMs4OL 2026 官方编号** 为准）

> **重要：** 研究范式论文（arXiv:2307.16648）的 A/B/C 与挑战赛编号不同。  
> - 2024 赛：A Term Typing / B Taxonomy / C NonTax RE  
> - 2025 赛：A Text2Onto / B Term Typing / C Taxonomy / D NonTax RE  
> - **2026 赛：A Flagship / B Reuse / C Taxonomy**（本项目主目标）

| 优先级 | 2026 Task | 名称 | 理由 |
|--------|-----------|------|------|
| **P0** | **Task A** | Flagship（端到端 OL） | 主榜指标 Graph Similarity；系统完整性卖点 |
| **P0** | **Task B** | Reuse（本体扩展） | 官方有 terms/types + seed ontology；最易做出可控增益 |
| **P1** | **Task C** | Taxonomy（跨域） | 依赖 OntoLearner 多域本体；hold-out 域测泛化 |
| **P0 方法** | Non-taxonomic RE | Flagship/Reuse 内的子能力 | pro-team 2026 在此为 0，是最大差异化缺口 |
| **P2** | Contamination protocol | 污染检测与 clean-split 报告 | 不占竞赛主分，但是论文主贡献 |

**执行顺序：**

1. 官方 2026 JSON loader + 与 `metrics/graph_similarity.py` 对齐的 eval
2. Task A baseline（RAG + 封闭词表对照）→ OntoBridg 开放关系 + 符号修复
3. Task B seed-extension 管线（vocabulary constraint + graph-aware 补全）
4. Task C：OntoLearner 多域训练 → hold-out 材料/生物/学术/生态
5. Contamination probes + clean subset 重评
6. 真实 LLM API 全量实验、消融、竞赛打包（对接 OntoLearner integration 阶段）

**数据集优先级（2026 官方）：**

| 优先级 | 数据 | 用途 |
|--------|------|------|
| P0 | `2026/TaskA-Flagship/train_task_a.json`（4303） | 端到端主实验 |
| P0 | `2026/TaskB-Reuse/train_task_b.json`（2774） | 扩展任务主实验 |
| P1 | OntoLearner 域本体（除 hold-out 外） | Task C 跨域 taxonomy |
| P2 | 禁止训练清单中的本体 | **不可用于训练**（合规红线） |

---

## 3. Paper Roadmap（Workshop → Conference）

```
Week 0–4   系统 MVP + baseline 表 + 污染 probe 原型
Week 5–8   完整实验 + 消融 + paper draft（workshop 长度）
Week 9–12  投 workshop（OM 相关 / 本体与 LLM 相关 CEUR 或等价）
           同步：竞赛数据若开放 → 正式榜单结果写入 revision
之后 3–6月 扩展版：更深 contamination 分析 + 更多数据集 + constraints 消融
           → ISWC / AAAI / EMNLP-ACL Findings / Knowledge-Based Systems
```

| 阶段 | 产出 | 目标 venue 类型 | 时点 |
|------|------|-----------------|------|
| M1 | 技术报告 / arXiv preprint | 自托管 | Week 8–10 |
| M2 | Workshop paper（4–8 页 CEUR 等） | OM workshop（ISWC 系列）或 ontology/LLM workshop | 赶最近可投 deadline；否则 2027 对应 workshop |
| M3 | 会议主会/Findings | ISWC / AAAI / EMNLP / ACL Findings | Workshop 反馈 + 竞赛结果后 |
| M4 | 期刊扩展 | Knowledge-Based Systems（或 Semantic Web Journal） | 需更完整实验与污染协议长文 |

**论文—竞赛双线绑定：** 竞赛提交材料中的系统描述直接复用 paper Method；竞赛结果表进入 paper Experiments（注明 competition snapshot 日期）。

**作者与许可注意：** UMLS 相关实验若不能公开原始数据，论文中写明获取路径与聚合结果；代码仓库提供非 UMLS 可复现路径。

---

## 4. Risk Matrix

| ID | 风险 | 类型 | 概率 | 影响 | 缓解措施 | Owner 动作 |
|----|------|------|------|------|----------|------------|
| R1 | **Training set contamination** 导致指标虚高、结论不可靠；审稿人/竞赛质疑 | 评测有效性 | 高 | 高 | 主线采用 contamination-aware protocol；全量 + clean subset 双报告；stress test | `src/ontobridg/eval` 优先实现；论文单设 Threats & Protocol 节 |
| R2 | **API/算力成本**超预算或速率限制打断实验 | 资源 | 中 | 中 | 统一 client 限流与缓存；baseline 用小模型；实验批处理；记录 token 用量 | `src/ontobridg/llm` 加 cache 层 |
| R3 | **UMLS license** 限制数据分发与论文复现 | 合规 | 高 | 中 | UMLS 仅本地；仓库不含受限文件；README 写明申请路径；主可复现集用 WordNet/schema.org | 检查所有 commit；`data/raw` gitignore |
| R4 | LLMs4OL 2027 **任务/数据集/指标变更** | 竞赛 | 中 | 高 | 模块化 stages；eval 指标可插拔；每周跟踪官方渠道 | `competition/README.md` 更新日志 |
| R5 | 纯 LLM 在部分 task 上 **弱于 symbolic/传统系统**，竞赛名次不佳 | 竞赛结果 | 中 | 中 | 定位 hybrid（LLM + constraints），不卖“纯 LLM”；论文贡献含协议，不单靠榜单 | 消融必须证明 constraints 必要性 |
| R6 | 2026 组合分公式与实现不一致 | 指标 | 中 | 中 | 同时报告 F1、graph sim、多种加权；以官方为准校准 | eval 可配置权重 |
| R7 | 模型 **版本漂移** 导致不可复现 | 科研质量 | 中 | 中 | 记录 model id、温度、日期、prompt 版本 hash；关键表冻结 API snapshot | experiments 元数据强制字段 |
| R8 | 污染 probe 本身 **误报/漏报** | 方法 | 中 | 中 | 多 probe 交叉验证；人工抽查；在论文中限定声明范围 | 协议节写清假设与局限 |

---

## 5. 8 周里程碑（可执行）

> 起始周记为 W1 = 2026-09-22（或项目实际 kick-off 周）。每周五对照 checklist 更新本表状态。

### W1 — 仓库与评测脚手架

- [ ] 确认目录与 git ignore（`data/raw`、任何 UMLS 文件）
- [ ] 实现数据加载：WordNet + schema.org 子集（Task A/B 字段）
- [ ] Gold 解析：types、is-a edges
- [ ] `eval` 骨架：Task A F1、Task B edge F1；graph similarity 占位实现
- [ ] 记录：LLMs4OL arXiv 2307.16648 任务定义摘要写入 `competition/README.md`
- **交付：** `python -m ontobridg.eval --dataset wordnet --task A` 在 dummy 预测上跑通

### W2 — Task A Baseline + LLM Client

- [ ] OpenAI-compatible client + 重试 + 调用日志
- [ ] Zero-shot Term Typing baseline（WordNet + schema.org）
- [ ] 结果表 v0：baseline F1
- [ ] Prompt 版本化（`prompts/taskA_v*.yaml`）
- **交付：** baseline 表进 `experiments/exp_2026W2_taskA_zs/`

### W3 — RAG few-shot + Type Constraints

- [ ] Train-only 检索库（防泄漏 checklist）
- [ ] Task A：+RAG；+inventory/type constraints
- [ ] 消融开关：RAG on/off、constraints on/off
- **交付：** 消融表；选定 Task A 默认配置

### W4 — Taxonomy Discovery Baseline + Graph Constraints

- [ ] Task B：LLM is-a 候选生成 + scoring
- [ ] 实现 acyclicity、depth control、multi-parent policy
- [ ] Baseline vs +constraints 对比（F1 + graph sim）
- **交付：** Task B 消融；系统架构图更新到 DESIGN.md

### W5 — Task C + Typed Filtering

- [ ] 关系 inventory 从 gold ontology 构建
- [ ] Domain/range 合法 pair 过滤
- [ ] LLM 抽取/分类 + symbolic 一致性检查
- **交付：** Task C 首份可报告 F1（可低于 A/B）

### W6 — Contamination Protocol MVP

- [ ] Membership-style probes（term/type/relation）
- [ ] 污染分级 + `contamination_scores.json`
- [ ] Clean subset 构建与重评
- [ ] Surface-form stress test（同义改写）
- **交付：** `eval_report.md` 含 clean vs full；论文 Threats 素材

### W7 — 组合分 + 全量实验 + 竞赛打包草案

- [ ] 实现/对齐 F1 + semantic graph similarity 组合
- [ ] 主数据集全量跑（WordNet、schema.org、GeoNames；UMLS 本地可选）
- [ ] 固定 config、seed/prompt hash、模型 id
- [ ] 提交格式草案（按往年 LLMs4OL 要求，缺失则 JSONL 自洽定义）
- **交付：** `experiments/exp_2026W7_main/` 冻结结果 + 打包脚本草案

### W8 — 论文 Draft + 复现包

- [ ] 按 `PAPER_OUTLINE.md` 完成 workshop 长度 draft
- [ ] 图：architecture、消融、contamination 对比
- [ ] README 复现命令与数据获取说明
- [ ] Preprint / 投稿决策记录（写入 STRATEGY 更新）
- **交付：** `papers/draft_v1/` + 可运行复现路径

### 每周固定动作（贯穿 W1–W8）

- 周一 30 min：检查 LLMs4OL / OAEI 官方更新 → 更新 `competition/README.md`
- 周五 30 min：里程碑勾选、风险表概率复核、实验日志补全
- 不阻塞主线的事项进 backlog，不占用 P0 任务时间

---

## 6. Decision Log

| 日期 | 决策 | 依据 |
|------|------|------|
| 2026-09-19 | 主方向 = LLMs4OL；备选 = OAEI 2027 | OAEI 2026 execution 09-05 已过；OAEI 经典赛道 hybrid symbolic 主导；LLMs4OL 更新、方法未收敛 |
| 2026-09-19 | 系统名 **OntoBridg**；方向 = hybrid neuro-symbolic + contamination-aware eval | 文章指出污染是系统性评测缺陷；纯 LLM 竞争力不足 |
| 2026-09-19 | 目标竞赛年默认 **2027**，立刻开始 build | 留足系统与论文时间 |

---

## 7. Links

- 项目设计：`D:\NLP\docs\DESIGN.md`
- 论文大纲：`D:\NLP\docs\PAPER_OUTLINE.md`
- 竞赛笔记：`D:\NLP\competition\README.md`
- LLMs4OL：arXiv 2307.16648
