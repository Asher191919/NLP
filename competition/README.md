# Competition Notes — OAEI & LLMs4OL

> 更新：2026-09-19 | 用途：竞赛事实、任务定义、链接。战略见 `docs/STRATEGY.md`。

---

## 1. OAEI vs LLMs4OL 对比

| 维度 | OAEI | LLMs4OL |
|------|------|---------|
| 全称 | Ontology Alignment Evaluation Initiative | LLMs for Ontology Learning |
| 起始年 | **2004** | **2024** |
| 核心问题 | Ontology matching / alignment | Ontology learning from text |
| 典型任务 | 实例/类/属性对齐、matching | Term typing、taxonomy discovery、non-taxonomic relation extraction |
| 方法格局 | Hybrid symbolic 系统主导：**LogMap、AML、BERTMap** 等 | 方法仍在演化；LLM-centric 与 hybrid 均在探索 |
| 纯 LLM 表现 | 经典赛道上 **往往弱于** 成熟 hybrid 系统 | 相对更有空间；仍需防污染与约束 |
| 成熟度 | 高；基线强、指标稳定 | 低–中；协议与数据仍在调整 |
| 本项目优先级 | **Secondary / backup（OAEI 2027）** | **Primary（目标 2027）** |
| 与论文契合 | Matching 侧可作污染分析延伸 | 与 contamination-aware OL 论文一体 |

---

## 2. OAEI 2026 Schedule（事实备忘）

| 节点 | 日期 | 备注 |
|------|------|------|
| Registration | **2026-07-31** | 已过 |
| Execution / 系统执行截止 | **2026-09-05** | **已过（今天 2026-09-19）→ 无法参加 OAEI 2026 执行轮** |
| Alignments 提交 | **2026-09-22** | 临近；仅观察结果，不作为参赛目标 |
| OM workshop | **2026-10-25** | **ISWC Bari**；听会/读 CEUR，不提交参赛系统 |

**含义：** OAEI 2026 竞赛窗口关闭。保留笔记与 baseline 复现，面向 **OAEI 2027** 预研。

---

## 3. LLMs4OL 赛题事实（2024 → 2026）

### 3.1 届次对照

| 年份 | 届次 | Venue | 任务编号 |
|------|------|-------|----------|
| 2024 | 1st | ISWC Baltimore | A Term Typing / B Taxonomy / C Non-Taxonomic RE（另有 FS/ZS 划分） |
| 2025 | 2nd | ISWC Nara | **A Text2Onto / B Term Typing / C Taxonomy / D Non-Taxonomic RE** |
| 2026 | 3rd | ISWC Bari（10/25–29） | **A Flagship / B Reuse / C Taxonomy** |

**注意：** 论文 arXiv 2307.16648 的 A/B/C 是研究范式任务，**不等于** 2025/2026 挑战赛编号。实现竞赛代码时以当年官网为准。

### 3.2 LLMs4OL 2026（本届已结束）

官网：https://sites.google.com/view/llms4ol2026  
数据：https://github.com/sciknoworg/LLMs4OL-Challenge/tree/main/2026  
工具：OntoLearner（https://ontolearner.readthedocs.io/，`pip install ontolearner`）

#### Task A — Flagship（End-to-End Ontology Learning）

- **输入：** raw domain text（title + content）
- **输出：** primitive ontology triples：terms、types、instance-of、is-a、non-taxonomic relations、disjoint-with 等
- **训练数据：** ~4033 samples；字段 `id / context / primitive-ontology-triples`
- **测试数据：** `id / context`，系统提交 triples
- **指标：**
  - P / R / F1（overall + Term Typing / Taxonomy / Non-Tax RE）
  - **Graph Similarity = (Edge F1 + Neighborhood Similarity + Taxonomy Similarity) / 3**
  - 官方脚本：`2026/graph_similarity_metric.py`
- **禁用训练本体（部分）：** OBI, FOAF, SchemaOrg, GeoNames, FoodOn, BFO, DBpedia 等（完整列表见官网）

#### Task B — Reuse（Ontology Extension）

- **输入：** context + `initial-primitive-ontology-triples` + given `terms` / `types`
- **输出：** `extended-primitive-ontology-triples`（新增 typing / is-a / non-tax）
- **训练数据：** ~2774 samples
- **指标：** P/R/F1 + 任务分项 F1（Term Typing / Taxonomy / Non-Tax RE）
- **强 baseline 思路：** vocabulary-constrained filtering + RAG few-shot；**非分类关系常为 0（可突破）**

#### Task C — Taxonomy（Cross-domain）

- **输入：** unseen domain 的 type list
- **输出：** `(child, is-a, parent)` edges
- **数据：** OntoLearner 180+ ontologies / 22 domains；train 可用除测试域外全部本体
- **Hold-out 测试域：** Materials Science & Engineering、Biology & Life Sciences、Scholarly Knowledge、Ecology & Environment
- **指标：** Precision / Recall / F1
- **官方示例：** `ontolearner` AutoRAGLearner + MiniLM + Qwen2.5

### 3.3 2026 时间线（均已过）

| 节点 | 日期 |
|------|------|
| 训练集发布 | 2026-04-20/26 |
| 测试集发布 | 2026-06-01 |
| 系统提交截止 | 2026-07-07（延后） |
| 参赛论文截止 | 2026-07-28（延后） |
| 录用通知 | 2026-08-10 |
| Camera-ready | 2026-08-31 |
| Workshop | 2026-10-25/26 @ ISWC Bari |

**含义：** 2026 参赛窗口关闭。当前策略 = **用公开数据与指标复现/超越 baseline → 写论文 → 冲 LLMs4OL 2027**。

### 3.4 2026 参赛方法（Accepted Papers，已录用）

| 队伍 | 任务 | 方法要点 |
|------|------|----------|
| LaboroAI | Flagship+Reuse | Less is More for LLM-based OL |
| ELLMO | A+B+C | Ontology Learning Across Contexts |
| LettRAGraph | Flagship | Prompt-Guided Extraction + Ensembling |
| pro-team | Flagship+Reuse | Qwen2.5-14B RAG + vocab filter；**non-tax=0** |
| GenuineKnowledge-KEOL | A+B+C | Hybrid Training-Free Pipelines |
| VUL337-OL | Flagship+Reuse | Task-Native Constraints |
| Volt-IQ | 全系列 | Constrained Graph Synthesis + Domain Calibration |
| semantic-swingers | A+B+C | Retrieval + FT + Seen-Unseen Gap |
| LIST-CortAIx | Flagship | Stage-wise Modular LLM Pipeline |
| XuanJi | Flagship | Partial Graph-Oriented System |
| HybridAI-Squared | Reuse | Divide and Conquer（结论：未必有用） |
| SEMA | Taxonomy | Continual Pretraining on Serialized Ontologies |
| BigDipperAI | Flagship | Schema-Aligned Auxiliary Supervision |

**我们的相对优势点：**
1. Open non-taxonomic relation vocabulary + 符号约束修复（直接打 pro-team 的 0）
2. Contamination-aware evaluation protocol（论文贡献）
3. Graph Similarity 导向的联合优化（Edge F1 + Neighborhood + Taxonomy 三分量）
4. 覆盖 Flagship + Reuse + Taxonomy 的统一 OntoBridg 管线

### 3.5 范式任务定义（研究对照，arXiv 2307.16648）

- **Term Typing：** term → type 集合
- **Taxonomy Discovery：** types → is-a 边
- **Non-Taxonomic RE：** (h, r, t) 非分类关系
- 历史数据源：WordNet / GeoNames / UMLS / schema.org / GO / FoodOn / DBpedia

### 3.6 Contamination 注记（本项目研究点）

- WordNet / UMLS / schema.org / GeoNames 等资源很可能出现在 LLM 预训练语料中。
- 2026 官方已列 **禁止用于训练的本体清单**（OBI, SchemaOrg, GeoNames, FoodOn…），说明主办方已意识到污染/泄漏风险。
- 风险：test term 的类型、is-a、关系被“背过”，F1 虚高。
- 项目协议：membership-style probes、clean/polluted 分报告、synthetic novel ontology stress test、surface-form 变换（见 `docs/DESIGN.md` Stage 6）。

---

## 4. OAEI 侧备忘（Backup 2027）

| 信息 | 内容 |
|------|------|
| 经典强系统 | LogMap、AML、BERTMap（及各 track 专用系统） |
| 纯 LLM 风险 | 在 classic matching tracks 上常 **弱于** hybrid symbolic + embedding 系统 |
| 2027 预研范围 | 仅跟踪结果、复现 1 个 baseline、评估 OntoBridg matching 模块移植成本 |
| 不在 2026 做的事 | 不提交 alignments；不把资源投入 OAEI 2026 收尾 |

---

## 5. Links

| 资源 | 链接 / 编号 |
|------|-------------|
| LLMs4OL series | https://sites.google.com/view/llms4ol |
| LLMs4OL 2026 | https://sites.google.com/view/llms4ol2026 |
| LLMs4OL 2025 | https://sites.google.com/view/llms4ol2025 |
| LLMs4OL 2024 | https://sites.google.com/view/llms4ol2024 |
| Challenge data | https://github.com/sciknoworg/LLMs4OL-Challenge |
| OntoLearner docs | https://ontolearner.readthedocs.io/ |
| OntoLearner GitHub | https://github.com/sciknoworg/OntoLearner |
| LLMs4OL paper | https://arxiv.org/abs/2307.16648 |
| 2024 overview | https://arxiv.org/abs/2409.10146 |
| Alexbek 2025 | https://arxiv.org/abs/2508.19428 |
| pro-team 2026 | https://arxiv.org/abs/2608.27101 |
| OAEI | https://oaei.ontologymatching.org/ |
| OM workshop | 与 ISWC 系列相关（2026：Bari，10-25） |
| ISWC | https://iswc2026.semanticweb.org/ |
| WordNet | https://wordnet.princeton.edu/ |
| schema.org | https://schema.org/ |
| GeoNames | https://www.geonames.org/ |
| UMLS | https://www.nlm.nih.gov/research/umls/ （**license 受限，需申请**） |

**内部文档：**

- `D:\NLP\docs\DESIGN.md` — 系统设计
- `D:\NLP\docs\STRATEGY.md` — 战略与里程碑
- `D:\NLP\docs\PAPER_OUTLINE.md` — 论文大纲
- `D:\NLP\README.md` — 项目总览

---

## 6. Update Log

| 日期 | 变更 |
|------|------|
| 2026-09-19 | 建立笔记；记录 OAEI 2026 时间事实；确认主方向 LLMs4OL（目标 2027） |
| 2026-09-19 | 按官网修订 LLMs4OL 2024–2026 任务编号、指标公式、2026 录用队伍与突破点 |
