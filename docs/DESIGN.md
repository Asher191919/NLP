# OntoBridg — 项目设计文档

> 版本：v0.2 | 日期：2026-09-20 | 状态：任务编号已对齐 LLMs4OL 2026
>
> **任务编号：** 对外一律用 2026 赛题名（Flagship / Reuse / Taxonomy）。研究范式子能力（Typing / Taxonomy / Non-tax RE）只作 pipeline 模块名，不再写成竞赛 Task A/B/C。

---

## 1. Style / Research Anchor

**研究锚点：** Neuro-symbolic Ontology Learning + Contamination-aware Evaluation

| 维度 | 选择 |
|------|------|
| 问题域 | Ontology Learning from Text（LLMs4OL 任务族） |
| 方法风格 | Hybrid：LLM 生成 + symbolic graph constraints + RAG few-shot |
| 评价风格 | Contamination-aware protocol（训练集污染检测 + 干净子集报告） |
| 交付物 | 竞赛系统（LLMs4OL 2027）+ 1–2 篇论文 |
| 系统代号 | **OntoBridg**（Ontology Bridge：文本 → 结构化本体的桥接管线） |

**核心假设（H1–H3）：**

- **H1**：在 LLMs4OL 的 Term Typing / Taxonomy Discovery / Relation Extraction 上，纯 LLM zero-shot 存在天花板；加入 symbolic constraints（类型一致性、taxonomy 无环、关系域/值约束）可显著提升 F1 与 graph similarity。
- **H2**：LLMs4OL 数据集（WordNet / UMLS / schema.org 等）在主流 LLM 预训练语料中存在不同程度的 **training set contamination**；污染会导致 reported metrics 虚高。
- **H3**：Contamination-aware evaluation（检测污染 + 在干净子集/合成子集上重测）是可发表的系统性贡献，且对竞赛排名本身也可能构成优势（更稳的泛化表现）。

---

## 2. Chosen Direction and Why

### 2.1 主方向：LLMs4OL（Ontology Learning）

| 理由 | 说明 |
|------|------|
| 竞赛窗口 | LLMs4OL 自 2024 年起，2027 轮次可现在开始准备；OAEI 2026 execution deadline（2026-09-05）已过 |
| 技术格局 | OAEI classic tracks 被 hybrid symbolic 系统主导（LogMap, AML, BERTMap 等）；纯 LLM 方案常更差、难以突破 |
| 方法空间 | LLMs4OL 任务定义与评测协议仍在演进，方法创新空间大，适合 awards + papers 双目标 |
| 论文契合 | 文章指出的 training set contamination 问题直接适用于 ontology matching/learning 的 LLM 评测，与主方向一体 |

### 2.2 备选方向：OAEI 2027 预研（Secondary / Backup）

- 不参与 OAEI 2026 执行，但跟踪其结果与 workshop（OM @ ISWC）。
- 若 LLMs4OL 2027 时间线不利或数据授权受阻，可在 2027 年上半年切换到 OAEI 预注册与系统移植（OntoBridg 的 matching 模块可复用）。
- 产出保留：OAEI 相关笔记与 baseline 复现记录（见 `competition/README.md`）。

### 2.3 明确不做的事

- 不做 OAEI 2026 参赛（deadline 已过）。
- 不做纯 prompt-only LLM 方案作为最终系统（仅作 baseline）。
- 不在未确认 license 前将 UMLS 衍生数据纳入公开仓库。

---

## 3. System Architecture: OntoBridg

### 3.1 总览

多阶段 pipeline：**Term Extraction → Term Typing → Taxonomy Discovery → Non-Taxonomic Relation Extraction**，全程注入 **symbolic graph constraints** 与 **RAG few-shot**，末端以 **contamination-aware eval** 收口。

```
                    ┌─────────────────────────────────────────┐
                    │           OntoBridg Pipeline            │
                    └─────────────────────────────────────────┘

  Raw Text / Docs          Stage 1              Stage 2
  + Gold Terms      ──►  Term Extraction ──►  Term Typing
  + Ontology TBox        (LLM + rules)        (LLM + RAG
                                                + type lattice)

       Stage 3                  Stage 4              Stage 5
  Taxonomy Discovery ──► Non-Tax Relation   ──►  Ontology
  (is-a edges +           Extraction              Assembly
   acyclicity +           (domain/range +
   transitive closure)     relation inventory)    (OWL/RDF)

                              │
                              ▼
                    Contamination-Aware Eval
                    (F1 + semantic graph sim
                     + clean/polluted split)
```

### 3.2 Stage 设计

#### Stage 0 — Data & Ontology Prep

- 主数据：LLMs4OL 2026 官方 `data/LLMs4OL-Challenge/2026/{TaskA-Flagship,TaskB-Reuse,TaskC-Taxonomy}`
- 辅助/污染分析：WordNet、GeoNames、UMLS、schema.org（注意官方禁止训练清单）
- 解析 gold ontology 的 TBox：classes、`is-a`、object properties、domain/range。
- 构建 **prompt-side** 与 **eval-side** 隔离：eval gold 不进入 few-shot 检索库（防泄漏）。

#### Stage 1 — Term Extraction（内部模块，非 LLMs4OL 官方 Task，但服务 pipeline）

- 输入：领域文本。
- 方法：LLM candidate generation + 候选规范化（lowercase、去停用词、noun phrase filter）。
- 输出：`terms.jsonl` — `{id, surface, context}`。

#### Stage 2 — Term Typing（服务 **2026 Task A Flagship / Task B Reuse**；研究范式 Task A）

- 输入：term + 可选 context + 目标 type inventory（来自 gold 或外部 schema）。
- 方法：
  - **RAG few-shot**：从 *训练划分*（非 test）检索相似 term–type 对，注入 prompt。
  - **LLM 分类**：输出 type 标签；多标签时输出 ranked list。
  - **Symbolic constraint**：type 必须存在于 inventory；与已有 taxonomy 层级一致（子类不可单独出现而父类缺失——作为 soft/hard 约束可配置）。
- 输出：`term_types.jsonl` — `{term, types[]}`。

#### Stage 3 — Taxonomy Discovery（服务 **2026 Task A/B/C**；研究范式 Task B）

- 输入：已 typing 的 term 集合 + 可选 embeddings。
- 方法：
  - **Candidate generation**：LLM 生成 is-a 候选对；embedding 近邻（若有）作辅助候选。
  - **Scoring**：LLM confidence + definition similarity（若有 gloss）+ 结构先验。
  - **Graph constraints（symbolic）**：
    - **Acyclicity**：禁止环（Union-Find / DFS 检测 + 违规边剪枝）。
    - **Single/multi-parent policy**：按数据集策略（WordNet 多父常见，UMLS 亦然；默认允许多父但惩罚过宽节点）。
    - **Transitive reduction / consistency**：输出前做边集一致化，避免冗余直接父边。
  - **层次深度控制**：防止全部边指向 root 或全部指向 leaf 的退化解。
- 输出：`taxonomy.jsonl` — `{child, parent, score}`。

#### Stage 4 — Non-Taxonomic Relation Extraction（服务 **2026 Task A/B 非分类三元组**；研究范式 Task C）

> 2026 赛题无独立 “Non-tax RE Task”；非分类关系属于 Flagship/Reuse 输出图。

- 输入：term 对 + type 信息 + 关系 inventory（来自 ontology 或开放抽取）。
- 方法：
  - **Typed pair filtering**：仅对合法 (domain, range) 类型组合抽取，降低组合爆炸。
  - **LLM relation classification / extraction**：给定 context 或 term 定义，输出 relation label + score。
  - **Symbolic constraint**：relation 的 domain/range 必须与 typing 结果一致；禁止 taxonomy 关系与 non-taxonomy 关系标签冲突。
- 输出：`relations.jsonl` — `{head, tail, relation, score}`。

#### Stage 5 — Ontology Assembly

- 合并 Stage 2–4 输出为统一 graph。
- 可选导出 OWL/RDF（`rdflib`），便于外部 reasoner 做 consistency check。
- 产出：`ontology_final.json` / `ontology_final.owl`。

#### Stage 6 — Contamination-Aware Evaluation（核心差异化）

- **指标（对外）**：
  - Official：Term-Typing F1、Taxonomy F1
  - **Official Graph Similarity** = `(EdgeF1 + NeighborhoodSim + TaxonomySim)/3`
    - 实现：`src/ontobridg/eval/official_graph_similarity.py`
    - 对齐：`data/LLMs4OL-Challenge/2026/metrics/graph_similarity.py`
    - **禁止**把内部 proxy SGS 当作官方 SGS 对外报告
  - 组合分参考：`(GraphSim + TypingF1 + TaxonomyF1)/3`
- **内部诊断：** proxy SGS、combo、per-type F1 等，仅 ablation。
- **污染评估协议（贡献点）**：
  1. **Membership-style probes**：对 gold term/type/relation 与随机对照，查询 LLM 补全/定义/子类关系，估计 memorization 信号。
  2. **Contamination score**：每个 dataset–task 对得到污染等级（low/med/high）。
  3. **Split reporting**：在 full test、clean subset（低污染项）、synthetic holdout（人工构造或时间切分）上分别报告。
  4. **Prompt-augmented stress test**：改写 gold 表述（同义替换、罕见表面形式）观察性能掉幅 — 掉幅大 ⇒ 依赖表面记忆。
- 产出：`eval_report.md` + `contamination_scores.json`。

### 3.3 数据流与目录约定

```
D:\NLP\
  data\
    raw\              # 下载的官方/公开数据（UMLS 受限则不入库）
    processed\        # 归一化后的 JSONL
  src\
    ontobridg\
      stages\         # stage0..stage6
      constraints\    # symbolic graph ops
      rag\            # retrieval + prompt builders
      llm\            # API client wrappers
      eval\           # metrics + contamination probes
  experiments\
    exp_YYYYMMDD_*\   # 配置 + 结果 + 日志
  papers\
    draft_*\          # 论文素材
  assets\             # 图、图示
  competition\        # 竞赛笔记与提交相关
  docs\               # 本文档集
```

---

## 4. Novel Contributions（论文级，3–4 条）

1. **OntoBridg：面向 LLMs4OL 的 hybrid neuro-symbolic ontology learning pipeline**  
   在统一框架中耦合 LLM 生成、RAG few-shot 与显式 graph constraints，在 **2026 Task A Flagship / Task B Reuse / Task C Taxonomy** 上给出可复现基线与消融；非分类关系作为 Flagship/Reuse 图完整性子模块单独消融。

2. **Ontology Learning 场景下的 training set contamination 系统性评测协议**  
   将 contamination 从“担忧”变成可测量协议：membership probes、污染分级、clean vs polluted 分报告、surface-form stress test。指出纯 F1 在高污染数据集上的不可靠性。

3. **Semantic graph similarity 与 contamination 的交互分析**  
   实证检验：高污染条件下 F1 与 semantic graph similarity 是否同步虚高；constraints 是否能在 clean subset 上稳定提升 graph-level 指标（而不仅是 edge F1）。

4. **（可选）Contamination-robust decoding / constraint scheduling**  
   如时间允许：在检测到高污染或低置信时切换约束强度（hard → soft）或改用 definition-conditioned prompts，报告对 clean subset 的增益。

---

## 5. Technical Stack

| 层 | 选择 | 备注 |
|----|------|------|
| 语言 | Python 3.11+ | 主实现 |
| LLM 访问 | Open LLM APIs（OpenAI-compatible / 本地推理） | 统一 client 抽象，便于换模型 |
| 可选本地模型 | `transformers` + 本地 checkpoint | 小模型做 typing baseline；大任务仍走 API |
| 图约束 | 自研 + `networkx` | acyclicity、closure、consistency |
| 可选 OWL/RDF | `rdflib`（可选 reasoner） | 导出与 consistency check |
| 检索 / RAG | `sentence-transformers` 或 API embeddings + 向量检索（numpy/faiss 可选） | few-shot 检索 |
| 数据处理 | `pandas`, `datasets`（HF） | 数据集加载与切分 |
| 实验跟踪 | 本地 JSON/Markdown + git | 不强制 MLflow |
| 评测 | 自研 `ontobridg.eval` | F1 + graph similarity + contamination probes |
| 配置 | YAML/JSON per experiment | `experiments/exp_*/config.yaml` |

**依赖原则：** 默认可仅靠 API + 少量库跑通；`transformers` / OWL 栈为可选增强，不阻塞主路径。

---

## 6. Success Criteria

### 6.1 竞赛指标（LLMs4OL 2027）

| 目标 | 标准 |
|------|------|
| 提交完整性 | 2026 Task A Flagship + Task B Reuse 必做；Task C Taxonomy 尽力（OntoLearner hold-out） |
| Official score | 目标进入各 task **top-3** 或总分 **top-3**（视当年设置）；底线：显著超过公开 LLM baseline |
| Ablation 可辩护 | constraints on/off、RAG on/off 的增益在报告中清晰 |
| 污染说明 | 提交附带 contamination 诊断附录（不丢分的前提下展示系统性） |

### 6.2 论文质量

| 目标 | 标准 |
|------|------|
| 一作工作流 | 8 周内产出可提交 workshop draft（OM / 相关 CEUR） |
| 方法完整 | Pipeline + contamination protocol + 消融齐备 |
| 可复现 | 公开代码结构、配置、非受限数据的 eval 脚本；UMLS 按 license 说明获取方式 |
| 下一档 | Workshop → 扩展投会议/期刊（ISWC / AAAI / EMNLP Findings / KBS） |
| 证据等级 | 所有 claim 有表/图支撑；污染协议有对照实验 |

### 6.3 工程完成度（DoD）

- [ ] Stage 2–6 可在至少 2 个数据集（建议 WordNet + schema.org 或 GeoNames）上端到端跑通
- [ ] `eval_report.md` 自动生成（含 clean/polluted 分列）
- [ ] 单条命令复现主实验（README 中记录 exact command）
- [ ] 实验目录规范：config + results + notes
- [ ] 未确认 license 的数据不进入 git

---

## 7. Open Questions / Risks（详见 STRATEGY.md）

- LLMs4OL 2027 确切时间线与任务是否变更（需跟踪官网/arXiv）
- 2026 提及的 F1 + semantic graph similarity 的精确公式与权重
- UMLS 许可证对代码/数据分发的限制
- API 成本与模型版本漂移对复现性的影响（记录 model id + date + temperature）

---

## 8. Related Pointers

- LLMs4OL paper: arXiv **2307.16648**
- Competition notes: `D:\NLP\competition\README.md`
- Strategy & milestones: `D:\NLP\docs\STRATEGY.md`
- Paper outline: `D:\NLP\docs\PAPER_OUTLINE.md`
