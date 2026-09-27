# Paper Outline — OntoBridg / Contamination-aware Ontology Learning

> 版本：v0.1 | 日期：2026-09-19 | 配套：`DESIGN.md` / `STRATEGY.md`

---

## 1. Title Options

**推荐主标题（workshop / 早期）：**

1. **OntoBridg: Hybrid Neuro-Symbolic Ontology Learning with Contamination-Aware Evaluation**

**备选：**

2. Training Set Contamination in LLM-Based Ontology Learning: A Diagnostic Protocol and a Hybrid Pipeline  
3. Beyond Raw F1: Contamination-Aware Evaluation for Ontology Learning with Large Language Models  
4. Constraints over Generators: Reliable Taxonomy Discovery for LLMs4OL  
5. When LLMs “Know” the Ontology: Measuring and Mitigating Contamination in Ontology Learning

**标题策略：** workshop 版用 1（系统 + 协议双卖点）；若污染实验成为主要贡献，扩展版用 2 或 3。

---

## 2. Abstract Skeleton

> 按句填充；删除线句为可选。目标 workshop：~200 词。

1. **背景：** Large language models (LLMs) 被广泛用于 ontology learning 与 matching，LLMs4OL 等评测推动了该方向。
2. **问题：** 但常用资源（如 WordNet、UMLS、schema.org）可能存在于 LLM 预训练数据中，导致 **training set contamination**，官方 F1 可能系统性偏高。
3. **方法：** 我们提出 **OntoBridg**，hybrid neuro-symbolic pipeline：LLM 在 term typing、taxonomy discovery 与 non-taxonomic relation extraction 上生成候选，**symbolic graph constraints**（type inventory、acyclicity、domain/range）进行校正，**RAG few-shot** 提供可追溯上下文。
4. **评测协议：** 同时提出 contamination-aware protocol：membership-style probes、污染分级、**clean vs polluted subset** 分报告，以及 surface-form stress test。
5. **结果：** 在 LLMs4OL 相关数据集（WordNet、schema.org，及可选 GeoNames/UMLS）上，constraints 在 clean subset 上稳定提升 F1 与 semantic graph similarity；污染等级与性能虚高幅度相关。
6. **结论：** 竞赛与论文均不应只报单一 F1；我们给出可复现系统与诊断协议。

---

## 3. Contributions（论文列表，与 DESIGN 对齐）

1. **系统：** OntoBridg — 面向 LLMs4OL 任务族的 hybrid pipeline（Stage 2–5 + 可插拔约束），提供可复现基线与消融。
2. **协议：** Ontology learning 场景下可操作的 **training set contamination** 评测协议（probes + 分级 + clean-split + stress test）。
3. **实证：** 证明在高污染子集上 F1 / semantic graph similarity 虚高；显式 graph constraints 在 clean subset 上的增益更可信、更稳定。
4. **（可选扩展）** Contamination-robust 配置调度或 definition-conditioned prompting，进一步缩小 clean/full 差距。

---

## 4. Method Sections（建议结构）

### 4.1 Task Formalization（对外用 **LLMs4OL 2026** 编号）

> 研究范式子能力（typing / taxonomy / non-tax RE）仍作为方法模块描述，但实验与主张对齐 2026 赛题。

**LLMs4OL 2026 竞赛任务：**

- **Task A Flagship：** 给定 raw text，构造 primitive ontology triples（terms, types, `instance-of`, `is-a`, non-tax relations）  
  \(\text{context} \mapsto \hat{T} \subseteq \mathcal{E}\times\mathcal{P}\times\mathcal{E}\)
- **Task B Reuse：** 给定 partial ontology + terms/types + text，预测 **extended** triples  
  \((\text{context}, O_{\text{init}}, \mathcal{T}_{\text{cand}}) \mapsto \hat{T}_{\text{ext}}\)
- **Task C Taxonomy：** 跨域 taxonomy discovery；源域本体训练，hold-out 域预测 `(child, is-a, parent)`

**方法子模块（pipeline stages，不与赛题编号混淆）：**

- Sub能力 Typing：\((t, \text{ctx}) \mapsto \hat{Y} \subseteq \mathcal{T}\) — 服务 Flagship/Reuse
- Sub能力 Taxonomy：\(C \mapsto \hat{E}_{\text{is-a}}\) — 服务 A/B/C
- Sub能力 Non-tax RE：开放词表 \(\hat{R} \subseteq \mathcal{E}\times\mathcal{R}_{\text{open}}\times\mathcal{E}\) — 服务 Flagship/Reuse 图完整性（pro-team 缺口）

**评测（对外主指标，禁用自造 SGS）：**

- Term-Typing F1、Taxonomy F1
- **Official Graph Similarity** = \((\text{EdgeF1}+\text{NeighborhoodSim}+\text{TaxonomySim})/3\)  
  实现：`ontobridg.eval.official_graph_similarity`（对齐 `data/LLMs4OL-Challenge/2026/metrics/graph_similarity.py`）
- 组合分参考 pro-team：\((\text{GraphSim}+\text{TypingF1}+\text{TaxonomyF1})/3\)
- 内部 proxy SGS 仅 ablation，**不得**写入主表/摘要

### 4.2 OntoBridg Architecture

- 总览图（多阶段 pipeline；标注 LLM vs symbolic 模块）
- **Stage 2 Typing：** RAG few-shot（train-only 索引）+ LLM 分类 + inventory/type-lattice 约束
- **Stage 3 Taxonomy：** 候选 is-a 生成 → scoring → **acyclicity / depth / multi-parent** 约束 → 边集一致化
- **Stage 4 Relations：** typed pair filtering + LLM 抽取 + domain/range 与 typing 一致性
- **Stage 5 Assembly：** 统一 graph；可选 OWL/RDF 一致性检查
- Prompt 设计附录索引；模型与 API 元数据（model id、温度、日期）

### 4.3 Contamination-Aware Evaluation Protocol

- **动机与威胁模型：** 何为 OL 场景下的 contamination；与 classic train/test leakage 的区别
- **Probes：** 定义/补全/子类判断等 membership-style 信号；随机对照
- **污染分数：** dataset–task 级 low/med/high
- **报告规范：** Full / Clean / Stress 三套指标；禁止只报 Full
- **与竞赛兼容：** 提交仍用官方分；论文与附录给诊断

### 4.4 Symbolic Constraints（实现细节）

- 图算法：环检测与剪枝、传递闭包策略、深度惩罚
- Type / relation 逻辑约束的形式
- 约束开关与“soft vs hard”配置

---

## 5. Experiments Plan

### 5.1 实验问题

| ID | 问题 |
|----|------|
| Q1 | OntoBridg vs baselines，在 **2026 Task A Flagship / Task B Reuse / Task C Taxonomy** 上 official GraphSim + F1 差多少？ |
| Q2 | RAG few-shot 与 symbolic constraints 各自贡献？（消融） |
| Q3 | 数据集污染等级如何？与性能虚高是否相关？ |
| Q4 | Clean subset 上 constraints 增益是否仍显著？ |
| Q5 | F1 与 semantic graph similarity 是否同步膨胀？ |
| Q6 | （可选）与公开 LLMs4OL / 相关 OL 系统对比？ |

### 5.2 数据集与划分

| 数据集 | Task | 公开可复现 | 备注 |
|--------|------|------------|------|
| WordNet | A, B | 是 | 主实验；污染研究材料多 |
| schema.org | A, C | 是 | type inventory 清晰 |
| GeoNames | A, B | 是 | 层次变体 |
| UMLS (NCI/MEDCIN/SNOMEDCT) | A/B/C | **否（license）** | 本地跑；论文聚合结果 + 获取说明 |

- 划分：严格 train-only 用于 RAG 索引；test 从不进入检索库。
- 记录：每条实验的 config hash、model id、温度、日期。

### 5.3 系统配置（消融表列）

| 配置 | RAG | Graph constraints | Contamination eval |
|------|-----|-------------------|--------------------|
| ZS-LM | 否 | 否 | 是（诊断） |
| +RAG | 是 | 否 | 是 |
| +Constraints | 否/是 | 是 | 是 |
| OntoBridg (full) | 是 | 是 | 是 |

### 5.4 指标

- Official task metrics：P/R/F1（按当年协议 micro/macro）
- Semantic graph similarity（实现自 `src/ontobridg/eval`；权重可配置）
- 组合分（对齐 2026 提法，公式写清）
- Contamination：probe AUC/分离度；clean–full ΔF1；stress 掉幅

### 5.5 表/图清单

| 编号 | 内容 |
|------|------|
| Fig.1 | OntoBridg architecture |
| Fig.2 | Contamination protocol flow |
| Tab.1 | Main results（dataset × task × system） |
| Tab.2 | Ablation（RAG / constraints） |
| Tab.3 | Contamination levels per dataset-task |
| Tab.4 | Full vs Clean vs Stress |
| Fig.3 | ΔF1 vs contamination level（scatter/bar） |

### 5.6 复现

- 主命令示例写入项目 README；非 UMLS 路径一键复现
- 随机性：温度与 top_p 固定；多次运行报均值±标准差（若预算允许）

---

## 6. Related Work Notes

### 6.1 必须覆盖

| 主题 | 要点 | 线索 |
|------|------|------|
| Ontology learning 经典 | 从文本构建 taxonomy/关系的传统 pipeline | 综述检索：ontology learning survey |
| LLM + ontology matching/learning | LLMs4OL 赛事与同期方法 | **arXiv 2307.16648** |
| Hybrid symbolic matching | LogMap、AML、BERTMap 等；说明为何 OL 与 matching 不同 | OAEI 文献 / OM CEUR |
| Neuro-symbolic / constrained decoding | 结构约束提升可靠生成 | 近年 ACL/EMNLP/NeurIPS 约束生成工作 |
| **Data contamination in LLM eval** | 核心相关工作：benchmark contamination、membership inference / memorization 评估 | 检索：LLM benchmark contamination, training data leakage |
| Ontology evaluation | F1 之外的 graph-level / semantic similarity 指标 | OAEI 与 ontology evaluation 文献 |

### 6.2 论述立场（Related Work 结尾）

- Matching（OAEI）已由 hybrid symbolic 系统定义了强基线；**OL（LLMs4OL）仍缺既强又“评测干净”的系统**。
- 既有 contamination 工作多针对 QA/benchmark；**本文将其落到 ontology learning 的类型、层次与关系三类输出上**。

### 6.3 检索 TODO

- [ ] LLMs4OL 后续年份报告与上榜系统（更新 SOTA 表）
- [ ] 3–5 篇 LLM benchmark contamination 代表作（记入 `papers/refs/`）
- [ ] Semantic graph similarity 在 ontology alignment/learning 中的先例定义

---

## 7. Target Venues

| 档次 | Venue | 类型 | 适合版本 | 备注 |
|------|-------|------|----------|------|
| Workshop | **OM workshop**（Ontology Matching，ISWC 系列，CEUR） | Workshop / CEUR | 首投 | 与 OAEI/OL 社群交集大；2026 为 10-25 Bari（参赛已不可，论文需看当年 CFP 是否仍收） |
| Workshop | 其他 LLM + Semantic Web / Knowledge Graph workshop | Workshop | 备选 | 随 ISWC/AAAI/ACL 附带 |
| Conference | **ISWC** | 语义 Web 主会 | 扩展版 | 社群对口 |
| Conference | **AAAI / EMNLP / ACL Findings** | AI / NLP | 扩展版 | 若污染+约束叙事足够一般化 |
| Journal | **Knowledge-Based Systems** | 期刊 | 长文 | 实验与协议需完整 |
| Journal | Semantic Web Journal（备选） | 期刊 | 长文 | 社群对口 |

**投稿策略：**

1. Week 8：workshop 长度 draft（系统 + 初步污染诊断）。
2. 有竞赛/扩展实验后：主会或 Findings。
3. 协议成熟 + 多数据集 + 更深分析：KBS 等期刊。

---

## 8. Author Checklist（提交前）

- [ ] 贡献列表与正文实验一一对应
- [ ] 所有数据集 license 语句正确（UMLS 不公开数据）
- [ ] Contamination 协议假设与局限写清（Threats to Validity）
- [ ] 官方指标公式注明出处/年份
- [ ] 图与表编号稳定；代码与 commit hash 可引用
- [ ] 匿名化（如需要）：去除路径与个人标识

---

## 9. File Plan

```
D:\NLP\papers\
  draft_workshop_v1\
    main.tex 或 main.md
    figures\
    tables\
    refs.bib
  notes\
    related_work.md
    venue_cfp.md
```
