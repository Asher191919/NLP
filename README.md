# OntoBridg — LLMs4OL 本体学习竞赛与论文项目

> **主方向：LLMs4OL（Ontology Learning）→ 目标 2027 + 论文**  
> 系统代号 **OntoBridg** | 本地根目录 `D:\NLP` | 更新 2026-09-27

## 一句话结论

OAEI/LLMs4OL 2026 提交窗口已过，项目面向 **LLMs4OL 2027 + 论文发表**。当前证据支持的主张是：无约束开放关系生成在 exact-match 下严重过生成；符号约束应隔离在关系通道，并在校准支持不足时 fail closed。冻结的 nonce 合成诊断曾显示 guarded-v2 的关系召回/F1 明显提高，但随后完成的官方训练文本迁移实验中两套系统均为 0/101 exact non-tax TP；guarded-v2 只减少误报并改善整体 GS。因此官方或自然语料上的稳定关系增益仍未建立，不能宣称已经超过 2026 优胜方法。

## 已就绪资产

| 类别 | 路径 | 状态 |
|------|------|------|
| 系统代码 | `src/ontobridg/` | 可运行（mock LLM 离线） |
| 竞赛仿真 | `src/run_competition_sim.py` | OntoBridg vs pro-team 对比出数 |
| 官方评测脚本 | `src/run_official_2026_eval.py` | 官方 2026 train 子集离线评测 |
| 官方数据 2024–2026 | `data/LLMs4OL-Challenge/` | train/test + metrics |
| OntoLearner 源码 | `data/OntoLearner/` | clone 完成 |
| 优胜方法代码 | `data/alexbek/`, `data/LLMs4OL-baseline/` | 对照 baseline |
| 设计/战略 | `docs/DESIGN.md`, `docs/STRATEGY.md`, `docs/INNOVATION.md` | 已写 |
| 竞赛事实 | `competition/README.md`, `competition/OFFICIAL_2026.md` | 已按 2026 官网修订 |
| 论文包 | `papers/paper_draft.md` 等 + refs PDF | 骨架完成 |
| 实验结果 | `experiments/results/*.json` | demo / sim / official eval |

## 当前主要证据（官方 2026 train，blind n = 30）

| Task | 系统 | Typing F1 | Taxonomy F1 | NonTax F1 | Official GS | non-tax pred/gold |
|------|------|-----------|-------------|-----------|-------------|-------------------|
| Flagship A | `closed_vocab` | **0.1400** | **0.4903** | 0.0000 | **0.4404** | 0 / 1 |
| Flagship A | `open_vocab` | 0.1356 | 0.3516 | 0.0000 | 0.1603 | 384 / 1 |
| Reuse B | `closed_vocab` | 0.8824 | **0.9474** | 0.0000 | **0.9375** | 0 / 0 |
| Reuse B | `open_vocab` | **1.0000** | 0.8966 | 0.0000 | 0.6449 | 150 / 0 |

这些是公开训练样本上的泄漏受控实验，不是隐藏测试集或榜单成绩。Task A/B 的 open GS 分别比 closed 低 0.2801/0.2927；由于该前缀只有 1/0 条 non-tax gold，关系召回结论主要依赖后续密度分层诊断。完整证据、guarded v1/v2 结果和限制见 `papers/results_skeleton.md`。

最新的 30 样本双模型弱标注只解锁了探索性本地校准：模型间 macro Jaccard 为 0.3646，产物被强制标记为 `deployable=false`、`confirmatory_eligible=false`，不能替代人工 gold 或触发 confirmatory API 实验。

最新的 v5 独立验证和 confirmatory 结果均为负面：单复数归一化加 bounded 反向补边没有恢复 TP，反而增加 8 个 FP，因此不进入生产 canonicalizer。随后冻结的 24 样本 endpoint-visible confirmatory slice 完成 24/24 调用；冻结 calibrator 将 Brier/ECE 从 0.6707/0.6840 改善到 0.2109/0.1238，却使 exact TP 从 4 降到 3、F1 从 0.1159 降到 0.0882，低于预注册的 5 个正例样本门槛。因此证据有效但 gate 失败，`scientific_deployable=false`、`confirmatory_eligible=false`，不得事后调阈值。

确认集的 10,000 次样本簇 bootstrap 显示不确定性仍然较宽：base/calibrated F1 的 95% 区间分别为 [0.0238, 0.2712] 和 [0, 0.2222]。冻结的官方 semantic secondary metric（阈值 0.90、Nomic revision 固定）对 base 和 calibrated 都是 0/24 样本发生变化，说明该负面结果不能由简单语义表面对齐挽回。n≈122 仅是把 pooled precision 半区间缩到约 0.10 的规划估算，目前不作为调参或立即扩样依据。

污染诊断 pilot 也已闭环。对同一冻结的 Task A/B 各 15 个样本，214 个平衡的 gold/control 闭卷事实项得到 AUROC 0.9780（样本簇 bootstrap 95% CI [0.9484, 0.9982]），说明这些官方事实对该端点高度可探测；但探针分数与逐样本 GS 的相关性弱且方向不一致。把所有合法输入中的 gold 端点替换为确定性不透明别名后，Task B 的 closed/open GS 分别下降 0.1557 [0.0362, 0.3022] 和 0.1917 [0.0609, 0.3291]，Task A 的点估计却改善 0.0921/0.1102，区间跨 0。该结果只支持任务依赖的表面敏感性，**不证明 benchmark 出现在模型训练集**，也不支持 non-tax 污染结论。

为补上 non-tax 支持不足的问题，冻结的 `nonce_relation_holdout_v1` 已完成 60/60 个单元（累计 67 次底层尝试；完成后的缓存审计为 60/60 命中、0 次外部调用）。在 120 条 non-tax gold 上，`open_vocab` 的 exact P/R/F1 为 0.2222/0.3167/0.2612，`open_vocab_guarded_v2` 为 0.8644/0.8500/0.8571；两者在 60 个否定共现 hard negatives 上均为 0 FP。guarded-v2 的 canonical/paraphrased F1 为 1.0000/0.7119，显示受控模板上的明显增益与表面改写缺口。该包使用 300 个唯一不透明实体，与 2026 官方 train 的实体字符串和 exact gold 三元组重叠均为 0，但仍只是低暴露合成诊断，不是自然语料、官方结果或已验证的 temporal holdout；`confirmatory_eligible=false`、`scientific_deployable=false`。

为检验上述合成增益能否迁移，`official_relation_transfer_v1` 已完成 60/60 个冻结单元（累计 81 次尝试，完成后生成/后处理缓存均为 60/60，离线重放 0 次外部调用）。它从未使用的 2026 Task A 官方训练文本中选取 30 个 gold-stratified 样本，按 `equivalent class` / `disjoint with` / 其他开放谓词各 10 个分层，包含 89 条端点可见和 101 条完整 non-self non-tax gold。`open_vocab` 与未校准 guarded-v2 的 exact NonTax P/R/F1 均为 0；后者把 non-self non-tax 误报从 424 降到 143（-66.3%），并把 official GS 从 0.1946 提高到 0.2593，但没有恢复任何 gold 关系。零调用 post-hoc 审计进一步发现：guarded-v2 的 225 个 raw 候选全部使用 gold 谓词集合之外的谓词，0 个 raw exact TP，82 个被拒候选中也没有 TP；因此主因是 predicate inventory / task definition 不对齐，而不是置信度或证据门误删。该审计看过 gold，不能用于在同一切片调参。该结果仍是 gold-stratified 迁移诊断，不是 unbiased、naturalistic、temporal、隐藏测试或部署证据。

后续独立检验 `official_predicate_routing_v1` 已完成（package ID `78d478f4f1d9d76ff979`）：27 个独立 Task A 样本按 `equivalent class` / `disjoint with` / `type` / domain 分配为 8/8/3/8，共有 64 条端点可见 target gold（69 条完整 target gold），与既有证据的样本、压缩 context 和 exact non-tax gold 重叠均为 0。54/54 个冻结单元在 57 次记录尝试内成功，完成后生成和后处理缓存均为 54/54，离线重放 0 次外部调用。与 relation-only v3 对照相比，routed-v6 将四分层 macro exact target F1 从 0 提高到 0.0526，但 pooled target P/R/F1 仅为 0.0143/0.0290/0.0191：140 个目标谓词预测中有 2 个 TP 和 138 个 FP。两个 TP 均来自 2 个独立 `type` 样本；`equivalent class`、`disjoint with` 和 `database_cross_reference` 仍为 0 TP。因正例样本数 2 低于预注册门槛 5，机制门失败，`scientific_deployable=false`、`confirmatory_eligible=false`；协议禁止同切片修补，并据此不启动 n≥100 扩样。

零调用错误审计进一步表明，routed-v6 的 140 个目标输出中有 120 个 `type`（85.7%）；138 个 official exact FP 中，121 个带显式证据、17 个带跨句证据，133 个预测的两个端点都逐字绑定到上下文。该结果定位的是“广泛输出文本可证据化关系但与稀疏 official gold 不对齐”，而不是单纯缺证据；由于未做独立语义裁决，off-gold 既不能自动视为语义错误，也不能视为正确关系，机制门保持关闭。

`experiments/results/competition_sim_metrics.json` 中的高分只用于 synthetic mock 方法自检，不能与官方数据或 pro-team 榜单结果比较。

## 如何运行

```powershell
$py = $env:MIMO_PYTHON
& $py D:\NLP\src\run_demo.py
& $py D:\NLP\src\run_competition_sim.py
& $py D:\NLP\src\run_official_2026_eval.py
& $py D:\NLP\src\run_official_train_baseline.py --task a --limit 40
```

### 接真实 LLM（本地环境变量）

旧 dots key 曾以明文保存在本地配置中，必须先在供应商控制台吊销并轮换。新 key 只放在当前 PowerShell 会话或私有凭据管理器中：

```powershell
$env:ONTOBRIDG_API_KEY = 'ak_你的新Key'
```

Base URL / 模型已写好：

- Base：`https://note3-prev-api.askdiandian.com/v1`
- Model：`dots3-note-prev`
- 文档：https://dots.ai/platform/docs（当前免费试用）

**之后两条命令：**

```powershell
# 冒烟
& D:\NLP\src\run_dots_smoke.ps1

# 官方 train 真实 LLM（默认 Task A × 20）
& D:\NLP\src\run_dots_baseline.ps1 -Limit 20
& D:\NLP\src\run_dots_baseline.ps1 -Task ab -Limit 50
```

换其它 OpenAI 兼容服务可设置 `ONTOBRIDG_BASE_URL` 和 `ONTOBRIDG_MODEL`。`src/dots_config.json` 是被 Git 忽略的本地兼容入口，不应提交真实 key。

## 阅读顺序

1. `competition/README.md` + `competition/OFFICIAL_2026.md` — 赛什么、指标、2026 结果  
2. `docs/DESIGN.md` — OntoBridg 架构  
3. `docs/INNOVATION.md` — 打哪里、凭什么赢  
4. `docs/STRATEGY.md` — 8 周到 2027 的路线  
5. `papers/paper_draft.md` — 论文定位与贡献

## 约定

- UMLS 等受限数据不进公开 git
- **对外 Graph Similarity** 只用 `src/ontobridg/eval/official_graph_similarity.py`（2026 exact_match）；内部 proxy SGS 禁止写入论文主表
- 官方 train 基线入口：`src/run_official_train_baseline.py`（offline 已跑；真实 LLM 需 API key）
- 任务编号：对外用 **2026 Flagship / Reuse / Taxonomy**；pipeline 子能力不再冒充竞赛 Task A/B/C
- mock / offline pattern 结果 ≠ 榜单分数
