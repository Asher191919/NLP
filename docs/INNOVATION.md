# OntoBridg Innovation Notes — Fail-Closed Open-Relation Learning

> Target: **LLMs4OL 2027**（复现/对齐 2026 Task A Flagship + Task B Reuse 指标后再参赛）
> Baselines: pro-team arXiv:2608.27101（2026）, Alexbek 2025 arXiv:2508.19428
> Version: v0.5 | 2026-09-26 | 真实实验以 `papers/results_skeleton.md` 为准

---

## 1. Research Gap and Current Finding

pro-team's strong 2026 baseline (Qwen2.5-14B + MiniLM RAG few-shot + **vocabulary-constrained filtering**) reported:

| Task | SGS | Term-Typing F1 | Taxonomy F1 | Non-tax relations |
|------|-----|----------------|-------------|-------------------|
| Task B | 0.8692 | 0.9200 | 0.8540 | **ZERO** |
| Task A | 0.7416 | — | — | **ZERO** |

**Research hypothesis:** a closed taxonomy vocabulary cannot emit associative relations.
Vocabulary-constrained decoding can only emit labels that exist in the seed type
inventory (`is-a`, type names). Associative relations — `part-of`, `causes`,
`located-in`, `treats`, `interacts-with`, `manufactured-by`, `has-feature` —
are **out of vocabulary**, so the extractor emits nothing.

This motivates an open relation channel, but it does not establish that the open
channel improves official scoring. In the leakage-controlled n = 30 prefix, the
current unconstrained open implementation lowers exact Graph Similarity by
0.2801 on Task A and 0.2927 on Task B because it over-generates unmatched
relations. The relation-density follow-up finds only one exact generated relation,
which the preregistered evidence-gap gate rejects. The supported contribution is
therefore a failure diagnosis and fail-closed protocol, not a winning claim.

| Task | Closed GS | Open GS | Open non-tax pred/gold |
|------|-----------|---------|----------------------------|
| Flagship A | 0.4404 | 0.1603 | 384 / 1 |
| Reuse B | 0.9375 | 0.6449 | 150 / 0 |

These are official 2026 **training-sample** measurements, not hidden-test or
leaderboard results. The sparse prefix is a strong over-generation diagnostic but
is not sufficient to reject the general open-vocabulary hypothesis.

---

## 1.1 指标政策（2026-09-20）

| 用途 | 指标 | 模块 | 对外可报告 |
|------|------|------|------------|
| 论文主表 / 竞赛对比 | Official Graph Similarity (exact_match) | `ontobridg.eval.official_graph_similarity` | **是** |
| 消融 / debug | internal proxy SGS | `semantic_graph.proxy_semantic_graph_similarity` | **否** |
| 官方 train 离线基线 | Typ F1 / Tax F1 / NonTax F1 / Official GS | `run_official_train_baseline.py` | 是（需注明 offline） |

官方 train 离线（无 gold 泄漏）样本结果见 `experiments/results/official_train_baseline_offline_*.json`：分数很低是预期的，用来替换「合成仿真高分」的误导性结论。

真实 LLM（本机当前 **无** `OPENAI_API_KEY`）：

```powershell
$env:OPENAI_API_KEY='sk-...'
$env:ONTOBRIDG_BASE_URL='https://api.openai.com/v1'
$env:ONTOBRIDG_MODEL='gpt-4o-mini'
& $env:MIMO_PYTHON D:\NLP\src\run_official_train_baseline.py --backend openai --task ab --limit 300
```

## 2. OntoBridg Contributions

### Contribution 1 — Open Relation Vocabulary Mining + LLM Generation + Symbolic Domain-Range Constraints

Module: `ontobridg/relations/open_vocab.py`

Instead of a closed label set we:

1. **Mine an open relation inventory** from corpus surface patterns
   (`X is used to treat Y`, `X causes Y`, `X is part of Y`, …).
2. **Invite LLM relation generation** over typed term pairs — the model may
   propose *any* relation string, not just seed-taxonomy labels.
3. **Apply symbolic domain–range constraints** mined from corpus + seed ontology
   priors (`treats: Drug→Disease`, `capital-of: City→Country`, …).
   - Hard mode: drop inconsistent triples.
   - Soft mode: penalize but allow high-score partially-compatible edges.
4. **Keep associative relations alive** under an open inventory; taxonomic
   labels (`is-a`, `type-of`) are routed to the taxonomy stage, not banned.

### Contribution 2 — Hybrid Neuro-Symbolic Repair

Module: `ontobridg/symbolic/repair.py`

| Operator | Purpose |
|----------|---------|
| Taxonomy transitivity reduction | Remove edges implied by `A⊑B⊑C` when redundant |
| Cycle breaking | Greedy drop of lowest-score edge on each `is-a` cycle |
| Type-consistency filtering | Term `T:Ty` may only `is-a` type `Ty` or an ancestor |
| Embedding soft-edge rescue | Near-threshold, type-compatible edges retained |
| Domain-range filter on non-tax | `treats(Drug,Disease)` etc.; novel labels get loose typed-endpoint gate |

The 2025 winner used cross-attention for taxonomy only; we also repair the
associative (non-taxonomic) layer.

### Contribution 3 — Contamination-Aware Evaluation Protocol

Module: `ontobridg/contamination/protocol.py`

WordNet / UMLS / GeoNames / schema.org are public artifacts and may have been
exposed through LLM pretraining. Reported F1 can therefore be confounded by
memorized factual associations. Our protocol:

1. **Closed-book factual probes** — distinguish official facts from matched
   non-gold controls; this measures probeability, not verified membership.
2. **Term holdout** — hide a fraction of terms; measure recovery on held-out terms.
3. **Paraphrase template suite** — re-surface gold facts in novel wording.
4. **Synthetic novel ontology stress test** — neologism terms + renamed relations.
   Pure-memorization baselines collapse; structure-aware systems should degrade
   more gracefully once patterns for novel labels exist.

We report **full / holdout / paraphrase / novel** side by side.

The first official-train pilot is complete on 15 Task A and 15 Task B samples.
Its 214 balanced closed-book items yield AUROC 0.9780 (sample-cluster 95% CI
[0.9484, 0.9982]), while probeability has weak, inconsistent associations with
per-sample Graph Similarity. A paired endpoint-alias stress lowers Task B GS by
0.1557/0.1917 for closed/open systems, but Task A moves in the opposite direction
(-0.0921/-0.1102 for original minus alias, with intervals spanning zero). This is
evidence of task-dependent surface sensitivity, not proof of training-set
membership; the sparse slice contains no usable non-tax contamination evidence.

The frozen `nonce_relation_holdout_v1` then supplies the missing relation-dense
diagnostic: all 60 units complete in 67 recorded attempts. On 120 balanced
non-tax positives, exact P/R/F1 rises from 0.2222/0.3167/0.2612 for raw
`open_vocab` to 0.8644/0.8500/0.8571 for uncalibrated
`open_vocab_guarded_v2`; both systems have 0/60 hard-negative false positives.
Guarded-v2 scores 1.0000 F1 on canonical language and 0.7119 on paraphrases,
so the result supports relation guards under controlled wording while exposing
a remaining surface-robustness gap. This is synthetic low-exposure evidence,
not an official, naturalistic, or verified temporal result.

### Contribution 4 — Relation-Channel Isolation and Fail-Closed Calibration

Module: `ontobridg/eval/semantic_graph.py`

Directly applying guarded generation to every channel damages typing/taxonomy.
The evaluated hybrid therefore preserves the legacy core graph and substitutes
only the guarded relation channel. Confidence calibration is deployable only when
candidate-level class support and at least five independent positive samples pass
the preregistered gate. Current support is four positives from three samples, so
the artifact correctly remains `deployable=false`.

A separate 30-sample dual-model weak-label experiment supplies 26 candidate
matches from 14 samples, but the model routes are same-origin and agreement is
low (macro Jaccard 0.3646). It is useful for exploratory engineering only: the
runner forces `deployable=false` and `confirmatory_eligible=false`, and its
in-sample calibration metrics are not evidence of held-out reliability.

A separate axiom-only v5 diagnostic isolates symmetric relations. On eight
2026 Task A calibration samples, a conservative singularizer plus bounded
reverse-edge expansion raises exact F1 from 0.1379 to 0.3117. However, the
preregistered disjoint eight-sample validation reverses that result: base F1 is
0.1667, while the rule adds eight false positives, recovers no true positive,
and lowers F1 to 0.1364. The rule is therefore rejected for the production
canonicalizer. Pooling both slices gives seven exact positives from six samples
and numerically passes grouped-CV support (Brier 0.1836; ECE 0.0830), but the
validation slice has then been reused for calibration. A subsequently frozen
24-sample endpoint-visible confirmatory slice produces 4 base TP from 4 samples.
Frozen calibration improves Brier from 0.6707 to 0.2109 but removes one TP and
leaves only 3 positive samples, below the preregistered minimum of 5. This valid
confirmatory negative result forces `scientific_deployable=false` and
`confirmatory_eligible=false`; no post-result retuning is allowed, and v5 is not
part of the v2 calibrator.

The locked error audit shows substantial predicate heterogeneity: base precision
is 0.1111 for `equivalent class` and 0.5000 for `disjoint with`. A 10,000-draw
sample-cluster bootstrap places base F1 at 0.1159 [0.0238, 0.2712] and calibrated
F1 at 0.0882 [0, 0.2222]. A planning-only estimate gives about 122 total samples
for pooled precision +/-0.10, but this cannot remove the endpoint-visible,
gold-stratified selection bias or authorize tuning. The frozen official semantic
secondary metric (threshold 0.90, pinned Nomic revisions) changes 0/24 samples;
semantic and exact relation-only Graph Similarity are identical for both systems.

---

## 3. Local Reproducible Results (synthetic multi-domain, mock LLM)

Command:

```powershell
& $env:MIMO_PYTHON D:\NLP\src\run_competition_sim.py
```

Artifacts: `D:\NLP\experiments\results\competition_sim_metrics.json`

### Task A / Task B averages across 4 datasets (biomedical, geo, product, multi)

| System | Task | TypF1 | TaxF1 | **NontaxF1** | SGS | Combo | NonTax pred/gold |
|--------|------|-------|-------|--------------|-----|-------|------------------|
| pro_team_closed_vocab | A | 0.9249 | 0.8709 | **0.0000** | 0.5290 | 0.5812 | **0 / 116** |
| ontobridg_open_vocab  | A | 0.9249 | 0.8868 | **0.7670** | 0.7700 | 0.8372 | 190 / 116 |
| pro_team_closed_vocab | B | 0.9312 | 0.8759 | **0.0000** | 0.5347 | 0.5855 | **0 / 116** |
| ontobridg_open_vocab  | B | 0.9312 | 0.8917 | **0.7789** | 0.7837 | 0.8464 | 184 / 116 |

**Delta (OntoBridg − pro_team):**

| Metric | Task A | Task B |
|--------|--------|--------|
| Combo | **+0.2560** | **+0.2609** |
| SGS | **+0.2410** | **+0.2490** |
| Nontax F1 | **+0.7670** | **+0.7789** |

Per-domain highlights (Task A):
- product: OntoBridg NontaxF1 **1.000** (20/20), Combo 0.956 — open vocab nearly solves associative product graph
- geo: NontaxF1 **0.818** (26/18), Combo 0.867
- biomedical: NontaxF1 **0.447** (65/20) — over-generation + loose `BiologicalEntity` ranges; precision work remains
- pro_team: **0 non-tax recovered on every domain** (KEY GAP reproduced)

### Contamination protocol (multi-domain; knowledge probe = high)

| System | Full combo / nontaxF1 | Paraphrase combo / nontaxF1 | Novel combo / nontaxF1 |
|--------|----------------------|----------------------------|------------------------|
| ontobridg | 0.841 / **0.803** | 0.809 / **0.699** | 0.101 / 0.000 |
| pro_team  | 0.580 / **0.000** | 0.585 / **0.000** | 0.093 / 0.000 |

Interpretation:
- OntoBridg keeps non-taxonomic signal under paraphrase (0.803→0.699); pro_team stays at 0.
- Novel ontology stress collapses both — mock backend has no prior on invented
  relation labels; next ablation is pattern-only open mining on novel surfaces.
- Holdout-with-full-corpus is a weak blindness test for the mock (lexicon still
  holds all terms); novel + paraphrase arms are the sharp probes.
- Note: these are **synthetic mock** numbers for method validation, not official
  leaderboard claims. Official 2026 reference numbers are preserved in the JSON
  under `meta.reference_2026_pro_team`.

---

## 4. Baselines Compared in Synthetic Simulation

| System | Mode | Non-tax recovery |
|--------|------|------------------|
| `pro_team_closed_vocab` | RAG + vocabulary-constrained (is-a only) | **0** |
| `ontobridg_open_vocab` | Open relation mining + mock generation + domain-range + symbolic repair | **> 0** (F1≈0.77, synthetic only) |

---

## 5. Pipeline Sketch

```
Corpus ──► Term discovery (LLM/RAG + lexicon filter)
        ──► Term typing (RAG few-shot + type inventory)
        ──► Taxonomy candidates (LLM + type hierarchy priors)
        ──► OPEN relation mining (surface patterns + LLM typed-pair generation)
        ──► Symbolic repair (cycles, transitivity, type-consistency, domain-range)
        ──► Ontology assembly
        ──► Contamination-aware eval (full / holdout / paraphrase / novel)
```

Closed-vocab systems stop at taxonomy. We continue through the open relation
channel and then *clean* the graph.

---

## 6. How to Reproduce

```powershell
& $env:MIMO_PYTHON D:\NLP\src\run_competition_sim.py
# fallback
py -3 D:\NLP\src\run_competition_sim.py
python D:\NLP\src\run_competition_sim.py
```

Outputs:
- `D:\NLP\experiments\results\competition_sim_metrics.json`
- Sample CSVs: `D:\NLP\data\samples\` (+ per-domain subfolders)
- Public LLMs4OL baseline clone: `D:\NLP\data\LLMs4OL-baseline\`
  (TaskA/B/C: Geonames, UMLS, WN18RR)
- Additional local dumps if present: `D:\NLP\data\LLMs4OL-Challenge\`,
  `D:\NLP\data\alexbek\`, `D:\NLP\data\OntoLearner\`

---

## 7. Next Steps Toward Confirmatory Evidence

1. Keep the completed dual-model labels explicitly exploratory; they cannot be
   reported as human annotation or used to open the confirmatory gate.
2. Obtain relation-level human validation (independent annotation or a documented
   audit with corrected/missing relations) before reconsidering deployment.
3. Refit locally from the existing generation cache with zero external calls, and
   launch confirmatory API evaluation only if every preregistered gate then
   passes; otherwise report the abstention result without relaxing thresholds.
4. Retain the completed v5 confirmatory failure without threshold or rule
   retuning; report probability calibration and exact decision quality separately.
5. Defer the estimated n≈122 expansion until its gold-stratified scope and cost
   are justified; if run, use it only to narrow estimation uncertainty.
6. Retain the completed pinned semantic analysis as secondary evidence; it
   changes no v5 confirmatory sample and cannot override exact-match decisions.
7. Retain the completed factual-probe and endpoint-alias pilot as diagnostic
   evidence only; next use genuinely temporal or synthetic holdouts rather than
   relabeling probe-score tertiles as verified clean/polluted data.
8. Retain the completed `nonce_relation_holdout_v1` as synthetic diagnostic
   evidence: 60/60 locked units, guarded-v2 exact NonTax F1 0.8571 versus
   0.2612 for raw open vocab, and 0/60 hard-negative false positives for both.
   Do not fit a calibrator, retune the frozen protocol, or treat the result as
   naturalistic/temporal confirmation; deployment and confirmatory gates remain
   closed.
9. The follow-up `official_relation_transfer_v1` is complete: 60/60 units in 81
   recorded attempts, followed by a 60/60 zero-call cache replay. On 30 unused
   Task A training texts and 101 non-self non-tax gold relations, raw open and
   guarded-v2 both obtain exact NonTax F1 0. Guarded-v2 reduces predictions from
   424 to 143 and raises official exact GS from 0.1946 to 0.2593, but recovers no
   gold relation. Treat this as negative gold-stratified transfer evidence, not
   an unbiased, hidden-test, naturalistic, temporal, or deployment result.
10. Its zero-call post-hoc error audit localizes the failure upstream of the
    guard: all 225 raw guarded candidates use predicates outside the exact gold
    inventory, with zero raw or rejected exact TP. Raw open likewise emits no
    gold predicate, and 53/424 outputs are typing/taxonomy aliases misserialized
    as non-tax. This supports a predicate-family/task-definition mismatch
    diagnosis, but gold inspection prohibits tuning on the same slice; an axiom
    routing follow-up must use a newly frozen independent holdout.
11. The independent follow-up `official_predicate_routing_v1` is frozen but has
    not been executed (package `78d478f4f1d9d76ff979`). It contains 27 Task A
    samples allocated 8/8/3/8 to `equivalent class`, `disjoint with`, `type`,
    and `database_cross_reference` strata, with 64 endpoint-visible target edges
    and 69 target edges in total. Sample indices, compact contexts, and exact
    non-tax gold have zero overlap with prior evidence. The locked comparison is
    `open_vocab_guarded_v3_relations` versus
    `open_vocab_guarded_v6_predicate_routed`; only prompt identity differs. Its
    54 units had zero generation or postprocess cache hits at freeze time and
    therefore imply 54 external calls. The primary endpoint is four-stratum
    macro exact target F1, and the mechanism gate requires both a strict macro-F1
    improvement and exact true positives in at least five independent samples.
    Only four of the eight routed predicates have independent empirical support:
    independent `type` capacity is limited to three samples, and
    the other prompt routes (`has role`, `same as`, `is_conjugate_acid_of`, and
    `is_conjugate_base_of`) cannot be evaluated by this packet. Same-slice repair
    is prohibited, and execution requires separate authorization.

---

## 8. One-Line Evidence Statement

> OntoBridg shows that unconstrained open-relation generation over-produces under
> exact scoring, while relation-only symbolic guards recover relations on a
> frozen controlled nonce diagnostic but recover none on the completed
> gold-stratified official-training transfer slice. Its primary contribution is
> a bounded positive synthetic result, reproducible negative official-train
> results, and a fail-closed protocol for deciding when open-relation predictions
> are safe to deploy.

## Related Paths

- Design: `D:\NLP\docs\DESIGN.md`
- Metrics JSON: `D:\NLP\experiments\results\competition_sim_metrics.json`
- Package: `D:\NLP\src\ontobridg\`
- Competition notes: `D:\NLP\competition\`
