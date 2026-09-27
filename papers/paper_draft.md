# OntoBridg: A Fail-Closed Study of Open-Relation Ontology Learning

> **Status**: Paper skeleton v0.6 | 2026-09-27 | Project root `D:\NLP` | leakage-controlled official-train, guarded density, calibration, contamination, nonce-holdout, and official-text transfer results are complete (see `results_skeleton.md`)
> **Direction**: LLMs4OL (ontology learning / ontology extension), **not** OAEI alignment
> **Notes**: Chinese comments mark drafting decisions; English is the intended paper language.

---

## Title

**OntoBridg: A Fail-Closed Study of Open-Relation Ontology Learning**

*Alternative short title for workshop tracks*: OntoBridg: Hybrid Neuro-Symbolic OL under Contamination-Aware Evaluation

*Keywords*: ontology learning; large language models; neuro-symbolic; LLMs4OL; training data contamination; open relation extraction; taxonomy induction

---

## Abstract (~200 words)

Large language models (LLMs) have become the default approach to ontology learning from text, yet two structural problems remain under-examined. First, competition-style leaderboards for LLMs4OL use public knowledge resources, creating plausible pretraining-exposure confounds. Second, strong systems often fail to extract **any** non-taxonomic relations when the inventory is closed or under-specified. We present **OntoBridg**, a hybrid neuro-symbolic pipeline coupling open relation generation with relation-only evidence, confidence, predicate, self-loop, and quantity constraints. A leakage-controlled official-train prefix exposes severe open-relation over-generation. On balanced Task A/B density strata, preserving legacy typing/taxonomy while guarding only relations raises exact Graph Similarity by 0.0315/0.0409, although exact NonTax F1 remains 0/0. A separately frozen 30-sample nonce holdout supplies 120 synthetic non-tax relations and 60 negated controls. Uncalibrated guarded-v2 raises exact NonTax F1 from 0.2612 to 0.8571 with zero hard-negative false positives, but drops from 1.0000 on canonical wording to 0.7119 on paraphrases. On a frozen 30-sample, gold-stratified official-training transfer slice, both systems recover 0/101 non-self non-tax gold edges; guarded-v2 reduces false positives from 424 to 143 and raises exact Graph Similarity from 0.1946 to 0.2593. A post-hoc audit finds that all 225 raw guarded candidates use predicates outside the exact gold inventory, so the guard did not hide a true positive. A contamination pilot finds high factual probeability (AUROC 0.9780) but opposing alias effects across tasks. Thus the controlled synthetic gain does not establish official or naturalistic relation recovery. No result is a hidden-test or leaderboard claim, and all deployment gates remain closed.

---

## 1 Introduction

### 1.1 Motivation

Ontology learning from text—the automatic construction or extension of class hierarchies, type assignments, and semantic relations from unstructured or semi-structured corpora—remains a bottleneck for knowledge-centric applications. The LLMs4OL shared task series formalized this setting into three operational problems: **Term Typing** (assign each term to one or more ontology types), **Taxonomy Discovery** (induce *is-a* edges among types or terms), and **Non-taxonomic Relation Extraction** (recover other semantic relations such as part-of, located-in, or used-for). Datasets span general lexical resources (WordNet), geographic knowledge (GeoNames), biomedical ontologies (UMLS modules), and web schema (schema.org).

LLM-based systems have flooded this evaluation. Zero-shot prompting of instruction-tuned models already produces non-trivial Term Typing F1, and the 2025 competition winner (Alexbek, arXiv:2508.19428) demonstrated that retrieval-augmented generation, embedding ensembles, and a cross-attention taxonomy module can push graph-level metrics further. The 2026 pro-team system (arXiv:2608.27101) adopted a Flagship configuration on Task A plus ontology-extension reuse on Task B, scoring with Semantic Graph Similarity combined with Term-Typing F1 and Taxonomy F1, using Qwen2.5-14B with RAG—yet **failed to extract any non-taxonomic relations**. That single failure mode is informative: relation extraction under open or poorly specified inventories is not solved by larger RAG stacks alone.

### 1.2 The Contamination Problem

A second, orthogonal problem complicates the scientific reading of these results. WordNet glosses, GeoNames place types, UMLS concept hierarchies, and schema.org property definitions are public and heavily cited, so exposure through large-scale pretraining is plausible but model-specific and normally unverifiable. If a model has memorized *hypernym*(“dog”, “mammal”), micro-F1 can reward retrieval of a stored association rather than induction from the supplied text. Prior work on OAEI-style ontology *matching* has documented that pure LLM matchers often underperform hybrid symbolic systems (LogMap, AML, BERTMap) on clean alignment tracks; ontology *learning* inherits the same exposure risk because the gold graph is itself a public knowledge artifact.

We treat possible training-data exposure as a **first-class evaluation risk**, not a footnote. Closed-book factual probes, surface-form stress tests, and genuinely temporal or synthetic holdouts should accompany LLMs4OL claims, with explicit separation between probeability and verified membership.

### 1.3 System and Contributions

This paper documents **OntoBridg**, a hybrid neuro-symbolic ontology learning system developed at `D:\NLP`. The pipeline couples LLM generation with RAG few-shot conditioning, multi-objective symbolic scoring (see `methodology.md`), hard graph constraints (acyclicity, domain/range typing, hierarchy-depth regularization), and an open relation vocabulary procedure that discovers relation labels from definitions when no closed property inventory is available.

**Contributions:**

1. **Leakage-controlled failure characterization.** A blind official-training evaluation shows that the current unconstrained open-relation path sharply lowers exact Graph Similarity through unmatched relation over-generation; the result is reported without hidden-test or leaderboard claims.

2. **Relation-channel-isolated neuro-symbolic constraints.** Evidence, confidence, canonicalization, self-loop, predicate, and quantity rules are applied only to non-taxonomic relations while preserving the legacy typing/taxonomy graph. This isolates graph-precision gains from core-task regressions.

3. **Fail-closed calibration protocol.** Calibration and confirmatory execution are blocked unless candidate-level class support and independent positive-sample support meet preregistered minima. The current four positives from three samples deliberately produce `deployable=false`.

4. **Contamination-aware evaluation with bounded transfer claims.** A frozen closed-book factual probe and endpoint-alias stress test separate factual probeability from task performance. A preregistered nonce holdout tests 120 newly generated relation facts and 60 negated controls without exposing gold to the model; a disjoint official-training transfer slice then shows that its positive controlled result does not recover any of 101 non-self official gold relations. Neither result is promoted to naturalistic, temporal, unseen-predicate, or deployment evidence.

*（写作备注：Introduction 目标 ~1.5–2 页；贡献条对应 methodology 与 experiments 的可检验 claim。）*

---

## 2 Related Work

### 2.1 LLMs4OL: Task Family and History

The LLMs4OL shared task originates in the formulation by Elizarova et al. (arXiv:2307.16648), which defines Term Typing, Taxonomy Discovery, and Non-taxonomic Relation Extraction over WordNet, GeoNames, UMLS, and schema.org. The 2024 overview (arXiv:2409.10146) consolidates baselines and metric conventions. Subsequent editions refined Task A (Flagship-style term typing) and Task B (ontology extension / reuse), and shifted composite scoring toward **Semantic Graph Similarity (SGS)** together with Term-Typing F1 and Taxonomy F1.

Key systems to position against:

- **Zero-shot / instruction LLM baselines** from the original LLMs4OL paper—prompt-only GPT/instruction models; adequate for typing, weak on graph consistency.
- **Fine-tuned encoder models** (Flan-T5, BERT-style) supervised on train splits—strong on typing when train/test distributions align; fragile on novel types.
- **Alexbek 2025 (arXiv:2508.19428)**—competition winner; RAG + embedding ensemble for typing, cross-attention module for taxonomy edges.
- **pro-team 2026 (arXiv:2608.27101)**—Qwen2.5-14B RAG; Task A Flagship + Task B Ontology Extension Reuse; SGS + F1 metrics; **zero non-taxonomic relations extracted**.

### 2.2 OAEI Contrast: When Hybrid Symbolic Still Wins

Outside LLMs4OL, the Ontology Alignment Evaluation Initiative (OAEI) provides a useful control condition. On clean, well-benchmarked alignment tracks, hybrid symbolic matchers (LogMap, AgreementMakerLight, BERTMap and descendants) remain competitive or superior to pure LLM matchers. LLM-only systems are often *worse* than classical string/structure matchers when gold alignments are not trivially present in pretraining data. OntoBridg inherits this lesson: LLM generation is a **candidate generator**, not a trusted oracle; symbolic constraints repair and veto.

*（备注：OAEI 不作为本文主实验赛道；仅作 Related Work 对照，支撑 H1「symbolic constraints 必要」。）*

### 2.3 Neuro-Symbolic Ontology Learning

Classical ontology learning combined linguistic patterns, clustering, and taxonomic heuristics (e.g., Hearst-style patterns, formal concept analysis). Neuro-symbolic variants replace pattern extractors with neural encoders or LLMs while retaining logical post-constraints: type consistency, acyclicity, cardinality, and domain/range checks. OntoBridg sits in this lineage: LLM proposes; multi-objective symbolic scorer ranks; hard constraints prune; optional OWL/RDF export enables external reasoner checks.

### 2.4 Training-Set Contamination in LLM Evaluation

Contamination has been studied for QA, code, and reasoning benchmarks through membership inference, paraphrase degradation, and canary-style probes. For ontology matching/learning, the **exposure risk is systematic** because gold ontologies are themselves public knowledge artifacts, although membership in any particular model's training set is usually unverifiable. We extend this risk analysis to LLMs4OL metrics (F1, official Graph Similarity) and operationalize a bounded-claim protocol (Section 5).

---

## 3 Method: OntoBridg

### 3.1 Architecture Overview

OntoBridg is a staged pipeline over a typed intermediate representation `OntologyDataset` (terms, types, taxonomy edges, relations, domain/range maps)—implemented in `src/ontobridg/`.

```
Raw Text / Gold Terms + TBox
        │
        ▼
 Stage 1  Term Extraction          (LLM candidates + NP filter)            → serves 2026 Task A
 Stage 2  Term Typing              (RAG + LLM + type-lattice constraint)  → serves 2026 Task A/B
 Stage 3  Taxonomy Discovery       (candidate gen + multi-obj score + acyclicity) → 2026 Task A/B/C
 Stage 4  Non-Tax Relation Extract (open vocab + domain/range filter)     → 2026 Task A/B non-tax triples
 Stage 5  Ontology Assembly        (merge → primitive-ontology-triples)
 Stage 6  Contamination-Aware Eval (official GraphSim + F1 + clean splits)
```

### 3.2 Multi-Objective Symbolic Scoring

Candidates from Stages 3–4 are scored by a weighted multi-objective function combining LLM log-probability/confidence, embedding similarity, definition overlap, and structural priors. Hard constraints are applied after soft ranking. Full formulas, open-relation procedure, symbolic repair rules, and contamination stress-test definitions are specified in **`methodology.md`**.

Sketch: for a taxonomy candidate edge \(e = (c \rightarrow p)\),

\[
S(e) = \alpha\, s_{\mathrm{LLM}}(e) + \beta\, s_{\mathrm{emb}}(e) + \gamma\, s_{\mathrm{def}}(e) + \delta\, s_{\mathrm{struct}}(e)
\]

subject to acyclicity and depth regularizers; rejected edges are pruned by Union-Find/DFS repair.

### 3.2 Open Relation Vocabulary (non-tax triples in 2026 Task A/B)

**Competition tasks (external numbering — use these in experiments/tables):**

- **Task A — Flagship (End-to-End Ontology Learning).** Given raw domain text `context`, construct primitive ontology triples `(subject, predicate, object)` covering terms, types, `instance-of`, `is-a`, and non-taxonomic relations.
- **Task B — Reuse (Ontology Extension).** Given `context`, an initial ontology \(O_{\mathrm{init}}\), and optional candidate `terms`/`types`, predict extension triples \(T_{\mathrm{ext}}\) (not regenerate \(O_{\mathrm{init}}\)).
- **Task C — Taxonomy (Cross-domain).** Train on source-domain ontologies; induce `is-a` edges for held-out domains (materials science, biology/life sciences, scholarly knowledge, ecology/environment).

**Pipeline sub-capabilities (internal method stages — do not relabel as competition Task A/B/C):**

- Typing sub-module: assign each term a conceptual type (used inside Flagship/Reuse).
- Taxonomy sub-module: induce `is-a` edges (used in Flagship/Reuse and competition Task C).
- Non-taxonomic relation sub-module: open-vocabulary predicates (Flagship/Reuse graph completeness; addresses pro-team zero-relation failure).

**Metrics (external reportable):**

- Term-Typing F1; Taxonomy F1.
- **Official Graph Similarity (exact):** \((EdgeF1 + NeighborhoodSim + TaxonomySim)/3\) via `ontobridg.eval.official_graph_similarity`, aligned with challenge `metrics/graph_similarity.py`.
- Composite reference style (pro-team 2026): \((GraphSim + TypingF1 + TaxonomyF1)/3\).
- Internal proxy SGS is **not** official and must not appear in the main results table.

Rather than classifying term pairs into a closed property list, Stage 4: (i) mines predicate phrases from definitions and contexts; (ii) requests confidence and either one explicit evidence span or two nearby spans for a cross-sentence relation; (iii) canonicalizes entity surfaces and direction-aware predicate aliases; (iv) rejects unsupported, low-confidence, generic, and self-loop relations and caps accepted non-tax triples per sample; and (v) calibrates confidence on a disjoint development split only when that split has adequate positive and negative support. Degenerate calibration artifacts fail closed. Both evaluated hybrids preserve the legacy typing/taxonomy output and substitute only the guarded relation channel. This is the primary technical contrast to pro-team 2026’s empty relation output.

### 3.4 Contamination Integration

Stage 6 supplements official metrics with frozen closed-book factual probes and surface-form stress tests (Section 5). Probe-derived strata are descriptive and are not labeled as verified clean/polluted data; probe outputs cannot tune generation, calibration, or deployment.

---

## 4 Experiments / Results

> **Result tables**: measured numbers are maintained in [`results_skeleton.md`](results_skeleton.md) (verified against `experiments/results/*.json`). Section 4.6 reports the primary blind n = 30 experiment; Section 4.7 reports the density-stratified guarded follow-up; Section 4.8 preserves the legacy n = 10 oracle-inventory pilot for provenance.  
> **Metric policy**: only the **official LLMs4OL 2026 Graph Similarity** (exact) and micro-F1 are reported; the **internal proxy SGS is not reported**. Primary results must use `llms4ol2026_blind_v1`, where Task A receives context only and Task B receives only official task inputs. The **2026 competition is closed** — results target the **2027 competition + paper**.

### 4.1 Datasets

| Dataset | Domains | Tasks | Contamination prior |
|---------|---------|-------|---------------------|
| WordNet | lexical / commonsense | A, B, C | High (glosses & hypernyms widely web-copied) |
| GeoNames | geographic | A, B, C | Medium–High |
| UMLS (NCI / MEDCIN / SNOMEDCT subsets) | biomedical | A, B, C | Medium (public subsets; license-sensitive) |
| schema.org | web schema | A, B, C | Medium–High (property docs public) |

Exact train/dev/test splits follow the official LLMs4OL release for the target year. *（UMLS 许可：未确认 license 前不入库、不公开衍生数据。）*

### 4.2 Metrics

- **Official**: Term-Typing micro F1; Taxonomy F1; Non-taxonomic Relation F1 (micro; empty-empty not scored as a hit).
- **Official Graph Similarity** (exact): \((EdgeF1 + NeighborhoodSim + TaxonomySim)/3\) via `ontobridg.eval.official_graph_similarity`, aligned with the challenge `metrics/graph_similarity.py`.
- **Composite (summary only)**: \((TypingF1 + TaxonomyF1 + GraphSim)/3\) in the pro-team reporting style; not an official leaderboard score.
- **Not reported**: internal proxy SGS (non-official); it must not appear in main results tables.
- **Diagnostics**: precision/recall decomposition, constraint violation counts before/after repair.
- **Exposure-risk diagnostics**: closed-book factual discrimination, descriptive probe-score strata, and paired endpoint-alias stress; these do not create verified clean/polluted labels.

### 4.3 Baselines

| Baseline | Source | Role |
|----------|--------|------|
| Zero-shot LLM (prompt-only) | LLMs4OL 2023 arXiv:2307.16648 | Lower bound / “LLM alone” |
| Flan-T5 / encoder fine-tune | Standard supervised OL | Closed-vocab strong typing baseline |
| Alexbek 2025 | arXiv:2508.19428 | RAG + emb ensemble + cross-attn taxonomy |
| pro-team 2026 | arXiv:2608.27101 | Qwen2.5-14B RAG, SGS+F1, Task A Flagship + B reuse |
| OntoBridg (ours) | this paper | Hybrid + open relations + contamination protocol |

Implementation notes: reproduce baselines under identical backbone LLM where possible (e.g., Qwen2.5-14B for RAG-style systems) to isolate method vs. model scale. Record model id, date, temperature, and prompt templates for reproducibility.

### 4.4 Ablations

1. **Constraints off** (no acyclicity / domain/range / depth repair).
2. **RAG off** (zero-shot vs. RAG few-shot).
3. **Closed vs. open relation vocabulary** on Task C.
4. **Single-objective vs. multi-objective score** (\(\alpha,\beta,\gamma,\delta\) variants).
5. **Original vs. endpoint-alias reporting** (does surface substitution change system behavior?).
6. Optional: hard vs. soft constraint scheduling under high factual probeability.

### 4.5 Research Questions

- **RQ1**: Do symbolic constraints improve Taxonomy F1 and official Graph Similarity?
- **RQ2**: Does open relation vocabulary recover non-taxonomic relations where closed-inventory RAG systems fail?
- **RQ3**: Are official facts closed-book probeable, and is probeability associated with task performance?
- **RQ4**: How do edge-wise F1 and official Graph Similarity change under endpoint aliasing?

### 4.6 Leakage-Controlled Official-Train Results (n = 30)

We evaluate `dots3-note-prev` under `llms4ol2026_blind_v1` on the first 30 official training samples for each task. Task A prompts contain context only. Task B prompts contain context, the official initial ontology, and official terms/types. Gold triples never enter prompts and are used locally for scoring. All 120 task-system-sample units completed; these are train-prefix measurements, not hidden-test or leaderboard results.

| Task | System | n | Typing F1 | Taxonomy F1 | NonTax F1 | **Official GS** | Composite | non-tax pred/gold |
|------|--------|---|-----------|-------------|-----------|-----------------|-----------|-------------------|
| A Flagship | `closed_vocab` | 30 | **0.1400** | **0.4903** | 0.0000 | **0.4404** | **0.3569** | 0 / 1 |
| A Flagship | `open_vocab` | 30 | 0.1356 | 0.3516 | 0.0000 | 0.1603 | 0.2159 | 384 / 1 |
| B Reuse | `closed_vocab` | 30 | 0.8824 | **0.9474** | 0.0000 | **0.9375** | **0.9224** | 0 / 0 |
| B Reuse | `open_vocab` | 30 | **1.0000** | 0.8966 | 0.0000 | 0.6449 | 0.8471 | 150 / 0 |

**Reading.** The open path does not improve the primary blind result. Relative to closed, its official GS changes by −0.2801 on Task A and −0.2927 on Task B. Task B typing improves by +0.1176, but taxonomy falls by 0.0508 and relation over-generation dominates graph-level scoring. The non-taxonomic gold support is too small to test the intended recall advantage: one edge in Task A and zero in Task B. The valid conclusion is therefore that the current open-relation prompt is insufficiently calibrated under exact scoring, not that open-vocabulary extraction is intrinsically ineffective. This sparsity motivates the density-stratified guarded follow-up in Section 4.7.

### 4.7 Non-Tax-Density-Stratified Guarded Evaluation

The prefix experiment is unsuitable for RQ2 because it contains almost no non-tax gold. We therefore select five official-training samples per task from each of three strata defined by non-self non-tax gold count: low (1), medium (2–4), and high (at least 5). Gold is used only for local sample selection and scoring; it is never serialized into a model prompt. The resulting slices contain 63 Task A non-tax edges and 53 Task B non-tax edges, of which 52 Task B edges are non-self. Guarded-v1 requires confidence and one verbatim span containing both endpoints. Guarded-v2 adds predicate/entity canonicalization and two-span cross-sentence evidence, with prespecified confidence ≥ 0.95 and evidence gap ≤ 600 normalized characters for implicit candidates. The legacy/v1 experiment completed 60 API units; v2 added 30/30 successful calls. Both hybrids are derived locally.

| Task | System | n | Typing F1 | Taxonomy F1 | NonTax F1 | **Official GS** | Composite | non-tax pred/gold |
|------|--------|---|-----------|-------------|-----------|-----------------|-----------|-------------------|
| A Flagship | `open_vocab` | 15 | 0.3077 | 0.3475 | 0.0000 | 0.1825 | 0.2793 | 250 / 63 |
| A Flagship | `open_vocab_guarded` | 15 | 0.0310 | 0.2405 | 0.0000 | 0.1512 | 0.1409 | 34 / 63 |
| A Flagship | `open_vocab_guarded_hybrid` | 15 | **0.3077** | **0.3475** | 0.0000 | **0.2140** | **0.2897** | 34 / 63 |
| A Flagship | `open_vocab_guarded_v2` | 15 | **0.3396** | **0.3771** | 0.0000 | 0.1974 | **0.3047** | 54 / 63 |
| A Flagship | `open_vocab_guarded_v2_hybrid` | 15 | 0.3077 | 0.3475 | 0.0000 | **0.2152** | 0.2901 | 54 / 63 |
| B Reuse | `open_vocab` | 15 | 0.6000 | 0.7500 | **0.0202** | 0.5598 | 0.6366 | 46 / 53 |
| B Reuse | `open_vocab_guarded` | 15 | 0.1961 | 0.6000 | 0.0000 | 0.4677 | 0.4213 | 4 / 53 |
| B Reuse | `open_vocab_guarded_hybrid` | 15 | **0.6000** | **0.7500** | 0.0000 | **0.6007** | **0.6502** | 4 / 53 |
| B Reuse | `open_vocab_guarded_v2` | 15 | 0.1923 | 0.6861 | 0.0000 | 0.4991 | 0.4592 | 19 / 53 |
| B Reuse | `open_vocab_guarded_v2_hybrid` | 15 | 0.6000 | 0.7500 | 0.0000 | 0.5766 | 0.6422 | 19 / 53 |

**Reading.** Guarding only the relation channel is the stable comparison. Relative to legacy open, the v1 hybrid removes 216 Task A and 42 Task B non-tax predictions (86.4% and 91.3%) while improving official GS by 0.0315 and 0.0409. The gains occur in all six task-density cells, although the Task A high-density gain is only 0.0006. Direct v1 generation sharply reduces core-task F1; v2 improves the direct Task A core scores but remains weaker on Task B, so the hybrid keeps typing/taxonomy fixed across relation variants.

The v1 relation result remains negative. Task A has no exact non-tax true positive. Task B legacy open has one (`infructescence axis --part_of--> infructescence`), while v1-hybrid has none because its single-span prompt generates no candidate for that edge. V2 corrects the generation failure: among 48 two-span candidates it generates the exact edge at confidence 0.95, but the evidence spans are 1,604 normalized characters apart and the prespecified 600-character gate rejects it. Consequently, the primary v2-hybrid still has zero exact TP. V2-hybrid improves official GS over legacy by 0.0327 on Task A and 0.0168 on Task B, but its Task B GS remains below the more conservative v1-hybrid by 0.0241.

The original within-evaluation calibration split is degenerate (0/34 positives) and is not used for deployment. We instead collect 39 independent relation-enriched calibration samples (27 Task A, 12 Task B), disjoint from the blind prefix and density evaluation. Gold is used locally for selection and candidate labels only. Of 177 raw candidates, 144 pass evidence/confidence gates before quantity truncation and define the calibration population; 4 exact positives come from 3 independent samples. Beta-isotonic calibration reduces Brier score from 0.8429 to 0.0270 and ECE from 0.9024 to 0.0113 on this population. However, the preregistered minimum is five independent positive samples, so the artifact is `deployable=false` with status `insufficient_independent_positive_samples`. A calibration-only predicate set `{part_of, develops_from, regulates}` combined with the best threshold retains 1 TP and 1 FP (recall 0.25), which is still too weak for deployment.

As an explicitly exploratory alternative, two same-origin model routes annotate the same frozen 30-text packet and a third route adjudicates it. They produce 114 and 105 relations. Sample-exact agreement is 0.2000, macro/micro relation-set Jaccard is 0.3646/0.3436, and positive agreement is 0.5114. The adjudicated artifact contains 169 weak relations. A zero-call cache replay finds 26 weak-label matches across 14 samples among 200 deployment-eligible candidates, numerically exceeding the five-sample support minimum. Resubstitution changes Brier/ECE from 0.7596/0.8005 to 0.1131/0.0037. Deterministic five-fold cross-validation grouped by sample scores all 200 eligible candidates once with zero sample overlap and yields OOF Brier/ECE 0.1134/0.0045 after calibration. A 10,000-replicate sample-cluster bootstrap gives calibrated Brier 0.1134 [0.0681, 0.1781] and ECE 0.0045 [0.0013, 0.0867]; the corresponding raw intervals are 0.7596 [0.6747, 0.8268] and 0.8005 [0.7062, 0.8693]. These intervals condition on the frozen OOF predictions and weak labels. Every fold collapses the 0.85--1.00 input scores to a single probability block (0.124--0.140), showing stable base-rate shrinkage rather than useful confidence ranking. Moreover, the official data contain only one non-tax edge in these 30 samples and it overlaps none of the weak labels. We therefore report this only as *exploratory dual-LLM weak-label calibration*; the runner enforces `deployable=false` and `confirmatory_eligible=false`.

![Weak-label relation calibration reliability](figures/weaklabel_relation_calibration_reliability.svg)

**Figure X | Weak-label OOF calibration and uncertainty.** **a,** Reliability bins for raw and sample-grouped out-of-fold calibrated confidence; vertical intervals use 10,000 sample-cluster bootstrap replicates. **b,** Brier and expected calibration error (ECE) with 95% sample-cluster percentile intervals. Labels are dual-LLM weak labels rather than human or official gold. The calibrated points reflect base-rate shrinkage and do not establish relation correctness or ranking resolution.

A relation-only v3 pilot on the remaining three supported Task A failures completes 3/3 calls and emits 11 accepted candidates but recovers no exact relation. A subsequent zero-call structural audit projects generated taxonomy edges into benchmark relations. The lexical rule `(whole) is-a * substance` -> `* substance derives from (whole)` is exact on two samples and remains 2/2 under leave-one-sample-out selection. In contrast, a matched measurement-to-size rule is 3/4 pooled and 1/2 out of fold; unrestricted taxonomy-to-`part_of` and inverse-`has part` rules are 3/219 and 1/219. A `nested position` rule recovers one `has part` edge but has no cross-sample support. Although the replicated lexical rule raises the diagnostic union from three to five positive samples, the projection channel itself has only two positive samples, below its separate five-sample requirement. We therefore do not freeze or run the confirmatory configuration.

We additionally test a cross-year source intended to increase relation-positive support without reusing the 2026 evaluation slices. Ten public 2025 Text2Onto documents are linked locally by endpoint co-occurrence to 11 unique Task D axioms (8 `disjoint with`, 3 `equivalent class`); documents containing any 2026 Task A/B train-gold triple are excluded in full, and gold is never serialized into the prompt. A broad v4 axiom-aware prompt produces 44 raw and 34 accepted candidates but zero exact TP because it emits no target axiom predicates. An axiom-only v5 prompt produces 10 raw and 8 accepted candidates, with 1 TP, 7 FP, and 10 FN (precision 0.1250, recall 0.0909, F1 0.1053). The positive support comes from one sample, all eligible confidences equal 0.95, and the calibrator remains `degenerate_too_few_positives`. The error audit also shows that an enumerative passage yields six plausible pairwise disjoint predictions while the linked ontology gold contains only two pairs. Endpoint co-occurrence therefore does not provide complete document-level relation labels. We retain these two passes as negative mechanism diagnostics, do not relax the ordinary-sibling prohibition, do not pool them with the v2 calibrator, and do not launch confirmatory evaluation.

A further axiom-only v5 extension uses eight disjoint 2026 Task A training samples with visible lexical evidence, excluding the blind prefix, density split, 39-unit v2 calibration set, and weak-label packet. The first pass exhausts a 1,024-token completion cap on all eight units (16 low-level failed attempts); the 16,384-token retry completes 8/8, for 16 unit attempts and 24 low-level attempts in the retained ledger. The system emits 23 accepted candidates against 35 non-tax gold edges and obtains 4 exact TP from 3 samples (precision 0.1739, recall 0.1143, F1 0.1379). Calibration resubstitution changes Brier/ECE from 0.7452/0.7765 to 0.1403/0.0528, but the exact five-positive-sample gate remains closed. Because `equivalent class` and `disjoint with` are semantically symmetric, we separately report a non-official endpoint-order-invariant diagnostic: it counts 12 TP across all 8 samples (precision 0.5217, recall 0.3429, F1 0.4138). A versioned conservative final-token singularizer raises the non-official order-and-surface-invariant result to 13 TP and F1 0.4483. For an official-exact serialization ablation, we preserve every original prediction and use only the remaining max-eight capacity for reverse symmetric edges. Combined with singularization, this produces 42 predictions, 12 TP, 30 FP, and F1 0.3117, recovering eight exact edges and losing none. On the cached 2025 v5 pressure check, the bounded rule reaches F1 0.1739 under incomplete endpoint-linked supervision.

We then execute the preregistered, disjoint eight-sample v5 validation set. All 8 calls succeed without retry. Base v5 produces 8 predictions against 28 gold edges, with 3 TP from 3 samples (precision 0.3750, recall 0.1071, F1 0.1667). Surface canonicalization recovers no exact edge; adding max-eight-aware reverse edges increases output to 16 predictions while retaining 3 TP, adds 8 FP, and reduces F1 to 0.1364. The earlier surface/direction gain therefore does not replicate, and the rule is rejected for the production canonicalizer. A subsequent pooled diagnostic over both v5 slices contains 31 eligible candidates, 7 exact positives from 6 samples, and passes the numeric five-sample gate. Five-fold grouped OOF calibration scores every candidate once with zero sample overlap and improves Brier/ECE from 0.6987/0.7245 to 0.1836/0.0830. Because the inspected validation slice is reused in this pool, that artifact explicitly forces `scientific_deployable=false` and `confirmatory_eligible=false`. Neither v5 slice is pooled with the v2 calibrator.

Finally, we freeze a new 24-sample Task A confirmatory slice before generation. It is balanced between endpoint-visible `equivalent class` and `disjoint with` gold and has zero overlap with all earlier experimental ledgers. No unused sample remains under the earlier strict same-segment lexical-explicit criterion, so this slice is reported as gold-stratified and endpoint-visible, not distribution-matched to the first 16 samples or unbiased. Generation succeeds for 24/24 units in 24 low-level attempts. Base v5 produces 15 accepted predictions with 4 TP from 4 samples (precision 0.2667, recall 0.0741, F1 0.1159). A zero-call replay of the frozen pooled calibrator improves candidate Brier/ECE from 0.6707/0.6840 to 0.2109/0.1238, yet its frozen threshold removes one exact TP and reduces task F1 to 0.0882 (3 TP in 14 predictions from 3 samples). It therefore fails the preregistered minimum of five exact-positive samples. We retain the valid confirmatory negative result, prohibit post-result retuning, and keep `scientific_deployable=false` and `confirmatory_eligible=false`.

Post-confirmatory analysis is descriptive and does not reopen model selection. Base `disjoint with` precision/F1 is 0.5000/0.1579 (3 TP in 6 predictions), compared with 0.1111/0.0714 for `equivalent class` (1 TP in 9 predictions). Nine false positives have no exact endpoint/predicate match, and equivalent-class errors commonly map a named class to a description rather than an equivalent label. In a 10,000-replicate sample-cluster bootstrap, base F1 is 0.1159 [0.0238, 0.2712], calibrated F1 is 0.0882 [0, 0.2222], and their difference is -0.0277 [-0.1064, 0]. The Brier difference is -0.4598 [-0.7517, 0.2739]. A planning-only approximation gives n≈122 total samples for a pooled precision half-width near 0.10, but the base `disjoint with` stratum alone would require about 194 samples at the observed prediction rate, exceeding the available 89 endpoint-visible samples before exclusions. Immediate expansion is therefore deferred: it would not justify retuning or remove the slice's gold-stratified selection bias.

As a frozen secondary analysis, we apply the official 2026 embedding `semantic_match` at threshold 0.90 using pinned Nomic model and remote-code revisions. On relation-only triples averaged per sample, semantic matching changes no result for either system: base exact and semantic Graph Similarity are both 0.0935, while calibrated exact and semantic Graph Similarity are both 0.0657. The exact-match failure is therefore not explained by surface similarity recoverable at this threshold. Exact remains the primary metric and the failed deployment gate is unchanged.

For provenance, an earlier A12+B12 artifact was replayed once on the density evaluation before the stricter sample-support gate was added. Predicate-only filtering reduced non-tax predictions from 54 to 28 on Task A and from 19 to 8 on Task B, while the calibrated setting abstained on every relation. Exact NonTax F1 remained 0 throughout. We therefore treat those numbers only as retrospective abstention diagnostics, not deployable validation. A separate post-hoc gap sensitivity at 1,604 characters accepts the Task B TP and yields NonTax F1 0.0274 (1/20 predictions versus legacy 0.0202 at 1/46), but also admits two Task A false positives; it remains a mechanism diagnostic rather than tuned evidence. All post-generation replays use separate cache signatures and zero additional external calls.

### 4.8 Legacy Oracle-Inventory Pilot (official train, n = 10)

Legacy component comparison on the official LLMs4OL 2026 Flagship (Task A) **train** prefix. Sources: `official_train_baseline_openai_a_n10.json` (closed) and `official_train_baseline_openai_a_n10_open_merged.json` (open, micro-merged from offset chunks). Backbone: `dots3-note-prev`, temperature 0; scoring uses the official Graph Similarity (exact) only. The legacy runner constructed term/type candidate inventories from gold triples before prompting; consequently, this table is an oracle-assisted diagnostic and must not be presented as end-to-end Flagship performance.

| System | n | Typing F1 | Taxonomy F1 | NonTax F1 | **Official GS** | Composite | non-tax pred/gold |
|--------|---|-----------|-------------|-----------|-----------------|-----------|-------------------|
| `closed_vocab` | 10 | 1.0000 | 0.0247 | 0.0000 | **0.1825** | 0.4024 | 0 / 1 |
| `open_vocab` | 10 | 1.0000 | **0.1176** | 0.0000 | **0.2251** | **0.4476** | 0 / 1 |
| Δ (open − closed) | | 0.0000 | **+0.0930** | 0.0000 | **+0.0426** | **+0.0452** | — |

**Reading.** Under this oracle-inventory condition, the open-vocabulary path improves taxonomy structure and official Graph Similarity relative to the closed-vocabulary path, while non-taxonomic F1 remains 0 for both systems because only one gold non-tax edge appears in the first ten samples. The saturated typing score is expected under the leaked candidate inventory and is not evidence of end-to-end term discovery. Offline pattern baselines (no LLM) collapse to Official GS 0.0000–0.0373 (Task A) and 0.0000–0.0343 (Task B) at n = 40.

**Scope and caveats.**

1. **Official Graph Similarity only**; the internal proxy SGS is not reported anywhere in the main tables.
2. **n = 10 is an oracle-inventory component pilot**, not an end-to-end validation or leaderboard claim.
3. **Task A/B blind n = 30** is complete under `llms4ol2026_blind_v1`; all 120 units succeeded after checkpointed retries.
4. **The density-stratified follow-up** completed 60/60 legacy/v1 API units, 30/30 new v2 calls, and local hybrid derivations; its gold-based selection is local and never exposed to the model.
5. **Independent calibration** contains 39/39 successful unique v2 units after six recorded connection failures (45 unit-level attempts; 47 low-level attempts including two internal retries); only three samples supply exact generated-relation positives, so the sample-support gate blocks deployment. The separate v3 pilot adds 3/3 successful calls and 0 exact TP. Cross-year v4/v5 diagnostics add 10/10 successful calls each but supply at most one exact positive sample under endpoint-linked, incomplete document supervision. The disjoint 2026 v5 extension adds 8/8 successful units after 8 token-cap failures and also supplies exact positives from only three samples. None of these separate channels is pooled into the v2 calibrator.
6. **Structural projection** is exploratory and remains fail-closed: its only replicated rule has two positive samples, below the separate five-sample support requirement.
7. Synthetic multi-domain checks in `competition_sim_metrics.json` are **NON-official** mock-backend method validation (closed NonTax F1 0.0000 vs. open 0.7685 macro-average) and are labeled as such in `results_skeleton.md` §4.5.
8. The **2026 competition is closed**; this paper targets the **2027 competition cycle and archival publication**.

Further tables (offline n = 40, contamination protocol drops, reproduction commands) are maintained in **`papers/results_skeleton.md`**.

---

## 5 Contamination Protocol

*（本节为论文独立贡献章节；详细定义见 methodology.md §Contamination Stress Tests。）*

We propose a four-part protocol for LLMs4OL-style benchmarks:

1. **Closed-book factual probes.** Query the evaluated LLM on official facts and matched same-predicate object-swap controls without source context. High discrimination establishes factual probeability, not training-set membership; controls may include unlabeled true facts, and self-reported confidence is not a likelihood.

2. **Surface-form stress tests.** Paraphrase, synonym-substitute, or alias gold terms/types/relations and re-evaluate. A change measures surface dependence but does not by itself distinguish memorization from ordinary lexical difficulty.

3. **Clean vs. polluted split reporting.** Partition the test set by probe score; report F1 and official Graph Similarity separately. Always publish both columns.

4. **Synthetic / temporal holdouts.** Where possible, construct novel term–type–relation triples not present in common web snapshots, or use post-training-cutoff resources, as a lower-contamination test bed.

In the frozen pilot, 30/30 closed-book units succeed and cover 214 balanced items. AUROC is 0.9780 (10,000-replicate sample-cluster 95% CI [0.9484, 0.9982]), but probeability has weak and inconsistent association with per-sample Graph Similarity. The paired endpoint-alias arm completes 60/60 units. Original-minus-alias GS is -0.0921 [-0.2577, 0.0798] and -0.1102 [-0.2507, 0.0206] for Task A closed/open, versus +0.1557 [0.0362, 0.3022] and +0.1917 [0.0609, 0.3291] for Task B. Restricting to fully endpoint-visible samples preserves the opposing Task A/B pattern. NonTax F1 remains zero throughout. These results support task-dependent surface sensitivity, not verified contamination or general metric inflation; no probe output is used for tuning or deployment.

Because the official pilot contains no recoverable non-tax support, we additionally freeze and execute a controlled low-exposure holdout. Thirty synthetic context-only samples contain 120 balanced non-tax relations and 60 negated co-occurrence controls over 300 newly generated opaque entities, with zero exact entity-string or gold-triple overlap against official 2026 Task A/B train. Canonical/paraphrased and explicit/cross-sentence strata are balanced. All 60 preregistered units complete in 67 recorded attempts. Raw `open_vocab` obtains exact NonTax P/R/F1 0.2222/0.3167/0.2612 and official exact GS 0.1805; uncalibrated `open_vocab_guarded_v2` obtains 0.8644/0.8500/0.8571 and GS 0.5690. Both systems produce zero false positives on all 60 hard-negative controls. Guarded-v2 exact F1 is 1.0000 on canonical samples and 0.7119 on paraphrased samples, with explicit/cross-sentence recall 0.7500/0.9500. Because the packet has no taxonomy gold, official GS is capped at 2/3 and exact NonTax micro-F1 remains primary. The result supports guarded extraction and canonicalization of new synthetic facts but not unseen-predicate discovery; controlled language can overestimate naturalistic performance. It is not an official or verified temporal benchmark, and it remains ineligible for tuning or deployment (`confirmatory_eligible=false`, `scientific_deployable=false`).

We next executed a frozen official-text transfer diagnostic. `official_relation_transfer_v1` contains 30 unused Task A training samples assigned 10/10/10 to equivalent-class, disjoint-with, and heterogeneous open-predicate strata. It supplies 89 endpoint-visible and 101 total non-self non-tax gold relations over eight predicates, with 2–8 visible relations per sample. The exclusion ledger covers executed experiments and earlier audit-only selections; sample indices, compact contexts, and exact non-tax gold have zero overlap with all prior project evidence. The locked 60-unit comparison reuses the raw-open and uncalibrated guarded-v2 identities without prompt, threshold, calibrator, or postprocessing changes. All units completed in 81 recorded attempts and replay from 60/60 generation and postprocess cache hits. Raw open and guarded-v2 both obtain exact NonTax P/R/F1 of 0/0/0. Guarded-v2 reduces non-self predictions from 424 to 143 and raises official exact Graph Similarity from 0.1946 to 0.2593, but exact recall remains zero for every prespecified stratum, gold predicate, density group, and endpoint-visibility group. A deterministic post-hoc audit finds 225 raw guarded candidates, of which 82 are rejected; none of the 225 is an exact TP and all use predicates outside the exact gold inventory. Among accepted predictions, `part_of` and `has part` contribute 89/143. The evidence/confidence guard therefore did not remove a latent correct relation; predicate-family/task-definition mismatch dominates. Because this diagnosis inspects protected gold, it cannot tune the same slice, and any axiom-routing follow-up requires an independently frozen holdout. The nonce gain therefore does not transfer to this slice. Because selection is gold-stratified on official training data and the source does not establish naturalistic provenance, this result cannot be described as unbiased, hidden-test, naturalistic, temporal, or deployment evidence.

We have frozen, but not executed, the independent predicate-routing follow-up `official_predicate_routing_v1` (package `78d478f4f1d9d76ff979`). Its 27 Task A samples are allocated 8/8/3/8 to `equivalent class`, `disjoint with`, `type`, and a domain-predicate stratum represented by `database_cross_reference`, providing 64 endpoint-visible target edges and 69 target edges in total. The packet has zero overlap with prior evidence in sample index, compact context, and exact non-tax gold. The 54-unit design compares `open_vocab_guarded_v3_relations` with `open_vocab_guarded_v6_predicate_routed` under a shared postprocessor, with prompt identity as the only changed system component. At freeze time it has zero generation-cache and zero postprocess-cache hits, so execution is estimated to require 54 external calls. The primary endpoint is four-stratum macro exact target F1. The preregistered mechanism gate requires a strict macro-F1 improvement for routed-v6 and exact true positives in at least five independent samples. No same-slice prompt repair or retuning is permitted after inspection. Execution requires separate authorization; consequently, this paragraph reports the frozen design and contains no evaluation result.

---

## 6 Limitations

- **Model and data access.** API-based LLMs drift; UMLS redistribution is license-constrained. We mitigate with frozen local checkpoints where feasible and public-dataset primary experiments.
- **Contamination measurement is imperfect.** Closed-book discrimination can reflect public factual knowledge without revealing where it was learned, constructed controls can be unlabeled truths, and endpoint aliasing changes lexical difficulty as well as possible memorization cues. The opposing Task A/B effects prevent a global inflation claim, and the sparse sample contains no non-tax evidence.
- **Open relation vocabulary evaluation.** The completed official-training transfer slice recovers no exact non-tax edge under either raw-open or guarded-v2 generation despite dense, endpoint-visible gold selection. Its post-hoc audit finds no exact gold predicate among 225 raw guarded candidates, but that diagnosis is gold-informed and cannot validate a revised prompt on the same samples. Projection or predicate-family routing could change results in either direction and must be frozen on independent data and reported separately.
- **Predicate-routing coverage.** The frozen independent routing packet supports only four predicates: `equivalent class`, `disjoint with`, `type`, and `database_cross_reference`. Independent capacity for `type` is only three samples, and the other four prompt routes (`has role`, `same as`, `is_conjugate_acid_of`, and `is_conjugate_base_of`) have no empirical support in this packet. Any eventual result therefore cannot establish performance for all eight routed predicates.
- **Sparse and reused calibration support.** The same-system independent v2 calibrator has only four positive candidates from three samples. A separate v5 pool reaches seven positives from six samples and improves grouped-OOF calibration metrics only after the inspected validation slice is reused for fitting. The subsequent untouched, endpoint-visible confirmatory slice validates probability shrinkage but not the decision rule: frozen calibration reduces Brier/ECE while losing one exact TP and retaining only three positive samples. The bounded surface/direction rule is also rejected. Neither v2 nor v5 opens the deployment gate.
- **Weak-label validation.** Dual-model adjudication supplies more text-grounded positives but has low inter-model agreement, same-origin annotators, no exact overlap with the single official non-tax edge in its packet, and no held-out human audit. Sample-grouped OOF evaluation confirms only base-rate shrinkage: all five calibrators have one probability block and no ranking resolution. These weak-label diagnostics cannot open the confirmatory gate.
- **Exact vs. semantic metric.** Exact Graph Similarity remains primary. The pinned official semantic secondary analysis at threshold 0.90 changes none of the 24 v5 confirmatory samples. This relation-only, per-sample analysis is not interchangeable with leaderboard-wide SGS variants.
- **Competition year dependency.** If LLMs4OL 2027 changes tasks, some Task A Flagship / Task B extension assumptions may need revision.
- **Scope.** This paper targets ontology *learning/extension*, not OAEI *alignment*; symbolic repair reuse for OAEI is future work.

---

## 7 Conclusion

OntoBridg frames ontology learning for LLMs as a **candidate-generation + symbolic-repair + fail-closed evaluation** problem rather than a pure prompting problem. Unconstrained open relations over-generate on official training slices, while relation-only guards recover high exact recall on the frozen controlled nonce holdout. That gain is bounded: familiar predicates and templated language test new-fact extraction and canonicalization, not unseen-predicate discovery, and paraphrasing reduces F1. More decisively, the frozen official-training transfer slice yields zero exact non-tax true positives for both raw-open and guarded-v2 generation; guarding reduces false positives but does not recover official relations. The protected-gold audit localizes this failure to predicate-family mismatch rather than guard rejection, but cannot authorize same-slice repair. The system therefore abstains when independent naturalistic calibration support is insufficient. The contamination pilot likewise fails closed at the claim level: official facts are highly probeable, but opposing endpoint-alias effects do not verify pretraining membership. The empirical contribution is consequently a controlled synthetic positive result, reproducible official-train negative results, and an auditable deployment gate; stable naturalistic open-vocabulary gains remain a confirmatory question.

---

## References

> Format: informal working bib. Convert to venue style (CEUR / JWS / KBS) at submission.

1. Elizarova, T., et al. **LLMs4OL: Large Language Models for Ontology Learning.** arXiv:2307.16648, 2023.
2. Elizarova, T., et al. **LLMs4OL 2024 Overview.** arXiv:2409.10146, 2024.
3. Alexbek et al. **[2025 LLMs4OL system paper: RAG + embedding ensemble + cross-attention taxonomy].** arXiv:2508.19428, 2025.
4. pro-team. **[2026 LLMs4OL system paper: Task A Flagship + Task B Ontology Extension Reuse; SGS + F1; Qwen2.5-14B RAG].** arXiv:2608.27101, 2026.
5. Article / preprint on **training set contamination as systematic evaluation flaw for LLM ontology matching and learning.** *（投稿前补全作者/题名/年份；DESIGN.md 中“文章 insight”需映射到正式文献。）*
6. Jiménez-Ruiz, E., et al. **LogMap / OAEI hybrid alignment systems.** (LogMap, AML, BERTMap references as applicable.)
7. Hearst, M. **Automatic acquisition of hyponyms from large text corpora.** COLING, 1992. *(classical OL patterns)*
8. Ganguly, D., et al. / relevant **benchmark contamination** literature (paraphrase degradation, membership inference). *（补齐具体条目。）*
9. schema.org / WordNet / GeoNames / UMLS documentation citations per LLMs4OL data cards.

---

## Draft Word-Count & Next Actions

| Section | Target words | Status |
|---------|--------------|--------|
| Abstract | 180–220 | skeleton written |
| 1 Introduction | 600–800 | skeleton written (~700) |
| 2 Related Work | 400–600 | skeleton written |
| 3 Method | 300–400 in draft + expand from methodology.md | pointer written |
| 4 Experiments / Results | 400–500 + tables | blind n = 30 (§4.6), stratified guard (§4.7), oracle n = 10 (§4.8) |
| 5 Contamination Protocol | 250–350 | skeleton written |
| 6 Limitations | 200–250 | written |
| 7 Conclusion | 120–180 | written |

**Next actions (执行顺序建议):**
1. 补全 References 5–8 的正式文献信息（含 contamination article 的准确出处）。
2. 从 arXiv:2608.27101 对照 pro-team 报告口径（其 SGS）与本节官方 Graph Similarity 的差异说明，写入 methodology + experiments。
3. 实现 Stage 2–6 最小可跑路径（WordNet + schema.org），产出 `eval_report.md`。
4. 当前 v2 gold-based calibrator 的 144 个 deployment-eligible candidates 中只有 4 个正例、来自 3 个样本。v5 合并 16 个样本后达到 7 个正例、6 个正例样本，但表面/反向补边规则在独立 validation 上只增加 8 个 FP。新冻结的 24 样本 endpoint-visible confirmatory slice 已完成：冻结 calibrator 虽改善 Brier/ECE，却把 exact TP 从 4 降至 3，正例样本数低于预注册门槛 5。因此 v5 保持 `scientific_deployable=false`、`confirmatory_eligible=false`，不得事后调阈值。30 样本双模型弱标注同样不得用于打开 gate。
5. 按 `target_venues.md` 选择首个投稿点（优先 LLMs4OL workshop / ISWC）。
