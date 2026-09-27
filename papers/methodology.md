# OntoBridg Methodology (Expandable)

> **Status**: Method text v0.1 | 2026-09-20 | Companion to `paper_draft.md`
> **Purpose**: Ready-to-expand formal method description for Section 3 of the paper.
> **Language**: Academic English for body; Chinese notes for implementation choices.

---

## 0. Notation and Data Structures

Let a dataset be an ontology learning instance

\[
\mathcal{D} = (T, C, H_{\mathrm{gold}}, R_{\mathrm{gold}}, \mathcal{C}tx)
\]

- \(T\): set of terms (strings)
- \(C\): type / class inventory (closed inventory for Task A typing)
- \(H_{\mathrm{gold}} \subseteq C \times C\) or \(T \times T\): gold *is-a* edges (taxonomy)
- \(R_{\mathrm{gold}} \subseteq \mathcal{T} \times \mathcal{P} \times \mathcal{T}\): gold non-taxonomic triples (head, predicate, tail)
- \(\mathcal{C}tx: T \rightarrow 2^{U}\): optional natural-language contexts or definitions

OntoBridg internal IR (`OntologyDataset` in code):

| Field | Meaning |
|-------|---------|
| `terms[]` | `{id, surface, context?}` |
| `types[]` | type inventory \(C\) |
| `taxonomy[]` | predicted edges `{child, parent, score}` |
| `relations[]` | predicted triples `{head, relation, tail, score}` |
| `domain_ranges` | partial map \(p \mapsto (\mathrm{dom}, \mathrm{ran})\) |

*（实现对应：`src/ontobridg/data/schema.py` 的 TermSample / TaxonomySample / RelationSample / OntologyDataset / OntologyKnowledge。）*

---

## 1. Pipeline Stages

### Stage 0 — Data & Ontology Prep

1. Load official LLMs4OL splits (train/dev/test) for WordNet, GeoNames, UMLS subsets, schema.org.
2. Parse gold TBox when provided: classes, `is-a`, object properties, domain/range.
3. **Leakage firewall**: gold *test* items never enter the RAG index; few-shot retrieval only over train (and optionally dev).
4. Build `OntologyKnowledge` priors only from *train* + external non-eval resources if licensed.

*输出：`data/processed/*.jsonl` + manifest (model ids, dates, hashes)。*

### Stage 1 — Term Extraction (Internal)

Not an official LLMs4OL task; needed when raw text is the input.

- LLM candidate generation: extract noun phrases that denote domain concepts.
- Normalization: Unicode NFC, lowercase for matching keys, keep surface form for display.
- Filters: stopword/noun-phrase heuristics; optional embedding deduplication (cosine threshold \(\tau_{\mathrm{dedup}}\)).

Output: \(\hat{T} \subseteq T_{\mathrm{gold}}\) (evaluation usually provides gold terms; Stage 1 then acts as pass-through).

### Stage 2 — Term Typing (serves LLMs4OL 2026 Task A Flagship / Task B Reuse)

**Input:** term \(t \in T\), optional context \(\mathcal{C}tx(t)\), type inventory \(C\).

**RAG conditioning.** Retrieve \(k\) nearest train pairs \((t_i, c_i)\) by embedding similarity \(\mathrm{sim}(\mathbf{e}(t), \mathbf{e}(t_i))\). Prompt includes retrieved exemplars + type glosses (if available) + the query term.

**LLM scoring.** For each \(c \in C\), obtain a relevance score. Two modes:

- *Generative*: model outputs ranked type names; parse to \(C\); invalid labels dropped.
- *Verbalized scoring*: for each \(c\), prompt “Is \(t\) a \(c\)?” and map Yes/No + logprob to \(s_{\mathrm{LLM}}(t,c) \in [0,1]\).

**Symbolic typing constraints.**

- **Inventory membership**: predicted types must lie in \(C\).
- **Lattice consistency** (soft): if \(c\) is predicted and \((c, p) \in \hat{H}\) with high support, optionally encourage \(p\) (ancestor closure) depending on metric definition (multi-label vs single-label official Task A).
- **Incompatibility** (optional hard): disjointness pairs from gold TBox, if available, cannot co-occur.

Output: \(\hat{Y}(t) \subseteq C\) (or ranked list).

### Stage 3 — Taxonomy Discovery (serves 2026 Task A/B and competition Task C Taxonomy)

**Input:** typed terms/types; optional embeddings; optional definitions.

**Candidate generation** \(\mathcal{E}_0\):

1. LLM pairwise or listwise proposals of \((c \rightarrow p)\).
2. Embedding k-NN: for each \(c\), nearest neighbors in \(C\) as candidate parents.
3. Pattern seeds (optional): Hearst-like patterns from definitions (“X is a kind of Y”).

**Multi-objective scoring.** For each candidate edge \(e = (c \rightarrow p)\), see Section 2.

**Graph assembly + repair.** Rank by \(S(e)\); greedily accept subject to constraints (Section 3); optional transitive reduction to remove redundant direct-parent edges; depth control to avoid “all-to-root” or “all-to-leaf” degeneracy.

Output: \(\hat{H} = \{(c,p) : e \in \text{accepted}\}\).

### Stage 4 — Non-Taxonomic Relation Extraction (2026 Task A/B non-tax triples; not a standalone 2026 task id)

**Open relation vocabulary procedure** (Section 4) + typed pair filtering + LLM labeling + domain/range veto.

Output: \(\hat{R} = \{(h, p, t)\}\).

### Stage 5 — Ontology Assembly

- Union typed nodes + \(\hat{H}\) + \(\hat{R}\) into one graph \(G_{\hat{\mathcal{O}}}\).
- Optional export OWL/RDF (`rdflib`); external reasoner consistency check if available.
- Emit `ontology_final.json` / `.owl`.

### Stage 6 — Contamination-Aware Evaluation

Official metrics on full test + clean/polluted splits + stress sets (Section 5, 6).

---

## 2. Multi-Objective Score (Formulas)

### 2.1 Taxonomy edge score

For candidate edge \(e = (c \rightarrow p)\):

\[
S_{\mathrm{tax}}(e)
=
\alpha\, s_{\mathrm{LLM}}(e)
+ \beta\, s_{\mathrm{emb}}(e)
+ \gamma\, s_{\mathrm{def}}(e)
+ \delta\, s_{\mathrm{struct}}(e)
\]

\[
\alpha + \beta + \gamma + \delta = 1, \quad \alpha,\beta,\gamma,\delta \ge 0
\]

**Components:**

1. **LLM confidence**  
   \[
   s_{\mathrm{LLM}}(e) = \sigma\!\big(f_{\mathrm{LLM}}(c, p; \mathrm{prompt}, \mathrm{RAG})\big) \in [0,1]
   \]
   where \(f_{\mathrm{LLM}}\) is verbalized probability or normalized log-odds that \(c\) is-a \(p\), and \(\sigma\) is a calibration map (identity or temperature-scaled).

2. **Embedding affinity**  
   \[
   s_{\mathrm{emb}}(e) = \cos\big(\mathbf{e}(c), \mathbf{e}(p)\big) \in [-1,1]
   \xrightarrow{\ \text{rescale}\ } [0,1]
   \]
   Optionally definition embeddings \(\mathbf{e}(g(c)), \mathbf{e}(g(p))\) when glosses \(g\) exist.

3. **Definition / lexical overlap**  
   \[
   s_{\mathrm{def}}(e)
   =
   \frac{| \mathrm{tok}(c) \cap \mathrm{tok}(g(p)) |}{|\mathrm{tok}(c)|}
   \]
   or ROUGE-style unigram recall of the child surface in the parent gloss. Encourages linguistically plausible hypernymy, not just vector proximity.

4. **Structural prior**  
   \[
   s_{\mathrm{struct}}(e)
   =
   \lambda_d\, \phi_{\mathrm{depth}}(e)
   + \lambda_b\, \phi_{\mathrm{branch}}(e)
   + \lambda_p\, \phi_{\mathrm{type}}(e)
   \]
   - \(\phi_{\mathrm{depth}}\): penalty for attaching everything to root or leaf (e.g., prefer mid-depth parents).
   - \(\phi_{\mathrm{branch}}\): discourage nodes with unbounded out-degree.
   - \(\phi_{\mathrm{type}}\): consistency between Stage-2 types of \(c\) and domain of parent \(p\) (type-conditional taxonomy when applicable).

**Acceptance rule (soft):** accept \(e\) if \(S_{\mathrm{tax}}(e) \ge \tau_{\mathrm{tax}}\) and constraints (Section 3) hold.

**Relation edge score** (Task C), for triple \(r = (h, p_{\mathrm{rel}}, t)\):

\[
S_{\mathrm{rel}}(r)
=
\alpha_r\, s_{\mathrm{LLM}}(r)
+ \beta_r\, s_{\mathrm{emb}}(r)
+ \gamma_r\, s_{\mathrm{def}}(r)
+ \delta_r\, s_{\mathrm{typed}}(r)
\]

where \(s_{\mathrm{typed}}(r) \in \{0,1\}\) (or soft) measures domain/range compatibility:

\[
s_{\mathrm{typed}}(r)
=
\mathbb{1}\big[
\hat{\mathrm{type}}(h) \subseteq \mathrm{dom}(p_{\mathrm{rel}})
\ \wedge\
\hat{\mathrm{type}}(t) \subseteq \mathrm{ran}(p_{\mathrm{rel}})
\big]
\]

If \(\mathrm{dom}/\mathrm{ran}\) unknown (open vocab), estimate from mined usage or set \(s_{\mathrm{typed}}\) via soft type compatibility from embeddings of type labels.

### 2.2 Composite task score (internal selection)

When tuning \(\alpha,\ldots\) on dev, optimize official composite if available:

\[
S_{\mathrm{comp}}
=
w_{\mathrm{SGS}}\cdot \mathrm{SGS}
+ w_{\mathrm{A}}\cdot F1_{\mathrm{typing}}
+ w_{\mathrm{B}}\cdot F1_{\mathrm{tax}}
+ w_{\mathrm{C}}\cdot F1_{\mathrm{rel}}
\]

*（2026 pro-team / 官方主指标：Official Graph Similarity + Term-Typing F1 + Taxonomy F1。Graph Similarity 必须用 `ontobridg.eval.official_graph_similarity`；非分类关系在 Flagship/Reuse 图内单独报告 F1。）*

### 2.3 Semantic Graph Similarity (implementation sketch)

Until the official formula is locked, implement a conservative graph similarity and label it *SGS-proxy*:

\[
\mathrm{SGS}(G_{\hat{\mathcal{O}}}, G_{\mathrm{gold}})
=
\eta_1\, \mathrm{sim}_{\mathrm{nodes}}
+ \eta_2\, \mathrm{sim}_{\mathrm{tax-edges}}
+ \eta_3\, \mathrm{sim}_{\mathrm{rel-edges}}
\]

with each \(\mathrm{sim}\) as edge/node F1 or embedding-smoothed set similarity. **Replace with official definition when available** and note any delta in the paper.

---

## 3. Symbolic Repair Rules

Applied after soft ranking; operate on \(\hat{H}\) and \(\hat{R}\).

| ID | Rule | Operation |
|----|------|-----------|
| R1 | **Inventory** | Drop nodes/types not in \(C \cup \hat{T}\) (or remap via string normalization). |
| R2 | **No self-loop** | Remove \((c,c)\) from \(\hat{H}\) and \((h,h)\) non-tax triples that are identity. |
| R3 | **Acyclicity** | DFS/Union-Find detect cycles in \(\hat{H}\); remove lowest-\(S_{\mathrm{tax}}\) edge on each cycle. |
| R4 | **Domain/range veto** | Drop \(r=(h,p,t)\) if typed heads/tails violate known \(\mathrm{dom}(p),\mathrm{ran}(p)\) hard constraints. |
| R5 | **Tax vs non-tax conflict** | If \((h,t) \in \hat{H}\) and a non-tax \(p\) is flagged as taxonomic alias (e.g., `subClassOf`, `is_a`), drop the redundant non-tax triple. |
| R6 | **Depth regularization** | If max depth \(> D_{\max}\) or too many nodes share the root, re-rank structurally and re-assemble. |
| R7 | **Transitive reduction** | Remove \((c,g)\) if \((c,p)\) and \((p,g)\) exist and metric/protocol prefers direct parents only. |
| R8 | **Orphan soft-repair** | For high-type-confidence orphans, attach to nearest scored parent above \(\tau_{\mathrm{orphan}}\) or leave disconnected with logged reason. |
| R9 | **Relation inventory hygiene** | Cluster synonym predicates (`part_of`, `partOf`, `hasPart`); canonicalize labels before scoring. |
| R10 | **Cardinality (optional)** | Cap children per parent or relations per head if degenerate hubs appear. |

*（代码位置规划：`src/ontobridg/constraints/`；networkx + 自研 DFS/UF。R3 与 R7 是 taxonomy 质量的关键。）*

**Repair is reported:** paper tables include “violations before/after repair” and delta-F1 from repair alone (supports H1).

---

## 4. Open Relation Vocabulary Procedure

Motivation: pro-team 2026 (arXiv:2608.27101) **failed to extract any non-taxonomic relations**, likely due to closed or ill-fitting inventories + aggressive typing filters + RAG conditioned only on taxonomy-like exemplars. OntoBridg treats relation labels as **open**.

### 4.1 Procedure (five steps)

**Step OR1 — Context harvest.** For candidate term pairs \((h,t)\) (co-occurrence in same definition/sentence, or typed-pair candidates from Stage 2), collect supporting spans.

**Step OR2 — Predicate mining.** Prompt the LLM to extract *phrasal predicates* linking \(h\) to \(t\):

```
Extract the relation phrase that connects HEAD to TAIL in the text.
Return: predicate_phrase | direction | confidence | evidence_span
```

Allow free-text phrases; do **not** constrain to a gold property list at mining time.

In the guarded implementation, each candidate is emitted as `subject`, `predicate`, `object`, `confidence`, and `evidence`. A non-tax candidate survives only if its confidence is at least 0.85, its evidence is a verbatim context span containing both endpoints, it is not a self-loop, and its predicate is not a generic narrative relation. Candidates are ranked by confidence and capped at eight per sample. Typing and taxonomy are taken from the legacy open path, while only the guarded non-tax channel is substituted; this hybrid separation prevents the stricter relation prompt from degrading the core ontology channels.

Guarded v2 separates evidence into two modes. `explicit` requires one verbatim span containing both endpoints. `cross_sentence` permits exactly two nearby verbatim spans, requires every span to anchor at least one endpoint, requires the combined spans to cover both endpoints, and uses a stricter raw-confidence threshold (0.95 vs. 0.85). This supports locally implicit relations without accepting unconstrained background-knowledge inference.

Before filtering, v2 canonicalizes entity separators/surface forms against the context and legal Task B inputs, and maps direction-preserving predicate aliases to a frozen ontology-style vocabulary. Directional aliases such as “gives rise to” explicitly reverse endpoints when mapped to `develops_from`; every transformation is logged. Confidence is then calibrated on a deterministic sample-level development split using a Beta-smoothed isotonic model. A calibration artifact is deployable only when both positive and negative support exceed minimum counts. A zero-positive split is reported as degenerate and is never silently applied.

Reproducibility uses a two-level cache contract. A prompt hash identifies the immutable raw LLM generation, while a separate postprocess hash covers canonicalization version, evidence constraints, thresholds, quantity cap, and calibrator parameters. Changing only the latter triggers local replay of cached `raw_text`, not another model request. This permits calibration and threshold ablations with zero additional external calls after the first v2 generation.

The axiom-only surface/serialization diagnostic is deliberately separate from the production canonicalizer. Version `relation_surface_morphology_v1` singularizes only the final token of an endpoint using low-ambiguity rules (`-ies` to `-y`, or terminal `s` removal when the token does not end in `-es`), while protecting short tokens and endings such as `-ss`, `-us`, `-is`, and `-ics`. For the symmetric predicates `equivalent class` and `disjoint with`, the quantity-aware serializer first retains every accepted base prediction in model order, then fills only the remaining slots up to the existing maximum of eight with reverse edges. Exact-match transition logs record every recovered and lost gold edge. The rule remains diagnostic and fail-closed until an independent source supplies at least five positive samples under complete supervision.

**Step OR3 — Clustering into vocabulary.** Embed mined phrases; agglomerative clustering (cosine threshold \(\tau_{\mathrm{rel}}\)). Name each cluster by medoid phrase or LLM canonicalization (`located_in`, `has_part`, `used_for`, …). Result: **dataset-adaptive inventory** \(\hat{P}_{\mathrm{open}}\).

**Step OR4 — Induce domain/range sketches.** For each \(\hat{p} \in \hat{P}_{\mathrm{open}}\), observe Stage-2 types of heads/tails:

\[
\hat{\mathrm{dom}}(\hat{p}) = \mathrm{mode}\{\hat{\mathrm{type}}(h)\}, \quad
\hat{\mathrm{ran}}(\hat{p}) = \mathrm{mode}\{\hat{\mathrm{type}}(t)\}
\]

Soft compatibility replaces hard gold domain/range when TBox properties are missing.

**Step OR5 — Score, filter, project.**

- Score triples with \(S_{\mathrm{rel}}\) (Section 2.1).
- Apply R2–R5, R9–R10.
- **Projection for official metrics** (if needed): map \(\hat{p}\) to gold \(p \in P_{\mathrm{gold}}\) via string/embedding match + LLM alignment; unmapped predicates count as FP under closed-set F1 but are reported under open-set precision separately.

### 4.2 Open-set vs closed-set reporting

| Mode | What is scored | Purpose |
|------|----------------|---------|
| Closed-set F1 | projected labels in \(P_{\mathrm{gold}}\) | comparable to official Task C |
| Open-set metrics | coverage of gold predicates + novelty precision | scientific claim on open vocab |

Paper must report **both**; claimed contribution is recovering *any* meaningful non-tax structure where baselines output \(\emptyset\).

### 4.3 Fallback when gold inventory is provided

If the task year supplies a full property inventory, OntoBridg still mines open vocab as an **auxiliary channel**, then unions projected predictions with closed-set classification. Ablation: `closed-only` vs `open-only` vs `hybrid`.

---

## 5. Contamination Stress Tests (Definitions)

### 5.1 Membership-style probes

For each gold item \(x\) (term–type pair, taxonomy edge, or relation triple) and matched control \(x'\) (random plausible alternative):

Define probe score

\[
m(x)
=
\frac{1}{|Q_x|}
\sum_{q \in Q_x}
\mathbb{1}\big[\mathrm{LLM}(q) \models \text{gold fact about } x\big]
\]

where \(Q_x\) includes:

- definition completion prompts
- “Is A a kind of B?” subclass probes
- cloze templates for relations (“A is ___ of B”)

**Contamination grade** for dataset–task pair \((\mathcal{D}, \tau)\):

\[
\mathrm{CG}(\mathcal{D}, \tau)
=
\begin{cases}
\text{high} & \text{if } \mathrm{AUC}(m(x)_{\mathrm{gold}}, m(x')_{\mathrm{ctrl}}) \ge 0.8 \\
\text{med}  & \text{if } 0.65 \le \mathrm{AUC} < 0.8 \\
\text{low}  & \text{if } \mathrm{AUC} < 0.65
\end{cases}
\]

*（阈值可在实验中校准；报告 AUC 连续值 + 离散 grade。）*

### 5.2 Surface-form stress tests

For paraphrase operator \(\pi\) (synonym substitution, rare surface form, word-order change for multiword terms):

\[
\Delta_{\mathrm{stress}}
=
\mathrm{F1}(X) - \mathrm{F1}(\pi(X))
\]

Large \(\Delta_{\mathrm{stress}}\) ⇒ surface memorization reliance. OntoBridg prediction: pure LLM baselines have larger \(\Delta\) than hybrid+constraints on the same backbone.

Variants:

- **Term paraphrase**: replace gold term surface; keep gold type.
- **Type paraphrase**: replace type label with synonym (careful: may leave inventory).
- **Relation paraphrase**: replace predicate phrase; test open-vocab vs closed-set.

### 5.3 Clean / polluted split

Partition test items:

\[
X_{\mathrm{clean}} = \{x : m(x) \le q_{30}\}, \quad
X_{\mathrm{polluted}} = \{x : m(x) \ge q_{70}\}
\]

Report \(F1\), \(SGS\), and all ablations on each split + full test. **Primary scientific table:** system ranking on `full` vs `clean`.

### 5.4 Synthetic / temporal holdout

- **Synthetic**: LLM-or-rule-generated novel terms with expert or cross-resource types not in gold; evaluate typing transfer.
- **Temporal**: prefer ontology versions / pages dated after assumed pretraining cutoff when available.

### 5.5 Protocol outputs

| Artifact | Content |
|----------|---------|
| `contamination_scores.json` | per-item \(m(x)\), AUC, grades |
| `eval_report.md` | official + split metrics + stress deltas |
| `ranking_flip.json` | system order on full vs clean |

---

## 6. Experimental Controls

- **Same backbone** across hybrid vs pure-LLM when comparing methods (e.g., Qwen2.5-14B) to isolate constraint/RAG effects from scale.
- **Seeds & decoding**: temperature \(= 0\) for main runs; \(n \ge 3\) seeds for stochastic retrieval.
- **Prompt freeze**: version prompts in `experiments/exp_*/prompts/`.
- **Leakage audit**: assert RAG index ∩ test gold \(= \emptyset\) in CI-style checks.
- **License**: UMLS-derived artifacts stay out of public git; document access path only.

---

## 7. Claims → Evidence Map

| Claim | Evidence needed |
|-------|-----------------|
| H1: symbolic constraints improve F1/SGS | Ablation constraints on/off, especially clean split |
| H2: contamination inflates LLMs4OL metrics | Probe AUC + large full-vs-clean gaps for zero-shot |
| H3: contamination-aware eval changes conclusions | Ranking flip table; SGS vs F1 sensitivity |
| Open vocab recovers relations | Task C non-zero vs pro-team-like \(\emptyset\); closed/open dual report |
| Hybrid matches or beats Alexbek/pro-team | Controlled baseline table (`baseline_comparison.md`) |

---

## 8. Implementation Notes (code-aligned)

- Package root: `D:\NLP\src\ontobridg\`
- Planned modules: `stages/`, `constraints/`, `rag/`, `llm/`, `eval/`
- Existing: `data/schema.py`, `data/loaders.py` (CSV/JSON ontology IO, `OntologyKnowledge` prior simulation)
- Config: `experiments/exp_YYYYMMDD_*/config.yaml`
- Reproduce command (to document once Stage 2–6 are online):

```powershell
# example target — fill exact entrypoint when implemented
python -m ontobridg.run --config experiments/exp_2026xxxx_main/config.yaml
```

---

## 9. Open Technical Questions

1. Exact **SGS** formula and weights from 2026/2027 task guidelines.
2. Whether Task A is single-label or multi-label in the target year.
3. Optimal \(\alpha,\beta,\gamma,\delta\) schedule: global vs per-dataset.
4. Hard vs soft constraint application under high contamination (adaptive schedule).
5. Whether open relation projection should use LLM alignment or embedding nearest gold property.

---

*End of methodology v0.1. Expand subsections in place when experiments land; keep formulas stable so the paper draft can cite them.*
