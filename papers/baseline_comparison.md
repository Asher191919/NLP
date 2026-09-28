# OntoBridg — Baseline Comparison Table

> **Archived planning matrix.** This document predates the completed official-train, nonce, transfer, and predicate-routing experiments. Submission-ready claims and measured tables now live in `paper_draft.md` and `appendix.md`; unresolved `TBD` cells below are not manuscript evidence.

> **Status**: Comparison matrix v0.1 | 2026-09-20
> **Use**: Feeds Section 4 of `paper_draft.md`; refresh numbers after reproduction.
> **Caveat**: Metrics for external systems are **as reported / qualitative** until OntoBridg reproduces them under the same backbone & splits. Official LLMs4OL year protocols may differ (F1-only vs SGS+F1).

---

## Main Comparison Matrix

| System | Year | Method | Non-tax rels | SGS / F1 | Gap we exploit |
|--------|------|--------|--------------|----------|----------------|
| **LLMs4OL zero-shot baseline** | 2023 | Prompt-only instruction LLM (original shared-task baselines); no RAG, no graph constraints | Weak / often empty or noisy; closed or ad-hoc relation prompts | Task-wise micro-F1 only (2023 protocol); typing non-trivial, taxonomy fragile, graph quality poor | No acyclicity/domain-range repair; high contamination sensitivity; no open relation vocab |
| **Flan-T5 fine-tune (supervised encoder/seq2seq)** | 2023–24 | Fine-tune Flan-T5-class model on train splits for typing / taxonomy labels | Closed relation inventory if supervised; poor on unseen predicates | Competitive typing F1 when train/test aligned; weak graph-level SGS without structure decode | Brittle under type shift & paraphrase; no symbolic graph assembly; closed vocab fails Task C open-world |
| **Alexbek 2025** | 2025 | **RAG + embedding ensemble + cross-attention taxonomy** module (competition winner) | Improved taxonomy; non-tax not the headline strength (inventory / stage focus) | **Winner-level** official scores (cite exact from arXiv:2508.19428); strong A/B | Contamination not audited; hybrid still primarily neural ranking — limited hard symbolic repair; open relation vocab not emphasized |
| **pro-team 2026** | 2026 | **Task A Flagship + Task B Reuse**; **Qwen2.5-14B + RAG**; eval **Official Graph Similarity + Term-Typing F1 + Taxonomy F1** | **FAILED — zero non-taxonomic relations** in Flagship/Reuse triples | GS 0.7416 (A) / 0.8692 (B); Typing 0.92; Taxonomy 0.854 (arXiv:2608.27101); did not target Task C Taxonomy | Closed/ill-specified relation inventory + typing filters; no open vocab; contamination-free framing |
| **OntoBridg (ours)** | 2026–27 | Hybrid neuro-symbolic: RAG + symbolic repair + **open relation vocab** + **official Graph Similarity reporting** + contamination-aware eval | **Target: non-zero** non-tax triples on Flagship/Reuse; Task C Taxonomy via OntoLearner | **Plan:** match/beat A/B on official GS+F1; primary claim stable gains on **clean** subset | Exploit: (1) zero non-tax in pro-team Flagship/Reuse, (2) contamination inflation, (3) no hard graph repair, (4) closed relation lists |

---

## Compact View (for paper teaser table)

| System | Year | Hybrid symbolic? | RAG? | Open relations? | Contamination protocol? | Non-tax outcome |
|--------|------|------------------|------|-----------------|-------------------------|-----------------|
| Zero-shot LLMs4OL | 2023 | No | No | No | No | Unreliable |
| Flan-T5 FT | 2023–24 | No | No | No (closed) | No | Closed-set only |
| Alexbek | 2025 | Partial (neural structured) | Yes | Limited | No | Not the focus |
| pro-team | 2026 | Light (reuse) | Yes (Qwen2.5-14B) | No | No | **Zero (failed)** |
| **OntoBridg** | 2026–27 | **Yes (hard constraints)** | Yes | **Yes** | **Yes (core contribution)** | **Target: recovered** |

---

## Gap Analysis (What We Exploit)

### Gap 1 — Empty non-taxonomic relation output (pro-team 2026)

- **Evidence:** arXiv:2608.27101 reported failure to extract any non-taxonomic relations despite strong RAG + large backbone.
- **Cause hypothesis:** closed inventory mismatch, aggressive domain/range filters without open-world mining, taxonomy-centric RAG exemplars.
- **OntoBridg response:** Stage 4 open relation vocabulary (methodology.md §4): mine predicates from definitions → cluster → induce soft domain/range → score → optional projection for official F1.
- **Success metric:** Task C F1 > 0 on at least WordNet/schema.org; open-set coverage of gold predicates reported alongside closed-set F1.

### Gap 2 — Training-set contamination as systematic eval flaw

- **Evidence:** LLMs4OL datasets (WordNet, GeoNames, UMLS, schema.org) are public artifacts; article insight: contamination systematically inflates LLM ontology matching/learning scores; OAEI contrast: hybrid symbolic still wins on **clean** tracks, LLM alone often worse.
- **OntoBridg response:** Stage 6 contamination protocol — membership probes, surface-form stress, clean/polluted split reporting for F1 **and** SGS.
- **Success metric:** Document high/med/low grades; show ranking sensitivity on clean subset; show smaller \(\Delta_{\mathrm{stress}}\) for hybrid vs zero-shot.

### Gap 3 — Soft neural scores without graph-level repair

- **Evidence:** Winner systems emphasize retrieval + attention; limited explicit acyclicity/transitive/domain-range enforcement in reported pipelines.
- **OntoBridg response:** Symbolic repair rules R1–R10 (methodology.md §3); report violations before/after and delta metrics.
- **Success metric:** Constraint ablation improves Taxonomy F1 / SGS especially on clean data (H1).

### Gap 4 — Metric regime change (F1 → SGS + F1) under contamination

- **Evidence:** 2026 scoring uses Semantic Graph Similarity + Term-Typing F1 + Taxonomy F1; unclear whether SGS is more memorization-sensitive than edge F1.
- **OntoBridg response:** Implement SGS-proxy + official weights when known; RQ4 on relative contamination sensitivity.
- **Success metric:** Correlation / gap analysis full vs clean for SGS vs F1 across systems.

---

## Reported / Planned Scores (Fill After Experiments)

> **Policy:** Do not invent numbers. External cells cite paper-reported values when verified; OntoBridg cells stay `TBD` until runs exist.

| System | WordNet Typing F1 | WordNet Tax F1 | WordNet Rel F1 | WordNet SGS | schema.org Typing F1 | Clean-split note |
|--------|-------------------|----------------|----------------|-------------|----------------------|------------------|
| Zero-shot LLMs4OL | TBD (repro) / cite 2307.16648 | TBD | TBD | N/A (2023) | TBD | Expect large drop on clean |
| Flan-T5 FT | TBD | TBD | TBD | N/A | TBD | Medium drop under paraphrase |
| Alexbek 2025 | Cite arXiv:2508.19428 | Cite | Cite if reported | Cite if reported | Cite | Unknown without repro |
| pro-team 2026 | Cite arXiv:2608.27101 | Cite | **0 (reported fail)** | Cite SGS/F1 | Cite | Contamination unaudited |
| **OntoBridg** | TBD | TBD | **Target > 0** | TBD | TBD | **Must report full vs clean** |

---

## Reproduction Protocol (Fair Comparison)

1. **Backbone control:** Re-run RAG baselines on the same backbone when possible (e.g., Qwen2.5-14B to mirror pro-team) and also a common open model for all hybrid variants.
2. **Identical splits:** Official LLMs4OL train/dev/test for the target year.
3. **Leakage firewall:** No test gold in RAG index (methodology.md Stage 0).
4. **Prompt freeze:** Store exact templates under `experiments/exp_*/prompts/`.
5. **Dual metric regime:** Official composite (SGS+F1…) **and** per-task F1 **and** contamination splits.
6. **Task C dual report:** closed-set projected F1 + open-set coverage.

---

## OAEI Contrast (Secondary Narrative — Not Main Track)

| Setting | Typical finding | Implication for OntoBridg |
|---------|-----------------|---------------------------|
| OAEI clean alignment tracks | Hybrid symbolic (LogMap, AML, BERTMap) often ≥ pure LLM | LLM = candidate generator; symbolic = repair/veto |
| LLMs4OL public OL graphs | LLM scores look strong; contamination untested | Need contamination-aware protocol before claiming induction ability |
| OntoBridg position | Borrow OAEI lesson into OL | Hybrid constraints + open vocab + clean-split honesty |

*（OAEI 2026 execution已过；对比仅用于 Related Work / positioning，不作为主实验。）*

---

## Citation Keys

| Key | Reference |
|-----|-----------|
| LLMs4OL-2023 | arXiv:2307.16648 |
| LLMs4OL-2024-overview | arXiv:2409.10146 |
| Alexbek-2025 | arXiv:2508.19428 |
| pro-team-2026 | arXiv:2608.27101 |
| Contamination-article | *(fill formal citation — systematic flaw for LLM ontology matching/learning)* |

---

## Update Checklist

- [ ] Pull exact numeric tables from arXiv:2508.19428 and arXiv:2608.27101 (datasets × metrics)
- [ ] Confirm 2026 SGS formula/weights
- [ ] Run OntoBridg pilots → fill TBD row
- [ ] Add per-dataset appendix (UMLS, GeoNames) when license/runs allow
- [ ] Sync tables into `paper_draft.md` §4 and workshop CEUR version
