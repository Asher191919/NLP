# Experimental Results (OntoBridg) — filled skeleton

> **Status**: blind Task A/B n = 30, guarded-v1/v2 density evaluation, calibration/axiom diagnostics, the 60-unit nonce holdout, the 60-unit official-text relation transfer, and the 54-unit predicate-routing evaluation plus error audit are complete; legacy oracle pilot retained for provenance | 2026-09-28
> **Metric policy**: external numbers use **official LLMs4OL 2026 Graph Similarity** (exact_match) + micro-F1. Internal proxy SGS is **not** reported as leaderboard-style SGS.  
> **Model**: dots3-note-prev (dots studio, OpenAI-compatible free trial)  
> **Data**: official LLMs4OL 2026 Flagship (Task A) / Reuse (Task B) train splits, no test-set claims.  
> **Scope**: the primary Task A/B n = 30 tables use leakage-controlled protocol `llms4ol2026_blind_v1`; the old n = 10 runner used gold-derived candidate inventories and is an **oracle-inventory pilot only**. All measurements use official **training** samples, not hidden test data or a 2026 leaderboard submission. The **2026 competition is closed** — results target the **2027 competition + paper**.

---

## 4.1 Experimental Setup

### Tasks

| Competition task (2026) | Local key | Input | Output |
|-------------------------|-----------|-------|--------|
| **A Flagship** | `task_a` | `context` (title + content) | `primitive-ontology-triples` |
| **B Reuse** | `task_b` | `context` + `initial-...` ontology + given terms/types | `extended-primitive-ontology-triples` |

### Systems

| ID | Description |
|----|-------------|
| `closed_vocab` | pro-team-style baseline: term typing + taxonomy only; **no non-taxonomic relations** (closed vocab filter drops them) |
| `open_vocab` | OntoBridg: same typing path + **open non-tax relation extraction** + symbolic repair hooks |
| `open_vocab_guarded` | Direct guarded generation: every non-tax triple carries confidence and a verbatim evidence span; local filters enforce evidence, predicate, confidence, self-loop, and quantity constraints |
| `open_vocab_guarded_hybrid` | Locally derived hybrid: legacy open supplies typing/taxonomy and guarded open supplies non-tax relations; requires no additional API calls |
| `open_vocab_guarded_v2` | Adds predicate/entity canonicalization, explicit or two-span cross-sentence evidence, separate implicit-confidence and distance gates, and fail-closed calibrated confidence |
| `open_vocab_guarded_v2_hybrid` | Legacy typing/taxonomy plus guarded-v2 relations, derived locally without an API call |
| `open_vocab_guarded_v3_relations` | Relation-only general-predicate control for the frozen routing test; uses the same postprocessor as routed v6 |
| `open_vocab_guarded_v6_predicate_routed` | Relation-only routed treatment restricted to the frozen eight-predicate inventory; evaluated against `open_vocab_guarded_v3_relations` with the same postprocessor |

All direct systems use the same backbone (`dots3-note-prev`), temperature 0, offline JSON prompts, and capped max_tokens.

### Metrics (external-reportable only)

- Term-Typing F1 (micro)
- Taxonomy F1 (micro, `is-a` edges)
- Non-taxonomic Relation F1 (micro)
- **Official Graph Similarity** = \((EdgeF1 + Neighborhood + Taxonomy)/3\)  
  via `ontobridg.eval.official_graph_similarity` (aligned with challenge `metrics/graph_similarity.py`)
- Composite = \((TypingF1 + TaxonomyF1 + GraphSim)/3\) (internal summary; not an official score)

### Scale & runtime notes

- Primary blind run: Task A/B × closed/open × n = 30 = **120/120 successful experimental units**; legacy oracle-inventory diagnostic n = 10; offline pattern baseline n = 40.
- Density-stratified follow-up: the legacy/guarded-v1 comparison completed **60/60 API units**. Guarded-v2 then added **30/30 successful new external calls** on the same samples; both hybrid variants are derived locally. Each task has five low-, five medium-, and five high-density samples.
- Independent relation calibration now contains **39/39 successful unique v2 units** (27 Task A, 12 Task B), disjoint from the blind prefix and density evaluation. Including six earlier connection failures, the deduplicated ledger contains 45 unit-level attempts; two successful units required an internal retry, giving 47 low-level runner/connection attempts. Gold was used locally for sample selection and candidate labeling only and never entered a prompt.
- A separate 2026 Task A axiom-only v5 extension completed **8/8 successful units** after an initial 8/8 token-cap failure at 1,024 completion tokens. The ledger retains all 16 unit attempts and 24 low-level attempts (16 failed plus 8 successful); the successful retry used a 16,384-token cap. These gold-selected calibration samples are disjoint from the blind prefix, density split, 39-unit v2 calibration set, and weak-label packet.
- The frozen independent axiom-only validation set contains eight additional Task A samples (`678, 2171, 2643, 2780, 2920, 2958, 3064, 3676`). All **8/8 calls succeeded** with eight low-level attempts, no retry, and a 16,384-token completion cap (median latency 29.234 s; maximum 156.852 s). The samples are disjoint from every earlier experimental ledger, gold remained prompt-hidden, and the completed caches now reproduce at 8/8 exact hits with zero estimated external calls.
- A relation-only v3 pilot added **3/3 successful calls** on the remaining supported Task A failures. It produced 11 accepted relation candidates and no exact true positive; it is reported as a negative mechanism diagnostic, not merged into calibration.
- A cross-year mechanism diagnostic added two **10/10 successful** passes over the same ten public 2025 documents. The documents were linked locally to 11 unique 2025 Task D axiom pairs and filtered to zero overlap with all 2026 Task A/B train gold. V4 generated 44 raw/34 accepted candidates but no exact TP; v5 generated 10 raw/8 accepted candidates and one exact TP from one sample. These endpoint-co-occurrence links are calibration diagnostics, not an unbiased evaluation set, and neither pass is merged into the v2 calibrator.
- Guarded-v2 raw generation and postprocessing use separate cache identities. After the 30 raw responses existed, canonicalizer, evidence-gap, predicate-policy, and calibrator changes were replayed with **0 external calls**.
- Final successful-cache latency was 14.5–814.5 s per unit (median 61.1 s; p95 209.0 s). Four final records needed a second runner attempt; per-sample cache and atomic checkpoints allowed safe resume.
- Some Task A generations exhausted 4,096 tokens and four conditions exhausted 8,192 tokens before emitting final JSON; the final reproducible cap is 16,384. A transient endpoint outage affected six Task B open-vocabulary samples, all recovered by checkpointed retry.
- The frozen nonce holdout completed **60/60 units** in 67 recorded attempts (32 raw-open, 35 guarded-v2). Its completed generation cache audits at 60/60 hits with zero external calls; prompt, threshold, and calibrator tuning remain prohibited.
- The frozen official-text transfer completed **60/60 units** in 81 recorded attempts (34 raw-open, 47 guarded-v2). Completion-length recovery raised only the API completion cap from 8,192 to 16,384; frozen samples, prompt hashes, thresholds, calibrator state, and postprocessing were unchanged. Its final generation and postprocess caches both audit at 60/60 hits with zero external calls.
- The independent predicate-routing packet `official_predicate_routing_v1` completed **54/54 units in 57 recorded attempts**. Routed-v6 raises four-stratum macro target F1 from 0 to 0.0526 but yields only 2 TP-positive samples versus the required 5. Completed generation and postprocess caches both audit at 54/54 hits with zero external calls.
- Primary result files: `official_blind_openai_a_n30_off0.json` and `official_blind_openai_b_n30_off0.json`.
- Stratified result and combined payload audit: `official_stratified_open_openai_ab_s5_seed20260922.json` and `official_stratified_open_audit_ab_open_vocab_open_vocab_guarded_open_vocab_guarded_v2_s5_seed20260922.json`.
- Frozen nonce manifest/result: `nonce_relation_holdout_manifest_n30_seed20260926.json` and `nonce_relation_holdout_openai_n30_seed20260926.json`.
- Frozen official-text transfer manifest/result: `official_relation_transfer_manifest_a_n30_seed20260927.json` and `official_relation_transfer_openai_a_n30_seed20260927.json`.
- Frozen predicate-routing package/manifest/result/error audit: `experiments/holdouts/official_predicate_routing_v1/`, `official_predicate_routing_manifest_a_n27_seed20260927.json`, `official_predicate_routing_openai_a_n27_seed20260927.json`, and `official_predicate_routing_error_audit_a_n27_seed20260927.json`.

---

## 4.2 Task A Flagship — main results

### 4.2.1 Legacy oracle-inventory pilot (dots3-note-prev), n = 10

Sources:

- closed: `experiments/results/official_train_baseline_openai_a_n10.json`
- open: `experiments/results/official_train_baseline_openai_a_n10_open_merged.json` (micro-merge of offset chunks)

| System | n | Typing F1 | Taxonomy F1 | NonTax F1 | **Official GS** | Composite | non-tax pred/gold |
|--------|---|-----------|-------------|-----------|-----------------|-----------|-------------------|
| `closed_vocab` | 10 | 1.0000 | 0.0247 | 0.0000 | **0.1825** | 0.4024 | 0 / 1 |
| `open_vocab` | 10 | 1.0000 | **0.1176** | 0.0000 | **0.2251** | **0.4476** | 0 / 1 |
| Δ (open − closed) | | 0.0000 | **+0.0930** | 0.0000 | **+0.0426** | **+0.0452** | — |

**Reading.** This table measures a legacy oracle-assisted component setting: candidate terms/types were derived from gold triples before prompting. It can diagnose open-vs-closed behavior within that runner, but it is not valid end-to-end Flagship evidence. Typing saturation must not be cited as model performance.

### 4.2.2 Leakage-controlled real LLM (dots3-note-prev), n = 30

_Target file:_ `official_blind_openai_a_n30_off0.json`  
_Protocol:_ Task A receives context only; gold triples remain local and scoring-only. All 60 Task A units completed.

| System | n | Typing F1 | Taxonomy F1 | NonTax F1 | Official GS | Composite | non-tax |
|--------|---|-----------|-------------|-----------|-------------|-----------|---------|
| `closed_vocab` | 30 | **0.1400** | **0.4903** | 0.0000 | **0.4404** | **0.3569** | 0 / 1 |
| `open_vocab` | 30 | 0.1356 | 0.3516 | 0.0000 | 0.1603 | 0.2159 | 384 / 1 |
| Δ (open − closed) | | -0.0044 | -0.1387 | 0.0000 | -0.2801 | -0.1411 | +384 / 0 |

**Reading.** Under blind context-only input, the closed path outperforms the open path on taxonomy, official Graph Similarity, and the internal composite. The 30-sample prefix contains only one gold non-taxonomic edge. The open path predicts 384 non-tax edges but does not recover that edge, so this slice provides no evidence of non-tax recall and instead exposes severe relation over-generation under exact scoring.

### 4.2.3 Offline pattern baseline (no LLM), n = 40

Source: `official_train_baseline_offline_a_n40.json`

| System | n | Typing F1 | Taxonomy F1 | NonTax F1 | Official GS | Composite |
|--------|---|-----------|-------------|-----------|-------------|-----------|
| `closed_vocab` | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `open_vocab` | 40 | 0.0045 | 0.0252 | 0.0000 | 0.0373 | 0.0223 |

**Reading.** Without an LLM, official Flagship is nearly unsolvable; confirms the need for generative extraction rather than surface patterns alone.

---

## 4.3 Task B Reuse — results

### 4.3.1 Offline pattern baseline (no LLM), n = 40

Source: `official_train_baseline_offline_b_n40.json`

| System | n | Typing F1 | Taxonomy F1 | NonTax F1 | Official GS | Composite |
|--------|---|-----------|-------------|-----------|-------------|-----------|
| `closed_vocab` | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `open_vocab` | 40 | 0.0010 | 0.0411 | 0.0000 | 0.0343 | 0.0255 |

### 4.3.2 Leakage-controlled real LLM (dots3-note-prev), n = 30

_Source:_ `official_blind_openai_b_n30_off0.json`. Task B receives context, the official initial ontology, and official terms/types; gold extension triples remain local and scoring-only. All 60 Task B units completed.

| System | n | Typing F1 | Taxonomy F1 | NonTax F1 | Official GS | Composite | non-tax |
|--------|---|-----------|-------------|-----------|-------------|-----------|---------|
| `closed_vocab` | 30 | 0.8824 | **0.9474** | 0.0000 | **0.9375** | **0.9224** | 0 / 0 |
| `open_vocab` | 30 | **1.0000** | 0.8966 | 0.0000 | 0.6449 | 0.8471 | 150 / 0 |
| Δ (open − closed) | | +0.1176 | -0.0508 | 0.0000 | -0.2927 | -0.0753 | +150 / 0 |

**Reading.** With the official Task B inputs, both LLM paths are strong on typing and taxonomy. The open path reaches perfect typing but loses 0.2927 official GS because its 150 non-tax predictions have no matching gold edges in this prefix; the closed path therefore has the stronger composite. Because the gold count is zero, NonTax F1 = 0 is a sparsity/false-positive diagnostic rather than a recall measurement.

---

## 4.4 Density-stratified guarded open evaluation

_Source:_ `official_stratified_open_openai_ab_s5_seed20260922.json`. Selection uses official gold **locally** to form strata, but gold never enters a model prompt. Strata are based on non-self non-tax gold: low = 1, medium = 2–4, high = at least 5. Each task contributes five samples per stratum (n = 15). The comparison contains legacy `open_vocab`, guarded-v1, and guarded-v2; each hybrid retains the legacy typing/taxonomy channel and replaces only its non-tax channel.

The guarded-v1 contract requires a confidence score and one verbatim context span containing both endpoints. Guarded-v2 additionally canonicalizes predicates/entities and permits two nearby verbatim spans for cross-sentence evidence. Prespecified v2 gates are confidence ≥ 0.85 for explicit evidence, confidence ≥ 0.95 and gap ≤ 600 normalized characters for cross-sentence evidence, plus generic-predicate/self-loop vetoes and a maximum of eight non-tax predictions per sample.

| Task | System | n | Typing F1 | Taxonomy F1 | NonTax F1 | Official GS | Composite | non-tax pred/gold |
|------|--------|---|-----------|-------------|-----------|-------------|-----------|-------------------|
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

Relative to legacy open, the v1 hybrid cuts non-tax predictions by 216/250 (86.4%) on Task A and 42/46 (91.3%) on Task B. Official GS rises by 0.0315 and 0.0409, respectively. Direct guarded-v1 generation harms typing and taxonomy, motivating isolation of the constraints to the relation channel.

Guarded-v2 is less sparse: its hybrid cuts predictions by 78.4% on Task A and 58.7% on Task B. Relative to legacy, v2-hybrid GS changes by +0.0327 and +0.0168. It slightly exceeds v1-hybrid GS on Task A (+0.0012) but trails it on Task B (−0.0241), so two-span recall support trades away some of v1's graph-level precision.

| Task | Stratum | non-self gold | legacy pred | hybrid pred | Δ Official GS |
|------|---------|---------------|-------------|-------------|---------------|
| A | low | 5 | 82 | 8 | +0.0209 |
| A | medium | 16 | 92 | 17 | +0.0729 |
| A | high | 42 | 76 | 9 | +0.0006 |
| B | low | 5 | 3 | 0 | +0.0287 |
| B | medium | 12 | 17 | 0 | +0.0650 |
| B | high | 35 | 26 | 4 | +0.0290 |

Guarded-v2 hybrid by stratum under the prespecified 600-character gap:

| Task | Stratum | non-self gold | legacy pred | v2-hybrid pred | Δ Official GS |
|------|---------|---------------|-------------|----------------|---------------|
| A | low | 5 | 82 | 26 | +0.0084 |
| A | medium | 16 | 92 | 18 | +0.0834 |
| A | high | 42 | 76 | 10 | +0.0061 |
| B | low | 5 | 3 | 5 | −0.0153 |
| B | medium | 12 | 17 | 9 | +0.0344 |
| B | high | 35 | 26 | 5 | +0.0311 |

**Guarded-v1 negative result and diagnosis.** Exact-match relation recall does not improve: Task A remains at zero true positives, while Task B loses the legacy system's sole hit, `infructescence axis --part_of--> infructescence`. The guarded model generated no non-tax candidate for that sample; the edge requires a cross-sentence inference and therefore cannot satisfy the single-span evidence contract. Filtering itself is not the main source of the reduction: Task A accepts 34 of 36 raw non-tax candidates and Task B accepts all 4. All emitted non-tax confidence values fall between 0.90 and 1.00, so the reported confidence has little discrimination. This diagnosis motivated v2 canonicalization, two-span evidence, and calibration support.

**Guarded-v2 evidence and calibration result.** The v2 run produced 111 raw non-tax candidates after core-relation canonicalization. Ten changed under predicate/entity canonicalization, but canonicalization alone recovered no exact non-tax match. Of 48 cross-sentence candidates, 27 pass the prespecified gates. The model generated the one exact Task B relation with two-span evidence at confidence 0.95, but its evidence gap is 1,604 characters, so the prespecified 600-character gate rejects it. Thus v2 improves the generation failure of v1, but the primary thresholded result still has zero exact relation TP.

The original within-evaluation split was degenerate (0/34 positive candidates) and is retained only as a failed diagnostic. The independent, relation-enriched calibration set now contains 39 samples (27 Task A, 12 Task B), excluding the blind prefix and all density-evaluation indices. It contains 177 raw non-tax candidates; 144 pass evidence/confidence rules before the per-sample quantity cap and form the calibrator population. Four exact positives come from three independent samples: two `part_of` edges in Task A sample 2100, one `regulates` edge in Task A sample 2963, and one cross-sentence `develops_from` edge in Task B sample 727. Beta-isotonic calibration reduces Brier score from 0.8429 to 0.0270 and ECE from 0.9024 to 0.0113 on this calibration population. Candidate-level checks pass, but the artifact is `deployable=false` with `insufficient_independent_positive_samples` because the preregistered minimum is five positive samples.

The disjoint axiom-only v5 extension targets eight 2026 Task A samples with visible lexical evidence for `equivalent class` or `disjoint with`. It emits 23 accepted candidates against 35 non-tax gold edges and obtains 4 exact TP from 3 samples (precision 0.1739, recall 0.1143, F1 0.1379). In-sample beta-isotonic Brier/ECE change from 0.7452/0.7765 to 0.1403/0.0528, but the five-positive-sample exact gate still fails. Eight additional candidates have the correct predicate and unordered endpoints for a symmetric gold axiom, raising the explicitly non-official order-invariant diagnostic to 12 TP from all 8 samples (precision 0.5217, recall 0.3429, F1 0.4138). Adding conservative final-token singularization raises the non-official order-and-surface-invariant diagnostic to 13 TP and F1 0.4483. Singularization alone recovers `aaqs --equivalent class--> ambient air quality standard` with no lost exact edge, giving exact F1 0.1724. A max-eight-aware serializer preserves all original predictions before adding reverse symmetric edges; combined with singularization it produces 42 predictions, 12 TP, 30 FP, and exact F1 0.3117. The transition audit records eight recovered and zero lost exact edges. This v5 channel is not the v2 generator and is not pooled into the v2 calibrator.

The frozen eight-sample independent validation reverses the apparent postprocessing gain. Base v5 obtains 3 exact TP in 8 predictions against 28 gold edges (precision 0.3750, recall 0.1071, F1 0.1667), with exact positives from 3 samples and constant confidence 0.95. Surface canonicalization recovers no exact edge. Surface canonicalization plus max-eight-aware reverse serialization emits 16 predictions but retains the same 3 TP, adding 8 FP and reducing exact F1 to 0.1364; its transition audit records 0 recovered and 0 lost exact edges. The surface/direction rule therefore fails independent validation and is not promoted to the production canonicalizer.

Pooling the two disjoint v5 slices yields 31 deployment-eligible candidates, 7 exact TP, and 6 positive samples, so the preregistered *numeric* support gate passes. Five-fold sample-grouped OOF calibration covers all 31 candidates exactly once across 14 candidate-bearing samples with zero train/validation sample overlap; all five fold calibrators fit successfully, and Brier/ECE change from 0.6987/0.7245 to 0.1836/0.0830. This is not confirmatory evidence: the formerly independent validation slice was inspected and then reused for fitting and OOF analysis. The pooled artifact therefore records `numeric_gate_passed=true` but forces `scientific_deployable=false` and `confirmatory_eligible=false`. This v5 pool remains separate from the v2 calibrator.

A new confirmatory slice was then frozen before generation. It contains 24 previously unused Task A samples, balanced 12/12 by endpoint-visible `equivalent class` and `disjoint with` gold, with zero overlap against the blind prefix, density evaluation, v2/v3 calibration, weak-label packet, and both prior v5 slices. The remaining official pool contains no unused sample satisfying the earlier strict same-segment lexical-explicit heuristic, so this is explicitly a gold-stratified endpoint-visible slice rather than the same distribution as the first 16 samples or an unbiased test set. All 24 external generations complete in 24 low-level attempts without retry (median latency 17.377 s; maximum 118.624 s). Base v5 emits 17 raw and 15 accepted relations and obtains 4 TP from 4 samples (precision 0.2667, recall 0.0741, F1 0.1159). Replaying the frozen pooled calibrator makes zero external calls; it improves candidate Brier/ECE from 0.6707/0.6840 to 0.2109/0.1238, but the frozen decision threshold removes one correct edge, leaving 3 TP in 14 predictions (precision 0.2143, recall 0.0556, F1 0.0882). The preregistered five-positive-sample gate therefore fails. The scoring artifact records `confirmatory_evidence_valid=true`, `scientific_deployable=false`, and `confirmatory_eligible=false`; no threshold or rule is retuned.

The locked post-confirmatory audit preserves that negative result. Base exact performance is substantially better for `disjoint with` (3 TP/6 predictions, precision 0.5000, F1 0.1579) than for `equivalent class` (1 TP/9 predictions, precision 0.1111, F1 0.0714). Nine of the 11 false positives have no exact endpoint/predicate match; the dominant equivalent-class error is treating a description as an equivalent name. One further prediction is a reverse-only symmetric near hit. A 10,000-replicate sample-cluster bootstrap gives base F1 0.1159 [0.0238, 0.2712], calibrated F1 0.0882 [0, 0.2222], and calibrated-minus-base F1 -0.0277 [-0.1064, 0]. The Brier change is -0.4598 [-0.7517, 0.2739], so the point improvement in probability calibration is uncertain at this sample size. A planning-only normal approximation estimates about 122 total samples (98 additional) for a pooled precision half-width near 0.10. Predicate-specific planning is much less favorable: the base `disjoint with` stratum would require about 194 samples at the observed prediction rate, while only 89 eligible endpoint-visible samples exist before exclusions. This is not a tuning budget or a claim that the gold-stratified slice becomes representative, so immediate expansion is deferred.

We also freeze a secondary semantic protocol before loading the embedding model: official 2026 `semantic_match`, threshold 0.90, `nomic-ai/nomic-embed-text-v1.5` at revision `e9b6763`, and its nested `nomic-bert-2048` remote code at revision `7710840`. On the relation-only per-sample mean, semantic matching changes no sample for either system (0 improved, 24 unchanged). Base exact and semantic Graph Similarity are both 0.0935; calibrated exact and semantic Graph Similarity are both 0.0657. Thus the confirmatory failure is not rescued by the frozen semantic alignment rule. Exact remains primary, and this secondary analysis cannot change the failed gate or authorize retuning.

An exploratory dual-LLM weak-label packet uses the same 30 frozen text-only samples but does not claim human annotation. The two same-origin model routes emit 114 and 105 relations; sample-exact agreement is 0.2000, macro/micro relation-set Jaccard is 0.3646/0.3436, and standard positive agreement (Dice) is 0.5114. Model adjudication freezes 169 text-grounded relations. Replaying the 30 cached guarded-v2 generations requires zero external calls and yields 234 raw/200 deployment-eligible candidates, of which 26 match weak labels across 14 samples. This clears the numeric five-sample support check. In-sample beta-isotonic resubstitution changes Brier 0.7596 -> 0.1131 and ECE 0.8005 -> 0.0037. A deterministic five-fold cross-validation groups by sample, covers all 200 eligible candidates exactly once with zero train/validation sample overlap, and gives OOF Brier 0.7596 -> 0.1134 and ECE 0.8005 -> 0.0045. The 200 candidates occur in 22 candidate-bearing samples; validation folds contain 38--42 candidates and 2--3 positive samples. A 10,000-replicate sample-cluster bootstrap gives calibrated Brier 0.1134 [0.0681, 0.1781] and ECE 0.0045 [0.0013, 0.0867], compared with raw Brier 0.7596 [0.6747, 0.8268] and ECE 0.8005 [0.7062, 0.8693]. These intervals are conditional on the frozen OOF predictions and weak labels. All five fitted calibrators collapse to one probability block (0.124--0.140), so the apparent calibration gain is stable base-rate shrinkage rather than learned confidence ranking. It does **not** clear the scientific gate: the producers are not independent humans, the two annotators agree poorly, and the 30 official samples contain only one non-tax gold edge with zero exact overlap against the 169 weak labels. All artifacts remain `deployable=false`, `confirmatory_eligible=false`; no confirmatory API run is authorized from these labels. The reliability figure is `papers/figures/weaklabel_relation_calibration_reliability.{svg,pdf,png}`.

On the expanded calibration set, the candidate-only F1 sweep selects 0.1538 but yields only 1 TP and 10 FP. Adding the calibration-derived predicate set (`part_of`, `develops_from`, `regulates`) improves this to 1 TP and 1 FP, still with recall 0.25. The sample-support gate therefore blocks deployment and no new confirmatory evaluation is run. For provenance, the earlier A12+B12 artifact produced the following exploratory zero-call replay before this stricter gate was added:

An exploratory, calibration-only structural projection audit then tests whether generated taxonomy edges can recover ontology-specific relations. The lexical rule `(whole) is-a * substance` -> `* substance derives from (whole)` is exact on Task A samples 938 and 2077 (2/2 pooled and 2/2 under leave-one-sample-out selection). It is the only cross-sample-replicated rule. A matched measurement-to-size rule reaches only 3/4 pooled precision and 1/2 out of fold; unrestricted taxonomy-to-`part_of` and inverse-`has part` reach only 3/219 and 1/219. The `nested position` inverse-`has part` rule recovers sample 2424 but has support from one sample and produces no held-out prediction when that sample is left out. Adding the replicated whole/substance hits to the base candidates would raise the diagnostic union from three to five positive samples, but the projection channel itself has only two positive samples, below its independent five-sample gate. The combined system therefore remains `deployable=false` and no confirmatory API run is started.

The cross-year diagnostic tests whether relation-rich 2025 data can supply the missing independent support without using 2026 evaluation samples. Ten Text2Onto documents are linked by endpoint co-occurrence to 11 unique Task D gold axioms (8 `disjoint with`, 3 `equivalent class`), with any document containing a 2026 train-gold triple excluded in full. In v4, the general relation inventory produces 44 raw and 34 accepted candidates, but none uses either target axiom predicate and exact TP remains zero. V5 restricts generation to the two axiom predicates: 10 raw candidates, 8 accepted, 1 TP, 7 FP, and 10 FN (precision 0.1250, recall 0.0909, F1 0.1053). The sole TP comes from one sample; all eligible candidates have confidence 0.95, so the beta-isotonic fit is `degenerate_too_few_positives` and the five-positive-sample gate fails. A second candidate is a reverse-direction plural surface variant of a symmetric gold pair, but it is not counted by either direction normalization or singularization alone. More importantly, one enumerative passage yields six plausible pairwise disjoint predictions while the linked Task D gold contains only two pairs. This exposes the limitation of endpoint co-occurrence: ontology-level Task D pairs are not a complete set of document-level assertions. As zero-call pressure checks, unconstrained bidirectional serialization lowers exact F1 from 0.1053 to 0.0741, while the max-eight-aware surface-plus-direction strategy recovers `freshwater ecosystem --disjoint with--> marine ecosystem` and reaches 2 TP in 12 predictions (F1 0.1739), with zero lost exact edges. The same rule changes neither TP nor F1 on the 39-sample independent v2 cache or the density slice. Because the only independent gain comes from one additional cross-year sample under incomplete endpoint-linked supervision, the rule remains an axiom-only candidate rather than a deployable default; it is not merged into the v2 confidence model and does not trigger confirmatory evaluation.

| Task | v2-hybrid operating point | NonTax F1 | Official GS | non-tax pred/gold |
|------|---------------------------|-----------|-------------|-------------------|
| A | default recall-oriented v2 | 0.0000 | 0.2152 | 54 / 63 |
| A | calibration predicate policy only | 0.0000 | 0.2218 | 28 / 63 |
| A | predicate policy + calibrated threshold 2/3 | 0.0000 | 0.2288 | 0 / 63 |
| B | default recall-oriented v2 | 0.0000 | 0.5766 | 19 / 53 |
| B | calibration predicate policy only | 0.0000 | 0.5850 | 8 / 53 |
| B | predicate policy + calibrated threshold 2/3 | 0.0000 | 0.6026 | 0 / 53 |

The earlier predicate-only ablation removes 26/54 Task A and 11/19 Task B false positives but recovers no exact relation. Its calibrated operating point abstains on every held-out relation. These are retrospective precision/abstention diagnostics, not deployable results or evidence of improved relation recall.

A clearly post-hoc gap sensitivity at 1,604 characters accepts the Task B TP but also admits two additional Task A false positives. Under this diagnostic only, v2-hybrid Task B NonTax F1 becomes 0.0274 (1/20 predictions versus legacy 1/46 and F1 0.0202), while GS is 0.5779; Task A remains at zero TP with GS 0.2137. Because the threshold was selected after inspecting the held-out TP, these numbers are not primary evidence. The artifact is `official_stratified_open_openai_ab_s5_seed20260922_posthoc_gap1604.json`.

The v2 cache separates `prompt_sha256` (raw generation) from `postprocess_sha256` (canonicalizer, evidence gates, predicate policy, cap, and calibrator). The initial v2 pass required exactly 30 new external calls, all successful. Subsequent canonicalizer fixes, calibration fitting, predicate-policy ablations, and gap sensitivity runs were local-only. Audit manifests record complete generation-cache coverage and `estimated_external_calls=0`, with postprocess-signature changes handled as local replays.

---

## 4.5 Contamination diagnostics and synthetic method check

### 4.5.1 Frozen official-train contamination pilot

The closed-book arm uses the same frozen 15 Task A and 15 Task B samples, with no source context in the prompt. Up to four official facts and four same-predicate object-swap controls are queried per sample. All 30 units succeed, producing 214 balanced items (107 official facts and 107 controls). Overall AUROC is 0.9780 under a 10,000-replicate sample-cluster bootstrap (95% CI [0.9484, 0.9982]); accuracy is 0.9346, with positive/negative-control accuracy 0.8692/1.0000. Task A/B AUROC is 0.9810/0.9882. No non-tax fact is covered by this sparse slice. Probeability has weak and inconsistent Spearman association with per-sample GS across the four task/system cells (point estimates from -0.1329 to 0.1381, all intervals spanning zero). Thus the valid claim is that benchmark facts are highly closed-book probeable, not that benchmark membership in pretraining has been established. Constructed controls may also include unlabeled true world facts, and self-reported confidence is not a model likelihood.

The paired surface arm replaces every gold endpoint visible in legal model inputs with a deterministic opaque alias while preserving predicates and relational sentence structure. It completes 60/60 units in 60 recorded attempts. The prespecified contrast is original minus alias; positive values mean performance fell after aliasing.

| Task | System | n | Original GS | Alias GS | Original - alias GS (95% paired sample-bootstrap CI) |
|------|--------|---|-------------|----------|----------------------------------------------------|
| A | `closed_vocab` | 15 | 0.3628 | 0.4549 | -0.0921 [-0.2577, 0.0798] |
| A | `open_vocab` | 15 | 0.0759 | 0.1861 | -0.1102 [-0.2507, 0.0206] |
| B | `closed_vocab` | 15 | 0.9394 | 0.7837 | +0.1557 [0.0362, 0.3022] |
| B | `open_vocab` | 15 | 0.6743 | 0.4825 | +0.1917 [0.0609, 0.3291] |

All 15 Task B samples have full endpoint visibility, so its sensitivity slice is unchanged. Restricting Task A to the 10/15 samples with `input_visible_endpoint_fraction == 1.0` retains the mixed direction: closed original-minus-alias GS is -0.1754 [-0.3482, -0.0116], and open is -0.1000 [-0.2919, 0.0791]. Whole samples are retained; no per-triple filtering changes the scoring scope. NonTax F1 is 0 in every surface comparison. The experiment therefore supports Task B surface dependence but rules out a simple global claim that surface memorization explains performance. It does not verify contamination, cannot tune generation/calibration/deployment, and gives no non-tax contamination conclusion.

### 4.5.2 Synthetic mock method check (**NON-official**)

Source: `competition_sim_metrics.json` (synthetic multi-domain, **mock backend — NON-official**, method validation only).

Macro-averages over the 4 synthetic domains (from the file's `summary` block):

| System | Task A NonTax F1 | Task A Official GS | Task B NonTax F1 | Task B Official GS |
|--------|------------------|--------------------|------------------|--------------------|
| `pro_team_closed_vocab` | **0.0000** | 0.6026 | **0.0000** | 0.6094 |
| `ontobridg_open_vocab` | **0.7685** | 0.7145 | **0.7683** | 0.7179 |

Reading: closed filter removes non-tax by design (0 non-tax predictions vs 116 gold on both Task A and B); open vocab recovers associative relations (~0.77 NonTax F1).

Contamination protocol (synthetic multi-domain, knowledge-scrubbed holdout; OntoBridg arm, official-GS composite drops from the same file):

- full → high combo / high non-tax F1 (composite level 0.8429)  
- holdout (corpus-blind + knowledge scrub) → large drop (combo drop 0.8429)  
- paraphrase → mild drop (combo drop 0.0041)  
- novel ontology → sharp drop (combo drop 0.7296)  

**These numbers are NON-official method validation only** and must not be compared to pro-team leaderboard scores (different data, backbone, and metric harness).

### 4.5.3 Frozen relation-rich nonce holdout (**COMPLETED SYNTHETIC DIAGNOSTIC**)

To address the official pilot's lack of non-tax support, we freeze `nonce_relation_holdout_v1` before any generation. It contains 30 context-only synthetic samples with 120 non-tax gold relations (four per sample), 60 explicitly negated hard controls, and 300 unique opaque entities. Six predicate families contribute 20 positives each; canonical/paraphrased sample strata are 15/15 and explicit/cross-sentence evidence strata are 60/60. An exact audit finds zero entity-string and zero gold-triple overlap with the official 2026 Task A/B training files. The model-input projection contains only `id` and `context`; protected gold remains local.

The evaluation manifest locks `open_vocab` and uncalibrated `open_vocab_guarded_v2`, exact NonTax micro-F1 as primary, official exact Graph Similarity as secondary, and hard-negative false-positive rate as a diagnostic. Per-predicate exact precision/recall/F1 and explicit/cross-sentence exact gold recall were also fixed before generation. Its immutable metadata intentionally retains the pre-generation `frozen_not_executed` state and 60-call estimate; the separate result artifact records completion. All 60 units completed in 67 recorded attempts (32/35 by system); the completed cache now audits as 60/60 hits and zero external calls.

| System / stratum | Exact P | Exact R | Exact NonTax F1 | Official exact GS | Hard-negative FPR |
|---|---:|---:|---:|---:|---:|
| `open_vocab`, all | 0.2222 | 0.3167 | 0.2612 | 0.1805 | 0/60 = 0.0000 |
| `open_vocab`, canonical | 0.4605 | 0.5833 | 0.5147 | 0.3312 | 0/30 = 0.0000 |
| `open_vocab`, paraphrased | 0.0316 | 0.0500 | 0.0387 | 0.0299 | 0/30 = 0.0000 |
| `open_vocab_guarded_v2`, all | 0.8644 | 0.8500 | 0.8571 | 0.5690 | 0/60 = 0.0000 |
| `open_vocab_guarded_v2`, canonical | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0/30 = 0.0000 |
| `open_vocab_guarded_v2`, paraphrased | 0.7241 | 0.7000 | 0.7119 | 0.4714 | 0/30 = 0.0000 |

For `open_vocab`, exact recall is 0.3167 for both explicit and cross-sentence evidence. For guarded-v2 it is 0.7500 and 0.9500, respectively.

| Predicate | `open_vocab` exact P/R/F1 | Guarded-v2 exact P/R/F1 |
|---|---:|---:|
| `part_of` | 0.7647 / 0.6500 / 0.7027 | 0.7727 / 0.8500 / 0.8095 |
| `has part` | 1.0000 / 0.1000 / 0.1818 | 0.7500 / 0.7500 / 0.7500 |
| `develops_from` | 1.0000 / 0.5000 / 0.6667 | 1.0000 / 1.0000 / 1.0000 |
| `derives from` | 1.0000 / 0.0500 / 0.0952 | 1.0000 / 1.0000 / 1.0000 |
| `located in` | 1.0000 / 0.1000 / 0.1818 | 1.0000 / 0.7500 / 0.8571 |
| `regulates` | 1.0000 / 0.5000 / 0.6667 | 1.0000 / 0.7500 / 0.8571 |

Because the packet has no typing or taxonomy gold, the official GS taxonomy component is zero and a perfect relation graph scores 2/3; NonTax F1 is the primary endpoint.

The frozen comparison therefore supports the relation-only guards on controlled synthetic facts and reduces factual memorization as an explanation. It does not establish unseen-predicate discovery: the predicate inventory is familiar, and the canonical-to-paraphrased drop shows residual surface dependence. Controlled language may overestimate naturalistic extraction. The packet prohibits prompt/threshold tuning and calibrator fitting/application; `confirmatory_eligible=false` and `scientific_deployable=false` remain final.

### 4.5.4 Completed official-text relation transfer

To test whether the nonce gain transfers beyond controlled templates, `official_relation_transfer_v1` freezes 30 previously unused 2026 Task A training samples. Gold-based local selection assigns 10 samples each to `equivalent class`, `disjoint with`, and heterogeneous other-open-predicate strata. Every sample has 2–8 non-self non-tax gold relations whose endpoints occur in the legal 6,000-character context window. The packet contains 89 endpoint-visible and 101 total non-self non-tax gold relations across eight predicates. Its exclusion ledger covers both executed experiments and retained audit-only selections. It has zero overlap with prior evidence in Task A sample index, compact context, or exact non-tax gold, zero exact gold overlap with the nonce packet, and no duplicated visible gold across selected samples.

The evaluation manifest freezes the same raw `open_vocab` and uncalibrated `open_vocab_guarded_v2` identities used in the nonce comparison. Exact NonTax micro-F1 is primary; official exact Graph Similarity is secondary; assigned-stratum, per-gold-predicate, 2–4 versus 5–8 visible-density, and endpoint-visible versus nonvisible recall were prespecified before generation. The immutable manifest retains its pre-generation 0-cache/60-call estimate; completion is recorded in a separate result artifact.

All 60 units completed in 81 recorded attempts. The initial 8,192-token execution exposed completion-length failures in guarded reasoning outputs. Technical recovery used a 16,384-token completion cap without changing samples, prompt hashes, thresholds, calibrator state, or postprocessing, and retained failed attempts in cache provenance. The completed cache replays with 60/60 generation hits, 60/60 postprocess hits, and zero external calls.

| System | Typing F1 | Taxonomy F1 | NonTax P / R / F1 | non-self pred / gold | Official GS |
|---|---:|---:|---:|---:|---:|
| `open_vocab` | 0.0840 | 0.4111 | 0.0000 / 0.0000 / 0.0000 | 424 / 101 | 0.1946 |
| `open_vocab_guarded_v2` | **0.2500** | **0.4180** | 0.0000 / 0.0000 / 0.0000 | 143 / 101 | **0.2593** |

Guarded-v2 reduces non-self non-tax false positives by 281 (66.3%) and raises overall GS by 0.0646, but neither system recovers any exact non-tax gold edge. Exact recall is zero in all three assigned strata, all eight gold predicates, both visible-density groups, and both endpoint-visibility groups (0/89 visible; 0/12 nonvisible). Thus the strong nonce result does not transfer to this official-training slice. Because selection uses official train gold and the source-text provenance does not establish naturalistic collection, this is negative gold-stratified transfer evidence, not an unbiased, hidden-test, naturalistic, temporal, or deployable result. `confirmatory_eligible=false` and `scientific_deployable=false` remain final; no retuning or calibrator fitting is authorized from this result.

The deterministic zero-call error audit localizes the dominant failure. Guarded-v2 produces 225 raw non-tax candidates: 143 accepted and 82 rejected. All 225 use predicates outside the exact eight-predicate gold inventory, none is an exact TP, and the rejected set contains no hidden TP. Rejections are led by the quantity cap (40), missing endpoint evidence (24), and non-verbatim evidence (10), but these gates do not explain zero recall. Among accepted predictions, 93 have no local gold overlap, 45 share one exact endpoint, two contain both gold endpoints but not as a gold pair, one has the exact endpoint pair with the wrong predicate, one reverses gold endpoints under the wrong predicate, and one has a single surface-normalized endpoint overlap. `part_of` (46) and `has part` (43) dominate the accepted vocabulary. Raw open likewise emits zero exact gold predicates; 51/424 predictions are taxonomy aliases and 2/424 are typing aliases misserialized as non-tax, while 371 remain outside the gold predicate inventory. The post-hoc diagnosis is predicate-inventory/task-definition mismatch rather than guard-induced recall loss. Because it inspects protected gold, it cannot tune this slice; any axiom/predicate-family routing experiment must be frozen on independent samples.

### 4.5.5 Completed independent predicate-routing test

`official_predicate_routing_v1` freezes an independent Task A packet after the official-text transfer diagnosis and before any routed generation. Package ID `78d478f4f1d9d76ff979` identifies 27 samples: 8 assigned to `equivalent class`, 8 to `disjoint with`, 3 to `type`, and 8 to the domain-relation stratum. The packet contains 64 endpoint-visible target gold edges and 69 target gold edges in total. It has zero overlap with prior evidence by sample index, compact context, and exact non-tax gold, and no visible target gold edge is duplicated within the packet. The only externally visible projection is `id/context`.

The comparison locks `open_vocab_guarded_v3_relations` as control and `open_vocab_guarded_v6_predicate_routed` as treatment. Both use the same postprocessor; prompt identity is the only system difference. Neither arm uses a calibrator or an output allowlist. The immutable freeze-time manifest records 54 planned units, 0/54 generation-cache hits, 0/54 postprocess-cache hits, and 54 estimated external calls. All units complete in 57 recorded attempts; the completed caches now replay at 54/54 generation and postprocess hits with zero external calls.

The primary endpoint is four-stratum macro exact target F1. Prespecified secondary endpoints are pooled target micro-F1, all-NonTax micro-F1, official exact Graph Similarity, predicate adherence, endpoint-visible and nonvisible target recall, and a 10,000-replicate stratified paired sample-cluster bootstrap. The mechanism gate requires both a strict routed-v6 macro-F1 improvement over the general v3 control and exact true positives in at least five independent samples. Failure cannot trigger prompt repair or other retuning on this slice.

The frozen prompt contains eight exact predicates, but the independent packet provides empirical support for only four: `equivalent class`, `disjoint with`, `type`, and `database_cross_reference`. Independent `type` capacity is only three samples after exclusions. The other four prompt routes (`has role`, `same as`, `is_conjugate_acid_of`, and `is_conjugate_base_of`) cannot be evaluated by this packet. The experiment therefore estimates routing behavior only over the four supported strata and cannot substantiate claims about the remaining routes.

| System | Macro target F1 | Pooled target P / R / F1 | TP / FP / FN | TP-positive samples | Official GS |
|---|---:|---:|---:|---:|---:|
| general relation-only v3 | 0.0000 | 0 / 0 / 0 | 0 / 0 / 69 | 0 | 0.0000 |
| predicate-routed v6 | **0.0526** | 0.0143 / 0.0290 / 0.0191 | 2 / 138 / 67 | 2 | 0.0016 |

Routed-v6 recovers two exact `type` edges, one in each of two independent samples. Its type-stratum F1 is 0.2105; equivalent-class, disjoint-with, and domain-stratum F1 remain zero. The routed arm emits 140 target-inventory predictions, led by 120 `type` predictions, and therefore achieves perfect predicate-inventory adherence without useful precision. The paired 10,000-replicate stratified sample-cluster bootstrap gives routed-minus-general macro F1 [0, 0.0556]. Strict macro improvement passes, but exact TP occur in only two samples. The preregistered five-sample mechanism gate fails; `scientific_deployable=false` and `confirmatory_eligible=false` remain final. Same-slice repair and n>=100 expansion are not launched.

The deterministic zero-call routing audit separates exact mismatch from evidence failure. Routed-v6 generates 193 raw target candidates, accepts 140, and rejects 53. Accepted predicates are `type` 120, `same as` 12, `has role` 6, and `equivalent class` 2. Among the 138 official exact false positives, 121 carry explicit evidence and 17 carry cross-sentence evidence; 133 bind both endpoints verbatim to context. The quantity cap accounts for 42 rejected candidates, 12 samples reach the accepted limit of eight, and six samples have quantity-limit rejections. Thus routing replaces the earlier predicate-inventory mismatch with broad evidence-bound off-gold generation. This audit does not establish that off-gold predictions are semantically false or true: official gold completeness is unknown and no independent semantic adjudication was performed. It cannot authorize retuning or reopen the mechanism gate.

---

## 4.6 External anchors (cited, not re-run)

| System | Year | Reported headline |
|--------|------|-------------------|
| pro-team (arXiv:2608.27101) | 2026 | Task A SGS **0.7416**; Task B SGS **0.8692**, Typing **0.9200**, Taxonomy **0.8540**; **zero non-tax relations** |
| Alexbek (arXiv:2508.19428) | 2025 | top-ranking on 2025 Tasks A/B/C with RAG + embedding ensemble + cross-attention taxonomy |

Our blind n = 30 official-train numbers are **not** directly comparable to leaderboard scores: they use a training prefix, exact-match official Graph Similarity, a different backbone, and no evaluation server. The n = 10 table is an oracle-inventory diagnostic only. The **2026 competition is closed** — we target **2027 + this paper**.

---

## 4.7 Takeaways (for paper Results section)

1. **Blind n = 30 is complete (120/120 units).** Closed-vocabulary official GS is 0.4404 on Task A and 0.9375 on Task B; open-vocabulary official GS is 0.1603 and 0.6449.
2. **Unconstrained open relations hurt this exact-match slice.** The open path emits 384 Task A and 150 Task B non-tax predictions without a true positive, reducing GS by 0.2801 and 0.2927 relative to closed.
3. **The stratified follow-up tests the intended failure mode.** It includes 63 Task A and 52 non-self Task B non-tax gold edges across balanced density strata, with 60/60 legacy/v1 units and 30/30 new v2 units successful.
4. **Guard only the relation channel.** The v1 hybrid reduces non-tax output by 86.4%/91.3% and improves GS by 0.0315/0.0409; v2-hybrid changes GS by +0.0327/+0.0168 relative to legacy.
5. **Two-span generation helps, but the prespecified gate still misses the TP.** V2 generates the exact Task B edge; the 600-character gap rejects it. A post-hoc 1,604-character diagnostic recovers it and reaches NonTax F1 0.0274, but is not inference-valid tuning.
6. **Independent v2 calibration remains below the deployment gate.** Among 144 deployment-eligible v2 calibration candidates, 4 positives come from 3 samples, below the preregistered minimum of 5 positive samples. The artifact fails closed even though candidate-level class checks pass.
7. **Surface form plus bounded symmetric serialization fails independent validation.** Although it raises exact F1 from 0.1379 to 0.3117 on the selected v5 calibration slice and from 0.1053 to 0.1739 on the incomplete cross-year pressure check, the frozen eight-sample validation set gains 0 TP, adds 8 FP, and lowers F1 from 0.1667 to 0.1364. The rule is rejected for production use.
8. **Structural projection finds one reproducible but under-supported rule.** Whole/substance derivation is 2/2 under leave-one-sample-out selection, while broader and nested rules fail precision or replication. The projection channel remains below its own five-sample gate.
9. **Cross-year axiom data do not resolve calibration support.** V4 has 0/34 exact accepted candidates; v5 reaches 1/8 from one sample, with a degenerate calibrator. Endpoint-linked gold is not complete document-level supervision, so these runs remain negative diagnostics rather than added calibration support.
10. **Weak-label calibration generalizes only as base-rate shrinkage.** Sample-grouped five-fold OOF metrics remain low (Brier 0.1134, ECE 0.0045), but every fold collapses to one probability block and therefore learns no confidence ranking. Same-origin weak labels still cannot open the gate.
11. **Pooled v5 calibration passes only the numeric gate, then fails confirmatory promotion.** The untouched 24-sample endpoint-visible slice yields 4 base TP from 4 samples. Frozen calibration improves Brier/ECE but removes one TP and leaves only 3 positive samples, below the preregistered minimum of 5. This is valid confirmatory negative evidence, so deployment remains false without retuning.
12. **Predicate routing produces a narrow recovery but fails its mechanism gate.** On the independent 27-sample packet, routed-v6 raises macro target F1 from 0 to 0.0526 and recovers two `type` edges, but pooled precision is 0.0143 with 138 FP. Two TP-positive samples do not satisfy the required five, so no same-slice repair or n>=100 expansion follows.
13. **Semantic matching does not rescue v5.** At the frozen 0.90 threshold, the official embedding alignment changes 0/24 samples for both base and calibrated outputs; relation-only per-sample Graph Similarity remains 0.0935 and 0.0657.
14. **Uncertainty is still wide and heterogeneous by predicate.** Base F1 has a sample-cluster 95% interval of [0.0238, 0.2712], while equivalent-class precision is 0.1111 and disjoint-with precision is 0.5000. The n≈122 estimate is for pooled precision estimation only.
15. **The legacy n = 10 direction does not survive blind evaluation.** Its open-path gains came from an oracle-inventory component setting and remain diagnostic only.
16. **Synthetic non-tax gains remain NON-official.** They cannot be used as evidence of official relation recovery.
17. **Official facts are highly probeable, but the surface effect is task-dependent.** Closed-book AUROC is 0.9780 [0.9484, 0.9982]. Endpoint aliasing lowers Task B GS but improves Task A point estimates, so neither verified pretraining membership nor a global contamination-induced inflation claim is supported.
18. **The frozen nonce diagnostic recovers controlled relations but exposes a wording gap.** Guarded-v2 raises exact NonTax F1 from 0.2612 to 0.8571 with 0/60 hard-negative false positives for both systems. Its canonical/paraphrased F1 split is 1.0000/0.7119, so this supports extraction of new synthetic facts under familiar predicates, not unseen-predicate discovery or naturalistic performance.
19. **The nonce gain does not transfer to the frozen official-text slice.** Both raw open and guarded-v2 obtain 0 exact TP over 101 non-self non-tax gold relations. Guarded-v2 cuts false positives from 424 to 143 and raises GS from 0.1946 to 0.2593, but exact recall remains zero across every prespecified stratum, predicate, density, and visibility diagnostic.
20. **The guard did not remove a latent correct relation.** A zero-call post-hoc audit finds 0 exact TP among all 225 raw guarded candidates and 0 among the 82 rejected candidates. Every raw guarded predicate falls outside the gold inventory; `part_of`/`has part` account for 89/143 accepted predictions. Predicate-family mismatch, not confidence/evidence filtering, is the dominant failure mechanism.

---

## 4.8 Ablation & scale checklist (to complete before camera-ready)

- [x] Task A/B real LLM **blind n = 30** tables — 120/120 units complete; payload audit and result JSON retained
- [x] Non-tax-density-stratified Task A/B evaluation — 60/60 legacy/v1 units, 30/30 new v2 units, and local hybrid derivations complete
- [x] Relation confidence, verbatim evidence, generic/self-loop vetoes, and max-eight quantity constraint
- [x] Guarded-v2 canonicalization, two-span implicit evidence parser, and fail-closed calibration implementation
- [x] Offline v1-cache replay — 40 candidates, 0 canonicalized TP, zero-positive calibration artifact marked non-deployable
- [x] Guarded-v2 stratified API run — 30/30 new calls successful; both Task A/B hybrids rebuilt locally
- [x] Predicate/entity canonicalization and implicit-evidence gap sensitivity on v2 generations
- [x] Original fail-closed audit — calibration 0/34 positives, held-out 1/77; not deployable
- [x] Independent relation calibration — 39/39 successful unique v2 units, 144 deployment-eligible candidates, 4 positives from 3 samples
- [x] Sample-level deployment gate — requires at least 5 independent positive samples and currently fails closed
- [x] Exploratory calibration-supported predicate policy replay — zero additional external calls, retained as non-deployable provenance
- [x] Relation-only v3 pilot — 3/3 calls, 11 accepted relation candidates, 0 exact TP
- [x] Calibration-only structural projection audit — one 2/2 replicated rule, but only 2 positive projection samples; fails closed
- [x] Cross-year 2025 axiom diagnostics — v4 0/34 exact; v5 1/8 exact from one sample; endpoint-linked supervision judged unsuitable for calibrator pooling
- [x] Disjoint 2026 axiom-only calibration extension — 8/8 successful after 8 token-cap failures; 4/23 exact candidates from 3 samples, so the exact deployment gate remains closed
- [x] Symmetric-predicate serialization audit — 2026 order-invariant diagnostic recovers 8 reversed endpoints, but the zero-call 2025 pressure check shows bidirectional output reduces exact F1 to 0.0741
- [x] Surface morphology + quantity-aware serialization audit — versioned final-token singularization and max-eight preservation recover 8/1 additional exact edges on 2026/cross-year axiom caches with zero losses; cross-year support remains only 2 samples
- [x] Independent axiom validation manifest — 8 disjoint, lexically explicit Task A samples frozen with no source text
- [x] Execute and score the frozen 8-sample axiom validation set — 8/8 successful calls; base exact F1 0.1667, surface-plus-direction F1 0.1364, so the rule is rejected
- [x] Pooled v5 grouped-CV diagnostic — 31/31 OOF candidates, 7 positives from 6 samples, Brier 0.1836 and ECE 0.0830; numeric gate passes but scientific deployment is forced closed
- [x] Confirmatory launch gate — remains closed; no confirmatory API calls were made
- [x] Text-only independent annotation packet — 30 unused 2026 train samples (15 A/15 B), zero overlap with prior experimental indices, selected without official gold or model output
- [x] Same-system annotation-packet generation — `open_vocab_guarded_v2` 30/30 cached; 234 raw and 128 accepted candidates; official-gold metrics discarded
- [x] Exploratory dual-model weak annotation — 30/30 + model adjudication, 169 frozen weak relations; macro Jaccard 0.3646 and `confirmatory_eligible=false`
- [x] Weak-label calibration replay — 30/30 cache hits, zero external calls, 26 weak-label matches from 14 samples; numeric support passes but deployment remains forcibly closed
- [x] Sample-grouped weak-label calibration CV — 5 deterministic folds, 200/200 OOF candidates, zero sample overlap, Brier 0.1134 and ECE 0.0045; all folds reduce to base-rate-only calibration
- [x] Weak-label OOF uncertainty and reliability figure — 10,000 sample-cluster bootstrap replicates; SVG/PDF/300 dpi PNG retained with conditional 95% intervals
- [ ] Two independent human annotations plus third-person adjudication — deferred because the current paper makes no human-calibration claim; all 30 rows remain pending
- [ ] Relation-level human audit of the weak labels — optional future enhancement; the current 27-sample worklist remains a triage index only
- [ ] Fit/freeze a human-label calibrator only if the paper scope is later expanded to a human-calibration claim
- [ ] Task A/B real LLM n >= 100 — not launched because the independent predicate-routing mechanism gate failed
- [x] Official embedding `semantic_match` secondary analysis — frozen threshold 0.90 and model revisions; 0/24 samples change for either v5 system
- [ ] Expand human/official-gold same-system independent calibration beyond five positive samples; the completed weak-label reliability analysis does not satisfy this gate
- [x] Freeze a new untouched, fully supervised v5 confirmatory set — 24 Task A samples, 12/12 predicate-balanced, zero overlap, frozen before 0/24-cache generation
- [x] Execute v5 confirmatory generation — 24/24 successful external calls, 24 low-level attempts, no failed final unit
- [x] Apply the frozen pooled calibrator locally — zero external calls; Brier/ECE improve but exact F1 falls from 0.1159 to 0.0882
- [x] Evaluate the preregistered confirmatory gate — fails closed with 3 calibrated positive samples versus the required 5; no retuning authorized
- [x] Confirmatory error/uncertainty audit — per-predicate exact metrics plus 10,000 sample-cluster bootstrap replicates; base/calibrated F1 intervals [0.0238, 0.2712]/[0, 0.2222]
- [x] Estimation-only scale planning — n≈122 total (98 additional) for pooled precision half-width near 0.10; no retuning authority and no immediate launch
- [x] Official-train contamination pilot — 30/30 closed-book units and 60/60 endpoint-alias units; 10,000 paired sample-cluster bootstrap replicates; probeability and task-dependent surface sensitivity only, not verified membership
- [x] Freeze relation-rich low-exposure holdout — 30 nonce-entity samples, 120 balanced non-tax gold relations, 60 negated controls, zero official-2026 exact entity/triple overlap
- [x] Execute the locked nonce holdout — 60/60 `open_vocab`/`open_vocab_guarded_v2` units in 67 attempts; exact NonTax F1 0.2612/0.8571, both with 0/60 hard-negative FP
- [x] Freeze official-text relation transfer — 30 unused Task A samples, 10/10/10 predicate strata, 89 visible non-tax gold, zero prior sample/context/exact-gold overlap
- [x] Execute official-text relation transfer — 60/60 locked raw-open/guarded-v2 units in 81 recorded attempts; exact NonTax F1 0/0, guarded-v2 reduces non-self FP 424→143; final dual-layer cache replay uses zero external calls
- [x] Official-text transfer error audit — zero-call post-hoc classification of 424/143 predictions and 225 raw guarded candidates; 0 raw/rejected TP and 0 guarded candidates using an exact gold predicate
- [x] Freeze independent predicate-routing test: package `78d478f4f1d9d76ff979`, 27 untouched Task A samples in 8/8/3/8 strata, 64 visible target edges, and zero prior sample/context/exact-gold overlap
- [x] Execute the frozen predicate-routing comparison — 54/54 locked `open_vocab_guarded_v3_relations`/`open_vocab_guarded_v6_predicate_routed` units completed in 57 recorded attempts; completed dual-layer cache replay uses zero external calls
- [x] Score the frozen routing gate — macro target F1 improves 0 -> 0.0526, but routed-v6 has exact TP in only 2 samples versus the required 5; gate fails and no same-slice retuning is authorized
- [x] Record predicate-routing API stability — v3/v6 use 28/29 attempts for 27 units each, with 1/2 retried samples and no failed final unit; median latency 33.639/42.690 s, maximum 110.180/135.899 s
- [x] Predicate-routing zero-call error audit — 120/140 outputs are `type`; 133/138 exact FP bind both endpoints verbatim, so evidence binding alone does not solve sparse-gold alignment

---

## 4.9 How to reproduce these numbers

```powershell
# config (Key only user fills)
$env:ONTOBRIDG_API_KEY = 'ak_<rotated-key>'

# smoke
& D:\NLP\src\run_dots_smoke.ps1

# single chunk (with offset)
& D:\NLP\src\run_dots_baseline.ps1 -Task a -Offset 0 -Limit 5 -Systems 'open_vocab' -VerboseRun

# inspect the external payload scope without sending source text
& D:\NLP\src\run_blind_eval.ps1 -Task ab -Limit 30 -Workers 2 -AuditOnly

# scale-up after explicit authorization (sample cache + atomic checkpoints)
& D:\NLP\src\run_blind_eval.ps1 -Task ab -Limit 30 -Workers 2 -VerboseRun

# inspect the stratified sample/payload manifest without an API call
& D:\NLP\src\run_stratified_open_eval.ps1 -Task ab -SamplesPerStratum 5 -AuditOnly

# combined guarded density-stratified comparison; current caches make this zero-call
& D:\NLP\src\run_stratified_open_eval.ps1 -Task ab -SamplesPerStratum 5 -Workers 2 `
  -Systems 'open_vocab,open_vocab_guarded,open_vocab_guarded_v2' -AuditOnly

# fit/audit the independent v2 calibration set locally (never calls the API)
python -B D:\NLP\src\run_relation_v2_offline.py `
  --result D:\NLP\experiments\results\relation_calibration_a_n27_seed20260923.json `
  --additional-result D:\NLP\experiments\results\relation_calibration_ab_n12_seed20260923.json `
  --source-system open_vocab_guarded_v2 `
  --all-samples-calibration `
  --output D:\NLP\experiments\results\relation_guarded_v2_independent_calibrator_a27_b12_seed20260923.json

# recompute the disjoint 2026 axiom-only exact/symmetric audit locally
python -B D:\NLP\src\run_relation_v2_offline.py `
  --result D:\NLP\experiments\results\relation_calibration_open_vocab_guarded_v5_axioms_only_a_indices1419-1554-1796-1824-1935-2676-3130-3516_seed20260924.json `
  --source-system open_vocab_guarded_v5_axioms_only `
  --all-samples-calibration --seed 20260924 `
  --output D:\NLP\experiments\results\relation_guarded_v5_axioms_only_explicit_calibrator_a8_seed20260924.json

# audit the current sample-gated calibrator; it remains fail-closed below five positive samples
& D:\NLP\src\run_stratified_open_eval.ps1 -Task ab -SamplesPerStratum 5 -Workers 2 `
  -Systems 'open_vocab,open_vocab_guarded_v2' `
  -CalibrationFile D:\NLP\experiments\results\relation_guarded_v2_independent_calibrator_a27_b12_seed20260923.json `
  -AuditOnly

# audit the cached cross-year v5 axiom diagnostic; current cache coverage makes this zero-call
& D:\NLP\src\run_cross_year_relation_calibration.ps1 -Workers 2 -AuditOnly

# after the audit reports external_calls=0, replay caches and rebuild its offline report
& D:\NLP\src\run_cross_year_relation_calibration.ps1 -Workers 2

# recompute weak-label sample-cluster intervals and the reliability figure locally
python -B D:\NLP\src\analyze_grouped_relation_calibration.py `
  --bootstrap-replicates 10000 --bootstrap-seed 20260924 --bins 10

# verify (without overwriting) the frozen v5 confirmatory manifest; caches now make this zero-call
python -B D:\NLP\src\run_v5_confirmatory.py --audit-only --model dots3-note-prev

# rebuild the locked confirmatory score from cached base/calibrated results; zero API calls
python -B D:\NLP\src\run_v5_confirmatory.py --score-only --model dots3-note-prev

# rebuild the deterministic post-confirmatory error/uncertainty audit; zero API calls
python -B D:\NLP\src\analyze_v5_confirmatory.py

# verify the frozen semantic protocol, then run the pinned official secondary metric
python -B D:\NLP\src\run_v5_semantic_secondary.py --freeze-only
$env:HF_HOME = 'D:\AI\hf-cache'
$env:HF_ENDPOINT = 'https://huggingface.co'
python -B D:\NLP\src\run_v5_semantic_secondary.py

# audit/replay the frozen closed-book probe, then rebuild its zero-call analysis
python -B D:\NLP\src\run_official_contamination_probe.py --audit-only --model dots3-note-prev
python -B D:\NLP\src\run_official_contamination_probe.py --model dots3-note-prev
python -B D:\NLP\src\analyze_official_contamination_probe.py

# replay 60 endpoint-alias units from cache and rebuild paired intervals; zero external calls
python -B D:\NLP\src\run_official_contamination_surface_stress.py --model dots3-note-prev
python -B D:\NLP\src\analyze_official_contamination_surface_stress.py

# verify the frozen relation-rich nonce holdout and completed cache; zero API calls
python -B D:\NLP\src\build_nonce_relation_holdout.py --audit-only
python -B D:\NLP\src\run_nonce_relation_holdout.py --audit-only --model dots3-note-prev

# rebuild the completed result from 60/60 generation-cache hits; zero API calls
python -B D:\NLP\src\run_nonce_relation_holdout.py --model dots3-note-prev --workers 2

# verify the frozen official-text transfer packet and completed cache; zero API calls
python -B D:\NLP\src\build_official_relation_transfer_holdout.py --audit-only
python -B D:\NLP\src\run_official_relation_transfer.py --model dots3-note-prev

# rebuild the completed transfer result from 60/60 cache hits; zero API calls
python -B D:\NLP\src\run_official_relation_transfer.py --execute `
  --model dots3-note-prev --workers 2 --max-completion-tokens 16384

# rebuild the protected-gold post-hoc error audit; zero API calls and no tuning
python -B D:\NLP\src\analyze_official_relation_transfer.py

# verify the frozen independent predicate-routing packet; zero API calls
python -B D:\NLP\src\build_official_predicate_routing_holdout.py --audit-only

# replay/audit the completed 54-unit v3-control/v6-treatment result; zero API calls
python -B D:\NLP\src\run_official_predicate_routing.py --model dots3-note-prev

# rebuild the post-hoc routing error audit; zero API calls and no retuning authority
python -B D:\NLP\src\analyze_official_predicate_routing.py
```

**Files:**

- results JSON under `D:\NLP\experiments\results\`
- checkpoints under `D:\NLP\experiments\results\checkpoints\`
- reliability figure under `D:\NLP\papers\figures\weaklabel_relation_calibration_reliability.{svg,pdf,png}`
- frozen v5 confirmatory manifest/result/score: `v5_axiom_confirmatory_{manifest,base,calibrated,score}_*.json`
- v5 error/uncertainty audit and semantic secondary result: `v5_axiom_confirmatory_{error_audit,semantic_secondary}_*.json`
- frozen contamination manifests/results/analyses: `official_contamination_{probe,surface}_*.json`
- frozen nonce package: `experiments/holdouts/nonce_relation_v1/`; evaluation manifest/result: `nonce_relation_holdout_{manifest,openai}_n30_seed20260926.json`
- frozen official-text transfer package: `experiments/holdouts/official_relation_transfer_v1/`; manifest/result/error audit: `official_relation_transfer_{manifest,openai,error_audit}_a_n30_seed20260927.json`
- frozen independent predicate-routing package: `experiments/holdouts/official_predicate_routing_v1/`; manifest/result/error audit: `experiments/results/official_predicate_routing_{manifest,openai,error_audit}_a_n27_seed20260927.json`
- official metric: `D:\NLP\src\ontobridg\eval\official_graph_similarity.py`
