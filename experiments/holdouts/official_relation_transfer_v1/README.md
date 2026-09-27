# Official Relation Transfer Holdout v1

Status: **frozen and completed (60/60 units)**. Package ID: `76e98ba860f1b72b4df3`.

This packet tests whether the positive guarded-v2 result on the synthetic nonce
holdout transfers to unused official Task A text. It is a gold-stratified
official-training diagnostic, not a hidden test set, an unbiased sample, a
verified temporal holdout, or a confirmed naturalistic benchmark.

## Frozen Design

| Property | Frozen value |
|---|---:|
| Samples | 30 |
| Assigned strata | 10 equivalent / 10 disjoint / 10 other open predicate |
| Endpoint-visible non-tax gold | 89 |
| All non-self non-tax gold | 101 |
| Visible relations per sample | 2-8 |
| Medium / high density samples | 27 / 3 |
| Planned systems | `open_vocab`, `open_vocab_guarded_v2` |
| Planned evaluation units | 60 |
| Cache hits at freeze | 0 |
| Estimated external calls | 60 |

Visible predicate support covers `equivalent class` (25), `disjoint with` (33),
`type` (13), `is_conjugate_acid_of` (5), `is_conjugate_base_of` (5),
`database_cross_reference` (5), `has role` (2), and `same as` (1).

The exclusion ledger covers the first 40 Task A/B samples, the completed density
evaluation, independent calibration slices, every retained audit-only selection,
both annotation packets, the v5 calibration/validation slices, and the 24-sample v5 confirmatory set. The frozen
selection has zero reused Task A indices, zero compact-context overlap, zero exact
non-tax gold overlap with prior evidence, zero exact gold overlap with the nonce
holdout, and no duplicated visible gold across selected samples.

## Files And Payload Boundary

- `dataset.json`: local source projection and protected official gold.
- `model_inputs.jsonl`: the only legal external payload fields, `id` and
  `context`.
- `manifest.json`: hashes, aggregate counts, exclusion provenance, and overlap
  audit; it contains neither context nor gold triples.
- Evaluation manifest:
  `experiments/results/official_relation_transfer_manifest_a_n30_seed20260927.json`.

Exact NonTax micro-F1 is primary and official exact Graph Similarity is
secondary. Assigned-stratum, gold-predicate, visible-density, and
endpoint-visible versus nonvisible recall diagnostics were frozen before any
generation. The systems use the same uncalibrated prompt, thresholds, quantity
limit, and postprocessing identity as the completed nonce comparison. Prompt or
threshold tuning, calibrator fitting/application, and deployment changes are
prohibited. `confirmatory_eligible=false` and `scientific_deployable=false`.

## Completed Result

All 60 units completed in 81 recorded attempts. The initial 8,192-token run
exposed completion-length failures on guarded reasoning outputs; technical
recovery used a 16,384-token completion cap while preserving every frozen
sample, prompt hash, threshold, calibrator setting, and postprocessor. Failed
attempts remain in the per-unit cache provenance. A final replay reports 60/60
generation-cache hits, 60/60 postprocess-cache hits, and zero external calls.

| System | Exact NonTax P / R / F1 | Non-self pred / gold | Official exact GS |
|---|---:|---:|---:|
| `open_vocab` | 0.0000 / 0.0000 / 0.0000 | 424 / 101 | 0.1946 |
| `open_vocab_guarded_v2` | 0.0000 / 0.0000 / 0.0000 | 143 / 101 | 0.2593 |

Neither system recovers an exact non-tax gold edge: recall is zero for all
three assigned strata, all eight gold predicates, and both endpoint-visibility
groups. Guarded-v2 reduces non-tax false positives by 281 (66.3%) and raises
overall Graph Similarity by 0.0646, but it does not transfer the nonce holdout's
relation-recovery gain to this official-training slice. This is negative,
gold-stratified transfer evidence, not an unbiased, hidden-test, naturalistic,
temporal, or deployment result. The completed result is
`experiments/results/official_relation_transfer_openai_a_n30_seed20260927.json`.

A zero-call post-hoc error audit shows that the guard did not discard a hidden
true positive. All 225 raw guarded non-tax candidates use predicates outside the
eight-predicate gold inventory; none is an exact gold triple, and none of the 82
rejected candidates is exact. Among 143 accepted predictions, 93 have no local
gold endpoint/predicate overlap and 45 share exactly one endpoint. Raw open also
emits no exact gold predicate; 53/424 apparent non-tax predictions are actually
typing/taxonomy aliases serialized outside the official core spelling. The
dominant failure is therefore predicate-inventory/task-definition mismatch, not
the evidence or confidence gate. This audit contains protected gold and is
post-hoc: it cannot tune this slice. Any predicate-family or axiom-routing
follow-up requires a newly frozen independent holdout. Audit artifact:
`experiments/results/official_relation_transfer_error_audit_a_n30_seed20260927.json`.

## Reproduce Without API Calls

```powershell
py -3 D:\NLP\src\build_official_relation_transfer_holdout.py --audit-only
py -3 D:\NLP\src\run_official_relation_transfer.py --model dots3-note-prev
py -3 D:\NLP\src\analyze_official_relation_transfer.py
```

The runner is audit-only by default. It now reports 60 generation-cache and 60
postprocess-cache hits, with zero external calls. The immutable evaluation
manifest intentionally retains its pre-generation `frozen_not_executed` state,
0 cache hits, and 60-call estimate; completion is recorded only in the separate
result artifact.
