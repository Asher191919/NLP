# Official Predicate Routing Holdout v1

Status: **completed; mechanism gate failed closed**. Package ID: `78d478f4f1d9d76ff979`.

This packet independently tests the protected-gold post-hoc hypothesis that the
official transfer failure was caused by predicate-family mismatch. It compares
the existing relation-only general prompt (`open_vocab_guarded_v3_relations`)
with the frozen inventory router (`open_vocab_guarded_v6_predicate_routed`). It
is a gold-stratified official-training diagnostic, not a hidden test, unbiased
sample, naturalistic benchmark, temporal holdout, or deployment experiment.

## Frozen Design

| Property | Frozen value |
|---|---:|
| Samples | 27 |
| Assigned strata | 8 equivalent / 8 disjoint / 3 type / 8 domain |
| Endpoint-visible target gold | 64 |
| All target gold | 69 |
| Planned systems | general relation-only v3 / routed v6 |
| Planned evaluation units | 54 |
| Generation cache hits at freeze | 0 |
| Estimated external calls | 54 |
| Completion-token cap | 16,384 |

The route inventory is frozen to `equivalent class`, `disjoint with`, `type`,
`database_cross_reference`, `has role`, `same as`,
`is_conjugate_acid_of`, and `is_conjugate_base_of`. After all prior selections
and exact-gold overlaps are excluded, the independent pool supports only four
of them: 30 visible `disjoint with`, 23 `equivalent class`, 3 `type`, and 8
`database_cross_reference` edges. The other four prompt routes are therefore
not empirically tested by this packet. Only three independent type samples
remain, so allocation is capacity-aware rather than balanced.

The primary metric is four-stratum macro exact target F1, with routed-v6 minus
general-v3 as the frozen contrast. Pooled target micro-F1, all-NonTax micro-F1,
official exact Graph Similarity, predicate adherence, visible/nonvisible target
recall, and a 10,000-replicate stratified paired sample-cluster bootstrap are
secondary diagnostics. Mechanism support requires a strict macro-F1 gain and
exact TP from at least five routed samples. Failure cannot trigger same-slice
repair or retuning.

## Independence And Payload Boundary

The exclusion ledger includes the blind prefix, all executed calibration and
validation slices, annotation packets, audit-only selections, the v5
confirmatory set, and the completed `official_relation_transfer_v1` packet.
The selected 27 samples have zero prior sample-index overlap, zero compact-text
overlap, zero exact non-tax-gold overlap, no duplicate IDs, and no repeated
visible target gold.

- `dataset.json`: local source projection and protected official gold.
- `model_inputs.jsonl`: the only legal external payload fields, `id` and
  `context`.
- `manifest.json`: hashes, aggregate counts, exclusion provenance, and overlap
  audit; no source text or gold triples.
- Evaluation manifest:
  `experiments/results/official_predicate_routing_manifest_a_n27_seed20260927.json`.

No calibrator or predicate allowlist is applied. The two systems share the same
evidence, confidence, distance, self-loop, generic-predicate, and max-eight
postprocessor. Only the frozen prompt identity differs. Prompt/threshold tuning,
calibration, same-slice repair, and deployment changes are prohibited.
`confirmatory_eligible=false` and `scientific_deployable=false`.

## Reproduce Without API Calls

```powershell
py -3 D:\NLP\src\build_official_predicate_routing_holdout.py --audit-only
py -3 D:\NLP\src\run_official_predicate_routing.py --model dots3-note-prev
py -3 D:\NLP\src\analyze_official_predicate_routing.py
```

All 54 frozen units completed in 57 recorded attempts. The completed runner now
reports 54/54 generation-cache hits, 54/54 postprocess-cache hits, and zero
external calls on replay.

## Completed Result

| Metric | general v3 | routed v6 |
|---|---:|---:|
| Four-stratum macro exact target F1 | 0.0000 | 0.0526 |
| Pooled target precision / recall / F1 | 0 / 0 / 0 | 0.0143 / 0.0290 / 0.0191 |
| Exact TP / FP / FN | 0 / 0 / 69 | 2 / 138 / 67 |
| TP-positive independent samples | 0 | 2 |
| Official exact Graph Similarity | 0.0000 | 0.0016 |

Both routed true positives are `type` relations from two of the three type
samples. Equivalent-class, disjoint-with, and domain strata remain at zero TP.
The routed arm satisfies the strict macro-F1 improvement condition but misses
the preregistered minimum of five TP-positive samples. The mechanism gate is
therefore false; same-slice repair, n>=100 expansion, calibration, and
deployment remain unauthorized. The stratified paired bootstrap interval for
routed-minus-general macro F1 is [0, 0.0556].

The strictly offline post-hoc audit finds 193 raw target candidates, 140
accepted target predictions, and 53 rejected target candidates. The accepted
inventory is dominated by `type` (120/140, 85.7%), followed by `same as` (12),
`has role` (6), and `equivalent class` (2). Of the 138 official exact false
positives, 121 use explicit evidence and 17 use cross-sentence evidence; 133
have both endpoints bound verbatim to the context. The quantity limit rejects
42 target candidates, 12 samples reach the eight-relation accepted cap, and six
samples contain a quantity-limit rejection. These are exact-score diagnostics,
not semantic adjudication: official gold completeness is not established, so
off-gold predictions cannot be called correct or incorrect without additional
annotation. The audit cannot authorize same-slice tuning or reopen the gate.

Result artifacts:
`experiments/results/official_predicate_routing_openai_a_n27_seed20260927.json`
and
`experiments/results/official_predicate_routing_error_audit_a_n27_seed20260927.json`.
