# Nonce Relation Holdout v1

Status: **frozen and executed**. Package ID: `aa1f0e2e61177e6144e2`.

This is a synthetic low-exposure diagnostic for text-grounded non-taxonomic
relation extraction. It is not an official LLMs4OL split, a naturalistic corpus,
or a verified temporal holdout.

## Design

| Property | Frozen value |
|---|---:|
| Samples | 30 |
| Non-tax gold relations | 120 (4/sample) |
| Hard-negative controls | 60 (2/sample) |
| Unique nonce entities | 300 |
| Predicate families | 6 (20 positives each) |
| Canonical / paraphrased samples | 15 / 15 |
| Explicit / cross-sentence positives | 60 / 60 |
| Exact entity strings overlapping official 2026 train | 0 |
| Exact gold triples overlapping official 2026 train | 0 |

The six relation labels are `part_of`, `has part`, `develops_from`,
`derives from`, `located in`, and `regulates`. These familiar labels test
extraction and canonicalization of new facts, not discovery of an unseen
predicate ontology. Canonical samples expose the target surface; paraphrased
samples require normalization. Every sample also contains explicitly negated
co-occurring entity pairs to measure false-positive extraction.

## Files

- `dataset.json`: local protected gold, evidence annotations, and controls.
- `model_inputs.jsonl`: the only legal external payload fields, `id` and
  `context`.
- `manifest.json`: hashes, aggregate counts, overlap audit, and locked protocol;
  it contains neither source text nor gold triples.

The evaluation manifest is
`experiments/results/nonce_relation_holdout_manifest_n30_seed20260926.json`.
It freezes `open_vocab` and `open_vocab_guarded_v2` before generation: 60 units,
exact NonTax micro-F1 primary, official exact Graph Similarity secondary, and
hard-negative false-positive rate diagnostic. No calibrator may be fit or
applied. Exact precision/recall/F1 is also frozen per target predicate, while
explicit versus cross-sentence evidence is compared by exact gold recall. This
packet cannot authorize prompt/threshold tuning or deployment.

The immutable evaluation manifest intentionally retains its pre-generation
`status=frozen_not_executed`, zero cache hits, and 60 estimated calls. Do not
rewrite those preregistration fields after observing outputs. Completion state,
the cumulative attempt ledger, and metrics live in
`experiments/results/nonce_relation_holdout_openai_n30_seed20260926.json`.

## Frozen Results

All 60 preregistered units completed with 67 recorded low-level attempts (32 for
`open_vocab`, 35 for `open_vocab_guarded_v2`). The completed cache now replays
with 60/60 hits and zero external calls.

| System / stratum | Exact P | Exact R | Exact NonTax F1 | Official exact GS | Hard-negative FPR |
|---|---:|---:|---:|---:|---:|
| `open_vocab`, all | 0.2222 | 0.3167 | 0.2612 | 0.1805 | 0/60 = 0.0000 |
| `open_vocab`, canonical | 0.4605 | 0.5833 | 0.5147 | 0.3312 | 0/30 = 0.0000 |
| `open_vocab`, paraphrased | 0.0316 | 0.0500 | 0.0387 | 0.0299 | 0/30 = 0.0000 |
| `open_vocab_guarded_v2`, all | 0.8644 | 0.8500 | 0.8571 | 0.5690 | 0/60 = 0.0000 |
| `open_vocab_guarded_v2`, canonical | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0/30 = 0.0000 |
| `open_vocab_guarded_v2`, paraphrased | 0.7241 | 0.7000 | 0.7119 | 0.4714 | 0/30 = 0.0000 |

Overall exact recall by evidence mode is 0.3167/0.3167
(explicit/cross-sentence) for `open_vocab` and 0.7500/0.9500 for
`open_vocab_guarded_v2`.

| Predicate | `open_vocab` P/R/F1 | Guarded-v2 P/R/F1 |
|---|---:|---:|
| `part_of` | 0.7647 / 0.6500 / 0.7027 | 0.7727 / 0.8500 / 0.8095 |
| `has part` | 1.0000 / 0.1000 / 0.1818 | 0.7500 / 0.7500 / 0.7500 |
| `develops_from` | 1.0000 / 0.5000 / 0.6667 | 1.0000 / 1.0000 / 1.0000 |
| `derives from` | 1.0000 / 0.0500 / 0.0952 | 1.0000 / 1.0000 / 1.0000 |
| `located in` | 1.0000 / 0.1000 / 0.1818 | 1.0000 / 0.7500 / 0.8571 |
| `regulates` | 1.0000 / 0.5000 / 0.6667 | 1.0000 / 0.7500 / 0.8571 |

This relation-only packet has no typing or taxonomy gold. The official Graph
Similarity taxonomy component is therefore zero, so even a perfect relation
graph scores 2/3; exact NonTax micro-F1 is the primary endpoint. The large
guarded-v2 gain shows that the frozen relation guards work on these controlled
templates, while its canonical-to-paraphrased F1 drop (1.0000 to 0.7119) warns
against extrapolation to natural text. The result remains
`confirmatory_eligible=false` and `scientific_deployable=false`.

## Reproduce

```powershell
# Recompute balance and official-overlap checks without writing
py -3 D:\NLP\src\build_nonce_relation_holdout.py --audit-only

# Verify the frozen evaluation manifest; no API calls
py -3 D:\NLP\src\run_nonce_relation_holdout.py --audit-only --model dots3-note-prev

# Replay the completed evaluation from its generation cache; zero external calls
py -3 D:\NLP\src\run_nonce_relation_holdout.py --model dots3-note-prev
```

Do not remove or bypass the completed cache when reproducing the result. If any
unit is missing, stop at `--audit-only`; a new external call requires separate
authorization.

The synthetic construction sharply reduces factual memorization as an
explanation because entity identifiers and facts were generated for this
packet. It does not reproduce natural text difficulty, establish a model
training cutoff, or replace independent human/official evaluation.
