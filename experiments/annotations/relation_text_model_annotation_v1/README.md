# Dual-model weak labelling packet

Protocol: `relation_text_model_annotation_v1`
Packet ID: `5b53d848edca6abb58f4` (same samples as the human packet)
Generated: 2026-09-24

This directory holds **model-produced weak labels**. It exists so that relation
calibration can be developed without waiting on human annotators. It is *not* a
human annotation packet, and nothing here may be reported as human-annotated.

The sibling directory `experiments/annotations/relation_text_annotation_v1/` is the
human packet and is untouched. It still has `complete=0/30` on both annotators.

## Why this exists

The earlier official-gold calibration path stalled at three samples with positive
relation support, and the preregistered deployment gate needs five; the separate
human packet remains entirely pending. Dual-model labelling was run to see
whether the engineering calibration could be unblocked for exploratory work while
the confirmatory gate stays shut. It can, and it did — see "Calibration outcome".

## Files

| File | Role |
|------|------|
| `manifest.json` | Sample manifest; `protocol=relation_text_model_annotation_v1`, `confirmatory_eligible=false` |
| `annotation_guidelines.md` | The 11 per-sample rules plus the annotator and adjudicator prompts |
| `model_annotator_a.jsonl` | Annotator A's completed labels (input `inputs` preserved byte-identical) |
| `model_annotator_b.jsonl` | Annotator B's completed labels |
| `model_adjudication.jsonl` | Comparison plus adjudicator C's per-disagreement rulings |
| `frozen_model_relations.json` | Frozen weak labels; the artifact the calibrator consumes |
| `model_agreement.json` | Inter-annotator agreement metrics |
| `human_audit_worklist.json` | Triage index for a future relation-level human audit (27 of 30 samples); not directly freezable |
| `_template_backup/` | Pristine pre-annotation templates, kept for audit |

## Provenance

| Role | Declared ID | Declared model | Routing |
|------|-------------|----------------|---------|
| Annotator A | `model_a_wb-reasoning` | `workbuddy-subagent:reasoning` | reasoning |
| Annotator B | `model_b_wb-default` | `workbuddy-subagent:default` | default |
| Adjudicator C | `model_c_wb-reasoning` | `workbuddy-subagent:reasoning` | reasoning |

- `prompt_version`: `relation_model_annotation_v1`
- Temperature 0. `completed_at`: A and B `2026-09-24T03:38:18+08:00`; C stamped itself.
- Prompt hashes: annotator `e93e2d38230c5e209de277d83b7ed07dbf841e10838c793f621699f02770ef00`,
  adjudicator `33a7a1b8aa33ce1b5b2f39ac76d9a837922b157c1131515b633387e1ffc84e77`,
  guidelines `029ff8f4539eb56129767668f843e1bcf12a06880c5dc960ee512e9c58986fa6`.

**Independence is nominal, not real.** All three roles are same-origin WorkBuddy
subagents. `model_id` records the routing class because the exact upstream model name
is not exposed to the annotator. Per the guidelines' own warning, three same-origin
agents are repeated sampling, not independent annotation. Treat the agreement numbers
below as an optimistic upper bound on independence.

Isolation was enforced by instruction: A read only `model_annotator_a.jsonl` and the
guidelines; B read only `model_annotator_b.jsonl` and the guidelines; C ran after both
finished. No annotator was given official gold, `dots` output, confidence scores, or
`experiments/results/`. The structural validator independently guarantees that every
emitted relation carries evidence that is a verbatim substring of that sample's own
`inputs.context`, with both endpoints present. It cannot independently verify the
agents' out-of-band access or prove absence of gold exposure; those remain declared
orchestration provenance rather than a technical guarantee.

## Annotator A/B and agreement

| Annotator | Relations | explicit | cross_sentence | Samples left empty |
|-----------|-----------|----------|----------------|--------------------|
| A (`model_a_wb-reasoning`) | 114 | 112 | 2 | 4 |
| B (`model_b_wb-default`) | 105 | 105 | 0 | 11 |

| Metric | Overall | Task A | Task B |
|--------|---------|--------|--------|
| Sample-level exact agreement | **0.2000** | 0.2000 | 0.2000 |
| Relation-set Jaccard (mean) | **0.3646** | 0.3758 | 0.3534 |
| Relation-set Jaccard (micro) | **0.3436** | 0.4127 | 0.3000 |
| Positive agreement (Dice) | **0.5114** | 0.5843 | 0.4615 |

Agreed 56, A-only 58, B-only 49, 107 disagreements. The two annotators mostly found
*different* relations rather than contradicting each other, so the low agreement
reflects enumeration breadth as much as judgement.

## Adjudication

Adjudicator C ruled 106 of 107 disagreements `entailed` and 1 `not_entailed`
(sample 23, `pepo fruit weight part_of yield optimization` — the text says managing the
trait is part of the optimisation programme, not that the trait is part of the
objective). C added 7 joint misses, kept every agreed relation after re-checking
evidence and direction, and set `accept_agreed_relations=false` on zero samples.

Frozen result: **169 accepted relations across 30 samples**, stored in
`frozen_model_relations.json` with `annotation_source=dual_model_adjudicated`,
`human_annotated=false`, `confirmatory_eligible=false`, `calibration_ready=false`,
`exploratory_calibration_ready=true`.

## Calibration outcome (exploratory only)

Command (zero external calls — all 30 guarded-v2 generations were already cached):

```powershell
python D:\NLP\src\run_annotated_relation_calibration.py `
  --manifest D:\NLP\experiments\annotations\relation_text_model_annotation_v1\manifest.json `
  --annotation-source model --model dots3-note-prev `
  --frozen-annotations D:\NLP\experiments\annotations\relation_text_model_annotation_v1\frozen_model_relations.json `
  --grouped-cv-folds 5 --grouped-cv-seed 20260924
```

Artifact: `experiments/results/model_weaklabel_relation_calibration_open_vocab_guarded_v2_5b53d848edca6abb58f4_n30.json`

Grouped-CV artifact: `experiments/results/model_weaklabel_relation_calibration_grouped_cv_5b53d848edca6abb58f4_n30.json`

| Quantity | Value |
|----------|-------|
| Raw candidates / deployment-eligible | 234 / 200 |
| Canonical "true positives" | 26 |
| Independent positive samples | **14** (gate needs 5) |
| Brier, raw to calibrated (same fit population) | 0.7596 → 0.1131 |
| ECE, raw to calibrated (same fit population) | 0.8005 → 0.0037 |
| Brier, raw to calibrated (5-fold sample-grouped OOF) | 0.7596 -> 0.1134 |
| ECE, raw to calibrated (5-fold sample-grouped OOF) | 0.8005 -> 0.0045 |
| OOF candidate coverage / sample overlap | 200/200 / 0 |
| `candidate_level_deployable` | true |
| `deployable` | **false** |
| `status` | `exploratory_weak_labels_confirmatory_gate_closed` |

The numeric support gate passes for the first time: 14 positive samples clears the
preregistered minimum of 5, which the human path could not reach. The runner
nonetheless reports `confirmatory_eligible=false` and refuses promotion, by design.
The Brier/ECE values are resubstitution measurements on the same weak-label population
used to fit the calibrator, not held-out evidence of probability generalization.

The companion five-fold evaluation groups candidates by `sample_key`, so no sample can
appear in both fit and validation data. The 200 eligible candidates come from 22 of the
30 packet samples; the other eight have no eligible candidate to score. Fold sizes are
38--42 candidates, each fold has 2--3 positive samples, all 200 candidates receive
exactly one out-of-fold prediction, and sample overlap is zero. The OOF result remains
close to the resubstitution result, but all five fitted calibrators collapse to one
probability block (0.124--0.140). The improvement is therefore stable base-rate
shrinkage of overconfident scores, not evidence that confidence ranks candidates.

### Sample-cluster bootstrap uncertainty

The frozen OOF predictions were resampled 10,000 times with `sample_key` as the
cluster (`seed=20260924`, percentile 95% intervals). This preserves dependence among
candidates from the same text. The fold models are not refit inside each replicate,
so the intervals are conditional on the frozen OOF predictions and frozen weak labels.

| Metric | Raw OOF, estimate [95% CI] | Calibrated OOF, estimate [95% CI] |
|--------|----------------------------|-----------------------------------|
| Brier | 0.7596 [0.6747, 0.8268] | 0.1134 [0.0681, 0.1781] |
| ECE | 0.8005 [0.7062, 0.8693] | 0.0045 [0.0013, 0.0867] |

Weak-label positive prevalence is 0.1300 [0.0687, 0.2180]. The paired bootstrap
intervals for calibrated-minus-raw exclude zero for Brier [-0.7563, -0.4980] and ECE
[-0.8125, -0.6225]. This establishes robust compression toward the weak-label base
rate, not relation correctness, human-label validity, or confidence ranking.

Uncertainty artifact:
`experiments/results/model_weaklabel_relation_calibration_grouped_cv_uncertainty_5b53d848edca6abb58f4_n30.json`

Figure (editable vector plus 300 dpi raster):
`papers/figures/weaklabel_relation_calibration_reliability.svg`, `.pdf`, and `.png`.

## Two caveats that must travel with these numbers

**1. "True positive" here means "agrees with the weak labels", not "is correct".**
Official gold non-taxonomic triples across these 30 samples total **one** edge. The
169 weak labels overlap it **zero** times. That is good news for integrity — there was
nothing to leak and the labels are text-grounded, not gold-grounded — but it also means
the calibration has **no external gold validation at all**. It measures agreement with
a noisy label set, and that label set had only 0.36 mean Jaccard between its own two
producers.

**2. The targeted-audit shortcut barely saves work here.**
Targeted auditing covers all disagreements (24 samples), all positives (26 samples) and
20% of agreed negatives (1 sample) — **27 of 30 samples**. At 0.20 agreement, nearly
every sample is either a disagreement or a positive, so the cheap path is close to a
full audit. The shortcut pays off only when agreement is high; this packet is not that
packet.

Correct claim: *exploratory dual-LLM weak-label calibration*.
Incorrect claim: *independently human-annotated calibration*.

## Next step: targeted human audit

`human_audit_worklist.json` identifies the 27 samples that need review, but its current
sample-level `human_verdict` field is only a triage marker. It does not contain context,
evidence, or per-relation corrections and therefore cannot be frozen or used to open a
gate. A valid audit must judge every proposed relation, record corrected/missing
relations with verbatim evidence, and retain human provenance. Until such an artifact
exists, the gate stays closed and no confirmatory API run starts.

## Reproduce

```powershell
# 1. regenerate the packet (reuses the human packet's sample selection)
python D:\NLP\src\relation_annotation_workflow.py prepare-model `
  --source-packet-dir D:\NLP\experiments\annotations\relation_text_annotation_v1

# 2. validate both annotations
python D:\NLP\src\relation_annotation_workflow.py validate --annotation-source model `
  --manifest D:\NLP\experiments\annotations\relation_text_model_annotation_v1\manifest.json `
  --annotations D:\NLP\experiments\annotations\relation_text_model_annotation_v1\model_annotator_a.jsonl

# 3. compare, then adjudicate the output by hand
python D:\NLP\src\relation_annotation_workflow.py compare --annotation-source model `
  --manifest D:\NLP\experiments\annotations\relation_text_model_annotation_v1\manifest.json `
  --annotations-a D:\NLP\experiments\annotations\relation_text_model_annotation_v1\model_annotator_a.jsonl `
  --annotations-b D:\NLP\experiments\annotations\relation_text_model_annotation_v1\model_annotator_b.jsonl `
  --output D:\NLP\experiments\annotations\relation_text_model_annotation_v1\model_adjudication.jsonl

# 4. freeze
python D:\NLP\src\relation_annotation_workflow.py freeze --annotation-source model `
  --manifest D:\NLP\experiments\annotations\relation_text_model_annotation_v1\manifest.json `
  --adjudication D:\NLP\experiments\annotations\relation_text_model_annotation_v1\model_adjudication.jsonl `
  --output D:\NLP\experiments\annotations\relation_text_model_annotation_v1\frozen_model_relations.json

# 5. recompute sample-cluster intervals and export the reliability figure locally
python D:\NLP\src\analyze_grouped_relation_calibration.py `
  --bootstrap-replicates 10000 --bootstrap-seed 20260924 --bins 10
```

Provenance is validated, never inferred. `--annotation-source model` rejects any file
declaring `annotator_type="human"` and requires `model_id` plus `prompt_version`;
`--annotation-source human` still rejects `annotator_type="model"`. A model file cannot
be relabelled as human by editing a field.
