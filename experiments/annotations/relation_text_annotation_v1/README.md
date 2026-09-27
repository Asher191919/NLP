# Independent relation calibration annotation packet

Packet ID: `5b53d848edca6abb58f4`

## Current Status (2026-09-24)

- [x] Thirty unused samples selected without gold/model-output access.
- [x] Annotator A and B packets pass schema validation (`rows=30`).
- [x] Guarded-v2 generations are cached for all 30 samples; fitting needs no API calls.
- [ ] Annotator A: `complete=0/30`.
- [ ] Annotator B: `complete=0/30`.
- [ ] Third-human adjudication and frozen labels.
- [ ] Local calibration fit and deployment-gate decision.

Do not run confirmatory API evaluation or expand to n >= 100 until the frozen
annotation fit passes every preregistered deployment gate.

This packet contains 30 previously unused LLMs4OL 2026 training samples: 15 Task A
and 15 Task B. Selection used text-only relation cues. It did not use official gold
or model output. The selected indices have zero overlap with the blind prefix,
density evaluation, A27+B12 calibration data, or the v3 pilot.

## Assignment

- Give `annotator_a.jsonl` to one human annotator.
- Give `annotator_b.jsonl` to a different human annotator.
- Do not give either annotator the other annotation file, official gold, model
  output, confidence summaries, or files under `experiments/results/blind_cache`.
- Give both annotators `annotation_guidelines.md`.
- Replace the placeholder `annotator_id`, set `annotator_type` to `human`, and add
  an ISO-8601 `completed_at` value before marking a row `complete`.
- Evidence must be copied verbatim from `inputs.context`. Task B terms, types, and
  initial triples may help identify entities but cannot serve as relation evidence.

The workflow validates declared provenance but cannot verify a person's identity.
Keep the external assignment record needed to support a human-annotation claim.
`assignment_record.template.md` provides a privacy-preserving template; retain the
mapping from pseudonymous IDs to real people outside the public repository.

## Validation

```powershell
python D:\NLP\src\relation_annotation_workflow.py validate `
  --manifest D:\NLP\experiments\annotations\relation_text_annotation_v1\manifest.json `
  --annotations D:\NLP\experiments\annotations\relation_text_annotation_v1\annotator_a.jsonl

python D:\NLP\src\relation_annotation_workflow.py validate `
  --manifest D:\NLP\experiments\annotations\relation_text_annotation_v1\manifest.json `
  --annotations D:\NLP\experiments\annotations\relation_text_annotation_v1\annotator_b.jsonl
```

Both commands must report `complete=30`.

## Compare And Adjudicate

```powershell
python D:\NLP\src\relation_annotation_workflow.py compare `
  --manifest D:\NLP\experiments\annotations\relation_text_annotation_v1\manifest.json `
  --annotations-a D:\NLP\experiments\annotations\relation_text_annotation_v1\annotator_a.jsonl `
  --annotations-b D:\NLP\experiments\annotations\relation_text_annotation_v1\annotator_b.jsonl `
  --output D:\NLP\experiments\annotations\relation_text_annotation_v1\adjudication.jsonl
```

A third human must inspect every sample, resolve every disagreement as `entailed`
or `not_entailed`, use a distinct `adjudicator_id`, and mark adjudication complete.
Then freeze the labels:

```powershell
python D:\NLP\src\relation_annotation_workflow.py freeze `
  --manifest D:\NLP\experiments\annotations\relation_text_annotation_v1\manifest.json `
  --adjudication D:\NLP\experiments\annotations\relation_text_annotation_v1\adjudication.jsonl `
  --output D:\NLP\experiments\annotations\relation_text_annotation_v1\frozen_relations.json
```

## Fit Without External Calls

All 30 guarded-v2 generations are already cached. After freezing annotations:

```powershell
& D:\NLP\src\run_annotated_relation_calibration.ps1 `
  -FrozenAnnotations D:\NLP\experiments\annotations\relation_text_annotation_v1\frozen_relations.json
```

The audit currently reports 30 generation-cache hits and zero external calls. The
fit remains fail-closed unless at least five independent samples contain accepted
true-positive candidates and the candidate-level class-support checks also pass.
