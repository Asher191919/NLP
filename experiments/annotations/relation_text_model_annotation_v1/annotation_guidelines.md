# Relation annotation guidelines (dual-model weak labelling)

Protocol: `relation_text_model_annotation_v1`
Prompt version: `relation_model_annotation_v1`

This packet is for exhaustive, text-grounded non-taxonomic relation annotation.
Model annotators must work in isolated contexts and must not inspect official gold,
model output, confidence scores, caches, or the other annotator's file.

## Per-sample rules

1. Read the whole `inputs.context`.
2. Enumerate every non-taxonomic relation the text states explicitly or entails
   reliably.
3. Do not annotate `is-a`, subclass, instance/type, or ordinary co-occurrence.
4. Prefer these canonical predicates: `part_of`, `has part`, `develops_from`,
   `derives from`, `located in`, `positively regulates`, `regulates`.
5. Use a different predicate only when none of the above is accurate.
6. Check direction strictly: `X part_of Y` must not be written `Y part_of X`.
7. `explicit` must carry one verbatim evidence quote.
8. `cross_sentence` must carry two verbatim evidence quotes.
9. `subject` and `object` must both occur in the evidence after ignoring case and
   collapsing repeated whitespace.
10. When no relation holds, emit an empty array. Never guess.
11. For Task B, `terms`, `types`, and `initial-primitive-ontology-triples` may only
    help locate entities. They are never evidence.

## Model annotation shape

Use `annotator_type: "model"`. Never write `hostname: "human"`.

```json
{
  "annotator_id": "model_a_<model-id>",
  "annotator_type": "model",
  "model_id": "<exact-model-name>",
  "prompt_version": "relation_model_annotation_v1",
  "completed_at": "2026-09-24T12:00:00+08:00",
  "status": "complete",
  "relations": [
    {
      "subject": "...",
      "predicate": "part_of",
      "object": "...",
      "evidence_mode": "explicit",
      "evidence": ["verbatim quote"],
      "notes": ""
    }
  ],
  "sample_notes": ""
}
```

## Independence and claim limits

- Give `model_annotator_a.jsonl` to annotator A and `model_annotator_b.jsonl` to
  annotator B. Neither may read the other file.
- Annotator C may read both files only after A and B are finished.
- Prefer different model families. Three same-origin agents are repeated sampling,
  not strong independence; record the real model IDs and say so.
- Use temperature 0 and record model name, version, date, and prompt hash.
- These are **weak labels**. They support exploratory engineering calibration. They
  do **not** support an independently human-annotated claim, and the confirmatory
  gate stays closed until a real human audit exists.

## Adjudication

1. Normalize both relation sets; match on the full `(subject, predicate, object)`
   triple.
2. Agreement still needs evidence and direction checked by C.
3. Judge every disagreement `entailed` or `not_entailed`.
4. C must also read each text to catch relations both annotators missed.
5. C records its own model, prompt, timestamp, and decision reasons.
6. After freezing, report sample-level exact agreement, relation-set Jaccard,
   positive agreement, and predicate-level disagreement.

## Lowest-cost credible compromise

Have A and B label all samples, have C adjudicate, then have **one human** review all
disagreements, all positives, and a random 20% of agreed negatives. Report that as
`dual-LLM annotation with targeted human audit`. That is far cheaper than full
three-person annotation and far more defensible than an unaudited model self-label.
