# Relation annotation guidelines

This packet is for exhaustive, text-grounded non-taxonomic relation annotation.
Annotators must work independently and must not inspect official gold, model output,
confidence scores, or the other annotator's file.

For every relation entailed by the visible inputs:

1. Record `subject`, canonical `predicate`, and `object`.
2. Prefer: `part_of`, `has part`, `develops_from`, `derives from`, `located in`,
   `positively regulates`, or `regulates`. Other specific non-taxonomic predicates
   are allowed when none of these is faithful.
3. Copy one exact evidence quote for an explicit relation. Use two exact quotes and
   `cross_sentence` only when the relation requires two nearby spans.
4. Both endpoint strings must occur in the combined evidence after case and
   whitespace normalization. Use the surface forms actually stated in the text.
5. Do not annotate `is-a`, subclass, instance/type, or ordinary co-occurrence.
6. Do not infer disjointness from ordinary sibling classes or equivalence from an
   ordinary subtype statement.
7. Set sample status to `complete` only after checking the full visible input. Use
   `abstain` when the text cannot be judged reliably; never treat abstention as a
   negative label.
8. For a completed row, set a stable pseudonymous `annotator_id`, set
   `annotator_type` to `human`, and record an ISO-8601 `completed_at` timestamp.
   Replace the template placeholder ID. The two annotators and the adjudicator
   must use three different IDs.

Relation object shape:

```json
{"subject":"...","predicate":"part_of","object":"...","evidence_mode":"explicit","evidence":["exact quote"],"notes":""}
```

After both files are complete, `compare` creates a separate adjudication file.
Only a human adjudicator may resolve disagreements and mark each sample `complete`.
`freeze` rejects pending, abstained, malformed, or unresolved samples.
