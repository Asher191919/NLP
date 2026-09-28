# OntoBridg Internal Submission Review

**Review date:** 2026-09-28  
**Target:** specialized ontology-learning, evaluation, or negative-results workshop  
**Recommendation before revision:** borderline / weak accept for a workshop; reject for a full ISWC research track

## Summary Assessment

The manuscript has a coherent and unusually auditable negative-results narrative. Its strongest contribution is not a state-of-the-art ontology learner, but a staged protocol that prevents a controlled synthetic gain from being promoted after two independent official-text failures. The evidence is sufficient for a focused workshop paper if the claim boundary remains explicit. It is not sufficient for a broad naturalistic-performance, contamination-membership, or deployment claim.

## Major Review Risks

### 1. Leakage language could overstate what was controlled

The prompts withhold local gold labels, but the hosted model may have encountered public ontologies during pretraining. Calling the protocol simply "leakage-controlled" invites the objection that training-data exposure was not controlled.

**Resolution:** the manuscript now uses "gold-withheld prompt protocol" and reserves contamination language for the probeability and alias diagnostics.

### 2. Local freezing is not public preregistration

The project records immutable manifests and precommitted gates before execution, but no public registry or trusted timestamp is claimed. "Preregistered" would imply stronger external verification.

**Resolution:** the manuscript now uses "manifest-frozen" and "precommitted" and the supplement explicitly states that the manifests were not registered in a public preregistration service.

### 3. External validity is narrow

The official-data evidence uses one hosted model alias and public training files. The strongest positive result is synthetic and predicate-familiar. This limits claims about model families, naturalistic text, unseen predicates, and future challenge data.

**Disposition:** retained as a central limitation. Expanding models or datasets is an archival-extension requirement, not a same-paper repair after the failed routing gate.

### 4. Sparse gold makes exact false positives ambiguous

The routing system emits 138 outputs absent from the official gold, even though most have accepted evidence and verbatim endpoints. Without a completeness statement or independent semantic adjudication, these are exact-scoring false positives rather than verified semantic errors.

**Disposition:** already stated in the figure caption, routing audit, discussion, and limitations. No semantic-correctness claim is made.

### 5. Method novelty is secondary to evaluation design

Confidence and evidence filtering are technically sensible but individually conventional. The paper is more defensible as an evaluation-protocol and failure-characterization contribution than as a new extraction architecture.

**Disposition:** title, abstract, contributions, and conclusion foreground fail-closed evaluation. The manuscript does not claim state-of-the-art performance.

### 6. Uncertainty reporting is selective

Bootstrap intervals exist for the confirmatory, contamination, weak-label, and routing analyses, but not every descriptive table reports an interval. A blanket statement that all intervals use a given protocol could be read as implying universal interval coverage.

**Resolution:** the methods section now says "Where reported" and emphasizes support counts and effect sizes for small stratified slices.

## Minor Review Risks

- The density-stratified comparisons are diagnostic because gold density influenced selection; this is stated in Table 1 and should remain visible after template conversion.
- Task B in the blind prefix has zero non-taxonomic gold. The text now avoids presenting this as a missed-recall result.
- The provider alias is reproducible only as an experiment identifier; immutable provider weights are unavailable.
- The weak-label packet must remain described as dual-model weak annotation without human audit.
- The semantic secondary metric must remain secondary and cannot reopen the failed exact gate.
- External baseline numbers should remain cited context, not a direct same-split comparison.

## Evidence Verdict

| Claim | Review status |
|---|---|
| Relation-channel filtering reduces unmatched edges | Supported on gold-stratified diagnostics |
| Guarded extraction succeeds on the nonce holdout | Supported for controlled synthetic text |
| The nonce gain transfers to official text | Refuted on the frozen transfer slice |
| Predicate routing establishes official relation recovery | Not supported; five-sample gate fails |
| Calibration is deployment-ready | Not supported |
| Public benchmark facts are probeable | Supported by the closed-book diagnostic |
| Verified pretraining membership or global score inflation | Not supported |
| Off-gold routed outputs are semantically wrong | Not adjudicated |

## Submission Recommendation

Proceed to a specialized workshop after anonymous artifact verification and venue-template conversion. Do not submit the current evidence package as a full research-track paper without independent naturalistic data, multi-model replication, or human semantic adjudication. Do not reopen the failed routing slice for prompt repair or launch the blocked n>=100 expansion.
