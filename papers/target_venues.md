# OntoBridg Venue Strategy

**Assessment date:** 2026-09-28

## Current Paper Type

OntoBridg is currently a reproducible negative-results and evaluation-protocol paper with one bounded controlled positive result. Its defensible contribution is:

- relation-channel constraints improve graph precision;
- a synthetic nonce holdout demonstrates controlled extraction;
- the gain fails to transfer to relation-rich official text;
- independent predicate routing fails a preregistered support gate;
- contamination diagnostics show probeability and task-dependent surface sensitivity without proving membership;
- the system remains fail-closed for calibration and deployment.

It is not currently a leaderboard paper, a hidden-test study, a multi-model benchmark, or a demonstrated naturalistic performance improvement.

## Recommended Sequence

### 1. LLMs4OL 2027 workshop or closely aligned ISWC workshop

**Recommendation:** primary target, subject to the official 2027 call.

**Fit:** high. The manuscript directly examines LLMs4OL Task A/B behavior, exact graph metrics, relation extraction, reproducibility, and evaluation safeguards. Shared-task and workshop audiences can evaluate a carefully bounded negative result without requiring the broader empirical coverage expected of an ISWC full research paper.

**Submission framing:**

> Evidence constraints recover controlled relations but do not transfer to official text; fail-closed gates prevent a synthetic gain from becoming an unsupported deployment claim.

**Required before submission:**

- verify the official call, page limit, anonymity policy, and artifact rules;
- move the Markdown manuscript into the required CEUR/LNCS template;
- add author metadata only after the anonymity policy is known;
- archive an anonymized artifact with a stable URL or DOI.

### 2. Negative-results, evaluation, or neuro-symbolic NLP workshop

**Recommendation:** fallback if no suitable LLMs4OL 2027 call appears.

**Fit:** moderate to high when framed around evaluation science, sparse structured gold, evidence-bound off-gold generation, and fail-closed decision rules. The ontology-learning context must remain explicit; this should not be reframed as generic relation extraction without additional baselines.

### 3. ISWC research track

**Recommendation:** do not submit the current package as a full research paper.

The present evidence uses one hosted model and public training data, and the main naturalistic relation result is negative. A competitive full-paper version would normally require at least one of the following:

- replication across multiple model families;
- a genuinely independent naturalistic or temporal dataset;
- a human-adjudicated completeness study for off-gold relations;
- a benchmark or annotation resource that can be released and reused;
- a new method that succeeds under the already frozen transfer protocol.

These additions must use new data or models; the failed routing slice cannot be repaired or tuned.

### 4. JWS, Semantic Web Journal, or Knowledge-Based Systems

**Recommendation:** later archival extension only.

A journal version should add multi-model replication, broader dataset coverage, human adjudication, and a mature public artifact. The current manuscript can serve as the protocol and negative-results core of that extension.

### 5. OAEI / Ontology Matching workshop

**Recommendation:** not a primary target.

Ontology matching is used only as related-work context. The evaluated task is ontology learning/extension, and no matching-system experiment is reported.

## Claims by Venue

| Claim | Workshop manuscript | Full research/journal extension |
|---|---|---|
| Relation-only guard improves graph precision | Ready | Replicate across models/data |
| Guarded nonce extraction reaches F1 0.8571 | Ready with synthetic qualifier | Add naturalistic validation |
| Official-text transfer recovers 0/101 | Ready as negative result | Diagnose with human completeness study |
| Predicate routing fails five-sample gate | Ready | Test a new preregistered method on new data |
| Facts are highly probeable | Ready as diagnostic | Add stronger contamination methodology |
| Verified pretraining contamination | Not supported | Requires corpus access or stronger evidence |
| Deployment-ready calibration | Not supported | Requires independent human/official support |

## Venue Checks Still Requiring External Confirmation

- [ ] Official LLMs4OL 2027 call and submission deadline
- [ ] Page limit and template
- [ ] Double-blind versus single-blind policy
- [ ] Prior-preprint policy
- [ ] Artifact review and anonymization requirements
- [ ] Whether workshop publication permits a later extended journal version

No deadline is recorded here until it has been verified from the official call.
