# OntoBridg Release Readiness

**Checked:** 2026-09-29  
**Artifact baseline commit:** `2e696b7`  
**Venue/release handoff commit:** `0a03592`  
**Anonymous artifact:** `dist/ontobridg-anonymous-artifact.zip`  
**Artifact SHA-256:** `b9e9f5bce1b004cca5a6ec30d7ac2ed0dc196fda6d520dadba676f2bb6f87777`

## Completed

- Source workspace: 170/170 tests pass.
- Anonymous artifact: 129/129 data-free portable tests pass in a fresh virtual environment.
- Manuscript checker, anonymous scan, figure regeneration, and figure hashes pass.
- The archive excludes API credentials, raw generation caches, checkpoints, third-party datasets, source contexts, protected gold, and author metadata.
- The live LLMs4OL series page was checked on 2026-09-29. No 2027 call or submission rules were published, so no deadline or template is recorded as fact.

## Required owner decisions before public release

1. Select the project code/artifact license and confirm that the rights holder authorizes it. The upstream challenge is MIT-licensed, but that does not select a license for this project or clear upstream ontology resources.
2. Supply author names, affiliations, ORCID identifiers, and the corresponding-author address after the target venue's anonymity policy is known.
3. Select the submission venue and convert `papers/paper_draft.md` / `papers/paper_draft.tex` to that venue's official template. Recheck page limits, figure dimensions, and accessibility requirements then.
4. Publish the anonymized ZIP at an approved repository and record its stable URL, DOI, and release tag in the camera-ready artifact statement.

Until these decisions are supplied, the package is technically release-ready but is not represented as publicly archived or venue-compliant.
