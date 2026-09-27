# OntoBridg — Target Venues (Ranked)

> **Status**: Venue shortlist v0.1 | 2026-09-20
> **Project**: OntoBridg | `D:\NLP` | LLMs4OL (not OAEI primary)
> **Notes**: Deadlines are approximate / “if known”; verify on official CFP before submitting. Chinese OK for planning notes.

---

## Ranking Criteria Used

1. Fit to **LLMs4OL / ontology learning + contamination** contribution
2. Chance of **competition-track** acceptance while system paper + protocol are still maturing
3. Path to **journal / archival** extension
4. Review cycle speed vs. paper readiness (code + experiments incomplete as of 2026-09)

---

## Tier 1 — Primary Targets (LLMs4OL / ISWC ecosystem)

### 1. LLMs4OL Workshop (co-located with ISWC or venue of that year)

| Item | Detail |
|------|--------|
| **Why #1** | Exact task family; competition/system papers expected; contamination critique is on-topic |
| **Paper type** | System + evaluation-protocol paper (competition track) |
| **Fit** | 2026 Task A Flagship / B Reuse / C Taxonomy + baselines Alexbek / pro-team + OntoBridg |
| **Deadline** | Typically ~ISWC workshop deadline (often May–July for autumn ISWC). **2027 track TBD** — monitor arXiv/LLMs4OL site |
| **Length** | Often CEUR 8–12 pages |
| **Risk** | Depends on 2027 task stability; if track changes, reframe using 2026 Flagship/Reuse/Taxonomy names |

**Action:** Track LLMs4OL 2027 CFP; pre-register system; submit contamination protocol as core differentiator.

### 2. ISWC (International Semantic Web Conference) — Research / Resource track

| Item | Detail |
|------|--------|
| **Why** | Core venue for ontology learning + neuro-symbolic OL; audience understands TBox, SGS, OAEI contrast |
| **Paper type** | Full research paper (contamination protocol + hybrid method) or resource/demo if engineering-led |
| **Fit** | Strong once Stage 2–6 experiments exist |
| **Deadline** | ISWC research track historically: abstract ~April, full ~May (for that year’s conference). **Verify current year CFP** |
| **Length** | ~16 pages Springer LNCS (+ references) or as CFP specifies |
| **Risk** | High bar; needs solid baselines + clean ablations |

**Action:** If LLMs4OL workshop happens first, workshop paper → extended ISWC/journal version.

### 3. OM Workshop @ ISWC (Ontology Matching) — CEUR — **Backup only**

| Item | Detail |
|------|--------|
| **Why backup** | Project direction is **LLMs4OL not OAEI**; OAEI 2026 execution deadline already passed (2026-09-05 per DESIGN) |
| **Use for** | (a) Hybrid symbolic vs LLM contrast notes; (b) if LLMs4OL 2027 timeline fails, pivot matching module to OAEI 2027 |
| **Paper type** | Short system / position paper on contamination in matching **or** OntoBridg-matching port |
| **Deadline** | Usually aligned with ISWC workshop deadlines (often August for autumn ISWC). **2027 TBD** |
| **Venue outlet** | CEUR-WS |
| **Risk** | Mismatch with current main claims; do not lead with OAEI for this paper |

---

## Tier 2 — Journals (Extension After Workshop / Strong Preprint)

### 4. Journal of Web Semantics (JWS)

| Item | Detail |
|------|--------|
| **Why** | Home venue for ontology learning, semantic web, hybrid systems |
| **Paper type** | Full archival article: OntoBridg + contamination protocol + extensive ablations |
| **Fit** | Excellent after experimental results; SGS/graph metrics welcome |
| **Timeline** | Rolling; major revision cycles common. Target **after** workshop acceptance + extra datasets |
| **Risk** | Slow review; needs completeness |

### 5. Knowledge-Based Systems (KBS)

| Item | Detail |
|------|--------|
| **Why** | Publishes neuro-symbolic + LLM hybrid KRR systems; good for method-heavy framing |
| **Paper type** | Research article |
| **Fit** | Strong for multi-objective scoring + symbolic repair + open relation vocab |
| **Timeline** | Rolling; relatively faster than some SW venues |
| **Risk** | Less “semantic web competition” audience; emphasize general OL method |

### 6. Semantic Web Journal (SWJ)

| Item | Detail |
|------|--------|
| **Why** | Direct OL/alignment audience; open review culture |
| **Paper type** | Research paper; possible special issue on LLMs + SW |
| **Fit** | Contamination + LLMs4OL + hybrid pipeline |
| **Timeline** | Rolling + special issues (watch SWJ calls) |
| **Risk** | Special-issue deadlines can be tight |

---

## Tier 3 — NLP Conferences / Findings (Alternate Framing)

### 7. EMNLP / ACL Findings (or *SEM)

| Item | Detail |
|------|--------|
| **Why** | Contamination methodology is broadly eval-science; open relation extraction is NLP |
| **Paper type** | Findings short/long; or workshop (BlackboxNLP, NLP4ConvAI not ideal — prefer **eval / resources** workshops) |
| **Fit** | Best if headline is *contamination-aware evaluation for structured prediction from LLMs* rather than full OL system |
| **Deadline** | ACL/EMNLP cycles ~May and ~Aug (varies by year; ARR + direct). **Verify** |
| **Risk** | Semantic-web depth (SGS, TBox repair) may need compression |

### 8. AAAI / IJCAI (secondary)

| Item | Detail |
|------|--------|
| **Why** | Neuro-symbolic AI framing |
| **Fit** | Only if method novelty (constraints + open vocab + contamination) is very strong |
| **Deadline** | AAAI often abstract July–Aug for following winter; IJCAI ~January |
| **Risk** | Crowded; OL-specific audience thinner |

### 9. LREC / ESWC / EKAW (niche SW/NLP)

| Item | Detail |
|------|--------|
| **ESWC** | Good ISWC sibling; research + resources; spring deadline typical |
| **EKAW** | Knowledge engineering / ontology learning focused when held |
| **LREC** | If contribution reframed as dataset/eval resource (contamination probes + clean splits) |

---

## Recommended Submission Sequences

### Sequence A — Competition-first (aligned with LLMs4OL 2027)

```
Now (2026-09) → implement Stage 2–6, pilots on WordNet + schema.org
      ↓
2026 H2 / 2027 H1 → track LLMs4OL 2027 CFP + ISWC workshop deadlines
      ↓
Submit system+protocol paper → LLMs4OL workshop (CEUR)
      ↓
Extend with full ablations + more datasets → JWS or SWJ (or ISWC research next cycle)
```

### Sequence B — Method-first (if competition timeline slips)

```
Draft contamination protocol + hybrid method on public LLMs4OL data
      ↓
Preprint arXiv
      ↓
Findings / ESWC / ISWC research track
      ↓
Optional: OM Workshop short paper for OAEI contrast / matching port
      ↓
Journal (KBS or JWS) extended version
```

### Sequence C — OAEI backup (only if LLMs4OL blocked)

```
Port matching module → LogMap-style hybrid + LLM candidates
      ↓
Pre-register OAEI 2027 → OM Workshop paper
      ↓
Not the current primary narrative; keep as DESIGN.md secondary path
```

---

## Deadline Checklist (Fill Official Dates)

| Venue | Approx window | Official CFP | Decision |
|-------|---------------|--------------|----------|
| LLMs4OL 2027 workshop | ~mid-year (ISWC-linked) | ☐ pending | **Primary** |
| ISWC 2027 Research | ~Apr–May 2027 | ☐ pending | Primary after workshop |
| OM @ ISWC 2027 | ~Aug 2027 | ☐ pending | Backup |
| ESWC 2027/2028 | ~Nov–Dec prior year | ☐ pending | Optional |
| EMNLP/ACL Findings | ARR cycle | ☐ pending | Alt framing |
| JWS / SWJ / KBS | Rolling | ☐ open now | Extension |

*Note: Project DESIGN.md states OAEI **2026** execution deadline was **2026-09-05** (passed) — reinforces LLMs4OL as primary.*

---

## Paper Packaging per Venue

| Venue | Lead pitch | Must include | Can cut |
|-------|------------|--------------|---------|
| LLMs4OL workshop | System + contamination protocol on official tasks | A/B/C results, Alexbek/pro-team compare, stress tests | Long OL history |
| ISWC research | Hybrid OL + eval science | Full method, repair rules, SGS analysis | Competition diary |
| OM CEUR | Contamination in matching / hybrid wins | OAEI contrast, LogMap/AML/BERTMap cites | Open vocab depth |
| JWS/SWJ/KBS | Complete method + protocol archive | All tables, release notes, license appendix | Workshop anecdotes |
| NLP Findings | Contamination eval for structured LLM tasks | Probes, stress tests, ranking-flip | OWL export details |

---

## Practical Notes

1. **Do not claim** Alexbek/pro-team numbers until reproduced or carefully cited from arXiv (2508.19428, 2608.27101).
2. **UMLS license**: journal/workshop submission can describe access path; avoid committing restricted data.
3. **Dual submission**: workshops (CEUR) often allow later extension; check each CFP.
4. **Name consistency**: system is **OntoBridg** (project DESIGN spelling); keep spelling stable across drafts.
5. **Evidence gate**: workshop paper OK with 2 datasets + clear protocol; ISWC/journal want 3–4 datasets + full ablations.

---

## Immediate Next Venue Actions

- [ ] Confirm LLMs4OL 2027 task definitions + SGS formula (from 2026 overview / organizers)
- [ ] Add official CFP links + exact dates to this file when known
- [ ] Choose Sequence A vs B after first end-to-end run on WordNet
- [ ] Keep `paper_draft.md` abstract non-numeric until results exist
