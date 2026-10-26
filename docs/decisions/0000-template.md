# 0000 — ADR Template (full)

**Status:** Template
**Date:** 2026-05-14
**Deciders:** —

> **How to use.** Stamp new ADRs from this file when authoring a *material decision* — typically architectural choices, scope foreclosures, methodology commitments, supersessions, or parking decisions that will be referenced later. For *trivial / single-decision* ADRs (expected <150 lines, no panel review, no supersede chain), stamp from `0000-template-minimal.md` instead. Renumber from `README.md` index (gap-aware: skip RESERVED numbers like 0019). Set Status to one of: `Accepted | Accepted — Parked | Superseded by NNNN | Partially superseded by NNNN | Deprecated`. Strip `<!-- OPTIONAL -->` sections you don't use; KEEP `<!-- RECOMMENDED -->` sections by default and only drop them if genuinely N/A.

<!-- OPTIONAL — include when the ADR was authored via a particular workflow (autonomous, panel-augmented, WWMD-gated, etc.) -->
> **Authoring note.** [Procedural context for future readers — e.g. "Authored via 3-stage workflow: @overseer brainstorm → perturb/critic/validate → Gemini Pro consent. Panel verdicts in transcript at <path>."]

<!-- OPTIONAL — include when the ADR's number doesn't follow simple sequential allocation (gaps from RESERVED placeholders, re-charters, etc.) -->
> **Numbering note.** [Why this number, not the next sequential. E.g. "0019 RESERVED per ADR 0018; this is 0020."]

<!-- OPTIONAL — include for ADRs with rich metadata: parking decisions, scoped supersedes, multi-ADR relationships. Single-purpose YAML block, NOT full frontmatter (project convention is bold-text metadata above). -->
```yaml
adr_metadata:
  supersedes: []                       # list of ADR numbers fully replaced
  supersedes_partial:                  # list of {id, scope} for partial supersession
    - {id: NNNN, scope: "what's superseded"}
  superseded_by: []                    # back-link if this ADR is later replaced
  related: []                          # list of ADR numbers / planning paths
  # for parked decisions, add:
  archived_date: YYYY-MM-DD
  archived_by: name
  retire_rationale: |
    why this is being parked
  reopen_trigger: |
    what would have to be true to reopen
  reopen_requires:
    - mechanical conditions
```

---

<!-- RECOMMENDED — for material decisions; skip for foundational ADRs not reactive to a triggering event. 6+ ADRs use this. -->
## Trigger and evidence

What changed that forced this ADR now. Cite specific commits, plan files, or prior ADR sections. Be concrete — `commit abc1234`, `Plan 02-04b T4 §8`, `data/run-reports/<run>/RUN-REPORT.md`. Future readers need to know whether the triggering evidence still holds before relying on this decision.

<!-- REQUIRED -->
## Context

What's the situation that forced a decision? What constraints, evidence, or prior decisions are in play? Be specific — the future reader needs to know whether the context still holds before relying on the decision.

<!-- REQUIRED -->
## Decision

What was decided. One paragraph minimum, multiple if the decision has parts. State the decision in the active voice ("We will...") so it can't be misread as a description of what others do.

<!-- OPTIONAL — for multi-branch decisions, sub-section each branch -->
### Sub-decisions

#### 3a — [Branch name]
[What this branch commits to.]

#### 3b — [Branch name]
[What this branch commits to.]

<!-- RECOMMENDED — explicit boundary-setting. 6+ ADRs use this. -->
## Scope

**In scope:**
- [bullets]

**Out of scope (explicit):**
- [bullets — what this ADR does NOT cover, even if related. Pre-empts scope-creep claims.]

<!-- OPTIONAL — for amendments / supersessions. Quote the prior ADR section verbatim and state how it carries forward (binding rule preserved, retired, modified). 3+ ADRs use this (emerging pattern). -->
## Binding cites / Carry-forward from prior ADRs

Verbatim cite of the prior ADR section being preserved or modified:

> "[exact quote from ADR NNNN §M]"

**This ADR affirms / amends / retires.** [How the cited rule carries forward under this ADR.]

<!-- OPTIONAL — for amendments / supersessions affecting multiple prior ADRs. 3+ ADRs use this (emerging pattern). -->
## Inheritance table

| Prior ADR | Status under this ADR | Notes |
|---|---|---|
| 00NN | Historical record / Carry-forward / Retires / Subsumed | [brief] |

<!-- OPTIONAL — codifies how this ADR can be amended. 5+ ADRs use this. Default is "by later ADR"; specify if stricter. -->
## Override clause

How this ADR can be amended:
- By default: superseded by a later ADR
- Stricter constraints (if applicable): [e.g. "requires panel review", "requires WWMD verdict", "unamendable for N months"]

<!-- REQUIRED -->
## Consequences

The good, the bad, and the ugly. What does this decision make easy? What does it make hard? What did we explicitly accept the cost of?

<!-- RECOMMENDED — what evidence would force this ADR to be revisited or superseded. 7+ ADRs codify this. For research decisions, qualitative triggers are fine. For engineering, prefer quantifiable thresholds. For parked decisions, this becomes the reopen-gate (move to `adr_metadata.reopen_trigger` instead). -->
### Falsifiability anchor / When to revisit

[Specific evidence or condition that would force this ADR to be reopened. E.g. "If the targeted harvest produces ≥3 schema deltas v0-skeleton can't accommodate, reopen the schema-as-output assumption." Or qualitative: "If a Pillar 3 design surfaces a divergent-friendly use case, reopen."]

<!-- RECOMMENDED — relates-to, not bibliography. Distinct from Sources below. 6+ ADRs use this. -->
## Cross-references

- Linked planning docs: `docs/<plan-or-design>.md`
- Linked schema / data files
- Prior ADRs this builds on (cite as `ADR NNNN §M` — section anchor; never `$N` notation)
- Related successor ADRs (forward-link only when bidirectional reciprocity is intentional; default is no forward-link, historical-record preservation)

<!-- REQUIRED — bibliography: external evidence, conversation logs, paper citations -->
## Sources

- External references (papers, conversations, tool docs) by URL or absolute path
- Source documents that informed the decision (transcripts, panel outputs, plan files)
- Author/decider credits if not in the top metadata
