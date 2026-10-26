# 0002 — Publish A Fresh Export Through A Leak Gate

**Status:** Accepted
**Date:** 2026-10-25
**Deciders:** Martin

> **Authoring note.** Decided in a brainstorm session on 2026-10-25, the day of the race. The full
> release design lives in the author's private working repo; this record carries the decision.

## Trigger and evidence

Scarebox ran at the HCA Halloween 5k on 2026-10-25 and the author wants it public at
`github.com/voronerd/scarebox`, as a showcase that a stranger or their coding agent can get running
on similar hardware. The private working repo had 131 commits. A scan of that history found no
committed secrets (the real camera password and page token appear zero times) and no images of
people, but it did find home and travel-router IP addresses, hardware MAC addresses, names of other
machines on the author's network, links to private AI-assistant sessions in commit trailers, and
personal names. All of that is fine in a private repo and wrong in a public one.

## Context

There are three ways to get a private repo public:

- **Push it as it is.** Leaks everything above.
- **Rewrite the history** with a filter tool to strip the patterns. A single missed commit stays
  public forever, and proving a 131-commit rewrite clean is hard.
- **Publish a fresh tree** with no shared history, built from the current files.

Separately, some values only make sense for one installation (the camera's hardware address, which
networks may open the control page without a token, the travel router's address). As long as they
live in tracked config files, every future edit can leak them again.

## Decision

1. **The public repo gets a fresh history.** Its first commit is a single export of the current tree;
   later updates are new commits made by re-running the export. The working repo's history is never
   pushed or rewritten for publication.
2. **Exports come from an allowlist.** `scripts/export_public.sh` copies only named paths (the
   package, tests, configs, deploy files, the line library, the voice reference clip, the synthetic
   sound effects, scripts and docs). Process files and evidence stay private by default.
3. **Every export passes a leak gate.** `scripts/leak_gate.py` refuses to publish if the tree
   contains a private-network address, a tailnet address, a MAC address, a machine name other than
   the example box `gbox1`, a private AI-session link, a personal name or email that shouldn't be
   public, or the literal value of any configured secret. It reports where, never what, and its own
   test proves each rule fires.
4. **Installation-specific values move out of tracked config** into a git-ignored local overlay, with
   an example file committed.

## Scope

**In scope:** what reaches `github.com/voronerd/scarebox` and how.

**Out of scope (explicit):** how the software works (ADR 0001 still governs that), and the author's
private working practices, which carry on in the working repo.

## Consequences

- Nothing in the public history can identify the author's network, because none of that history is
  shared.
- The private repo's story (bake-offs, tuning runs, the decisions log) isn't public. The public docs
  summarise what mattered instead (a design doc, a hardware guide, a race report).
- Publishing is one command plus a review, and the gate makes a careless edit fail loudly instead of
  leaking quietly.
- Contributions from outside land in the public repo and have to be brought back into the working
  repo by hand. Acceptable for a showcase; revisit if outside contributions become regular.

### Falsifiability anchor / When to revisit

- If anything private is ever found in the public repo, the gate missed a pattern: add the rule and
  its test before the next export.
- If the public repo becomes where development actually happens (regular outside contributions),
  make it the canonical repo and retire the export.

## Cross-references

- ADR 0001: the architecture this repo implements.
- `scripts/export_public.sh`, `scripts/leak_gate.py`: the mechanism.

## Sources

- History scan of the working repo, 2026-10-25 (counts only; no values printed).
