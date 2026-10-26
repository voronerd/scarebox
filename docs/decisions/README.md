# Architectural Decision Records

One file per material decision about scarebox (an architectural choice, a scope cut, a
licence or publication policy), in MADR format. Each record says what was decided, why, what
it rules out, and when to revisit it. Two templates:

- **`0000-template.md`** (full, default): for material decisions such as architectural
  choices, scope foreclosures, methodology commitments and supersessions.
- **`0000-template-minimal.md`** (bare MADR): for small, single-decision records.

When in doubt, start with the full template and strip the optional sections.

## Index

- [`0000-template.md`](0000-template.md) — Full ADR template (recommended).
- [`0000-template-minimal.md`](0000-template-minimal.md) — Minimal MADR template, for trivial decisions.
- [`0001-offline-live-pipeline-with-vetted-library-fallback.md`](0001-offline-live-pipeline-with-vetted-library-fallback.md) — Offline live generation under a 3.5 s deadline, falling back to a human-vetted line library; permissive licences only; RTSP+CGI not ONVIF.
- [`0002-publish-a-fresh-export-through-a-leak-gate.md`](0002-publish-a-fresh-export-through-a-leak-gate.md) — The public repo gets a fresh history from an allowlisted export that must pass a leak gate every time; the private history is never pushed or rewritten.

Add a line here when a new record lands.

## Conventions

- **Numbering:** zero-padded, sequential (`0003-...`, `0004-...`). Never reuse a number, even
  for a superseded record.
- **Status:** `Accepted | Accepted — Parked | Superseded by NNNN | Partially superseded by NNNN | Deprecated`.
  When superseding, the later record points back to the earlier one; the earlier record's
  body stays as the historical record.
- **Filenames:** `NNNN-kebab-case-title.md`. The title inside the file uses the human form.
- **Citation:** cite by `ADR NNNN §M` (section anchor). Inline citations are preferred over
  footnotes.
- **Source links:** records reference shipped files and prior records by relative path, and
  external material (papers, vendor docs) by URL.
