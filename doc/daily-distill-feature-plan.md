# Daily Distill Feature Plan

## Goal

Add a daily distillation job that converts session-scoped knowledge into durable shared persona knowledge baselines for:

- [agent_memory/executive/Knowledge.md](/Users/dinhtran/Projects/Qidyyy/edtronaunt-ai-co-worker/agent_memory/executive/Knowledge.md)
- [agent_memory/people/Knowledge.md](/Users/dinhtran/Projects/Qidyyy/edtronaunt-ai-co-worker/agent_memory/people/Knowledge.md)
- [agent_memory/operations/Knowledge.md](/Users/dinhtran/Projects/Qidyyy/edtronaunt-ai-co-worker/agent_memory/operations/Knowledge.md)

The purpose is to preserve durable cross-session learnings without letting raw per-session notes leak directly into every future conversation.

## Problem

The current system writes runtime knowledge to `agent_memory/sessions/<session_id>/<route>/Knowledge.md`. That isolates live sessions correctly, but it creates two gaps:

- useful insights learned in one session never become part of the shared persona baseline
- session knowledge will grow without pruning unless it is summarized, archived, or discarded

## Product Behavior

The daily distill job should:

1. Scan session knowledge files for each persona.
2. Select entries that are new since the last successful distill run.
3. Summarize them into durable persona-level notes.
4. Merge the summary into the shared persona knowledge files.
5. Record which source entries were processed.
6. Avoid duplicating the same insight across multiple runs.

The job should not:

- copy full transcripts into shared knowledge
- merge private or session-specific identifiers into shared knowledge
- overwrite `SOUL.md`
- rewrite the entire shared knowledge file on every run unless necessary

## Proposed Architecture

### Inputs

- `agent_memory/sessions/*/executive/Knowledge.md`
- `agent_memory/sessions/*/people/Knowledge.md`
- `agent_memory/sessions/*/operations/Knowledge.md`

### Outputs

- appended or refreshed summaries in the shared persona files:
  - `agent_memory/executive/Knowledge.md`
  - `agent_memory/people/Knowledge.md`
  - `agent_memory/operations/Knowledge.md`
- a machine-readable run ledger, for example:
  - `agent_memory/distill_state.json`

### New Runtime Pieces

- `scripts/daily_distill.py`
  - command-line entrypoint for the distillation job
- `my-app/coworker_engine/utils/distill.py`
  - parsing, dedupe, summarization, merge, and ledger helpers
- optional automation or cron entry
  - run once per day in the repo workspace

## Knowledge Model

Each shared persona `Knowledge.md` should gain a dedicated section for distilled knowledge, for example:

```md
## Distilled Cross-Session Knowledge

### 2026-03-20
- Pilot scope should stay narrow when adoption burden is uncertain.
- Manager enablement is a recurring dependency before regional rollout.
- Local staffing readiness is a gating factor for launch sequencing.
```

This keeps the shared baseline separate from manually written notes and task-journal content.

## Distillation Rules

Use strict inclusion rules. A fact should be promoted into shared knowledge only if it is:

- durable across sessions
- relevant to the persona’s role
- stated as a decision, constraint, recurring user need, or repeated operating pattern

Do not promote content that is:

- specific to one user session
- personally identifying
- a one-off draft artifact
- contradictory without enough evidence to resolve the conflict

## Deduplication Strategy

Apply dedupe in two stages:

1. Source-level dedupe
   - identify source journal entries already processed via a ledger key based on file path, timestamp heading, and content hash
2. Insight-level dedupe
   - compare candidate distilled bullets against existing shared bullets using normalized text similarity

This prevents both repeated processing and repeated phrasing in the shared files.

## Implementation Plan

### Phase 1: Parser and Ledger

- Parse session `Knowledge.md` files into structured entries.
- Extract heading timestamp, title, and bullet lines.
- Add `distill_state.json` to track processed entries and the last run timestamp.

### Phase 2: Heuristic Distill

- Start with a deterministic heuristic distiller before adding an LLM step.
- Keep entries whose titles and bullets indicate decisions, constraints, rollout dependencies, adoption patterns, or trade-offs.
- Collapse similar entries into concise bullets per persona.

### Phase 3: Shared File Merge

- Add or update a `## Distilled Cross-Session Knowledge` section in each shared persona file.
- Append a dated subsection for each run.
- Keep older distilled sections intact unless a cleanup rule is added later.

### Phase 4: Optional LLM Summarization

- After the heuristic pipeline is stable, optionally summarize candidate bullets with the configured LLM.
- Guard this step behind a flag so the job still runs offline or in test mode.

### Phase 5: Scheduling

- Run once per day through cron, a shell script, or Codex automation.
- Make the job idempotent so reruns are safe.

## Suggested CLI

```bash
python scripts/daily_distill.py
python scripts/daily_distill.py --dry-run
python scripts/daily_distill.py --persona executive
python scripts/daily_distill.py --since 2026-03-20
```

## Testing Plan

Add tests for:

- parsing journal entries from session `Knowledge.md`
- ignoring already-processed entries
- persona-specific output paths
- no cross-persona contamination
- dedupe across repeated session entries
- merge behavior when the distilled section already exists
- dry-run mode producing no file writes

## Risks

- Low-quality summarization could pollute shared knowledge with weak or redundant insights.
- If the ledger is wrong, the job may either skip valid entries or duplicate them.
- Shared knowledge could become too large unless a later pruning or compaction step is added.

## Open Decisions

- Whether distilled bullets should be heuristic-only or LLM-assisted by default
- Whether the shared files should keep every daily batch or periodically compact older batches
- Whether supervisor session knowledge should also be distilled into a shared supervisor baseline
