# Evals as a loop: climbing the hill

An eval is not a gate you pass once. Its job is to find the next thing to fix. You run it, read the
failures, find the reason for each, fix it, and run again. Each pass is a small step up the hill, and the fix
can land on either side: on the **agent** or on the **harness**.

## 1. Run with repeats

```bash
uv run assayer run hr_assistant --run-id v1 --repeats 3
```

Models vary between calls, so one attempt per case says little about reliability. Look at both:

* **pass^k**: share of cases that pass *every* repeat. This is the reliability users feel.
* **pass@k**: share that pass *at least once*. This is capability: "can it do this at all?"

A gap between them means the agent can do the task but not dependably. That calls for a different fix
(tighter tool contracts, a deterministic step) from a case that never passes.

## 2. Error analysis: read before you count

```bash
uv run assayer failures v1
uv run assayer summary v1      # failure causes, by check and by judge criterion
```

1. **Read** each failing attempt in the HTML report (`assayer report v1`): the transcript, the tool calls, the
   checks that failed and the judge's quoted findings.
2. **Write one sentence** about the *first* thing that went wrong, not every symptom.
3. **Group** those sentences into causes: wrong tool, wrong arguments, missing context, policy breach,
   wasted steps, false claim of success.
4. **Count** them and fix the largest cluster first. The "failure causes" table is the mechanical
   starting point; your notes are the real one.

Keep reading until new failures stop producing new categories.

## 3. Fix the agent or the harness

| Symptom | Fix on the agent side | Fix on the harness side |
|---|---|---|
| Misreads an instruction | A prompt rule | |
| Calls the wrong tool or gets too little back | A tool change | |
| Its input was wrong | A data fix | |
| A failure seen in production | | A new scenario file (a permanent guard) |
| A verdict you disagree with | | A sharper check, a rubric line, or relabelled gold |
| The judge drops many findings | | Calibrate it: `assayer calibrate sample/import/stats` |

When a control matters, prefer moving it into code over adding a prompt line. A prompt is advice; a check or
a tool that refuses is a control. The HR environment enforces permissions in the tool, not in the prompt.

## 4. Did it work? Compare paired, not in aggregate

```bash
uv run assayer run hr_assistant --run-id v2 --repeats 3
uv run assayer diff v1 v2 --fail-on-drop 0.05 --html reports/v1-v2.html
```

`diff` pairs the runs on the same `(case, repeat)`, lists exactly which cases flipped, and runs an exact
McNemar test on the discordant pairs. It also prints the smallest change the sample can resolve: with a
handful of cases, only large changes are distinguishable from noise. Add cases before trusting small
movements.

## 5. Did the agent change, or did the grading?

```bash
uv run assayer replay v1 --into v1-regraded
uv run assayer diff v1 v1-regraded
```

Replay re-scores v1's stored outputs with the current checks and rubric, with **no calls to the agent**.
Judgements of identical evidence under an unchanged judge are reused from the cache. If the numbers move
under replay, the grading moved, not the agent.

## Case study: a real local model (measured)

Measured on 2026-11-14 with Ollama on a laptop (RTX 3070 Laptop GPU): agent `llama3.2:3b`,
judge `qwen2.5:7b` (a different family, ADR-0012), one repeat per case. Six cases cannot
resolve small changes, and the diffs say so; read this as a demonstration of the loop, not as
a benchmark. Reports for these runs are on the demo site.

**`resume_extraction`**

| Step | What changed | Pass rate | What the harness showed |
|---|---|---|---|
| v1 | baseline | 0/6 | One failure cluster: every field right except `years_experience`, computed from the *current* role only. |
| v2 | prompt rule with a worked example | 0/6 | The model now copied the reference year from the prompt: `years_experience = 2026` for every case. A prompt is advice. |
| v3 | **numbers come from code**: the model extracts `first_job_start_year` (written in the document) and the runner derives the years (`derive` in `eval.yaml`) | 1/6 | All deterministic checks pass. The remaining failures all came from the judge. |
| v3, regraded | **judge sensor**: `current_role` findings must quote the *output* (`quote_from: subject`), because the 7B judge had "proved" violations by quoting correct lines of the résumé | 5/6 | `assayer replay --rejudge`: zero agent calls; paired diff 4 improved, 0 regressed (McNemar p = 0.125, not significant at n = 6). |

The last regrade changed only the grader, and replay made that visible: the agent's outputs were
identical, the numbers moved, and the diff attributes every flip to grading. The remaining failure
is a `no_fabrication` finding that looks like another judge false positive; that is what
`assayer calibrate` is for, and it needs human labels, not a stronger prompt.

**`hr_assistant`** (8 conversations, about 25 minutes per run on this laptop)

| Step | What changed | Graded | Trajectory sensors |
|---|---|---|---|
| v1 | baseline | 0/8 | 2 wrong-year dates, 4 permission errors (guessed employee ids), 2 duplicate requests after "fingers crossed" |
| v2 | environment `briefing()` gives identity and today's date; `submit_pto_request` made idempotent | 0/7 (+1 judge harness error) | 0, 0 and 0 |
| v3 | the same agent re-run on v0.2.1 (stricter environment after the code reviews) | 0/8 | unchanged: the remaining failures are model capability |

The end-to-end rate did not move, and the diff says so. The tool-level errors it targeted went to zero.
The remaining failures are the 3B agent acting before confirming and repeating calls, and a 3B
user simulator that rarely ends a conversation (turn-limit endings). The next steps would be a
stronger agent model, a stronger simulator, or turning "quote the change, then wait for a yes" into
a gate in the tool rather than an instruction.

## 6. Keep it cheap to go round again

* **Safe to re-run.** Same run id resumes; finished work is never redone; a budget caps every run.
* **Cheap to extend.** A scenario is a YAML file; a criterion is one line.
* **Observable.** Every run records git sha and content hashes of the prompt, rubric and graders.
