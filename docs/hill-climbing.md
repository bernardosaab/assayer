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

## 6. Keep it cheap to go round again

* **Safe to re-run.** Same run id resumes; finished work is never redone; a budget caps every run.
* **Cheap to extend.** A scenario is a YAML file; a criterion is one line.
* **Observable.** Every run records git sha and content hashes of the prompt, rubric and graders.
