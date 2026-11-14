# 0010. Judge protocol: reasoning first, bounded re-asks, verdict derived from verified findings

- Status: accepted
- Date: 2026-11-13

## Context and problem statement

Some criteria cannot be settled by code. "The current role is reported, not a previous one" and
"no fabricated values" are examples. For those we use an LLM judge. Judges are unreliable
instruments: they pick a verdict first and rationalise it afterwards, emit unparseable JSON,
contradict themselves ("fail" with no findings), and invent quotes. A judge verdict therefore
has to be checked like any other measurement.

## Considered options

1. A single verdict field (`pass`/`fail`), trusted as returned.
2. A 1-10 score with a threshold.
3. Structured findings, each with a verbatim quote, and the final verdict computed in code from
   the findings that verify.

## Decision

Option 3. The protocol is in `domain/judging.py` (pure) and `application/judge.py:LlmJudge`.

1. **Reasoning before verdict.** `JudgeResponse` declares `reasoning`, then `findings`, then
   `verdict`. The JSON schema keeps that order, so a structured-output model writes its
   reasoning before it commits. The prompt (`_RULES`) says that a finding must quote the
   EVIDENCE character for character, that a suspected violation without a quote is not a
   finding, and that the verdict is `fail` if and only if there are findings.
2. **Unparseable reply:** one repair re-ask, quoting the validation error (truncated to 300
   characters). A second failure returns `judge_error(JUDGE_UNPARSEABLE)`.
3. **Self-contradiction** (`contradiction()`: `fail` without findings, or `pass` with
   findings): one re-ask naming the problem. A second contradiction returns
   `judge_error(JUDGE_INCONSISTENT)`. The two re-asks are budgeted independently, so a judgement
   costs at most three calls, and `reasks` is recorded.
4. **Verification.** `settle_verdict` keeps a finding only if `verify_quote` accepts its quote
   (ADR-0011) and its `criterion_id` is one of the criteria asked about.
   `verdict_final = "fail"` if and only if a verified finding remains. The model's own word is
   stored as `verdict_raw` and is never used for grading.

A judge harness error grades the attempt `harness_error` (ADR-0008), not pass or fail. The
evidence is the source document plus the extracted output for task evals, and the rendered
transcript (including tool calls and results) for conversations (`scoring.judge_inputs`).
`RunAnalytics.judge` reports re-asks, findings, dropped findings, drop rate, and
`raw_final_disagreements` (the judge said fail, but no finding survived). The judge is itself
calibrated against blind human labels (`application/calibration.py`: TPR, TNR, Cohen's kappa,
and the Rogan-Gladen corrected pass rate).

The judge is a critic and never the signer: its verdict word is advice, and the verdict that
counts is computed by code from evidence code has checked.

## Consequences

**Positive:** an invented quote cannot fail an attempt. Malformed or inconsistent judges are
measured (re-asks, drop rate) instead of silently trusted. The judge's error rate against
people is a number we can report.

**Negative / trade-offs:**

- The protocol is biased toward `pass` for violations of *omission*. "The agent never confirmed
  the dates" has nothing to quote, so the finding is dropped. Such criteria belong in
  deterministic checks.
- Re-asks raise judge cost by up to 3x per judgement, and the limit of one re-ask each is
  arbitrary.
- A quote only has to occur somewhere in the evidence. For extraction, that includes the
  model's own output, so verification shows the text exists, not that it supports the
  criterion.
