# 0021. Run events through an observer port; the live view is just one listener

- Status: accepted
- Date: 2026-11-14

## Context and problem statement

A run against a local model takes minutes (one conversation eval took about 25 minutes on a laptop
GPU). People need to see what is happening while it happens: which attempt is running, what the
agent just did, which tool call failed, what the judge decided. The use cases must not depend on a
terminal library to make that possible.

## Considered options

1. Print from inside the use cases.
2. Parse the log output.
3. Typed events emitted through an observer port, rendered by the CLI.

## Decision

Option 3. `application/events.py` defines small frozen event types (`StageStarted`,
`AttemptStarted`, `AttemptFinished`, `TurnEvent`, `Judged`, `Graded`, `Halted`) and a
`RunObserver` protocol. `Executor`, `run_conversation` and `ScoreRun` emit them; `RunEval` passes
the observer through. Every emission goes through `safe_notify`, so an observer bug is logged and
never fails a run. `cli/live.py:LiveObserver` renders progress bars, per-attempt results and, with
`--trace`, every conversation turn and tool call. Tests use `RecordingObserver`.

## Consequences

**Positive:** the live view and any future listener (a web UI, a metrics exporter) need no change
to the application. The isolation was tested by accident on day one: a Windows cp1252 console could
not encode the view's glyphs, the observer failed, and the run carried on; the view now falls back
to ASCII glyphs when the console cannot encode Unicode.

**Negative / trade-offs:** events are fire-and-forget, so a listener that falls behind sees a burst
and nothing is replayable from them (the store remains the record). Concurrent attempts interleave
their turns in `--trace`; `--concurrency 1` gives a readable single stream.
