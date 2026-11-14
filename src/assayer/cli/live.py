"""Live terminal view of a run: stage progress bars, per-attempt results, and (with ``--trace``)
the conversation itself, turn by turn, including every tool call and tool error.

It is a :class:`RunObserver`; the application does not know it exists.
"""

from __future__ import annotations

from types import TracebackType

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from assayer.application.events import (
    AttemptFinished,
    AttemptStarted,
    Event,
    Graded,
    Halted,
    Judged,
    StageStarted,
    TurnEvent,
)

_LABEL = {"execute": "running attempts", "judge": "judging"}
_ACTOR_STYLE = {"user": "cyan", "agent": "green", "tool": "magenta"}


def _glyphs(console: Console) -> dict[str, str]:
    """Unicode glyphs only when the terminal can encode them (e.g. not a cp1252 Windows console)."""
    if (console.encoding or "").lower().replace("-", "").startswith("utf"):
        return {"bar": "│", "arrow": "→", "cross": "✗", "ellipsis": "…"}
    return {"bar": "|", "arrow": "->", "cross": "x", "ellipsis": "..."}


def _clip(text: str, limit: int = 220, ellipsis: str = "...") -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - len(ellipsis)] + ellipsis


class LiveObserver:
    def __init__(self, console: Console, *, trace: bool = False) -> None:
        self._console = console
        self._trace = trace
        self._g = _glyphs(console)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self._tasks: dict[str, TaskID] = {}
        self._running: set[str] = set()

    def __enter__(self) -> LiveObserver:
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._progress.stop()

    def _print(self, text: Text | str) -> None:
        self._progress.console.print(text)

    def _describe_running(self) -> None:
        task = self._tasks.get("execute")
        if task is None:
            return
        shown = ", ".join(sorted(self._running)[:3])
        more = f" +{len(self._running) - 3}" if len(self._running) > 3 else ""
        suffix = f" [dim]({shown}{more})[/]" if shown else ""
        self._progress.update(task, description=f"{_LABEL['execute']}{suffix}")

    def notify(self, event: Event) -> None:  # noqa: PLR0912 - one branch per event type
        if isinstance(event, StageStarted):
            if event.total and event.stage in _LABEL:
                self._tasks[event.stage] = self._progress.add_task(
                    _LABEL[event.stage], total=event.total
                )
        elif isinstance(event, AttemptStarted):
            self._running.add(f"{event.case_id}#{event.repeat}")
            self._describe_running()
        elif isinstance(event, AttemptFinished):
            self._running.discard(f"{event.case_id}#{event.repeat}")
            self._describe_running()
            if event.released:
                # Not an error and not done: it goes back to pending and runs on resume.
                self._print(
                    Text.assemble(
                        ("  not run ", "dim"),
                        f"{event.case_id}#{event.repeat}",
                        ("  (budget)", "dim"),
                    )
                )
                return
            if "execute" in self._tasks:
                self._progress.advance(self._tasks["execute"])
            took = f"{event.duration_ms / 1000:.1f}s" if event.duration_ms is not None else ""
            if event.ok:
                self._print(
                    Text.assemble(
                        ("  ran ", "dim"), f"{event.case_id}#{event.repeat}", (f"  {took}", "dim")
                    )
                )
            else:
                self._print(
                    Text.assemble(
                        ("  harness error ", "yellow"),
                        f"{event.case_id}#{event.repeat}  ",
                        (_clip(event.error or ""), "dim"),
                    )
                )
        elif isinstance(event, TurnEvent):
            if self._trace:
                style = "red" if event.error else _ACTOR_STYLE.get(event.actor, "white")
                arrow = self._g["arrow"] + " " if event.actor == "tool" else ""
                body = _clip(event.text) + (
                    f"  {self._g['cross']} {_clip(event.error, 120)}" if event.error else ""
                )
                self._print(
                    Text.assemble(
                        ("    " + self._g["bar"] + " ", "dim"),
                        (f"{event.case_id} t{event.turn} ", "dim"),
                        (f"{event.actor:>5} ", style),
                        arrow + body,
                    )
                )
        elif isinstance(event, Judged):
            if "judge" in self._tasks:
                self._progress.advance(self._tasks["judge"])
            notes = []
            if event.skipped:
                notes.append("skipped: not cached, use --rejudge")
            if event.cached:
                notes.append("cached")
            if event.reasks:
                notes.append(f"{event.reasks} re-ask")
            if event.dropped:
                notes.append(f"{event.dropped} unverified quote dropped")
            if notes:
                label = "skipped" if event.skipped else event.verdict or "error"
                self._print(
                    Text.assemble(
                        ("  judge ", "dim"),
                        f"{event.case_id}#{event.repeat} {label} ",
                        (f"({', '.join(notes)})", "dim"),
                    )
                )
        elif isinstance(event, Graded):
            style = {"pass": "green", "fail": "red"}.get(event.outcome, "yellow")
            reason = f"  {_clip(event.reasons[0], 120)}" if event.reasons else ""
            self._print(
                Text.assemble(
                    (f"  {event.outcome.upper():<13}", f"bold {style}"),
                    f"{event.case_id}#{event.repeat}",
                    (reason, "dim"),
                )
            )
        elif isinstance(event, Halted):
            self._print(Text(f"  halted: {event.reason}", style="yellow"))
