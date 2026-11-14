"""Static HTML reports with Jinja2 (ADR-0020): one self-contained file, no JavaScript.

This is presentation, like the terminal renderer: it turns application view models into HTML,
so it lives with the CLI (outermost layer) rather than among the adapters.

A static file opens anywhere, diffs in a pull request, and can be published to GitHub Pages
as-is. Interactivity is limited to native ``<details>`` disclosure.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from assayer.application.report import DiffReport, RunReport
from assayer.domain.money import Money


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _usd(value: Money | None) -> str:
    return "n/a" if value is None else f"${value.to_usd():.4f}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f} ms"


class JinjaReportRenderer:
    def __init__(self) -> None:
        self._env = Environment(
            loader=PackageLoader("assayer.cli", "templates"),
            autoescape=select_autoescape(["html", "j2"]),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        self._env.filters.update(pct=_pct, usd=_usd, ms=_ms, money=lambda m: _usd(Money(m)))

    def render_run(self, report: RunReport) -> str:
        return self._env.get_template("run.html.j2").render(r=report)

    def render_diff(self, report: DiffReport) -> str:
        return self._env.get_template("diff.html.j2").render(r=report)


def write(html: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8", newline="\n")
    return path
