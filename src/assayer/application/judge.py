"""The LLM judge (ADR-0010): a critic that never signs.

Protocol, in order:

1. Ask for :class:`JudgeResponse` JSON (reasoning, then findings, then verdict).
2. Unparseable reply: one repair re-ask quoting the validation error, then ``judge_unparseable``.
3. Self-contradictory reply (``fail`` without findings, ``pass`` with findings): one re-ask naming
   the contradiction, then ``judge_inconsistent``. Both are harness errors, excluded from rates.
4. Verify every finding's quote against the evidence and drop unverified ones; the final
   verdict is ``fail`` only if a verified finding remains.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from assayer.domain.errors import ErrorKind
from assayer.domain.evalspec import JudgeSpec
from assayer.domain.fixtures import Criterion
from assayer.domain.jsontext import parse_json_object
from assayer.domain.judging import (
    JudgeResponse,
    JudgeVerdict,
    contradiction,
    judge_error,
    settle_verdict,
)
from assayer.domain.messages import Message
from assayer.ports.sut import ChatModel

_RULES = """\
You are a strict evaluator. Grade the SUBJECT against each criterion below.

Rules:
- Work through every criterion in "reasoning" BEFORE deciding the verdict.
- A finding is a violation of one criterion. Each finding MUST include "quote": text copied
  character-for-character from the EVIDENCE that shows the violation. Do not paraphrase.
- If you cannot quote evidence for a suspected violation, it is not a finding.
- "verdict" is "fail" if and only if "findings" is non-empty; otherwise "pass".
- Reply with a single JSON object matching the schema, nothing else.
"""


def build_judge_prompt(spec: JudgeSpec | None, criteria: list[Criterion]) -> str:
    lines = [_RULES]
    if spec is not None and spec.instructions.strip():
        lines.append(spec.instructions.strip())
    lines.append("Criteria (use these ids as criterion_id):")
    lines.extend(
        f"- {c.id}: {c.text}"
        + (
            " (quote the SUBJECT/EXTRACTED OUTPUT, not the source)"
            if c.quote_from == "subject"
            else ""
        )
        for c in criteria
    )
    lines.append(f"\nJSON schema:\n{json.dumps(JudgeResponse.model_json_schema())}")
    return "\n".join(lines)


class LlmJudge:
    def __init__(self, spec: JudgeSpec | None = None) -> None:
        self._spec = spec
        self._min_chars = spec.min_quote_chars if spec else 8

    async def judge(
        self, criteria: list[Criterion], evidence: str, subject: str, llm: ChatModel
    ) -> JudgeVerdict:
        messages = [
            Message.system(build_judge_prompt(self._spec, criteria)),
            Message.user(
                f"EVIDENCE:\n{evidence}"
                + ("" if subject in evidence else f"\n\nSUBJECT:\n{subject}")
            ),
        ]
        schema = JudgeResponse.model_json_schema()
        known = {c.id for c in criteria}
        reasks = 0
        repaired = reconsidered = False

        while True:
            reply = await llm.chat(messages, response_schema=schema)
            raw = reply.message.content or ""
            try:
                response = JudgeResponse.model_validate(parse_json_object(raw))
            except (ValueError, ValidationError) as exc:
                if repaired:
                    return judge_error(ErrorKind.JUDGE_UNPARSEABLE, str(exc)[:500], reasks=reasks)
                repaired, reasks = True, reasks + 1
                messages += [
                    Message.assistant(raw),
                    Message.user(
                        f"Your reply did not match the schema: {str(exc)[:300]}. "
                        "Reply again with only the JSON object."
                    ),
                ]
                continue

            problem = contradiction(response)
            if problem is not None:
                if reconsidered:
                    return judge_error(ErrorKind.JUDGE_INCONSISTENT, problem, reasks=reasks)
                reconsidered, reasks = True, reasks + 1
                messages += [
                    Message.assistant(raw),
                    Message.user(
                        f"Your answer is inconsistent: {problem}. Re-examine the criteria and "
                        "reply again with a consistent JSON object."
                    ),
                ]
                continue

            return settle_verdict(
                response,
                evidence,
                reasks=reasks,
                known_criteria=known,
                min_chars=self._min_chars,
                sources={c.id: subject for c in criteria if c.quote_from == "subject"},
            )
