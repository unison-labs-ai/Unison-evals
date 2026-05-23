"""LLM-as-judge grading using Letta's published rubric.

Mirrors Letta's own RubricGrader (vendor/letta-evals/letta_evals/graders/rubric.py)
so our cell is directly comparable to their leaderboard:

  - Same rubric template, substituted by string.Formatter (their util)
  - Same JSON-schema response format {score: float[0,1], rationale: str}
  - Same temperature coercion: 0.0 → 1.0 for gpt-5*/o1*/o3*/o4* families
    (OpenAI's API rejects temperature=0 on those; Letta's grader bumps
    it to 1.0 server-side, see their rubric.py:134-137)
  - Same post-parse clamp on score to [0.0, 1.0]
  - Same provider auto-routing (OpenAI / Anthropic / Google)

We diverge in one place: we expose a single `grade()` function that
returns (score, raw_json_str) so the runner can persist the rationale
alongside the per-row JSON without parsing it twice. The score is
identical to what Letta's grader would emit on the same row.
"""

from __future__ import annotations

import json
import os
import string
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
RUBRIC_PATH = (
    _REPO_ROOT / "vendor" / "letta-evals" / "letta-leaderboard" / "filesystem-agent" / "rubric.txt"
)

# Locked to Letta's leaderboard configuration. Changing breaks
# apples-to-apples comparison with leaderboard.letta.com.
DEFAULT_JUDGE_MODEL = "gpt-5-mini"


def _load_rubric_template() -> str:
    if not RUBRIC_PATH.exists():
        raise FileNotFoundError(
            f"Rubric not vendored. Expected at {RUBRIC_PATH}. "
            "Run `git submodule update --init vendor/letta-evals`."
        )
    return RUBRIC_PATH.read_text()


_TEMPLATE: str | None = None


def _template() -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _load_rubric_template()
    return _TEMPLATE


def derive_provider(model: str) -> str:
    """Map a judge model name → provider. Fail loudly on unknown."""
    m = model.lower()
    if m.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "google"
    raise ValueError(
        f"Unknown judge model {model!r}. Set --judge-model to a "
        "gpt-*, claude-*, or gemini-* model, or unset $JUDGE_MODEL "
        "to fall back to the leaderboard default (gpt-5-mini)."
    )


# Letta's _JudgeResponse schema, ported verbatim. Sent to all three
# providers as a structured-output schema so the judge can't return
# free-form text (which our prior regex parser couldn't always handle).
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Score between 0.0 and 1.0",
        },
        "rationale": {
            "type": "string",
            "description": "Explanation of the grading decision",
        },
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}


def _build_prompt(question: str, ground_truth: str, submission: str) -> str:
    """Mirror Letta's build_judge_prompt — string.Formatter substitution."""
    substitutions = {
        "input": question,
        "ground_truth": ground_truth,
        "submission": submission,
    }
    return string.Formatter().vformat(_template(), (), substitutions)


def _coerce_temperature(model: str, base: float) -> float:
    """Match Letta's rubric.py:134-137 — OpenAI's gpt-5*/o1*/o3*/o4*
    families reject temperature != 1.0. Their grader silently bumps
    0.0 → 1.0 for those; we do the same so the cell matches theirs."""
    m = model.lower()
    if base == 0.0 and (m.startswith(("o1", "o3", "o4")) or "gpt-5" in m):
        return 1.0
    return base


def _grade_openai(prompt: str, model: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    temperature = _coerce_temperature(model, 0.0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "JudgeResponse",
                "schema": _JUDGE_SCHEMA,
                "strict": True,
            },
        },
    )
    return json.loads(resp.choices[0].message.content or "{}")


def _grade_anthropic(prompt: str, model: str) -> dict:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # Anthropic doesn't have an OpenAI-style response_format yet, so we
    # use tool-use to force structured output — same effect, equivalent
    # to what Letta does via their transform_schema path.
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "name": "submit_score",
                "description": "Submit the judged score and rationale per the rubric.",
                "input_schema": _JUDGE_SCHEMA,
            }
        ],
        tool_choice={"type": "tool", "name": "submit_score"},
    )
    for block in resp.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "submit_score":
            return dict(block.input)
    return {}


def grade(
    question: str,
    ground_truth: str,
    submission: str,
    model: str | None = None,
) -> tuple[float, str]:
    """Run the judge on one row.

    Returns (score, raw_json_str). Score is clamped to [0.0, 1.0] per
    Letta's grader. The raw JSON contains both score and rationale and
    is persisted alongside the per-row JSON for audit."""
    judge_model = model or DEFAULT_JUDGE_MODEL
    provider = derive_provider(judge_model)

    prompt = _build_prompt(question, ground_truth, submission)

    if provider == "openai":
        parsed = _grade_openai(prompt, judge_model)
    elif provider == "anthropic":
        parsed = _grade_anthropic(prompt, judge_model)
    else:
        raise ValueError(f"Judge provider {provider!r} not yet wired (model={judge_model!r}).")

    raw_score = float(parsed.get("score", 0.0))
    # Same clamp Letta applies — not every provider honours the schema bounds.
    score = max(0.0, min(1.0, raw_score))
    return score, json.dumps(parsed, ensure_ascii=False)
