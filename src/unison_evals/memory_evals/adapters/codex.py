"""OpenAI Codex CLI adapter — subprocess to `codex exec`.

Codex CLI has no persistent memory layer, so Track 3 and Track 2 behave
the same: we always pass any oracle context inside the prompt. This is the
honest comparison — Codex is a general-purpose coding agent without a brain,
so persistent-memory benchmarks should expose that gap.

CLI invocation: `codex exec --json "<prompt>"`

With `--json`, Codex streams JSONL events to stdout; without it, only the
final agent message goes to stdout. We use `--json` so we can parse usage
stats and fall back to plain-text mode if the flag is unavailable on older
installs. Pricing is a placeholder — Codex CLI delegates to whichever
OpenAI model is configured by the user (default: codex-1 / o4-mini tier).
We use GPT-5 pricing as a placeholder until Codex publishes per-token costs.

Install: npm install -g @openai/codex
Docs:    https://github.com/openai/codex
"""

from __future__ import annotations

import asyncio
import json
import time
from shutil import which

from loguru import logger

from ...config import get_settings
from ...types import AdapterResult, Document
from .base import AgentAdapter

# Pricing placeholder: GPT-5 rates as of 2026-05 (per million tokens).
# Codex CLI routes to whichever model the user has configured; update
# these constants once OpenAI publishes Codex-specific per-token pricing.
CODEX_INPUT_USD_PER_MTOK = 5.0
CODEX_OUTPUT_USD_PER_MTOK = 15.0


class CodexAdapter(AgentAdapter):
    name = "codex"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._codex_bin: str | None = None

    async def setup(self) -> None:
        bin_path = which("codex")
        if bin_path is None:
            raise RuntimeError(
                "`codex` binary not found on PATH. Install OpenAI Codex CLI: "
                "https://github.com/openai/codex"
            )
        self._codex_bin = bin_path
        # Smoke test the CLI exists and is auth'd by checking version.
        proc = await asyncio.create_subprocess_exec(
            bin_path,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"`codex --version` failed: {stderr.decode()[:300]}. "
                "Make sure Codex CLI is installed and authenticated."
            )
        logger.debug("codex adapter ready", version=stdout.decode().strip())

    async def answer(
        self,
        question: str,
        oracle_context: str | None = None,
        seed_docs: list[Document] | None = None,
    ) -> AdapterResult:
        assert self._codex_bin is not None, "setup() must be called first"

        if oracle_context is not None and seed_docs is not None:
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=0.0,
                raw={},
                error="seed_docs and oracle_context are mutually exclusive",
            )

        # Codex CLI has no persistent memory layer. For Track 3 (seed_docs),
        # we inline the docs into the prompt, concatenated similarly to oracle_context.
        effective_context: str | None = oracle_context
        if seed_docs is not None:
            effective_context = _format_seed_docs(seed_docs)

        inlined_chars = len(effective_context) if effective_context else 0
        prompt = _build_prompt(question, effective_context)
        start = time.perf_counter()

        try:
            proc = await asyncio.create_subprocess_exec(
                self._codex_bin,
                "exec",
                "--json",  # JSONL event stream on stdout
                "--ephemeral",  # do not persist session rollout files
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.settings.adapter_timeout,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return AdapterResult(
                    answer="",
                    cost_usd=0.0,
                    latency_ms=elapsed_ms,
                    raw={},
                    error=f"timeout after {self.settings.adapter_timeout}s",
                )

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if proc.returncode != 0:
                return AdapterResult(
                    answer="",
                    cost_usd=0.0,
                    latency_ms=elapsed_ms,
                    raw={"stderr": stderr.decode()[:1000]},
                    error=f"exit {proc.returncode}: {stderr.decode()[:200]}",
                )

            answer, usage = _parse_output(stdout.decode())
            cost = _estimate_cost(usage, prompt, answer)
            input_tok = usage.get("input_tokens") or len(prompt) // 4
            output_tok = usage.get("output_tokens") or len(answer) // 4
            tokens_real = bool(usage.get("input_tokens"))
            return AdapterResult(
                answer=answer,
                cost_usd=cost,
                latency_ms=elapsed_ms,
                input_tokens=input_tok,
                output_tokens=output_tok,
                tokens_unavailable=not tokens_real,
                raw={
                    "stdout": stdout.decode()[:5000],
                    "usage": usage,
                    "inlined_chars": inlined_chars,
                    "inlined_tokens": input_tok,
                },
            )

        except (OSError, ValueError) as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=elapsed_ms,
                raw={},
                error=f"subprocess error: {e}",
            )


def _build_prompt(question: str, oracle_context: str | None) -> str:
    if oracle_context is None:
        return question
    return (
        "Use the following context to answer the question. Do not use any "
        "other knowledge or tools.\n\n"
        f"<context>\n{oracle_context}\n</context>\n\n"
        f"Question: {question}"
    )


def _format_seed_docs(seed_docs: list[Document]) -> str:
    return "\n\n".join(f"[Document: {doc.path}]\n{doc.body}" for doc in seed_docs)


def _parse_output(stdout: str) -> tuple[str, dict[str, int]]:
    """Best-effort parse of `codex exec --json` JSONL output.

    With --json, Codex emits one JSON object per line (JSONL). The final
    agent message lives in an event of type 'item.completed' with an
    'assistant_message' item, or in 'turn.completed'. Without --json (or on
    older CLI versions that ignore the flag), stdout contains only the plain
    final message.

    We scan lines in reverse to find the last assistant message event; if
    JSON parsing fails entirely we return the raw text as the answer.
    """
    text = stdout.strip()
    if not text:
        return "", {}

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Try JSONL mode: scan for assistant message events.
    usage: dict[str, int] = {}
    answer_text: str | None = None

    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(event, dict):
            continue

        event_type = event.get("type", "")

        # turn.completed carries token usage in some CLI versions.
        if event_type == "turn.completed" and "usage" in event:
            u = event["usage"]
            if isinstance(u, dict):
                usage = {
                    "input_tokens": u.get("input_tokens", 0),
                    "output_tokens": u.get("output_tokens", 0),
                }

        # item.completed with assistant_message content = final answer.
        if event_type == "item.completed" and answer_text is None:
            item = event.get("item", {})
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content", "")
                if isinstance(content, str) and content:
                    answer_text = content
                elif isinstance(content, list):
                    # content blocks: [{type: text, text: "..."}]
                    parts = [
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ]
                    candidate = " ".join(parts).strip()
                    if candidate:
                        answer_text = candidate

    if answer_text is not None:
        return answer_text, usage

    # Fallback 1: single JSON object (non-JSONL response)
    if lines:
        try:
            data = json.loads(lines[-1])
            if isinstance(data, dict):
                candidate = str(data.get("result") or data.get("answer") or data.get("text") or "")
                if candidate:
                    return candidate, data.get("usage", {}) if isinstance(
                        data.get("usage"), dict
                    ) else {}
        except json.JSONDecodeError:
            pass

    # Fallback 2: no JSON at all — return last non-empty line as the answer.
    for line in reversed(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError:
            return line, {}

    # Everything was valid JSON but we couldn't extract an answer — return raw.
    return text, {}


def _estimate_cost(usage: dict[str, int], prompt: str, answer: str) -> float:
    """Estimate USD cost. Uses real usage if available, else token-count
    approximation (1 token ≈ 4 chars)."""
    input_tokens = usage.get("input_tokens") or len(prompt) // 4
    output_tokens = usage.get("output_tokens") or len(answer) // 4
    return (
        input_tokens * CODEX_INPUT_USD_PER_MTOK + output_tokens * CODEX_OUTPUT_USD_PER_MTOK
    ) / 1_000_000.0
