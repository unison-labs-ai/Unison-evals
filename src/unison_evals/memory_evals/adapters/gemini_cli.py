"""Google Gemini CLI adapter — subprocess to `gemini -p`.

Gemini CLI has no persistent memory layer, so Track 3 and Track 2 behave
the same: we always pass any oracle context inside the prompt. This is the
honest comparison — Gemini CLI is a general-purpose agent without a brain,
so persistent-memory benchmarks should expose that gap.

CLI invocation: `gemini -p "<prompt>" --output-format json`

With `--output-format json`, the CLI returns a single JSON object:
  {"response": "<text>", "stats": {"input_token_count": N, ...}}

Headless mode is triggered by the `-p`/`--prompt` flag; the CLI does not
require a TTY when this flag is present.

Install: npm install -g @google/gemini-cli
Docs:    https://github.com/google-gemini/gemini-cli
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

# Gemini 2.5 Pro pricing as of 2026-05 (per million tokens).
# Update when pricing changes — affects $/task numbers in the leaderboard.
GEMINI_INPUT_USD_PER_MTOK = 1.25
GEMINI_OUTPUT_USD_PER_MTOK = 10.0


class GeminiCliAdapter(AgentAdapter):
    name = "gemini-cli"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._gemini_bin: str | None = None

    async def setup(self) -> None:
        bin_path = which("gemini")
        if bin_path is None:
            raise RuntimeError(
                "`gemini` binary not found on PATH. Install Google Gemini CLI: "
                "https://github.com/google-gemini/gemini-cli"
            )
        self._gemini_bin = bin_path
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
                f"`gemini --version` failed: {stderr.decode()[:300]}. "
                "Make sure Gemini CLI is installed and authenticated."
            )
        logger.debug("gemini-cli adapter ready", version=stdout.decode().strip())

    async def answer(
        self,
        question: str,
        oracle_context: str | None = None,
        seed_docs: list[Document] | None = None,
    ) -> AdapterResult:
        assert self._gemini_bin is not None, "setup() must be called first"

        if oracle_context is not None and seed_docs is not None:
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=0.0,
                raw={},
                error="seed_docs and oracle_context are mutually exclusive",
            )

        # Gemini CLI has no persistent memory layer. For Track 3 (seed_docs),
        # we inline the docs into the prompt, concatenated similarly to oracle_context.
        effective_context: str | None = oracle_context
        if seed_docs is not None:
            effective_context = _format_seed_docs(seed_docs)

        inlined_chars = len(effective_context) if effective_context else 0
        prompt = _build_prompt(question, effective_context)
        start = time.perf_counter()

        try:
            proc = await asyncio.create_subprocess_exec(
                self._gemini_bin,
                "-p",  # --prompt: non-interactive mode
                prompt,
                "--output-format",
                "json",  # single JSON object with response + stats
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
    """Best-effort parse of `gemini -p ... --output-format json` output.

    Expected JSON shape (from Gemini CLI headless docs):
      {
        "response": "<final answer text>",
        "stats": {
          "input_token_count": N,
          "output_token_count": N,
          ...
        }
      }

    Falls back to plain-text if JSON parse fails (older CLI versions or
    when --output-format is not recognised).
    """
    text = stdout.strip()
    if not text:
        return "", {}

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            answer = str(data.get("response") or data.get("result") or data.get("text") or text)
            stats = data.get("stats", {})
            usage: dict[str, int] = {}
            if isinstance(stats, dict):
                usage = {
                    "input_tokens": int(stats.get("input_token_count", 0)),
                    "output_tokens": int(stats.get("output_token_count", 0)),
                }
            return answer, usage
    except json.JSONDecodeError:
        pass

    # Plain-text fallback (no JSON) — return raw stdout as the answer.
    return text, {}


def _estimate_cost(usage: dict[str, int], prompt: str, answer: str) -> float:
    """Estimate USD cost. Uses real usage if available, else token-count
    approximation (1 token ≈ 4 chars)."""
    input_tokens = usage.get("input_tokens") or len(prompt) // 4
    output_tokens = usage.get("output_tokens") or len(answer) // 4
    return (
        input_tokens * GEMINI_INPUT_USD_PER_MTOK + output_tokens * GEMINI_OUTPUT_USD_PER_MTOK
    ) / 1_000_000.0
