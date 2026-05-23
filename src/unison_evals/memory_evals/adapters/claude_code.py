"""Claude Code adapter — subprocess to `claude --print`.

Claude Code has no persistent memory layer, so Track 3 and Track 2 behave
the same: we always pass any oracle context inside the prompt. This is the
honest comparison — Claude Code is a general-purpose agent without a brain,
so persistent-memory benchmarks should expose that gap.

Cost is estimated from token counts since `claude --print` doesn't return
structured cost data. We use Anthropic API pricing for sonnet-4-5.
"""

from __future__ import annotations

import asyncio
import time
from shutil import which

from loguru import logger

from ...config import get_settings
from ...types import AdapterResult, Document
from .base import AgentAdapter

# Claude 4.5 Sonnet pricing as of 2026-05 (per million tokens)
# Update when pricing changes — affects $/task numbers in the leaderboard.
SONNET_INPUT_USD_PER_MTOK = 3.0
SONNET_OUTPUT_USD_PER_MTOK = 15.0


class ClaudeCodeAdapter(AgentAdapter):
    name = "claude-code"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._claude_bin: str | None = None

    async def setup(self) -> None:
        bin_path = which("claude")
        if bin_path is None:
            raise RuntimeError(
                "`claude` binary not found on PATH. Install Claude Code: "
                "https://claude.com/claude-code"
            )
        self._claude_bin = bin_path
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
                f"`claude --version` failed: {stderr.decode()[:300]}. "
                "Make sure Claude Code is installed and authenticated."
            )
        logger.debug("claude-code adapter ready", version=stdout.decode().strip())

    async def answer(
        self,
        question: str,
        oracle_context: str | None = None,
        seed_docs: list[Document] | None = None,
    ) -> AdapterResult:
        assert self._claude_bin is not None, "setup() must be called first"

        if oracle_context is not None and seed_docs is not None:
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=0.0,
                raw={},
                error="seed_docs and oracle_context are mutually exclusive",
            )

        # Claude Code has no persistent memory layer. For Track 3 (seed_docs),
        # we inline the docs into the prompt, concatenated similarly to oracle_context.
        effective_context: str | None = oracle_context
        if seed_docs is not None:
            effective_context = "\n\n".join(
                f"[Document: {doc.path}]\n{doc.body}" for doc in seed_docs
            )

        inlined_chars = len(effective_context) if effective_context else 0
        prompt = self._build_prompt(question, effective_context)
        start = time.perf_counter()

        # Run Claude Code in --bare mode for a clean, reproducible eval.
        # --bare:
        #   * skips hooks, LSP, plugin sync, attribution, auto-memory
        #   * skips background prefetches + keychain reads
        #   * skips CLAUDE.md auto-discovery
        #   * auth is STRICTLY ANTHROPIC_API_KEY (no OAuth, no keychain)
        # Plus we explicitly disable all tools and pin a clean cwd so no
        # local Notion/Linear/etc. MCP can attach.
        #
        # Critical: we must pass ANTHROPIC_API_KEY into the subprocess env
        # since pydantic-settings reads it into Settings, not os.environ.
        import os
        import tempfile

        if not self.settings.anthropic_api_key:
            return AdapterResult(
                answer="",
                cost_usd=0.0,
                latency_ms=0.0,
                raw={},
                error="ANTHROPIC_API_KEY not set — required for claude-code in --bare mode",
            )

        env = {
            **os.environ,
            "ANTHROPIC_API_KEY": self.settings.anthropic_api_key,
        }

        with tempfile.TemporaryDirectory(prefix="unison-evals-claude-") as tmp_cwd:
            try:
                proc = await asyncio.create_subprocess_exec(
                    self._claude_bin,
                    "--bare",  # minimal mode; auth = ANTHROPIC_API_KEY only
                    "--print",  # headless: prints final answer + exits
                    "--output-format",
                    "json",
                    "--tools",
                    "",  # disable all tools (Track 2/3 don't need agent tool use)
                    "--no-session-persistence",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tmp_cwd,
                    env=env,
                )
                return await self._communicate_and_parse(proc, prompt, start, inlined_chars)
            except (OSError, ValueError) as e:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return AdapterResult(
                    answer="",
                    cost_usd=0.0,
                    latency_ms=elapsed_ms,
                    raw={},
                    error=f"subprocess error: {e}",
                )

    async def _communicate_and_parse(
        self,
        proc: asyncio.subprocess.Process,
        prompt: str,
        start: float,
        inlined_chars: int = 0,
    ) -> AdapterResult:
        try:
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=prompt.encode()),
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

            answer, usage = self._parse_output(stdout.decode())
            cost = self._estimate_cost(usage, prompt, answer)
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

    @staticmethod
    def _build_prompt(question: str, oracle_context: str | None) -> str:
        if oracle_context is None:
            return question
        return (
            "Use the following context to answer the question. Do not use any "
            "other knowledge or tools.\n\n"
            f"<context>\n{oracle_context}\n</context>\n\n"
            f"Question: {question}"
        )

    @staticmethod
    def _parse_output(stdout: str) -> tuple[str, dict[str, int]]:
        """Best-effort parse of claude --print --output-format json.

        Output format varies by CLI version; fall back to raw stdout if JSON
        parse fails.
        """
        import json

        text = stdout.strip()
        try:
            data = json.loads(text)
            # Modern format: {"result": "...", "usage": {"input_tokens": N, ...}}
            if isinstance(data, dict):
                answer = str(data.get("result") or data.get("answer") or data.get("text") or text)
                usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
                return answer, usage
        except json.JSONDecodeError:
            pass
        return text, {}

    @staticmethod
    def _estimate_cost(usage: dict[str, int], prompt: str, answer: str) -> float:
        """Estimate USD cost. Uses real usage if available, else token-count
        approximation (1 token ≈ 4 chars)."""
        input_tokens = usage.get("input_tokens") or len(prompt) // 4
        output_tokens = usage.get("output_tokens") or len(answer) // 4
        return (
            input_tokens * SONNET_INPUT_USD_PER_MTOK + output_tokens * SONNET_OUTPUT_USD_PER_MTOK
        ) / 1_000_000.0
