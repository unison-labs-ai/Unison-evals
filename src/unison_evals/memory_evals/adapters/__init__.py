"""Adapters — one per system.

Two contracts:
  * AgentAdapter (Tracks 2 + 3) — answers questions
  * BrainAdapter (Track 1) — ingests docs and returns ranked chunks

Adding a new system: subclass the appropriate base, implement the abstract
methods, register in the matching REGISTRY.
"""

from .anthropic_raw import AnthropicRawAdapter
from .base import AgentAdapter, BrainAdapter
from .claude_code import ClaudeCodeAdapter
from .codex import CodexAdapter
from .gemini_cli import GeminiCliAdapter
from .google_gemini import GoogleGeminiAdapter
from .letta import LettaBrainAdapter
from .mem0 import Mem0AgentAdapter, Mem0BrainAdapter
from .openai_gpt5 import OpenAIGpt5Adapter
from .pgvector_naive import PgvectorNaiveBrainAdapter
from .unison_agent import UnisonAgentAdapter, UnisonAgentPipelineAdapter
from .unison_brain import UnisonBrainAdapter
from .zep import ZepBrainAdapter

REGISTRY: dict[str, type[AgentAdapter]] = {
    "unison-agent": UnisonAgentAdapter,
    "unison-agent-pipeline": UnisonAgentPipelineAdapter,
    "claude-code": ClaudeCodeAdapter,
    "codex": CodexAdapter,
    "gemini-cli": GeminiCliAdapter,
    "mem0-agent": Mem0AgentAdapter,
    "anthropic-raw": AnthropicRawAdapter,
    "openai-gpt5": OpenAIGpt5Adapter,
    "google-gemini": GoogleGeminiAdapter,
}

BRAIN_REGISTRY: dict[str, type[BrainAdapter]] = {
    "pgvector-naive": PgvectorNaiveBrainAdapter,
    "unison-brain": UnisonBrainAdapter,
    "mem0": Mem0BrainAdapter,
    "letta": LettaBrainAdapter,
    "zep": ZepBrainAdapter,
}


def get_adapter(name: str) -> AgentAdapter:
    """Factory — returns an instantiated AgentAdapter by registry name."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown agent adapter '{name}'. Available: {available}")
    return REGISTRY[name]()


def get_brain_adapter(name: str) -> BrainAdapter:
    """Factory — returns an instantiated BrainAdapter by registry name."""
    if name not in BRAIN_REGISTRY:
        available = ", ".join(sorted(BRAIN_REGISTRY)) or "(none registered yet)"
        raise KeyError(f"Unknown brain adapter '{name}'. Available: {available}")
    return BRAIN_REGISTRY[name]()


__all__ = [
    "BRAIN_REGISTRY",
    "REGISTRY",
    "AgentAdapter",
    "AnthropicRawAdapter",
    "BrainAdapter",
    "ClaudeCodeAdapter",
    "CodexAdapter",
    "GeminiCliAdapter",
    "GoogleGeminiAdapter",
    "LettaBrainAdapter",
    "Mem0AgentAdapter",
    "Mem0BrainAdapter",
    "OpenAIGpt5Adapter",
    "PgvectorNaiveBrainAdapter",
    "UnisonAgentAdapter",
    "UnisonAgentPipelineAdapter",
    "UnisonBrainAdapter",
    "ZepBrainAdapter",
    "get_adapter",
    "get_brain_adapter",
]
