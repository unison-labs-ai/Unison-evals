"""Adapters — one per system.

Contract: AgentAdapter (Tracks 2 + 3) — answers questions.

Adding a new system: subclass AgentAdapter, implement the abstract
methods, register in REGISTRY.
"""

from .base import AgentAdapter
from .unison_agent import UnisonAgentAdapter, UnisonAgentPipelineAdapter
from .unison_brain_context import UnisonBrainContextAdapter

REGISTRY: dict[str, type[AgentAdapter]] = {
    "unison-agent": UnisonAgentAdapter,
    "unison-agent-pipeline": UnisonAgentPipelineAdapter,
    "unison-brain-context": UnisonBrainContextAdapter,
}


def get_adapter(name: str) -> AgentAdapter:
    """Factory — returns an instantiated AgentAdapter by registry name."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown agent adapter '{name}'. Available: {available}")
    return REGISTRY[name]()


__all__ = [
    "REGISTRY",
    "AgentAdapter",
    "UnisonAgentAdapter",
    "UnisonAgentPipelineAdapter",
    "UnisonBrainContextAdapter",
    "get_adapter",
]
