"""Datasets — one loader per benchmark. All return Question objects."""

from .base import Dataset
from .locomo import LocomoDataset
from .longmemeval import LongMemEvalDataset
from .memoryagentbench import MemoryAgentBenchDataset

REGISTRY: dict[str, type[Dataset]] = {
    "longmemeval": LongMemEvalDataset,
    "locomo": LocomoDataset,
    "memoryagentbench": MemoryAgentBenchDataset,
}


def get_dataset(name: str) -> Dataset:
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return REGISTRY[name]()


__all__ = [
    "REGISTRY",
    "Dataset",
    "LocomoDataset",
    "LongMemEvalDataset",
    "MemoryAgentBenchDataset",
    "get_dataset",
]
