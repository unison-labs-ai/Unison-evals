"""Datasets — one loader per benchmark. All return Question objects."""

from .base import Dataset
from .longmemeval import LongMemEvalDataset
from .memoryagentbench import MemoryAgentBenchDataset

REGISTRY: dict[str, type[Dataset]] = {
    "longmemeval": LongMemEvalDataset,
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
    "LongMemEvalDataset",
    "MemoryAgentBenchDataset",
    "get_dataset",
]
