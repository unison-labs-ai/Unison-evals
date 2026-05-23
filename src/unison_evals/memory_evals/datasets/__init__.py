"""Datasets — one loader per benchmark. All return Question objects."""

from .base import Dataset
from .bitempoqa import BitempoQADataset
from .frames import FramesDataset
from .longmemeval import LongMemEvalDataset
from .memoryagentbench import MemoryAgentBenchDataset
from .msmarco import MsMarcoDataset
from .musique import MuSiQueDataset

REGISTRY: dict[str, type[Dataset]] = {
    "longmemeval": LongMemEvalDataset,
    "frames": FramesDataset,
    "musique": MuSiQueDataset,
    "memoryagentbench": MemoryAgentBenchDataset,
    "bitempoqa": BitempoQADataset,
    "msmarco": MsMarcoDataset,
}


def get_dataset(name: str) -> Dataset:
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return REGISTRY[name]()


__all__ = [
    "REGISTRY",
    "BitempoQADataset",
    "Dataset",
    "FramesDataset",
    "LongMemEvalDataset",
    "MemoryAgentBenchDataset",
    "MsMarcoDataset",
    "MuSiQueDataset",
    "get_dataset",
]
