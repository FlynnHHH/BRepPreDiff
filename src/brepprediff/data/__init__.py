from .graph import BRepGraph, GraphBatch
from .classification import read_class_label

__all__ = [
    "BRepGraph",
    "GraphBatch",
    "MultiSourceDataset",
    "StepSegDataset",
    "build_dataloader",
    "read_class_label",
]


def __getattr__(name: str):
    if name in {"MultiSourceDataset", "StepSegDataset", "build_dataloader"}:
        from .dataset import MultiSourceDataset, StepSegDataset, build_dataloader

        return {
            "MultiSourceDataset": MultiSourceDataset,
            "StepSegDataset": StepSegDataset,
            "build_dataloader": build_dataloader,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
