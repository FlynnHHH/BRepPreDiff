from .graph import BRepGraph, GraphBatch

__all__ = ["BRepGraph", "GraphBatch", "StepSegDataset", "build_dataloader"]


def __getattr__(name: str):
    if name in {"StepSegDataset", "build_dataloader"}:
        from .dataset import StepSegDataset, build_dataloader

        return {"StepSegDataset": StepSegDataset, "build_dataloader": build_dataloader}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
