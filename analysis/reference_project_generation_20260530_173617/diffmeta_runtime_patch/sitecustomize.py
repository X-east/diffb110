"""Runtime compatibility shims for DiffMeta smoke tests."""

try:
    from typing import Tuple, Union

    import torch
    import torch_geometric.typing as pyg_typing

    if not hasattr(pyg_typing, "InputEdges"):
        pyg_typing.InputEdges = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
except Exception:
    pass
