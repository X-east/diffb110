"""Minimal graph_tool.all shim.

This is enough for DiffMeta's molecular-generation import path. SBM graph
metrics still require the real graph-tool package and will raise clearly.
"""


class Graph:
    """Placeholder graph-tool Graph object for unused SBM metric paths."""

    def add_edge_list(self, _edges):
        """Accept edge-list setup for compatibility before raising later."""
        return None


def minimize_blockmodel_dl(_graph):
    """Raise when SBM metrics try to use graph-tool functionality."""
    raise RuntimeError("graph_tool is not installed; SBM graph metrics are unavailable.")


def contiguous_map(_blocks):
    """Raise when SBM metrics try to use graph-tool functionality."""
    raise RuntimeError("graph_tool is not installed; SBM graph metrics are unavailable.")
