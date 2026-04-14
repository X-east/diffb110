"""Protocol adapter for the Lightning DDPO mainline.

Bug4 Fix-1: re-exports DataProto/TensorDict from verl_diffusion.protocol,
providing a single canonical import point for all ddpo-internal code.

Do NOT import directly from verl_diffusion.protocol in new ddpo code.
Use: from ddpo.runtime.protocol import DataProto, TensorDict
"""
from __future__ import annotations

from verl_diffusion.protocol import DataProto, TensorDict

__all__ = ["DataProto", "TensorDict"]
