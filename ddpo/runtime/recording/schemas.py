from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RunStatus(str, Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass
class BatchFacts:
    epoch: int
    batch_idx: int
    rank: int
    generated_count: int
    valid_count: int
    reward_mean: float
    reward_min: float
    reward_max: float
    avg_qed: float
    avg_sa: float
    avg_docking: float
    filtering_ratio: float
    novelty_penalty_ratio: float
    avg_qed_norm: float = 0.0
    avg_sa_norm: float = 0.0
    avg_docking_norm: float = 0.0
    failure_counts: Dict[str, int] = field(default_factory=dict)
    molecule_rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EpochFacts:
    epoch: int
    timestamp: str
    avg_loss: float
    reward_mean: float
    reward_min: float
    reward_max: float
    avg_qed: float
    avg_sa: float
    avg_docking: float
    generated_total: float
    rdkit_valid: float
    rdkit_valid_ratio: float
    filtering_ratio_mean: float = 0.0
    novelty_penalty_ratio_mean: float = 0.0
    failure_counts: Dict[str, int] = field(default_factory=dict)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_due: bool = False


@dataclass
class CheckpointFacts:
    epoch: int
    path: str
    kind: str
    monitor: Optional[str]
    monitor_value: Optional[float]
    exists: bool
    size_bytes: int
    confirmed: bool
    saved_at: str


@dataclass
class RunLifecycleFacts:
    run_id: str
    run_name: str
    output_dir: str
    status: RunStatus
    started_at: str
    ended_at: Optional[str] = None
    message: Optional[str] = None
