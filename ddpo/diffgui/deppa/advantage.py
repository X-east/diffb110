"""DePPA-style rank-normal advantages for DiffGui training.

The original DePPA reward is a batch-relative objective: component scores are
ranked within the sampled batch, converted from percentile to a normal score,
and then combined with fixed weights.  This module keeps that logic isolated
from the existing DiffGui reward stack so old aromatic/ring/guard rewards do
not leak into DePPA experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


DEFAULT_WEIGHTS: dict[str, float] = {
    "qed": 0.20,
    "sa": 0.20,
    "vina_score": 0.50,
    "distance": 0.10,
    "strain": 0.0,
}

DEFAULT_ENABLED_METRICS = ("qed", "sa", "vina_score", "distance")
DEFAULT_DIVERSITY_RADIUS = 2
DEFAULT_DIVERSITY_BITS = 2048

# Keep the legacy metric name ``distance`` in configs/CSVs, but compute it as
# DePPA's Morgan-fingerprint diversity term.
_DIVERSITY_FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=DEFAULT_DIVERSITY_RADIUS,
    fpSize=DEFAULT_DIVERSITY_BITS,
)


@dataclass
class DePPAAdvantageResult:
    """Graph-local outputs of DePPA rank-normal reward construction."""

    rewards_by_graph: list[float]
    advantages: torch.Tensor
    fields_by_graph: dict[int, dict[str, Any]]
    summary: dict[str, float]


def is_deppa_mode_enabled(config: dict[str, Any]) -> bool:
    """Return whether DiffGui should use the isolated DePPA reward path."""
    cfg = config.get("diffgui_deppa_mode", {}) or {}
    return bool(cfg.get("enabled", False))


def build_local_deppa_records(
    *,
    reconstruction_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    rank: int,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build serializable per-graph records for DDP all-gather.

    Records intentionally contain only plain Python scalars so they can be
    gathered through ``torch.distributed.all_gather_object``.
    """
    cfg = config or {}
    reward_by_graph = _reward_rows_by_graph_index(reconstruction_rows, reward_rows)
    records: list[dict[str, Any]] = []
    for graph_index, reconstruction_row in enumerate(reconstruction_rows):
        reward_row = reward_by_graph.get(graph_index, {})
        reconstruction_success = str(reconstruction_row.get("reconstruction_status", "")) == "ok"
        rdkit_valid = _row_bool(reward_row, "rdkit_valid")
        metrics = {
            "qed": _extract_qed(reward_row, cfg),
            "sa": _extract_sa(reward_row, cfg),
            "vina_score": _extract_vina_score(reward_row, cfg),
            "strain": _extract_optional_metric("strain", reward_row, cfg),
        }
        records.append(
            {
                "rank": int(rank),
                "graph_index": int(graph_index),
                "reconstruction_success": int(reconstruction_success),
                "reward_found": int(bool(reward_row)),
                "rdkit_valid": int(rdkit_valid),
                "smiles": str(
                    reward_row.get("canonical_smiles")
                    or reward_row.get("smiles")
                    or reconstruction_row.get("smiles")
                    or ""
                ),
                "metrics": metrics,
            }
        )
    return records


def compute_deppa_advantages(
    *,
    local_records: list[dict[str, Any]],
    gathered_records: list[list[dict[str, Any]]] | list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> DePPAAdvantageResult:
    """Compute DePPA rewards and advantages for the local rank.

    ``gathered_records`` may be either the output of all-gather
    (``list[list[record]]``) or a flat list for single-process tests.
    """
    cfg = config or {}
    all_records = _flatten_records(gathered_records)
    metric_weights = _metric_weights(cfg)
    enabled_metrics = [name for name, weight in metric_weights.items() if abs(float(weight)) > 0.0]
    _attach_distance_metric(all_records)
    global_record_by_key = {_record_key(record): record for record in all_records}

    valid_keys = _valid_record_keys(all_records, enabled_metrics)
    normal_scores = _rank_normal_scores(all_records, enabled_metrics, valid_keys, cfg)
    combined_by_key: dict[tuple[int, int], float] = {}
    for record in all_records:
        key = _record_key(record)
        if key not in valid_keys:
            continue
        combined = 0.0
        for metric_name in enabled_metrics:
            combined += float(metric_weights[metric_name]) * float(normal_scores[key][metric_name])
        combined_by_key[key] = float(combined)

    invalid_reward = _invalid_reward(combined_by_key, cfg)
    local_rewards: list[float] = []
    fields_by_graph: dict[int, dict[str, Any]] = {}
    for record in local_records:
        key = _record_key(record)
        reward = combined_by_key.get(key, invalid_reward)
        local_rewards.append(float(reward))
        fields_record = global_record_by_key.get(key, record)
        fields = _deppa_fields(
            fields_record,
            enabled_metrics,
            metric_weights,
            normal_scores.get(key, {}),
            reward,
            key in valid_keys,
        )
        fields_by_graph[int(record.get("graph_index", 0))] = fields

    advantages = torch.tensor(local_rewards, dtype=torch.float32)
    advantages = _postprocess_advantages(advantages, cfg)
    summary = _summary(local_rewards, fields_by_graph, enabled_metrics)
    return DePPAAdvantageResult(
        rewards_by_graph=local_rewards,
        advantages=advantages,
        fields_by_graph=fields_by_graph,
        summary=summary,
    )


def _reward_rows_by_graph_index(
    reconstruction_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    ok_indices = [int(row["mol_index"]) for row in reconstruction_rows if row.get("reconstruction_status") == "ok"]
    mapped: dict[int, dict[str, Any]] = {}
    for reward_row in reward_rows:
        graph_index = _reward_graph_index(reward_row, ok_indices)
        if graph_index is not None:
            mapped[graph_index] = reward_row
    return mapped


def _reward_graph_index(reward_row: dict[str, Any], ok_indices: list[int]) -> int | None:
    """Map a possibly PB-filtered reward row to its original graph index."""
    try:
        direct_index = int(float(reward_row.get("source_graph_index", "")))
    except (TypeError, ValueError):
        direct_index = -1
    if direct_index in set(ok_indices):
        return direct_index
    try:
        source_index = int(reward_row.get("source_index", -1))
    except (TypeError, ValueError):
        return None
    if 0 <= source_index < len(ok_indices):
        return ok_indices[source_index]
    return None


def _metric_weights(config: dict[str, Any]) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    weights.update({str(key): float(value) for key, value in (config.get("weights", {}) or {}).items()})
    metrics_cfg = config.get("metrics", {}) or {}
    enabled = config.get("enabled_metrics", None)
    if enabled is None:
        enabled_names = set(DEFAULT_ENABLED_METRICS)
    else:
        enabled_names = {str(name) for name in enabled}
    for name, metric_cfg in metrics_cfg.items():
        metric_name = str(name)
        if isinstance(metric_cfg, dict):
            if "weight" in metric_cfg:
                weights[metric_name] = float(metric_cfg["weight"])
            if "enabled" in metric_cfg and not bool(metric_cfg["enabled"]):
                weights[metric_name] = 0.0
    for name in list(weights):
        if name not in enabled_names and name not in metrics_cfg:
            weights[name] = 0.0
    return weights


def _extract_qed(row: dict[str, Any], config: dict[str, Any]) -> float | None:
    return _extract_first_finite("qed", row, config, default_fields=("qed_raw", "qed_reward", "qed"))


def _extract_sa(row: dict[str, Any], config: dict[str, Any]) -> float | None:
    value = _extract_first_finite("sa", row, config, default_fields=("sa_reward", "sa_norm"))
    if value is not None:
        return value
    raw = _first_finite(row, ("sa_raw",))
    if raw is None:
        return None
    return float(max(0.0, min(1.0, (10.0 - raw) / 9.0)))


def _extract_vina_score(row: dict[str, Any], config: dict[str, Any]) -> float | None:
    value = _extract_first_finite("vina_score", row, config, default_fields=("vina_score",))
    if value is not None:
        return value
    docking_raw = _first_finite(row, ("docking_raw",))
    if docking_raw is None:
        return None
    if "docking_success" in row and not _row_bool(row, "docking_success"):
        return None
    if "docking_success" not in row and abs(float(docking_raw)) <= 1.0e-12:
        return None
    direction = str((config.get("metrics", {}) or {}).get("vina_score", {}).get("direction", "lower")).lower()
    return -float(docking_raw) if direction == "lower" else float(docking_raw)


def _extract_optional_metric(name: str, row: dict[str, Any], config: dict[str, Any]) -> float | None:
    return _extract_first_finite(name, row, config, default_fields=(name,))


def _extract_first_finite(
    name: str,
    row: dict[str, Any],
    config: dict[str, Any],
    *,
    default_fields: tuple[str, ...],
) -> float | None:
    metric_cfg = (config.get("metrics", {}) or {}).get(name, {}) or {}
    fields = metric_cfg.get("source_fields", metric_cfg.get("source_field", default_fields))
    if isinstance(fields, str):
        fields = (fields,)
    value = _first_finite(row, tuple(str(field) for field in fields))
    if value is None:
        return None
    direction = str(metric_cfg.get("direction", "higher")).lower()
    if direction == "lower":
        return -float(value)
    return float(value)


def _first_finite(row: dict[str, Any], fields: tuple[str, ...]) -> float | None:
    for field in fields:
        value = row.get(field)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return float(numeric)
    return None


def _attach_distance_metric(records: list[dict[str, Any]]) -> None:
    keyed_fps: list[tuple[tuple[int, int], Any]] = []
    for record in records:
        smiles = str(record.get("smiles", ""))
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            continue
        keyed_fps.append((_record_key(record), _DIVERSITY_FINGERPRINT_GENERATOR.GetFingerprint(mol)))
    if len(keyed_fps) < 2:
        distances = {key: 0.0 for key, _fp in keyed_fps}
    else:
        distances = {}
        for index, (key, fingerprint) in enumerate(keyed_fps):
            similarities = [
                float(DataStructs.TanimotoSimilarity(fingerprint, other_fp))
                for other_index, (_other_key, other_fp) in enumerate(keyed_fps)
                if other_index != index
            ]
            distances[key] = float(1.0 - sum(similarities) / len(similarities)) if similarities else 0.0
    for record in records:
        metrics = dict(record.get("metrics", {}) or {})
        key = _record_key(record)
        metrics["distance"] = distances.get(key, None)
        record["metrics"] = metrics


def _valid_record_keys(records: list[dict[str, Any]], enabled_metrics: list[str]) -> set[tuple[int, int]]:
    valid_keys: set[tuple[int, int]] = set()
    for record in records:
        if not _record_is_base_valid(record):
            continue
        metrics = record.get("metrics", {}) or {}
        if all(_is_finite(metrics.get(metric_name)) for metric_name in enabled_metrics):
            valid_keys.add(_record_key(record))
    return valid_keys


def _rank_normal_scores(
    records: list[dict[str, Any]],
    enabled_metrics: list[str],
    valid_keys: set[tuple[int, int]],
    config: dict[str, Any],
) -> dict[tuple[int, int], dict[str, float]]:
    scores: dict[tuple[int, int], dict[str, float]] = {key: {} for key in valid_keys}
    eps = float(config.get("normal_eps", 1.0e-3))
    normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    for metric_name in enabled_metrics:
        values: list[tuple[tuple[int, int], float]] = []
        for record in records:
            key = _record_key(record)
            if key not in valid_keys:
                continue
            values.append((key, float((record.get("metrics", {}) or {}).get(metric_name))))
        for key, percentile in _rank_percentiles(values).items():
            clipped = min(1.0 - eps, max(eps, float(percentile)))
            scores[key][metric_name] = float(normal.icdf(torch.tensor(clipped)).item())
    return scores


def _rank_percentiles(values: list[tuple[tuple[int, int], float]]) -> dict[tuple[int, int], float]:
    if not values:
        return {}
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    ranks: dict[tuple[int, int], float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (float(start) + float(end - 1)) / 2.0
        for idx in range(start, end):
            ranks[ordered[idx][0]] = average_rank
        start = end
    n_valid = float(len(ordered))
    return {key: (rank + 0.5) / n_valid for key, rank in ranks.items()}


def _invalid_reward(combined_by_key: dict[tuple[int, int], float], config: dict[str, Any]) -> float:
    fallback = float(config.get("invalid_fallback_reward", -1.0))
    if not combined_by_key:
        return fallback
    values = torch.tensor(list(combined_by_key.values()), dtype=torch.float32)
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    penalty = mean - float(config.get("invalid_penalty_std", 3.0)) * std
    if std <= float(config.get("invalid_std_epsilon", 1.0e-8)):
        penalty = min(float(penalty), fallback)
    floor = config.get("invalid_penalty_floor", None)
    if floor is not None:
        penalty = max(float(floor), float(penalty))
    return float(penalty)


def _postprocess_advantages(values: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    out = values.clone().float()
    if bool(config.get("normalize_advantages", False)) and out.numel() > 1:
        epsilon = float(config.get("advantage_epsilon", 1.0e-6))
        std = out.std(unbiased=False)
        if float(std.item()) > epsilon:
            out = (out - out.mean()) / (std + epsilon)
        else:
            out = out - out.mean()
    clamp_min = config.get("advantage_clip_min", None)
    clamp_max = config.get("advantage_clip_max", None)
    if clamp_min is not None or clamp_max is not None:
        min_value = -float("inf") if clamp_min is None else float(clamp_min)
        max_value = float("inf") if clamp_max is None else float(clamp_max)
        out = out.clamp(min=min_value, max=max_value)
    return out


def _deppa_fields(
    record: dict[str, Any],
    enabled_metrics: list[str],
    metric_weights: dict[str, float],
    normal_scores: dict[str, float],
    reward: float,
    valid: bool,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "deppa_enabled": 1,
        "deppa_valid": int(valid),
        "deppa_reward": float(reward),
        "deppa_metric_count": int(len(enabled_metrics)),
    }
    metrics = record.get("metrics", {}) or {}
    for metric_name in enabled_metrics:
        fields[f"deppa_{metric_name}_raw"] = metrics.get(metric_name, "")
        fields[f"deppa_{metric_name}_normal"] = normal_scores.get(metric_name, "")
        fields[f"deppa_{metric_name}_weight"] = float(metric_weights.get(metric_name, 0.0))
    return fields


def _summary(
    local_rewards: list[float],
    fields_by_graph: dict[int, dict[str, Any]],
    enabled_metrics: list[str],
) -> dict[str, float]:
    values = torch.tensor(local_rewards, dtype=torch.float32)
    valid_count = sum(int(row.get("deppa_valid", 0)) for row in fields_by_graph.values())
    summary = {
        "deppa_enabled": 1.0,
        "deppa_metric_count": float(len(enabled_metrics)),
        "deppa_valid_count": float(valid_count),
        "deppa_invalid_count": float(max(0, len(fields_by_graph) - valid_count)),
        "deppa_reward_mean": float(values.mean().item()) if values.numel() else 0.0,
        "deppa_reward_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "deppa_reward_min": float(values.min().item()) if values.numel() else 0.0,
        "deppa_reward_max": float(values.max().item()) if values.numel() else 0.0,
    }
    return summary


def _flatten_records(records: list[list[dict[str, Any]]] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    first = records[0]
    if isinstance(first, dict):
        return [dict(record) for record in records]  # type: ignore[arg-type]
    flat: list[dict[str, Any]] = []
    for group in records:  # type: ignore[assignment]
        flat.extend(dict(record) for record in group)
    return flat


def _record_is_base_valid(record: dict[str, Any]) -> bool:
    return (
        _row_bool(record, "reconstruction_success")
        and _row_bool(record, "reward_found")
        and _row_bool(record, "rdkit_valid")
    )


def _record_key(record: dict[str, Any]) -> tuple[int, int]:
    return (int(record.get("rank", 0)), int(record.get("graph_index", 0)))


def _row_bool(row: dict[str, Any], key: str) -> bool:
    try:
        return int(float(row.get(key, 0) or 0)) == 1
    except (TypeError, ValueError):
        return False


def _is_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)
