"""Per-timestep PPO diagnostics for DiffGui epoch recording.

Writes ``kl_by_timestep.csv``, ``clipfrac_by_timestep.csv``, and
``ratio_by_timestep.csv`` from the actor's ``last_step_metrics``,
grouping by ``step_idx``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    import statistics
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})


def export_per_timestep(
    rank_dir: Path,
    step_metrics: list[dict[str, Any]],
    *,
    epoch: int,
    rank: int,
) -> None:
    """Write per-timestep CSV files from actor step metrics.

    Writes into *rank_dir*:
    - ``kl_by_timestep.csv``
    - ``clipfrac_by_timestep.csv``
    - ``ratio_by_timestep.csv`` (includes pos/node/halfedge logprob components)
    """
    rank_dir.mkdir(parents=True, exist_ok=True)

    by_timestep: dict[int, list[dict[str, Any]]] = {}
    for row in step_metrics:
        t = int(row.get("step_idx", 0))
        by_timestep.setdefault(t, []).append(row)

    # KL by timestep
    kl_rows: list[dict[str, Any]] = []
    for t in sorted(by_timestep):
        group = by_timestep[t]
        kls = [float(r["approx_kl"]) for r in group]
        kl_rows.append({
            "epoch": epoch, "rank": rank, "step_idx": t,
            "n_samples": len(kls),
            "approx_kl_mean": _mean(kls), "approx_kl_std": _std(kls),
        })
    _write_csv(rank_dir / "kl_by_timestep.csv", kl_rows)

    # Clipfrac by timestep
    cf_rows: list[dict[str, Any]] = []
    for t in sorted(by_timestep):
        group = by_timestep[t]
        cfs = [float(r["clipfrac"]) for r in group]
        cf_rows.append({
            "epoch": epoch, "rank": rank, "step_idx": t,
            "n_samples": len(cfs),
            "clipfrac_mean": _mean(cfs), "clipfrac_std": _std(cfs),
        })
    _write_csv(rank_dir / "clipfrac_by_timestep.csv", cf_rows)

    # Ratio + component logprobs by timestep
    ratio_rows: list[dict[str, Any]] = []
    for t in sorted(by_timestep):
        group = by_timestep[t]
        ratios = [float(r["ratio_mean"]) for r in group]
        lp_pos = [float(r.get("logprob_pos_mean", 0)) for r in group]
        lp_node = [float(r.get("logprob_node_mean", 0)) for r in group]
        lp_hedge = [float(r.get("logprob_halfedge_mean", 0)) for r in group]
        ratio_rows.append({
            "epoch": epoch, "rank": rank, "step_idx": t,
            "n_samples": len(ratios),
            "ratio_mean": _mean(ratios), "ratio_std": _std(ratios),
            "logprob_pos_mean": _mean(lp_pos),
            "logprob_node_mean": _mean(lp_node),
            "logprob_halfedge_mean": _mean(lp_hedge),
        })
    _write_csv(rank_dir / "ratio_by_timestep.csv", ratio_rows)


def merge_per_timestep(
    rows: list[dict[str, Any]],
    output_path: Path,
    world_size: int,
) -> Path | None:
    """Merge per-timestep diagnostics across ranks by averaging.

    Returns the output path on success, or None if *rows* is empty.
    """
    if not rows:
        return None
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        t = int(row.get("step_idx", 0))
        by_step.setdefault(t, []).append(row)
    merged: list[dict[str, Any]] = []
    for t in sorted(by_step):
        group = by_step[t]
        ck_row = dict(group[0])
        ck_row["n_ranks"] = len(group)
        for key in list(ck_row.keys()):
            if key.endswith("_mean") or key.endswith("_std"):
                vals = [float(r.get(key, 0)) for r in group if key in r]
                ck_row[key] = _mean(vals) if vals else 0.0
        merged.append(ck_row)
    _write_csv(output_path, merged)
    return output_path
