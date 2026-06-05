"""Generation-time atom prior bias and final coordinate refinement.

The coordinate refinement follows the mature training-free guidance pattern used
by BInD/MolSnapper/DecompOpt: define differentiable geometry energies, take
autograd gradients with respect to ligand coordinates, and apply small clipped
updates before molecule reconstruction.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AtomBiasConfig:
    """Configuration for pre-argmax atom score prior correction."""

    bias_by_element: Mapping[str, float] = field(default_factory=dict)
    temperature: float = 1.0


@dataclass(frozen=True)
class AtomLocalMaskConfig:
    """Configuration for local chemistry-aware atom score masking."""

    enabled: bool = False
    candidate_cutoff: float = 2.20
    terminal_degree_max: int = 1
    terminal_only_elements: tuple[str, ...] = ("F", "Cl", "Br", "I")
    disable_elements: tuple[str, ...] = ("P", "Br", "I", "B")
    mask_value: float = -1.0e6


@dataclass(frozen=True)
class GeometryRefineConfig:
    """Configuration for final-only geometry guidance before reconstruction."""

    enabled: bool = False
    steps: int = 10
    learning_rate: float = 0.05
    max_step: float = 0.05
    max_total_shift: float = 0.30
    candidate_cutoff: float = 2.20
    soft_tau: float = 0.10
    min_distance_factor: float = 0.80
    bond_lower_factor: float = 0.85
    bond_upper_margin: float = 0.25
    close_pair_weight: float = 1.00
    bond_length_weight: float = 0.50
    degree_weight: float = 0.20
    triangle_weight: float = 0.05
    pocket_clash_weight: float = 0.20
    tether_weight: float = 5.00
    pocket_clash_distance: float = 1.20
    use_pocket_clash: bool = True
    max_triangle_atoms: int = 80


@dataclass(frozen=True)
class GeometryRefineStats:
    """Summary statistics for a coordinate refinement call."""

    mean_shift: float
    max_shift: float
    mean_energy_before: float
    mean_energy_after: float


class AtomScoreDiagnosticsCollector:
    """Collect raw and corrected atom scores before final atom-type argmax."""

    def __init__(self, atom_decoder: Sequence[str]) -> None:
        self.atom_decoder = list(atom_decoder)
        self._raw_chunks: list[torch.Tensor] = []
        self._corrected_chunks: list[torch.Tensor] = []

    def update(self, raw_scores: torch.Tensor, corrected_scores: torch.Tensor) -> None:
        """Append one generated batch of atom scores."""
        if raw_scores.shape != corrected_scores.shape:
            raise ValueError("Raw and corrected atom score tensors must have the same shape.")
        if raw_scores.shape[-1] != len(self.atom_decoder):
            raise ValueError(
                "Atom score tensor width does not match atom_decoder length: "
                f"{raw_scores.shape[-1]} != {len(self.atom_decoder)}."
            )
        self._raw_chunks.append(raw_scores.detach().float().cpu())
        self._corrected_chunks.append(corrected_scores.detach().float().cpu())

    @property
    def has_data(self) -> bool:
        """Whether any scores have been collected."""
        return len(self._raw_chunks) > 0

    def raw_scores(self) -> torch.Tensor:
        """Return all raw score rows."""
        return torch.cat(self._raw_chunks, dim=0) if self._raw_chunks else torch.empty(0, len(self.atom_decoder))

    def corrected_scores(self) -> torch.Tensor:
        """Return all corrected score rows."""
        return (
            torch.cat(self._corrected_chunks, dim=0)
            if self._corrected_chunks else torch.empty(0, len(self.atom_decoder))
        )


def parse_element_float_map(text: str | None) -> dict[str, float]:
    """Parse comma-separated element:value pairs."""
    if text is None or str(text).strip() == "":
        return {}

    values: dict[str, float] = {}
    for item in str(text).split(","):
        if item.strip() == "":
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid element map item {item!r}; expected ELEMENT:VALUE."
            )
        element, value = item.split(":", 1)
        key = element.strip()
        if key == "":
            raise ValueError(f"Invalid empty element name in {item!r}.")
        values[key] = float(value)
    return values


def parse_element_tuple(text: str | None) -> tuple[str, ...]:
    """Parse a comma-separated element list."""
    if text is None or str(text).strip() == "":
        return ()
    return tuple(item.strip() for item in str(text).split(",") if item.strip())


def dataset_atom_distribution(dataset_info: Mapping[str, object]) -> dict[str, float]:
    """Return normalized atom frequencies from a DiffSBDD dataset info dict."""
    hist = dataset_info.get("atom_hist")
    if not isinstance(hist, Mapping):
        raise ValueError("dataset_info must contain an atom_hist mapping.")

    positive_counts = {str(k): float(v) for k, v in hist.items() if float(v) > 0}
    total = sum(positive_counts.values())
    if total <= 0:
        raise ValueError("atom_hist has no positive atom counts.")
    return {element: count / total for element, count in positive_counts.items()}


def build_log_ratio_atom_bias(
    *,
    atom_decoder: Sequence[str],
    target_fraction: Mapping[str, float],
    baseline_fraction: Mapping[str, float],
    strength: float,
    clamp: float,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Build a mature prior-correction bias from target and generated fractions."""
    bias: dict[str, float] = {}
    for element in atom_decoder:
        target = float(target_fraction.get(element, 0.0))
        baseline = float(baseline_fraction.get(element, 0.0))
        if target <= 0.0:
            continue
        value = strength * math.log((target + eps) / (baseline + eps))
        bias[element] = max(-clamp, min(clamp, value))
    return bias


def atom_bias_tensor(
    atom_decoder: Sequence[str],
    config: AtomBiasConfig | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Convert element-name atom biases into an atom-score tensor."""
    if config is None or len(config.bias_by_element) == 0:
        return None
    bias = torch.zeros(len(atom_decoder), device=device, dtype=dtype)
    index_by_element = {element: idx for idx, element in enumerate(atom_decoder)}
    for element, value in config.bias_by_element.items():
        if element not in index_by_element:
            raise ValueError(
                f"Atom bias element {element!r} is not in atom_decoder "
                f"{list(atom_decoder)!r}."
            )
        bias[index_by_element[element]] = float(value)
    return bias


def apply_atom_score_bias(
    atom_scores: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply temperature and prior bias before atom-type argmax."""
    if temperature <= 0:
        raise ValueError("Atom bias temperature must be positive.")
    corrected = atom_scores / float(temperature)
    if bias is not None:
        corrected = corrected + bias.to(device=atom_scores.device, dtype=atom_scores.dtype)
    return corrected


def apply_local_atom_score_mask(
    atom_scores: torch.Tensor,
    positions: torch.Tensor,
    batch_mask: torch.Tensor,
    atom_decoder: Sequence[str],
    config: AtomLocalMaskConfig | None,
) -> torch.Tensor:
    """Mask locally implausible atom choices before final atom-type argmax.

    The first rule targets terminal atoms such as F/Cl/Br/I. If a generated
    node has more than ``terminal_degree_max`` close candidate neighbors, those
    terminal-only elements are masked for that node. A second optional rule
    disables rare/high-risk elements globally unless explicitly enabled.
    """
    if config is None or not config.enabled:
        return atom_scores
    if atom_scores.ndim != 2:
        raise ValueError("atom_scores must have shape [n_atoms, n_atom_types].")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [n_atoms, 3].")
    if atom_scores.shape[0] != positions.shape[0]:
        raise ValueError("atom_scores and positions must have the same row count.")
    if atom_scores.shape[1] != len(atom_decoder):
        raise ValueError(
            "atom_scores width does not match atom_decoder length: "
            f"{atom_scores.shape[1]} != {len(atom_decoder)}."
        )

    corrected = atom_scores.clone()
    index_by_element = {element: idx for idx, element in enumerate(atom_decoder)}

    disabled = [
        index_by_element[element]
        for element in config.disable_elements
        if element in index_by_element
    ]
    if disabled:
        disabled_idx = torch.as_tensor(disabled, device=corrected.device, dtype=torch.long)
        corrected[:, disabled_idx] = float(config.mask_value)

    terminal = [
        index_by_element[element]
        for element in config.terminal_only_elements
        if element in index_by_element
    ]
    if not terminal:
        return corrected

    degree = _candidate_degree(positions, batch_mask, cutoff=float(config.candidate_cutoff))
    internal_mask = degree > int(config.terminal_degree_max)
    if bool(internal_mask.any().item()):
        terminal_idx = torch.as_tensor(terminal, device=corrected.device, dtype=torch.long)
        row_idx = internal_mask.nonzero(as_tuple=False).flatten()
        corrected[row_idx.unsqueeze(1), terminal_idx.unsqueeze(0)] = float(config.mask_value)
    return corrected


def _candidate_degree(
    positions: torch.Tensor,
    batch_mask: torch.Tensor,
    *,
    cutoff: float,
) -> torch.Tensor:
    """Estimate local candidate degree from intra-molecule close neighbors."""
    degree = torch.zeros(positions.shape[0], device=positions.device, dtype=torch.long)
    for batch_id in torch.unique(batch_mask.detach()).tolist():
        selected = batch_mask == int(batch_id)
        local_indices = selected.nonzero(as_tuple=False).flatten()
        if local_indices.numel() < 2:
            continue
        local_pos = positions[local_indices]
        distances = torch.cdist(local_pos, local_pos)
        close = (distances <= float(cutoff)) & (distances > 1.0e-6)
        degree[local_indices] = close.sum(dim=1).long()
    return degree


def _quantile(values: torch.Tensor, q: float) -> float:
    """Return a scalar quantile for a 1D tensor."""
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), q).item())


def _score_element_rows(
    scores: torch.Tensor,
    atom_decoder: Sequence[str],
    *,
    stage: str,
) -> list[dict[str, float | str]]:
    """Build per-element score distribution rows."""
    rows: list[dict[str, float | str]] = []
    for idx, element in enumerate(atom_decoder):
        values = scores[:, idx].float()
        rows.append({
            "stage": stage,
            "element": element,
            "mean": float(values.mean().item()) if values.numel() else 0.0,
            "std": float(values.std(unbiased=False).item()) if values.numel() else 0.0,
            "min": float(values.min().item()) if values.numel() else 0.0,
            "q10": _quantile(values, 0.10),
            "q25": _quantile(values, 0.25),
            "q50": _quantile(values, 0.50),
            "q75": _quantile(values, 0.75),
            "q90": _quantile(values, 0.90),
            "max": float(values.max().item()) if values.numel() else 0.0,
        })
    return rows


def _top_count_rows(
    scores: torch.Tensor,
    atom_decoder: Sequence[str],
    *,
    stage: str,
) -> list[dict[str, float | str]]:
    """Build top-1 and top-2 atom score count rows."""
    if scores.numel() == 0:
        return []
    topk = torch.topk(scores, k=min(2, scores.shape[1]), dim=1).indices
    rows: list[dict[str, float | str]] = []
    total = float(scores.shape[0])
    for rank_idx, rank_name in enumerate(["top1", "top2"][:topk.shape[1]]):
        counts = torch.bincount(topk[:, rank_idx], minlength=len(atom_decoder))
        for idx, element in enumerate(atom_decoder):
            count = int(counts[idx].item())
            rows.append({
                "stage": stage,
                "rank": rank_name,
                "element": element,
                "count": count,
                "fraction": count / total if total else 0.0,
            })
    return rows


def _margin_rows(scores: torch.Tensor, *, stage: str) -> list[dict[str, float | str]]:
    """Build top-1 minus top-2 score margin summary rows."""
    if scores.shape[0] == 0 or scores.shape[1] < 2:
        return []
    top2 = torch.topk(scores, k=2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).float()
    return [{
        "stage": stage,
        "mean": float(margin.mean().item()),
        "std": float(margin.std(unbiased=False).item()),
        "q05": _quantile(margin, 0.05),
        "q10": _quantile(margin, 0.10),
        "q25": _quantile(margin, 0.25),
        "q50": _quantile(margin, 0.50),
        "q75": _quantile(margin, 0.75),
        "q90": _quantile(margin, 0.90),
    }]


def _confusion_rows(
    raw_scores: torch.Tensor,
    corrected_scores: torch.Tensor,
    atom_decoder: Sequence[str],
) -> list[dict[str, float | str]]:
    """Build raw top-1 to corrected top-1 transition rows."""
    if raw_scores.numel() == 0:
        return []
    raw_top1 = torch.argmax(raw_scores, dim=1)
    corrected_top1 = torch.argmax(corrected_scores, dim=1)
    total = float(raw_top1.numel())
    rows: list[dict[str, float | str]] = []
    for raw_idx, raw_element in enumerate(atom_decoder):
        raw_sel = raw_top1 == raw_idx
        raw_total = int(raw_sel.sum().item())
        if raw_total == 0:
            continue
        for corrected_idx, corrected_element in enumerate(atom_decoder):
            count = int(((corrected_top1 == corrected_idx) & raw_sel).sum().item())
            if count == 0:
                continue
            rows.append({
                "raw_top1": raw_element,
                "corrected_top1": corrected_element,
                "count": count,
                "fraction_all": count / total if total else 0.0,
                "fraction_within_raw": count / raw_total,
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write dictionaries to CSV if rows are available."""
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_atom_score_diagnostics(
    outdir: Path | str,
    collector: AtomScoreDiagnosticsCollector,
    *,
    extra_summary: Mapping[str, object] | None = None,
) -> None:
    """Write raw/corrected atom score diagnostics as CSV, JSON, and plots."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not collector.has_data:
        raise ValueError("No atom score diagnostics were collected.")

    atom_decoder = collector.atom_decoder
    raw_scores = collector.raw_scores()
    corrected_scores = collector.corrected_scores()

    element_rows = _score_element_rows(raw_scores, atom_decoder, stage="raw")
    element_rows += _score_element_rows(corrected_scores, atom_decoder, stage="corrected")
    top_rows = _top_count_rows(raw_scores, atom_decoder, stage="raw")
    top_rows += _top_count_rows(corrected_scores, atom_decoder, stage="corrected")
    margin_rows = _margin_rows(raw_scores, stage="raw")
    margin_rows += _margin_rows(corrected_scores, stage="corrected")
    confusion_rows = _confusion_rows(raw_scores, corrected_scores, atom_decoder)

    _write_csv(outdir / "atom_score_element_stats.csv", element_rows)
    _write_csv(outdir / "atom_score_top_counts.csv", top_rows)
    _write_csv(outdir / "atom_score_margin_stats.csv", margin_rows)
    _write_csv(outdir / "atom_score_change_confusion.csv", confusion_rows)

    raw_top1 = torch.argmax(raw_scores, dim=1)
    corrected_top1 = torch.argmax(corrected_scores, dim=1)
    changed = raw_top1 != corrected_top1
    summary = {
        "atom_decoder": atom_decoder,
        "n_atom_scores": int(raw_scores.shape[0]),
        "n_elements": int(raw_scores.shape[1]),
        "changed_count": int(changed.sum().item()),
        "changed_fraction": float(changed.float().mean().item()),
    }
    if extra_summary is not None:
        summary["extra"] = dict(extra_summary)
    with (outdir / "atom_score_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    _write_atom_score_plots(outdir, atom_decoder, raw_scores, corrected_scores)


def _write_atom_score_plots(
    outdir: Path,
    atom_decoder: Sequence[str],
    raw_scores: torch.Tensor,
    corrected_scores: torch.Tensor,
) -> None:
    """Write compact atom score diagnostic plots."""
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(atom_decoder))
    width = 0.35

    raw_top1 = torch.bincount(torch.argmax(raw_scores, dim=1), minlength=len(atom_decoder)).float()
    corrected_top1 = torch.bincount(torch.argmax(corrected_scores, dim=1), minlength=len(atom_decoder)).float()
    raw_top1 = raw_top1 / raw_top1.sum().clamp_min(1.0)
    corrected_top1 = corrected_top1 / corrected_top1.sum().clamp_min(1.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, raw_top1.numpy(), width=width, label="raw")
    ax.bar(x + width / 2, corrected_top1.numpy(), width=width, label="corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(atom_decoder)
    ax.set_ylabel("top1 fraction")
    ax.set_title("Atom score top-1 fractions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "atom_score_top1_fractions.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(atom_decoder, raw_scores.mean(dim=0).numpy(), marker="o", label="raw mean")
    ax.plot(atom_decoder, corrected_scores.mean(dim=0).numpy(), marker="o", label="corrected mean")
    ax.set_ylabel("score")
    ax.set_title("Mean atom scores by element")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "atom_score_element_means.png", dpi=200)
    plt.close(fig)

    if raw_scores.shape[1] >= 2:
        raw_margin = torch.topk(raw_scores, k=2, dim=1).values
        raw_margin = raw_margin[:, 0] - raw_margin[:, 1]
        corrected_margin = torch.topk(corrected_scores, k=2, dim=1).values
        corrected_margin = corrected_margin[:, 0] - corrected_margin[:, 1]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(raw_margin.numpy(), bins=50, alpha=0.6, label="raw")
        ax.hist(corrected_margin.numpy(), bins=50, alpha=0.6, label="corrected")
        ax.set_xlabel("top1 - top2 score margin")
        ax.set_ylabel("count")
        ax.set_title("Atom score decision margins")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "atom_score_margin_hist.png", dpi=200)
        plt.close(fig)


def refine_ligand_batch_positions(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    batch_mask: torch.Tensor,
    dataset_info: Mapping[str, object],
    config: GeometryRefineConfig | None,
    *,
    pocket_positions: torch.Tensor | None = None,
    pocket_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, GeometryRefineStats | None]:
    """Refine generated ligand coordinates molecule-by-molecule."""
    if config is None or not config.enabled or config.steps <= 0:
        return positions, None

    refined = positions.detach().clone()
    energy_before: list[float] = []
    energy_after: list[float] = []

    unique_batches = torch.unique(batch_mask.detach().cpu()).tolist()
    for batch_id in unique_batches:
        lig_sel = batch_mask == int(batch_id)
        if int(lig_sel.sum().item()) < 2:
            continue

        pocket_x = None
        if pocket_positions is not None and pocket_mask is not None:
            pocket_sel = pocket_mask == int(batch_id)
            if int(pocket_sel.sum().item()) > 0:
                pocket_x = pocket_positions[pocket_sel].to(
                    device=positions.device, dtype=positions.dtype
                )

        x0 = positions[lig_sel]
        types = atom_types[lig_sel]
        refined_single, before, after = refine_single_ligand_positions(
            x0,
            types,
            dataset_info,
            config,
            pocket_positions=pocket_x,
        )
        refined[lig_sel] = refined_single
        energy_before.append(before)
        energy_after.append(after)

    shifts = torch.linalg.norm(refined - positions.detach(), dim=-1)
    stats = GeometryRefineStats(
        mean_shift=float(shifts.mean().item()) if len(shifts) > 0 else 0.0,
        max_shift=float(shifts.max().item()) if len(shifts) > 0 else 0.0,
        mean_energy_before=float(sum(energy_before) / len(energy_before))
        if energy_before else 0.0,
        mean_energy_after=float(sum(energy_after) / len(energy_after))
        if energy_after else 0.0,
    )
    return refined, stats


def refine_single_ligand_positions(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    dataset_info: Mapping[str, object],
    config: GeometryRefineConfig,
    *,
    pocket_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float]:
    """Apply clipped geometry-guidance updates to one generated ligand."""
    x0 = positions.detach()
    pair_data = _build_pair_data(x0, atom_types, dataset_info, config)
    if pair_data is None:
        return x0.clone(), 0.0, 0.0

    with torch.enable_grad():
        before = float(
            _geometry_energy(x0, x0, atom_types, pair_data, config, pocket_positions).item()
        )
        x = x0.clone()
        for _ in range(config.steps):
            x_var = x.detach().clone().requires_grad_(True)
            energy = _geometry_energy(
                x_var, x0, atom_types, pair_data, config, pocket_positions
            )
            grad = torch.autograd.grad(energy, x_var, allow_unused=False)[0]
            if not torch.isfinite(grad).all():
                break

            delta = -float(config.learning_rate) * grad
            delta = _clip_vector_norm(delta, config.max_step)
            proposed = x.detach() + delta
            total_delta = _clip_vector_norm(proposed - x0, config.max_total_shift)
            x = x0 + total_delta

        after = float(
            _geometry_energy(x, x0, atom_types, pair_data, config, pocket_positions).item()
        )
    return x.detach(), before, after


def _build_pair_data(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    dataset_info: Mapping[str, object],
    config: GeometryRefineConfig,
) -> dict[str, torch.Tensor] | None:
    """Create fixed pair metadata for stable final-only refinement."""
    n_atoms = int(positions.shape[0])
    if n_atoms < 2:
        return None

    pair_index = torch.triu_indices(n_atoms, n_atoms, offset=1, device=positions.device)
    atom_i = atom_types[pair_index[0]].long()
    atom_j = atom_types[pair_index[1]].long()

    bond_tables = [
        _pair_table(dataset_info, key, positions)
        for key in ("bonds1", "bonds2", "bonds3")
    ]
    bond_stack = torch.stack(
        [table[atom_i, atom_j] for table in bond_tables],
        dim=0,
    )
    has_bond_value = bond_stack > 0
    large = torch.full_like(bond_stack, 1e6)
    shortest_bond = torch.where(has_bond_value, bond_stack, large).min(dim=0).values
    shortest_bond = torch.where(shortest_bond >= 1e6, torch.zeros_like(shortest_bond), shortest_bond)
    single_bond = bond_tables[0][atom_i, atom_j]

    d0 = torch.linalg.norm(
        positions[pair_index[0]] - positions[pair_index[1]], dim=-1
    ).detach()
    plausible = (shortest_bond > 0) & (d0 <= float(config.candidate_cutoff))
    upper = single_bond + float(config.bond_upper_margin)
    lower = shortest_bond * float(config.bond_lower_factor)
    min_distance = shortest_bond * float(config.min_distance_factor)

    # Fixed BInD-style candidate weights avoid an optimization loophole where a
    # pair can reduce its own guidance weight instead of correcting geometry.
    candidate_score = torch.sigmoid((upper - d0) / float(config.soft_tau))
    candidate_weight = torch.where(plausible, candidate_score, torch.zeros_like(d0))

    return {
        "pair_index": pair_index,
        "atom_i": atom_i,
        "atom_j": atom_j,
        "plausible": plausible,
        "candidate_weight": candidate_weight.detach(),
        "lower": lower.detach(),
        "upper": upper.detach(),
        "min_distance": min_distance.detach(),
    }


def _geometry_energy(
    x: torch.Tensor,
    x0: torch.Tensor,
    atom_types: torch.Tensor,
    pair_data: Mapping[str, torch.Tensor],
    config: GeometryRefineConfig,
    pocket_positions: torch.Tensor | None,
) -> torch.Tensor:
    """Compute differentiable geometry guidance energy."""
    pair_index = pair_data["pair_index"]
    d = torch.linalg.norm(x[pair_index[0]] - x[pair_index[1]], dim=-1).clamp_min(1e-6)
    plausible = pair_data["plausible"]

    close = F.relu(pair_data["min_distance"] - d).pow(2)
    close = torch.where(plausible, close, torch.zeros_like(close)).mean()

    w_fixed = pair_data["candidate_weight"]
    bond_window = (
        F.relu(pair_data["lower"] - d).pow(2)
        + F.relu(d - pair_data["upper"]).pow(2)
    )
    bond_window = (w_fixed * bond_window).sum() / w_fixed.sum().clamp_min(1.0)

    degree = _soft_degree(x, atom_types, pair_data, config)
    max_degree = _allowed_degree(atom_types, config, x)
    degree_energy = F.relu(degree - max_degree).pow(2).mean()

    triangle_energy = _soft_triangle_energy(x, pair_data, config)

    if config.use_pocket_clash and pocket_positions is not None and len(pocket_positions) > 0:
        pocket_d = torch.cdist(x, pocket_positions).clamp_min(1e-6)
        pocket_energy = F.relu(float(config.pocket_clash_distance) - pocket_d).pow(2).mean()
    else:
        pocket_energy = x.new_tensor(0.0)

    tether = torch.linalg.norm(x - x0, dim=-1).pow(2).mean()

    return (
        float(config.close_pair_weight) * close
        + float(config.bond_length_weight) * bond_window
        + float(config.degree_weight) * degree_energy
        + float(config.triangle_weight) * triangle_energy
        + float(config.pocket_clash_weight) * pocket_energy
        + float(config.tether_weight) * tether
    )


def _soft_degree(
    x: torch.Tensor,
    atom_types: torch.Tensor,
    pair_data: Mapping[str, torch.Tensor],
    config: GeometryRefineConfig,
) -> torch.Tensor:
    """Return differentiable candidate-bond degree per atom."""
    del atom_types
    pair_index = pair_data["pair_index"]
    d = torch.linalg.norm(x[pair_index[0]] - x[pair_index[1]], dim=-1).clamp_min(1e-6)
    plausible = pair_data["plausible"].to(dtype=x.dtype)
    w = torch.sigmoid((pair_data["upper"] - d) / float(config.soft_tau)) * plausible
    degree = x.new_zeros(x.shape[0])
    degree.scatter_add_(0, pair_index[0], w)
    degree.scatter_add_(0, pair_index[1], w)
    return degree


def _soft_triangle_energy(
    x: torch.Tensor,
    pair_data: Mapping[str, torch.Tensor],
    config: GeometryRefineConfig,
) -> torch.Tensor:
    """Approximate triangle pressure from a differentiable soft adjacency."""
    n_atoms = int(x.shape[0])
    if n_atoms < 3 or n_atoms > int(config.max_triangle_atoms):
        return x.new_tensor(0.0)

    pair_index = pair_data["pair_index"]
    d = torch.linalg.norm(x[pair_index[0]] - x[pair_index[1]], dim=-1).clamp_min(1e-6)
    plausible = pair_data["plausible"].to(dtype=x.dtype)
    w = torch.sigmoid((pair_data["upper"] - d) / float(config.soft_tau)) * plausible
    adj = x.new_zeros((n_atoms, n_atoms))
    adj[pair_index[0], pair_index[1]] = w
    adj[pair_index[1], pair_index[0]] = w
    triangle = torch.trace(adj @ adj @ adj) / 6.0
    normalizer = max(float(n_atoms), 1.0)
    return triangle / normalizer


def _allowed_degree(
    atom_types: torch.Tensor,
    config: GeometryRefineConfig,
    x: torch.Tensor,
) -> torch.Tensor:
    """Return conservative valence caps for DiffSBDD atom type indices."""
    del config
    # DiffSBDD crossdock/bindingmoad ordering: C,N,O,S,B,Br,Cl,P,I,F.
    caps_by_index = x.new_tensor([4.0, 3.0, 2.0, 4.0, 3.0, 1.0, 1.0, 5.0, 1.0, 1.0])
    caps = x.new_full((len(atom_types),), 4.0)
    in_range = atom_types.long() < len(caps_by_index)
    caps[in_range] = caps_by_index[atom_types.long()[in_range]]
    return caps


def _pair_table(
    dataset_info: Mapping[str, object],
    key: str,
    like: torch.Tensor,
) -> torch.Tensor:
    """Return a pm-to-angstrom pair table tensor."""
    values = dataset_info.get(key)
    if values is None:
        raise ValueError(f"dataset_info is missing pair table {key!r}.")
    return torch.as_tensor(values, device=like.device, dtype=like.dtype) / 100.0


def _clip_vector_norm(vectors: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip per-row vector norms without changing zero vectors."""
    if max_norm <= 0:
        return torch.zeros_like(vectors)
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(float(max_norm) / norms, max=1.0)
    return vectors * scale
