#!/usr/bin/env python
"""Build structure-driven ESRPO preference pairs from R1-R5 pathology flags.

Unlike the docking-threshold builder (``build_alidiff_preference_pairs.py``),
this builder selects the good/bad pools primarily on **structural sanity**
(R1-R4) rather than docking score. R5 is retained as an independent carbon
drift guard, but is not used to define good or bad. Motivation: the docking-selected "good"
pool is ~69% structurally pathological, so a docking-only label teaches the
model to prefer large, complex, strained molecules that score well.

The output schema is identical to the docking builder so the existing ESRPO
trainer can consume it unchanged.

Pool definitions (configurable)
-------------------------------
good : ``clean``            structurally clean (no R1-R4 violation)
       ``clean_dock``       clean AND docking_raw <= --good-dock-max
       ``dock_only``        legacy docking-threshold behaviour
bad  : ``all_patho``        any R1-R4 structural violation
       ``min_viol``         structural_violation_count >= --min-violation
       ``dock_only``        legacy docking-threshold behaviour

Reward
------
continuous : reward = -structural_violation_count (structurally cleaner -> higher reward),
             with a small docking tie-break controlled by --dock-weight.
binary     : winner = 1.0, loser = 0.0.
legacy     : reward = -docking_raw (original docking behaviour).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors

LOGGER = logging.getLogger("DiffSBDD-StructurePairs")

# Output schema kept identical to build_alidiff_preference_pairs.py.
CSV_COLUMNS = [
    "pair_index",
    "split",
    "winner_id",
    "loser_id",
    "winner_docking_raw",
    "loser_docking_raw",
    "reward_winner",
    "reward_loser",
    "reward_gap",
    "winner_shard",
    "loser_shard",
    "winner_index",
    "loser_index",
    "winner_n_nodes",
    "loser_n_nodes",
    "n_nodes",
    "source_rule",
]

# Additional columns that only this builder produces.
EXTRA_COLUMNS = [
    "winner_structural_violation_count",
    "loser_structural_violation_count",
    "structural_violation_gap",
    "winner_r5_carbon_guard_flag",
    "loser_r5_carbon_guard_flag",
    "winner_heavy_atom_count",
    "loser_heavy_atom_count",
    "winner_hetero_atom_count",
    "loser_hetero_atom_count",
    "winner_mw",
    "loser_mw",
    "winner_tpsa",
    "loser_tpsa",
    "winner_logp",
    "loser_logp",
    "abs_node_gap",
    "abs_mw_gap",
    "abs_tpsa_gap",
    "abs_logp_gap",
    "abs_docking_gap",
    "match_cost",
    "composition_key",
]

RULE_FIELDS = (
    "r1_large_ring_violation",
    "r2_ring_system_violation",
    "r3_four_ring_violation",
    "r4a_three_ring_hetero_violation",
    "r4b_three_ring_over_attached_violation",
    "r4c_three_ring_spiro_or_fused_violation",
    "r5_carbon_fraction_violation",
)

# R5 is a carbon-composition drift guard, not a structural bad-label rule.
STRUCTURAL_RULE_FIELDS = RULE_FIELDS[:-1]
SAMPLE_ID_RE = re.compile(r"^r\d+_s\d+_m(?P<molecule_index>\d+)$")


@dataclass(frozen=True)
class Pair:
    """One structural preference pair."""

    pair_index: int
    split: str
    winner_id: str
    loser_id: str
    winner_docking_raw: float
    loser_docking_raw: float
    reward_winner: float
    reward_loser: float
    reward_gap: float
    winner_shard: str
    loser_shard: str
    winner_index: int
    loser_index: int
    winner_n_nodes: int
    loser_n_nodes: int
    n_nodes: int
    source_rule: str
    winner_structural_violation_count: int
    loser_structural_violation_count: int
    structural_violation_gap: int
    winner_r5_carbon_guard_flag: int
    loser_r5_carbon_guard_flag: int
    winner_heavy_atom_count: int
    loser_heavy_atom_count: int
    winner_hetero_atom_count: int
    loser_hetero_atom_count: int
    winner_mw: float
    loser_mw: float
    winner_tpsa: float
    loser_tpsa: float
    winner_logp: float
    loser_logp: float
    abs_node_gap: int
    abs_mw_gap: float
    abs_tpsa_gap: float
    abs_logp_gap: float
    abs_docking_gap: float
    match_cost: float
    composition_key: str


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a JSONL file into a list of dicts."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of dicts."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_value(value: object, default: int = 0) -> int:
    """Convert a sidecar value to an integer, treating blanks as default."""
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected integer-like value, got {value!r}") from exc


def _finite_float(value: object, *, field: str) -> float:
    """Convert a value to a finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def _structural_violation_count(pathology_row: dict[str, str]) -> int:
    """Return the R1-R4 count, explicitly excluding R5."""
    explicit = pathology_row.get("structural_violation_count")
    if explicit not in (None, ""):
        return _int_value(explicit)
    return sum(
        _int_value(pathology_row.get(field, 0))
        for field in STRUCTURAL_RULE_FIELDS
    )


def _r5_flag(pathology_row: dict[str, str]) -> int:
    """Return R5 as a separate carbon-drift guard flag."""
    return _int_value(pathology_row.get("r5_carbon_fraction_violation", 0))


def _composition_descriptor(mol: Chem.Mol) -> tuple[str, int, int, float]:
    """Return exact heavy-atom formula key and composition statistics."""
    counts = Counter(
        atom.GetAtomicNum()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() >= 2
    )
    composition_key = ";".join(
        f"{atomic_num}:{counts[atomic_num]}" for atomic_num in sorted(counts)
    )
    heavy_atom_count = int(sum(counts.values()))
    carbon_count = int(counts.get(6, 0))
    hetero_atom_count = heavy_atom_count - carbon_count
    carbon_fraction = carbon_count / heavy_atom_count if heavy_atom_count else 0.0
    return composition_key, heavy_atom_count, hetero_atom_count, carbon_fraction


def _candidate_record(
    episode_row: dict[str, Any],
    pathology_row: dict[str, str],
    reward_row: dict[str, str],
    source_run_dir: Path,
) -> dict[str, Any] | None:
    """Build a descriptor-rich candidate from the frozen input sidecars."""
    if str(reward_row.get("docking_success", "1")) != "1":
        return None
    try:
        docking_raw = _finite_float(reward_row.get("docking_raw"), field="docking_raw")
    except ValueError:
        return None

    smiles = str(episode_row.get("smiles") or pathology_row.get("smiles") or "")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    n_nodes = _int_value(episode_row.get("num_nodes"))
    if n_nodes <= 0:
        return None
    sample_match = SAMPLE_ID_RE.fullmatch(str(episode_row["sample_id"]))
    if sample_match is None:
        raise ValueError(f"invalid sample_id format: {episode_row['sample_id']}")
    shard_value = Path(str(episode_row.get("shard_path", "")))
    shard_path = shard_value if shard_value.is_absolute() else source_run_dir / shard_value
    composition_key, heavy_atom_count, hetero_atom_count, carbon_fraction = (
        _composition_descriptor(mol)
    )
    return {
        "sample_id": str(episode_row["sample_id"]),
        "shard": str(shard_path.resolve()),
        # shard_index is the shard number; mXXXX is the sample's position
        # inside that shard and is the index required by EndpointStore.
        "index": int(sample_match.group("molecule_index")),
        "n_nodes": n_nodes,
        "docking_raw": docking_raw,
        "structural_violation_count": _structural_violation_count(pathology_row),
        "r5_flag": _r5_flag(pathology_row),
        "composition_key": composition_key,
        "heavy_atom_count": heavy_atom_count,
        "hetero_atom_count": hetero_atom_count,
        "carbon_fraction": carbon_fraction,
        "mw": float(Descriptors.MolWt(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
    }


def _match_cost(
    winner: dict[str, Any],
    loser: dict[str, Any],
    *,
    tpsa_scale: float,
    logp_scale: float,
    docking_scale: float,
) -> float:
    """Calculate a normalized secondary matching cost."""
    return (
        abs(float(winner["tpsa"]) - float(loser["tpsa"])) / tpsa_scale
        + abs(float(winner["logp"]) - float(loser["logp"])) / logp_scale
        + abs(float(winner["docking_raw"]) - float(loser["docking_raw"])) / docking_scale
    )


def _eligible_match(
    winner: dict[str, Any],
    loser: dict[str, Any],
    *,
    max_tpsa_gap: float,
    max_docking_gap: float,
) -> bool:
    """Check hard formula, node-count, TPSA, and docking constraints."""
    return (
        winner["composition_key"] == loser["composition_key"]
        and winner["n_nodes"] == loser["n_nodes"]
        and abs(float(winner["tpsa"]) - float(loser["tpsa"])) <= max_tpsa_gap
        and abs(float(winner["docking_raw"]) - float(loser["docking_raw"])) <= max_docking_gap
    )


def _best_bad_index(
    winner: dict[str, Any],
    bad: list[dict[str, Any]],
    unused_bad: set[int],
    *,
    max_tpsa_gap: float,
    max_docking_gap: float,
    tpsa_scale: float,
    logp_scale: float,
    docking_scale: float,
) -> tuple[int, float] | None:
    """Find the nearest currently unused bad endpoint."""
    candidates: list[tuple[float, str, int]] = []
    for index in unused_bad:
        loser = bad[index]
        if not _eligible_match(
            winner,
            loser,
            max_tpsa_gap=max_tpsa_gap,
            max_docking_gap=max_docking_gap,
        ):
            continue
        candidates.append(
            (
                _match_cost(
                    winner,
                    loser,
                    tpsa_scale=tpsa_scale,
                    logp_scale=logp_scale,
                    docking_scale=docking_scale,
                ),
                str(loser["sample_id"]),
                index,
            )
        )
    if not candidates:
        return None
    cost, _, index = min(candidates)
    return index, cost


def build_pairs(
    groups: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    reward_mode: str,
    dock_weight: float,
    max_tpsa_gap: float,
    max_docking_gap: float,
    tpsa_scale: float,
    logp_scale: float,
    docking_scale: float,
    source_rule: str,
) -> tuple[dict[str, list[Pair]], dict[str, dict[str, int]]]:
    """Build unique condition-matched pairs independently within each split.

    The primary key is ``(n_nodes, exact heavy-atom composition)``.  Within
    each key, each winner is assigned to the unused loser with the smallest
    normalized TPSA/logP/docking cost, subject to hard TPSA and docking
    tolerances.  No endpoint is reused.
    """
    outputs: dict[str, list[Pair]] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    pair_index = 0

    for split in ("train", "validation", "test"):
        split_groups = groups.get(split, {"good": [], "bad": []})
        good = split_groups["good"]
        bad = split_groups["bad"]
        bad_by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for record in bad:
            bad_by_key[(int(record["n_nodes"]), str(record["composition_key"]))].append(record)
        good_by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for record in good:
            good_by_key[(int(record["n_nodes"]), str(record["composition_key"]))].append(record)

        pairs: list[Pair] = []
        unmatched_good = 0
        candidate_groups = 0
        for key, good_group in sorted(good_by_key.items()):
            bad_group = bad_by_key.get(key, [])
            if not bad_group:
                unmatched_good += len(good_group)
                continue
            candidate_groups += 1
            unused_bad = set(range(len(bad_group)))
            # Most constrained winners are matched first, avoiding a greedy
            # assignment consuming a rare candidate needed by another winner.
            constrained: list[tuple[int, str, dict[str, Any]]] = []
            for winner in good_group:
                candidate_count = sum(
                    1
                    for loser in bad_group
                    if _eligible_match(
                        winner,
                        loser,
                        max_tpsa_gap=max_tpsa_gap,
                        max_docking_gap=max_docking_gap,
                    )
                )
                constrained.append((candidate_count, str(winner["sample_id"]), winner))

            for candidate_count, _, winner in sorted(constrained):
                if candidate_count == 0:
                    unmatched_good += 1
                    continue
                candidates: list[tuple[float, str, int]] = []
                for bad_index in unused_bad:
                    loser = bad_group[bad_index]
                    if not _eligible_match(
                        winner,
                        loser,
                        max_tpsa_gap=max_tpsa_gap,
                        max_docking_gap=max_docking_gap,
                    ):
                        continue
                    cost = (
                        abs(float(winner["tpsa"]) - float(loser["tpsa"])) / tpsa_scale
                        + abs(float(winner["logp"]) - float(loser["logp"])) / logp_scale
                        + abs(float(winner["docking_raw"]) - float(loser["docking_raw"])) / docking_scale
                    )
                    candidates.append((cost, str(loser["sample_id"]), bad_index))
                if not candidates:
                    unmatched_good += 1
                    continue
                cost, _, bad_index = min(candidates)
                unused_bad.remove(bad_index)
                loser = bad_group[bad_index]
                w_struct = int(winner["structural_violation_count"])
                l_struct = int(loser["structural_violation_count"])
                w_dock = float(winner["docking_raw"])
                l_dock = float(loser["docking_raw"])
                if reward_mode == "binary":
                    rw, rl = 1.0, 0.0
                elif reward_mode == "legacy":
                    rw, rl = -w_dock, -l_dock
                else:
                    rw = -float(w_struct) - dock_weight * w_dock
                    rl = -float(l_struct) - dock_weight * l_dock
                pairs.append(
                    Pair(
                        pair_index=pair_index,
                        split=split,
                        winner_id=str(winner["sample_id"]),
                        loser_id=str(loser["sample_id"]),
                        winner_docking_raw=w_dock,
                        loser_docking_raw=l_dock,
                        reward_winner=rw,
                        reward_loser=rl,
                        reward_gap=rw - rl,
                        winner_shard=str(winner["shard"]),
                        loser_shard=str(loser["shard"]),
                        winner_index=int(winner["index"]),
                        loser_index=int(loser["index"]),
                        winner_n_nodes=int(winner["n_nodes"]),
                        loser_n_nodes=int(loser["n_nodes"]),
                        n_nodes=int(winner["n_nodes"]),
                        source_rule=source_rule,
                        winner_structural_violation_count=w_struct,
                        loser_structural_violation_count=l_struct,
                        structural_violation_gap=l_struct - w_struct,
                        winner_r5_carbon_guard_flag=int(winner["r5_flag"]),
                        loser_r5_carbon_guard_flag=int(loser["r5_flag"]),
                        winner_heavy_atom_count=int(winner["heavy_atom_count"]),
                        loser_heavy_atom_count=int(loser["heavy_atom_count"]),
                        winner_hetero_atom_count=int(winner["hetero_atom_count"]),
                        loser_hetero_atom_count=int(loser["hetero_atom_count"]),
                        winner_mw=float(winner["mw"]),
                        loser_mw=float(loser["mw"]),
                        winner_tpsa=float(winner["tpsa"]),
                        loser_tpsa=float(loser["tpsa"]),
                        winner_logp=float(winner["logp"]),
                        loser_logp=float(loser["logp"]),
                        abs_node_gap=abs(int(winner["n_nodes"]) - int(loser["n_nodes"])),
                        abs_mw_gap=abs(float(winner["mw"]) - float(loser["mw"])),
                        abs_tpsa_gap=abs(float(winner["tpsa"]) - float(loser["tpsa"])),
                        abs_logp_gap=abs(float(winner["logp"]) - float(loser["logp"])),
                        abs_docking_gap=abs(w_dock - l_dock),
                        match_cost=cost,
                        composition_key=str(winner["composition_key"]),
                    )
                )
                pair_index += 1
        outputs[split] = pairs
        diagnostics[split] = {
            "good_pool": len(good),
            "bad_pool": len(bad),
            "composition_groups_with_candidates": candidate_groups,
            "pairs": len(pairs),
            "unmatched_good": unmatched_good,
            "unused_bad": len(bad) - len(pairs),
        }
    return outputs, diagnostics


def main() -> None:
    """Build and audit condition-matched train/validation/test manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--pathology-sidecar", type=Path, required=True,
                        help="structure_pathology_sidecar.csv from R1-R5 builder")
    parser.add_argument("--reward-sidecar", type=Path, required=True,
                        help="reward_sidecar.csv containing docking_raw")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True,
                        help="Directory containing the relative shards/ tree.")
    parser.add_argument("--good-mode", choices=["clean", "clean_dock", "dock_only"],
                        default="clean")
    parser.add_argument("--bad-mode", choices=["all_patho", "min_viol", "dock_only"],
                        default="min_viol")
    parser.add_argument("--good-dock-max", type=float, default=-7.0)
    parser.add_argument("--bad-dock-min", type=float, default=-5.0)
    parser.add_argument("--min-violation", type=int, default=1,
                        help="Minimum R1-R4 count for the bad pool; R5 is excluded.")
    parser.add_argument("--reward-mode", choices=["continuous", "binary", "legacy"],
                        default="continuous")
    parser.add_argument("--dock-weight", type=float, default=0.0,
                        help="Tie-break weight on docking for continuous reward.")
    parser.add_argument("--max-tpsa-gap", type=float, default=30.0)
    parser.add_argument("--max-docking-gap", type=float, default=1.5)
    parser.add_argument("--tpsa-scale", type=float, default=30.0)
    parser.add_argument("--logp-scale", type=float, default=1.0)
    parser.add_argument("--docking-scale", type=float, default=1.0)
    parser.add_argument("--source-rule", default="structure_r1_r4_exact_formula_tpsa_dock_nn_v2")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for path in (
        args.episode_index,
        args.split_manifest,
        args.pathology_sidecar,
        args.reward_sidecar,
        args.source_run_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.min_violation <= 0:
        raise ValueError("--min-violation must be positive")
    if args.max_tpsa_gap < 0 or args.max_docking_gap < 0:
        raise ValueError("descriptor tolerances must be non-negative")
    for name in ("tpsa_scale", "logp_scale", "docking_scale"):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("loading inputs")
    episode_rows = load_jsonl(args.episode_index)
    split_rows = load_jsonl(args.split_manifest)
    pathology_rows = load_csv(args.pathology_sidecar)
    reward_rows = load_csv(args.reward_sidecar)
    pathology = {row["sample_id"]: row for row in pathology_rows}
    rewards = {row["sample_id"]: row for row in reward_rows}
    if len(pathology) != len(pathology_rows):
        raise ValueError("pathology sidecar contains duplicated sample_id")
    if len(rewards) != len(reward_rows):
        raise ValueError("reward sidecar contains duplicated sample_id")

    # The split manifest stores one row per sample_id (the offline label
    # splitter writes per-sample rows, not a sample_ids list).
    split_of: dict[str, str] = {}
    for row in split_rows:
        sample_ids = row.get("sample_ids")
        if isinstance(sample_ids, list):
            for sample_id in sample_ids:
                if str(sample_id) in split_of:
                    raise ValueError(f"duplicate split assignment for {sample_id}")
                split_of[str(sample_id)] = str(row.get("split", "train"))
        elif "sample_id" in row:
            if str(row["sample_id"]) in split_of:
                raise ValueError(f"duplicate split assignment for {row['sample_id']}")
            split_of[str(row["sample_id"])] = str(row.get("split", "train"))

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "train": {"good": [], "bad": []},
        "validation": {"good": [], "bad": []},
        "test": {"good": [], "bad": []},
    }
    skipped = Counter()
    seen_ids: set[str] = set()

    for row in episode_rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id in seen_ids:
            raise ValueError(f"duplicate episode sample_id: {sample_id}")
        seen_ids.add(sample_id)
        if not row.get("rdkit_valid"):
            skipped["not_rdkit_valid"] += 1
            continue
        split = split_of.get(sample_id)
        if split not in groups:
            skipped["no_split"] += 1
            continue
        path = pathology.get(sample_id)
        rew = rewards.get(sample_id)
        if path is None:
            skipped["no_pathology"] += 1
            continue
        if rew is None:
            skipped["no_reward"] += 1
            continue
        record = _candidate_record(
            row,
            path,
            rew,
            args.source_run_dir.resolve(),
        )
        if record is None:
            skipped["unusable_descriptor_or_docking"] += 1
            continue
        skipped["eligible"] += 1
        good_list = groups[split]["good"]
        bad_list = groups[split]["bad"]
        structural_count = int(record["structural_violation_count"])
        docking_raw = float(record["docking_raw"])

        if args.good_mode == "clean":
            if structural_count == 0:
                good_list.append(record)
        elif args.good_mode == "clean_dock":
            if structural_count == 0 and docking_raw <= args.good_dock_max:
                good_list.append(record)
        else:  # dock_only
            if docking_raw <= args.good_dock_max:
                good_list.append(record)

        if args.bad_mode == "all_patho":
            if structural_count > 0:
                bad_list.append(record)
        elif args.bad_mode == "min_viol":
            if structural_count >= args.min_violation:
                bad_list.append(record)
        else:  # dock_only
            if docking_raw >= args.bad_dock_min:
                bad_list.append(record)

    LOGGER.info("candidate accounting: %s", dict(skipped))
    for split in groups:
        LOGGER.info(
            "split %s: good=%d bad=%d",
            split,
            len(groups[split]["good"]),
            len(groups[split]["bad"]),
        )

    outputs, diagnostics = build_pairs(
        groups,
        reward_mode=args.reward_mode,
        dock_weight=args.dock_weight,
        max_tpsa_gap=args.max_tpsa_gap,
        max_docking_gap=args.max_docking_gap,
        tpsa_scale=args.tpsa_scale,
        logp_scale=args.logp_scale,
        docking_scale=args.docking_scale,
        source_rule=args.source_rule,
    )
    pairs = [pair for split in outputs for pair in outputs[split]]
    endpoint_ids = [
        endpoint
        for pair in pairs
        for endpoint in (pair.winner_id, pair.loser_id)
    ]
    if len(endpoint_ids) != len(set(endpoint_ids)):
        raise ValueError("matched builder reused an endpoint across output pairs")
    LOGGER.info("built matched pairs: %d", len(pairs))

    for split in ("train", "validation", "test"):
        split_path = output_dir / f"{split}_pairs.csv"
        with split_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS + EXTRA_COLUMNS)
            writer.writeheader()
            writer.writerows(asdict(pair) for pair in outputs[split])
    combined_path = output_dir / "preference_pairs.csv"
    with combined_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS + EXTRA_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(pair) for pair in pairs)

    def _mean(field: str, rows: list[Pair]) -> float:
        return sum(float(getattr(row, field)) for row in rows) / len(rows) if rows else 0.0

    pair_summary: dict[str, dict[str, float | int]] = {}
    for split in ("train", "validation", "test"):
        split_pairs = outputs[split]
        winner_dock = _mean("winner_docking_raw", split_pairs)
        loser_dock = _mean("loser_docking_raw", split_pairs)
        pair_summary[split] = {
            "pairs": len(split_pairs),
            "mean_node_gap": _mean("abs_node_gap", split_pairs),
            "mean_mw_gap": _mean("abs_mw_gap", split_pairs),
            "mean_tpsa_gap": _mean("abs_tpsa_gap", split_pairs),
            "mean_logp_gap": _mean("abs_logp_gap", split_pairs),
            "mean_abs_docking_gap": _mean("abs_docking_gap", split_pairs),
            "mean_docking_winner": winner_dock,
            "mean_docking_loser": loser_dock,
            "mean_docking_loser_minus_winner": loser_dock - winner_dock,
            "mean_structural_count_winner": _mean(
                "winner_structural_violation_count", split_pairs
            ),
            "mean_structural_count_loser": _mean(
                "loser_structural_violation_count", split_pairs
            ),
            "r5_rate_winner": _mean("winner_r5_carbon_guard_flag", split_pairs),
            "r5_rate_loser": _mean("loser_r5_carbon_guard_flag", split_pairs),
        }
    audit = {
        "status": "completed",
        "config": {key: str(value) for key, value in vars(args).items()},
        "inputs": {
            "episode_index": {"path": str(args.episode_index),
                              "sha256": sha256_file(args.episode_index)},
            "split_manifest": {"path": str(args.split_manifest),
                               "sha256": sha256_file(args.split_manifest)},
            "pathology_sidecar": {"path": str(args.pathology_sidecar),
                                  "sha256": sha256_file(args.pathology_sidecar)},
            "reward_sidecar": {"path": str(args.reward_sidecar),
                               "sha256": sha256_file(args.reward_sidecar)},
        },
        "structural_rule_fields": list(STRUCTURAL_RULE_FIELDS),
        "r5_guard_field": RULE_FIELDS[-1],
        "candidate_accounting": dict(skipped),
        "pool_sizes": {
            split: {
                "good": len(groups[split]["good"]),
                "bad": len(groups[split]["bad"]),
            }
            for split in groups
        },
        "pair_count": len(pairs),
        "matching_diagnostics": diagnostics,
        "pair_summary": pair_summary,
        "manifest_sha256": {
            split: sha256_file(output_dir / f"{split}_pairs.csv")
            for split in ("train", "validation", "test")
        },
    }
    audit_path = output_dir / "pair_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    LOGGER.info("wrote %s", audit_path)
    LOGGER.info("pair summary: %s", pair_summary)


if __name__ == "__main__":
    main()
