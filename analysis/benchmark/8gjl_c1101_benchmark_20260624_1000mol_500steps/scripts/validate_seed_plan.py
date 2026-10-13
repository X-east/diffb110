"""Validate sharded benchmark seed plans before launching generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer field {field!r} in row: {row}") from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is available in project envs.
        raise RuntimeError("PyYAML is required to validate generated config files") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return data


def _validate_config(row: dict[str, str], errors: list[str]) -> None:
    model = row["model"]
    config_text = row.get("config_path", "")
    if not config_text:
        return
    config_path = Path(config_text)
    if not config_path.exists():
        errors.append(f"{model} gpu={row['gpu_id']} config missing: {config_path}")
        return

    data = _load_yaml(config_path)
    sample = data.get("sample")
    if not isinstance(sample, dict):
        errors.append(f"{model} gpu={row['gpu_id']} config has no sample mapping: {config_path}")
        return

    expected_seed = _as_int(row, "seed")
    expected_timesteps = _as_int(row, "timesteps")
    actual_seed = sample.get("seed")
    if actual_seed != expected_seed:
        errors.append(
            f"{model} gpu={row['gpu_id']} seed mismatch: "
            f"plan={expected_seed} config={actual_seed} path={config_path}"
        )

    timestep_key = "num_timesteps" if model == "diffgui" else "num_steps"
    actual_timesteps = sample.get(timestep_key)
    if actual_timesteps != expected_timesteps:
        errors.append(
            f"{model} gpu={row['gpu_id']} timestep mismatch: "
            f"plan={expected_timesteps} config[{timestep_key}]={actual_timesteps} path={config_path}"
        )


def _previous_seed_index(paths: list[Path]) -> dict[str, set[int]]:
    previous: dict[str, set[int]] = defaultdict(set)
    for path in paths:
        if not path.exists():
            continue
        for row in _read_csv(path):
            model = row.get("model", "").strip()
            if not model or "seed" not in row:
                continue
            try:
                previous[model].add(int(row["seed"]))
            except ValueError:
                continue
    return previous


def validate(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_csv(args.seed_plan)
    models = args.models.split(",")
    gpus = [int(item) for item in args.gpus.split(",")]
    counts = [int(item) for item in args.counts.split(",")]
    if len(gpus) != len(counts):
        raise ValueError("--gpus and --counts must have the same length")

    expected_counts = dict(zip(gpus, counts, strict=True))
    expected_keys = {(model, gpu) for model in models for gpu in gpus}
    seen_keys: set[tuple[str, int]] = set()
    rows_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    errors: list[str] = []

    required_fields = {"model", "gpu_id", "count", "seed", "timesteps", "output_dir"}
    previous = _previous_seed_index(args.previous_manifest)

    for row in rows:
        missing = sorted(required_fields - set(row))
        if missing:
            errors.append(f"row missing fields {missing}: {row}")
            continue
        model = row["model"]
        gpu_id = _as_int(row, "gpu_id")
        count = _as_int(row, "count")
        seed = _as_int(row, "seed")
        timesteps = _as_int(row, "timesteps")
        key = (model, gpu_id)
        if key in seen_keys:
            errors.append(f"duplicate row for model={model} gpu={gpu_id}")
        seen_keys.add(key)
        rows_by_model[model].append(row)

        if key not in expected_keys:
            errors.append(f"unexpected model/gpu row: model={model} gpu={gpu_id}")
        if count != expected_counts.get(gpu_id):
            errors.append(
                f"{model} gpu={gpu_id} count mismatch: plan={count} "
                f"expected={expected_counts.get(gpu_id)}"
            )
        expected_seed = gpu_id + args.seed_offset
        if seed != expected_seed:
            errors.append(f"{model} gpu={gpu_id} seed mismatch: plan={seed} expected={expected_seed}")
        if timesteps != args.timesteps:
            errors.append(f"{model} gpu={gpu_id} timesteps mismatch: plan={timesteps} expected={args.timesteps}")
        if args.replicate is not None and row.get("replicate") not in {"", str(args.replicate)}:
            errors.append(f"{model} gpu={gpu_id} replicate mismatch: plan={row.get('replicate')} expected={args.replicate}")
        if seed in previous.get(model, set()):
            errors.append(f"{model} gpu={gpu_id} seed={seed} overlaps with previous manifest")
        if args.check_output_empty:
            output_dir = Path(row["output_dir"])
            if output_dir.exists() and any(output_dir.iterdir()):
                errors.append(f"{model} gpu={gpu_id} output_dir is not empty: {output_dir}")
        _validate_config(row, errors)

    missing_keys = sorted(expected_keys - seen_keys)
    for model, gpu_id in missing_keys:
        errors.append(f"missing row for model={model} gpu={gpu_id}")

    expected_total = sum(counts)
    per_model_totals: dict[str, int] = {}
    per_model_seeds: dict[str, list[int]] = {}
    for model in models:
        model_rows = rows_by_model.get(model, [])
        total = sum(_as_int(row, "count") for row in model_rows)
        seeds = sorted(_as_int(row, "seed") for row in model_rows)
        per_model_totals[model] = total
        per_model_seeds[model] = seeds
        if total != expected_total:
            errors.append(f"{model} count total mismatch: plan={total} expected={expected_total}")
        if len(seeds) != len(set(seeds)):
            errors.append(f"{model} has duplicate seeds: {seeds}")

    summary = {
        "ok": not errors,
        "seed_plan": str(args.seed_plan),
        "replicate": args.replicate,
        "seed_offset": args.seed_offset,
        "timesteps": args.timesteps,
        "models": models,
        "gpus": gpus,
        "counts": counts,
        "per_model_totals": per_model_totals,
        "per_model_seeds": per_model_seeds,
        "errors": errors,
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--models", default="diffgui,diffsbdd,pmdm,gcdm,targetdiff")
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--counts", default="143,143,143,143,143,143,142")
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--previous-manifest", type=Path, action="append", default=[])
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--check-output-empty", action="store_true")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
