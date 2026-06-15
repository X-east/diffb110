"""Generate short-run configs for DiffSBDD logprob mode comparison."""

from __future__ import annotations

from pathlib import Path

import yaml


BASE_CONFIG = Path("OUT/qed-sa-docking-8gjl-8gpu-kl-0p02_20260511_171944/config.yaml")
OUT_DIR = Path("launch/generated/logprob_modes")
EPOCHS = 5

MODES = {
    "mean_per_element": {
        "logprob_mode": "mean_per_element",
        "logprob_clamp_min": -100.0,
        "logprob_clamp_max": 0.0,
    },
    "mean_per_atom": {
        "logprob_mode": "mean_per_atom",
        "logprob_clamp_min": -100.0,
        "logprob_clamp_max": 0.0,
    },
    "full_mean_per_atom": {
        "logprob_mode": "full_mean_per_atom",
        "logprob_clamp_min": -100.0,
        "logprob_clamp_max": None,
    },
}


def main() -> None:
    """Write one YAML config per logprob mode."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    for mode, settings in MODES.items():
        config = dict(base)
        config["name"] = f"logprob-mode-{mode}-kl0p02-5ep"
        config.setdefault("train", {})["epochs"] = EPOCHS
        config.setdefault("model", {}).update(settings)
        out_path = OUT_DIR / f"8gpu_kl0p02_5ep_{mode}.yaml"
        out_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(out_path)


if __name__ == "__main__":
    main()
