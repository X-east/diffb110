"""Compute R1-R5 structural pathology + physchem on DiffGui ring-last3 epoch SDFs.

Goal (P0 from 260904-01 planning doc): upgrade the "TPSA dropped" signal into a
molecule-level R1-R5 verdict, using the *actual generated SDFs* so that chapter
2 (RL success) and chapter 3 (hacking diagnosis) share the same metric language.

We compare the first full epoch (epoch_0001) against the last training epoch
(epoch_0023) of:
    runs/diffgui/diffgui-8gpu-ring-last3-lr3e5_20260610_114650/

Usage:
    conda run -n dif python analysis/diffgui_rl_r1r5_drift/compute_ring_last3_sdf_r1r5.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable so `scripts.offline_rl...` resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from scripts.offline_rl.structure_pathology_rules import (
    compute_structure_pathology_from_smiles,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("ring_last3_sdf_r1r5")

RUN_DIR = Path(
    "runs/diffgui/diffgui-8gpu-ring-last3-lr3e5_20260610_114650"
)
EPOCH_FIRST = "epoch_0001"
EPOCH_LAST = "epoch_0023"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def analyze_sdf(sdf_path: Path) -> dict[str, object]:
    """Aggregate R1-R5 pathology and physchem over an epoch SDF."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    n_total = 0
    n_parsed = 0
    carbon_fractions: list[float] = []
    tpsa: list[float] = []
    logp: list[float] = []
    qed: list[float] = []
    r1 = r2 = r3 = r4a = r4b = r4c = r5 = 0
    structural = 0
    any_path = 0

    for mol in suppl:
        n_total += 1
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:  # noqa: BLE001 - keep counting even if sanitize fails
            continue
        n_parsed += 1
        smi = Chem.MolToSmiles(mol)
        facts = compute_structure_pathology_from_smiles(smi)
        if facts is None:
            continue
        r1 += facts.r1_large_ring_violation
        r2 += facts.r2_ring_system_violation
        r3 += facts.r3_four_ring_violation
        r4a += facts.r4a_three_ring_hetero_violation
        r4b += facts.r4b_three_ring_over_attached_violation
        r4c += facts.r4c_three_ring_spiro_or_fused_violation
        r5 += facts.r5_carbon_fraction_violation
        structural += facts.is_structurally_pathological
        any_path += facts.is_pathological
        carbon_fractions.append(facts.carbon_fraction)
        try:
            tpsa.append(Descriptors.TPSA(mol))
            logp.append(Descriptors.MolLogP(mol))
            qed.append(QED.qed(mol))
        except Exception:  # noqa: BLE001
            pass

    n = max(n_parsed, 1)
    return {
        "sdf": sdf_path.name,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "R1_large_ring_%": 100.0 * r1 / n,
        "R2_ring_system_%": 100.0 * r2 / n,
        "R3_four_ring_%": 100.0 * r3 / n,
        "R4a_three_hetero_%": 100.0 * r4a / n,
        "R4b_three_over_attached_%": 100.0 * r4b / n,
        "R4c_three_spiro_fused_%": 100.0 * r4c / n,
        "R5_carbon_%": 100.0 * r5 / n,
        "R1-R4_%": 100.0 * (r1 + r2 + r3 + r4a + r4b + r4c) / n,
        "R1-R5_%": 100.0 * any_path / n,
        "structural_pathology_%": 100.0 * structural / n,
        "carbon_fraction_mean": _mean(carbon_fractions),
        "TPSA_mean": _mean(tpsa),
        "LogP_mean": _mean(logp),
        "QED_mean": _mean(qed),
    }


def main() -> None:
    results = {}
    for label, epoch in (("first", EPOCH_FIRST), ("last", EPOCH_LAST)):
        sdf = RUN_DIR / "epochs" / epoch / "molecules.sdf"
        if not sdf.exists():
            LOGGER.warning("missing %s", sdf)
            continue
        res = analyze_sdf(sdf)
        res = {"label": label, "epoch": epoch, **res}
        results[label] = res
        LOGGER.info("%s epoch=%s parsed=%d", label, epoch, res["n_parsed"])

    if len(results) == 2:
        f, l = results["first"], results["last"]
        print("\n=== DiffGui ring-last3: first vs last epoch (molecule-level) ===")
        keys = [
            "n_parsed",
            "R1_large_ring_%",
            "R2_ring_system_%",
            "R3_four_ring_%",
            "R4a_three_hetero_%",
            "R4b_three_over_attached_%",
            "R4c_three_spiro_fused_%",
            "R5_carbon_%",
            "R1-R4_%",
            "R1-R5_%",
            "structural_pathology_%",
            "carbon_fraction_mean",
            "TPSA_mean",
            "LogP_mean",
            "QED_mean",
        ]
        print(f"{'metric':<28}{'first':>12}{'last':>12}{'Δ':>10}")
        for k in keys:
            fv, lv = f[k], l[k]
            if isinstance(fv, float) and isinstance(lv, float):
                delta = lv - fv
                print(f"{k:<28}{fv:>12.3f}{lv:>12.3f}{delta:>10.3f}")
            else:
                print(f"{k:<28}{str(fv):>12}{str(lv):>12}")
        # Verdict
        hacking_signals = []
        if l["R5_carbon_%"] - f["R5_carbon_%"] > 10:
            hacking_signals.append("R5 carbon fraction violation up sharply")
        if l["carbon_fraction_mean"] - f["carbon_fraction_mean"] > 0.05:
            hacking_signals.append("mean carbon fraction up")
        if f["TPSA_mean"] - l["TPSA_mean"] > 10:
            hacking_signals.append("TPSA collapsed")
        if l["R1-R5_%"] - f["R1-R5_%"] > 5:
            hacking_signals.append("overall R1-R5 rate up")
        print("\nVerdict:")
        if hacking_signals:
            print("  REWARD HACKING suspected: " + "; ".join(hacking_signals))
        else:
            print("  No classic hacking signal; molecules did not collapse to carbon.")


if __name__ == "__main__":
    main()
