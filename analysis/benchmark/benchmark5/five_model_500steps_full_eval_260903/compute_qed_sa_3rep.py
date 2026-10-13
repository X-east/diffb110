"""QED and SA score per model for the Table 2.1 batch (500 steps, 3 repeats).

QED is read back from the per-repeat model_summary.csv (qed_mean over
sanitize-valid molecules). SA is recomputed from the canonical SMILES with
RDKit's sascorer (Ertl & Schuffenhauer), since no summary carries it.

Reps: rep4_seed22_29 / rep5_seed30_37 / rep6_seed38_45
      = the three repeats behind thesis Tables 2.1 / 3.2.

Output: model_qed_sa_500steps_3rep.csv (next to this script)
"""

from pathlib import Path
import os
import sys
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")

H = Path(__file__).resolve().parent
REPEATS = ["rep4_seed22_29", "rep5_seed30_37", "rep6_seed38_45"]
BASE = H.parent / "five_model_repeats_500steps_260826"
DIRNAME = {"DiffGui": "diffgui", "DiffSBDD": "diffsbdd", "GCDM-SBDD": "gcdm", "PMDM": "pmdm", "TargetDiff": "targetdiff"}
ORDER = list(DIRNAME)


def sa_of(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else float(sascorer.calculateScore(mol))


rows = []
for rep in REPEATS:
    ev = BASE / rep / "evaluation/sanitize_valid_only"
    summ = pd.read_csv(ev / "model_summary.csv").set_index("model")
    per = pd.read_csv(ev / "per_molecule_metrics.csv")
    for model in ORDER:
        sub = per[per["model"] == DIRNAME[model]]
        vals = [sa_of(s) for s in sub["smiles"].astype(str)]
        vals = [v for v in vals if v is not None]
        rows.append(
            {
                "rep": rep,
                "model": model,
                "qed_mean": float(summ.loc[DIRNAME[model], "qed_mean"]),
                "sa_mean": sum(vals) / len(vals),
                "n": len(vals),
            }
        )
t = pd.DataFrame(rows)
out = H / "model_qed_sa_500steps_3rep.csv"
t.to_csv(out, index=False)

print(t.pivot(index="model", columns="rep", values="sa_mean").loc[ORDER].round(3).to_string())
print()
print(t.groupby("model")[["qed_mean", "sa_mean"]].mean().loc[ORDER].round(4).to_string())
print("\nWrote", out)
