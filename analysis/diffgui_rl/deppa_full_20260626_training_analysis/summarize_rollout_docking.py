"""Pool the DiffGui main-training rollout molecules and summarise docking.

Source : runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809/
         epochs/epoch_*/molecules.csv  (64 molecules per epoch, 90 epochs)
Filter : docking_success == 1 and docking_raw < 0
         (32 failed dockings carry a 0 fallback and 343 carry NaN; both dropped)

Produces diffgui_rollout_docking_summary.csv next to this script.
Note: this is the TRAINING-ROLLOUT protocol (diffusion steps = 200), which is
NOT the same protocol as thesis Table 2.1 (500 steps, clean inference,
n = 2604, mean = -10.7886). The two must not be quoted side by side.
"""

from pathlib import Path
import glob
import pandas as pd

H = Path(__file__).resolve().parent
RUN = H.parents[2] / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809/epochs"

COLS = ["epoch", "docking_raw", "docking_success", "rdkit_valid", "qed_raw", "sa_raw"]
files = sorted(glob.glob(str(RUN / "epoch_*/molecules.csv")))
raw = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)
ok = raw[(raw["docking_success"] == 1) & (raw["docking_raw"] < 0)].copy()

SCOPES = [
    ("ep1", 1),
    ("ep1-5", 5),
    ("ep1-10", 10),
    ("ep1-20", 20),
    ("all 89 epochs", 10**9),
]

rows = []
for label, upto in SCOPES:
    sub = ok[ok["epoch"] <= upto]
    if sub.empty:
        continue
    s = sub["docking_raw"].astype(float)
    rows.append(
        {
            "scope": label,
            "n_attempted": int((raw["epoch"] <= upto).sum()),
            "n_docked": len(s),
            "dock_mean": round(float(s.mean()), 4),
            "dock_median": round(float(s.median()), 4),
            "dock_std": round(float(s.std()), 4),
            "qed_mean": round(float(pd.to_numeric(sub["qed_raw"], errors="coerce").mean()), 4),
            "sa_mean": round(float(pd.to_numeric(sub["sa_raw"], errors="coerce").mean()), 4),
        }
    )

t = pd.DataFrame(rows)
out = H / "diffgui_rollout_docking_summary.csv"
t.to_csv(out, index=False)
print(t.to_string(index=False))
print()
print("reference: clean-inference benchmark, 500 steps, n=2604, mean = -10.7886")
print("Wrote", out)
