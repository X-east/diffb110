from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
H = Path(__file__).parent
o = pd.read_csv(H/"official_epoch_summary_selected.csv").sort_values("epoch")
c = pd.read_csv(H/"component_epoch_summary.csv")
c = c[c.source=="official_merged"].sort_values("epoch").reset_index(drop=True)
X = [c.epoch, c.epoch, o.epoch, o.epoch]
Y = [c.qed_raw_mean, c.sa_raw_mean, o.docking_raw_mean, o.molecule_rdkit_valid_ratio]
T = ["QED (RDKit)", "SA score (1-10)", "UniDock score (kcal/mol)", "Valid ratio (RDKit)"]
K = ["#27ae60", "#a04000", "#c0392b", "#2e86c1"]
fig, axs = plt.subplots(2, 2, figsize=(10.6, 6.2), dpi=300)
for a, x, y, t, k in zip(axs.ravel(), X, Y, T, K):
    a.plot(x, y, "-o", ms=2.8, lw=1.2, color=k)
    a.set_title(t, fontsize=11)
    a.set_xlabel("epoch", fontsize=9.5)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(alpha=0.28, lw=0.7)
    for ep, cl in ((59, "#8e44ad"), (79, "#16a085")):
        a.axvline(ep, color=cl, ls="--", lw=1.0, alpha=0.7)
fig.suptitle("Fig 2.3  QED / SA / docking / validity (ep1-89)", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(H / "four_panel_raw.png", dpi=300)
fig.savefig(H / "four_panel_raw.svg")
print("ok")
