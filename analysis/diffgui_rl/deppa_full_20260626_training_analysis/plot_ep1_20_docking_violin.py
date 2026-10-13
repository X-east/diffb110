"""Table + violin for the DiffGui training-early window (epochs 1-20).

Protocol: diffusion steps = 200, training rollout sampling.
This is NOT the clean-inference 500-step benchmark protocol of thesis Table 2.1,
so the two must never be quoted side by side.

Source : runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809/
         epochs/epoch_01..20/molecules.csv
Filter : docking_success == 1 and docking_raw < 0
Output : diffgui_ep1_20_summary.csv, diffgui_ep1_20_docking_violin.png/.svg
"""

from pathlib import Path
import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

H = Path(__file__).resolve().parent
RUN = H.parents[2] / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809/epochs"
LAST_EPOCH = 20

COLS = ["epoch", "docking_raw", "docking_success", "rdkit_valid", "qed_raw", "sa_raw"]
files = sorted(glob.glob(str(RUN / "epoch_*/molecules.csv")))[:LAST_EPOCH]
raw = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)

per = raw.groupby("epoch").apply(
    lambda d: pd.Series(
        {
            "valid_pct": d["rdkit_valid"].mean() * 100,
            "dock_mean": d.loc[(d["docking_success"] == 1) & (d["docking_raw"] < 0), "docking_raw"].mean(),
            "qed": pd.to_numeric(d["qed_raw"], errors="coerce").mean(),
            "sa": pd.to_numeric(d["sa_raw"], errors="coerce").mean(),
        }
    ),
    include_groups=False,
)

summary = pd.DataFrame(
    [
        {
            "model": "DiffGui (training early, ep1-20, T=200)",
            "validity": f"{per.valid_pct.mean():.2f}% ± {per.valid_pct.std():.2f}%",
            "dock_mean": f"{per.dock_mean.mean():.3f} ± {per.dock_mean.std():.3f}",
            "QED": f"{per.qed.mean():.3f} ± {per.qed.std():.3f}",
            "SA": f"{per.sa.mean():.2f} ± {per.sa.std():.2f}",
        }
    ]
)
summary.to_csv(H / "diffgui_ep1_20_summary.csv", index=False)
print(summary.to_string(index=False))

ok = raw[(raw["docking_success"] == 1) & (raw["docking_raw"] < 0)]
vals = ok["docking_raw"].astype(float)
mean = float(vals.mean())

fig, ax = plt.subplots(figsize=(4.6, 6.0), dpi=300)
parts = ax.violinplot([vals.to_numpy()], positions=[0], widths=0.8, showextrema=False)
for body in parts["bodies"]:
    body.set_facecolor("#c0392b")
    body.set_edgecolor("#7b241c")
    body.set_alpha(0.85)
ax.hlines(mean, -0.30, 0.30, color="#2c3e50", linewidth=3.2, zorder=4)
ax.scatter([0], [mean], s=54, color="white", edgecolor="#2c3e50", linewidth=1.8, zorder=5)
ax.text(
    0, mean + 0.22, f"{mean:.2f}", ha="center", va="bottom", fontsize=13,
    fontweight="bold", color="#2c3e50", zorder=6,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.4),
)
ax.set_xticks([0])
ax.set_xticklabels(["DiffGui"], fontsize=19, fontweight="bold")
ax.set_ylabel("UniDock score (kcal/mol)", fontsize=16)
ax.tick_params(axis="y", labelsize=14)
ax.set_xlim(-0.62, 0.72)
ax.set_ylim(float(vals.min()) - 0.6, float(vals.max()) + 0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.28, linewidth=0.7)
fig.subplots_adjust(left=0.20, right=0.96, top=0.975, bottom=0.12)
fig.savefig(H / "diffgui_ep1_20_docking_violin.png", dpi=300)
fig.savefig(H / "diffgui_ep1_20_docking_violin.svg")
plt.close(fig)
print(f"\nn={len(vals)}  violin mean={mean:.4f}")
print("Wrote diffgui_ep1_20_docking_violin.png")
