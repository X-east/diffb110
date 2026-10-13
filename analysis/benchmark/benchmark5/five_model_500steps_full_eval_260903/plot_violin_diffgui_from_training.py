"""Five-model docking violin, but DiffGui taken from the training-early window.

DiffGui  : epochs 1-20 of the main DiffGui RL run (diffusion steps = 200,
           training-rollout sampling), docking_success == 1 and docking_raw < 0
Other 4  : clean-inference 500-step benchmark (rep4/5/6), as in thesis Table 2.1

WARNING - the two protocols differ (200 vs 500 diffusion steps and rollout vs
clean sampling). The footnote on the figure records this so the panel is not
read as a like-for-like comparison.

Output: five_model_violin_diffgui_ep1_20.png / .svg
"""

from pathlib import Path
import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

H = Path(__file__).resolve().parent
RUN = H.parents[3] / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809/epochs"

MODEL_ORDER = ["DiffGui", "DiffSBDD", "GCDM-SBDD", "PMDM", "TargetDiff"]
MODEL_DIR = {"DiffGui": "diffgui", "DiffSBDD": "diffsbdd", "GCDM-SBDD": "gcdm_sbdd",
             "PMDM": "pmdm", "TargetDiff": "targetdiff"}
REPEATS = ("rep4", "rep5", "rep6")
PALETTE = {"DiffGui": "#c0392b", "DiffSBDD": "#2e86c1", "GCDM-SBDD": "#7f8c8d",
           "PMDM": "#e67e22", "TargetDiff": "#27ae60"}

NOTES = {
    "DiffGui": "DiffGui = training-early window (ep1-20, 200 steps, rollout sampling)",
    "others": "other four models = clean-inference 500-step benchmark (Table 2.1)",
}


def load_benchmark() -> pd.DataFrame:
    frames = []
    for rep in REPEATS:
        for model in MODEL_ORDER[1:]:
            p = H / "docking" / rep / MODEL_DIR[model] / "docking_results.csv"
            d = pd.read_csv(p)
            keep = d["success"].astype(bool) & d["score"].notna()
            frames.append(pd.DataFrame({"model": model, "score": d.loc[keep, "score"].astype(float)}))
    return pd.concat(frames, ignore_index=True)


def load_diffgui_early() -> pd.DataFrame:
    files = sorted(glob.glob(str(RUN / "epoch_*/molecules.csv")))[:20]
    t = pd.concat([pd.read_csv(f, usecols=["epoch", "docking_raw", "docking_success"]) for f in files],
                  ignore_index=True)
    keep = (t["docking_success"] == 1) & (t["docking_raw"] < 0)
    return pd.DataFrame({"model": "DiffGui", "score": t.loc[keep, "docking_raw"].astype(float)})


data = pd.concat([load_diffgui_early(), load_benchmark()], ignore_index=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(9.6, 5.9), dpi=300)

shape_data = data[data["score"] <= 0.0]

sns.violinplot(data=shape_data, x="model", y="score", order=MODEL_ORDER, hue="model",
               hue_order=MODEL_ORDER, palette=PALETTE, inner=None, cut=0,
               linewidth=0.9, saturation=1.0, width=0.82, legend=False, ax=ax)

for i, model in enumerate(MODEL_ORDER):
    vals = data.loc[data["model"] == model, "score"].to_numpy()
    mean = float(np.mean(vals))
    print(f"{model:<12s} n={len(vals):>5d}  mean={mean:.4f}")
    ax.hlines(mean, i - 0.30, i + 0.30, color="#2c3e50", linewidth=3.2, zorder=4)
    ax.scatter([i], [mean], s=54, color="white", edgecolor="#2c3e50", linewidth=1.8, zorder=5)
    ax.text(i, mean + 0.20, f"{mean:.2f}", ha="center", va="bottom", fontsize=13,
            fontweight="bold", color="#2c3e50", zorder=6,
            bbox=dict(facecolor="white", alpha=0.80, edgecolor="none", pad=1.4))

ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels(MODEL_ORDER, fontsize=19.0, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("UniDock score (kcal/mol)", fontsize=16)
ax.tick_params(axis="y", labelsize=14)
ax.set_ylim(float(shape_data["score"].min()) - 0.6, float(shape_data["score"].max()) + 0.85)
ax.set_xlim(-0.62, len(MODEL_ORDER) - 1 + 0.72)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.subplots_adjust(left=0.095, right=0.985, top=0.975, bottom=0.115)
fig.savefig(H / "five_model_violin_diffgui_ep1_20.png", dpi=300)
fig.savefig(H / "five_model_violin_diffgui_ep1_20.svg")
plt.close(fig)
print("Wrote five_model_violin_diffgui_ep1_20.png")
