import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

stages = ["Original\n(pre-RL)", "base end\n(epoch20)", "continuation end\n(epoch50)"]
x = [0, 20, 50]
dock = [-9.899, -9.361, -9.687]          # original / Next6-04 early / Next6-04 late
carbon = [None, 0.791, 0.846]           # epoch20=0.791, epoch50=0.846
ORIG = -9.899

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1]})

# ---------- top: docking ----------
ax1.plot(x, dock, "-o", color="#1f77b4", lw=2.2, ms=9, zorder=5)
ax1.axhline(ORIG, ls="--", color="#888888", lw=1.4, zorder=4)
ax1.text(2, ORIG + 0.012, f"Original baseline  {ORIG}", color="#555555", fontsize=9)

ax1.fill_between(x, ORIG, max(dock) + 0.05, where=[v > ORIG for v in dock],
                 color="#ff9999", alpha=0.25, zorder=1)
ax1.axhspan(ORIG, max(dock) + 0.05, color="#ff9999", alpha=0.10, zorder=0)

ax1.annotate("", xy=(20, -9.361), xytext=(0, -9.899),
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
ax1.text(9, -9.50, "first 20 RL epochs:\ndrops ~0.54", color="#d62728", fontsize=9,
         ha="center", va="center")
ax1.annotate("", xy=(50, -9.687), xytext=(20, -9.361),
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.6))
ax1.text(35, -9.43, "continuation:\nrecovers ~0.33", color="#2ca02c", fontsize=9,
         ha="center", va="center")

for xi, yi in zip(x, dock):
    ax1.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                 xytext=(0, 12 if yi > ORIG else -16), ha="center", fontsize=9,
                 fontweight="bold")

ax1.set_ylabel("Docking score (more negative = better)", fontsize=10)
ax1.set_ylim(ORIG - 0.12, max(dock) + 0.10)
ax1.set_xlim(-3, 53)
ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=9)
ax1.grid(axis="y", ls=":", alpha=0.5)
ax1.set_title("Next6-04 continuation: docking never reaches the original model",
              fontsize=12, fontweight="bold")

# ---------- bottom: carbon fraction drift ----------
xc = [20, 50]
ax2.plot(xc, carbon[1:], "-o", color="#ff7f0e", lw=2.2, ms=9, zorder=5)
ax2.annotate(f"{carbon[1]:.3f}", (20, carbon[1]), textcoords="offset points",
             xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
ax2.annotate(f"{carbon[2]:.3f}", (50, carbon[2]), textcoords="offset points",
             xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
ax2.set_ylabel("Mean carbon fraction", fontsize=10)
ax2.set_ylim(0.70, 0.90)
ax2.set_xlim(-3, 53)
ax2.set_xticks(x)
ax2.set_xticklabels(stages, fontsize=9)
ax2.grid(axis="y", ls=":", alpha=0.5)
ax2.set_xlabel("Training stage (epoch)", fontsize=10)
ax2.fill_between(xc, carbon[1], carbon[2], color="#ff7f0e", alpha=0.15)
ax2.text(35, 0.82, "carbon drift up", color="#ff7f0e", fontsize=9, ha="center")

legend = [Patch(facecolor="#ff9999", alpha=0.3, label="worse than original (higher score)")]
ax1.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)

fig.tight_layout()
out = "/home/md666/data/disk6/DiffB110/next6_continuation_curve.png"
fig.savefig(out, dpi=150)
print("saved", out)
