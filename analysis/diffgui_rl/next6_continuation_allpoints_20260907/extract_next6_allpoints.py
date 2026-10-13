import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = "/home/md666/data/disk6/DiffB110/runs/diffgui/diffgui-next6-20260618-04-ring_last3_pathology_guard_20260619_112410/training_summary.csv"
CONT = "/home/md666/data/disk6/DiffB110/runs/diffgui/diffgui-next6-20260618-04-ring_last3_pathology_guard-continue-to50_20260621_225705/training_summary.csv"

COLS = ["epoch", "docking_raw_mean", "carbon_fraction", "reward_total_mean",
        "logp_mean", "tpsa_mean", "reconstruction_success_ratio", "docking_success_ratio"]

def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({c: float(row[c]) if row[c] not in ("", None) else None for c in COLS})
            except Exception:
                continue
    rows.sort(key=lambda d: d["epoch"])
    return rows

base, cont = load(BASE), load(CONT)
allrows = sorted(base + cont, key=lambda d: d["epoch"])
x = [r["epoch"] for r in allrows]
dock = [r["docking_raw_mean"] for r in allrows]
carb = [r["carbon_fraction"] for r in allrows]

ORIG_EP1 = allrows[0]["docking_raw_mean"]   # this run's pre-RL (epoch 1)
ORIG_ALT = -9.899                            # alternate original probe (DG-01 epoch1)

# ---- print full table ----
print(f'{"ep":>3} {"dock":>8} {"carbon":>7} {"reward":>7} {"logp":>6} {"tpsa":>6} {"recon":>6} {"dsucc":>6}')
for r in allrows:
    print(f'{int(r["epoch"]):>3} {r["docking_raw_mean"]:>8.3f} {r["carbon_fraction"]:>7.3f} '
          f'{r["reward_total_mean"]:>7.3f} {r["logp_mean"]:>6.2f} {r["tpsa_mean"]:>6.1f} '
          f'{r["reconstruction_success_ratio"]:>6.3f} {r["docking_success_ratio"]:>6.3f}')

with open("/home/md666/data/disk6/DiffB110/next6_allpoints.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
    for r in allrows: w.writerow(r)

# ---- plot ALL points ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1]})
ax1.plot(x, dock, "-o", color="#1f77b4", lw=1.8, ms=4, zorder=5)
ax1.axhline(ORIG_EP1, ls="--", color="#888888", lw=1.4, zorder=4)
ax1.text(1, ORIG_EP1 + 0.015, f"original this run (ep1) {ORIG_EP1:.3f}", color="#555555", fontsize=8)
ax1.axhline(ORIG_ALT, ls=":", color="#bb5555", lw=1.2, zorder=4)
ax1.text(1, ORIG_ALT + 0.015, f"alt original probe (DG-01) {ORIG_ALT}", color="#aa4444", fontsize=8)
ax1.axvline(20.5, color="#999999", ls=":", lw=1)
ax1.text(21, max(dock) + 0.01, "continuation start", color="#999999", fontsize=8)
ax1.fill_between(x, ORIG_ALT, max(dock) + 0.05, where=[v > ORIG_ALT for v in dock],
                 color="#ffb3b3", alpha=0.15, zorder=1)
ax1.set_ylabel("Docking score (more negative = better)", fontsize=10)
ax1.set_ylim(min(dock) - 0.20, max(dock) + 0.10)
ax1.set_xlim(0.5, 50.5)
ax1.set_xticks(range(0, 51, 5))
ax1.grid(axis="y", ls=":", alpha=0.5)
ax1.set_title("Next6-04 full training (epoch 1-50): all 50 epochs", fontsize=12, fontweight="bold")

ax2.plot(x, carb, "-o", color="#ff7f0e", lw=1.8, ms=4, zorder=5)
ax2.set_ylabel("Mean carbon fraction", fontsize=10)
ax2.set_ylim(0.70, 0.92)
ax2.set_xlim(0.5, 50.5)
ax2.set_xticks(range(0, 51, 5))
ax2.grid(axis="y", ls=":", alpha=0.5)
ax2.set_xlabel("Epoch", fontsize=10)
ax2.axvline(20.5, color="#999999", ls=":", lw=1)

leg = [Patch(facecolor="#ffb3b3", alpha=0.4, label="worse than alt original (-9.899)")]
ax1.legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.9)

fig.tight_layout()
out = "/home/md666/data/disk6/DiffB110/next6_continuation_curve_allpoints.png"
fig.savefig(out, dpi=150)
print("saved", out)
