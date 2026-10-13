import sys
from pathlib import Path

run = Path(
    "runs/diffgui-deppa-pbfast-cpu4-lr1e5-wd1e4-b8x7-t200-100ep_20260715_2018"
)
ep_dirs = sorted(run.glob("epochs/epoch_*"), key=lambda p: int(p.name.split("_")[1]))
print("total epoch dirs:", len(ep_dirs))
for k in (10, 20):
    last = ep_dirs[-k:]
    tot = 0
    for e in last:
        sdf = e / "molecules.sdf"
        if sdf.exists():
            n = sum(1 for _ in open(sdf)) and 0
            with open(sdf) as fh:
                n = sum(1 for line in fh if "V2000" in line)
            tot += n
    print(f"last {k} epochs total molecules: {tot}")
