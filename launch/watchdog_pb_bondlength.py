#!/usr/bin/env python3
"""自动止血线：监控 DiffGui DDPO 训练的键长质量，跌破阈值即停训。

- 每个轮询周期读取最新 epoch 的 pb_check_bond_lengths（8 rank 聚合）
- 若均值 < BOND_STOP（默认 0.80）→ pkill 主训练进程并退出
- PPO_ApproxKL 过高仅告警，不直接停（几何才是硬止损）
"""
import sys, csv, glob, os, time, subprocess, statistics, datetime

OUT_PREFIX = sys.argv[1] if len(sys.argv) > 1 else "OUT/diffgui-ddpo-qed-sa-docking-b8-t200-100ep-nofreeze-lr3e5"
POLL = int(sys.argv[2]) if len(sys.argv) > 2 else 120
BOND_STOP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.80
KL_WARN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10


def log(msg):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def find_run():
    cand = sorted(glob.glob(OUT_PREFIX + "_*"), key=os.path.getmtime, reverse=True)
    return cand[0] if cand else None


def agg_bond(run):
    eps = sorted(glob.glob(os.path.join(run, "epochs", "epoch_*")))
    if not eps:
        return None, None
    ep = eps[-1]
    rates = []
    for f in glob.glob(os.path.join(ep, "rank_*", "posebusters.csv")):
        try:
            with open(f) as fh:
                rd = csv.DictReader(fh)
                cols = rd.fieldnames or []
                blc = [c for c in cols if "bond_length" in c.lower()]
                if not blc:
                    continue
                for row in rd:
                    try:
                        rates.append(float(row[blc[0]]))
                    except Exception:
                        pass
        except Exception:
            pass
    if not rates:
        return None, ep
    return statistics.mean(rates), ep


def last_kl(run):
    m = os.path.join(run, "metrics", "metrics.csv")
    if not os.path.exists(m):
        return None
    try:
        rows = list(csv.DictReader(open(m)))
        if not rows:
            return None
        v = rows[-1].get("PPO_ApproxKL")
        return float(v) if v not in (None, "") else None
    except Exception:
        return None


def main():
    log(f"watchdog start prefix={OUT_PREFIX} poll={POLL}s bond_stop={BOND_STOP} kl_warn={KL_WARN}")
    while True:
        run = find_run()
        if not run:
            log("run dir not found yet, waiting...")
            time.sleep(POLL)
            continue
        bond, ep = agg_bond(run)
        kl = last_kl(run)
        if bond is None:
            log(f"epoch not ready ({os.path.basename(ep) if ep else 'NA'}), waiting...")
            time.sleep(POLL)
            continue
        epnum = os.path.basename(ep)
        klstr = f"{kl:.3e}" if kl is not None else "NA"
        log(f"check {epnum} bond_lengths_mean={bond:.3f} PPO_ApproxKL={klstr}")
        if bond < BOND_STOP:
            log(f"*** STOP-LOSS: bond_lengths {bond:.3f} < {BOND_STOP} at {epnum} -> killing training")
            subprocess.run("pkill -f main_diffgui_docking_ddp", shell=True)
            log("training killed by watchdog. exit.")
            return
        if kl is not None and kl > KL_WARN:
            log(f"!! WARN: PPO_ApproxKL {kl:.3e} > {KL_WARN} (high but bond ok)")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
