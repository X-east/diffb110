#!/usr/bin/env python3
"""从全量 ChEMBL 指纹缓存中随机采样子集，保存为训练可加载格式。"""
from __future__ import annotations

import argparse
import gzip
import logging
import pickle
import random
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("sample_chembl_fp")


def main():
    parser = argparse.ArgumentParser(description="Sample ChEMBL fingerprint cache subset")
    parser.add_argument("--ecfp4-cache", type=str,
                        default="data/chembl/chembl_36_ecfp4_fingerprints.pkl.gz",
                        help="Full ECFP4 fingerprint cache path")
    parser.add_argument("--fcfp4-cache", type=str,
                        default="data/chembl/chembl_36_fcfp4_fingerprints.pkl.gz",
                        help="Full FCFP4 fingerprint cache path")
    parser.add_argument("--sample-size", type=int, default=100_000,
                        help="Number of fingerprints to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic sampling")
    parser.add_argument("--output-dir", type=str, default="data/chembl",
                        help="Output directory for sampled caches")
    parser.add_argument("--output-tag", type=str, default="100k",
                        help="Tag inserted into output filenames")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp_type, cache_path in [("ecfp4", args.ecfp4_cache), ("fcfp4", args.fcfp4_cache)]:
        cache_path = Path(cache_path)
        if not cache_path.is_file():
            LOGGER.warning("Cache not found: %s, skipping %s", cache_path, fp_type)
            continue

        LOGGER.info("Loading %s cache from %s ...", fp_type, cache_path)
        with gzip.open(cache_path, "rb") as f:
            payload = pickle.load(f)

        fingerprints = list(payload["fingerprints"])
        records = list(payload.get("records", []))
        total = len(fingerprints)

        sample_size = min(args.sample_size, total)
        sampled_indices = sorted(rng.sample(range(total), sample_size))
        sampled_fps = [fingerprints[i] for i in sampled_indices]
        sampled_records = [records[i] for i in sampled_indices] if records else []

        LOGGER.info("  Total: %d, Sampled: %d (seed=%d)", total, sample_size, args.seed)

        # 输出文件名: chembl_36_100k_ecfp4_fingerprints.pkl.gz
        out_path = out_dir / f"chembl_36_{args.output_tag}_{fp_type}_fingerprints.pkl.gz"
        sampled_payload = {
            "fingerprints": sampled_fps,
            "records": sampled_records,
            "radius": payload.get("radius", 2),
            "n_bits": payload.get("n_bits", 2048),
            "fingerprint_type": fp_type,
            "chembl_version": payload.get("chembl_version", "36"),
            "source": str(cache_path),
            "sampled_from_total": total,
            "sampling_seed": args.seed,
        }
        with gzip.open(out_path, "wb") as f:
            pickle.dump(sampled_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        LOGGER.info("  Saved %s (%.0f KB)", out_path, out_path.stat().st_size / 1024)

    LOGGER.info("Done.")


if __name__ == "__main__":
    raise SystemExit(main())
